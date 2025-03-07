import os

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = (
    "false"  # Avoid preallocating GPU memory for JAX
)
import argparse
from functools import partial

import jax
import jax.numpy as jnp
from jax.random import split
import numpy as np
import evosax
from tqdm.auto import tqdm

import substrates
import foundation_models
from rollout import rollout_simulation
import asal_metrics
from video_text_models import LLAVA
import util

parser = argparse.ArgumentParser()
group = parser.add_argument_group("meta")
group.add_argument("--seed", type=int, default=0, help="the random seed")
group.add_argument("--save_dir", type=str, default=None, help="path to save results to")

group = parser.add_argument_group("substrate")
group.add_argument(
    "--substrate", type=str, default="boids", help="name of the substrate"
)
group.add_argument(
    "--rollout_steps",
    type=int,
    default=None,
    help="number of rollout timesteps, leave None for the default of the substrate",
)

group = parser.add_argument_group("evaluation")
group.add_argument(
    "--foundation_model",
    type=str,
    default="clip",
    help="the foundation model to use (don't touch this)",
)
group.add_argument(
    "--prompts",
    type=str,
    default="a biological cell;two biological cells",
    help="prompts to optimize for seperated by ';'",
)
group.add_argument(
    "--coef_prompt", type=float, default=1.0, help="coefficient for ASAL prompt loss"
)
group.add_argument(
    "--coef_softmax",
    type=float,
    default=0.0,
    help="coefficient for softmax loss (only for multiple temporal prompts)",
)
group.add_argument(
    "--coef_oe",
    type=float,
    default=0.0,
    help="coefficient for ASAL open-endedness loss (only for single prompt)",
)
group.add_argument(
    "--coef_recon",
    type=float,
    default=0.0,
    help="coefficient for ASAL recon loss",
)

group = parser.add_argument_group("optimization")
group.add_argument(
    "--bs", type=int, default=1, help="number of init states to average simulation over"
)
group.add_argument(
    "--pop_size", type=int, default=16, help="population size for Sep-CMA-ES"
)
group.add_argument(
    "--n_iters", type=int, default=1000, help="number of iterations to run"
)
group.add_argument("--sigma", type=float, default=0.1, help="mutation rate")


def parse_args(*args, **kwargs):
    args = parser.parse_args(*args, **kwargs)
    for k, v in vars(args).items():
        if isinstance(v, str) and v.lower() == "none":
            setattr(args, k, None)  # Turn "none" strings into actual None
    return args


def main(args):
    prompts = args.prompts.split(";")
    # Splitting prompts allows multiple targets (Eq. (2) in ASAL can be extended over time)
    # if args.time_sampling < len(
    #     prompts
    # ):  # If we have more prompts than frames, match them
    #     args.time_sampling = len(prompts)
    print(args)

    # Create a video-language model for similarity scoring and reconstruction
    vtm = LLAVA(
        model_id="llava-hf/llava-interleave-qwen-0.5b-hf",
        device="cpu",
    )

    fm = foundation_models.create_foundation_model(args.foundation_model)
    # Create a vision-language model for alignment

    substrate = substrates.create_substrate(args.substrate)
    # Substrate defines the ALife simulation space (parameters + dynamics)
    
    substrate = substrates.FlattenSubstrateParameters(substrate)
    # Flatten parameters into a single vector for optimization
    
    if args.rollout_steps is None:
        args.rollout_steps = (
            substrate.rollout_steps
        )  # Use default steps if none provided
    
    # This partial function returns a dictionary with:
    #   "z":  the image embeddings (already in the FM's space)
    #   "rgb": a list/array of frames
    rollout_fn = partial(
        rollout_simulation,
        s0=None,
        substrate=substrate,
        fm=fm,
        rollout_steps=args.rollout_steps,
        time_sampling="video",  # capture all frames
        img_size=224,
        return_state=False,
    )

    # Text embedding of the (one or more) prompts
    print(prompts)
    z_txt = fm.embed_txt(prompts)  # shape [P, D]

    rng = jax.random.PRNGKey(args.seed)
    strategy = evosax.Sep_CMA_ES(
        popsize=args.pop_size, num_dims=substrate.n_params, sigma_init=args.sigma
    )
    es_params = strategy.default_params

    # Initialize the ES state
    rng, init_rng = split(rng)
    es_state = strategy.initialize(init_rng, es_params)

    def compute_loss_unbatched(rng, params):
        """
        1) Run the substrate + FM to get frames and embeddings (rollout_fn).
        2) Convert frames from JAX -> NumPy, call PIL/HF code.
        3) Compute losses in normal Python space, return a Python float and dict.
        """
        # Run JAX-based simulation
        rollout_data = rollout_fn(rng, params)
        z_frames = rollout_data["z"]  # shape [rollout_length, D]
        rgb_frames = rollout_data["rgb"]  # JAX array of shape [T, H, W, 3]

        # Convert frames to NumPy (outside jit/vmap)
        video_frames = np.array((rgb_frames * 255).clip(0, 255), dtype=np.uint8)

        # Use the video-text model to describe the video (PIL, HF calls)
        vtm_desc = vtm.describe_video(video_frames, sample_rate=20)
        print("Video Model Description:", vtm_desc)
        z_desc = fm.embed_txt([vtm_desc])

        # Different ASAL losses
        loss_recon = asal_metrics.calc_reconstruction_loss(z_txt, z_desc)
        loss_prompt = asal_metrics.calc_supervised_target_score(z_frames, z_txt)
        loss_softmax = asal_metrics.calc_supervised_target_softmax_score(z_frames, z_txt)
        loss_oe = asal_metrics.calc_open_endedness_score(z_frames)

        # Weighted sum
        loss = (
            args.coef_prompt * loss_prompt
            + args.coef_softmax * loss_softmax
            + args.coef_oe * loss_oe
            + args.coef_recon * loss_recon
        )
        loss_dict = dict(
            loss=loss,
            loss_prompt=loss_prompt,
            loss_softmax=loss_softmax,
            loss_oe=loss_oe,
            loss_recon=loss_recon,
        )
        return float(loss), loss_dict  # Return Python floats

    def do_iter(es_state, rng):
        """
        One iteration of evolution:
          1) Ask for a population of parameter vectors
          2) Evaluate each using a normal Python loop (avoids JAX tracer -> PIL issues)
          3) Tell the ES the fitness values
        """
        rng, ask_rng = split(rng)
        params_pop, next_es_state = strategy.ask(ask_rng, es_state, es_params)

        # Evaluate each member (and average over bs random seeds)
        pop_losses = []
        pop_loss_dicts = []
        for params_i in params_pop:
            # Optionally evaluate multiple init states
            accum_loss = 0.0
            accum_dict = None
            for _ in range(args.bs):
                rng, roll_rng = split(rng)
                loss_val, loss_dict = compute_loss_unbatched(roll_rng, params_i)
                accum_loss += loss_val
                accum_dict = loss_dict
            mean_loss = accum_loss / args.bs
            pop_losses.append(mean_loss)
            pop_loss_dicts.append(accum_dict)

        # Convert losses to JAX array
        pop_losses_jax = jnp.array(pop_losses, dtype=jnp.float32)
        
        # Update the ES
        next_es_state = strategy.tell(params_pop, pop_losses_jax, next_es_state, es_params)
        
        # Return the ES state plus any data you want to record
        data_out = dict(
            best_loss=next_es_state.best_fitness,
            loss_dict=pop_loss_dicts,  # or you could just store the mean or best
        )
        return next_es_state, data_out

    # Main ES loop
    data = []
    pbar = tqdm(range(args.n_iters))
    for i_iter in pbar:
        rng, iter_rng = split(rng)
        es_state, di = do_iter(es_state, iter_rng)
        data.append(di)

        # Live progress
        pbar.set_postfix(best_loss=es_state.best_fitness.item())

        # Periodic saving
        if args.save_dir is not None and (
            i_iter % (args.n_iters // 10) == 0 or i_iter == args.n_iters - 1
        ):
            data_save = jax.tree_map(lambda *x: np.array(jnp.stack(x, axis=0)), *data)
            util.save_pkl(args.save_dir, "data", data_save)

            best_tuple = (es_state.best_member, es_state.best_fitness)
            best = jax.tree_map(lambda x: np.array(x), best_tuple)
            util.save_pkl(args.save_dir, "best", best)

if __name__ == "__main__":
    main(parse_args())
