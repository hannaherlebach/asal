import os
import argparse
import numpy as np
import imageio
from functools import partial

import jax
import jax.numpy as jnp
from jax.random import split

import evosax
import substrates
import foundation_models
from rollout import rollout_simulation

def create_simulation_video(rng, params, rollout_fn, output_path='simulation.mp4', fps=15):
    """
    Runs a simulation and saves the video.

    Parameters
    ----------
    rng : jax.random.PRNGKey
        The random key for reproducibility.
    params : jnp.ndarray
        Parameters defining the simulation.
    rollout_fn : Callable
        A partially applied function that calls rollout_simulation with time_sampling='video'.
    output_path : str
        The filename of the output video file.
    fps : int
        Frames per second for the video.
    """
    # Run the simulation with 'video' mode to capture frames at each timestep
    data = rollout_fn(rng, params)

    # Convert frames from float [0,1] to uint8 [0,255] for video
    rgb_frames = np.array(data['rgb'])
    video_frames = (rgb_frames * 255).clip(0, 255).astype(np.uint8)

    # Save frames as an mp4 video
    imageio.mimsave(output_path, video_frames, fps=fps)
    print(f"Video saved to {output_path}")

if __name__ == '__main__':
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Run a simulation and save a video")
    parser.add_argument('--substrate', type=str, default='boids', help="Simulation substrate")
    parser.add_argument('--rollout_steps', type=int, default=256, help="Number of simulation timesteps")
    parser.add_argument('--fps', type=int, default=15, help="Frames per second for the video")
    parser.add_argument('--output_path', type=str, default='simulation.mp4', help="Filename for the saved video")
    parser.add_argument('--seed', type=int, default=0, help="Random seed for reproducibility")
    args = parser.parse_args()

    # Initialize JAX PRNG key
    rng = jax.random.PRNGKey(args.seed)

    # Load foundation model and substrate
    fm = foundation_models.create_foundation_model('clip')
    substrate = substrates.create_substrate(args.substrate)
    substrate = substrates.FlattenSubstrateParameters(substrate)

    # Get default parameters
    params = substrate.default_params(rng)

    # Create a rollout function configured for video output
    rollout_fn = partial(
        rollout_simulation,
        s0=None,
        substrate=substrate,
        fm=fm,
        rollout_steps=args.rollout_steps,
        time_sampling='video',  # Capture all frames
        img_size=224,
        return_state=False
    )

    # Generate and save the simulation video
    create_simulation_video(rng, params, rollout_fn, output_path=args.output_path, fps=args.fps)
