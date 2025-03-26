import wandb
from einops import rearrange
import numpy as np
import jax, jax.numpy as jnp
from functools import partial

from rollout import rollout_simulation

class WandbLogger:
    def __init__(self, project, group, entity, config, substrate=None):
        self.run = wandb.init(project=project, group=group, entity=entity, config=config)
        if substrate is not None: # hacky fix later
            self.initialise_rollout(substrate)

    def initialise_rollout(self, substrate):
        self.rollout_fn = partial(rollout_simulation, s0=None, substrate=substrate, fm=None, rollout_steps=substrate.rollout_steps, time_sampling='video', img_size=224, return_state=False)

    def initialise_prompt_logging(self):
        """
        Run this once if logging prompts to initialise table.
        """
        self.table = wandb.Table(columns=["prompts"])

    def log_prompt(self, prompt):
        """
        Log prompt to new row.
        """
        self.table.add_data(prompt)

    def log_losses(self, data):
        """
        Takes data output of do_iter and logs the losses.
        """
        self.run.log({"best_loss": data["best_loss"]})
        best_losses = jax.tree_util.tree_map(lambda x: jnp.min(x), data['loss_dict'])
        self.run.log({k: v for k, v in best_losses.items()})

    def log_video(self, rng, params, name, caption, rgb=False, fps=30):
        """
        Takes rng and current best params, and logs a rollout video.
        Args:
            rng
            params
            name (str): name of plot
            rgb (bool): whether imgs already in RGB form (channels are ints from 0 to 255). if not, convert to RGB
            fps
        """
        rollout_data = self.rollout_fn(rng, params)
        img = np.array(rollout_data['rgb'])
        img = rearrange(img, "T H W D -> T D H W")
        if not rgb:
            img = (img*255).clip(0,255)
        
        img = img.astype(np.uint8)
        self.run.log({name: wandb.Video(img, fps=fps, caption=caption)})