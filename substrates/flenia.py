
import flax.linen as nn
import jax
import jax.numpy as jnp
from einops import rearrange, reduce, repeat
from jax.random import split

from .flenia_impl import Config as ConfigFLenia
from .flenia_impl import FlowLenia as FlowLeniaImpl

"""
The Flow Lenia substrate.
The implementation of Flow Lenia is from https://github.com/erwanplantec/FlowLenia/tree/main/flowlenia.
"""

class FlowLenia:
    def __init__(self, grid_size=128, n_channels=1, c0=[0], c1=[[0]], k=10, dd=5, dt=0.2, sigma=0.65, border="wall"):
        self.grid_size = grid_size
        self.config_flenia = ConfigFLenia(X=grid_size, Y=grid_size, C=n_channels, c0=c0, c1=c1, k=k, dd=dd, dt=dt, sigma=sigma, border=border)
        self.flenia = FlowLeniaImpl(self.config_flenia)

        # clip?

    def default_params(self, rng):
        # Unsure whether to return the params attached to self.flenia, or generate some new random ones... depends what they're for

        # TODO version using rng 

        return jnp.array([self.flenia.R, self.flenia.r, self.flenia.m, self.flenia.s, self.flenia.h, self.flenia.a, self.flenia.b, self.flenia.w])
    
    def init_state(self, rng, params):
        state = self.flenia.initialize(rng)
        return state # Can I leave as is?

    def step_state(self, rng, state, params):
        new_state = self.flenia(state, rng)

        return new_state # In lenia, this is a dict... now its a namedtuple... fine? unless the type for state is important in ASAL
    
    def render_state(self, state, params, img_size=None):
        img = state.A # I think this is right?
        if img_size is not None:
            img = jax.image.resize(img, (img_size, img_size, 3), method='nearest')
        return img