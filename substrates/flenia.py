
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from einops import rearrange, reduce, repeat
from jax.random import split

from .flenia_impl import Config as ConfigFLenia
from .flenia_impl import FlowLenia as FlowLeniaImpl
from .flenia_impl import conn_from_matrix

"""
The Flow Lenia substrate.
The implementation of Flow Lenia is from https://github.com/erwanplantec/FlowLenia/tree/main/flowlenia.
"""

# Maybe make these a dict
mat_0 = np.array([[2, 1, 0],
                  [0, 2, 1],
                  [1, 0, 2]], dtype=int)
mat_1 = np.array([[3, 1, 0],
              [0, 3, 1],
              [1, 0, 3]], dtype=int)


class FlowLenia:
    def __init__(self, grid_size=128, C=1, c0=[0], c1=[[0]], k=9, dd=5, dt=0.2, sigma=0.65, border="wall", seed=42, matrix=mat_1):
        self.grid_size = grid_size

        # Set up channels using matrix
        if matrix is not None:
            k = matrix.sum()
            c0, c1 = conn_from_matrix(matrix)
            C = matrix.shape[0]

        self.config_flenia = ConfigFLenia(X=grid_size, Y=grid_size, C=C, c0=c0, c1=c1, k=k, dd=dd, dt=dt, sigma=sigma, border=border)
        key = jax.random.PRNGKey(seed)
        self.flenia = FlowLeniaImpl(self.config_flenia, key)

        # clip?

    def default_params(self, rng):
        # Unsure whether to return the params attached to self.flenia, or generate some new random ones... depends what they're for

        # TODO version using rng 

        # Flenia impl never actually uses params... this might be a problem though cos params are what we wanna optimise with ASAL right?
        return None

        return jnp.array([self.flenia.R, self.flenia.r, self.flenia.m, self.flenia.s, self.flenia.h, self.flenia.a, self.flenia.b, self.flenia.w])
    
    def init_state(self, rng, params):
        rng, rng_init = split(rng)
        s = self.flenia.initialize(rng_init)
        # n = int(self.grid_size/3) # don't know why
        # locs = jnp.arange(n) + (self.config_flenia.X//2-10)
        # A = s.A.at[jnp.ix_(locs, locs)].set(jax.random.uniform(rng, (40, 40, self.config_flenia.C)))
        A = s.A.at[44:84, 44:84, :].set(jax.random.uniform(rng, (40, 40, self.config_flenia.C)))
        s = s._replace(A=A)

        return s # Can I leave as is?

    def step_state(self, rng, state, params):
        new_state = self.flenia(state, rng)

        return new_state # In lenia, this is a dict... now its a namedtuple... fine? unless the type for state is important in ASAL
    
    def render_state(self, state, params, img_size=None):
        A = state.A # I think this is right?
        C = A.shape[-1]
        # C = self.config_flenia.C
        print(f"{C=}")
        if C==1:
            img = A
        elif C==2:
            img=jnp.dstack([A[...,0], A[...,0], A[...,1]])
        else:
            img=A[...,:3]
        if img_size is not None:
            img = jax.image.resize(img, (img_size, img_size, 3), method='nearest')
        return img