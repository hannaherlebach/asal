
import flax.linen as nn
import jax
import jax.numpy as jnp
import equinox as eqx
import numpy as np
from einops import rearrange, reduce, repeat
from jax.random import split
from collections import namedtuple

from .flenia_impl_old import Config as ConfigFLenia
from .flenia_impl_old import FlowLenia as FlowLeniaImpl
from .flenia_impl_old import conn_from_matrix

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

Params = namedtuple('Params', 'R r m s h a b w')

class FlowLenia:
    """
    This class wraps FlowLeniaImpl to work with ASAL.
    """
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

        # Get flattened parameters and record original tree structure.
        self.base_params, self.params_treedef = self._extract_params(self.flenia)

        # clip?

    def default_params(self, rng):
        """Returns a random PyTree of the same structure as base_params.
        TODO is this what we're meant to do?
        """
        # params = jax.tree_util.tree_map(lambda x: jax.random.normal(rng, x.shape) * 0.1, self.base_params)
        # Shouldn't be normal tbh
        params = jax.random.normal(rng, self.base_params.shape) * 0.1
        print("default params", params.shape)
        return params

    def _extract_params(self, model):
        """Extract flattened parameters from equinox module as PyTree."""
        params = eqx.filter(model, eqx.is_array)

        shapes = jax.tree_map(lambda x: x.shape, params)
        sizes = jax.tree_map(lambda x: x.size, params)

        import pdb; pdb.set_trace()
        leaves, treedef = jax.tree_util.tree_flatten(params)
        params_flattened = jnp.concatenate([jnp.ravel(leaf) for leaf in leaves])

        return params_flattened, treedef

    def _get_real_params(self, params):
        # Recover PyTree from params and parameter structure of the model.
        print("get real params", params.shape)
        params_reconstructed = self.params_treedef.unflatten(params)
        return params_reconstructed
    
    def init_state(self, rng, params):
        """
        params: flattened params.
        """
        print("init state params", params.shape)
        self._update_model(params) # this updates self.flenia
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
    
    def _update_model(self, params_flattened):
        params = self._get_real_params(params_flattened)
        self.flenia = eqx.tree_at(
            lambda m: eqx.filter(m, eqx.is_array),
            self.flenia,
            params
        )
    
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