import jax
import jax.numpy as jnp
import numpy as np
from einops import rearrange, reduce, repeat
from jax.random import split
from collections import namedtuple

from .flenia_impl_alt import FL_Config as ConfigFLenia
from .flenia_impl_alt import FlowLenia as FlowLeniaImpl
from .flenia_impl_alt import FL_State as State
from .flenia_impl_alt import Params, CompiledParams
from .flenia_impl_alt import conn_from_matrix, compile_kernel_computer

"""
Wrapper for FlowLenia alternative, to work with evosax.
"""

# Maybe make these a dict
mat_0 = np.array([[2, 1, 0],
                  [0, 2, 1],
                  [1, 0, 2]], dtype=int)
mat_1 = np.array([[3, 1, 0],
              [0, 3, 1],
              [1, 0, 3]], dtype=int)

class FlowLenia:
    """
    This class wraps FlowLeniaImpl ALT to work with ASAL.
    """
    def __init__(self, grid_size=128, C=1, c0=[0], c1=[[0]], k=9, dd=5, dt=0.2, sigma=0.65, n=2, theta_A: float=1., border="wall", matrix=mat_1):
        self.grid_size = grid_size

        # Set up channels using matrix
        if matrix is not None:
            k = matrix.sum()
            c0, c1 = conn_from_matrix(matrix)
            C = matrix.shape[0]

        self.config_flenia = ConfigFLenia(SX=grid_size, SY=grid_size, C=C, c0=c0, c1=c1, nb_k=k, dd=dd, dt=dt, sigma=sigma, theta_A=theta_A, border=border)

        self.flenia = FlowLeniaImpl(self.config_flenia)
        self.step_fn = self.flenia._build_step_fn()
        self.compile_params = compile_kernel_computer(SX=grid_size, SY=grid_size, nb_k=k)

    def default_params(self, rng):
        params = self.flenia.rule_space.sample(rng)
        return params # Unsure if these are the right shape!

    def init_state(self, rng, params):
        """
        Note that in this version, State only has one field A, doesn't have fK.
        """
        A = jnp.zeros((self.config_flenia.SX, self.config_flenia.SY, self.config_flenia.C))
        return State(A=A)

    def step_state(self, rng, state, params):
        """
        Converts params to CompiledParams
        """
        compiled_params = self.compile_params(params)
        next_state = self.step_fn(state, compiled_params)
        return next_state

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

