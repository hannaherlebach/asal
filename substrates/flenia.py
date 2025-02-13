
import flax.linen as nn
import jax
import jax.numpy as jnp
from einops import rearrange, reduce, repeat
import jax.random as jr
import equinox as eqx
from typing import NamedTuple, Optional, Tuple
from jaxtyping import Float, Array

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

from functools import partial

from substrates.flenia_utils import *


class ConfigFL(NamedTuple):
    X: int=128
    Y: int=128
    C: int=1
    c0: list[int]=[0]
    c1: list[list[int]]=[[0]]
    k: int=10
    dd: int=5
    dt: float=0.2
    sigma: float=.65
    border: str="wall"

class Params(NamedTuple):
    R: Float[Array, ""]
    r: Float[Array, "k"]
    m: Float[Array, "k"]
    s: Float[Array, "k"]
    h: Float[Array, "k"]
    a: Float[Array, "k 3"]
    b: Float[Array, "k 3"]
    w: Float[Array, "k 3"]

class State(NamedTuple):
    A: Float[Array, "X Y C"]
    fK: Float[Array, "X Y k"]

"""
The Flow Lenia substrate.
"""
class FlowLenia():
    def __init__(self, cfg: ConfigFL):
        self.cfg = cfg
        self.RT = ReintegrationTracking(cfg.X, cfg.Y, cfg.dt, cfg.dd, cfg.sigma, cfg.border, has_hidden=False)

    def default_params(self, rng):
        kR, kr, km, ks, kh, ka, kb, kw = jr.split(rng, 8)
        R = jr.uniform(kR, (    ), minval=2.000, maxval=25.0)
        r = jr.uniform(kr, (self.cfg.k,  ), minval=0.200, maxval=1.00)
        m = jr.uniform(km, (self.cfg.k,  ), minval=0.050, maxval=0.50) 
        s = jr.uniform(ks, (self.cfg.k,  ), minval=0.001, maxval=0.18)
        h = jr.uniform(kh, (self.cfg.k,  ), minval=0.010, maxval=1.00)
        a = jr.uniform(ka, (self.cfg.k, 3), minval=0.000, maxval=1.00)
        b = jr.uniform(kb, (self.cfg.k, 3), minval=0.001, maxval=1.00)
        w = jr.uniform(kw, (self.cfg.k, 3), minval=0.010, maxval=0.50)
        
        # Unsure if this is the right thing to return
        return Params(R=R, r=r, m=m, s=s, h=h, a=a, b=b, w=w)
    
    def init_state(self, rng, params: Params):
        fK = get_kernels_fft(self.cfg.X, self.cfg.Y, self.cfg.k, params.R, params.r, params.a, params.w, params.b)
        A = jnp.zeros((self.cfg.X, self.cfg.Y, self.cfg.C))
        state = State(A=A, fK=fK)
        return self.step_state(rng, state, params)

        
        base_dyn, base_init = self.base_params[:45], self.base_params[45:]
        params_dyn, params_init = params[:45], params[45:]

        params_dyn = jax.nn.sigmoid(base_dyn + jnp.clip(params_dyn, -self.clip1, self.clip1))
        params_init = jax.nn.sigmoid(base_init + jnp.clip(params_init, -self.clip2, self.clip2))
        params = jnp.concatenate([params_dyn, params_init], axis=0)
        # params = jax.nn.sigmoid(jnp.clip(params, -self.clip_genotype, self.clip_genotype)+self.base_params)

        carry = self.lenia.express_genotype(self.init_carry, params)
        state = dict(carry=carry, img=jnp.zeros((self.phenotype_size, self.phenotype_size, 3)))
        # return state
        return self.step_state(rng, state, params) # so init img is not zeros lol
    
    def step_state(self, rng, state, params):
        # --- Lenia ---
        A = state.A

        fA = jnp.fft.fft2(A, axes=(0,1))  # (x,y,c)

        fAk = fA[:, :, self.cfg.c0]  # (x,y,k)

        U = jnp.real(jnp.fft.ifft2(state.fK * fAk, axes=(0,1)))  # (x,y,k)

        U = growth(U, params.m, params.s) * params.h  # (x,y,k)

        U = jnp.dstack([ U[:, :, self.cfg.c1[c]].sum(axis=-1) for c in range(self.cfg.C) ])  # (x,y,c)

        # --- Flow ---

        nabla_U = sobel(U) #(x, y, 2, c)

        nabla_A = sobel(A.sum(axis = -1, keepdims = True)) #(x, y, 2, 1)

        alpha = jnp.clip((A[:, :, None, :]/self.cfg.C)**2, .0, 1.)

        F = nabla_U * (1 - alpha) - nabla_A * alpha
        nA = self.RT(A, F) #type:ignore

        return state._replace(A=nA)
    
    def render_state(self, state, params, img_size=None):
        img = state['img']
        if img_size is not None:
            img = jax.image.resize(img, (img_size, img_size, 3), method='nearest')
        return img
    
    # remove
    def rollout_(self, state: State, params, key: Optional[jax.Array]=None, 
                 steps: int=100)->State:
        return jax.lax.fori_loop(0, steps, lambda i,s: self.step_state(key, s, params), state)


if __name__ == '__main__':
    # import matplotlib.pyplot as plt
    # cfg = Config(X=64, Y=64, C=3, k=9)
    # M = np.array([[2, 1, 0],
    #               [0, 2, 1],
    #               [1, 0, 2]])
    # c0, c1 = conn_from_matrix(M)
    # cfg = cfg._replace(c0=c0, c1=c1)
    # fl = FlowLenia(cfg)
    # params = fl.default_params(jr.key(0))
    # s = fl.init_state(jr.key(2), params)
    # locs = jnp.arange(20) + (cfg.X//2-10)
    # A = s.A.at[jnp.ix_(locs, locs)].set(jr.uniform(jr.key(2), (20, 20, 1)))
    # s = s._replace(A=A)
    # s = fl.rollout_(s, params, None, 100)
    # plt.imshow(s.A); plt.show()

    # Initialize configuration and FlowLenia model
    cfg = ConfigFL(X=64, Y=64, C=3, k=9)
    M = np.array([[2, 1, 0],
                  [0, 2, 1],
                  [1, 0, 2]])
    c0, c1 = conn_from_matrix(M)
    cfg = cfg._replace(c0=c0, c1=c1)
    
    fl = FlowLenia(cfg)
    params = fl.default_params(jr.key(0))
    s = fl.init_state(jr.key(2), params)

    # Initialize a random seed patch
    locs = jnp.arange(20) + (cfg.X // 2 - 10)
    A = s.A.at[jnp.ix_(locs, locs)].set(jr.uniform(jr.key(2), (20, 20, 1)))
    s = s._replace(A=A)

    # Set up the figure
    fig, ax = plt.subplots()
    im = ax.imshow(s.A, cmap='viridis', interpolation='nearest')

    def update(frame):
        """Update function for animation"""
        global s
        s = fl.step_state(jr.key(frame), s, params)  # Step the simulation
        im.set_array(s.A)  # Update image
        return [im]

    # Create animation
    ani = animation.FuncAnimation(fig, update, frames=100, interval=50, blit=True)

    # Show animation
    plt.show()
