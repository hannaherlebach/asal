
import flax.linen as nn
import jax
import jax.numpy as jnp
from einops import rearrange, reduce, repeat
import jax.random as jr
import equinox as eqx
from typing import NamedTuple, Optional, Tuple, Callable
from jaxtyping import Float, Array

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

from functools import partial

from substrates.flenia_utils import *

class ConfigFLP(NamedTuple):
    """
    """
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
    mix_rule: str="stoch"

class State(NamedTuple):
    """
    """
    A: Float[Array, "X Y C"] #Cells activations
    P: Float[Array, "X Y K"] #Embedded parameters
    fK:jax.Array             #Kernels fft

class Params(NamedTuple):
    R: Float[Array, ""]
    r: Float[Array, "k"]
    m: Float[Array, "k"]
    s: Float[Array, "k"]
    a: Float[Array, "k 3"]
    b: Float[Array, "k 3"]
    w: Float[Array, "k 3"]

"""
The Flow Lenia substrate.
"""
class FlowLeniaParams():
    def __init__(self, cfg: ConfigFLP, callback: Optional[Callable]=None):
        self.cfg = cfg
        self.RT = ReintegrationTracking(cfg.X, cfg.Y, cfg.dt, cfg.dd, cfg.sigma, cfg.border, has_hidden=True, mix=cfg.mix_rule)
        self.clbck = callback

    def default_params(self, rng):
        kR, kr, km, ks, ka, kb, kw = jr.split(rng, 7)
        R = jr.uniform(kR, (    ), minval=2.000, maxval=25.0)
        r = jr.uniform(kr, (self.cfg.k,  ), minval=0.200, maxval=1.00)
        m = jr.uniform(km, (self.cfg.k,  ), minval=0.050, maxval=0.50) 
        s = jr.uniform(ks, (self.cfg.k,  ), minval=0.001, maxval=0.18)
        a = jr.uniform(ka, (self.cfg.k, 3), minval=0.000, maxval=1.00)
        b = jr.uniform(kb, (self.cfg.k, 3), minval=0.001, maxval=1.00)
        w = jr.uniform(kw, (self.cfg.k, 3), minval=0.010, maxval=0.50)
        
        # Unsure if this is the right thing to return
        return Params(R=R, r=r, m=m, s=s, a=a, b=b, w=w)
    
    def init_state(self, rng, params: Params):
        """Compute the kernels fft and put dummy arrays as placeholders for A and P"""
        A = jnp.zeros((self.cfg.X, self.cfg.Y, self.cfg.C))
        P = jnp.zeros((self.cfg.X, self.cfg.Y, self.cfg.k))
        fK = get_kernels_fft(self.cfg.X, self.cfg.Y, self.cfg.k, params.R, params.r, params.a, params.w, params.b)
        state = State(A=A, P=P, fK=fK)
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
        A, P = state.A, state.P
            # --- Original Lenia ---
        fA = jnp.fft.fft2(A, axes=(0,1))  # (x,y,c)

        fAk = fA[:, :, self.cfg.c0]  # (x,y,k)

        U = jnp.real(jnp.fft.ifft2(state.fK * fAk, axes=(0,1)))  # (x,y,k)

        U = growth(U, params.m, params.s) * P # (x,y,k)

        U = jnp.dstack([ U[:, :, self.cfg.c1[c]].sum(axis=-1) for c in range(self.cfg.C) ])  # (x,y,c)

        # --- FLOW ---

        F = sobel(U) #(x, y, 2, c) : Flow

        C_grad = sobel(A.sum(axis = -1, keepdims = True)) #(x, y, 2, 1) : concentration gradient

        alpha = jnp.clip((A[:, :, None, :]/2)**2, .0, 1.)

        F = jnp.clip(F * (1 - alpha) - C_grad * alpha, 
                     -(self.cfg.dd-self.cfg.sigma), 
                     self.cfg.dd - self.cfg.sigma)

        nA, nP = self.RT(A, P, F) #type:ignore

        state = state._replace(A=nA, P=nP)

        # --- Callback ---

        if self.clbck is not None:
            state = self.clbck(state, rng)
        
        # ---
        return state
    
    def render_state(self, state, params, img_size=None):
        A, P = state.A, state.P
        img = P[..., :3] * A.sum(-1, keepdims=True)
        if img_size is not None:
            img = jax.image.resize(img, (img_size, img_size, 3), method='nearest')
        return img
    
    # remove
    def rollout_(self, state: State, params, key: Optional[jax.Array]=None, 
                 steps: int=100)->State:
        return jax.lax.fori_loop(0, steps, lambda i,s: self.step_state(key, s, params), state)
    
    def rollout(self, state: State, params, key: Optional[jax.Array]=None, 
                steps: int=100)->Tuple[State, State]:
        def _step(c, x):
            s, k = c
            k, k_ = jr.split(k)
            s = self.step_state(k_, s, params)
            return [s,k],s
        [s, _], S = jax.lax.scan(_step, [state,key], None, steps)
        return s, S


#===========================================================================================
#====================================Simulaton utils========================================
#===========================================================================================


def beam_mutation(state: State, key: jax.Array, sz: int=20, p: float=0.01):
    kmut, kloc, kp = jr.split(key, 3)
    P = state.P
    k = P.shape[-1]
    mut = jnp.ones((sz,sz,k)) * jr.normal(kmut, (1,1,k))
    loc = jr.randint(kloc, (3,), minval=0, maxval=P.shape[0]-sz).at[-1].set(0)
    dP = jax.lax.dynamic_update_slice(jnp.zeros_like(P), mut, loc)
    m = (jr.uniform(kp, ()) < p).astype(float)
    P = P + dP*m
    return state._replace(P=P)

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    cfg = ConfigFLP(X=500, Y=500, C=3, k=9)
    M = np.array([[2, 1, 0],
                  [0, 2, 1],
                  [1, 0, 2]])
    c0, c1 = conn_from_matrix(M)
    cfg = cfg._replace(c0=c0, c1=c1)
    flp = FlowLeniaParams(cfg, callback=partial(beam_mutation, sz=20, p=0.1))
    params = flp.default_params(jr.key(0))
    s = flp.init_state(jr.key(1), params)
    locs = jnp.arange(20) + (cfg.X//2-10)
    A = s.A.at[jnp.ix_(locs, locs)].set(jr.uniform(jr.key(2), (20, 20, 3)))
    P = s.P.at[jnp.ix_(locs, locs)].set(jnp.ones((20, 20, 9))*jr.uniform(jr.key(111), (1, 1, 9)))
    s = s._replace(A=A, P=P)
    s, S = flp.rollout(s, params, jr.key(1), 2)
    display_flp(S)