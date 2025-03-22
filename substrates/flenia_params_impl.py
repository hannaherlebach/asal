"""
This is an implementation of FlowLeniaParams https://github.com/erwanplantec/FlowLenia/blob/main/flowlenia/flowlenia_params.py in the form of the code taken from the EvoFlow notebook for ES optimisation of FlowLenia parameters. https://colab.research.google.com/drive/18OGQqdzqAZeiTJjHukJ0ieFBlswi4eiG?usp=sharing#scrollTo=92ME_WBRNXnw
"""
import jax
import jax.numpy as jnp
import jax.scipy as jsp
import chex
import numpy as np
from functools import partial
from tqdm import tqdm
import typing as t
import matplotlib.pyplot as plt
import pickle
import os

from jax.experimental import host_callback

def sigmoid(x):
    return 0.5 * (jnp.tanh(x / 2) + 1)

# The division by w seems to have been causing the NaNs, so clip it in compute_kernels()
ker_f = lambda x, a, w, b : (b * jnp.exp( - (x[..., None] - a)**2 / w)).sum(-1)

bell = lambda x, m, s: jnp.exp(-((x-m)/s)**2 / 2)

def growth(U, m, s):
    return bell(U, m, s)*2-1

kx = jnp.array([
                [1., 0., -1.],
                [2., 0., -2.],
                [1., 0., -1.]
])
ky = jnp.transpose(kx)
def sobel_x(A):
    """
    A : (x, y, c)
    ret : (x, y, c)
    """
    return jnp.dstack([jsp.signal.convolve2d(A[:, :, c], kx, mode = 'same')
                    for c in range(A.shape[-1])])
def sobel_y(A):
    return jnp.dstack([jsp.signal.convolve2d(A[:, :, c], ky, mode = 'same')
                    for c in range(A.shape[-1])])

@jax.jit
def sobel(A):
    return jnp.concatenate((sobel_y(A)[:, :, None, :], sobel_x(A)[:, :, None, :]),
                            axis = 2)



def get_kernels(SX: int, SY: int, nb_k: int, params):
    mid = SX//2
    Ds = [ np.linalg.norm(np.mgrid[-mid:mid, -mid:mid], axis=0) /
          ((params['R']+15) * params['r'][k]) for k in range(nb_k) ]  # (x,y,k)
    K = jnp.dstack([sigmoid(-(D-1)*10) * ker_f(D, params["a"][k], params["w"][k], params["b"][k])
                    for k, D in zip(range(nb_k), Ds)])
    nK = K / jnp.sum(K, axis=(0,1), keepdims=True)
    return nK


def conn_from_matrix(mat):
    C = mat.shape[0]
    c0 = []
    c1 = [[] for _ in range(C)]
    i = 0
    for s in range(C):
        for t in range(C):
            n = mat[s, t]
            if n:
                c0 = c0 + [s]*n
                c1[t] = c1[t] + list(range(i, i+n))
            i+=n
    return c0, c1


def conn_from_lists(c0, c1, C):
    return c0, [[i == c1[i] for i in range(len(c0))] for c in range(C)]


@jax.jit
def stack_trees(*trees):
    return jax.tree_map(lambda *trees : jnp.stack(list(trees)), *trees)

@partial(jax.jit, static_argnums=(1,))
def unstack_tree(tree, dims):
    return [jax.tree_map(lambda x : x[i], tree) for i in range(dims)]

@jax.jit
def add_trees(t1, t2):
    return jax.tree_map(lambda x1, x2 : x1 + x2, t1, t2)

@jax.jit
def clip_tree(tree, lower, upper):
    return jax.tree_map(lambda x, l, u : jnp.clip(x, l, u), tree, lower, upper)

def center_of_mass(im, SX, SY):
    """
    im: array (W, H, C)
    SX: int (width of image)
    SY: int (height of image)
    """
    im = im[:, :, 0]
    mass = im.sum()
    x, y = jnp.arange(SX), jnp.arange(SY)
    xx, yy = jnp.meshgrid(x, y)
    X, Y = xx - SX / 2, yy - SY / 2

    # Centroids
    cx = (X * im).sum() / (mass + 1e-10)
    cy = (Y * im).sum() / (mass + 1e-10)

    z = jnp.zeros(2)
    z = z.at[0].set(cx/SX)
    z = z.at[1].set(cy/SY)

    return z

def ring_sampling(key, dmin, dmax, n=30):
    ang_key, dist_key = jax.random.split(key)
    angles = jax.random.uniform(ang_key, shape=(n,), minval=0, maxval=2 * np.pi)  # in radians
    dists = jax.random.uniform(dist_key, shape = (n,), minval=dmin, maxval=dmax)
    return jnp.dstack([dists * jnp.cos(angles), dists * jnp.sin(angles)])[0]

# ---- FlowLeniaParams---

# Same parameters as FlowLenia

@chex.dataclass
class Params:
    r: jnp.ndarray # neighbourhood
    b: jnp.ndarray # kernel param
    w: jnp.ndarray # kernel param
    a: jnp.ndarray # kernel param
    m: jnp.ndarray # growth fn param
    s: jnp.ndarray # growth fn param
    h: jnp.ndarray # kernel param
    R: float

# I'm guessing same CompiledParams works as well?
@chex.dataclass
class CompiledParams:
    fK: jnp.ndarray
    m: jnp.ndarray
    s: jnp.ndarray
    h: jnp.ndarray


### ADD RULESPACE AND UTILS


### --- FLOW LENIA PARAM ---

@chex.dataclass
class FLP_Config :
    SX: int
    SY: int
    nb_k: int 
    C: int
    c0: t.Iterable
    c1: t.Iterable
    dt: float = .2
    dd: int = 5
    sigma: float = .65
    n: int = 2
    theta_A : float = 1.
    border: str = 'wall'
    mix_rule: str = 'stoch' # diff from flenia

@chex.dataclass
class FLP_State :
    A: jnp.ndarray # (X, Y, C) cell activations
    P: jnp.ndarray # (X, Y, K) embedded parameters

class FlowLeniaParams:
    def __init__(self, config: FLP_Config):
        self.cfg = config
        pass

    def _build_step_fn(self):
        pass
    
        def step_flow(rollx:int, rolly:int, A: jnp.ndarray, P: jnp.ndarray, mus: jnp.ndarray):
            """
            Computes quantity of matter arriving from neighbours.
            
            (Is this reintegration tracking?)
            """
            rollA = jnp.roll(A, (rollx, rolly), axis=(0,1))
            rollP = jnp.roll(P, (rollx, rolly), axis=(0,1))
            rollmu = jnp.roll(mus, (rollx, rolly), axis=(0,1))

            if self.border == 'torus':
            pass

            return nA, nP

        def step(state: FLP_State, params: CompiledParams):
            A, P = state.A, state.P

            # --- Original Lenia ---
            fA = jnp.fft.fft2(A, axes=(0,1))  # (x,y,c)
            fAk = fA[:, :, self.config.c0]  # (x,y,k)
            U = jnp.real(jnp.fft.ifft2(params.fK * fAk, axes=(0,1)))  # (x,y,k)
            U = growth(U, params.m, params.s) * params.h  # (x,y,k)
            U = jnp.dstack([ U[:, :, self.config.c1[c]].sum(axis=-1) for c in range(self.config.C) ])  # (x,y,c)

            # --- Flow ---
            F = sobel(U) #(x, y, 2, c) : Flow
            C_grad = sobel(A.sum(axis = -1, keepdims = True)) #(x, y, 2, 1) : concentration gradient
            alpha = jnp.clip((A[:, :, None, :]/2)**2, .0, 1.)
            F = jnp.clip(F * (1 - alpha) - C_grad * alpha, 
                        -(self.cfg.dd-self.cfg.sigma), 
                        self.cfg.dd - self.cfg.sigma)
            
            # Now it gets different, gotta use step_flow rather than self.RT

            # Original FlowLeniaParam:

            # nA, nP = self.RT(A, P, F) #type:ignore
            # state = state._replace(A=nA, P=nP)

            # Alternative: 

            nA, nP = self.step_flow(...)


        return step
    
    # Don't think I need those extra rollout functions