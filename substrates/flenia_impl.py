"""
This combines flowlenia.py, utils.py, vizutils.py and reintegration_tracking.py from https://github.com/erwanplantec/FlowLenia/tree/main/flowlenia.
"""
import numpy as np
import jax
import jax.numpy as jnp
import jax.scipy as jsp
import jax.random as jr
import equinox as eqx
from typing import NamedTuple, Optional, Tuple
from jaxtyping import Float, Array
from functools import partial
import matplotlib.pyplot as plt
import matplotlib.animation as animation


# --- FlowLenia utils ---

def sigmoid(x):
    return 0.5 * (jnp.tanh(x / 2) + 1)

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



def get_kernels_fft(X, Y, k, R, r, a, w, b):

    """Compute kernels and return a dic containing kernels fft
    
    Args:
        params (Params): raw params of the system
    
    Returns:
        CompiledParams: compiled params which can be used as update rule
    """
    mid = X//2
    Ds = [ np.linalg.norm(np.mgrid[-mid:mid, -mid:mid], axis=0) / 
          ((R+15) * r[k]) for k in range(k) ]  # (x,y,k)
    K = jnp.dstack([sigmoid(-(D-1)*10) * ker_f(D, a[k], w[k], b[k]) 
                    for k, D in zip(range(k), Ds)])
    nK = K / jnp.sum(K, axis=(0,1), keepdims=True)  # Normalize kernels 
    fK = jnp.fft.fft2(jnp.fft.fftshift(nK, axes=(0,1)), axes=(0,1))  # Get kernels fft

    return fK


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
    return c0, [[i == c1[i] for i in range(len(c0))] for _ in range(C)]

# --- Reintegration tracking ---

class ReintegrationTracking:

    #-------------------------------------------------------------------

    def __init__(self, SX=256, SY=256, dt=.2, dd=5, sigma=.65, border="wall", has_hidden=False, 
                 mix="stoch"):
        self.SX = SX
        self.SY = SY
        self.dt = dt
        self.dd = dd
        self.sigma = sigma
        self.has_hidden = has_hidden
        self.border = border if border in ['wall', 'torus'] else 'wall'
        self.mix = mix

    #-------------------------------------------------------------------

    def __call__(self, *args, **kwargs):
        
        if self.has_hidden:
            return self._apply_with_hidden(*args, **kwargs)
        else:
            return self._apply_without_hidden(*args, **kwargs)

    #-------------------------------------------------------------------

    def _apply_without_hidden(self, A: jax.Array, F: jax.Array)->jax.Array:

        x, y = jnp.arange(self.SX), jnp.arange(self.SY)
        X, Y = jnp.meshgrid(x, y)
        pos = jnp.dstack((Y, X)) + .5 #(SX, SY, 2)
        dxs = []
        dys = []
        dd = self.dd
        for dx in range(-dd, dd+1):
            for dy in range(-dd, dd+1):
                dxs.append(dx)
                dys.append(dy)
        dxs = jnp.array(dxs)
        dys = jnp.array(dys)

        @partial(jax.vmap, in_axes=(None, None, 0, 0))
        def step(A, mu, dx, dy):
            Ar = jnp.roll(A, (dx, dy), axis=(0, 1))
            mur = jnp.roll(mu, (dx, dy), axis=(0, 1))
            if self.border == 'torus':
                dpmu = jnp.min(jnp.stack(
                    [jnp.absolute(pos[..., None] - (mur + jnp.array([di, dj])[None, None, :, None])) 
                    for di in (-self.SX, 0, self.SX) for dj in (-self.SY, 0, self.SY)]
                ), axis = 0)
            else :
                dpmu = jnp.absolute(pos[..., None] - mur)
            sz = .5 - dpmu + self.sigma
            area = jnp.prod(jnp.clip(sz, 0, min(1, 2*self.sigma)) , axis = 2) / (4 * self.sigma**2)
            nA = Ar * area
            return nA

        ma = self.dd - self.sigma  # upper bound of the flow maggnitude
        mu = pos[..., None] + jnp.clip(self.dt * F, -ma, ma) #(x, y, 2, c) : target positions (distribution centers)
        if self.border == "wall":
            mu = jnp.clip(mu, self.sigma, self.SX-self.sigma)

        nA = step(A, mu, dxs, dys).sum(0)
        
        return nA

    #-------------------------------------------------------------------

    def _apply_with_hidden(self, A: jax.Array, H: jax.Array, F: jax.Array):

        x, y = jnp.arange(self.SX), jnp.arange(self.SY)
        X, Y = jnp.meshgrid(x, y)
        pos = jnp.dstack((Y, X)) + .5 #(SX, SY, 2)
        dxs = []
        dys = []
        dd = self.dd
        for dx in range(-dd, dd+1):
            for dy in range(-dd, dd+1):
                dxs.append(dx)
                dys.append(dy)
        dxs = jnp.array(dxs)
        dys = jnp.array(dys)
        
        @partial(jax.vmap, in_axes = (None, None, None, 0, 0))
        def step_flow(A, H, mu, dx, dy):
            """Summary
            """
            Ar = jnp.roll(A, (dx, dy), axis = (0, 1))
            Hr = jnp.roll(H, (dx, dy), axis = (0, 1)) #(x, y, k)
            mur = jnp.roll(mu, (dx, dy), axis = (0, 1))

            if self.border == 'torus':
                dpmu = jnp.min(jnp.stack(
                    [jnp.absolute(pos[..., None] - (mur + jnp.array([di, dj])[None, None, :, None])) 
                    for di in (-self.SX, 0, self.SX) for dj in (-self.SY, 0, self.SY)]
                ), axis = 0)
            else :
                dpmu = jnp.absolute(pos[..., None] - mur)

            sz = .5 - dpmu + self.sigma
            area = jnp.prod(jnp.clip(sz, 0, min(1, 2*self.sigma)) , axis = 2) / (4 * self.sigma**2)
            nA = Ar * area
            return nA, Hr

        ma = self.dd - self.sigma  # upper bound of the flow maggnitude
        mu = pos[..., None] + jnp.clip(self.dt * F, -ma, ma) #(x, y, 2, c) : target positions (distribution centers)
        if self.border == "wall":
            mu = jnp.clip(mu, self.sigma, self.SX-self.sigma)
        nA, nH = step_flow(A, H, mu, dxs, dys)

        if self.mix == 'avg':
            nH = jnp.sum(nH * nA.sum(axis = -1, keepdims = True), axis = 0)  
            nA = jnp.sum(nH, axis = 0)
            nH = nH / (nA.sum(axis = -1, keepdims = True)+1e-10)

        elif self.mix == "softmax":
            expnA = jnp.exp(nA.sum(axis = -1, keepdims = True)) - 1
            nA = jnp.sum(nA, axis = 0)
            nH = jnp.sum(nH * expnA, axis = 0) / (expnA.sum(axis = 0)+1e-10) #avg rule

        elif self.mix == "stoch":
            categorical=jax.random.categorical(
              jax.random.PRNGKey(42), 
              jnp.log(nA.sum(axis=-1, keepdims=True)), 
              axis=0)
            mask=jax.nn.one_hot(categorical,num_classes=(2*self.dd+1)**2,axis=-1)
            mask=jnp.transpose(mask,(3,0,1,2)) 
            nH = jnp.sum(nH * mask, axis = 0)
            nA = jnp.sum(nA, axis = 0)

        elif self.mix == "stoch_gene_wise":
            mask = jnp.concatenate(
              [jax.nn.one_hot(jax.random.categorical(
                                                    jax.random.PRNGKey(42), 
                                                    jnp.log(nA.sum(axis = -1, keepdims = True)), 
                                                    axis=0),
                              num_classes=(2*dd+1)**2,axis=-1)
              for _ in range(H.shape[-1])], 
              axis = 2)
            mask=jnp.transpose(mask,(3,0,1,2)) # (2dd+1**2, x, y, nb_k)
            nH = jnp.sum(nH * mask, axis = 0)
            nA = jnp.sum(nA, axis = 0)
        
        return nA, nH

    #-------------------------------------------------------------------

# --- FlowLenia implementation ---

class Config(NamedTuple):
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

class State(NamedTuple):
    A: Float[Array, "X Y C"]
    fK: Float[Array, "X Y k"]

class FlowLenia(eqx.Module):
    """
    """
    #-------------------------------------------------------------------
    # Parameters:
    R: Float
    r: Float[Array, "k"]
    m: Float[Array, "k"]
    s: Float[Array, "k"]
    h: Float[Array, "k"]
    a: Float[Array, "k 3"]
    b: Float[Array, "k 3"]
    w: Float[Array, "k 3"]
    # Statics
    cfg: Config
    RT: ReintegrationTracking
    #-------------------------------------------------------------------

    def __init__(self, cfg: Config, key: jax.Array):

        # ---
        self.cfg = cfg
        # ---
        kR, kr, km, ks, kh, ka, kb, kw = jr.split(key, 8)
        self.R = jr.uniform(kR, (    ), minval=2.000, maxval=25.0)
        self.r = jr.uniform(kr, (cfg.k,  ), minval=0.200, maxval=1.00)
        self.m = jr.uniform(km, (cfg.k,  ), minval=0.050, maxval=0.50) 
        self.s = jr.uniform(ks, (cfg.k,  ), minval=0.001, maxval=0.18)
        self.h = jr.uniform(kh, (cfg.k,  ), minval=0.010, maxval=1.00)
        self.a = jr.uniform(ka, (cfg.k, 3), minval=0.000, maxval=1.00)
        self.b = jr.uniform(kb, (cfg.k, 3), minval=0.001, maxval=1.00)
        self.w = jr.uniform(kw, (cfg.k, 3), minval=0.010, maxval=0.50)
        # ---
        self.RT = ReintegrationTracking(cfg.X, cfg.Y, cfg.dt, cfg.dd, cfg.sigma, 
                                        cfg.border, has_hidden=False)

    #-------------------------------------------------------------------

    @staticmethod
    def default_config():
        return Config()

    #-------------------------------------------------------------------

    def __call__(self, state: State, key: Optional[jax.Array]=None)->State:
        
        # --- Lenia ---
        A = state.A

        fA = jnp.fft.fft2(A, axes=(0,1))  # (x,y,c)

        fAk = fA[:, :, self.cfg.c0]  # (x,y,k)

        U = jnp.real(jnp.fft.ifft2(state.fK * fAk, axes=(0,1)))  # (x,y,k)

        U = growth(U, self.m, self.s) * self.h  # (x,y,k)

        U = jnp.dstack([ U[:, :, self.cfg.c1[c]].sum(axis=-1) for c in range(self.cfg.C) ])  # (x,y,c)

        # --- Flow ---

        nabla_U = sobel(U) #(x, y, 2, c)

        nabla_A = sobel(A.sum(axis = -1, keepdims = True)) #(x, y, 2, 1)

        alpha = jnp.clip((A[:, :, None, :]/self.cfg.C)**2, .0, 1.)

        F = nabla_U * (1 - alpha) - nabla_A * alpha
        nA = self.RT(A, F) #type:ignore

        return state._replace(A=nA)
    
    #-------------------------------------------------------------------

    def rollout(self, state: State, key: Optional[jax.Array]=None, 
                steps: int=100)->Tuple[State, State]:

        def _step(s, x):
            return self.__call__(s), s
        return jax.lax.scan(_step, state, None, steps)

    #-------------------------------------------------------------------

    def rollout_(self, state: State, key: Optional[jax.Array]=None, 
                 steps: int=100)->State:
        return jax.lax.fori_loop(0, steps, lambda i,s: self.__call__(s), state)

    #-------------------------------------------------------------------

    def initialize(self, key: jax.Array)->State:
        
        fK = get_kernels_fft(self.cfg.X, self.cfg.Y, self.cfg.k, self.R, self.r, 
                             self.a, self.w, self.b)
        A = jnp.zeros((self.cfg.X, self.cfg.Y, self.cfg.C))
        return State(A=A, fK=fK)

# --- Visualisation utils ---

def display_fl(states):
    ims = []
    fig, ax = plt.subplots()
    for i in range(100):
        A = states.A[i]
        C = A.shape[-1]
        if C==1:
            img = A
        if C==2:
            img=jnp.dstack([A[...,0], A[...,0], A[...,1]])
        else:
            img = A[...,:3]
        im = ax.imshow(img, animated=True)
        ims.append([im])
    _ = animation.ArtistAnimation(fig, ims, interval=50, blit=True,
                                    repeat_delay=1000)
    plt.show()

def display_flp(states):
    ims = []
    fig, ax = plt.subplots()
    for i in range(100):
        A, P = states.A[i], states.P[i]
        im = ax.imshow(P[..., :3] * A.sum(-1, keepdims=True), animated=True)
        ims.append([im])
    _ = animation.ArtistAnimation(fig, ims, interval=50, blit=True,
                                    repeat_delay=1000)
    plt.show()

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    cfg = Config(X=64, Y=64, C=3, k=9)
    M = np.array([[2, 1, 0],
                  [0, 2, 1],
                  [1, 0, 2]])
    c0, c1 = conn_from_matrix(M)
    cfg = cfg._replace(c0=c0, c1=c1)
    fl = FlowLenia(cfg, key=jr.key(101))
    s = fl.initialize(jr.key(2))
    locs = jnp.arange(20) + (cfg.X//2-10)
    A = s.A.at[jnp.ix_(locs, locs)].set(jr.uniform(jr.key(2), (20, 20, 1)))
    s = s._replace(A=A)
    s = fl.rollout_(s, None, 100)
    plt.imshow(s.A); plt.show()