import numpy as np
import jax
import jax.numpy as jnp
import jax.scipy as jsp
from functools import partial

import matplotlib.pyplot as plt
import matplotlib.animation as animation


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

# --- FlowLenia ----

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

#----------FlowLenia Params----------

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