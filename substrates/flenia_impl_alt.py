"""
This contains coThis contains code taken from the EvoFlow notebook for ES optimisation of FlowLenia parameters. https://colab.research.google.com/drive/18OGQqdzqAZeiTJjHukJ0ieFBlswi4eiG?usp=sharing#scrollTo=92ME_WBRNXnw
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

def save_scan(chkpt_file, save_rate=20, aux_infos={}):
    """Decorator for scanned func
    Saves carried state of scanned function every save_rate steps"""

    def _save(it_data, transform):
        iter, data = it_data
        with open(f"{chkpt_file}_{iter}.pickle", 'wb') as handle:
            pickle.dump({**data, **aux_infos}, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def save(iter, carry, y):
        _ = jax.lax.cond(
            iter % save_rate == 0,
            lambda _: host_callback.id_tap(_save,
                                           [iter, {'carry': carry,
                                                  'y': y}],
                                           result=iter),
            lambda _: iter,
            operand=None
        )

    def _save_scan(func):

        def _wrapped_func(carry, x):
            if type(x) is tuple:
                iter, *_ = x
            else:
                iter = x
            n_carry, y = func(carry, x)
            save(iter, carry, y)
            return n_carry, y

        return _wrapped_func

    return _save_scan

def plot_scan(plot_rate=20, xlabel="", ylabel="", title="", **scatter_kws):
    xs = []
    ys = []

    def _plot(xy, *_):
        x, y = xy
        xs.append(x); ys.append(y)
        if not x % plot_rate:
            clear_output(True)
            plt.scatter(xs, ys, **scatter_kws)
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
            plt.title(title)
            plt.show()
        return x

    def plot(iter, fits):
        _ = host_callback.call(_plot, [iter, fits], result_shape=(1))
        return _

    def _plot_scan(func):

        def _wrapped(carry, x):
            if type(x) is tuple:
                iter, *_ = x
            else:
                iter = x
            carry, ys = func(carry, x)
            plot(x, jnp.max(ys))
            #host_callback.barrier_wait()
            return carry, ys

        return _wrapped

    return _plot_scan

# --- FLowLenia ---

#==================================================================================================================
#====================================================UTILS=========================================================
#==================================================================================================================

@chex.dataclass
class Params:

    """Summary
    """

    r: jnp.ndarray # neighbourhood
    b: jnp.ndarray # kernel param
    w: jnp.ndarray # kernel param
    a: jnp.ndarray # kernel param
    m: jnp.ndarray # growth fn param
    s: jnp.ndarray # growth fn param
    h: jnp.ndarray # kernel param
    R: float


@chex.dataclass
class CompiledParams:

    """Summary
    """

    fK: jnp.ndarray
    m: jnp.ndarray
    s: jnp.ndarray
    h: jnp.ndarray



class Rule_space :

    """Rule space for Flow Lenia system

    Attributes:
        nb_k (int): number of kernels of the system
        spaces (TYPE): Description

    Deleted Attributes:
        init_shape (TYPE): Description
    """

    #-----------------------------------------------------------------------------
    def __init__(self, nb_k: int):
        """

        Args:
            nb_k (int): number of kernels in the update rule
        """
        self.nb_k = nb_k
        self.kernel_keys = 'r b w a m s h'.split()
        self.spaces = {
            "r" : {'low' : .2, 'high' : 1., 'mut_std' : .2, 'shape' : None},
            "b" : {'low' : .001, 'high' : 1., 'mut_std' : .2, 'shape' : (3,)},
            "w" : {'low' : .01, 'high' : .5, 'mut_std' : .2, 'shape' : (3,)},
            "a" : {'low' : .0, 'high' : 1., 'mut_std' : .2, 'shape' : (3,)},
            "m" : {'low' : .05, 'high' : .5, 'mut_std' : .2, 'shape' : None},
            "s" : {'low' : .001, 'high' : .18, 'mut_std' : .01, 'shape' : None},
            "h" : {'low' : .01, 'high' : 1., 'mut_std' : .2, 'shape' : None},
            'R' : {'low' : 2., 'high' : 25., 'mut_std' : .2, 'shape' : None},
        }
    #-----------------------------------------------------------------------------
    def sample(self, key: jnp.ndarray)->Params:
        """sample a random set of parameters

        Returns:
            Params: sampled parameters
        """
        kernels = {}
        for k in 'rmsh':
            key, subkey = jax.random.split(key)
            kernels[k] = jax.random.uniform(
              key=subkey, minval=self.spaces[k]['low'], maxval=self.spaces[k]['high'],
              shape=(self.nb_k,)
            )
        for k in "awb":
            key, subkey = jax.random.split(key)
            kernels[k] = jax.random.uniform(
              key=subkey, minval=self.spaces[k]['low'], maxval=self.spaces[k]['high'],
              shape=(self.nb_k, 3)
            )
        R = jax.random.uniform(key=key, minval=self.spaces['R']['low'], maxval=self.spaces['R']['high'])
        return Params(R=R, **kernels)



def compile_kernel_computer(SX: int, SY: int, nb_k: int)->t.Callable[[Params], CompiledParams]:
    """return a jit compiled function taking as input lenia raw params and returning computed kernels (compiled params)

    Args:
        SX (int): size of world in X
        SY (int): size of world in Y
        nb_k (int): number of kernels

    Returns:
        t.Callable[Params, CompiledParams]: function to compile raw params
    """
    mid = SX // 2
    def compute_kernels(params: Params)->CompiledParams:
        """Compute kernels and return a dic containing fft kernels, T and R

        Args:
            params (Params): raw params of the system

        Returns:
            CompiledParams: compiled params which can be used in update rule
        """

        Ds = [ np.linalg.norm(np.mgrid[-mid:mid, -mid:mid], axis=0) /
              ((params.R+15) * params.r[k]) for k in range(nb_k) ]  # (x,y,k)
        K = jnp.dstack([sigmoid(-(D-1)*10) * ker_f(D, params.a[k], params.w[k], params.b[k])
                        for k, D in zip(range(nb_k), Ds)])
        nK = K / jnp.sum(K, axis=(0,1), keepdims=True)  # Normalize kernels
        fK = jnp.fft.fft2(jnp.fft.fftshift(nK, axes=(0,1)), axes=(0,1))  # Get kernels fft

        return CompiledParams(fK=fK, m=params.m, s=params.s, h=params.h)

    return jax.jit(compute_kernels)

#==================================================================================================================
#==================================================FLOW LENIA======================================================
#==================================================================================================================

@chex.dataclass
class FL_Config :

    """Configuration of Flow Lenia system
    """
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

@chex.dataclass
class FL_State :

    """State of the system
    """

    A: jnp.ndarray

class FlowLenia :

    """class building the main functions of Flow Lenia

    Attributes:
        config (FL_Config): config of the system
        kernel_computer (TYPE): kernel computer
        rollout_fn (TYPE): rollout function
        rule_space (TYPE): Rule space of the system
        step_fn (Callable): system step function
    """

    #------------------------------------------------------------------------------

    def __init__(self, config: FL_Config):
        """

        Args:
            config (FL_Config): config of the system
        """
        self.config = config

        self.rule_space = Rule_space(config.nb_k)

        self.kernel_computer = compile_kernel_computer(config.SX, config.SY, config.nb_k)

        self.step_fn = self._build_step_fn()

        self.rollout_fn = self._build_rollout()

    #------------------------------------------------------------------------------

    def _build_step_fn(self)->t.Callable[[FL_State, CompiledParams], FL_State]:
        """Build step function of the system according to config

        Returns:
            t.Callable[t.Tuple[FL_State, CompiledParams], FL_State]: step function which outputs next state
            given a state and params
        """
        x, y = jnp.arange(self.config.SX), jnp.arange(self.config.SY)
        X, Y = jnp.meshgrid(x, y)
        pos = jnp.dstack((Y, X)) + .5 #(SX, SY, 2)

        rolls = []
        rollxs = []
        rollys = []
        dd = self.config.dd
        for dx in range(-dd, dd+1):
            for dy in range(-dd, dd+1):
                rolls.append((dx, dy))
                rollxs.append(dx)
                rollys.append(dy)
        rollxs = jnp.array(rollxs)
        rollys = jnp.array(rollys)


        @partial(jax.vmap, in_axes = (0, 0, None, None))
        def step_flow(rollx: int, rolly: int, A: jnp.ndarray, mus: jnp.ndarray)->jnp.ndarray:
            """Computes quantity of matter arriving from neighbors at x + [rollx, rolly] for all
            xs in the system (all locations)

            Args:
                rollx (int): offset of neighbors in x direction
                rolly (int): offset of neighbors in y direction
                A (jnp.ndarray): state of the system (SX, SY, C)
                mus (jnp.ndarray): target locations of all cells (SX, SY, 2, C)

            Returns:
                jnp.ndarray: quantities of matter arriving to all cells from their respective neighbor (SX, SY, C)
            """
            rollA = jnp.roll(A, (rollx, rolly), axis = (0, 1))
            rollmu = jnp.roll(mus, (rollx, rolly), axis = (0, 1))
            if self.config.border == 'torus':
                dpmu = jnp.min(jnp.stack(
                    [jnp.absolute(pos[..., None] - (rollmu + jnp.array([di, dj])[None, None, :, None]))
                    for di in (-self.config.SX, 0, self.config.SX) for dj in (-self.config.SY, 0, self.config.SY)]
                ), axis = 0)
            else :
                dpmu = jnp.absolute(pos[..., None] - rollmu)
            sz = .5 - dpmu + self.config.sigma
            area = jnp.prod(jnp.clip(sz, 0, min(1, 2*self.config.sigma)) , axis = 2) / (4 * self.config.sigma**2)
            nA = rollA * area
            return nA

        def step(state: FL_State, params: CompiledParams)->FL_State:
            """
            Main step

            Args:
                state (FL_State): state of the system
                params (CompiledParams): params

            Returns:
                FL_State: new state of the system

            """
            #---------------------------Original Lenia------------------------------------
            A = state.A

            fA = jnp.fft.fft2(A, axes=(0,1))  # (x,y,c)

            fAk = fA[:, :, self.config.c0]  # (x,y,k)

            U = jnp.real(jnp.fft.ifft2(params.fK * fAk, axes=(0,1)))  # (x,y,k)

            U = growth(U, params.m, params.s) * params.h  # (x,y,k)

            U = jnp.dstack([ U[:, :, self.config.c1[c]].sum(axis=-1) for c in range(self.config.C) ])  # (x,y,c)

            #-------------------------------FLOW------------------------------------------

            nabla_U = sobel(U) #(x, y, 2, c)

            nabla_A = sobel(A.sum(axis = -1, keepdims = True)) #(x, y, 2, 1)

            alpha = jnp.clip((A[:, :, None, :]/self.config.theta_A)**self.config.n, .0, 1.)

            F = nabla_U * (1 - alpha) - nabla_A * alpha

            ma = self.config.dd - self.config.sigma  # upper bound of the flow maggnitude
            mus = pos[..., None] + jnp.clip(self.config.dt * F, -ma, ma) #(x, y, 2, c) : target positions (distribution centers)
            if self.config.border == "wall":
                mus = jnp.clip(mus, self.config.sigma, self.config.SX-self.config.sigma)
            nA = step_flow(rollxs, rollys, A, mus).sum(axis = 0)

            return FL_State(A=nA)

        return step

    #------------------------------------------------------------------------------

    def _build_rollout(self)->t.Callable[[CompiledParams, FL_State, int], t.Tuple[FL_State, FL_State]]:
        """build rollout function

        Returns:
            t.Callable[t.Tuple[CompiledParams, FL_State, int], t.Tuple[FL_State, FL_State]]: Description
        """
        def scan_step(carry: t.Tuple[FL_State, CompiledParams], x)->t.Tuple[t.Tuple[FL_State, CompiledParams], FL_State]:
            """Summary

            Args:
                carry (t.Tuple[FL_State, CompiledParams]): Description
                x (TYPE): Description

            Returns:
                t.Tuple[t.Tuple[FL_State, CompiledParams], FL_State]: Description
            """
            state, params = carry
            nstate = jax.jit(self.step_fn)(state, params)
            return (nstate, params), nstate

        def rollout(params: CompiledParams, init_state: FL_State, steps: int) -> t.Tuple[FL_State, FL_State]:
            """Summary

            Args:
                params (CompiledParams): Description
                init_state (FL_State): Description
                steps (int): Description

            Returns:
                t.Tuple[FL_State, FL_State]: Description
            """
            return jax.lax.scan(scan_step, (init_state, params), None, length = steps)

        return rollout
    
def rollout_seq(c_params, As, rollout_fn):
    trajs  = stack_trees(
        *[rollout_fn(c_param, A, 500)[1] for c_param, A in zip(unstack_tree(c_params, 16),
                                                               unstack_tree(As, 16))]
    )
    return trajs

# (skip Lenia impl)

# --- VizUtils ---

# TODO they're currently for ipynb

# --- Metrics ---

def progress_bar_scan(num_samples, message=None):
    "Progress bar for a JAX scan"
    if message is None:
        #message = f"Running for {num_samples:,} iterations"
        message=""
    tqdm_bars = {}

    print_rate = 5 # if you run the sampler for less than 20 iterations
    remainder = num_samples % print_rate

    def _define_tqdm(arg, transform):
        tqdm_bars[0] = tqdm(range(num_samples))
        tqdm_bars[0].set_description(message, refresh=False)

    def _update_tqdm(arg, transform):
        tqdm_bars[0].update(arg)

    def _update_progress_bar(iter_num):
        "Updates tqdm progress bar of a JAX scan or loop"
        _ = jax.lax.cond(
            iter_num == 0,
            lambda _: host_callback.id_tap(_define_tqdm, None, result=iter_num),
            lambda _: iter_num,
            operand=None,
        )

        _ = jax.lax.cond(
            # update tqdm every multiple of `print_rate` except at the end
            (iter_num % print_rate == 0) & (iter_num != num_samples-remainder),
            lambda _: host_callback.id_tap(_update_tqdm, print_rate, result=iter_num),
            lambda _: iter_num,
            operand=None,
        )

        _ = jax.lax.cond(
            # update tqdm by `remainder`
            iter_num == num_samples-remainder,
            lambda _: host_callback.id_tap(_update_tqdm, remainder, result=iter_num),
            lambda _: iter_num,
            operand=None,
        )

    def _close_tqdm(arg, transform):
        tqdm_bars[0].close()

    def close_tqdm(result, iter_num):
        return jax.lax.cond(
            iter_num == num_samples-1,
            lambda _: host_callback.id_tap(_close_tqdm, None, result=result),
            lambda _: result,
            operand=None,
        )


    def _progress_bar_scan(func):
        """Decorator that adds a progress bar to `body_fun` used in `lax.scan`.
        Note that `body_fun` must either be looping over `np.arange(num_samples)`,
        or be looping over a tuple who's first element is `np.arange(num_samples)`
        This means that `iter_num` is the current iteration number
        """

        def wrapper_progress_bar(carry, x):
            if type(x) is tuple:
                iter_num, *_ = x
            else:
                iter_num = x
            _update_progress_bar(iter_num)
            result = func(carry, x)
            return close_tqdm(result, iter_num)

        return wrapper_progress_bar

    return _progress_bar_scan


# Test the FlowLenia class and visualise it
if __name__=="__main__":
    M = np.array([[3, 1, 0],
                [0, 3, 1],
                [1, 0, 3]], dtype=int)
    k = M.sum()
    c0, c1 = conn_from_matrix(M)
    nb_k = len(c0)
    C=1
    cfg = FL_Config(SX=128, SY=128, C=1, nb_k=nb_k,
                      c0=c0, c1=c1, dt=0.2, dd=4)
    fl = FlowLenia(cfg)

    params = fl.rule_space.sample(jax.random.PRNGKey(0))
    A = jnp.zeros((128,128,C)).at[44:84, 44:84, :].set(jax.random.uniform(jax.random.PRNGKey(42), (40, 40, C)))
    state = FL_State(A=A)

    c_params = compile_kernel_computer(128, 128, k)(params)

    traj = fl.rollout_fn(fl.kernel_computer(params), state, 1000)
    # traj[0] is tuple length 2, 0 item is FL_State, 1st item is a CompiledParams
    # traj[1] is a FL_State, where A has shape (10, 128, 128, 1) - this is a sequence of images


    img_sequence = traj[1].A
    # Animate the sequence along the first dim
    import matplotlib.animation as animation

    ims = []
    fig, ax = plt.subplots()
    for i in range(img_sequence.shape[0]):
        im = ax.imshow(img_sequence[i], animated=True)
        ims.append([im])
    ani = animation.ArtistAnimation(fig, ims, interval=5, blit=True,
                                    repeat_delay=1000)
    
    # Save as GIF
    save_path = "animations/flenia_impl_alt_2.gif"
    os.makedirs("animations", exist_ok=True)
    ani.save(save_path, writer=animation.PillowWriter(fps=20))  # Adjust FPS as needed

    plt.show()
