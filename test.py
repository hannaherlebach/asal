import os
os.environ["JAX_PLATFORM_NAME"] = "cpu"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Disable GPU if necessary

import jax
jax.config.update("jax_platform_name", "cpu")  # Force CPU usage

print("JAX is using:", jax.default_backend())  # Should print "cpu"
print("Available devices:", jax.devices())  # Should list only CPU

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from functools import partial
import substrates
import foundation_models
from rollout import rollout_simulation
import asal_metrics
import jax.numpy as jnp

fm = foundation_models.create_foundation_model('clip')
substrate = substrates.create_substrate('flenia')
rollout_fn = partial(rollout_simulation, s0=None, substrate=substrate, fm=fm, rollout_steps=10000, time_sampling=100, img_size=224, return_state=True) # create the rollout function
rollout_fn = jax.jit(rollout_fn) # jit for speed
# now you can use rollout_fn as you need...
rng = jax.random.PRNGKey(0)
params = substrate.default_params(rng) # sample random parameters
rollout_data = rollout_fn(rng, params)
rgb = rollout_data['rgb'] # shape: (8, 224, 224, 3)
z = rollout_data['z'] # shape: (8, 512)

# Uncomment if time_sampling is not 'final' (needs multiple steps)
# oe_score = asal_metrics.calc_open_endedness_score(z) # shape: ()

# Display the final image
# plt.imshow(rgb[0])
# plt.show()

# Display the simulation rollout as an animation
# fig, ax = plt.subplots()
# ims = []
# for i in range(rgb.shape[0]):
#     im = ax.imshow(rgb[i])
#     ims.append([im])
# ani = animation.ArtistAnimation(fig, ims, interval=1000, blit=True, repeat_delay=1000)
# plt.show()


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


display_fl(rollout_data['state'])