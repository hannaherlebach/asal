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
rollout_fn = partial(rollout_simulation, s0=None, substrate=substrate, fm=fm, rollout_steps=5000, time_sampling=100, img_size=224, return_state=True) # create the rollout function
rollout_fn = jax.jit(rollout_fn) # jit for speed
# now you can use rollout_fn as you need...
rng = jax.random.PRNGKey(0)
params = substrate.default_params(rng) # sample random parameters
rollout_data = rollout_fn(rng, params)
rgb = rollout_data['rgb'] # shape: (8, 224, 224, 3)
z = rollout_data['z'] # shape: (8, 512)
states = rollout_data['state'] # shape: (8, 128, 128, 3) # wait, what is this state object?

# Uncomment if time_sampling is not 'final' (needs multiple steps)
# oe_score = asal_metrics.calc_open_endedness_score(z) # shape: ()


# Display the simulation rollout as an animation
fig, ax = plt.subplots()
ims = []
print(rgb[0])
for i in range(rgb.shape[0]):
    im = ax.imshow(rgb[i], animated=True)
    ims.append([im])
ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True, repeat_delay=1000)
plt.show()


# Save animation as a GIF
save_path = "animations/flenia_alt.gif"
os.makedirs("animations", exist_ok=True)
ani.save(save_path, writer=animation.PillowWriter(fps=10))  # Adjust FPS as needed

print(f"Animation saved as {save_path}")
