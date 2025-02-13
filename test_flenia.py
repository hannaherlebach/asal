
import jax
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from functools import partial
import substrates
import foundation_models
from rollout import rollout_simulation
import asal_metrics

fm = foundation_models.create_foundation_model('clip')
substrate = substrates.create_substrate('flenia_params')
rollout_fn = partial(rollout_simulation, s0=None, substrate=substrate, fm=fm, rollout_steps=substrate.rollout_steps, time_sampling=8, img_size=224, return_state=False) # create the rollout function
rollout_fn = jax.jit(rollout_fn) # jit for speed
# now you can use rollout_fn as you need...
rng = jax.random.PRNGKey(0)
params = substrate.default_params(rng) # sample random parameters
rollout_data = rollout_fn(rng, params)
rgb = rollout_data['rgb'] # shape: (8, 224, 224, 3)
z = rollout_data['z'] # shape: (8, 512)
oe_score = asal_metrics.calc_open_endedness_score(z) # shape: ()

# Display the simulation rollout as an animation
# fig, ax = plt.subplots()
# ims = []
# for i in range(rgb.shape[0]):
#     im = ax.imshow(rgb[i])
#     ims.append([im])
# ani = animation.ArtistAnimation(fig, ims, interval=200, blit=True, repeat_delay=1000)
print(oe_score)
plt.imshow(rgb[0])
plt.show()
