import os
import argparse
import numpy as np
import imageio
from functools import partial
from tqdm.auto import tqdm

import jax
import jax.numpy as jnp
from jax.random import split

import torch
from transformers import (
    LlavaProcessor, 
    LlavaForConditionalGeneration,
    BitsAndBytesConfig,
)
from PIL import Image

import evosax
import substrates
import foundation_models
from rollout import rollout_simulation

# Load LLaVA model for video description
# Not Using MPS as Leads to `Generated Simulation Description: !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!`
# device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
model_id = "llava-hf/llava-interleave-qwen-0.5b-hf"

dtype = torch.float16 if device == "cuda" else torch.float32

model = LlavaForConditionalGeneration.from_pretrained(
    model_id, 
    torch_dtype=dtype, 
    low_cpu_mem_usage=True, 
).to(device)

processor = LlavaProcessor.from_pretrained(model_id)

def describe_video(video_frames, sample_rate=20):
    """
    Generate a description for the video by sampling frames and using LLaVA.
    """
    # Sample frames
    frames = [Image.fromarray(video_frames[i]).convert("RGB")  # Convert to PIL Image
              for i in range(0, len(video_frames), sample_rate)]
    
    print(f"Describing {len(frames)} frames")
    
    # Ensure we use the correct prompt format
    user_prompt = "Describe what is happening in the video."
    image_tokens = "<image>" * len(frames)
    prompt = f"<|im_start|>user{image_tokens}\n{user_prompt}<|im_end|><|im_start|>assistant"

    # Process images with the correct prompt
    inputs = processor(text=prompt, images=frames, return_tensors="pt").to(model.device, model.dtype)

    # Generate response
    output_ids = model.generate(**inputs, max_new_tokens=200, do_sample=False)
    final_description = processor.decode(output_ids[0][2:], skip_special_tokens=True)[len(user_prompt)+10:]

    return final_description

if __name__ == '__main__':
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Run a simulation and describe it")
    parser.add_argument('--substrate', type=str, default='gol', help="Simulation substrate")
    parser.add_argument('--rollout_steps', type=int, default=256, help="Number of simulation timesteps")
    parser.add_argument('--fps', type=int, default=15, help="Frames per second for the video")
    parser.add_argument('--output_path', type=str, default='simulation.mp4', help="Filename for the saved video")
    parser.add_argument('--seed', type=int, default=0, help="Random seed for reproducibility")
    args = parser.parse_args()

    # Initialize JAX PRNG key
    rng = jax.random.PRNGKey(args.seed)

    # Load foundation model and substrate
    fm = foundation_models.create_foundation_model('clip')
    substrate = substrates.create_substrate(args.substrate)
    substrate = substrates.FlattenSubstrateParameters(substrate)

    # Get default parameters
    params = substrate.default_params(rng)

    # Create a rollout function configured for video output
    rollout_fn = partial(
        rollout_simulation,
        s0=None,
        substrate=substrate,
        fm=fm,
        rollout_steps=args.rollout_steps,
        time_sampling='video',  # Capture all frames
        img_size=224,
        return_state=False
    )

    # Run the simulation and get image data
    data = rollout_fn(rng, params)

    # Convert frames from float [0,1] to uint8 [0,255] for video
    rgb_frames = np.array(data['rgb'])
    video_frames = (rgb_frames * 255).clip(0, 255).astype(np.uint8)

    # # Save frames as an mp4 video
    # imageio.mimsave(args.output_path, video_frames, fps=args.fps, codec="libx264")
    # print(f"Video saved to {args.output_path}")

    # Generate a textual description of the simulation
    video_prompt = describe_video(video_frames)
    print("\nGenerated Simulation Description:\n", video_prompt)
