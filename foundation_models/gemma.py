import jax, jax.numpy as jnp
from gemma import gm

class Gemma3():
    def __init__(self, gemma_checkpoint_path=gm.ckpts.CheckpointPath.GEMMA3_4B_IT):
        model = gm.nn.Gemma3_4B()
        params = gm.ckpts.load_params(gemma_checkpoint_path)
        tokenizer = gm.text.Gemma3Tokenizer()

        self.sampler = gm.text.Sampler(
            model=model,
            params=params,
            tokenizer=tokenizer
        )

    def generate_response(self, user_instruction, images, max_length=128):
        """
        Takes user instruction and images, correctly formats them to pass to Gemma 3, and returns the model response.

        Args:
            instruction_prompt (str): instruction prompt.
            images (array): (N, H, W, C) array of images.
        Returns:
            response (str): model response..
        """
        prompt = f"<start_of_turn>user\n{user_instruction}"
        prompt += "\n".join("<start_of_image>" for _ in range(images.shape[0]))
        prompt += "<end_of_turn>\n<start_of_turn>model"

        response = self.sampler.sample(prompt=prompt, images=images, max_new_tokens=max_length)

        return response