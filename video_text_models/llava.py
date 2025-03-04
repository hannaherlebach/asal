import torch
from transformers import LlavaProcessor, LlavaForConditionalGeneration
from PIL import Image

class LLAVA:
    def __init__(
        self,
        model_id="llava-hf/llava-interleave-qwen-0.5b-hf",
        device=None,
        torch_dtype=None,
    ):
        """
        Initializes the LLaVA model and its processor.

        Args:
            model_id (str): The identifier for the LLaVA model you want to load.
            device (str, optional): Device to run the model on, e.g. 'cuda', 'cpu'.
                                    If None, we infer from torch availability.
            torch_dtype (torch.dtype, optional): Data type to load the model with.
                                                 If None, defaults to float16 if
                                                 running on CUDA, else float32.
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        if torch_dtype is None:
            torch_dtype = torch.float16 if device == "cuda" else torch.float32
        self.torch_dtype = torch_dtype

        # Load processor & model
        self.processor = LlavaProcessor.from_pretrained(model_id)
        self.model = LlavaForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=self.torch_dtype,
            low_cpu_mem_usage=True,
        ).to(self.device)

    def describe_video(self, video_frames, sample_rate=20, user_prompt="Describe what is happening in the video."):
        """
        Generate a description for the video by sampling frames and using LLaVA.
        
        Args:
            video_frames (list or np.ndarray): A list/array of frames (shape HxWx3),
                                               typically in [0, 255] integer range
                                               or float range [0,1].
            sample_rate (int): Step size to sample frames. For example, a sample_rate
                               of 20 means we take every 20th frame.
            user_prompt (str): Text prompt appended after <image> tokens.

        Returns:
            final_description (str): The model-generated description.
        """
        # Convert frames to PIL Images and sample them
        frames = [
            Image.fromarray(frame).convert("RGB")
            for i, frame in enumerate(video_frames)
            if i % sample_rate == 0
        ]
        print(f"Describing {len(frames)} frames")

        # Construct prompt: Insert one <image> token for each frame
        image_tokens = "<image>" * len(frames)
        # LLaVA prompt format
        prompt = f"<|im_start|>user{image_tokens}\n{user_prompt}<|im_end|><|im_start|>assistant"

        # Processor expects the prompt + images
        inputs = self.processor(
            text=prompt,
            images=frames,
            return_tensors="pt"
        ).to(self.device, self.torch_dtype)

        # Generate response
        output_ids = self.model.generate(**inputs, max_new_tokens=200, do_sample=False)
        
        # Strip out the system tokens and partial user prompt
        # The [2:] is often used to remove system tokens; adjust as necessary
        raw_output = self.processor.decode(output_ids[0][2:], skip_special_tokens=True)
        # Remove the user_prompt (plus some buffer for offset)
        final_description = raw_output[len(user_prompt)+10:]

        return final_description
