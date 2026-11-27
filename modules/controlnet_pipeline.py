"""
Hugging Face Diffusers and ControlNet Pipeline integration.
Handles CUDA/CPU device placement, model caching, and graceful fallback.
"""

import os
import logging
from typing import Dict, Any, List, Optional
from PIL import Image
from utils.hardware import detect_hardware

logger = logging.getLogger("IndustrialDesignAI.controlnet")

class DiffusersControlNetPipeline:
    """Manages Stable Diffusion + ControlNet loading and inference."""

    def __init__(
        self,
        base_model_id: str = "runwayml/stable-diffusion-v1-5",
        controlnet_model_id: str = "lllyasviel/sd-controlnet-canny",
        cache_dir: Optional[str] = None
    ):
        self.base_model_id = base_model_id
        self.controlnet_model_id = controlnet_model_id
        self.cache_dir = cache_dir or os.getenv("HF_HOME")
        self.hw_info = detect_hardware()
        self.pipe = None
        self.is_loaded = False

    def load_pipeline(self) -> bool:
        """Attempts to load pipeline if dependencies and weights are available."""
        if not self.hw_info["has_diffusers"] or not self.hw_info["has_torch"]:
            logger.info("PyTorch or Diffusers not present. ControlNet pipeline in inactive state.")
            return False

        try:
            import torch
            from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, DPMSolverMultistepScheduler

            logger.info(f"Loading ControlNet model: {self.controlnet_model_id}")
            controlnet = ControlNetModel.from_pretrained(
                self.controlnet_model_id,
                torch_dtype=torch.float16 if self.hw_info["has_cuda"] else torch.float32,
                cache_dir=self.cache_dir
            )

            logger.info(f"Loading Base SD model: {self.base_model_id}")
            self.pipe = StableDiffusionControlNetPipeline.from_pretrained(
                self.base_model_id,
                controlnet=controlnet,
                torch_dtype=torch.float16 if self.hw_info["has_cuda"] else torch.float32,
                cache_dir=self.cache_dir,
                safety_checker=None
            )

            self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(self.pipe.scheduler.config)

            if self.hw_info["has_cuda"]:
                self.pipe.to("cuda")
                self.pipe.enable_attention_slicing()
            else:
                self.pipe.to("cpu")

            self.is_loaded = True
            logger.info("Stable Diffusion + ControlNet pipeline loaded successfully.")
            return True

        except Exception as e:
            logger.warning(f"Could not load Diffusers ControlNet pipeline: {e}")
            self.is_loaded = False
            return False

    def generate(
        self,
        prompt: str,
        negative_prompt: str,
        control_image: Image.Image,
        seed: int = 42,
        steps: int = 25,
        guidance_scale: float = 7.5,
        controlnet_conditioning_scale: float = 1.0,
        num_images: int = 1,
        width: int = 512,
        height: int = 512
    ) -> List[Image.Image]:
        """Runs inference through ControlNet pipeline."""
        if not self.is_loaded or self.pipe is None:
            raise RuntimeError("Diffusers ControlNet pipeline is not loaded on this system.")

        import torch
        generator = torch.Generator(device="cuda" if self.hw_info["has_cuda"] else "cpu").manual_seed(seed)

        # Ensure control image is RGB and matches dimensions
        control_resized = control_image.resize((width, height)).convert("RGB")

        outputs = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=control_resized,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            controlnet_conditioning_scale=controlnet_conditioning_scale,
            num_images_per_prompt=num_images,
            generator=generator,
            width=width,
            height=height
        )
        return outputs.images
