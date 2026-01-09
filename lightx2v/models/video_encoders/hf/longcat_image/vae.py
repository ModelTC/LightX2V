import gc
import os

import torch

from lightx2v.utils.envs import *
from lightx2v_platform.base.global_var import AI_DEVICE

try:
    from diffusers import AutoencoderKL
    from diffusers.image_processor import VaeImageProcessor
except ImportError:
    AutoencoderKL = None
    VaeImageProcessor = None

torch_device_module = getattr(torch, AI_DEVICE)


class LongCatImageVAE:
    """VAE for LongCat Image model.

    Uses standard AutoencoderKL with LongCat-specific scaling/shift factors.
    Unlike QwenImage which uses 2x2 packing, LongCat uses patch_size=1.
    """

    def __init__(self, config):
        self.config = config
        self.cpu_offload = config.get("vae_cpu_offload", config.get("cpu_offload", False))
        if self.cpu_offload:
            self.device = torch.device("cpu")
        else:
            self.device = torch.device(AI_DEVICE)
        self.dtype = GET_DTYPE()

        # LongCat VAE parameters
        self.latent_channels = config.get("in_channels", 64)
        self.vae_scale_factor = config.get("vae_scale_factor", 8)

        # Scaling factors for LongCat
        # From diffusers LongCatImagePipeline: scaling_factor=0.3611, shift_factor=0.1159
        self.scaling_factor = 0.3611
        self.shift_factor = 0.1159

        self.load()

    def load(self):
        """Load the VAE model."""
        vae_path = self.config.get("vae_path", os.path.join(self.config["model_path"], "vae"))
        self.model = AutoencoderKL.from_pretrained(vae_path).to(self.device).to(self.dtype)
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)

        if self.config.get("use_tiling_vae", False):
            self.model.enable_tiling()

    @staticmethod
    def _unpack_latents(latents, height, width):
        """Unpack latents from [B, (H//2)*(W//2), C*4] to [B, C, H, W].

        Reverses the 2x2 packing used by Flux/LongCat.
        height and width are the packed spatial dims (half of VAE latent dims).
        """
        # Handle both [B, L, C] and [L, C] formats
        if latents.dim() == 2:
            latents = latents.unsqueeze(0)

        batch_size = latents.shape[0]
        # Packed channels = 64, original channels = 16
        channels = latents.shape[-1] // 4
        # -> [B, H//2, W//2, C, 2, 2]
        latents = latents.view(batch_size, height, width, channels, 2, 2)
        # -> [B, C, H//2, 2, W//2, 2]
        latents = latents.permute(0, 3, 1, 4, 2, 5)
        # -> [B, C, H, W] where H and W are full latent dims
        latents = latents.reshape(batch_size, channels, height * 2, width * 2)
        return latents

    @torch.no_grad()
    def decode(self, latents, input_info):
        """Decode latents to images.

        Args:
            latents: Latent tensor [B, (H//2)*(W//2), 64] - packed format
            input_info: Input information containing dimensions

        Returns:
            List of PIL images
        """
        if self.cpu_offload:
            self.model.to(AI_DEVICE)

        width, height = input_info.auto_width, input_info.auto_height
        # Full VAE latent dimensions
        full_latent_height = height // self.vae_scale_factor
        full_latent_width = width // self.vae_scale_factor
        # Packed dimensions (half due to 2x2 packing)
        packed_height = full_latent_height // 2
        packed_width = full_latent_width // 2

        # Unpack latents: [B, (H//2)*(W//2), 64] -> [B, 16, H, W]
        latents = self._unpack_latents(latents, packed_height, packed_width)
        latents = latents.to(self.dtype)

        # Apply inverse scaling: latents = (latents / scaling_factor) + shift_factor
        latents = (latents / self.scaling_factor) + self.shift_factor

        # Decode - latents is now [B, 16, H, W]
        images = self.model.decode(latents, return_dict=False)[0]
        images = self.image_processor.postprocess(images, output_type="pil")

        if self.cpu_offload:
            self.model.to(torch.device("cpu"))
            torch_device_module.empty_cache()
            gc.collect()

        return images
