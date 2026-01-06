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

ASPECT_RATIO_MAP = {
    "16:9": [1664, 928],
    "9:16": [928, 1664],
    "1:1": [1328, 1328],
    "4:3": [1472, 1140],
    "3:4": [768, 1024],
}


class AutoencoderKLZImageVAE:
    def __init__(self, config):
        self.config = config

        self.cpu_offload = config.get("vae_cpu_offload", config.get("cpu_offload", False))
        if self.cpu_offload:
            self.device = torch.device("cpu")
        else:
            self.device = torch.device(AI_DEVICE)
        self.dtype = GET_DTYPE()
        self.latent_channels = 16
        self.vae_latents_mean = None
        self.vae_latents_std = None
        self.load()

    def load(self):
        self.model = AutoencoderKL.from_pretrained(os.path.join(self.config["model_path"], "vae")).to(self.device).to(torch.bfloat16)
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.config["vae_scale_factor"] * 2)

    @staticmethod
    def _unpack_latents(latents, latent_height, latent_width):
        batchsize, num_patches, channels = latents.shape
        num_channels_latents = channels // 4

        patch_height = latent_height // 2
        patch_width = latent_width // 2

        latents = latents.view(batchsize, patch_height, patch_width, num_channels_latents, 2, 2)
        latents = latents.permute(0, 3, 1, 4, 2, 5)
        latents = latents.reshape(batchsize, num_channels_latents, latent_height, latent_width)

        return latents

    @torch.no_grad()
    def decode(self, latents, input_info):
        if self.cpu_offload:
            self.model.to(torch.device("cuda"))

        latents = latents.to(next(self.model.parameters()).dtype)
        if hasattr(self.model.config, "scaling_factor") and hasattr(self.model.config, "shift_factor"):
            scaling_factor = self.model.config.scaling_factor
            shift_factor = self.model.config.shift_factor
            latents = (latents / scaling_factor) + shift_factor
        images = self.model.decode(latents, return_dict=False)[0]

        images_postprocessed = self.image_processor.postprocess(images, output_type="pil")
        images = images_postprocessed
        if self.cpu_offload:
            self.model.to(torch.device("cpu"))
            torch.cuda.empty_cache()
            gc.collect()
        return images

    @staticmethod
    def _pack_latents(latents, batchsize, num_channels_latents, height, width):
        latents = latents.view(batchsize, num_channels_latents, height // 2, 2, width // 2, 2)
        latents = latents.permute(0, 2, 4, 1, 3, 5)  # (batch_size, height//2, width//2, num_channels, 2, 2)
        latents = latents.reshape(batchsize, (height // 2) * (width // 2), num_channels_latents * 4)
        return latents

    def _encode_vae_image(self, image: torch.Tensor):
        encoder_output = self.model.encode(image)
        if hasattr(encoder_output, "latent_dist"):
            image_latents = encoder_output.latent_dist.mode()
        elif hasattr(encoder_output, "latents"):
            image_latents = encoder_output.latents
        else:
            raise AttributeError("Could not access latents from VAE encoder output")

        return image_latents

    @torch.no_grad()
    def encode_vae_image(self, image):
        if self.cpu_offload:
            self.model.to(torch.device("cuda"))

        image = image.to(self.model.device)

        if image.shape[1] != self.latent_channels:
            image_latents = self._encode_vae_image(image=image)
            # Apply scaling (inverse of decoding: decode does latents/scaling_factor + shift_factor)
            if hasattr(self.model.config, "scaling_factor") and hasattr(self.model.config, "shift_factor"):
                image_latents = (image_latents - self.model.config.shift_factor) * self.model.config.scaling_factor
        else:
            image_latents = image
        image_latents = torch.cat([image_latents], dim=0)
        if self.cpu_offload:
            self.model.to(torch.device("cpu"))
            torch.cuda.empty_cache()
            gc.collect()
        return image_latents
