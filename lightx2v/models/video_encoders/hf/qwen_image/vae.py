import json
import os

import torch  # type: ignore
from diffusers import AutoencoderKLQwenImage
from diffusers.image_processor import VaeImageProcessor


class AutoencoderKLQwenImageVAE:
    def __init__(self, config):
        self.config = config
        self.model = AutoencoderKLQwenImage.from_pretrained(os.path.join(config.model_path, "vae")).to(torch.device("cuda")).to(torch.bfloat16)
        self.image_processor = VaeImageProcessor(vae_scale_factor=config.vae_scale_factor * 2)
        with open(os.path.join(config.model_path, "vae", "config.json"), "r") as f:
            vae_config = json.load(f)
            self.vae_scale_factor = 2 ** len(vae_config["temperal_downsample"]) if "temperal_downsample" in vae_config else 8
        self.dtype = torch.bfloat16

    @staticmethod
    def _unpack_latents(latents, height, width, vae_scale_factor):
        batch_size, num_patches, channels = latents.shape

        # VAE applies 8x compression on images but we must also account for packing which requires
        # latent height and width to be divisible by 2.
        height = 2 * (int(height) // (vae_scale_factor * 2))
        width = 2 * (int(width) // (vae_scale_factor * 2))

        latents = latents.view(batch_size, height // 2, width // 2, channels // 4, 2, 2)
        latents = latents.permute(0, 3, 1, 4, 2, 5)

        latents = latents.reshape(batch_size, channels // (2 * 2), 1, height, width)

        return latents

    @torch.no_grad()
    def decode(self, latents):
        width, height = self.config.aspect_ratios[self.config.aspect_ratio]
        latents = self._unpack_latents(latents, height, width, self.config.vae_scale_factor)
        latents = latents.to(self.dtype)
        latents_mean = torch.tensor(self.config.vae_latents_mean).view(1, self.config.vae_z_dim, 1, 1, 1).to(latents.device, latents.dtype)
        latents_std = 1.0 / torch.tensor(self.config.vae_latents_std).view(1, self.config.vae_z_dim, 1, 1, 1).to(latents.device, latents.dtype)
        latents = latents / latents_std + latents_mean
        images = self.model.decode(latents, return_dict=False)[0][:, :, 0]
        images = self.image_processor.postprocess(images, output_type="pil")
        return images
