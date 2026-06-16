import gc
import os

import torch

from lightx2v.utils.envs import GET_DTYPE
from lightx2v_platform.base.global_var import AI_DEVICE

try:
    from diffusers.image_processor import VaeImageProcessor
    from diffusers.models import AutoencoderKLFlux2
except ImportError:
    VaeImageProcessor = None
    AutoencoderKLFlux2 = None

torch_device_module = getattr(torch, AI_DEVICE)


class ErnieImageVAE:
    def __init__(self, config):
        self.config = config
        self.cpu_offload = config.get("vae_cpu_offload", config.get("cpu_offload", False))
        self.vae_scale_factor = config.get("vae_scale_factor", 16)
        self.load()

    def load(self):
        if AutoencoderKLFlux2 is None or VaeImageProcessor is None:
            raise ImportError("ERNIE-Image VAE requires diffusers with AutoencoderKLFlux2 support.")
        vae_path = self.config.get("vae_path", os.path.join(self.config["model_path"], "vae"))
        target_device = "cpu" if self.cpu_offload else AI_DEVICE
        self.vae = AutoencoderKLFlux2.from_pretrained(vae_path, torch_dtype=GET_DTYPE()).to(target_device)
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)
        if self.config.get("use_tiling_vae", False):
            self.vae.enable_tiling()

    @staticmethod
    def _unpatchify_latents(latents):
        batch_size, channels, height, width = latents.shape
        latents = latents.reshape(batch_size, channels // 4, 2, 2, height, width)
        latents = latents.permute(0, 1, 4, 2, 5, 3)
        return latents.reshape(batch_size, channels // 4, height * 2, width * 2)

    @torch.no_grad()
    def decode(self, latents, input_info=None):
        if self.cpu_offload:
            self.vae.to(AI_DEVICE)

        latents = latents.to(device=AI_DEVICE, dtype=GET_DTYPE())
        bn_mean = self.vae.bn.running_mean.view(1, -1, 1, 1).to(device=latents.device, dtype=latents.dtype)
        bn_std = torch.sqrt(self.vae.bn.running_var.view(1, -1, 1, 1) + 1e-5).to(
            device=latents.device,
            dtype=latents.dtype,
        )
        latents = latents * bn_std + bn_mean
        latents = self._unpatchify_latents(latents)
        images = self.vae.decode(latents, return_dict=False)[0]
        output_type = "pt" if input_info is not None and input_info.return_result_tensor else "pil"
        images = self.image_processor.postprocess(images, output_type=output_type)

        if self.cpu_offload:
            self.vae.to("cpu")
            torch_device_module.empty_cache()
            gc.collect()
        return images
