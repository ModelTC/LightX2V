import os

import torch

from diffusers.models import AutoencoderKLFlux2
from diffusers.pipelines.flux2.image_processor import Flux2ImageProcessor
from lightx2v.utils.envs import GET_DTYPE
from lightx2v_platform.base.global_var import AI_DEVICE


class Flux2KleinVAE:
    def __init__(self, config):
        self.config = config
        self.cpu_offload = config.get("vae_cpu_offload", config.get("cpu_offload", False))
        self.latent_channels = config.get("latent_channels", 16)
        self.vae_scale_factor = config.get("vae_scale_factor", 8)
        self.load()

    def load(self):
        model_path = self.config["model_path"]
        kwargs = {}
        if not os.path.exists(model_path):
            vae_path = model_path
            kwargs["subfolder"] = "vae"
        else:
            vae_path = self.config.get("vae_path", os.path.join(model_path, "vae"))
        target_device = "cpu" if self.cpu_offload else AI_DEVICE
        self.vae = AutoencoderKLFlux2.from_pretrained(vae_path, torch_dtype=GET_DTYPE(), **kwargs).to(target_device)
        self.image_processor = Flux2ImageProcessor(vae_scale_factor=self.vae_scale_factor)

        if self.config.get("use_tiling_vae", False):
            self.vae.enable_tiling()

    def to(self, device):
        self.vae.to(device)
        return self

    @torch.no_grad()
    def encode_vae_image(self, image):
        if self.cpu_offload:
            self.vae.to(AI_DEVICE)

        encoded = self.vae.encode(image.to(AI_DEVICE, dtype=GET_DTYPE()))

        if self.cpu_offload:
            self.vae.to("cpu")

        return encoded

    @torch.no_grad()
    def decode(self, latents, input_info=None):
        if self.cpu_offload:
            self.vae.to(AI_DEVICE)

        image = self.vae.decode(latents.to(AI_DEVICE, dtype=GET_DTYPE()))[0]
        image = self.image_processor.postprocess(image)

        if self.cpu_offload:
            self.vae.to("cpu")

        return image
