import gc
import json
import os

import torch

from lightx2v.utils.envs import *
from lightx2v_platform.base.global_var import AI_DEVICE

try:
    from diffusers import AutoencoderKLQwenImage
    from diffusers.image_processor import VaeImageProcessor
except ImportError:
    AutoencoderKLQwenImage = None
    VaeImageProcessor = None


class AutoencoderKLQwenImageVAE:
    def __init__(self, config):
        self.config = config

        self.cpu_offload = config.get("vae_cpu_offload", config.get("cpu_offload", False))
        if self.cpu_offload:
            self.device = torch.device("cpu")
        else:
            self.device = torch.device(AI_DEVICE)
        self.dtype = GET_DTYPE()
        self.latent_channels = 16
        self.vae_latents_mean = [-0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508, 0.4134, -0.0715, 0.5517, -0.3632, -0.1922, -0.9497, 0.2503, -0.2921]
        self.vae_latents_std = [2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743, 3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.916]
        self.load()

    def load(self):
        self.model = AutoencoderKLQwenImage.from_pretrained(os.path.join(self.config["model_path"], "vae")).to(self.device).to(self.dtype)
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.config["vae_scale_factor"] * 2)
        with open(os.path.join(self.config["model_path"], "vae", "config.json"), "r") as f:
            vae_config = json.load(f)
            self.vae_scale_factor = 2 ** len(vae_config["temperal_downsample"]) if "temperal_downsample" in vae_config else 8
        if self.config.get("use_tiling_vae", False):
            self.model.enable_tiling()

    @staticmethod
    def _unpack_latents(latents, height, width, vae_scale_factor):
        batchsize, num_patches, channels = latents.shape

        # VAE applies 8x compression on images but we must also account for packing which requires
        # latent height and width to be divisible by 2.
        height = 2 * (int(height) // (vae_scale_factor * 2))
        width = 2 * (int(width) // (vae_scale_factor * 2))

        latents = latents.view(batchsize, height // 2, width // 2, channels // 4, 2, 2)
        latents = latents.permute(0, 3, 1, 4, 2, 5)

        latents = latents.reshape(batchsize, channels // (2 * 2), 1, height, width)

        return latents

    @torch.no_grad()
    def decode(self, latents, input_info):
        if self.cpu_offload:
            self.model.to(torch.device("cuda"))
        width, height = input_info.auto_width, input_info.auto_height
        latents = self._unpack_latents(latents, height, width, self.config["vae_scale_factor"])
        latents = latents.to(self.dtype)
        latents_mean = torch.tensor(self.vae_latents_mean).view(1, self.latent_channels, 1, 1, 1).to(latents.device, latents.dtype)
        latents_std = 1.0 / torch.tensor(self.vae_latents_std).view(1, self.latent_channels, 1, 1, 1).to(latents.device, latents.dtype)
        latents = latents / latents_std + latents_mean
        images = self.model.decode(latents, return_dict=False)[0][:, :, 0]
        images = self.image_processor.postprocess(images, output_type="pil")
        if self.cpu_offload:
            self.model.to(torch.device("cpu"))
            torch.cuda.empty_cache()
            gc.collect()
        return images

    @staticmethod
    # Copied from diffusers.pipelines.qwenimage.pipeline_qwenimage.QwenImagePipeline._pack_latents
    def _pack_latents(latents, batchsize, num_channels_latents, height, width):
        latents = latents.view(batchsize, num_channels_latents, height // 2, 2, width // 2, 2)
        latents = latents.permute(0, 2, 4, 1, 3, 5)
        latents = latents.reshape(batchsize, (height // 2) * (width // 2), num_channels_latents * 4)
        return latents

    def _encode_vae_image(self, image: torch.Tensor):
        image_latents = self.model.encode(image).latent_dist.mode()
        latents_mean = torch.tensor(self.model.config["latents_mean"]).view(1, self.latent_channels, 1, 1, 1).to(image_latents.device, image_latents.dtype)
        latents_std = torch.tensor(self.model.config["latents_std"]).view(1, self.latent_channels, 1, 1, 1).to(image_latents.device, image_latents.dtype)
        image_latents = (image_latents - latents_mean) / latents_std

        return image_latents

    @torch.no_grad()
    def encode_vae_image(self, image):
        if self.cpu_offload:
            self.model.to(torch.device("cuda"))

        num_channels_latents = self.config["transformer_in_channels"] // 4
        image = image.to(self.model.device)

        if image.shape[1] != self.latent_channels:
            image_latents = self._encode_vae_image(image=image)
        else:
            image_latents = image
        image_latents = torch.cat([image_latents], dim=0)
        image_latent_height, image_latent_width = image_latents.shape[3:]
        image_latents = self._pack_latents(image_latents, 1, num_channels_latents, image_latent_height, image_latent_width)

        if self.cpu_offload:
            self.model.to(torch.device("cpu"))
            torch.cuda.empty_cache()
            gc.collect()
        return image_latents
