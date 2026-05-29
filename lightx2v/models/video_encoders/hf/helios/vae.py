import gc
import os

import torch
from diffusers import AutoencoderKLWan
from diffusers.utils import load_image
from diffusers.video_processor import VideoProcessor

from lightx2v_platform.base.global_var import AI_DEVICE


class HeliosVAE:
    def __init__(self, config):
        self.config = config
        self.cpu_offload = config.get("vae_cpu_offload", config.get("cpu_offload", False))
        self.device = torch.device("cpu") if self.cpu_offload else torch.device(AI_DEVICE)
        self.dtype = torch.float32
        self.model = AutoencoderKLWan.from_pretrained(config["vae_path"], torch_dtype=self.dtype).to(self.device)
        self.vae_scale_factor_temporal = getattr(self.model.config, "scale_factor_temporal", 4)
        self.vae_scale_factor_spatial = getattr(self.model.config, "scale_factor_spatial", 8)
        self.video_processor = VideoProcessor(vae_scale_factor=self.vae_scale_factor_spatial)
        self.latents_mean = torch.tensor(self.model.config.latents_mean).view(1, self.model.config.z_dim, 1, 1, 1)
        self.latents_std = 1.0 / torch.tensor(self.model.config.latents_std).view(1, self.model.config.z_dim, 1, 1, 1)
        if config.get("use_tiling_vae", False):
            self.model.enable_tiling()

    def _to_device(self):
        if self.cpu_offload:
            self.model.to(torch.device(AI_DEVICE))

    def _to_cpu(self):
        if self.cpu_offload:
            self.model.to(torch.device("cpu"))
            torch.cuda.empty_cache()
            gc.collect()

    def preprocess_image(self, image_path_or_pil, height, width):
        image = image_path_or_pil
        if isinstance(image, (str, os.PathLike)):
            image = load_image(str(image))
        return self.video_processor.preprocess(image, height=height, width=width)

    def prepare_image_latents(self, image, generator, num_latent_frames_per_chunk, height, width, dtype=torch.float32):
        self._to_device()
        image = self.preprocess_image(image, height, width).unsqueeze(2).to(device=self.model.device, dtype=self.model.dtype)
        latents_mean = self.latents_mean.to(device=self.model.device, dtype=self.model.dtype)
        latents_std = self.latents_std.to(device=self.model.device, dtype=self.model.dtype)
        image_latents = self.model.encode(image).latent_dist.sample(generator=generator)
        image_latents = (image_latents - latents_mean) * latents_std

        min_frames = (num_latent_frames_per_chunk - 1) * self.vae_scale_factor_temporal + 1
        fake_video = image.repeat(1, 1, min_frames, 1, 1)
        fake_latents = self.model.encode(fake_video).latent_dist.sample(generator=generator)
        fake_latents = (fake_latents - latents_mean) * latents_std
        fake_latents = fake_latents[:, :, -1:, :, :]
        self._to_cpu()
        return image_latents.to(dtype=dtype), fake_latents.to(dtype=dtype)

    def decode(self, latents):
        self._to_device()
        latents_mean = self.latents_mean.to(device=self.model.device, dtype=self.model.dtype)
        latents_std = self.latents_std.to(device=self.model.device, dtype=self.model.dtype)
        current_latents = latents.to(self.model.device, dtype=self.model.dtype) / latents_std + latents_mean
        decoded = self.model.decode(current_latents, return_dict=False)[0]
        self._to_cpu()
        return decoded
