import json
import math
import os

import numpy as np
import torch
from diffusers.pipelines.flux2.pipeline_flux2 import compute_empirical_mu
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import retrieve_timesteps
from diffusers.schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler

from lightx2v.models.schedulers.scheduler import BaseScheduler
from lightx2v.utils.envs import GET_DTYPE
from lightx2v_platform.base.global_var import AI_DEVICE


def get_timestep_embedding(
    timesteps: torch.Tensor,
    embedding_dim: int = 256,
    flip_sin_to_cos: bool = True,
    downscale_freq_shift: float = 0,
    scale: float = 1,
    max_period: int = 10000,
) -> torch.Tensor:
    if len(timesteps.shape) != 1:
        raise ValueError("Timesteps should be a 1D tensor")

    half_dim = embedding_dim // 2
    exponent = -math.log(max_period) * torch.arange(start=0, end=half_dim, dtype=torch.float32, device=timesteps.device)
    exponent = exponent / (half_dim - downscale_freq_shift)

    emb = torch.exp(exponent)
    emb = timesteps[:, None].float() * emb[None, :]
    emb = scale * emb
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)

    if flip_sin_to_cos:
        emb = torch.cat([emb[:, half_dim:], emb[:, :half_dim]], dim=-1)

    if embedding_dim % 2 == 1:
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))

    return emb


def randn_tensor(shape, generator=None, device=None, dtype=None):
    if isinstance(device, str):
        device = torch.device(device)
    device = device or torch.device("cpu")
    latents = torch.randn(shape, generator=generator, device=device, dtype=dtype)
    return latents


class Flux2KleinScheduler(BaseScheduler):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        scheduler_path = config.get("scheduler_path", os.path.join(config["model_path"], "scheduler"))
        self.scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(scheduler_path)

        with open(os.path.join(config["model_path"], "scheduler", "scheduler_config.json"), "r") as f:
            self.scheduler_config = json.load(f)

        self.dtype = GET_DTYPE()
        self.sample_guide_scale = config.get("sample_guide_scale", 4.0)
        self.infer_steps = config.get("infer_steps", 50)
        self.sigmas = None
        self.timesteps = None

    def prepare(self, input_info):
        self.input_info = input_info
        self.generator = torch.Generator(device=AI_DEVICE).manual_seed(input_info.seed)

        # Preserve latent and position ids for transformer id-aware routines
        if hasattr(input_info, "latent_image_ids"):
            self.latent_image_ids = input_info.latent_image_ids
        else:
            self.latent_image_ids = None

        if hasattr(input_info, "txt_ids"):
            self.txt_ids = input_info.txt_ids
        else:
            self.txt_ids = None

        self.latents = randn_tensor(input_info.latent_shape, generator=self.generator, device=AI_DEVICE, dtype=self.dtype)

        self.set_timesteps()

    def set_timesteps(self):
        """Set timesteps for the scheduler."""
        self.sigmas = np.linspace(1.0, 1 / self.infer_steps, self.infer_steps)
        image_seq_len = self.latents.shape[1]
        mu = compute_empirical_mu(image_seq_len=image_seq_len, num_steps=self.infer_steps)
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler,
            self.infer_steps,
            AI_DEVICE,
            sigmas=self.sigmas,
            mu=mu,
        )
        self.timesteps = timesteps
        self.infer_steps = num_inference_steps

        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
        self._num_timesteps = len(timesteps)
        self.num_warmup_steps = num_warmup_steps

    def step_pre(self, step_index):
        super().step_pre(step_index)
        timestep_input = torch.tensor([self.timesteps[self.step_index]], device=AI_DEVICE, dtype=self.dtype)
        self.timesteps_proj = get_timestep_embedding(timestep_input).to(self.dtype)

    def step_post(self):
        t = self.timesteps[self.step_index]
        latents = self.scheduler.step(self.noise_pred, t, self.latents, return_dict=False)[0]
        self.latents = latents

    def _encode_image(self, image):
        image = image.to(device=AI_DEVICE, dtype=GET_DTYPE())
        encoder_output = self.vae.encode_vae_image(image)

        if hasattr(encoder_output, "latent_dist"):
            image_latents = encoder_output.latent_dist.mode()
        else:
            image_latents = encoder_output.latents

        batch_size, num_channels_latents, height, width = image_latents.shape
        image_latents = image_latents.view(batch_size, num_channels_latents, height // 2, 2, width // 2, 2)
        image_latents = image_latents.permute(0, 1, 3, 5, 2, 4)
        image_latents = image_latents.reshape(batch_size, num_channels_latents * 4, height // 2, width // 2)

        bn = self.vae.vae.bn
        latents_bn_mean = bn.running_mean.view(1, -1, 1, 1).to(image_latents.device, image_latents.dtype)
        latents_bn_std = torch.sqrt(bn.running_var.view(1, -1, 1, 1) + self.vae.vae.config.batch_norm_eps).to(image_latents.device, image_latents.dtype)
        image_latents = (image_latents - latents_bn_mean) / latents_bn_std

        return image_latents

    def _prepare_image_ids(self, image_latents: list[torch.Tensor], scale: int = 10):
        t_coords = [scale + scale * t for t in torch.arange(0, len(image_latents))]
        t_coords = [t.view(-1) for t in t_coords]

        image_latent_ids = []
        for x, t in zip(image_latents, t_coords):
            x = x.squeeze(0)
            _, height, width = x.shape
            x_ids = torch.cartesian_prod(t, torch.arange(height), torch.arange(width), torch.arange(1))
            image_latent_ids.append(x_ids)

        image_latent_ids = torch.cat(image_latent_ids, dim=0)
        image_latent_ids = image_latent_ids.unsqueeze(0)
        return image_latent_ids

    def _pack_latents(self, latents):
        if isinstance(latents, list):
            packed_list = []
            for lat in latents:
                batch_size, num_channels, height, width = lat.shape
                packed = lat.reshape(batch_size, num_channels, height * width).permute(0, 2, 1)
                packed_list.append(packed)
            return torch.cat(packed_list, dim=1)
        else:
            batch_size, num_channels, height, width = latents.shape
            latents = latents.reshape(batch_size, num_channels, height * width).permute(0, 2, 1)
            return latents

    def prepare_i2i(self, input_info, input_image, vae):
        self.vae = vae
        self.prepare(input_info)

        image_latents = []
        for img in input_image:
            imagge_latent = self._encode_image(img)
            image_latents.append(imagge_latent)

        if "task_variant" in self.config:
            self.task_variant = self.config["task_variant"]
            if self.task_variant == "edit":  # 取出第一个融入latents
                ref_img_latent = image_latents[0]
                image_latents = image_latents[1:]

                ref_img_latent = self._pack_latents(ref_img_latent).squeeze(0)
                ref_img_latent = ref_img_latent.unsqueeze(0).to(AI_DEVICE, dtype=self.dtype)

                self.latents = (1 - self.sigmas[0]) * ref_img_latent + self.sigmas[0] * self.latents

        image_latent_ids = self._prepare_image_ids(image_latents, scale=10)

        packed = self._pack_latents(image_latents).squeeze(0)

        packed_latents = packed.unsqueeze(0).repeat(1, 1, 1).to(AI_DEVICE, dtype=self.dtype)
        image_latent_ids = image_latent_ids.repeat(1, 1, 1).to(AI_DEVICE)

        self.input_image_latents = packed_latents
        self.input_image_ids = image_latent_ids
