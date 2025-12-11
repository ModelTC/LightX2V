import math

import numpy as np
import torch
from loguru import logger

from lightx2v.models.schedulers.wan.scheduler import WanScheduler
from lightx2v.utils.envs import *
from lightx2v.utils.utils import masks_like
from lightx2v_platform.base.global_var import AI_DEVICE


class EulerScheduler(WanScheduler):
    def __init__(self, config):
        super().__init__(config)
        d = config["dim"] // config["num_heads"]
        self.rope_t_dim = d // 2 - 2 * (d // 6)

        if self.config["parallel"]:
            self.sp_size = self.config["parallel"].get("seq_p_size", 1)
        else:
            self.sp_size = 1

        if self.config["model_cls"] == "wan2.2_audio":
            self.prev_latents = None
            self.prev_len = 0

    def set_audio_adapter(self, audio_adapter):
        self.audio_adapter = audio_adapter

    def step_pre(self, step_index):
        super().step_pre(step_index)
        if self.audio_adapter.cpu_offload:
            self.audio_adapter.time_embedding.to("cuda")
        self.audio_adapter_t_emb = self.audio_adapter.time_embedding(self.timestep_input).unflatten(1, (3, -1))
        if self.audio_adapter.cpu_offload:
            self.audio_adapter.time_embedding.to("cpu")

        if self.config["model_cls"] == "wan2.2_audio":
            _, lat_f, lat_h, lat_w = self.latents.shape
            F = (lat_f - 1) * self.config["vae_stride"][0] + 1
            per_latent_token_len = lat_h * lat_w // (self.config["patch_size"][1] * self.config["patch_size"][2])
            max_seq_len = ((F - 1) // self.config["vae_stride"][0] + 1) * per_latent_token_len
            max_seq_len = int(math.ceil(max_seq_len / self.sp_size)) * self.sp_size

            temp_ts = (self.mask[0][:, ::2, ::2] * self.timestep_input).flatten()
            self.timestep_input = torch.cat([temp_ts, temp_ts.new_ones(max_seq_len - temp_ts.size(0)) * self.timestep_input]).unsqueeze(0)

            self.timestep_input = torch.cat(
                [
                    self.timestep_input,
                    torch.zeros(
                        (1, per_latent_token_len),  # padding for reference frame latent
                        dtype=self.timestep_input.dtype,
                        device=self.timestep_input.device,
                    ),
                ],
                dim=1,
            )

    def prepare_latents(self, seed, latent_shape, dtype=torch.float32):
        self.generator = torch.Generator(device=AI_DEVICE).manual_seed(seed)
        self.latents = torch.randn(
            latent_shape[0],
            latent_shape[1],
            latent_shape[2],
            latent_shape[3],
            dtype=dtype,
            device=AI_DEVICE,
            generator=self.generator,
        )
        if self.config["model_cls"] == "wan2.2_audio":
            self.mask = masks_like(self.latents, zero=True, prev_len=self.prev_len)
            if self.prev_latents is not None:
                self.latents = (1.0 - self.mask) * self.prev_latents + self.mask * self.latents

    def prepare(self, seed, latent_shape, image_encoder_output=None):
        self.prepare_latents(seed, latent_shape, dtype=torch.float32)
        timesteps = np.linspace(self.num_train_timesteps, 0, self.infer_steps + 1, dtype=np.float32)

        self.timesteps = torch.from_numpy(timesteps).to(dtype=torch.float32, device=AI_DEVICE)
        self.timesteps_ori = self.timesteps.clone()

        self.sigmas = self.timesteps_ori / self.num_train_timesteps
        self.sigmas = self.sample_shift * self.sigmas / (1 + (self.sample_shift - 1) * self.sigmas)

        self.timesteps = self.sigmas * self.num_train_timesteps

        self.freqs[latent_shape[1] // self.patch_size[0] :, : self.rope_t_dim] = 0

        if self.config.get("f2v_process", False):
            f = latent_shape[1] // self.patch_size[0]
        else:
            f = latent_shape[1] // self.patch_size[0] + 1
        self.cos_sin = self.prepare_cos_sin((f, latent_shape[2] // self.patch_size[1], latent_shape[3] // self.patch_size[2]))

    def step_post(self):
        model_output = self.noise_pred.to(torch.float32)
        sample = self.latents.to(torch.float32)
        sigma = self.unsqueeze_to_ndim(self.sigmas[self.step_index], sample.ndim).to(sample.device, sample.dtype)
        sigma_next = self.unsqueeze_to_ndim(self.sigmas[self.step_index + 1], sample.ndim).to(sample.device, sample.dtype)
        x_t_next = sample + (sigma_next - sigma) * model_output
        self.latents = x_t_next
        if self.config["model_cls"] == "wan2.2_audio" and self.prev_latents is not None:
            self.latents = (1.0 - self.mask) * self.prev_latents + self.mask * self.latents

    def reset(self, seed, latent_shape, image_encoder_output=None):
        if self.config["model_cls"] == "wan2.2_audio":
            self.prev_latents = image_encoder_output["prev_latents"]
            self.prev_len = image_encoder_output["prev_len"]
        self.prepare_latents(seed, latent_shape, dtype=torch.float32)

    def unsqueeze_to_ndim(self, in_tensor, tgt_n_dim):
        if in_tensor.ndim > tgt_n_dim:
            logger.warning(f"the given tensor of shape {in_tensor.shape} is expected to unsqueeze to {tgt_n_dim}, the original tensor will be returned")
            return in_tensor
        if in_tensor.ndim < tgt_n_dim:
            in_tensor = in_tensor[(...,) + (None,) * (tgt_n_dim - in_tensor.ndim)]
        return in_tensor


class ConsistencyModelScheduler(EulerScheduler):
    def step_post(self):
        model_output = self.noise_pred.to(torch.float32)
        sample = self.latents.to(torch.float32)
        sigma = self.unsqueeze_to_ndim(self.sigmas[self.step_index], sample.ndim).to(sample.device, sample.dtype)
        sigma_next = self.unsqueeze_to_ndim(self.sigmas[self.step_index + 1], sample.ndim).to(sample.device, sample.dtype)
        x0 = sample - model_output * sigma
        x_t_next = x0 * (1 - sigma_next) + sigma_next * torch.randn(x0.shape, dtype=x0.dtype, device=x0.device, generator=self.generator)
        self.latents = x_t_next
        if self.config["model_cls"] == "wan2.2_audio" and self.prev_latents is not None:
            self.latents = (1.0 - self.mask) * self.prev_latents + self.mask * self.latents
