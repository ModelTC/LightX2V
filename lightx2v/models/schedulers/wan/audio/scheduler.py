import gc

import numpy as np
import torch
from loguru import logger

from lightx2v.models.schedulers.wan.scheduler import WanScheduler
from lightx2v.utils.envs import *
from lightx2v.utils.utils import masks_like


class ConsistencyModelScheduler(WanScheduler):
    def __init__(self, config):
        super().__init__(config)

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

    def prepare_latents(self, target_shape, dtype=torch.float32):
        self.generator = torch.Generator(device=self.device).manual_seed(self.config.seed)
        self.latents = torch.randn(
            target_shape[0],
            target_shape[1],
            target_shape[2],
            target_shape[3],
            dtype=dtype,
            device=self.device,
            generator=self.generator,
        )
        if self.config["model_cls"] == "wan2.2_audio":
            self.mask = masks_like(self.latents, zero=True, prev_len=self.prev_len)
            if self.prev_latents is not None:
                self.latents = (1.0 - self.mask) * self.prev_latents + self.mask * self.latents

    def prepare(self, previmg_encoder_output=None):
        self.prepare_latents(self.config.target_shape, dtype=torch.float32)
        timesteps = np.linspace(self.num_train_timesteps, 0, self.infer_steps + 1, dtype=np.float32)

        self.timesteps = torch.from_numpy(timesteps).to(dtype=torch.float32, device=self.device)
        self.timesteps_ori = self.timesteps.clone()

        self.sigmas = self.timesteps_ori / self.num_train_timesteps
        self.sigmas = self.sample_shift * self.sigmas / (1 + (self.sample_shift - 1) * self.sigmas)

        self.timesteps = self.sigmas * self.num_train_timesteps

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

    def reset(self, previmg_encoder_output=None):
        if self.config["model_cls"] == "wan2.2_audio":
            self.prev_latents = previmg_encoder_output["prev_latents"]
            self.prev_len = previmg_encoder_output["prev_len"]
        self.prepare_latents(self.config.target_shape, dtype=torch.float32)
        gc.collect()
        torch.cuda.empty_cache()

    def unsqueeze_to_ndim(self, in_tensor, tgt_n_dim):
        if in_tensor.ndim > tgt_n_dim:
            logger.warning(f"the given tensor of shape {in_tensor.shape} is expected to unsqueeze to {tgt_n_dim}, the original tensor will be returned")
            return in_tensor
        if in_tensor.ndim < tgt_n_dim:
            in_tensor = in_tensor[(...,) + (None,) * (tgt_n_dim - in_tensor.ndim)]
        return in_tensor
