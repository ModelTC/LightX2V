import gc
import math
import warnings

import numpy as np
import torch
from torch import Tensor

from lightx2v.models.schedulers.scheduler import BaseScheduler
from lightx2v.utils.envs import *


def unsqueeze_to_ndim(in_tensor: Tensor, tgt_n_dim: int):
    if in_tensor.ndim > tgt_n_dim:
        warnings.warn(f"the given tensor of shape {in_tensor.shape} is expected to unsqueeze to {tgt_n_dim}, the original tensor will be returned")
        return in_tensor
    if in_tensor.ndim < tgt_n_dim:
        in_tensor = in_tensor[(...,) + (None,) * (tgt_n_dim - in_tensor.ndim)]
    return in_tensor


class EulerSchedulerTimestepFix(BaseScheduler):
    def __init__(self, config, **kwargs):
        # super().__init__(**kwargs)
        self.init_noise_sigma = 1.0
        self.config = config
        self.latents = None
        self.device = torch.device("cuda")
        self.infer_steps = self.config.infer_steps
        self.target_video_length = self.config.target_video_length
        self.sample_shift = self.config.sample_shift
        self.num_train_timesteps = 1000
        self.step_index = None

    def step_pre(self, step_index):
        self.step_index = step_index
        if GET_DTYPE() == "BF16":
            self.latents = self.latents.to(dtype=torch.bfloat16)

    def prepare(self, image_encoder_output=None):
        self.prepare_latents(self.config.target_shape, dtype=torch.float32)

        if self.config.task in ["t2v"]:
            self.seq_len = math.ceil((self.config.target_shape[2] * self.config.target_shape[3]) / (self.config.patch_size[1] * self.config.patch_size[2]) * self.config.target_shape[1])
        elif self.config.task in ["i2v"]:
            self.seq_len = ((self.config.target_video_length - 1) // self.config.vae_stride[0] + 1) * self.config.lat_h * self.config.lat_w // (self.config.patch_size[1] * self.config.patch_size[2])

        timesteps = np.linspace(self.num_train_timesteps, 0, self.infer_steps + 1, dtype=np.float32)

        self.timesteps = torch.from_numpy(timesteps).to(dtype=torch.float32, device=self.device)
        self.timesteps_ori = self.timesteps.clone()

        self.sigmas = self.timesteps_ori / self.num_train_timesteps
        self.sigmas = self.sample_shift * self.sigmas / (1 + (self.sample_shift - 1) * self.sigmas)

        self.timesteps = self.sigmas * self.num_train_timesteps

    def prepare_latents(self, target_shape, dtype=torch.float32):
        self.generator = torch.Generator(device=self.device).manual_seed(self.config.seed)
        self.latents = (
            torch.randn(
                target_shape[0],
                target_shape[1],
                target_shape[2],
                target_shape[3],
                dtype=dtype,
                device=self.device,
                generator=self.generator,
            )
            * self.init_noise_sigma
        )

    def step_post(self):
        model_output = self.noise_pred.to(torch.float32)
        sample = self.latents.to(torch.float32)

        sigma = unsqueeze_to_ndim(self.sigmas[self.step_index], sample.ndim).to(sample.device, sample.dtype)
        sigma_next = unsqueeze_to_ndim(self.sigmas[self.step_index + 1], sample.ndim).to(sample.device, sample.dtype)
        x_t_next = sample + (sigma_next - sigma) * model_output

        self.latents = x_t_next

    def reset(self):
        self.prepare_latents(self.config.target_shape, dtype=torch.float32)
        gc.collect()
        torch.cuda.empty_cache()


class ConsistencyModelScheduler(EulerSchedulerTimestepFix):
    def step_post(self):
        model_output = self.noise_pred.to(torch.float32)
        sample = self.latents.to(torch.float32)
        sigma = unsqueeze_to_ndim(self.sigmas[self.step_index], sample.ndim).to(sample.device, sample.dtype)
        sigma_next = unsqueeze_to_ndim(self.sigmas[self.step_index + 1], sample.ndim).to(sample.device, sample.dtype)
        x0 = sample - model_output * sigma
        x_t_next = x0 * (1 - sigma_next) + sigma_next * torch.randn(x0.shape, dtype=x0.dtype, device=x0.device, generator=self.generator)
        self.latents = x_t_next
