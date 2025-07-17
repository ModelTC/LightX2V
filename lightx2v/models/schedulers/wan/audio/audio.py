import os
import gc
import math
import numpy as np
import torch
from typing import List, Optional, Tuple, Union
from lightx2v.utils.envs import *

from diffusers.configuration_utils import register_to_config
from torch import Tensor
from .utils import unsqueeze_to_ndim
from diffusers import (
    FlowMatchEulerDiscreteScheduler as FlowMatchEulerDiscreteSchedulerBase,  # pyright: ignore
)


def get_timesteps(num_steps, max_steps: int = 1000):
    return np.linspace(max_steps, 0, num_steps + 1, dtype=np.float32)


def timestep_shift(timesteps, shift: float = 1.0):
    return shift * timesteps / (1 + (shift - 1) * timesteps)


class FlowMatchEulerDiscreteScheduler(FlowMatchEulerDiscreteSchedulerBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.init_noise_sigma = 1.0

    def add_noise(self, x0: Tensor, noise: Tensor, timesteps: Tensor):
        dtype = x0.dtype
        device = x0.device
        sigma = timesteps.to(device, torch.float32) / self.config.num_train_timesteps
        sigma = unsqueeze_to_ndim(sigma, x0.ndim)
        xt = x0.float() * (1 - sigma) + noise.float() * sigma
        return xt.to(dtype)

    def get_velocity(self, x0: Tensor, noise: Tensor, timesteps: Tensor | None = None):
        return noise - x0

    def velocity_loss_to_x_loss(self, v_loss: Tensor, timesteps: Tensor):
        device = v_loss.device
        sigma = timesteps.to(device, torch.float32) / self.config.num_train_timesteps
        return v_loss.float() * (sigma**2)


class EulerSchedulerTimestepFix(FlowMatchEulerDiscreteScheduler):
    def __init__(self, config):
        self.config = config
        self.step_index = 0
        self.latents = None
        self.caching_records = [True] * config.infer_steps
        self.flag_df = False
        self.transformer_infer = None

        self.device = torch.device("cuda")
        self.infer_steps = self.config.infer_steps
        self.target_video_length = self.config.target_video_length
        self.sample_shift = self.config.sample_shift

        self.num_train_timesteps = 1000

        self.noise_pred = None

    def step_pre(self, step_index):
        self.step_index = step_index
        if GET_DTYPE() == "BF16":
            self.latents = self.latents.to(dtype=torch.bfloat16)

    def set_shift(self, shift: float = 1.0):
        self.sigmas = self.timesteps_ori / self.num_train_timesteps
        self.sigmas = timestep_shift(self.sigmas, shift=shift)
        self.timesteps = self.sigmas * self.num_train_timesteps

    def set_timesteps(
        self,
        infer_steps: Union[int, None] = None,
        device: Union[str, torch.device] = None,
        sigmas: Optional[List[float]] = None,
        mu: Optional[Union[float, None]] = None,
        shift: Optional[Union[float, None]] = None,
    ):
        timesteps = get_timesteps(num_steps=infer_steps, max_steps=self.num_train_timesteps)
        self.timesteps = torch.from_numpy(timesteps).to(dtype=torch.float32, device=device or self.device)
        self.timesteps_ori = self.timesteps.clone()
        self.set_shift(self.sample_shift)
        self._step_index = None
        self._begin_index = None

    def prepare(self, image_encoder_output=None):
        self.generator = torch.Generator(device=self.device)
        self.generator.manual_seed(self.config.seed)

        self.prepare_latents(self.config.target_shape, dtype=torch.float32)

        if os.path.isfile(self.config.image_path):
            self.seq_len = ((self.config.target_video_length - 1) // self.config.vae_stride[0] + 1) * self.config.lat_h * self.config.lat_w // (self.config.patch_size[1] * self.config.patch_size[2])
        else:
            self.seq_len = math.ceil((self.config.target_shape[2] * self.config.target_shape[3]) / (self.config.patch_size[1] * self.config.patch_size[2]) * self.config.target_shape[1])

        self.model_outputs = [None] * self.solver_order
        self.timestep_list = [None] * self.solver_order
        self.last_sample = None

        self.set_timesteps(infer_steps=self.infer_steps, device=self.device, shift=self.sample_shift)

    def prepare_latents(self, target_shape, dtype=torch.float32):
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
        timestep = self.timesteps[self.step_index]
        sample = self.latents.to(torch.float32)

        if self.step_index is None:
            self._init_step_index(timestep)
        sample = sample.to(torch.float32)  # pyright: ignore
        sigma = unsqueeze_to_ndim(self.sigmas[self.step_index], sample.ndim)
        sigma_next = unsqueeze_to_ndim(self.sigmas[self.step_index + 1], sample.ndim)
        # x0 = sample - model_output * sigma
        x_t_next = sample + (sigma_next - sigma) * model_output
        self._step_index += 1
        return x_t_next

    def reset(self):
        self.model_outputs = [None] * self.solver_order
        self.timestep_list = [None] * self.solver_order
        self.last_sample = None
        self.noise_pred = None
        self.this_order = None
        self.lower_order_nums = 0
        self.prepare_latents(self.config.target_shape, dtype=torch.float32)
        gc.collect()
        torch.cuda.empty_cache()
