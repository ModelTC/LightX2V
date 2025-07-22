import os
import gc
import math
import numpy as np
import torch
from typing import List, Optional, Tuple, Union
from lightx2v.utils.envs import *
from lightx2v.models.schedulers.scheduler import BaseScheduler
from loguru import logger

from diffusers.configuration_utils import register_to_config
from torch import Tensor
from diffusers import (
    FlowMatchEulerDiscreteScheduler as FlowMatchEulerDiscreteSchedulerBase,  # pyright: ignore
)


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
        self.caching_records = [True] * config.infer_steps
        self.flag_df = False
        self.transformer_infer = None
        self.solver_order = 2
        self.device = torch.device("cuda")
        self.infer_steps = self.config.infer_steps
        self.target_video_length = self.config.target_video_length
        self.sample_shift = self.config.sample_shift
        self.shift = 1
        self.num_train_timesteps = 1000
        self.step_index = None
        self.noise_pred = None

        self._step_index = None
        self._begin_index = None

    def step_pre(self, step_index):
        self.step_index = step_index
        if GET_DTYPE() == "BF16":
            self.latents = self.latents.to(dtype=torch.bfloat16)

    def set_timesteps(
        self,
        infer_steps: Union[int, None] = None,
        device: Union[str, torch.device] = None,
        sigmas: Optional[List[float]] = None,
        mu: Optional[Union[float, None]] = None,
        shift: Optional[Union[float, None]] = None,
    ):
        sigmas = np.linspace(self.sigma_max, self.sigma_min, infer_steps + 1).copy()[:-1]

        if shift is None:
            shift = self.shift
        sigmas = shift * sigmas / (1 + (shift - 1) * sigmas)

        sigma_last = 0

        timesteps = sigmas * self.num_train_timesteps
        sigmas = np.concatenate([sigmas, [sigma_last]]).astype(np.float32)

        self.sigmas = torch.from_numpy(sigmas)
        self.timesteps = torch.from_numpy(timesteps).to(device=device, dtype=torch.int64)

        assert len(self.timesteps) == self.infer_steps
        self.model_outputs = [
            None,
        ] * self.solver_order
        self.lower_order_nums = 0
        self.last_sample = None
        self._begin_index = None
        self.sigmas = self.sigmas.to("cpu")

    def prepare(self, image_encoder_output=None):
        self.prepare_latents(self.config.target_shape, dtype=torch.float32)

        if self.config.task in ["t2v"]:
            self.seq_len = math.ceil((self.config.target_shape[2] * self.config.target_shape[3]) / (self.config.patch_size[1] * self.config.patch_size[2]) * self.config.target_shape[1])
        elif self.config.task in ["i2v"]:
            self.seq_len = ((self.config.target_video_length - 1) // self.config.vae_stride[0] + 1) * self.config.lat_h * self.config.lat_w // (self.config.patch_size[1] * self.config.patch_size[2])

        alphas = np.linspace(1, 1 / self.num_train_timesteps, self.num_train_timesteps)[::-1].copy()
        sigmas = 1.0 - alphas
        sigmas = torch.from_numpy(sigmas).to(dtype=torch.float32)

        sigmas = self.shift * sigmas / (1 + (self.shift - 1) * sigmas)

        self.sigmas = sigmas
        self.timesteps = sigmas * self.num_train_timesteps

        self.model_outputs = [None] * self.solver_order
        self.timestep_list = [None] * self.solver_order
        self.last_sample = None

        self.sigmas = self.sigmas.to("cpu")
        self.sigma_min = self.sigmas[-1].item()
        self.sigma_max = self.sigmas[0].item()

        self.set_timesteps(self.infer_steps, device=self.device, shift=self.sample_shift)

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

        sample = sample.to(torch.float32)
        sigma = unsqueeze_to_ndim(self.sigmas[self.step_index], sample.ndim).to(sample.device, sample.dtype)
        sigma_next = unsqueeze_to_ndim(self.sigmas[self.step_index + 1], sample.ndim).to(sample.device, sample.dtype)
        # x0 = sample - model_output * sigma
        x_t_next = sample + (sigma_next - sigma) * model_output

        self.latents = x_t_next

    def reset(self):
        self.model_outputs = [None]
        self.timestep_list = [None]
        self.last_sample = None
        self.noise_pred = None
        self.this_order = None
        self.lower_order_nums = 0
        self.prepare_latents(self.config.target_shape, dtype=torch.float32)
        gc.collect()
        torch.cuda.empty_cache()
