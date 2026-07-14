import math
from typing import List, Optional, Union

import torch
from loguru import logger

from lightx2v.models.schedulers.scheduler import BaseScheduler
from lightx2v_platform.base.global_var import AI_DEVICE


class NeoppMoeScheduler(BaseScheduler):
    def __init__(self, config):
        super().__init__(config)
        self.time_schedule = self.config.get("time_schedule", "standard")
        self.timestep_shift = config.get("timestep_shift", 1.0)
        self.noise_scale_init = self.config.get("noise_scale", 1.0)
        self.noise_scale_mode = self.config.get("noise_scale_mode", "resolution")
        self.noise_scale_base_image_seq_len = self.config.get("noise_scale_base_image_seq_len", 64)
        self.noise_scale_max_value = self.config.get("noise_scale_max_value", 8.0)
        self.patch_size = self.config.get("patch_size", 16)
        self.merge_size = 2

    def prepare(self, seed, latent_shape, image_encoder_output=None):
        self.prepare_latents(seed, latent_shape)
        self.set_timesteps(self.infer_steps, device=AI_DEVICE, shift=self.timestep_shift)

    def prepare_latents(self, seed, latent_shape, dtype=torch.bfloat16):
        self.grid_h = latent_shape[2] // self.patch_size
        self.grid_w = latent_shape[3] // self.patch_size
        self.grid_hw = torch.tensor([[self.grid_h, self.grid_w]] * latent_shape[0], device=AI_DEVICE)

        noise_scale = self.noise_scale_init
        if self.noise_scale_mode in ("resolution", "dynamic", "dynamic_sqrt"):
            noise_scale = math.sqrt((self.grid_h * self.grid_w) / (self.merge_size**2) / self.noise_scale_base_image_seq_len)
            base = float(self.noise_scale_base_image_seq_len)
            scale = math.sqrt((self.grid_h * self.grid_w) / (self.merge_size**2) / base)
            noise_scale = scale * float(self.noise_scale_init)
            if self.noise_scale_mode == "dynamic_sqrt":
                noise_scale = math.sqrt(noise_scale)
        self.noise_scale = min(noise_scale, self.noise_scale_max_value)
        self.image_prediction = self.noise_scale * torch.randn(
            (
                latent_shape[0],
                latent_shape[1],
                latent_shape[2],
                latent_shape[3],
            ),
            device=AI_DEVICE,
            dtype=dtype,
        )

    def set_timesteps(
        self,
        infer_steps: Union[int, None] = None,
        device: Union[str, torch.device] = None,
        sigmas: Optional[List[float]] = None,
        mu: Optional[Union[float, None]] = None,
        shift: Optional[Union[float, None]] = None,
    ):
        logger.info(f"scheduler: timestep_shift={shift}")
        timesteps = torch.linspace(0.0, 1.0, self.infer_steps + 1, device=device)
        self.timesteps = self._apply_time_schedule(timesteps, timestep_shift=shift)

    def _apply_time_schedule(self, t: torch.Tensor, timestep_shift: float) -> torch.Tensor:
        sigma = 1 - t
        if timestep_shift != 1:
            self.time_schedule = "standard"
        if self.time_schedule == "standard":
            shift = timestep_shift
            sigma = shift * sigma / (1 + (shift - 1) * sigma)
        # elif self.time_schedule == "dynamic":
        #     """
        #     not support yet
        #     """
        #     mu = self._calculate_dynamic_mu(image_seq_len)
        #     mu_t = t.new_tensor(mu)
        #     if self.time_shift_type == "exponential":
        #         shift = torch.exp(mu_t)
        #         sigma = shift * sigma / (1 + (shift - 1) * sigma)
        #     elif self.time_shift_type == "linear":
        #         sigma = mu_t / (mu_t + (1 / sigma - 1))
        #     else:
        #         raise ValueError(f"Unsupported time_shift_type: {self.time_shift_type}")
        else:
            raise ValueError(f"Unsupported time_schedule: {self.time_schedule}")
        return 1 - sigma

    def step_pre(self, step_index):
        self.step_index = step_index

    def step_post(self):
        pass
