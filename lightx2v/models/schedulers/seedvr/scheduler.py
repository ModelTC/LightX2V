"""
Scheduler for SeedVR video super-resolution model.

SeedVR uses a standard diffusion scheduler with:
- Linear interpolation (lerp) schedule
- Velocity prediction (v_lerp)
- CFG support
"""

import torch

from lightx2v.models.schedulers.scheduler import BaseScheduler
from lightx2v.models.schedulers.seedvr.diffusion.config import (
    create_sampler_from_config,
    create_sampling_timesteps_from_config,
    create_schedule_from_config,
)
from lightx2v_platform.base.global_var import AI_DEVICE


class SeedVRScheduler(BaseScheduler):
    """Scheduler for SeedVR model.

    SeedVR uses a linear interpolation schedule with velocity prediction.
    """

    def __init__(self, config):
        super().__init__(config)

        self.num_train_timesteps = 1000.0

        schedule_cfg = {"type": "lerp", "T": self.num_train_timesteps}
        sampling_cfg = {"type": "uniform_trailing", "steps": 1}
        sampler_cfg = {"type": "euler", "prediction_type": "v_lerp"}

        self.schedule = create_schedule_from_config(schedule_cfg, device=AI_DEVICE)
        self.sampling_timesteps = create_sampling_timesteps_from_config(sampling_cfg, schedule=self.schedule, device=AI_DEVICE)
        self.sampler = create_sampler_from_config(sampler_cfg, schedule=self.schedule, timesteps=self.sampling_timesteps)

    def prepare(self, seed, latent_shape, image_encoder_output=None):
        pass

    def step_pre(self, step_index):
        """Prepare for a single step.

        Args:
            step_index: Current step index
        """
        self.step_index = step_index

    def step_post(self):
        """Process after a single step."""
        pass

    def _timestep_transform(self, timesteps: torch.Tensor, latents_shapes: torch.Tensor):
        transform = self.config.get("diffusion", {}).get("timesteps", {}).get("transform", True)
        if not transform:
            return timesteps

        vt = 4
        vs = 8
        frames = (latents_shapes[:, 0] - 1) * vt + 1
        heights = latents_shapes[:, 1] * vs
        widths = latents_shapes[:, 2] * vs

        def get_lin_function(x1, y1, x2, y2):
            m = (y2 - y1) / (x2 - x1)
            b = y1 - m * x1
            return lambda x: m * x + b

        img_shift_fn = get_lin_function(x1=256 * 256, y1=1.0, x2=1024 * 1024, y2=3.2)
        vid_shift_fn = get_lin_function(x1=256 * 256 * 37, y1=1.0, x2=1280 * 720 * 145, y2=5.0)
        shift = torch.where(
            frames > 1,
            vid_shift_fn(heights * widths * frames),
            img_shift_fn(heights * widths),
        )

        schedule_T = float(self.schedule.T) if self.schedule is not None else self.num_train_timesteps
        timesteps = timesteps / schedule_T
        timesteps = shift * timesteps / (1 + (shift - 1) * timesteps)
        timesteps = timesteps * schedule_T
        return timesteps

    def _add_noise(self, x: torch.Tensor, aug_noise: torch.Tensor, cond_noise_scale: float = 0.0):
        t = torch.tensor([self.num_train_timesteps * cond_noise_scale], device=x.device, dtype=x.dtype)
        shape = torch.tensor(x.shape[1:], device=x.device)[None]
        t = self._timestep_transform(t, shape)
        if self.schedule is not None:
            return self.schedule.forward(x, aug_noise, t)
        return (1 - (t / self.num_train_timesteps)) * x + (t / self.num_train_timesteps) * aug_noise
