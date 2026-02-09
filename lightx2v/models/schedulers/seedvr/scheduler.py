"""
Scheduler for SeedVR video super-resolution model.

SeedVR uses a standard diffusion scheduler with:
- Linear interpolation (lerp) schedule
- Velocity prediction (v_lerp)
- CFG support
"""

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
