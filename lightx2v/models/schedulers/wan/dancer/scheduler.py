import torch

from lightx2v.models.schedulers.scheduler import BaseScheduler
from lightx2v.models.schedulers.wan.step_distill.scheduler import WanStepDistillScheduler
from lightx2v.utils.envs import GET_DTYPE
from lightx2v_platform.base.global_var import AI_DEVICE


class WanDancerScheduler(BaseScheduler):
    """Wan-Dancer's shifted flow-matching Euler scheduler."""

    def __init__(self, config):
        super().__init__(config)
        self.sample_shift = float(config["sample_shift"])
        self.sample_guide_scale = float(config["sample_guide_scale"])
        self.num_train_timesteps = 1000
        self.keep_latents_dtype_in_scheduler = True
        self.noise_pred = None

    def prepare(self, seed, latent_shape, image_encoder_output=None):
        self.generator = torch.Generator(device="cpu").manual_seed(int(seed))
        self.latents = torch.randn(latent_shape, generator=self.generator, dtype=torch.float32, device="cpu").to(device=AI_DEVICE, dtype=GET_DTYPE())
        sigmas = torch.linspace(1.0, 0.0, self.infer_steps + 1, dtype=torch.float32)[:-1]
        self.sigmas = self.sample_shift * sigmas / (1.0 + (self.sample_shift - 1.0) * sigmas)
        self.timesteps = self.sigmas * self.num_train_timesteps

    def reset(self, seed, latent_shape, step_index=None):
        self.prepare(seed, latent_shape)
        self.step_index = 0 if step_index is None else step_index
        self.noise_pred = None

    def step_pre(self, step_index):
        self.step_index = step_index
        self.timestep_input = self.timesteps[step_index].reshape(1).to(device=AI_DEVICE, dtype=GET_DTYPE())

    def step_post(self):
        sigma = self.sigmas[self.step_index]
        next_sigma = self.sigmas[self.step_index + 1] if self.step_index + 1 < self.infer_steps else torch.tensor(0.0)
        self.latents.add_(self.noise_pred * (next_sigma - sigma))

    def clear(self):
        self.generator = None
        self.latents = None
        self.noise_pred = None


class WanDancerStepDistillScheduler(WanStepDistillScheduler):
    """Wan 4-step scheduler with Dancer's per-segment deterministic reset."""

    def reset(self, seed, latent_shape, step_index=None):
        self.generator = None
        self.prepare(seed, latent_shape)
        self.step_index = 0 if step_index is None else step_index
        self.noise_pred = None
