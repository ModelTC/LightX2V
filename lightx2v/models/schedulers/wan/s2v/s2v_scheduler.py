import torch

from lightx2v.models.schedulers.scheduler import BaseScheduler
from lightx2v.models.schedulers.wan.s2v.fm_solvers_unipc import FlowUniPCMultistepScheduler
from lightx2v_platform.base.global_var import AI_DEVICE


class WanS2VScheduler(BaseScheduler):
    """UniPC scheduler for Wan2.2 S2V."""

    def __init__(self, config):
        super().__init__(config)
        self.num_train_timesteps = int(config.get("num_train_timesteps", 1000))
        self.sample_shift = float(config["sample_shift"])
        self.sample_guide_scale = float(config["sample_guide_scale"])
        self.unipc = None
        self.timesteps = None

    def prepare_clip(self, seed, latent_shape, dtype):
        self.generator = torch.Generator(device=AI_DEVICE).manual_seed(seed)
        self.latents = torch.randn(
            *latent_shape,
            dtype=dtype,
            device=AI_DEVICE,
            generator=self.generator,
        )
        self.unipc = FlowUniPCMultistepScheduler(
            num_train_timesteps=self.num_train_timesteps,
            shift=1,
            use_dynamic_shifting=False,
        )
        self.unipc.set_timesteps(self.infer_steps, device=AI_DEVICE, shift=self.sample_shift)
        self.timesteps = self.unipc.timesteps
        self.step_index = 0

    def step_pre(self, step_index):
        self.step_index = step_index
        self.timestep_input = torch.stack([self.timesteps[step_index]])

    def step_post(self):
        if getattr(self, "noise_pred", None) is None:
            return
        t = self.timesteps[self.step_index]
        self.latents = self.unipc.step(
            self.noise_pred.unsqueeze(0),
            t,
            self.latents.unsqueeze(0),
            return_dict=False,
            generator=self.generator,
        )[0].squeeze(0)

    def step(self, noise_pred, t):
        self.latents = self.unipc.step(
            noise_pred.unsqueeze(0),
            t,
            self.latents.unsqueeze(0),
            return_dict=False,
            generator=self.generator,
        )[0].squeeze(0)
        return self.latents
