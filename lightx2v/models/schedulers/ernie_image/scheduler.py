import os

import torch
from diffusers.schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler

from lightx2v.models.schedulers.scheduler import BaseScheduler
from lightx2v.utils.envs import GET_DTYPE
from lightx2v_platform.base.global_var import AI_DEVICE


class ErnieImageScheduler(BaseScheduler):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        scheduler_path = config.get("scheduler_path", os.path.join(config["model_path"], "scheduler"))
        self.scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(scheduler_path)
        self.dtype = GET_DTYPE()
        self.sample_guide_scale = config.get("sample_guide_scale", 1.0)
        self.timestep = None
        self.noise_pred = None

    def prepare_latents(self, input_info):
        shape = tuple(input_info.target_shape)
        self.latents = torch.randn(
            shape,
            generator=self.generator,
            device=AI_DEVICE,
            dtype=self.dtype,
        )
        self.noise_pred = None

    def set_timesteps(self):
        sigmas = torch.linspace(
            1.0,
            0.0,
            self.config["infer_steps"] + 1,
            device=AI_DEVICE,
            dtype=torch.float32,
        )
        self.scheduler.set_timesteps(sigmas=sigmas[:-1], device=AI_DEVICE)
        self.timesteps = self.scheduler.timesteps
        self.infer_steps = len(self.timesteps)

    def prepare(self, input_info):
        if self.generator is None:
            self.generator = torch.Generator(device=AI_DEVICE).manual_seed(input_info.seed)
        self.prepare_latents(input_info)
        self.set_timesteps()

    def step_pre(self, step_index):
        super().step_pre(step_index)
        self.timestep = self.timesteps[step_index].to(device=AI_DEVICE, dtype=self.dtype)

    def step_post(self):
        latents = self.scheduler.step(
            self.noise_pred,
            self.timesteps[self.step_index],
            self.latents,
            return_dict=False,
        )[0]
        self.latents = latents
