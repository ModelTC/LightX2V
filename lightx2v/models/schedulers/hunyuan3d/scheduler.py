"""Scheduler for Hunyuan3D-2.1 shape flow-matching inference."""

from __future__ import annotations

import numpy as np
import torch
from diffusers.utils.torch_utils import randn_tensor

from lightx2v.models.networks.hunyuan3d.utils.checkpoint import load_pipeline_config, resolve_model_dir
from lightx2v.models.schedulers.hunyuan3d.flow_match_euler import FlowMatchEulerDiscreteScheduler
from lightx2v.models.schedulers.scheduler import BaseScheduler
from lightx2v.utils.envs import GET_DTYPE
from lightx2v_platform.base.global_var import AI_DEVICE


class Hunyuan3DShapeScheduler(BaseScheduler):
    """LightX2V scheduler wrapping Hunyuan3D flow-match Euler steps."""

    def __init__(self, config):
        super().__init__(config)
        model_path = config["model_path"]
        subfolder = config.get("subfolder", "hunyuan3d-dit-v2-1")
        model_dir = resolve_model_dir(model_path, subfolder)
        pipeline_cfg = load_pipeline_config(f"{model_dir}/config.yaml")
        scheduler_params = pipeline_cfg["scheduler"]["params"]
        self.flow_scheduler = FlowMatchEulerDiscreteScheduler(**scheduler_params)
        self.num_train_timesteps = scheduler_params["num_train_timesteps"]
        self.keep_latents_dtype_in_scheduler = True
        self.noise_pred = None
        self.generator = None
        self.device = torch.device(config.get("device", AI_DEVICE))
        self.dtype = GET_DTYPE()
        self.current_timestep = None

    def prepare(self, seed, batch_size=1, latent_shape=None):
        self.infer_steps = self.config["infer_steps"]

        sigmas = np.linspace(0, 1, self.infer_steps)
        self.flow_scheduler.set_timesteps(sigmas=sigmas, device=self.device)
        self.timesteps = self.flow_scheduler.timesteps

        self.generator = torch.Generator(device=self.device).manual_seed(seed)

        if latent_shape is None:
            raise ValueError("latent_shape must be provided to Hunyuan3DShapeScheduler.prepare")
        self.latents = randn_tensor(latent_shape, generator=self.generator, device=self.device, dtype=self.dtype)
        self.noise_pred = None
        self.step_index = 0

    def step_pre(self, step_index):
        super().step_pre(step_index)
        self.current_timestep = self.timesteps[step_index]

    def step_post(self):
        outputs = self.flow_scheduler.step(self.noise_pred, self.current_timestep, self.latents)
        self.latents = outputs.prev_sample
        self.noise_pred = None

    def clear(self):
        self.latents = None
        self.noise_pred = None
        self.generator = None
        self.current_timestep = None
        self.timesteps = None
        self.flow_scheduler.timesteps = None
        self.flow_scheduler.sigmas = None
