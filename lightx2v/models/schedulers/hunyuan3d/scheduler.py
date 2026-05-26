"""Scheduler for Hunyuan3D-2.1 shape flow-matching inference."""

from __future__ import annotations

import inspect
from typing import List, Optional, Union

import numpy as np
import torch
from diffusers.utils.torch_utils import randn_tensor

from lightx2v.models.networks.hunyuan3d.utils.checkpoint import load_pipeline_config, resolve_model_dir
from lightx2v.models.schedulers.hunyuan3d.flow_match_euler import FlowMatchEulerDiscreteScheduler
from lightx2v.models.schedulers.scheduler import BaseScheduler
from lightx2v.utils.envs import GET_DTYPE
from lightx2v_platform.base.global_var import AI_DEVICE


def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    **kwargs,
):
    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed.")
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            raise ValueError(f"{scheduler.__class__}'s `set_timesteps` does not support custom timesteps.")
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accept_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accept_sigmas:
            raise ValueError(f"{scheduler.__class__}'s `set_timesteps` does not support custom sigmas.")
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps


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

    def prepare(self, seed=None, batch_size=1, latent_shape=None):
        infer_steps = int(self.config.get("infer_steps", 50))
        self.infer_steps = infer_steps

        sigmas = np.linspace(0, 1, infer_steps)
        timesteps, _ = retrieve_timesteps(
            self.flow_scheduler,
            infer_steps,
            self.device,
            sigmas=sigmas,
        )
        self.timesteps = timesteps

        if seed is not None:
            self.generator = torch.Generator(device=self.device).manual_seed(int(seed))
        else:
            self.generator = None

        if latent_shape is None:
            raise ValueError("latent_shape must be provided to Hunyuan3DShapeScheduler.prepare")
        latents = randn_tensor(latent_shape, generator=self.generator, device=self.device, dtype=self.dtype)
        init_noise_sigma = getattr(self.flow_scheduler, "init_noise_sigma", 1.0)
        self.latents = latents * init_noise_sigma
        self.noise_pred = None
        self.step_index = 0

    def step_pre(self, step_index):
        super().step_pre(step_index)
        self.current_timestep = self.timesteps[step_index]

    def step_post(self):
        outputs = self.flow_scheduler.step(self.noise_pred, self.current_timestep, self.latents)
        self.latents = outputs.prev_sample
        self.noise_pred = None
