import inspect
import json
import os
from typing import List, Optional, Union

import numpy as np
import torch
from diffusers.schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler
from diffusers.utils.torch_utils import randn_tensor

from lightx2v.models.schedulers.scheduler import BaseScheduler


def calculate_shift(
    image_seq_len,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
):
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    mu = image_seq_len * m + b
    return mu


# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.retrieve_timesteps
def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    **kwargs,
):
    r"""
    Calls the scheduler's `set_timesteps` method and retrieves timesteps from the scheduler after the call. Handles
    custom timesteps. Any kwargs will be supplied to `scheduler.set_timesteps`.

    Args:
        scheduler (`SchedulerMixin`):
            The scheduler to get timesteps from.
        num_inference_steps (`int`):
            The number of diffusion steps used when generating samples with a pre-trained model. If used, `timesteps`
            must be `None`.
        device (`str` or `torch.device`, *optional*):
            The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        timesteps (`List[int]`, *optional*):
            Custom timesteps used to override the timestep spacing strategy of the scheduler. If `timesteps` is passed,
            `num_inference_steps` and `sigmas` must be `None`.
        sigmas (`List[float]`, *optional*):
            Custom sigmas used to override the timestep spacing strategy of the scheduler. If `sigmas` is passed,
            `num_inference_steps` and `timesteps` must be `None`.

    Returns:
        `Tuple[torch.Tensor, int]`: A tuple where the first element is the timestep schedule from the scheduler and the
        second element is the number of inference steps.
    """
    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values")
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accept_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accept_sigmas:
            raise ValueError(f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom sigmas schedules. Please check whether you are using the correct scheduler.")
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps


class QwenImageScheduler(BaseScheduler):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(os.path.join(config.model_path, "scheduler"))
        with open(os.path.join(config.model_path, "scheduler", "scheduler_config.json"), "r") as f:
            self.scheduler_config = json.load(f)
        self.generator = torch.Generator(device="cuda").manual_seed(config.seed)
        self.device = torch.device("cuda")
        self.dtype = torch.bfloat16
        self.guidance_scale = 1.0

    @staticmethod
    def _pack_latents(latents, batch_size, num_channels_latents, height, width):
        latents = latents.view(batch_size, num_channels_latents, height // 2, 2, width // 2, 2)
        latents = latents.permute(0, 2, 4, 1, 3, 5)
        latents = latents.reshape(batch_size, (height // 2) * (width // 2), num_channels_latents * 4)
        return latents

    @staticmethod
    def _unpack_latents(latents, height, width, vae_scale_factor):
        batch_size, num_patches, channels = latents.shape

        # VAE applies 8x compression on images but we must also account for packing which requires
        # latent height and width to be divisible by 2.
        height = 2 * (int(height) // (vae_scale_factor * 2))
        width = 2 * (int(width) // (vae_scale_factor * 2))

        latents = latents.view(batch_size, height // 2, width // 2, channels // 4, 2, 2)
        latents = latents.permute(0, 3, 1, 4, 2, 5)

        latents = latents.reshape(batch_size, channels // (2 * 2), 1, height, width)

        return latents

    @staticmethod
    def _prepare_latent_image_ids(batch_size, height, width, device, dtype):
        latent_image_ids = torch.zeros(height, width, 3)
        latent_image_ids[..., 1] = latent_image_ids[..., 1] + torch.arange(height)[:, None]
        latent_image_ids[..., 2] = latent_image_ids[..., 2] + torch.arange(width)[None, :]

        latent_image_id_height, latent_image_id_width, latent_image_id_channels = latent_image_ids.shape

        latent_image_ids = latent_image_ids.reshape(latent_image_id_height * latent_image_id_width, latent_image_id_channels)

        return latent_image_ids.to(device=device, dtype=dtype)

    def prepare_latents(self):
        shape = self.config.target_shape
        width, height = shape[-1], shape[-2]
        latents = randn_tensor(shape, generator=self.generator, device=self.device, dtype=self.dtype)
        latents = self._pack_latents(latents, self.config.batchsize, self.config.num_channels_latents, height, width)
        latent_image_ids = self._prepare_latent_image_ids(self.config.batchsize, height // 2, width // 2, self.device, self.dtype)
        self.latents = latents
        self.latent_image_ids = latent_image_ids
        self.noise_pred = None

    def set_timesteps(self):
        sigmas = np.linspace(1.0, 1 / self.config.infer_steps, self.config.infer_steps)
        image_seq_len = self.latents.shape[1]
        mu = calculate_shift(
            image_seq_len,
            self.scheduler_config.get("base_image_seq_len", 256),
            self.scheduler_config.get("max_image_seq_len", 4096),
            self.scheduler_config.get("base_shift", 0.5),
            self.scheduler_config.get("max_shift", 1.15),
        )
        num_inference_steps = self.config.infer_steps
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler,
            num_inference_steps,
            self.device,
            sigmas=sigmas,
            mu=mu,
        )

        self.timesteps = timesteps
        self.infer_steps = num_inference_steps

        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
        self._num_timesteps = len(timesteps)
        self.num_warmup_steps = num_warmup_steps

    def prepare_guidance(self):
        # handle guidance
        if self.config.guidance_embeds:
            guidance = torch.full([1], self.guidance_scale, device=self.device, dtype=torch.float32)
            guidance = guidance.expand(self.latents.shape[0])
        else:
            guidance = None
        self.guidance = guidance

    def set_img_shapes(self, inputs):
        if self.config.task == "t2i":
            width, height = self.config.aspect_ratios[self.config.aspect_ratio]
            self.img_shapes = [(1, height // self.config.vae_scale_factor // 2, width // self.config.vae_scale_factor // 2)] * self.config.batchsize
        elif self.config.task == "i2i":
            image_height, image_width = inputs["text_encoder_output"]["image_info"]
            self.img_shapes = [
                [
                    (1, self.config.auto_hight // self.config.vae_scale_factor // 2, self.config.auto_width // self.config.vae_scale_factor // 2),
                    (1, image_height // self.config.vae_scale_factor // 2, image_width // self.config.vae_scale_factor // 2),
                ]
            ]

    def prepare(self, inputs):
        self.prepare_latents()
        self.prepare_guidance()
        self.set_img_shapes(inputs)
        self.set_timesteps()

    def step_post(self):
        # compute the previous noisy sample x_t -> x_t-1
        t = self.timesteps[self.step_index]
        latents = self.scheduler.step(self.noise_pred, t, self.latents, return_dict=False)[0]
        self.latents = latents
