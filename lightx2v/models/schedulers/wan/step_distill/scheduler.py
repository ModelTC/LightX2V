import math
from typing import Union

import torch

from lightx2v.models.schedulers.wan.scheduler import WanScheduler


class WanStepDistillScheduler(WanScheduler):
    def __init__(self, config):
        super().__init__(config)
        self.denoising_step_list = config.denoising_step_list
        self.infer_steps = len(self.denoising_step_list)
        self.sample_shift = self.config.sample_shift

        self.num_train_timesteps = 1000
        self.sigma_max = 1.0
        self.sigma_min = 0.0

    def prepare(self, image_encoder_output):
        self.generator = torch.Generator(device=self.device)
        self.generator.manual_seed(self.config.seed)

        self.prepare_latents(self.config.target_shape, dtype=torch.float32)

        if self.config.task in ["t2v"]:
            self.seq_len = math.ceil((self.config.target_shape[2] * self.config.target_shape[3]) / (self.config.patch_size[1] * self.config.patch_size[2]) * self.config.target_shape[1])
        elif self.config.task in ["i2v"]:
            self.seq_len = self.config.lat_h * self.config.lat_w // (self.config.patch_size[1] * self.config.patch_size[2]) * self.config.target_shape[1]

        self.set_denoising_timesteps(device=self.device)

    def set_denoising_timesteps(self, device: Union[str, torch.device] = None):
        sigma_start = self.sigma_min + (self.sigma_max - self.sigma_min)
        self.sigmas = torch.linspace(sigma_start, self.sigma_min, self.num_train_timesteps + 1)[:-1]
        self.sigmas = self.sample_shift * self.sigmas / (1 + (self.sample_shift - 1) * self.sigmas)
        self.timesteps = self.sigmas * self.num_train_timesteps

        self.denoising_step_index = [self.num_train_timesteps - x for x in self.denoising_step_list]
        self.timesteps = self.timesteps[self.denoising_step_index].to(device)
        self.sigmas = self.sigmas[self.denoising_step_index].to("cpu")

    def reset(self):
        self.prepare_latents(self.config.target_shape, dtype=torch.float32)

    def add_noise(self, original_samples, noise, sigma):
        sample = (1 - sigma) * original_samples + sigma * noise
        return sample.type_as(noise)

    def step_post(self):
        flow_pred = self.noise_pred.to(torch.float32)
        sigma = self.sigmas[self.step_index].item()
        noisy_image_or_video = self.latents.to(torch.float32) - sigma * flow_pred
        if self.step_index < self.infer_steps - 1:
            sigma = self.sigmas[self.step_index + 1].item()
            noise = torch.randn(noisy_image_or_video.shape, dtype=torch.float32, device=self.device, generator=self.generator)
            noisy_image_or_video = self.add_noise(noisy_image_or_video, noise=noise, sigma=self.sigmas[self.step_index + 1].item())
        self.latents = noisy_image_or_video.to(self.latents.dtype)


class Wan22StepDistillScheduler(WanStepDistillScheduler):
    def __init__(self, config):
        super().__init__(config)
        self.boundary_step_index = config.boundary_step_index

    def set_denoising_timesteps(self, device: Union[str, torch.device] = None):
        super().set_denoising_timesteps(device)
        self.sigma_boundary = self.sigmas[self.boundary_step_index].item()

    def step_post(self):
        flow_pred = self.noise_pred.to(torch.float32)
        sigma = self.sigmas[self.step_index].item()
        noisy_image_or_video = self.latents.to(torch.float32) - sigma * flow_pred
        if self.step_index < self.boundary_step_index:
            noisy_image_or_video = noisy_image_or_video / self.sigma_boundary
        if self.step_index < self.infer_steps - 1:
            sigma = self.sigmas[self.step_index + 1].item()
            noisy_image_or_video = self.add_noise(noisy_image_or_video, torch.randn_like(noisy_image_or_video), self.sigmas[self.step_index + 1].item())
        self.latents = noisy_image_or_video.to(self.latents.dtype)
