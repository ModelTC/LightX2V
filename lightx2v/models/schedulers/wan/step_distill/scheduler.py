import math
from typing import Union

import torch

from lightx2v.models.schedulers.wan.scheduler import WanScheduler


class WanStepDistillScheduler(WanScheduler):
    def __init__(self, config):
        super().__init__(config)
        self.denoising_step_list = config["denoising_step_list"]
        self.infer_steps = len(self.denoising_step_list)
        self.sample_shift = self.config["sample_shift"]

        self.num_train_timesteps = 1000
        self.sigma_max = 1.0
        self.sigma_min = 0.0

    def prepare(self, seed, latent_shape, image_encoder_output=None):
        self.prepare_latents(seed, latent_shape, dtype=torch.float32)
        self.set_denoising_timesteps(device=self.device)

    def set_denoising_timesteps(self, device: Union[str, torch.device] = None):
        sigma_start = self.sigma_min + (self.sigma_max - self.sigma_min)
        self.sigmas = torch.linspace(sigma_start, self.sigma_min, self.num_train_timesteps + 1)[:-1]
        self.sigmas = self.sample_shift * self.sigmas / (1 + (self.sample_shift - 1) * self.sigmas)
        self.timesteps = self.sigmas * self.num_train_timesteps

        self.denoising_step_index = [self.num_train_timesteps - x for x in self.denoising_step_list]
        self.timesteps = self.timesteps[self.denoising_step_index].to(device)
        self.sigmas = self.sigmas[self.denoising_step_index].to("cpu")

    def reset(self, seed, latent_shape, step_index=None):
        self.prepare_latents(seed, latent_shape, dtype=torch.float32)

    def add_noise(self, original_samples, noise, sigma):
        sample = (1 - sigma) * original_samples + sigma * noise
        return sample.type_as(noise)

    def step_post(self):
        flow_pred = self.noise_pred.to(torch.float32)
        sigma = self.sigmas[self.step_index].item()
        noisy_image_or_video = self.latents.to(torch.float32) - sigma * flow_pred
        if self.step_index < self.infer_steps - 1:
            sigma_n = self.sigmas[self.step_index + 1].item()
            noisy_image_or_video = noisy_image_or_video + flow_pred * sigma_n
        self.latents = noisy_image_or_video.to(self.latents.dtype)


class Wan22StepDistillScheduler(WanStepDistillScheduler):
    def __init__(self, config):
        super().__init__(config)
        self.boundary_step_index = config["boundary_step_index"]

    def set_denoising_timesteps(self, device: Union[str, torch.device] = None):
        super().set_denoising_timesteps(device)
        self.sigma_bound = self.sigmas[self.boundary_step_index].item()

    def calculate_alpha_beta_high(self, sigma):
        alpha = (1 - sigma) / (1 - self.sigma_bound)
        beta = math.sqrt(sigma**2 - (alpha * self.sigma_bound) ** 2)
        return alpha, beta

    def step_post(self):
        flow_pred = self.noise_pred.to(torch.float32)
        sigma = self.sigmas[self.step_index].item()
        noisy_image_or_video = self.latents.to(torch.float32) - flow_pred * sigma
        # self.latent: x_t
        if self.step_index < self.infer_steps - 1:
            sigma_n = self.sigmas[self.step_index + 1].item()
            noisy_image_or_video = noisy_image_or_video + flow_pred * sigma_n

        self.latents = noisy_image_or_video.to(self.latents.dtype)
