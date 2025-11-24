import torch
from lightx2v.models.schedulers.hunyuan_video.scheduler import HunyuanVideo15Scheduler


class HunyuanVideo15StepDistillScheduler(HunyuanVideo15Scheduler):
    def __init__(self, config):
        super().__init__(config)
        self.denoising_step_list = config["denoising_step_list"]
        self.infer_steps = len(self.denoising_step_list)

        self.num_train_timesteps = 1000
        self.sigma_max = 1.0
        self.sigma_min = 0.0

    def set_timesteps(self, num_inference_steps, device, shift):
        sigma_start = self.sigma_min + (self.sigma_max - self.sigma_min)
        self.sigmas = torch.linspace(sigma_start, self.sigma_min, self.num_train_timesteps + 1)[:-1]
        self.sigmas = self.sample_shift * self.sigmas / (1 + (self.sample_shift - 1) * self.sigmas)
        self.timesteps = self.sigmas * self.num_train_timesteps

        self.denoising_step_index = [self.num_train_timesteps - x for x in self.denoising_step_list]
        self.timesteps = self.timesteps[self.denoising_step_index].to(device)
        self.sigmas = self.sigmas[self.denoising_step_index].to("cpu")

    def step_post(self):
        flow_pred = self.noise_pred.to(torch.float32)
        sigma = self.sigmas[self.step_index].item()
        noisy_image_or_video = self.latents.to(torch.float32) - sigma * flow_pred
        if self.step_index < self.infer_steps - 1:
            sigma_n = self.sigmas[self.step_index + 1].item()
            noisy_image_or_video = noisy_image_or_video + flow_pred * sigma_n
        self.latents = noisy_image_or_video.to(self.latents.dtype)