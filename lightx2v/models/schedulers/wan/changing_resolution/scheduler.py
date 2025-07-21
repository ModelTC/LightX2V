import torch
from lightx2v.models.schedulers.wan.scheduler import WanScheduler


class WanScheduler4ChangingResolution(WanScheduler):
    def __init__(self, config):
        super().__init__(config)
        self.resolution_rate = config.get("resolution_rate", 0.75)
        self.changing_resolution_steps = config.get("changing_resolution_steps", config.infer_steps // 2)

    def prepare_latents(self, target_shape, dtype=torch.float32):
        self.latents = torch.randn(
            target_shape[0],
            target_shape[1],
            int(target_shape[2] * self.resolution_rate) // 2 * 2,
            int(target_shape[3] * self.resolution_rate) // 2 * 2,
            dtype=dtype,
            device=self.device,
            generator=self.generator,
        )

        self.noise_original_resolution = torch.randn(
            target_shape[0],
            target_shape[1],
            target_shape[2],
            target_shape[3],
            dtype=dtype,
            device=self.device,
            generator=self.generator,
        )

    def step_post(self):
        if self.step_index == self.changing_resolution_steps:
            self.step_post_upsample()
        else:
            super().step_post()

    def step_post_upsample(self):
        # 1. denoised sample to clean noise
        model_output = self.noise_pred.to(torch.float32)
        sample = self.latents.to(torch.float32)
        sigma_t = self.sigmas[self.step_index]
        x0_pred = sample - sigma_t * model_output
        denoised_sample = x0_pred.to(sample.dtype)

        # 2. upsample clean noise to target shape
        denoised_sample_5d = denoised_sample.unsqueeze(0)  # (C,T,H,W) -> (1,C,T,H,W)
        clean_noise = torch.nn.functional.interpolate(denoised_sample_5d, size=(self.config.target_shape[1], self.config.target_shape[2], self.config.target_shape[3]), mode="trilinear")
        clean_noise = clean_noise.squeeze(0)  # (1,C,T,H,W) -> (C,T,H,W)

        # 3. add noise to clean noise
        noisy_sample = self.add_noise(clean_noise, self.noise_original_resolution, self.timesteps[self.step_index + 1])

        # 4. update latents
        self.latents = noisy_sample

        # self.disable_corrector = [24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37] # maybe not needed

        # 5. update timesteps using shift + 2 更激进的去噪
        self.set_timesteps(self.infer_steps, device=self.device, shift=self.sample_shift + 2)

    def add_noise(self, original_samples, noise, timesteps):
        sigma = self.sigmas[self.step_index]
        alpha_t, sigma_t = self._sigma_to_alpha_sigma_t(sigma)
        noisy_samples = alpha_t * original_samples + sigma_t * noise
        return noisy_samples
