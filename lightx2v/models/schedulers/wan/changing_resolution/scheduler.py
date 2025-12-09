import torch

from lightx2v_platform.base.global_var import AI_DEVICE


class WanScheduler4ChangingResolutionInterface:
    def __new__(cls, father_scheduler, config):
        class NewClass(WanScheduler4ChangingResolution, father_scheduler):
            def __init__(self, config):
                father_scheduler.__init__(self, config)
                WanScheduler4ChangingResolution.__init__(self, config)

        return NewClass(config)


class WanScheduler4ChangingResolution:
    def __init__(self, config):
        if "resolution_rate" not in config:
            config["resolution_rate"] = [0.75]
        if "changing_resolution_steps" not in config:
            config["changing_resolution_steps"] = [config.infer_steps // 2]
        assert len(config["resolution_rate"]) == len(config["changing_resolution_steps"])

    def prepare_latents(self, seed, latent_shape, dtype=torch.float32):
        self.generator = torch.Generator(device=AI_DEVICE).manual_seed(seed)
        self.latents_list = []
        for i in range(len(self.config["resolution_rate"])):
            self.latents_list.append(
                torch.randn(
                    latent_shape[0],
                    latent_shape[1],
                    int(latent_shape[2] * self.config["resolution_rate"][i]) // 2 * 2,
                    int(latent_shape[3] * self.config["resolution_rate"][i]) // 2 * 2,
                    dtype=dtype,
                    device=AI_DEVICE,
                    generator=self.generator,
                )
            )

        # add original resolution latents
        self.latents_list.append(
            torch.randn(
                latent_shape[0],
                latent_shape[1],
                latent_shape[2],
                latent_shape[3],
                dtype=dtype,
                device=AI_DEVICE,
                generator=self.generator,
            )
        )

        # set initial latents
        self.latents = self.latents_list[0]
        self.changing_resolution_index = 0

    def step_post(self):
        if self.step_index + 1 in self.config["changing_resolution_steps"]:
            self.step_post_upsample()
            self.changing_resolution_index += 1
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

        shape_to_upsampled = self.latents_list[self.changing_resolution_index + 1].shape[1:]
        clean_noise = torch.nn.functional.interpolate(denoised_sample_5d, size=shape_to_upsampled, mode="trilinear")
        clean_noise = clean_noise.squeeze(0)  # (1,C,T,H,W) -> (C,T,H,W)

        # 3. add noise to clean noise
        noisy_sample = self.add_noise(clean_noise, self.latents_list[self.changing_resolution_index + 1], self.timesteps[self.step_index + 1])

        # 4. update latents
        self.latents = noisy_sample

        # self.disable_corrector = [24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37] # maybe not needed

        # 5. update timesteps using shift + self.changing_resolution_index + 1 更激进的去噪
        self.set_timesteps(self.infer_steps, device=AI_DEVICE, shift=self.sample_shift + self.changing_resolution_index + 1)

    def add_noise(self, original_samples, noise, timesteps):
        sigma = self.sigmas[self.step_index]
        alpha_t, sigma_t = self._sigma_to_alpha_sigma_t(sigma)
        noisy_samples = alpha_t * original_samples + sigma_t * noise
        return noisy_samples
