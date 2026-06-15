import torch

from .flow_matching import RectifiedFlowMatchingScheduler


class DMDFlowMatchingScheduler(RectifiedFlowMatchingScheduler):
    def __init__(self, config, dmd_config={}):
        super().__init__(config)
        self.renoise_shift = float(dmd_config.get("renoise_shift", 5.0))
        self.renoise_sigma_min = float(dmd_config.get("renoise_sigma_min", dmd_config.get("sigma_min", 0.02)))
        self.renoise_sigma_max = float(dmd_config.get("renoise_sigma_max", dmd_config.get("sigma_max", 1.0)))
        self.renoise_discrete_samples = int(dmd_config.get("renoise_discrete_samples", dmd_config.get("discrete_samples", 1000)))

    @staticmethod
    def linear_shift(mu, t):
        return mu / (mu + (1 / t - 1))

    def set_timesteps(self, num_inference_steps, sigmas=None, latent_hw=None, device=None):
        super().set_timesteps(num_inference_steps, sigmas=sigmas, latent_hw=latent_hw)
        if device is not None:
            self.infer_sigmas = self.infer_sigmas.to(device)
            self.infer_timesteps = self.infer_timesteps.to(device)
        self.sigmas = self.infer_sigmas
        self.timesteps = self.infer_timesteps

    def sigma_at(self, step_idx, batch_size, device=None, dtype=None):
        sigma = self.sigmas[int(step_idx)].expand(int(batch_size))
        if device is not None or dtype is not None:
            sigma = sigma.to(device=device, dtype=dtype)
        return sigma

    def sample_renoise_sigma(self, batch_size, device=None, dtype=None):
        device = device or self.device
        raw = torch.rand((int(batch_size),), device=device, dtype=torch.float32)
        if self.renoise_discrete_samples > 0:
            raw = torch.ceil(raw * self.renoise_discrete_samples) / self.renoise_discrete_samples
        raw = torch.clamp(raw, 1e-7, 1 - 1e-7)
        sigma = torch.clamp(self.linear_shift(self.renoise_shift, raw), self.renoise_sigma_min, self.renoise_sigma_max)
        if dtype is not None:
            sigma = sigma.to(dtype=dtype)
        return sigma

    def add_noise(self, latent, noise, sigmas):
        sigmas = sigmas.to(device=latent.device)
        sigmas = self._expand_to_ndim(sigmas, latent.ndim)
        return ((1.0 - sigmas) * latent + sigmas * noise).to(dtype=latent.dtype)

    def step_by_index(self, velocity, step_idx, sample):
        sigma = self.sigma_at(step_idx, sample.shape[0], device=sample.device)
        sigma_next = self.sigma_at(int(step_idx) + 1, sample.shape[0], device=sample.device)
        sigma = self._expand_to_ndim(sigma, sample.ndim)
        sigma_next = self._expand_to_ndim(sigma_next, sample.ndim)
        next_sample = sample + (sigma_next - sigma) * velocity
        x0 = sample - sigma * velocity
        return next_sample.to(sample.dtype), x0.to(sample.dtype)
