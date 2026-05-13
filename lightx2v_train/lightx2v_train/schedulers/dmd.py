import torch

from .flow_matching import RectifiedFlowMatchingScheduler


class DMDFlowMatchingScheduler(RectifiedFlowMatchingScheduler):
    def __init__(self, config, dmd_config=None):
        super().__init__(config)
        dmd_config = dmd_config or {}
        self.inference_shift = float(dmd_config.get("inference_shift", 3.0))
        self.renoise_shift = float(dmd_config.get("renoise_shift", 5.0))
        self.min_sigma = float(dmd_config.get("sigma_min", 0.02))
        self.max_sigma = float(dmd_config.get("sigma_max", 1.0))
        self.discrete_samples = int(dmd_config.get("discrete_samples", 1000))

    @staticmethod
    def expand_to(value, target):
        value = value.to(device=target.device)
        while value.ndim < target.ndim:
            value = value.view(*value.shape, 1)
        return value

    @staticmethod
    def linear_shift(mu, t):
        return mu / (mu + (1 / t - 1))

    def set_timesteps(self, num_inference_steps, device=None):
        self.num_inference_steps = int(num_inference_steps)
        device = device or self.device
        timesteps = torch.linspace(
            self.num_train_timesteps,
            0,
            self.num_inference_steps + 1,
            dtype=torch.float32,
            device=device,
        )
        self.sigmas = self.linear_shift(self.inference_shift, timesteps / self.num_train_timesteps)
        self.timesteps = self.sigmas * self.num_train_timesteps

    def sigma_at(self, step_idx, batch_size, device=None, dtype=None):
        sigma = self.sigmas[int(step_idx)].expand(int(batch_size))
        if device is not None or dtype is not None:
            sigma = sigma.to(device=device, dtype=dtype)
        return sigma

    def sample_renoise_sigma(self, batch_size, device=None, dtype=None):
        device = device or self.device
        raw = torch.rand((int(batch_size),), device=device, dtype=torch.float32)
        if self.discrete_samples > 0:
            raw = torch.ceil(raw * self.discrete_samples) / self.discrete_samples
        raw = torch.clamp(raw, 1e-7, 1 - 1e-7)
        sigma = torch.clamp(self.linear_shift(self.renoise_shift, raw), self.min_sigma, self.max_sigma)
        if dtype is not None:
            sigma = sigma.to(dtype=dtype)
        return sigma

    def add_noise(self, latent, noise, sigmas):
        sigmas = self.expand_to(sigmas, latent).to(dtype=torch.float32)
        return ((1.0 - sigmas) * latent.float() + sigmas * noise.float()).to(dtype=latent.dtype)

    def euler_step(self, sample, velocity, sigma, target_sigma):
        sigma = self.expand_to(sigma, sample).to(dtype=torch.float32)
        target_sigma = self.expand_to(target_sigma, sample).to(dtype=torch.float32)
        return sample.float() + (target_sigma - sigma) * velocity.float()

    def step_by_index(self, model_output, step_idx, sample):
        sigma = self.sigma_at(step_idx, sample.shape[0], device=sample.device)
        sigma_next = self.sigma_at(int(step_idx) + 1, sample.shape[0], device=sample.device)
        x0 = sample.float() - self.expand_to(sigma, sample).float() * model_output.float()
        next_sample = self.euler_step(sample, model_output, sigma, sigma_next)
        return next_sample.to(sample.dtype), x0.to(sample.dtype)
