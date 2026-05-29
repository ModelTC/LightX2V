import torch

from .flow_matching import RectifiedFlowMatchingScheduler


class DMDFlowMatchingScheduler(RectifiedFlowMatchingScheduler):
    def __init__(self, config, dmd_config={}):
        super().__init__(config)
        self.inference_shift = float(dmd_config.get("inference_shift", 3.0))
        self.renoise_shift = float(dmd_config.get("renoise_shift", 5.0))
        self.min_sigma = float(dmd_config.get("sigma_min", 0.02))
        self.max_sigma = float(dmd_config.get("sigma_max", 1.0))
        self.discrete_samples = int(dmd_config.get("discrete_samples", 1000))

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

    def set_random_timesteps(
        self,
        num_steps_min,
        num_steps_max,
        sigma_min=0.25,
        sigma_max=0.95,
        sampling_method="stratified",
        device=None,
    ):
        device = device or self.device
        num_steps = int(torch.randint(num_steps_min, num_steps_max + 1, (1,), device=device).item())
        inner_count = max(0, num_steps - 1)
        if inner_count:
            if sampling_method == "uniform":
                inner_sigmas = torch.empty(inner_count, dtype=torch.float32, device=device).uniform_(sigma_min, sigma_max)
            elif sampling_method == "stratified":
                bin_edges = torch.linspace(sigma_min, sigma_max, inner_count + 1, dtype=torch.float32, device=device)
                bin_lows = bin_edges[:-1]
                bin_highs = bin_edges[1:]
                inner_sigmas = bin_lows + torch.rand(inner_count, dtype=torch.float32, device=device) * (bin_highs - bin_lows)
            else:
                raise ValueError(f"Unsupported random sigma sampling_method: {sampling_method}")
            inner_sigmas = self.linear_shift(self.inference_shift, inner_sigmas)
            inner_sigmas = torch.sort(inner_sigmas, descending=True).values
            sigmas = torch.cat(
                [
                    torch.ones(1, dtype=torch.float32, device=device),
                    inner_sigmas,
                    torch.zeros(1, dtype=torch.float32, device=device),
                ]
            )
        else:
            sigmas = torch.tensor([1.0, 0.0], dtype=torch.float32, device=device)
        self.sigmas = sigmas
        self.timesteps = self.sigmas * self.num_train_timesteps
        self.num_inference_steps = int(sigmas.numel() - 1)

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
        return ((1.0 - sigmas) * latent + sigmas * noise).to(dtype=latent.dtype)

    def step_by_index(self, velocity, step_idx, sample):
        sigma = self.sigma_at(step_idx, sample.shape[0], device=sample.device)
        sigma_next = self.sigma_at(int(step_idx) + 1, sample.shape[0], device=sample.device)
        next_sample = sample + (sigma_next - sigma) * velocity
        x0 = sample - sigma * velocity
        return next_sample.to(sample.dtype), x0.to(sample.dtype)
