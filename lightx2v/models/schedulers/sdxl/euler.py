"""Euler-discrete schedule matching the official SDXL base scheduler config."""

import numpy as np
import torch


class SDXLEulerScheduler:
    def __init__(self, config, steps, device="cuda"):
        if config["beta_schedule"] != "scaled_linear" or config["prediction_type"] != "epsilon" or config.get("use_karras_sigmas", False):
            raise ValueError("Only the official SDXL base Euler schedule is supported")
        n = config["num_train_timesteps"]
        if not isinstance(steps, int) or not 1 <= steps <= n:
            raise ValueError(f"infer_steps must be an integer in [1,{n}]")
        betas = torch.linspace(config["beta_start"] ** 0.5, config["beta_end"] ** 0.5, n, dtype=torch.float32) ** 2
        alphas = torch.cumprod(1 - betas, dim=0)
        training_sigmas = ((1 - alphas) / alphas).sqrt().numpy()
        spacing = config.get("timestep_spacing", "linspace")
        if spacing == "leading":
            timesteps = (np.arange(steps) * (n // steps)).round()[::-1].copy().astype(np.float32)
            timesteps += config.get("steps_offset", 0)
        elif spacing == "linspace":
            timesteps = np.linspace(0, n - 1, steps, dtype=np.float32)[::-1].copy()
        else:
            raise ValueError(f"Unsupported timestep spacing: {spacing}")
        sigmas = np.interp(timesteps, np.arange(n), training_sigmas)
        # The native scheduler keeps sigmas on CPU. A CPU scalar and a CUDA
        # zero-dimensional tensor follow different mixed-precision promotion
        # paths; moving this cache to CUDA changes FP16 noise and Euler steps.
        self.sigmas = torch.tensor(np.concatenate((sigmas, [0])), dtype=torch.float32, device="cpu")
        self.timesteps = torch.tensor(timesteps, device=device)
        self.init_noise_sigma = (self.sigmas.max() ** 2 + 1).sqrt() if spacing == "leading" else self.sigmas.max()

    def scale_model_input(self, x, index):
        return x / (self.sigmas[index] ** 2 + 1).sqrt()

    def step(self, noise, index, sample):
        sigma = self.sigmas[index]
        current = sample.float()
        predicted = current - sigma * noise
        derivative = (current - predicted) / sigma
        return (current + derivative * (self.sigmas[index + 1] - sigma)).to(noise.dtype)
