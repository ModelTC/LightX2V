import math

import torch

from lightx2v_train.runtime.distributed import get_device
from lightx2v_train.schedulers.time_shift import build_time_shift_mu
from lightx2v_train.utils.utils import get_running_dtype


class RectifiedFlowMatchingScheduler:
    def __init__(self, config):
        self.config = config
        self.device = get_device()

        scheduler_config = config["scheduler"]
        self.num_train_timesteps = scheduler_config.get("num_train_timesteps", 1000)
        self.timestep_distribution = scheduler_config.get("timestep_distribution", "logitnormal")

        self.logitnormal_mean = scheduler_config.get("logitnormal_mean", 0.0)
        self.logitnormal_std = scheduler_config.get("logitnormal_std", 1.0)

        self.min_t = scheduler_config.get("min_t", 0.001)
        self.max_t = scheduler_config.get("max_t", 1.0)

        time_shift_settings = scheduler_config["time_shift_settings"]
        self.do_time_shift = time_shift_settings.get("do_time_shift", False)
        self.time_shift_power = time_shift_settings.get("time_shift_power", 1.0)
        self.shift_type = time_shift_settings.get("shift_type", "linear")
        self.time_shift_mu = build_time_shift_mu(time_shift_settings)

        self.running_dtype = get_running_dtype(config["model"]["running_dtype"])

        # ==============================
        # The following attributes are for inference only
        # ==============================
        self.infer_sigmas = None
        self.infer_timesteps = None
        self.num_inference_steps = None

    def sample_timestep_or_sigma(self, num_samples, latent_hw=None):
        if self.timestep_distribution == "logitnormal":
            timestep_or_sigma = torch.randn((num_samples,), device=self.device, dtype=torch.float32) * self.logitnormal_std + self.logitnormal_mean
            timestep_or_sigma = torch.sigmoid(timestep_or_sigma)
            timestep_or_sigma = timestep_or_sigma * (self.max_t - self.min_t) + self.min_t  # [0, 1] -> [min_t, max_t]
        elif self.timestep_distribution == "uniform":
            timestep_or_sigma = torch.rand((num_samples,), device=self.device)
            timestep_or_sigma = timestep_or_sigma * (self.max_t - self.min_t) + self.min_t  # [0, 1] -> [min_t, max_t]
        else:
            raise ValueError(f"Unsupported timestep distribution: {self.timestep_distribution}")
        if self.do_time_shift:
            timestep_or_sigma = self.time_shift(timestep_or_sigma, latent_hw=latent_hw)
        return timestep_or_sigma.to(self.running_dtype)

    def time_shift(self, t, latent_hw=None, num_steps=None):
        mu = self.time_shift_mu(latent_hw=latent_hw, num_steps=num_steps)
        if self.shift_type == "exponential":
            mu = math.exp(mu)
        return mu / (mu + (1 / t - 1) ** self.time_shift_power)

    def add_noise(self, latent, noise, sigmas):
        return (1.0 - sigmas) * latent + sigmas * noise

    def build_train_gt(self, latent, noise):
        return noise - latent

    # ==============================
    # The following methods are for inference only
    # ==============================
    def set_timesteps(self, num_inference_steps, sigmas=None, latent_hw=None):
        self.num_inference_steps = num_inference_steps

        if sigmas is None:
            sigmas = torch.linspace(1.0, 1.0 / num_inference_steps, num_inference_steps)
            if self.do_time_shift:
                sigmas = self.time_shift(sigmas, latent_hw=latent_hw, num_steps=num_inference_steps)
        else:
            sigmas = torch.tensor(sigmas, dtype=torch.float32)
        self.infer_sigmas = torch.cat([sigmas, torch.zeros(1)]).to(self.device)
        self.infer_timesteps = (sigmas * self.num_train_timesteps).to(self.device)

    def step(self, model_output, step_index, latent):
        f"""
        ADD NOISE:
            x_t = (1 - sigma_t) * x_0 + sigma_t * N  ------ self.add_noise(...)
            =>  x_t = sigma_t * (N - x_0) + x_0
            =>  x_t = sigma_t * v + x_0
        REMOVE NOISE:
            x_t = sigma_t * v + x_0
            x_t-1 = sigma_t-1 * v + x_0
            =>  x_t - x_t-1 = (sigma_t - sigma_t-1) * v
            =>  x_t-1 = x_t + (sigma_t-1 - sigma_t) * v
            =>  x_t-1 = x_t + (sigma_next - sigma) * model_output  ------------------------ (*)
        """
        sigma = self.infer_sigmas[step_index]
        sigma_next = self.infer_sigmas[step_index + 1]
        prev_sample = latent + (sigma_next - sigma) * model_output  # --------------------- (*) from above
        return prev_sample
