import torch
from diffusers.schedulers.scheduling_utils import SchedulerOutput

from lightx2v_train.utils.utils import get_running_dtype


class RectifiedFlowMatchingScheduler:
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda")

        scheduler_training_config = config["scheduler"]["training"]
        self.num_train_timesteps = scheduler_training_config.get("num_train_timesteps", 1000)
        self.timestep_distribution = scheduler_training_config.get("timestep_distribution", "logitnormal")

        self.logitnormal_mean = scheduler_training_config.get("logitnormal_mean", 0.0)
        self.logitnormal_std = scheduler_training_config.get("logitnormal_std", 1.0)

        self.min_t = scheduler_training_config.get("min_t", 0.001)
        self.max_t = scheduler_training_config.get("max_t", 1.0)

        self.do_time_shift = scheduler_training_config.get("do_time_shift", False)
        self.time_shift_mu = scheduler_training_config.get("time_shift_mu", 5.0)
        self.time_shift_power = scheduler_training_config.get("time_shift_power", 1.0)

        _sigmas = torch.linspace(1.0, 1.0 / self.num_train_timesteps, self.num_train_timesteps)
        self._train_sigmas = _sigmas
        self._train_timesteps = _sigmas * self.num_train_timesteps

        self.sigmas = torch.cat([_sigmas, torch.zeros(1)])
        self.timesteps = self._train_timesteps
        self.num_inference_steps = None

        self.running_dtype = get_running_dtype(config["model"]["running_dtype"])

    def sample_timestep_or_sigma(self, num_samples):
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
            timestep_or_sigma = self.time_shift(timestep_or_sigma)
        return timestep_or_sigma.to(self.running_dtype)

    def time_shift(self, t):
        return self.time_shift_mu / (self.time_shift_mu + (1 / t - 1) ** self.time_shift_power)

    def add_noise(self, latent, noise, sigmas):
        return (1.0 - sigmas) * latent + sigmas * noise

    def build_train_gt(self, latent, noise):
        return noise - latent

    def set_timesteps(self, num_inference_steps, device=None):
        self.num_inference_steps = num_inference_steps
        device = device or self.device

        sigmas = torch.linspace(1.0, 1.0 / num_inference_steps, num_inference_steps)
        self.sigmas = torch.cat([sigmas, torch.zeros(1)]).to(device)
        self.timesteps = (sigmas * self.num_train_timesteps).to(device)

    def step(self, model_output, timestep, sample, return_dict=True):
        step_index = (self.timesteps == timestep).nonzero()[0].item()
        sigma = self.sigmas[step_index]
        sigma_next = self.sigmas[step_index + 1]

        prev_sample = sample + (sigma_next - sigma) * model_output

        if not return_dict:
            return (prev_sample,)
        return SchedulerOutput(prev_sample=prev_sample)

    def scale_model_input(self, sample, timestep=None):
        return sample
