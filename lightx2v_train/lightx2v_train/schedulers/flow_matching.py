from types import SimpleNamespace

import torch
from diffusers.schedulers.scheduling_utils import SchedulerOutput
from diffusers.training_utils import compute_density_for_timestep_sampling, compute_loss_weighting_for_sd3


class RectifiedFlowMatchingScheduler:
    def __init__(self, num_train_timesteps=1000):
        self.num_train_timesteps = num_train_timesteps
        self.device = torch.device("cuda")
        self.config = SimpleNamespace(num_train_timesteps=num_train_timesteps)

        _sigmas = torch.linspace(1.0, 1.0 / num_train_timesteps, num_train_timesteps)
        self._train_sigmas = _sigmas
        self._train_timesteps = _sigmas * num_train_timesteps

        self.sigmas = torch.cat([_sigmas, torch.zeros(1)])
        self.timesteps = self._train_timesteps
        self.num_inference_steps = None

    def sample_timesteps(self, num_samples, latent_device):
        u = compute_density_for_timestep_sampling(
            weighting_scheme="none",
            batch_size=num_samples,
            logit_mean=0.0,
            logit_std=1.0,
            mode_scale=1.29,
        )
        indices = (u * self.num_train_timesteps).long()
        return self._train_timesteps[indices].to(device=latent_device)

    def get_sigmas(self, timesteps, n_dim, dtype):
        sigmas = self._train_sigmas.to(device=self.device, dtype=dtype)
        schedule_timesteps = self._train_timesteps.to(self.device)
        timesteps = timesteps.to(self.device)
        sigma_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]
        sigma = sigmas[sigma_indices].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma

    def add_noise(self, latent, noise, sigmas):
        return (1.0 - sigmas) * latent + sigmas * noise

    def build_train_gt(self, latent, noise):
        return noise - latent

    def loss_weighting(self, sigmas):
        return compute_loss_weighting_for_sd3(weighting_scheme="none", sigmas=sigmas)

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
