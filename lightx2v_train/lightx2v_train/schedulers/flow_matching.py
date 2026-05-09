from types import SimpleNamespace

import torch
from diffusers.schedulers.scheduling_utils import SchedulerOutput
from diffusers.training_utils import compute_density_for_timestep_sampling, compute_loss_weighting_for_sd3


class FlowMatchingScheduler:
    """Unified flow-matching scheduler for both training and inference.

    Replicates the sigma / timestep layout of FlowMatchEulerDiscreteScheduler:

        sigmas    = linspace(1.0, 1/T, T)     e.g. [1.000, 0.999, ..., 0.001]
        timesteps = sigmas * T                  e.g. [1000.,  999., ...,    1.]

    Training helpers (sample_timesteps, get_sigmas, add_noise, …) always draw
    from the full T-step table.  Inference helpers (set_timesteps, step) operate
    on a subsampled table set at inference time, matching the diffusers scheduler
    interface expected by pipeline classes.
    """

    def __init__(self, num_train_timesteps: int = 1000, device: torch.device = None):
        self.num_train_timesteps = num_train_timesteps
        self.device = device or torch.device("cpu")
        self.config = SimpleNamespace(num_train_timesteps=num_train_timesteps)

        # Full training table — never mutated after init
        _sigmas = torch.linspace(1.0, 1.0 / num_train_timesteps, num_train_timesteps)
        self._train_sigmas = _sigmas  # shape: (T,)
        self._train_timesteps = _sigmas * num_train_timesteps  # shape: (T,)

        # Inference table — overwritten by set_timesteps(); defaults to full table
        self.sigmas = torch.cat([_sigmas, torch.zeros(1)])  # shape: (T+1,)
        self.timesteps = self._train_timesteps  # shape: (T,)
        self.num_inference_steps: int | None = None

    # ------------------------------------------------------------------
    # Training — timestep sampling
    # ------------------------------------------------------------------

    def sample_timesteps(self, num_samples: int, latent_device: torch.device) -> torch.Tensor:
        u = compute_density_for_timestep_sampling(
            weighting_scheme="none",
            batch_size=num_samples,
            logit_mean=0.0,
            logit_std=1.0,
            mode_scale=1.29,
        )
        indices = (u * self.num_train_timesteps).long()
        return self._train_timesteps[indices].to(device=latent_device)

    # ------------------------------------------------------------------
    # Training — sigma utilities
    # ------------------------------------------------------------------

    def get_sigmas(self, timesteps: torch.Tensor, n_dim: int, dtype: torch.dtype) -> torch.Tensor:
        sigmas = self._train_sigmas.to(device=self.device, dtype=dtype)
        schedule_timesteps = self._train_timesteps.to(self.device)
        timesteps = timesteps.to(self.device)
        sigma_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]
        sigma = sigmas[sigma_indices].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma

    # ------------------------------------------------------------------
    # Training — noise / target helpers
    # ------------------------------------------------------------------

    def add_noise(self, latent: torch.Tensor, noise: torch.Tensor, sigmas: torch.Tensor) -> torch.Tensor:
        """Forward diffusion: x_t = (1 - σ) * x_0 + σ * ε."""
        return (1.0 - sigmas) * latent + sigmas * noise

    def build_train_gt(self, latent: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """Flow-matching velocity target: v = ε - x_0."""
        return noise - latent

    def loss_weighting(self, sigmas: torch.Tensor) -> torch.Tensor:
        return compute_loss_weighting_for_sd3(weighting_scheme="none", sigmas=sigmas)

    # ------------------------------------------------------------------
    # Inference — diffusers scheduler interface
    # ------------------------------------------------------------------

    def set_timesteps(self, num_inference_steps: int, device=None) -> None:
        """Set up a subsampled sigma / timestep table for inference."""
        self.num_inference_steps = num_inference_steps
        device = device or self.device

        sigmas = torch.linspace(1.0, 1.0 / num_inference_steps, num_inference_steps)
        self.sigmas = torch.cat([sigmas, torch.zeros(1)]).to(device)  # shape: (N+1,)
        self.timesteps = (sigmas * self.num_train_timesteps).to(device)  # shape: (N,)

    def step(
        self,
        model_output: torch.Tensor,
        timestep: torch.Tensor | float,
        sample: torch.Tensor,
        return_dict: bool = True,
    ) -> SchedulerOutput | tuple:
        """Single Euler denoising step.

        Args:
            model_output: velocity prediction v = ε - x_0 from the model.
            timestep:     current timestep (scalar).
            sample:       current noisy sample x_t.

        Returns:
            SchedulerOutput with prev_sample = x_{t-1}.
        """
        step_index = (self.timesteps == timestep).nonzero()[0].item()
        sigma = self.sigmas[step_index]
        sigma_next = self.sigmas[step_index + 1]  # always exists; last entry is 0

        prev_sample = sample + (sigma_next - sigma) * model_output

        if not return_dict:
            return (prev_sample,)
        return SchedulerOutput(prev_sample=prev_sample)

    def scale_model_input(self, sample: torch.Tensor, timestep=None) -> torch.Tensor:
        """No-op: flow matching does not scale the model input."""
        return sample
