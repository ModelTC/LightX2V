"""FewStepRL's single stochastic transition with a deterministic Euler suffix.

The velocity callback uses normalized sigma. Both rollout and actor replay use
the bridge in algorithms.fewstep, including its non-batch mean log probability.
Adapted from FewStepRL 3dd8e422 (Apache-2.0).
"""

from collections.abc import Callable
from dataclasses import dataclass

import torch

from lightx2v_rl.algorithms.fewstep import FewStepConfig, bridge_step, build_timestep_paths, euler_step


@dataclass(frozen=True)
class FewStepTrajectory:
    final_latents: torch.Tensor
    all_latents: torch.Tensor
    all_timesteps: torch.Tensor
    all_next_timesteps: torch.Tensor
    log_probs: torch.Tensor
    selected_step: int | None

    def training_tensors(self) -> dict[str, torch.Tensor]:
        """Return batch-first tensors consumed by the external training adapter."""
        if self.selected_step is None:
            raise ValueError("Deterministic evaluation does not produce a trainable stochastic transition.")
        return {
            "all_latents": self.all_latents,
            "all_timesteps": self.all_timesteps,
            "all_next_timesteps": self.all_next_timesteps,
            "rollout_log_probs": self.log_probs,
        }


@torch.no_grad()
def sample_trajectory(
    predict_velocity: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    latents: torch.Tensor,
    config: FewStepConfig,
    *,
    cycle_index: int = 0,
    generator: torch.Generator | None = None,
    deterministic: bool = False,
    paths=None,
) -> FewStepTrajectory:
    """Sample one video batch, retaining only the transition selected for RL.

    ``latents`` is [B, ...], and the callback receives [B] sigmas in [0, 1].
    Stored timesteps use the Wan convention t=1000*sigma. Evaluation follows
    the same time path with Euler updates at every step and returns no logprob.
    Supplied paths allow deterministic comparisons across model backends.
    """
    if latents.ndim < 2 or latents.shape[0] == 0:
        raise ValueError("latents must have a nonempty batch dimension and at least one feature dimension")
    if paths is None:
        paths = build_timestep_paths(config, latents.shape[0], cycle_index, device=latents.device, generator=generator)
    expected = (latents.shape[0], paths.num_steps)
    if paths.sigmas.shape != expected or paths.next_sigmas.shape != expected:
        raise ValueError(f"Timestep paths must have shape {expected}")
    if not 0 <= paths.train_step_index < paths.num_steps - 1:
        raise ValueError("The selected stochastic transition must precede the final transition to sigma=0")
    if paths.sigmas.device != latents.device or paths.next_sigmas.device != latents.device:
        raise ValueError("Timestep paths and latents must be on the same device")
    if not torch.isfinite(paths.sigmas).all() or not torch.isfinite(paths.next_sigmas).all():
        raise ValueError("Timestep paths must contain finite values")
    if not torch.all(paths.sigmas[:, 0] == 1) or not torch.all(paths.next_sigmas[:, -1] == 0):
        raise ValueError("Timestep paths must start at sigma=1 and end at sigma=0")
    if not torch.equal(paths.next_sigmas[:, :-1], paths.sigmas[:, 1:]):
        raise ValueError("Timestep paths must be continuous between consecutive transitions")
    if not torch.all((paths.sigmas > paths.next_sigmas) & (paths.next_sigmas >= 0)):
        raise ValueError("Timestep paths must be strictly decreasing")

    state = latents.float()
    recorded = None
    for index in range(paths.num_steps):
        sigma = paths.sigmas[:, index]
        next_sigma = paths.next_sigmas[:, index]
        velocity = predict_velocity(state, sigma)
        if velocity.shape != state.shape:
            raise ValueError("The velocity callback must return the same shape as the latent input")
        if not deterministic and index == paths.train_step_index:
            transition = bridge_step(velocity, state, sigma, next_sigma, config=config, generator=generator)
            recorded = (
                torch.stack((state, transition.next_sample), dim=1),
                (sigma * 1000.0).unsqueeze(1),
                (next_sigma * 1000.0).unsqueeze(1),
                transition.log_prob.unsqueeze(1),
            )
            state = transition.next_sample
        else:
            state = euler_step(velocity, state, sigma, next_sigma)

    if recorded is None:
        empty = state.new_empty((state.shape[0], 0))
        return FewStepTrajectory(state, state.new_empty((state.shape[0], 0, *state.shape[1:])), empty, empty, empty, None)
    return FewStepTrajectory(state, *recorded, paths.train_step_index)
