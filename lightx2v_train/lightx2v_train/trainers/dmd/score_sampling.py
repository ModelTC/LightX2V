"""Score-noise sampling policies shared by distribution-matching trainers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass

import torch

from lightx2v_train.schedulers.flow_matching import RectifiedFlowMatchingScheduler


@dataclass(frozen=True)
class ScoreSigmaContext:
    denoised_timestep_from: int | None
    denoised_timestep_to: int | None
    num_train_timesteps: int
    device: torch.device
    scheduler: RectifiedFlowMatchingScheduler
    latent_hw: tuple[int, int] | None = None
    num_steps: int | None = None


class ScoreSigmaSampler(ABC):
    @abstractmethod
    def sample(self, context: ScoreSigmaContext) -> torch.Tensor:
        """Sample one base sigma in noise-ward coordinates."""


@dataclass(frozen=True)
class DiscreteTimestepScoreSigmaSampler(ScoreSigmaSampler):
    """Sample a discrete timestep, apply scheduler shift, then clamp."""

    use_rollout_min: bool = False
    use_rollout_max: bool = False

    def sample(self, context: ScoreSigmaContext) -> torch.Tensor:
        lower = context.denoised_timestep_to if self.use_rollout_min and context.denoised_timestep_to is not None else 0
        upper = context.denoised_timestep_from if self.use_rollout_max and context.denoised_timestep_from is not None else context.num_train_timesteps
        lower = max(0, int(lower))
        upper = min(context.num_train_timesteps, int(upper))
        if upper <= lower:
            upper = min(context.num_train_timesteps, lower + 1)
        if upper <= lower:
            raise ValueError(f"No score timestep remains in [{lower}, {upper}) for num_train_timesteps={context.num_train_timesteps}.")

        timestep = torch.randint(lower, upper, (1,), device=context.device, dtype=torch.long).float()
        sigma = context.scheduler.time_shift(timestep / context.num_train_timesteps, latent_hw=context.latent_hw, num_steps=context.num_steps)
        return context.scheduler.clamp_training_sigma(sigma)


@dataclass(frozen=True)
class ContinuousUniformScoreSigmaSampler(ScoreSigmaSampler):
    """Sample continuous uniform noise, apply scheduler shift, then clamp."""

    def sample(self, context: ScoreSigmaContext) -> torch.Tensor:
        sigma = torch.rand((1,), device=context.device, dtype=torch.float32)
        sigma = context.scheduler.time_shift(sigma, latent_hw=context.latent_hw, num_steps=context.num_steps)
        return context.scheduler.clamp_training_sigma(sigma)


def build_score_sigma_sampler(
    config,
    *,
    use_rollout_min: bool,
    use_rollout_max: bool,
) -> ScoreSigmaSampler:
    """Build the sampling policy; the scheduler owns time shifting and bounds."""

    if config is None:
        config = {}
    if not isinstance(config, Mapping):
        raise ValueError("training.dmd.score_sampling must be a mapping.")

    kind = str(config.get("type", "discrete_timestep")).lower()
    if kind == "discrete_timestep":
        return DiscreteTimestepScoreSigmaSampler(
            use_rollout_min=bool(config.get("use_rollout_min", use_rollout_min)),
            use_rollout_max=bool(config.get("use_rollout_max", use_rollout_max)),
        )
    if kind == "continuous_uniform":
        return ContinuousUniformScoreSigmaSampler()
    raise ValueError(f"Unsupported training.dmd.score_sampling.type={kind!r}; expected 'discrete_timestep' or 'continuous_uniform'.")
