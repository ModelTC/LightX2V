"""Score-noise sampling policies shared by distribution-matching trainers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class ScoreSigmaContext:
    denoised_timestep_from: int | None
    denoised_timestep_to: int | None
    num_train_timesteps: int
    device: torch.device


class ScoreSigmaSampler(ABC):
    @abstractmethod
    def sample(self, context: ScoreSigmaContext) -> torch.Tensor:
        """Sample one base sigma in noise-ward coordinates."""


@dataclass(frozen=True)
class DiscreteTimestepScoreSigmaSampler(ScoreSigmaSampler):
    """Sample a discrete training timestep, preserving the legacy DMD policy."""

    sample_min_timestep: int
    clamp_min_timestep: int
    clamp_max_timestep: int
    timestep_shift: float = 1.0
    use_rollout_min: bool = False
    use_rollout_max: bool = False

    def __post_init__(self) -> None:
        if self.sample_min_timestep < 0:
            raise ValueError(f"Minimum score sampling timestep cannot be negative, got {self.sample_min_timestep}.")
        if self.clamp_min_timestep < 0 or self.clamp_max_timestep <= self.clamp_min_timestep:
            raise ValueError(f"Score timestep clamp bounds must satisfy 0 <= min < max, got [{self.clamp_min_timestep}, {self.clamp_max_timestep}].")
        if self.timestep_shift <= 0:
            raise ValueError(f"Score timestep shift must be positive, got {self.timestep_shift}.")

    def sample(self, context: ScoreSigmaContext) -> torch.Tensor:
        lower = context.denoised_timestep_to if self.use_rollout_min and context.denoised_timestep_to is not None else self.sample_min_timestep
        upper = context.denoised_timestep_from if self.use_rollout_max and context.denoised_timestep_from is not None else context.num_train_timesteps
        lower = max(0, int(lower))
        upper = min(context.num_train_timesteps, int(upper))
        if upper <= lower:
            upper = min(context.num_train_timesteps, lower + 1)
        if upper <= lower:
            raise ValueError(f"No score timestep remains in [{lower}, {upper}) for num_train_timesteps={context.num_train_timesteps}.")

        timestep = torch.randint(lower, upper, (1,), device=context.device, dtype=torch.long).float()
        if self.timestep_shift > 1.0:
            normalized = timestep / context.num_train_timesteps
            timestep = self.timestep_shift * normalized / (1.0 + (self.timestep_shift - 1.0) * normalized) * context.num_train_timesteps
        return timestep.clamp(self.clamp_min_timestep, self.clamp_max_timestep) / context.num_train_timesteps


@dataclass(frozen=True)
class ContinuousUniformScoreSigmaSampler(ScoreSigmaSampler):
    """Sample an unshifted continuous sigma from a uniform distribution."""

    min_sigma: float = 0.02
    max_sigma: float = 0.98

    def __post_init__(self) -> None:
        if not 0.0 <= self.min_sigma < self.max_sigma <= 1.0:
            raise ValueError(f"Score sigma bounds must satisfy 0 <= min < max <= 1, got [{self.min_sigma}, {self.max_sigma}].")

    def sample(self, context: ScoreSigmaContext) -> torch.Tensor:
        return torch.empty((1,), device=context.device, dtype=torch.float32).uniform_(self.min_sigma, self.max_sigma)


def build_score_sigma_sampler(
    config,
    *,
    sample_min_timestep: int,
    clamp_min_timestep: int,
    clamp_max_timestep: int,
    timestep_shift: float,
    use_rollout_min: bool,
    use_rollout_max: bool,
) -> ScoreSigmaSampler:
    """Build a policy, defaulting to the pre-existing discrete DMD behavior."""

    if config is None:
        config = {}
    if not isinstance(config, Mapping):
        raise ValueError("training.dmd.score_sampling must be a mapping.")

    kind = str(config.get("type", "discrete_timestep")).lower()
    if kind == "discrete_timestep":
        return DiscreteTimestepScoreSigmaSampler(
            sample_min_timestep=int(config.get("sample_min_timestep", sample_min_timestep)),
            clamp_min_timestep=int(config.get("clamp_min_timestep", clamp_min_timestep)),
            clamp_max_timestep=int(config.get("clamp_max_timestep", clamp_max_timestep)),
            timestep_shift=float(config.get("timestep_shift", timestep_shift)),
            use_rollout_min=bool(config.get("use_rollout_min", use_rollout_min)),
            use_rollout_max=bool(config.get("use_rollout_max", use_rollout_max)),
        )
    if kind == "continuous_uniform":
        return ContinuousUniformScoreSigmaSampler(
            min_sigma=float(config.get("min_sigma", 0.02)),
            max_sigma=float(config.get("max_sigma", 0.98)),
        )
    raise ValueError(f"Unsupported training.dmd.score_sampling.type={kind!r}; expected 'discrete_timestep' or 'continuous_uniform'.")
