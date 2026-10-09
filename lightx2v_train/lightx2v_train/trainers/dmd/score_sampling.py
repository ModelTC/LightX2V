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
        """Sample noise levels understood by the model's capability.

        Normally this is one base sigma; legacy H3 uses a physical AV pair.
        """


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

    discrete_samples: int = 0

    def sample(self, context: ScoreSigmaContext) -> torch.Tensor:
        sigma = torch.rand((1,), device=context.device, dtype=torch.float32)
        if self.discrete_samples:
            sigma = torch.ceil(sigma * self.discrete_samples) / self.discrete_samples
        sigma = context.scheduler.time_shift(sigma, latent_hw=context.latent_hw, num_steps=context.num_steps)
        return context.scheduler.clamp_training_sigma(sigma)


@dataclass(frozen=True)
class H3ShiftedUniformScoreSigmaSampler(ScoreSigmaSampler):
    """Return base sigma, or the old trainer's exact physical pair in legacy mode.

    The pair is consumed only by H3's legacy-numerics capability. Keeping it
    intact avoids inverse-shift/forward-shift roundoff in both modalities.
    """

    video_flow_shift: float = 6.0
    discrete_samples: int = 1000
    min_sigma: float = 0.02
    max_sigma: float = 1.0
    legacy_numerics: bool = False
    audio_flow_shift: float = 3.0

    def sample(self, context: ScoreSigmaContext) -> torch.Tensor:
        base = torch.rand(() if self.legacy_numerics else (1,), device=context.device, dtype=torch.float32)
        if self.discrete_samples:
            base = torch.ceil(base * self.discrete_samples) / self.discrete_samples
        shift = self.video_flow_shift
        video_sigma = (shift * base / (1.0 + (shift - 1.0) * base)).clamp(
            self.min_sigma,
            self.max_sigma,
        )
        if self.legacy_numerics:
            audio_sigma = self.audio_flow_shift * video_sigma / (shift + (self.audio_flow_shift - shift) * video_sigma)
            return torch.stack((video_sigma, audio_sigma))
        return video_sigma / (shift - (shift - 1.0) * video_sigma)


def build_score_sigma_sampler(
    config,
    *,
    use_rollout_min: bool,
    use_rollout_max: bool,
) -> ScoreSigmaSampler:
    """Build the configured score-noise sampling policy."""

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
        discrete_samples = int(config.get("discrete_samples", 0))
        if discrete_samples < 0:
            raise ValueError("score_sampling.discrete_samples must be non-negative.")
        return ContinuousUniformScoreSigmaSampler(discrete_samples=discrete_samples)
    if kind == "h3_shifted_uniform":
        sampler = H3ShiftedUniformScoreSigmaSampler(
            video_flow_shift=float(config.get("video_flow_shift", 6.0)),
            discrete_samples=int(config.get("discrete_samples", 1000)),
            min_sigma=float(config.get("min_sigma", 0.02)),
            max_sigma=float(config.get("max_sigma", 1.0)),
            legacy_numerics=bool(config.get("legacy_numerics", False)),
            audio_flow_shift=float(config.get("audio_flow_shift", 3.0)),
        )
        if sampler.video_flow_shift <= 0 or sampler.audio_flow_shift <= 0 or sampler.discrete_samples < 0:
            raise ValueError("H3 score sampling requires positive video/audio flow shifts and non-negative discrete_samples.")
        if not 0 <= sampler.min_sigma < sampler.max_sigma <= 1:
            raise ValueError("H3 score sampling requires 0 <= min_sigma < max_sigma <= 1.")
        return sampler
    raise ValueError(f"Unsupported training.dmd.score_sampling.type={kind!r}; expected 'discrete_timestep', 'continuous_uniform', or 'h3_shifted_uniform'.")
