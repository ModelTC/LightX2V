"""Few-step path construction adapted from FewStepRL 3dd8e422 (Apache-2.0)."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .config import FewStepConfig


@dataclass(frozen=True)
class TimestepPathBatch:
    """Normalized source and target paths, each shaped [batch, num_steps]."""

    sigmas: torch.Tensor
    next_sigmas: torch.Tensor
    train_step_index: int
    num_steps: int

    @property
    def timesteps(self) -> torch.Tensor:
        """Source times in the 0--1000 convention of the original Wan pipeline."""
        return self.sigmas * 1000.0

    @property
    def next_timesteps(self) -> torch.Tensor:
        return self.next_sigmas * 1000.0


def timestep_shift(sigmas: torch.Tensor, shift: float = 1.0) -> torch.Tensor:
    """Apply UniShift to normalized times in [0, 1]."""
    return shift * sigmas / (1 + (shift - 1) * sigmas)


def build_timestep_paths(
    config: FewStepConfig,
    batch_size: int,
    cycle_index: int,
    *,
    device: torch.device | str,
    generator: torch.Generator | None = None,
) -> TimestepPathBatch:
    """Select one nonterminal bridge; the other transitions use Euler.

    As in FewStepRL, disabling train-index cycling selects the penultimate
    transition, while path length cycles between min_steps and max_steps.
    Random paths preserve the original uniform sampling and clamp bounds.
    """
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer.")
    if isinstance(cycle_index, bool) or not isinstance(cycle_index, int) or cycle_index < 0:
        raise ValueError("cycle_index must be a nonnegative integer.")
    device = torch.device(device)
    num_steps = config.min_steps + cycle_index % (config.max_steps - config.min_steps + 1)
    train_step_index = cycle_index % (num_steps - 1) if config.cycle_train_idx else num_steps - 2
    if config.timestep_sampling_mode == "fixed_uniform":
        sources = torch.linspace(1.0, 1.0 / num_steps, num_steps, device=device, dtype=torch.float32)
        sources = sources.unsqueeze(0).expand(batch_size, -1).clone()
    else:
        rng_device = device if generator is None else generator.device
        inner = torch.rand((batch_size, num_steps - 1), generator=generator, device=rng_device, dtype=torch.float32)
        inner = inner.to(device).clamp_(1e-4, 1.0 - 1e-4)
        inner = torch.sort(inner, dim=1, descending=True).values
        sources = torch.cat([torch.ones((batch_size, 1), device=device), inner], dim=1)
    sigmas = timestep_shift(sources, config.timestep_shift).clamp_(0.0, 1.0)
    next_sigmas = torch.cat([sigmas[:, 1:], torch.zeros((batch_size, 1), device=device)], dim=1)
    return TimestepPathBatch(sigmas, next_sigmas, train_step_index, num_steps)
