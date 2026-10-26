"""Configuration for the FewStepRL Wan algorithms, independent of any trainer.

Migrated from FewStepRL commit 3dd8e422 (Apache-2.0). Geo-FSRL (``geo``)
is the name used here for its IMMB transition and matched objective.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class FewStepConfig:
    """Algorithm parameters only; rewards and execution belong to the host trainer."""

    bridge: str = "geo"
    min_steps: int = 4
    max_steps: int = 4
    cycle_train_idx: bool = True
    timestep_sampling_mode: str = "fixed_uniform"
    timestep_shift: float = 5.0
    noise_level: float = 0.3
    diffusion_clip: bool = True
    diffusion_clip_value: float = 0.45
    loss_reweighting: str = "matched"
    geo_reweight_multiplier: float = 0.5
    difficulty_reweighting: str = "constant"
    clip_range: float = 0.001
    adv_clip_max: float = 5.0
    kl_beta: float = 0.0003

    def __post_init__(self) -> None:
        choices = {
            "bridge": {"geo", "cps", "flow_grpo"},
            "timestep_sampling_mode": {"fixed_uniform", "random_uniform"},
            "loss_reweighting": {"matched", "constant", "none"},
            "difficulty_reweighting": {"constant", "linear_t2_minus_s2"},
        }
        for name, allowed in choices.items():
            value = getattr(self, name)
            if not isinstance(value, str) or value not in allowed:
                raise ValueError(f"{name} must be one of {sorted(allowed)}, got {value!r}.")
        for name in ("min_steps", "max_steps"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 2:
                raise ValueError(f"{name} must be an integer of at least 2.")
        if self.max_steps < self.min_steps:
            raise ValueError("max_steps must be greater than or equal to min_steps.")
        for name in ("cycle_train_idx", "diffusion_clip"):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be a boolean.")
        if self.cycle_train_idx and self.min_steps != self.max_steps:
            raise ValueError("cycle_train_idx requires min_steps == max_steps.")
        for name in (
            "timestep_shift",
            "noise_level",
            "diffusion_clip_value",
            "geo_reweight_multiplier",
            "adv_clip_max",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be a finite positive number.")
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be a finite positive number.")
        if self.bridge == "cps" and self.noise_level > 1:
            raise ValueError("CPS noise_level must be in (0, 1].")
        for name in ("clip_range", "kl_beta"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be a finite nonnegative number.")
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be a finite nonnegative number.")
        if self.clip_range >= 1:
            raise ValueError("clip_range must be less than 1.")
