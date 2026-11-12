"""Configuration for FastGen-style parallel decoding distillation."""

import math
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class PDDConfig:
    grid_size: int = 32
    num_inference_steps: int = 8
    video_shift: float = 12.0
    audio_shift: float = 3.0
    block_size_min: int = 4
    block_size_max: int = 8
    num_supervised_heads: int = 2
    teacher_solver: str = "euler"
    backward_simulation: str = "staggered"

    @classmethod
    def from_config(cls, config):
        values = dict(config["training"]["pdd"])
        values.pop("generation_shapes", None)
        result = cls(**values)
        for name in ("grid_size", "num_inference_steps", "block_size_min", "block_size_max", "num_supervised_heads"):
            if type(getattr(result, name)) is not int or getattr(result, name) < 1:
                raise ValueError(f"training.pdd.{name} must be a positive integer")
        if not result.num_supervised_heads <= result.block_size_min <= result.block_size_max <= result.grid_size:
            raise ValueError("PDD requires supervised_heads <= block_size_min <= block_size_max <= grid_size")
        # H3 recipes use exact equal-sized blocks; avoid silently changing their
        # start-node distribution when an arbitrary NFE is requested.
        if result.grid_size % result.block_size_min or result.grid_size % result.num_inference_steps:
            raise ValueError("PDD grid_size must divide into whole training and inference blocks")
        if not result.block_size_min <= result.grid_size // result.num_inference_steps <= result.block_size_max:
            raise ValueError("Requested PDD inference NFE is outside the trained block range")
        if any(not math.isfinite(value) or value <= 0 for value in (result.video_shift, result.audio_shift)):
            raise ValueError("PDD video/audio shifts must be finite and positive")
        if result.teacher_solver not in {"euler", "midpoint"}:
            raise ValueError("PDD teacher_solver must be euler or midpoint")
        if result.backward_simulation != "staggered":
            raise ValueError("This H3 PDD trainer implements backward_simulation=staggered")
        return result

    @property
    def rollout_steps(self):
        return self.grid_size // self.block_size_min

    def base_grid(self, device=None):
        sigma = torch.linspace(1, 0, self.grid_size + 1, dtype=torch.float64, device=device)
        return self.video_shift * sigma / (1 + (self.video_shift - 1) * sigma)
