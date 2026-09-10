"""Flow-matching action sampler for OpenPI."""

from __future__ import annotations

import torch

from ..observation import Observation


class OpenPITransformerInfer:
    def __init__(self, num_steps: int = 10):
        if num_steps <= 0:
            raise ValueError("num_steps must be positive")
        self.num_steps = num_steps

    @torch.no_grad()
    def infer(
        self,
        model,
        observation: Observation,
        device: torch.device | str,
        noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return model.sample_actions(device, observation, noise=noise, num_steps=self.num_steps)
