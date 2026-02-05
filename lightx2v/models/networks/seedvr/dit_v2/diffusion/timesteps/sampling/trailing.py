

import torch

from ...types import SamplingDirection
from ..base import SamplingTimesteps


class UniformTrailingSamplingTimesteps(SamplingTimesteps):
    def __init__(
        self,
        T: int,
        steps: int,
        shift: float = 1.0,
        device: torch.device = "cpu",
    ):
        timesteps = torch.arange(1.0, 0.0, -1.0 / steps, device=device)
        timesteps = shift * timesteps / (1 + (shift - 1) * timesteps)
        if isinstance(T, float):
            timesteps = timesteps * T
        else:
            timesteps = timesteps.mul(T + 1).sub(1).round().int()
        super().__init__(T=T, timesteps=timesteps, direction=SamplingDirection.backward)
