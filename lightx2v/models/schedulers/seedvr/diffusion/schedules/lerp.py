from typing import Union

import torch

from .base import Schedule


class LinearInterpolationSchedule(Schedule):
    def __init__(self, T: Union[int, float] = 1.0):
        self._T = T

    @property
    def T(self) -> Union[int, float]:
        return self._T

    def A(self, t: torch.Tensor) -> torch.Tensor:
        return 1 - (t / self.T)

    def B(self, t: torch.Tensor) -> torch.Tensor:
        return t / self.T
