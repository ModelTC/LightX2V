

from abc import ABC, abstractmethod
from typing import Sequence, Union
import torch

from ..types import SamplingDirection


class Timesteps(ABC):
    def __init__(self, T: Union[int, float]):
        assert T > 0
        self._T = T

    @property
    def T(self) -> Union[int, float]:
        return self._T

    def is_continuous(self) -> bool:
        return isinstance(self.T, float)


class SamplingTimesteps(Timesteps):
    def __init__(
        self,
        T: Union[int, float],
        timesteps: torch.Tensor,
        direction: SamplingDirection,
    ):
        assert timesteps.ndim == 1
        super().__init__(T)
        self.timesteps = timesteps
        self.direction = direction

    def __len__(self) -> int:
        return len(self.timesteps)

    def __getitem__(self, idx: Union[int, torch.IntTensor]) -> torch.Tensor:
        return self.timesteps[idx]

    def index(self, t: torch.Tensor) -> torch.Tensor:
        i, j = t.reshape(-1, 1).eq(self.timesteps).nonzero(as_tuple=True)
        idx = torch.full_like(t, fill_value=-1, dtype=torch.int)
        idx.view(-1)[i] = j.int()
        return idx
