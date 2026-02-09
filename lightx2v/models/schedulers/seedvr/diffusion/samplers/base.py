from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable

import torch
from tqdm import tqdm

from ..schedules.base import Schedule
from ..timesteps.base import SamplingTimesteps
from ..types import PredictionType, SamplingDirection
from ..utils import assert_schedule_timesteps_compatible


@dataclass
class SamplerModelArgs:
    x_t: torch.Tensor
    t: torch.Tensor
    i: int


class Sampler(ABC):
    def __init__(
        self,
        schedule: Schedule,
        timesteps: SamplingTimesteps,
        prediction_type: PredictionType,
        return_endpoint: bool = True,
    ):
        assert_schedule_timesteps_compatible(
            schedule=schedule,
            timesteps=timesteps,
        )
        self.schedule = schedule
        self.timesteps = timesteps
        self.prediction_type = prediction_type
        self.return_endpoint = return_endpoint

    @abstractmethod
    def sample(
        self,
        x: torch.Tensor,
        f: Callable[[SamplerModelArgs], torch.Tensor],
    ) -> torch.Tensor: ...

    def get_next_timestep(
        self,
        t: torch.Tensor,
    ) -> torch.Tensor:
        T = self.timesteps.T
        steps = len(self.timesteps)
        curr_idx = self.timesteps.index(t)
        next_idx = curr_idx + 1
        bound = -1 if self.timesteps.direction == SamplingDirection.backward else T + 1

        s = self.timesteps[next_idx.clamp_max(steps - 1)]
        s = s.where(next_idx < steps, bound)
        return s

    def get_endpoint(
        self,
        pred: torch.Tensor,
        x_t: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        x_0, x_T = self.schedule.convert_from_pred(pred, self.prediction_type, x_t, t)
        return x_0 if self.timesteps.direction == SamplingDirection.backward else x_T

    def get_progress_bar(self):
        return tqdm(
            iterable=range(len(self.timesteps) - (0 if self.return_endpoint else 1)),
            dynamic_ncols=True,
            desc=self.__class__.__name__,
        )
