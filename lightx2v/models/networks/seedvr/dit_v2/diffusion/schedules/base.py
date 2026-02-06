from abc import ABC, abstractmethod, abstractproperty
from typing import Tuple, Union

import torch

from ..types import PredictionType
from ..utils import expand_dims


class Schedule(ABC):
    @abstractproperty
    def T(self) -> Union[int, float]: ...

    @abstractmethod
    def A(self, t: torch.Tensor) -> torch.Tensor: ...

    @abstractmethod
    def B(self, t: torch.Tensor) -> torch.Tensor: ...

    def snr(self, t: torch.Tensor) -> torch.Tensor:
        return (self.A(t) ** 2) / (self.B(t) ** 2)

    def isnr(self, snr: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def is_continuous(self) -> bool:
        return isinstance(self.T, float)

    def forward(self, x_0: torch.Tensor, x_T: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t = expand_dims(t, x_0.ndim)
        return self.A(t) * x_0 + self.B(t) * x_T

    def convert_from_pred(self, pred: torch.Tensor, pred_type: PredictionType, x_t: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        t = expand_dims(t, x_t.ndim)
        A_t = self.A(t)
        B_t = self.B(t)

        if pred_type == PredictionType.x_T:
            pred_x_T = pred
            pred_x_0 = (x_t - B_t * pred_x_T) / A_t
        elif pred_type == PredictionType.x_0:
            pred_x_0 = pred
            pred_x_T = (x_t - A_t * pred_x_0) / B_t
        elif pred_type == PredictionType.v_cos:
            pred_x_0 = A_t * x_t - B_t * pred
            pred_x_T = A_t * pred + B_t * x_t
        elif pred_type == PredictionType.v_lerp:
            pred_x_0 = (x_t - B_t * pred) / (A_t + B_t)
            pred_x_T = (x_t + A_t * pred) / (A_t + B_t)
        else:
            raise NotImplementedError

        return pred_x_0, pred_x_T

    def convert_to_pred(self, x_0: torch.Tensor, x_T: torch.Tensor, t: torch.Tensor, pred_type: PredictionType) -> torch.FloatTensor:
        t = expand_dims(t, x_0.ndim)
        A_t = self.A(t)
        B_t = self.B(t)

        if pred_type == PredictionType.x_T:
            pred = x_T
        elif pred_type == PredictionType.x_0:
            pred = x_0
        elif pred_type == PredictionType.v_cos:
            pred = A_t * x_T - B_t * x_0
        elif pred_type == PredictionType.v_lerp:
            pred = x_T - x_0
        else:
            raise NotImplementedError

        return pred
