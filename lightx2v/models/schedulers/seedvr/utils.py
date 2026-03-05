from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Tuple, Union

import torch
from tqdm import tqdm


class PredictionType(str, Enum):
    x_0 = "x_0"
    x_T = "x_T"
    v_cos = "v_cos"
    v_lerp = "v_lerp"


class SamplingDirection(str, Enum):
    backward = "backward"
    forward = "forward"


def expand_dims(tensor: torch.Tensor, ndim: int):
    shape = tensor.shape + (1,) * (ndim - tensor.ndim)
    return tensor.reshape(shape)


def assert_schedule_timesteps_compatible(schedule, timesteps):
    if schedule.T != timesteps.T:
        raise ValueError("Schedule and timesteps must have the same T.")
    if schedule.is_continuous() != timesteps.is_continuous():
        raise ValueError("Schedule and timesteps must have the same continuity.")


class Schedule(ABC):
    @property
    @abstractmethod
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
        a_t = self.A(t)
        b_t = self.B(t)

        if pred_type == PredictionType.x_T:
            pred_x_t = pred
            pred_x_0 = (x_t - b_t * pred_x_t) / a_t
        elif pred_type == PredictionType.x_0:
            pred_x_0 = pred
            pred_x_t = (x_t - a_t * pred_x_0) / b_t
        elif pred_type == PredictionType.v_cos:
            pred_x_0 = a_t * x_t - b_t * pred
            pred_x_t = a_t * pred + b_t * x_t
        elif pred_type == PredictionType.v_lerp:
            pred_x_0 = (x_t - b_t * pred) / (a_t + b_t)
            pred_x_t = (x_t + a_t * pred) / (a_t + b_t)
        else:
            raise NotImplementedError

        return pred_x_0, pred_x_t

    def convert_to_pred(self, x_0: torch.Tensor, x_T: torch.Tensor, t: torch.Tensor, pred_type: PredictionType) -> torch.Tensor:
        t = expand_dims(t, x_0.ndim)
        a_t = self.A(t)
        b_t = self.B(t)

        if pred_type == PredictionType.x_T:
            pred = x_T
        elif pred_type == PredictionType.x_0:
            pred = x_0
        elif pred_type == PredictionType.v_cos:
            pred = a_t * x_T - b_t * x_0
        elif pred_type == PredictionType.v_lerp:
            pred = x_T - x_0
        else:
            raise NotImplementedError

        return pred


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


class EulerSampler(Sampler):
    def sample(
        self,
        x: torch.Tensor,
        f: Callable[[SamplerModelArgs], torch.Tensor],
    ) -> torch.Tensor:
        timesteps = self.timesteps.timesteps
        progress = self.get_progress_bar()
        i = 0
        for t, s in zip(timesteps[:-1], timesteps[1:]):
            pred = f(SamplerModelArgs(x, t, i))
            x = self.step_to(pred, x, t, s)
            i += 1
            progress.update()

        if self.return_endpoint:
            t = timesteps[-1]
            pred = f(SamplerModelArgs(x, t, i))
            x = self.get_endpoint(pred, x, t)
            progress.update()
        return x

    def step(
        self,
        pred: torch.Tensor,
        x_t: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        return self.step_to(pred, x_t, t, self.get_next_timestep(t))

    def step_to(
        self,
        pred: torch.Tensor,
        x_t: torch.Tensor,
        t: torch.Tensor,
        s: torch.Tensor,
    ) -> torch.Tensor:
        t = expand_dims(t, x_t.ndim)
        s = expand_dims(s, x_t.ndim)
        T = self.schedule.T
        pred_x_0, pred_x_T = self.schedule.convert_from_pred(pred, self.prediction_type, x_t, t)
        pred_x_s = self.schedule.forward(pred_x_0, pred_x_T, s.clamp(0, T))
        pred_x_s = pred_x_s.where(s >= 0, pred_x_0)
        pred_x_s = pred_x_s.where(s <= T, pred_x_T)
        return pred_x_s


def _cfg_get(config, key, default=None):
    if isinstance(config, dict):
        return config.get(key, default)
    return getattr(config, key, default)


def create_schedule_from_config(
    config,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> Schedule:
    del device, dtype
    if _cfg_get(config, "type") == "lerp":
        return LinearInterpolationSchedule(T=_cfg_get(config, "T", 1.0))
    raise NotImplementedError


def create_sampler_from_config(
    config,
    schedule: Schedule,
    timesteps: SamplingTimesteps,
) -> Sampler:
    if _cfg_get(config, "type") == "euler":
        pred_type = _cfg_get(config, "prediction_type")
        if isinstance(pred_type, str):
            pred_type = PredictionType(pred_type)
        return EulerSampler(
            schedule=schedule,
            timesteps=timesteps,
            prediction_type=pred_type,
        )
    raise NotImplementedError


def create_sampling_timesteps_from_config(
    config,
    schedule: Schedule,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> SamplingTimesteps:
    del dtype
    if _cfg_get(config, "type") == "uniform_trailing":
        return UniformTrailingSamplingTimesteps(
            T=schedule.T,
            steps=_cfg_get(config, "steps"),
            shift=_cfg_get(config, "shift", 1.0),
            device=device,
        )
    raise NotImplementedError
