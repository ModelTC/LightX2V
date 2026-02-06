import torch

from .samplers.base import Sampler
from .samplers.euler import EulerSampler
from .schedules.base import Schedule
from .schedules.lerp import LinearInterpolationSchedule
from .timesteps.base import SamplingTimesteps
from .timesteps.sampling.trailing import UniformTrailingSamplingTimesteps
from .types import PredictionType


def _cfg_get(config, key, default=None):
    if isinstance(config, dict):
        return config.get(key, default)
    return getattr(config, key, default)


def create_schedule_from_config(
    config,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> Schedule:
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
    if _cfg_get(config, "type") == "uniform_trailing":
        return UniformTrailingSamplingTimesteps(
            T=schedule.T,
            steps=_cfg_get(config, "steps"),
            shift=_cfg_get(config, "shift", 1.0),
            device=device,
        )
    raise NotImplementedError
