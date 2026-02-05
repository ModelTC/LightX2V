

from typing import Callable
import torch


def expand_dims(tensor: torch.Tensor, ndim: int):
    shape = tensor.shape + (1,) * (ndim - tensor.ndim)
    return tensor.reshape(shape)


def assert_schedule_timesteps_compatible(schedule, timesteps):
    if schedule.T != timesteps.T:
        raise ValueError("Schedule and timesteps must have the same T.")
    if schedule.is_continuous() != timesteps.is_continuous():
        raise ValueError("Schedule and timesteps must have the same continuity.")


def classifier_free_guidance(
    pos: torch.Tensor,
    neg: torch.Tensor,
    scale: float,
    rescale: float = 0.0,
):
    cfg = neg + scale * (pos - neg)
    if rescale != 0.0:
        pos_std = pos.std(dim=list(range(1, pos.ndim)), keepdim=True)
        cfg_std = cfg.std(dim=list(range(1, cfg.ndim)), keepdim=True)
        factor = pos_std / cfg_std
        factor = rescale * factor + (1 - rescale)
        cfg *= factor
    return cfg


def classifier_free_guidance_dispatcher(
    pos: Callable,
    neg: Callable,
    scale: float,
    rescale: float = 0.0,
):
    if scale == 1.0:
        return pos()
    return classifier_free_guidance(
        pos=pos(),
        neg=neg(),
        scale=scale,
        rescale=rescale,
    )
