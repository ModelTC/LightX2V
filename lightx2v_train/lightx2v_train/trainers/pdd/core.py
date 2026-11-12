# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""PDD integration and detached teacher regression, adapted from NVIDIA FastGen.

Reference: NVlabs/FastGen 898e757836dc244192533fe6ea797906799dd745,
methods/trajectory_matching/pdd.py and networks/utils/pdd_mixin.py.

All heads and teacher callbacks use native ``q = noise - clean = dx/dsigma``.
The descending base grid is the primary modality's *physical* sigma. A callable
sigma_schedule maps it to one sigma per modality; thus AV updates use each
modality's exact sigma increments without applying its shift a second time.
Tensor, dict-of-tensors and H3 JointLatent values share the same algorithms.
"""

from collections.abc import Mapping

import torch

FASTGEN_REVISION = "898e757836dc244192533fe6ea797906799dd745"
RK_STAGES = {
    "euler": ((), (1.0,), None),
    "midpoint": (((0.5, 0),), (0.0, 1.0), None),
    "heun": (((1.0, 0),), (0.5, 0.5), 0),
    "kutta4": (((0.5, 0), (0.5, 1), (1.0, 2)), (1 / 6, 2 / 6, 2 / 6, 1 / 6), 1),
}


def modalities(value):
    if torch.is_tensor(value):
        return {None: value}
    if isinstance(value, Mapping):
        return dict(value)
    return {"video": value.video, "audio": value.audio}


def restore(template, values):
    if torch.is_tensor(template):
        return values[None]
    if isinstance(template, Mapping):
        return dict(values)
    return type(template)(values["video"], values["audio"], template.shape)


def map_tensors(operation, value):
    return restore(value, {key: operation(tensor) for key, tensor in modalities(value).items()})


def _sigmas(time, schedule):
    return time if schedule is None else schedule(time)


def _for_modality(value, key):
    return value[key] if isinstance(value, Mapping) else value


def _expand(scale, value):
    return scale.reshape(*scale.shape, *((1,) * (value.ndim - scale.ndim)))


def _batch_indices(index, batch_size, device):
    if isinstance(index, int):
        return torch.full((batch_size,), index, dtype=torch.long, device=device)
    if index.dtype != torch.long or index.shape != (batch_size,) or index.device != device:
        raise ValueError("PDD indices must be int or device-local torch.long[batch]")
    return index


def block_weights(start_idx, end_idx, base_grid, *, sigma_schedule=None, allow_empty=False):
    """Return normalized head weights and exact endpoint sigma differences.

    Int endpoints produce [N] weights; [B] endpoints produce [B,N]; an end
    index of shape [B,K] produces [B,K,N]. Empty blocks have zero weight.
    """
    shared = isinstance(start_idx, int)
    if shared != isinstance(end_idx, int):
        raise TypeError("PDD block endpoints must both be integers or both be tensors")
    if shared:
        valid = 0 <= start_idx <= end_idx <= len(base_grid) - 1
        valid = valid and (allow_empty or start_idx < end_idx)
    else:
        if start_idx.dtype != torch.long or end_idx.dtype != torch.long or start_idx.ndim != 1:
            raise ValueError("PDD tensor indices must be long, with start shape [B]")
        if end_idx.ndim not in (1, 2) or end_idx.shape[0] != start_idx.shape[0]:
            raise ValueError("PDD endpoint indices must have shape [B] or [B,K]")
        if start_idx.device != base_grid.device or end_idx.device != base_grid.device:
            raise ValueError("PDD indices and base grid must share a device")
        if end_idx.ndim == 2:
            start_idx = start_idx[:, None]
        ordered = start_idx <= end_idx if allow_empty else start_idx < end_idx
        valid = bool(((start_idx >= 0) & ordered & (end_idx < len(base_grid))).all())
    if not valid:
        raise ValueError("PDD block requires 0 <= start < end <= grid_size (or equal with allow_empty)")
    head_indices = torch.arange(len(base_grid) - 1, device=base_grid.device)
    active = (head_indices >= start_idx) & (head_indices < end_idx) if shared else ((head_indices >= start_idx[..., None]) & (head_indices < end_idx[..., None]))
    grids = _sigmas(base_grid, sigma_schedule)
    starts = _sigmas(base_grid[start_idx], sigma_schedule)
    ends = _sigmas(base_grid[end_idx], sigma_schedule)

    def normalized(grid, start, end):
        delta = grid[1:] - grid[:-1]
        if not bool((torch.isfinite(delta) & (delta < 0)).all()):
            raise ValueError("PDD modality sigma grid must be finite and strictly decreasing")
        total = end - start
        divisor = torch.where(active.any(dim=-1), total, torch.ones_like(total))
        return torch.where(active, delta, torch.zeros_like(delta)) / divisor[..., None], total

    if isinstance(grids, Mapping):
        pairs = {key: normalized(grid, starts[key], ends[key]) for key, grid in grids.items()}
        return {key: pair[0] for key, pair in pairs.items()}, {key: pair[1] for key, pair in pairs.items()}
    return normalized(grids, starts, ends)


def advance_blocks(state, pred, start_idx, end_idx, base_grid, *, sigma_schedule=None, fused=False, allow_empty=False, precision_decoding=torch.float32):
    """Integrate materialized [B,N,...] heads or a fused [B,...] block mean."""
    weights, total = block_weights(start_idx, end_idx, base_grid, sigma_schedule=sigma_schedule, allow_empty=allow_empty)
    shared = isinstance(start_idx, int)
    multi = not shared and end_idx.ndim == 2
    if multi and fused:
        raise ValueError("Multi-endpoint advances require materialized heads")
    predictions = modalities(pred)
    outputs = {}
    for key, original in modalities(state).items():
        dtype = precision_decoding or original.dtype
        current, prediction = original.to(dtype), predictions[key].to(dtype)
        head_weights = _for_modality(weights, key).to(dtype)
        scale = _for_modality(total, key).to(dtype)
        with torch.autocast(device_type=current.device.type, enabled=False):
            if fused:
                mean = prediction
            else:
                if prediction.shape[:2] != (current.shape[0], len(base_grid) - 1):
                    raise ValueError("PDD predictions must have shape [B,grid_size,...]")
                if shared:
                    mean = torch.einsum("n,bn...->b...", head_weights[start_idx:end_idx], prediction[:, start_idx:end_idx])
                else:
                    mean = torch.einsum("bkn,bn...->bk..." if multi else "bn,bn...->b...", head_weights, prediction)
            if multi:
                current = current[:, None]
            outputs[key] = (current + mean * _expand(scale, mean)).to(original.dtype)
    return restore(state, outputs)


def sample_supervision_indices(start_idx, grid_size, block_size_max, num_supervised_heads=1, *, batch_size=1, device=None, generator=None, block_end_idx=None):
    """Sample distinct absolute heads in each block, as FastGen's multinomial."""
    device = torch.device(device) if device is not None else (start_idx.device if torch.is_tensor(start_idx) else torch.device("cpu"))
    starts = _batch_indices(start_idx, batch_size, device)
    end = (starts + block_size_max).clamp(max=grid_size) if block_end_idx is None else _batch_indices(block_end_idx, batch_size, device)
    if not bool(((starts >= 0) & (end > starts) & (end <= grid_size)).all()) or num_supervised_heads < 1:
        raise ValueError("PDD supervision requires valid nonempty blocks and a positive head count")
    indices = torch.arange(grid_size, device=device)[None]
    mask = (indices >= starts[:, None]) & (indices < end[:, None])
    if not bool((mask.sum(dim=1) >= num_supervised_heads).all()):
        raise ValueError("num_supervised_heads exceeds a sampled block's head count")
    return torch.multinomial(mask.float(), num_supervised_heads, replacement=False, generator=generator)


@torch.no_grad()
def teacher_target(state, step_idx, base_grid, predict_teacher, *, sigma_schedule=None, solver="euler"):
    """Return an interval's native q target; all RK stages use a common clock.

    The teacher callback takes (state, primary_sigma) and returns noise-clean.
    Heun/RK4 still run their terminal evaluation on all ranks, then replace its
    contribution on the clean endpoint exactly as FastGen does.
    """
    if solver not in RK_STAGES:
        raise ValueError(f"Unknown PDD teacher solver: {solver}")
    first = next(iter(modalities(state).values()))
    indices = _batch_indices(step_idx, first.shape[0], base_grid.device)
    if not bool(((indices >= 0) & (indices < len(base_grid) - 1)).all()):
        raise ValueError("PDD teacher head index lies outside the grid")
    start, end = base_grid[indices], base_grid[indices + 1]
    stages, coefficients, fallback = RK_STAGES[solver]
    velocities = [predict_teacher(state, start)]
    initial_sigmas = _sigmas(start, sigma_schedule)
    for fraction, source in stages:
        time = end if fraction == 1.0 else start + (end - start) * fraction
        sigmas = _sigmas(time, sigma_schedule)
        velocity = modalities(velocities[source])
        stage_values = {}
        for key, original in modalities(state).items():
            value = velocity[key]
            delta = (_for_modality(sigmas, key) - _for_modality(initial_sigmas, key)).to(value.dtype)
            stage_values[key] = original.to(value.dtype) + value * _expand(delta, value)
        velocities.append(predict_teacher(restore(state, stage_values), time))
    values = [modalities(value) for value in velocities]
    output = {key: sum(weight * value[key] for weight, value in zip(coefficients, values)) for key in values[0]}
    if fallback is not None:
        output = {key: torch.where(_expand(end <= base_grid[-1], value), values[fallback][key], value) for key, value in output.items()}
    return restore(state, output)


def pdd_loss_from_prediction(
    state,
    pred,
    start_idx,
    base_grid,
    predict_teacher,
    *,
    sigma_schedule=None,
    supervised_indices=None,
    block_size_max=None,
    num_supervised_heads=1,
    next_idx=None,
    solver="euler",
    loss_scale=None,
    precision_decoding=torch.float32,
    generator=None,
):
    """Detached intra-block rollout, frozen RK targets, and ordinary head MSE.

    Exactly one student backbone evaluation precedes this function. Each
    selected head invokes the teacher the same number of times on every rank,
    even when its per-sample loss_scale is zero. Returns (loss_map, next_state);
    next_state is a detached clone suitable for cross-iteration carry.
    """
    first = next(iter(modalities(state).values()))
    batch, device = first.shape[0], first.device
    starts = _batch_indices(start_idx, batch, device)
    if supervised_indices is None:
        if block_size_max is None:
            raise ValueError("Supply supervised_indices or block_size_max")
        supervised_indices = sample_supervision_indices(starts, len(base_grid) - 1, block_size_max, num_supervised_heads, batch_size=batch, device=device, generator=generator)
    if supervised_indices.ndim != 2 or supervised_indices.shape[0] != batch or supervised_indices.shape[1] < 1:
        raise ValueError("supervised_indices must have shape [B,K] with K > 0")
    if not bool((supervised_indices < len(base_grid) - 1).all()):
        raise ValueError("Supervision indices must name heads, not the terminal grid node")
    endpoints = supervised_indices
    if next_idx is not None:
        ends = _batch_indices(next_idx, batch, device)
        if not bool((ends > starts).all()):
            raise ValueError("Carry endpoint must follow the current block start")
        endpoints = torch.cat((endpoints, ends[:, None]), dim=1)
    with torch.no_grad():
        states = advance_blocks(state, pred, starts, endpoints, base_grid, sigma_schedule=sigma_schedule, allow_empty=True, precision_decoding=precision_decoding)
    predictions, advanced = modalities(pred), modalities(states)
    batch_idx = torch.arange(batch, device=device)
    losses = {key: [] for key in predictions}
    for column in range(supervised_indices.shape[1]):
        head = supervised_indices[:, column]
        target = teacher_target(restore(state, {key: value[:, column] for key, value in advanced.items()}), head, base_grid, predict_teacher, sigma_schedule=sigma_schedule, solver=solver)
        for key, value in modalities(target).items():
            prediction = predictions[key][batch_idx, head]
            if prediction.shape != value.shape or prediction.device != value.device:
                raise ValueError("PDD head and teacher target shapes/devices must match exactly")
            # FastGen's common_loss.mse_loss uses HIGH_PRECISION=float64.
            squared = (prediction.double() - value.double()).square()
            if loss_scale is None:
                loss = squared.mean()
            else:
                if loss_scale.shape != (batch,):
                    raise ValueError("loss_scale must have shape [batch]")
                loss = (squared.flatten(1).mean(dim=1) * loss_scale).mean()
            losses[key].append(loss)
    loss_map = {(f"{key}_pdd_loss" if key is not None else "pdd_loss"): torch.stack(values).mean() for key, values in losses.items()}
    loss_map["total_loss"] = sum(loss_map.values())
    next_state = None if next_idx is None else restore(state, {key: value[:, -1].clone() for key, value in advanced.items()})
    return loss_map, next_state
