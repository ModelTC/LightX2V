"""FewStepRL's per-reward GRPO advantages, followed by a weighted sum.

Adapted from FewStepRL ``advantages.py`` and ``stat_tracking.py`` (Apache-2.0).
Statistics use only the supplied batch, with population standard deviations
and float64 accumulation matching the original NumPy implementation.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from numbers import Real

import torch


@torch.no_grad()
def compute_weighted_advantages(
    rewards: Mapping[str, torch.Tensor],
    group_ids: Sequence,
    weights: Mapping[str, float],
    *,
    global_std: bool = False,
    max_group_std: bool = True,
    norm_by_std: bool = True,
    epsilon: float = 1e-4,
) -> torch.Tensor:
    """Normalize each active reward within prompt groups, then weight and sum.

    Active rewards must share shape ``[B]`` or ``[B, T]`` and device. Means
    always use each prompt's rows. The denominator is either the per-column
    full-batch std, the maximum over all groups and columns, or each group's
    per-column std. Both std selectors cannot be enabled simultaneously.

    Zero-weight rewards need not exist. Nothing is clipped or renormalized
    after summation. The output preserves shape/device, with float64 dtype if
    any active input is float64 and float32 otherwise; gradients are detached.
    Padding and missing active scores must be removed or resolved by callers.
    """
    if not isinstance(rewards, Mapping) or not isinstance(weights, Mapping):
        raise ValueError("rewards and weights must be mappings")
    if any(not isinstance(value, bool) for value in (global_std, max_group_std, norm_by_std)):
        raise ValueError("Advantage normalization switches must be booleans")
    if global_std and max_group_std:
        raise ValueError("global_std and max_group_std cannot both be enabled")
    if isinstance(epsilon, bool) or not isinstance(epsilon, Real) or not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("epsilon must be positive and finite")

    active = {}
    for name, weight in weights.items():
        if not isinstance(name, str) or not name:
            raise ValueError("Reward names must be nonempty strings")
        if isinstance(weight, bool) or not isinstance(weight, Real) or not math.isfinite(weight) or weight < 0:
            raise ValueError(f"Reward weight {name!r} must be finite and nonnegative")
        if weight > 0:
            active[name] = float(weight)
    if not active:
        raise ValueError("At least one reward weight must be positive")

    shape, device = None, None
    output_dtype = torch.float32
    for name in active:
        if name not in rewards:
            raise ValueError(f"Missing active reward {name!r}")
        value = rewards[name]
        if not isinstance(value, torch.Tensor) or value.layout != torch.strided:
            raise ValueError(f"Reward {name!r} must be a dense tensor")
        if value.ndim not in (1, 2) or value.numel() == 0:
            raise ValueError(f"Reward {name!r} must have nonempty shape [B] or [B, T]")
        if value.is_complex() or value.dtype == torch.bool:
            raise ValueError(f"Reward {name!r} must contain real numeric scores")
        if shape is None:
            shape, device = value.shape, value.device
        elif value.shape != shape or value.device != device:
            raise ValueError("Active rewards must share the same shape and device")
        if not torch.isfinite(value).all():
            raise ValueError(f"Reward {name!r} contains nonfinite scores")
        if value.dtype == torch.float64:
            output_dtype = torch.float64

    if isinstance(group_ids, (str, bytes)):
        raise ValueError("group_ids must contain one hashable ID per sample")
    try:
        ids = list(group_ids)
    except TypeError as exc:
        raise ValueError("group_ids must contain one hashable ID per sample") from exc
    if len(ids) != shape[0]:
        raise ValueError("group_ids length must equal the reward batch size")
    grouped = {}
    for index, group in enumerate(ids):
        if isinstance(group, Real) and not math.isfinite(group):
            raise ValueError("group_ids must not contain nonfinite IDs")
        try:
            grouped.setdefault(group, []).append(index)
        except TypeError as exc:
            raise ValueError("group_ids must contain hashable IDs") from exc
    indices = [torch.tensor(rows, dtype=torch.long, device=device) for rows in grouped.values()]

    result = torch.zeros(shape, dtype=torch.float64, device=device)
    for name, weight in active.items():
        values = rewards[name].to(dtype=torch.float64)
        groups = [values.index_select(0, rows) for rows in indices]
        stds = [group.std(dim=0, correction=0, keepdim=True) for group in groups] if norm_by_std else []
        if norm_by_std and global_std:
            shared_std = values.std(dim=0, correction=0, keepdim=True)
        elif norm_by_std and max_group_std:
            shared_std = torch.stack([std.max() for std in stds]).max()
        else:
            shared_std = None
        for position, (rows, group) in enumerate(zip(indices, groups, strict=True)):
            centered = group - group.mean(dim=0, keepdim=True)
            if norm_by_std:
                std = shared_std if shared_std is not None else stds[position]
                centered = centered / (std + epsilon)
            result.index_add_(0, rows, centered * weight)
    result = result.to(dtype=output_dtype)
    if not torch.isfinite(result).all():
        raise ValueError("Weighted advantages are nonfinite; check reward magnitudes and weights")
    return result
