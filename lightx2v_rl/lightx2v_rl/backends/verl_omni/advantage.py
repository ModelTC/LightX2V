"""Adapt streaming reward details to FewStepRL's per-reward advantages."""

from collections.abc import Mapping

import numpy as np
import torch

from lightx2v_rl.algorithms.fewstep.advantages import compute_weighted_advantages
from lightx2v_rl.training.advantage import validate_advantage_config


def _group_ids(data, size):
    values = data.non_tensor_batch.get("uid")
    if values is None or len(values) != size:
        raise ValueError("Reward advantages require one prompt uid per sample")
    result = []
    for value in values:
        if hasattr(value, "data") and not isinstance(value, str):
            value = value.data
        if value is None or not str(value):
            raise ValueError("Reward advantages require nonempty prompt uids")
        result.append(str(value))
    return result


def _component(data, name, rows, size, device):
    # Native reward managers expose flat columns; streaming TransferQueue keeps
    # one reward_extra_info mapping per sample. Prefer the flat manager contract.
    flat = data.non_tensor_batch.get(name)
    if flat is not None:
        if len(flat) != size:
            raise ValueError(f"Reward component {name!r} is not aligned with the sample batch")
        values = [flat[index] for index in rows]
    else:
        nested = data.non_tensor_batch.get("reward_extra_info")
        if nested is None or len(nested) != size:
            raise ValueError(f"Missing active reward component {name!r}; check reward_advantage.weights and scorer outputs")
        values = []
        for index in rows:
            detail = nested[index]
            if not isinstance(detail, Mapping) or name not in detail or detail[name] is None:
                raise ValueError(f"Missing active reward component {name!r} at sample {index}")
            values.append(detail[name])
    try:
        # Reward heads emit scalar scores, independent of the number of replay
        # timesteps. float64 retains the original FewStepRL statistic precision.
        if any(isinstance(value, (bool, np.bool_)) for value in values):
            raise ValueError("boolean values are not reward scores")
        array = np.asarray(values)
        if array.shape != (len(rows),) or array.dtype.kind not in "fiu":
            raise ValueError("expected one numeric scalar per sample")
        return torch.as_tensor(array, dtype=torch.float64, device=device)
    except (TypeError, ValueError, RuntimeError) as exc:
        raise ValueError(f"Invalid reward component {name!r}: {exc}") from exc


def compute_advantages(*, data, config):
    """Return advantages/returns while retaining the original logged rewards.

    Disabled mode delegates directly to the original Flow-GRPO implementation.
    Enabled mode excludes copied padding rows from statistics, combines the
    independently normalized components, and broadcasts the result over replay
    timesteps. Loss-side advantage clipping remains the sole clipping step.
    """
    weights = validate_advantage_config(config, require_weights=True)
    rewards = data.batch["sample_level_rewards"]
    if rewards.ndim != 2 or not rewards.is_floating_point() or not rewards.numel():
        raise ValueError("sample_level_rewards must be a nonempty floating [batch, timesteps] tensor")
    size = rewards.shape[0]
    groups = _group_ids(data, size)
    if not config.get("weight_advantages", True):
        from verl_omni.trainer.diffusion.diffusion_algos import compute_flow_grpo_outcome_advantage

        return compute_flow_grpo_outcome_advantage(
            sample_level_rewards=rewards,
            index=np.asarray(groups, dtype=object),
            norm_adv_by_std_in_grpo=config.get("norm_adv_by_std_in_grpo", True),
            global_std=config.get("global_std", True),
            config=config,
        )

    is_pad = np.asarray(data.non_tensor_batch.get("_balance_is_pad", np.zeros(size, dtype=bool)))
    if is_pad.shape != (size,) or is_pad.dtype.kind != "b":
        raise ValueError("_balance_is_pad must be one boolean per sample")
    rows = np.flatnonzero(~is_pad).tolist()
    if not rows:
        raise ValueError("Cannot compute reward advantages from an all-padding batch")
    components = {name: _component(data, name, rows, size, rewards.device) for name in weights}
    options = config.get("reward_advantage", {})
    normalization = options.get("normalization", "max_group_std")
    with torch.no_grad():
        combined = compute_weighted_advantages(
            components,
            [groups[index] for index in rows],
            weights,
            global_std=normalization == "global",
            max_group_std=normalization == "max_group_std",
            norm_by_std=config.get("norm_adv_by_std_in_grpo", True),
            epsilon=options.get("epsilon", 1e-4),
        )
        advantages = torch.zeros_like(rewards)
        advantages[rows] = combined.to(dtype=rewards.dtype)[:, None]
    if not torch.isfinite(advantages).all():
        raise ValueError("Combined reward advantages are nonfinite in the training dtype")
    return advantages, advantages
