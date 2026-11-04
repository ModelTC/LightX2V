"""Adapt streaming reward details to FewStepRL's per-reward advantages."""

from collections.abc import Mapping
from numbers import Real

import numpy as np
import torch

from lightx2v_rl.algorithms.fewstep.advantages import compute_weighted_advantages
from lightx2v_rl.training.advantage import reward_weights_for_task, validate_advantage_config


def _group_ids(data, size, rows=None):
    values = data.non_tensor_batch.get("uid")
    if values is None or len(values) != size:
        raise ValueError("Reward advantages require one prompt uid per sample")
    result = [None] * size
    for index in range(size) if rows is None else rows:
        value = values[index]
        if hasattr(value, "data") and not isinstance(value, str):
            value = value.data
        if value is None or not str(value):
            raise ValueError("Reward advantages require nonempty prompt uids")
        result[index] = str(value)
    return result


def _task_ids(data, rows, size, task_weights):
    """Resolve actual tasks from dataset metadata or scorer output, never parity."""
    columns = {}
    for key in ("task_type", "data_source", "task_is_i2i", "extra_info", "reward_extra_info"):
        column = data.non_tensor_batch.get(key)
        if column is not None:
            if len(column) != size:
                raise ValueError(f"Task metadata {key!r} is not aligned with the sample batch")
            columns[key] = column

    result = {}
    source_tasks = {"t2i", "i2i", *task_weights}
    for index in rows:
        candidates = set()

        def collect(metadata):
            task = metadata.get("task_type")
            if task is not None:
                if not isinstance(task, str) or not task.strip() or task != task.strip():
                    raise ValueError(f"Invalid task_type at sample {index}: expected a nonempty task name")
                candidates.add(task)
            source = metadata.get("data_source")
            if isinstance(source, str) and source.rsplit("/", 1)[-1] in source_tasks:
                candidates.add(source.rsplit("/", 1)[-1])
            flag = metadata.get("task_is_i2i")
            if flag is not None:
                if isinstance(flag, (bool, np.bool_)) or not isinstance(flag, Real) or flag not in (0, 1):
                    raise ValueError(f"Invalid task_is_i2i at sample {index}: expected 0 or 1")
                candidates.add("i2i" if flag == 1 else "t2i")

        collect({key: values[index] for key, values in columns.items()})
        for key in ("extra_info", "reward_extra_info"):
            if key in columns:
                metadata = columns[key][index]
                if isinstance(metadata, Mapping):
                    collect(metadata)
        if len(candidates) != 1:
            reason = "Missing" if not candidates else "Conflicting"
            raise ValueError(f"{reason} task metadata at sample {index}; task reward weights require actual task labels")
        result[index] = candidates.pop()
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
    timesteps. With task overrides, each task uses its own active components and
    statistics. Loss-side advantage clipping remains the sole clipping step.
    """
    weights = validate_advantage_config(config, require_weights=True)
    rewards = data.batch["sample_level_rewards"]
    if rewards.ndim != 2 or not rewards.is_floating_point() or not rewards.numel():
        raise ValueError("sample_level_rewards must be a nonempty floating [batch, timesteps] tensor")
    size = rewards.shape[0]
    if not config.get("weight_advantages", True):
        from verl_omni.trainer.diffusion.diffusion_algos import compute_flow_grpo_outcome_advantage

        return compute_flow_grpo_outcome_advantage(
            sample_level_rewards=rewards,
            index=np.asarray(_group_ids(data, size), dtype=object),
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
    groups = _group_ids(data, size, rows)
    options = config.get("reward_advantage", {})
    normalization = options.get("normalization", "max_group_std")
    task_weights = options.get("task_weights", {})
    partitions = {}
    if task_weights:
        for index, task in _task_ids(data, rows, size, task_weights).items():
            partitions.setdefault(task, []).append(index)
    else:
        partitions[None] = rows
    with torch.no_grad():
        advantages = torch.zeros_like(rewards)
        for task, task_rows in partitions.items():
            active = reward_weights_for_task(config, task) if task_weights else weights
            # Inactive heads need not be returned by the scorer. Ignore even
            # stale/nonfinite values rather than turning absent scores into zero.
            components = {name: _component(data, name, task_rows, size, rewards.device) for name in active}
            combined = compute_weighted_advantages(
                components,
                [groups[index] for index in task_rows],
                active,
                global_std=normalization == "global",
                max_group_std=normalization == "max_group_std",
                norm_by_std=config.get("norm_adv_by_std_in_grpo", True),
                epsilon=options.get("epsilon", 1e-4),
            )
            advantages[task_rows] = combined.to(dtype=rewards.dtype)[:, None]
    if not torch.isfinite(advantages).all():
        raise ValueError("Combined reward advantages are nonfinite in the training dtype")
    return advantages, advantages
