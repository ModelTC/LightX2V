"""Validate reward balancing without importing the distributed runtime."""

import math
from collections.abc import Mapping
from numbers import Real

ADVANTAGE_FUNCTION = "lightx2v_rl.backends.verl_omni.advantage.compute_advantages"
NORMALIZATIONS = frozenset({"group", "global", "max_group_std"})


def _validate_weights(weights, label):
    if not isinstance(weights, Mapping):
        raise ValueError(f"{label} must map reward detail names to weights")
    for name, weight in weights.items():
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Reward advantage component names must be nonempty strings")
        if isinstance(weight, bool) or not isinstance(weight, Real) or not math.isfinite(weight) or weight < 0:
            raise ValueError(f"Reward advantage weight for {name!r} must be finite and nonnegative")


def reward_weights_for_task(algorithm, task):
    """Resolve validated partial task overrides, excluding inactive components."""
    options = algorithm.get("reward_advantage", {})
    weights = dict(options.get("weights", {}))
    weights.update(options.get("task_weights", {}).get(task, {}))
    return {name: float(weight) for name, weight in weights.items() if weight > 0}


def validate_advantage_config(algorithm, *, require_weights=False):
    enabled = algorithm.get("weight_advantages", True)
    if not isinstance(enabled, bool):
        raise ValueError("algorithm.weight_advantages must be a boolean")
    if algorithm.get("custom_advantage_function") != ADVANTAGE_FUNCTION:
        raise ValueError(f"FewStep recipes require algorithm.custom_advantage_function={ADVANTAGE_FUNCTION}")
    if not isinstance(algorithm.get("norm_adv_by_std_in_grpo", True), bool):
        raise ValueError("algorithm.norm_adv_by_std_in_grpo must be a boolean")
    if not enabled:
        return {}
    options = algorithm.get("reward_advantage", {})
    if not isinstance(options, Mapping):
        raise ValueError("algorithm.reward_advantage must be a mapping")
    normalization = options.get("normalization", "max_group_std")
    if not isinstance(normalization, str) or normalization not in NORMALIZATIONS:
        raise ValueError("reward_advantage.normalization must be group, global, or max_group_std")
    epsilon = options.get("epsilon", 1e-4)
    if isinstance(epsilon, bool) or not isinstance(epsilon, Real) or not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("reward_advantage.epsilon must be positive and finite")
    weights = options.get("weights", {})
    _validate_weights(weights, "reward_advantage.weights")
    task_weights = options.get("task_weights", {})
    if not isinstance(task_weights, Mapping):
        raise ValueError("reward_advantage.task_weights must map task names to component weight overrides")
    for task, overrides in task_weights.items():
        if not isinstance(task, str) or not task.strip() or task != task.strip():
            raise ValueError("Reward advantage task names must be nonempty strings without surrounding whitespace")
        _validate_weights(overrides, f"reward_advantage.task_weights.{task}")
        if (weights or require_weights) and not reward_weights_for_task(algorithm, task):
            raise ValueError(f"Enabled reward advantages require at least one positive component weight for task {task!r}")
    active = {name: float(weight) for name, weight in weights.items() if weight > 0}
    if (weights or require_weights) and not active:
        raise ValueError("Enabled reward advantages require at least one positive component weight")
    return active
