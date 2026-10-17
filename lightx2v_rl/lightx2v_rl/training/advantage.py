"""Validate reward balancing without importing the distributed runtime."""

import math
from collections.abc import Mapping
from numbers import Real

ADVANTAGE_FUNCTION = "lightx2v_rl.backends.verl_omni.advantage.compute_advantages"
NORMALIZATIONS = frozenset({"group", "global", "max_group_std"})


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
    if not isinstance(weights, Mapping):
        raise ValueError("reward_advantage.weights must map reward detail names to weights")
    active = {}
    for name, weight in weights.items():
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Reward advantage component names must be nonempty strings")
        if isinstance(weight, bool) or not isinstance(weight, Real) or not math.isfinite(weight) or weight < 0:
            raise ValueError(f"Reward advantage weight for {name!r} must be finite and nonnegative")
        if weight:
            active[name] = float(weight)
    if (weights or require_weights) and not active:
        raise ValueError("Enabled reward advantages require at least one positive component weight")
    return active
