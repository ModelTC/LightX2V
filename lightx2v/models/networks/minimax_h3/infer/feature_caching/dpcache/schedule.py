"""Path-aware scheduling with an explicit (previous, current) DP state."""

import numpy as np


def select_steps(cost, steps, budget, first=3, last=1):
    if not 1 <= first <= budget <= steps or not 0 <= last <= steps:
        raise ValueError("DPCache requires 1 <= first <= budget <= steps and 0 <= last <= steps")
    mandatory = set(range(first)) | set(range(steps - last, steps)) | {0, steps}
    if len(mandatory) > budget + 1:
        raise ValueError("DPCache budget is smaller than the mandatory step count")
    if cost.shape != (steps, steps + 1, steps + 1) or np.isnan(cost).any():
        raise ValueError("DPCache cost tensor has an invalid shape or NaNs")
    # The edge cost depends on BOTH preceding anchors. Keeping only the best
    # path to current (as upstream does) can discard the globally best path.
    states = {(0, 0): (0.0, [0])}
    max_jump = max(int(steps * 0.3), 1)
    for count in range(1, budget + 1):
        following = {}
        for (previous, current), (score, path) in states.items():
            for target in range(current + 1, min(steps, current + max_jump) + 1):
                if (target == steps) != (count == budget):
                    continue
                if any(current < required < target for required in mandatory):
                    continue
                candidate = score + float(cost[previous, current, target])
                key = (current, target)
                if np.isfinite(candidate) and candidate < following.get(key, (np.inf,))[0]:
                    following[key] = (candidate, path + [target])
        states = following
    if not states:
        raise ValueError("No DPCache path reaches the sentinel with this budget and mandatory steps")
    return min(states.values(), key=lambda item: item[0])[1][:-1]
