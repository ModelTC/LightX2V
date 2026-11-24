"""Fixed-grid Euler integration used by the pinned HY-Motion checkpoint."""

import itertools

import torch


def odeint(function, initial, timesteps, method="euler"):
    if method != "euler":
        raise ValueError("The HY-Motion adapter supports the released Euler sampler")
    current = initial
    trajectory = [current]
    for start, end in itertools.pairwise(timesteps):
        current = current + (end - start) * function(start, current)
        trajectory.append(current)
    return torch.stack(trajectory)
