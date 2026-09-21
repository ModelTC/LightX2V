"""Convert normalized model actions to LIBERO's 7-D action space."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from .pre_infer import load_norm_stats

LIBERO_ACTION_DIM = 7


class OpenPIPostInfer:
    def __init__(self, norm_stats_path: str | Path):
        self.stats = load_norm_stats(norm_stats_path)["actions"]

    def infer(self, actions: torch.Tensor) -> np.ndarray:
        if actions.ndim != 3 or actions.shape[0] != 1 or actions.shape[-1] < LIBERO_ACTION_DIM:
            raise ValueError(f"Expected actions with shape [1, horizon, padded_action_dim], got {tuple(actions.shape)}")
        normalized = actions[0, :, :LIBERO_ACTION_DIM].detach().to(torch.float32).cpu().numpy()
        q01 = self.stats["q01"][:LIBERO_ACTION_DIM]
        q99 = self.stats["q99"][:LIBERO_ACTION_DIM]
        physical = (normalized + 1.0) / 2.0 * (q99 - q01 + 1e-6) + q01
        return np.asarray(physical)
