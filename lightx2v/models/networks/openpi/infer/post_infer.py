"""Convert padded normalized model actions back to LIBERO's 7-D space."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from .pre_infer import load_norm_stats


class OpenPIPostInfer:
    def __init__(self, norm_stats_path: str | Path, output_action_dim: int = 7):
        self.stats = load_norm_stats(norm_stats_path)["actions"]
        self.output_action_dim = int(output_action_dim)

    def infer(self, actions: torch.Tensor | np.ndarray) -> np.ndarray:
        if isinstance(actions, torch.Tensor):
            actions = actions.detach().to(torch.float32).cpu().numpy()
        actions = np.asarray(actions, dtype=np.float32)
        if actions.ndim == 3:
            if actions.shape[0] != 1:
                raise ValueError(f"Only batch size 1 is supported by the policy API, got {actions.shape}")
            actions = actions[0]
        if actions.ndim != 2 or actions.shape[-1] < self.output_action_dim:
            raise ValueError(f"Expected [horizon, padded_action_dim], got {actions.shape}")
        q01 = self.stats["q01"][: self.output_action_dim]
        q99 = self.stats["q99"][: self.output_action_dim]
        physical = (actions[:, : self.output_action_dim] + 1.0) / 2.0 * (q99 - q01 + 1e-6) + q01
        return np.asarray(physical, dtype=np.float32)
