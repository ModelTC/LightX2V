from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np


@dataclass
class Observation:
    images: dict[str, np.ndarray]  # HWC uint8 RGB, semantic camera names
    state: np.ndarray  # physical units; model adapter normalizes
    prompt: str
    step: int = 0


@dataclass
class ActionChunk:
    actions: np.ndarray  # physical units, [horizon, action_dim]
    action_space: str  # libero_delta_eef or robotwin_joint_position
    timing: dict[str, float] = field(default_factory=dict)

    def validate(self, action_space, action_dim):
        array = np.asarray(self.actions, dtype=np.float32)
        if self.action_space != action_space:
            raise ValueError(f"Action space mismatch: {self.action_space} != {action_space}")
        if array.ndim != 2 or array.shape[1] != action_dim or len(array) < 1 or not np.isfinite(array).all():
            raise ValueError(f"Invalid action chunk: shape={array.shape}, expected [N,{action_dim}] finite values")
        return array


class Policy(Protocol):
    def reset_episode(self, metadata: dict[str, Any]) -> None: ...
    def predict_action_chunk(self, observation: Observation) -> ActionChunk: ...
    def close(self) -> None: ...


class Environment(Protocol):
    action_dim: int
    action_space: str
    max_steps: int

    def reset(self, episode_index: int) -> Observation: ...
    def step(self, action: np.ndarray) -> tuple[Observation, bool, bool]: ...
    def episode_metadata(self) -> dict: ...
    def close(self) -> None: ...
