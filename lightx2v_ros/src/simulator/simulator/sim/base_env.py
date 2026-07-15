"""Environment-agnostic simulator interface.

Every concrete simulator (LIBERO, RoboTwin, ...) is exposed to the ROS layer
through the same small contract: it produces an `Observation` (a dict of
camera RGB frames plus a flat proprio-state vector) and consumes an action
vector. The generic `SimulatorNode` only ever talks to this interface, so
adding a new environment never requires touching the node/topic plumbing.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
from common.contract import EnvContract


@dataclass
class Observation:
    # logical camera name -> HxWx3 uint8 RGB image, already oriented for publishing
    images: Dict[str, np.ndarray]
    # flat proprio-state vector, shape (contract.state_dim,)
    state: np.ndarray


class BaseSimEnv(ABC):
    def __init__(self, contract: EnvContract):
        self.contract = contract

    @property
    @abstractmethod
    def task_description(self) -> str:
        """Current natural-language task/instruction."""

    @abstractmethod
    def reset(self) -> Observation:
        """Reset the environment and return the first observation."""

    @abstractmethod
    def step(self, action: np.ndarray) -> Tuple[Observation, bool]:
        """Apply one action, returning (observation, success)."""

    def new_episode(self) -> Observation:
        """Start a fresh episode and return its first observation.

        Used by the node's continuous-eval loop. The default simply calls
        ``reset()``; environments that need to re-randomize a scene (e.g.
        RoboTwin) override this to tear down and rebuild the episode.
        """
        return self.reset()

    @property
    def max_steps(self):
        """Optional per-episode step cap hint (None = let the node decide)."""
        return None

    # ------------------------------------------------- optional capabilities
    @property
    def supports_task_switch(self) -> bool:
        """Whether `set_task` can rebuild the env with a different task/config."""
        return False

    def list_tasks(self):
        """Available task names for `set_task` (empty if unsupported)."""
        return []

    def list_task_configs(self):
        """Available task-config/scenario names for `set_task` (empty if unsupported)."""
        return []

    def set_task(self, task_name: str, task_config: str = "", seed=None) -> Observation:
        raise NotImplementedError(f"env '{self.contract.name}' does not support runtime task switching")

    def set_frame_callback(self, callback) -> None:
        """Register a callback(images: Dict[str, ndarray]) invoked with intermediate
        frames rendered while an action is being executed (viewer-only frames).
        Environments that cannot render mid-action may ignore this."""
        return None

    # Concrete envs may also expose plain attributes `task_name`, `task_config`
    # and `seed`; the node reads them via getattr for status reporting.

    def close(self) -> None:
        return None

    def validate(self, obs: Observation) -> None:
        missing = [cam for cam in self.contract.cameras if cam not in obs.images]
        if missing:
            raise ValueError(f"env '{self.contract.name}' observation is missing cameras: {missing}")
        for cam in self.contract.cameras:
            image = np.asarray(obs.images[cam])
            if image.ndim != 3 or image.shape[2] != 3:
                raise ValueError(f"env '{self.contract.name}' camera '{cam}' must be HxWx3, got {image.shape}")
        state = np.asarray(obs.state, dtype=np.float32).reshape(-1)
        if state.size != self.contract.state_dim:
            raise ValueError(f"env '{self.contract.name}' state dim {state.size} != contract {self.contract.state_dim}")
