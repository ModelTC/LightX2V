"""LIBERO implementation of the generic `BaseSimEnv` contract."""

import math

import numpy as np
from common.contract import EnvContract

from ..sim.base_env import BaseSimEnv, Observation
from .observer import LiberoActionObserver, default_libero_root


def quat_to_axis_angle(quat):
    quat = np.asarray(quat, dtype=np.float32).copy()
    quat[3] = np.clip(quat[3], -1.0, 1.0)
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(float(den), 0.0):
        return np.zeros(3, dtype=np.float32)
    return ((quat[:3] * 2.0 * math.acos(float(quat[3]))) / den).astype(np.float32)


class LiberoEnv(BaseSimEnv):
    # logical camera name -> LIBERO observation key
    CAMERA_OBS_KEYS = {
        "agentview": "agentview_image",
        "wrist": "robot0_eye_in_hand_image",
        "frontview": "frontview_image",
        "galleryview": "galleryview_image",
    }

    def __init__(
        self,
        contract: EnvContract,
        *,
        benchmark="libero_spatial",
        task_id=0,
        init_state_id=0,
        image_size=224,
        seed=0,
        libero_root=None,
    ):
        super().__init__(contract)
        self.observer = LiberoActionObserver(
            benchmark_name=benchmark,
            task_id=int(task_id),
            init_state_id=int(init_state_id),
            image_size=int(image_size),
            seed=int(seed),
            libero_root=libero_root,
        )

    @property
    def task_description(self) -> str:
        return self.observer.task_description

    def reset(self) -> Observation:
        return self._observation()

    def step(self, action):
        _, _, success, _ = self.observer.step(action)
        return self._observation(), bool(success)

    def _observation(self) -> Observation:
        obs = self.observer.obs
        # LIBERO renders upside-down/mirrored relative to the policy expectation.
        images = {cam: np.ascontiguousarray(obs[key][::-1, ::-1]) for cam, key in self.CAMERA_OBS_KEYS.items() if cam in self.contract.cameras}
        return Observation(images=images, state=self._state(obs))

    def _state(self, obs) -> np.ndarray:
        pos = np.asarray(obs["robot0_eef_pos"], dtype=np.float32)
        axis_angle = quat_to_axis_angle(np.asarray(obs["robot0_eef_quat"], dtype=np.float32))
        gripper = np.asarray(obs["robot0_gripper_qpos"], dtype=np.float32)
        return np.concatenate([pos, axis_angle, gripper]).astype(np.float32)

    def close(self) -> None:
        self.observer.close()


def build_libero_env(node) -> LiberoEnv:
    contract = node.contract
    node.declare_parameter("libero_root", str(default_libero_root()))
    node.declare_parameter("benchmark", "libero_spatial")
    node.declare_parameter("task_id", 0)
    node.declare_parameter("init_state_id", 0)
    node.declare_parameter("image_size", contract.image_size)
    node.declare_parameter("seed", 0)

    return LiberoEnv(
        contract,
        benchmark=node.get_parameter("benchmark").value,
        task_id=int(node.get_parameter("task_id").value),
        init_state_id=int(node.get_parameter("init_state_id").value),
        image_size=int(node.get_parameter("image_size").value),
        seed=int(node.get_parameter("seed").value),
        libero_root=node.get_parameter("libero_root").value,
    )
