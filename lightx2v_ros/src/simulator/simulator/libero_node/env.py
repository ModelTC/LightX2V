"""LIBERO implementation of the generic `BaseSimEnv` contract."""

import math

import numpy as np
from common.contract import EnvContract

from ..sim.base_env import BaseSimEnv, Observation
from .observer import LiberoActionObserver, build_task_catalog, default_libero_root


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
        self.image_size = int(image_size)
        self.libero_root = libero_root
        self.observer = LiberoActionObserver(
            benchmark_name=benchmark,
            task_id=int(task_id),
            init_state_id=int(init_state_id),
            image_size=self.image_size,
            seed=int(seed),
            libero_root=self.libero_root,
        )
        self._task_catalog = build_task_catalog(self.observer.benchmark_module)
        self._sync_metadata()

    def _sync_metadata(self):
        self.benchmark = self.observer.benchmark_name
        self.task_id = self.observer.task_id
        self.init_state_id = self.observer.init_state_id
        self.task_name = self.observer.task_key
        self.task_config = str(self.init_state_id)
        self.seed = self.observer.seed

    @property
    def task_description(self) -> str:
        return self.observer.task_description

    def reset(self) -> Observation:
        self.observer.reset()
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

    @property
    def supports_task_switch(self) -> bool:
        return True

    def list_tasks(self):
        return [
            {
                "value": key,
                "label": f"[{item['benchmark']} {item['task_id']}] {item['language']}",
            }
            for key, item in self._task_catalog.items()
        ]

    def list_task_configs(self):
        return [str(index) for index in range(self.observer.num_init_states)]

    def set_task(self, task_name: str, task_config: str = "", seed=None) -> Observation:
        task_key = str(task_name).strip()
        task = self._task_catalog.get(task_key)
        if task is None:
            raise ValueError(f"unknown LIBERO task {task_key!r}")

        init_state_id = self.init_state_id if str(task_config).strip() == "" else int(task_config)
        new_seed = self.seed + 1 if seed is None or str(seed).strip() == "" else int(seed)

        # Construct the replacement first so an invalid task/config leaves the
        # currently displayed environment alive and usable.
        new_observer = LiberoActionObserver(
            benchmark_name=task["benchmark"],
            task_id=task["task_id"],
            init_state_id=init_state_id,
            image_size=self.image_size,
            seed=new_seed,
            libero_root=self.libero_root,
        )
        old_observer = self.observer
        self.observer = new_observer
        self._sync_metadata()
        try:
            old_observer.close()
        except Exception:
            pass
        return self._observation()

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
