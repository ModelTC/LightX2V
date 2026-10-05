"""LIBERO implementation of the generic `BaseSimEnv` contract."""

import numpy as np
from common.contract import EnvContract

from ..sim.base_env import BaseSimEnv, Observation
from .observer import LiberoActionObserver, build_task_catalog, default_libero_root
from .runtime import CAMERA_OBS_KEYS, observation_components, observation_state
from .runtime import quat_to_axis_angle as quat_to_axis_angle


class LiberoEnv(BaseSimEnv):
    # logical camera name -> LIBERO observation key
    CAMERA_OBS_KEYS = CAMERA_OBS_KEYS

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
        success = bool(success)
        return self._observation(), success, success

    def _observation(self) -> Observation:
        images, state = observation_components(self.observer.obs, self.contract.cameras)
        return Observation(images=images, state=state)

    def _state(self, obs) -> np.ndarray:
        return observation_state(obs)

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
