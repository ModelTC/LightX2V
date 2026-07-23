import os
import sys
from pathlib import Path

import numpy as np
from common.contract import ROBOLAB_CONTRACT

from simulator.sim.base_env import BaseSimEnv, Observation


def default_robolab_root():
    return Path(os.environ.get("ROBOLAB_ROOT", "/app/RoboLab"))


class RoboLabEnv(BaseSimEnv):
    """Single-environment RoboLab DROID adapter for the generic ROS node."""

    def __init__(
        self,
        *,
        robolab_root,
        task_name="BananaInBowlTask",
        task_config="",
        instruction_type="default",
        device="cuda:0",
        seed=0,
        output_dir="",
        double_reset=True,
    ):
        super().__init__(ROBOLAB_CONTRACT)
        self.robolab_root = Path(robolab_root).expanduser().resolve()
        if not self.robolab_root.is_dir():
            raise FileNotFoundError(f"RoboLab root does not exist: {self.robolab_root}")
        if str(self.robolab_root) not in sys.path:
            sys.path.insert(0, str(self.robolab_root))

        self.task_name = str(task_name)
        self.task_config = str(task_config)
        self.instruction_type = str(instruction_type)
        self.device = str(device)
        self.seed = int(seed)
        self.output_dir = Path(output_dir).expanduser() if output_dir else self.robolab_root / "output" / "lightx2v_ros"
        self.double_reset = bool(double_reset)
        self._env = None
        self._env_cfg = None
        self._raw_obs = None
        self._register_and_create()

    def _register_and_create(self):
        import robolab.constants
        from robolab.constants import set_output_dir
        from robolab.core.environments.factory import get_envs
        from robolab.core.environments.runtime import create_env
        from robolab.registrations.droid.auto_env_registrations_jointpos import auto_register_droid_envs
        from robolab.registrations.droid.camera_presets import WRIST_LEFT_RIGHT_HEAD

        robolab.constants.VERBOSE = False
        robolab.constants.DEBUG = False
        robolab.constants.RECORD_IMAGE_DATA = False
        set_output_dir(str(self.output_dir))

        auto_register_droid_envs(task=[self.task_name], cameras=WRIST_LEFT_RIGHT_HEAD, enable_cameras=True)
        if self.task_config:
            candidates = get_envs(env=self.task_config)
        else:
            candidates = get_envs(task=self.task_name)
        if not candidates:
            raise ValueError(f"RoboLab task {self.task_name!r} has no registered environment")

        self.task_config = candidates[0]
        self._env, self._env_cfg = create_env(
            self.task_config,
            device=self.device,
            seed=self.seed,
            num_envs=1,
            instruction_type=self.instruction_type,
            policy="cosmos3",
            use_fabric=True,
        )

    @property
    def task_description(self):
        return str(self._env_cfg.instruction)

    @property
    def max_steps(self):
        return int(self._env.max_episode_length) if self._env is not None else None

    def _reset_raw(self):
        if hasattr(self._env, "reset_eval_state"):
            self._env.reset_eval_state()
        raw_obs, _ = self._env.reset()
        # Match RoboLab's official Cosmos3 evaluation path and the captured
        # offline sample, both of which perform two initial resets.
        if self.double_reset:
            raw_obs, _ = self._env.reset()
        self._raw_obs = raw_obs
        return raw_obs

    @staticmethod
    def _as_rgb(value):
        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()
        image = np.asarray(value)
        if image.ndim == 4:
            image = image[0]
        if np.issubdtype(image.dtype, np.floating):
            if image.size and float(np.nanmax(image)) <= 1.0:
                image = image * 255.0
            image = np.clip(image, 0, 255).round().astype(np.uint8)
        else:
            image = image.astype(np.uint8, copy=False)
        if image.ndim != 3 or image.shape[-1] != 3:
            raise ValueError(f"RoboLab camera observation must be HxWx3 RGB, got {image.shape}")
        return np.ascontiguousarray(image)

    @staticmethod
    def _as_vector(value):
        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()
        value = np.asarray(value, dtype=np.float32)
        if value.ndim > 1:
            value = value[0]
        return value.reshape(-1)

    def _convert_observation(self, raw_obs):
        image_obs = raw_obs["image_obs"]
        proprio = raw_obs["proprio_obs"]
        images = {camera: self._as_rgb(image_obs[camera]) for camera in self.contract.cameras}
        state = np.concatenate(
            [
                self._as_vector(proprio["arm_joint_pos"]),
                self._as_vector(proprio["gripper_pos"]),
            ]
        ).astype(np.float32, copy=False)
        return Observation(images=images, state=state)

    def reset(self):
        return self._convert_observation(self._reset_raw())

    def new_episode(self):
        return self.reset()

    def step(self, action):
        import torch

        action = np.asarray(action, dtype=np.float32).reshape(1, -1)
        action_tensor = torch.as_tensor(action, dtype=torch.float32, device=self._env.device)
        raw_obs, _, terminated, truncated, _ = self._env.step(action_tensor)
        self._raw_obs = raw_obs
        success = bool(terminated.reshape(-1)[0].item())
        # A timeout is reported as failure by SimulatorNode's matching step cap.
        if bool(truncated.reshape(-1)[0].item()) and not success:
            success = False
        return self._convert_observation(raw_obs), success

    def close(self):
        if self._env is not None:
            self._env.close()
            self._env = None


def build_robolab_env(node):
    node.declare_parameter("robolab_root", str(default_robolab_root()))
    node.declare_parameter("task_name", "BananaInBowlTask")
    node.declare_parameter("task_config", "")
    node.declare_parameter("instruction_type", "default")
    node.declare_parameter("device", os.environ.get("ROBOLAB_DEVICE", "cuda:0"))
    node.declare_parameter("seed", 0)
    node.declare_parameter("output_dir", "")
    node.declare_parameter("double_reset", True)
    return RoboLabEnv(
        robolab_root=node.get_parameter("robolab_root").value,
        task_name=node.get_parameter("task_name").value,
        task_config=node.get_parameter("task_config").value,
        instruction_type=node.get_parameter("instruction_type").value,
        device=node.get_parameter("device").value,
        seed=node.get_parameter("seed").value,
        output_dir=node.get_parameter("output_dir").value,
        double_reset=node.get_parameter("double_reset").value,
    )
