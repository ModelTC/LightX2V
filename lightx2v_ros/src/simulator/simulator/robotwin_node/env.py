"""RoboTwin (SAPIEN, dual-arm) implementation of the generic `BaseSimEnv`.

This adapter wraps a vendored RoboTwin task so the generic `SimulatorNode` can
drive it exactly like LIBERO. RoboTwin's own evaluation orchestration lives in
``third_party/RoboTwin/script/eval_policy.py``; that script is RoboTwin-driven
(it owns the rollout loop and calls a policy plugin). Here we invert the control
flow for ROS: we build the same task `args`, run ``setup_demo`` once, and then
expose ``reset`` / ``step`` (``get_obs`` + ``take_action(qpos)``), publishing the
three RoboTwin cameras and the 14-dim joint-state vector over ROS.

RoboTwin heavy dependencies (sapien, mplib, curobo, ...) and assets are imported
lazily inside ``reset``/construction, so the ROS package builds and imports even
on machines where the RoboTwin runtime is not installed yet.
"""

import importlib
import os
import sys
from pathlib import Path

import numpy as np
from common.contract import EnvContract

from ..sim.base_env import BaseSimEnv, Observation


def default_robotwin_root() -> Path:
    return Path(__file__).resolve().parent / "RoboTwin"


def _add_python_path(path) -> None:
    path = str(Path(path))
    if path not in sys.path:
        sys.path.insert(0, path)


class RoboTwinEnv(BaseSimEnv):
    """Single-episode RoboTwin environment exposed through the BaseSimEnv contract."""

    def __init__(
        self,
        contract: EnvContract,
        *,
        task_name: str = "click_alarmclock",
        task_config: str = "demo_clean",
        embodiment: str = "aloha-agilex",
        instruction_type: str = "unseen",
        instruction: str = "",
        seed: int = 0,
        robotwin_root=None,
    ):
        super().__init__(contract)
        self.robotwin_root = Path(robotwin_root or default_robotwin_root()).expanduser()
        self.task_name = str(task_name)
        self.task_config = str(task_config)
        self.embodiment = str(embodiment).strip()
        self.instruction_type = str(instruction_type)
        self._fixed_instruction = str(instruction).strip()
        self.seed = int(seed)
        self._episode_index = 0

        self._task_description = ""
        self._configs_path = self.robotwin_root / "task_config"

        self._prepare_runtime()
        self.args = self._build_task_args()
        self.env = self._instantiate_task()
        self._setup_episode()

    # ------------------------------------------------------------------ setup
    def _prepare_runtime(self) -> None:
        root = self.robotwin_root
        if not (root / "envs").is_dir():
            raise FileNotFoundError(f"RoboTwin is not vendored at {root}. See robotwin_node/RoboTwin/README and run the RoboTwin install/asset-download steps.")
        # RoboTwin source uses root-relative imports such as `from envs import ...`
        # and `from generate_episode_instructions import *`.
        _add_python_path(root)
        _add_python_path(root / "description" / "utils")

    def _require_config(self, *parts) -> Path:
        path = self._configs_path.joinpath(*parts)
        if not path.exists():
            raise FileNotFoundError(f"Missing RoboTwin config: {path}. Populate `task_config/` (and `assets/`) from the official RoboTwin repo (see robotwin_node/RoboTwin/script).")
        return path

    def _build_task_args(self) -> dict:
        """Replicates third_party/RoboTwin/script/eval_policy.py:main() arg assembly."""
        import yaml

        with open(self._require_config(f"{self.task_config}.yml"), "r", encoding="utf-8") as f:
            args = yaml.load(f.read(), Loader=yaml.FullLoader)

        args["task_name"] = self.task_name
        args["task_config"] = self.task_config

        # Allow the launch parameter to pin the embodiment (e.g. "aloha-agilex").
        if self.embodiment:
            args["embodiment"] = [self.embodiment]
        embodiment_type = args.get("embodiment")
        if not isinstance(embodiment_type, list):
            raise ValueError(f"task_config embodiment must be a list, got {embodiment_type!r}")

        with open(self._require_config("_embodiment_config.yml"), "r", encoding="utf-8") as f:
            embodiment_types = yaml.load(f.read(), Loader=yaml.FullLoader)

        def embodiment_file(key):
            robot_file = embodiment_types[key]["file_path"]
            if robot_file is None:
                raise ValueError(f"No embodiment file for '{key}'")
            return os.path.join(str(self.robotwin_root), robot_file) if not os.path.isabs(robot_file) else robot_file

        def embodiment_config(robot_file):
            with open(os.path.join(robot_file, "config.yml"), "r", encoding="utf-8") as f:
                return yaml.load(f.read(), Loader=yaml.FullLoader)

        with open(self._require_config("_camera_config.yml"), "r", encoding="utf-8") as f:
            camera_config = yaml.load(f.read(), Loader=yaml.FullLoader)

        head_camera_type = args["camera"]["head_camera_type"]
        args["head_camera_h"] = camera_config[head_camera_type]["h"]
        args["head_camera_w"] = camera_config[head_camera_type]["w"]

        if len(embodiment_type) == 1:
            args["left_robot_file"] = embodiment_file(embodiment_type[0])
            args["right_robot_file"] = embodiment_file(embodiment_type[0])
            args["dual_arm_embodied"] = True
        elif len(embodiment_type) == 3:
            args["left_robot_file"] = embodiment_file(embodiment_type[0])
            args["right_robot_file"] = embodiment_file(embodiment_type[1])
            args["embodiment_dis"] = embodiment_type[2]
            args["dual_arm_embodied"] = False
        else:
            raise ValueError("embodiment items should be 1 or 3")

        args["left_embodiment_config"] = embodiment_config(args["left_robot_file"])
        args["right_embodiment_config"] = embodiment_config(args["right_robot_file"])
        args["eval_mode"] = True
        # Headless: never spawn the on-screen SAPIEN viewer.
        args["render_freq"] = 0
        return args

    def _instantiate_task(self):
        module = importlib.import_module(f"envs.{self.task_name}")
        task_cls = getattr(module, self.task_name)
        return task_cls()

    def _setup_episode(self) -> None:
        self.env.setup_demo(now_ep_num=self._episode_index, seed=self.seed, is_test=True, **self.args)
        instruction = self._resolve_instruction()
        self.env.set_instruction(instruction=instruction)
        self._task_description = instruction

    def _resolve_instruction(self) -> str:
        if self._fixed_instruction:
            return self._fixed_instruction
        try:
            from generate_episode_instructions import generate_episode_descriptions

            episode_info_list = [self.env.info["info"]]
            results = generate_episode_descriptions(self.task_name, episode_info_list, 1)
            return str(np.random.choice(results[0][self.instruction_type]))
        except Exception:
            return f"Complete the {self.task_name.replace('_', ' ')} task."

    # ------------------------------------------------------------- contract API
    @property
    def task_description(self) -> str:
        return self._task_description

    def reset(self) -> Observation:
        return self._observation()

    @property
    def max_steps(self):
        # RoboTwin sets a per-task rollout cap (`step_lim`) during setup_demo.
        return getattr(self.env, "step_lim", None)

    def new_episode(self, max_setup_retries: int = 25) -> Observation:
        """Tear down the current episode and set up a fresh one (new layout).

        Advances the seed so each episode gets a different object placement, and
        retries the next seeds if setup raises (e.g. RoboTwin ``UnStableError``).
        """
        last_err = None
        for _ in range(max(1, max_setup_retries)):
            self.seed += 1
            try:
                try:
                    self.env.close_env(clear_cache=True)
                except Exception:
                    pass
                self._episode_index += 1
                self._setup_episode()
                return self._observation()
            except Exception as exc:  # e.g. UnStableError on an unlucky seed
                last_err = exc
        raise RuntimeError(f"RoboTwin failed to set up a new episode after {max_setup_retries} seeds; last error: {last_err}")

    def step(self, action):
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        # RoboTwin policies output absolute joint targets (qpos), matching FastWAM.
        self.env.take_action(action, action_type="qpos")
        obs = self._observation()
        success = bool(getattr(self.env, "eval_success", False)) or bool(self.env.check_success())
        return obs, success

    def _observation(self) -> Observation:
        raw = self.env.get_obs()
        cameras = raw["observation"]
        images = {}
        for cam in self.contract.cameras:
            if cam not in cameras or "rgb" not in cameras[cam]:
                raise KeyError(f"RoboTwin observation missing camera '{cam}' rgb; got {list(cameras)}")
            rgb = np.asarray(cameras[cam]["rgb"])[..., :3]
            images[cam] = np.ascontiguousarray(rgb.astype(np.uint8))
        state = np.asarray(raw["joint_action"]["vector"], dtype=np.float32).reshape(-1)
        return Observation(images=images, state=state)

    def close(self) -> None:
        env = getattr(self, "env", None)
        if env is not None:
            try:
                env.close_env()
            except Exception:
                pass


def build_robotwin_env(node) -> RoboTwinEnv:
    contract = node.contract
    node.declare_parameter("robotwin_root", str(default_robotwin_root()))
    node.declare_parameter("task_name", "click_alarmclock")
    node.declare_parameter("task_config", "demo_clean")
    node.declare_parameter("embodiment", "aloha-agilex")
    node.declare_parameter("instruction_type", "unseen")
    node.declare_parameter("instruction", "")
    node.declare_parameter("seed", 0)

    return RoboTwinEnv(
        contract,
        task_name=node.get_parameter("task_name").value,
        task_config=node.get_parameter("task_config").value,
        embodiment=node.get_parameter("embodiment").value,
        instruction_type=node.get_parameter("instruction_type").value,
        instruction=node.get_parameter("instruction").value,
        seed=int(node.get_parameter("seed").value),
        robotwin_root=node.get_parameter("robotwin_root").value,
    )
