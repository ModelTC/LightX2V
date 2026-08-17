import importlib
import os
import sys
from pathlib import Path

import numpy as np
from common.contract import EnvContract

from simulator.sim.base_env import BaseSimEnv, Observation

DEFAULT_TASK_NAME = "stack_bowls"
DEFAULT_ENV_CFG_TYPE = "arx_x5"


def default_robodojo_root() -> Path:
    return Path(os.environ.get("ROBODOJO_ROOT", "/app/RoboDojo"))


def resolve_robodojo_root(path=None) -> Path:
    raw_path = str(path).strip() if path is not None else ""
    if not raw_path:
        return default_robodojo_root().resolve()
    return Path(os.path.expandvars(raw_path)).expanduser().resolve()


def add_robodojo_python_paths(root: Path) -> None:
    # Official eval_policy.sh adds both entries for env/task/utils and
    # client_server.ws imports.
    paths = [str(root), str(root / "XPolicyLab")]
    for path in paths:
        while path in sys.path:
            sys.path.remove(path)
    sys.path[:0] = paths


class NoopModelClient:
    """Small stand-in for EvalEnv construction; policy inference happens via ROS."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def call(self, func_name=None, obs=None, **kwargs):
        return None

    def close(self):
        return None


def _to_numpy(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _as_vector(value, expected_dim, key):
    arr = _to_numpy(value).astype(np.float32, copy=False)
    if arr.ndim > 1 and arr.shape[0] == 1:
        arr = arr[0]
    arr = arr.reshape(-1)
    if arr.size != expected_dim:
        raise ValueError(f"RoboDojo state key '{key}' expected {expected_dim} values, got {arr.size}")
    return arr


def _as_rgb(value, key):
    image = _to_numpy(value)
    if image.ndim == 4 and image.shape[0] == 1:
        image = image[0]
    if image.ndim != 3 or image.shape[-1] < 3:
        raise ValueError(f"RoboDojo camera '{key}' must be HxWx3 RGB, got {image.shape}")
    image = image[..., :3]
    if np.issubdtype(image.dtype, np.floating):
        if image.size and float(np.nanmax(image)) <= 1.0:
            image = image * 255.0
        image = np.clip(image, 0, 255).round().astype(np.uint8)
    else:
        image = image.astype(np.uint8, copy=False)
    return np.ascontiguousarray(image)


def _first_bool(value):
    arr = _to_numpy(value)
    if arr.shape == ():
        return bool(arr.item())
    return bool(arr.reshape(-1)[0])


def _as_text(value):
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        return _as_text(value[0]) if value else ""
    if hasattr(value, "item"):
        value = value.item()
    return str(value)


def split_joint_action(action):
    action = np.asarray(action, dtype=np.float32).reshape(-1)
    if action.size != 14:
        raise ValueError(f"RoboDojo joint action expects 14 values, got {action.size}")
    # Presence of arm_joint_state keys makes EvalEnv infer joint action mode.
    return {
        "left_arm_joint_state": action[0:6].copy(),
        "left_ee_joint_state": action[6:7].copy(),
        "right_arm_joint_state": action[7:13].copy(),
        "right_ee_joint_state": action[13:14].copy(),
    }


class RoboDojoEnv(BaseSimEnv):
    """RoboDojo official EvalEnv exposed through LightX2V's simulator contract."""

    CAMERA_KEYS = {
        "head_camera": ("cam_head", "color"),
        "left_camera": ("cam_left_wrist", "color"),
        "right_camera": ("cam_right_wrist", "color"),
    }

    STATE_KEYS = (
        ("left_arm_joint_state", 6),
        ("left_ee_joint_state", 1),
        ("right_arm_joint_state", 6),
        ("right_ee_joint_state", 1),
    )

    def __init__(
        self,
        contract: EnvContract,
        *,
        simulation_app,
        robodojo_root=None,
        task_name=DEFAULT_TASK_NAME,
        env_cfg_type=DEFAULT_ENV_CFG_TYPE,
        eval_seed=0,
        layout_id=0,
        device_id=None,
        policy_name="FastWAM",
        additional_info="lightx2v_ros,action_type=joint",
    ):
        super().__init__(contract)
        if simulation_app is None:
            raise ValueError("RoboDojoEnv requires an already-started Isaac simulation app.")
        self.simulation_app = simulation_app
        self.robodojo_root = resolve_robodojo_root(robodojo_root)
        if not self.robodojo_root.is_dir():
            raise FileNotFoundError(f"RoboDojo root does not exist: {self.robodojo_root}")
        add_robodojo_python_paths(self.robodojo_root)

        self.task_name = str(task_name)
        self.env_cfg_type = str(env_cfg_type)
        self.task_config = self.env_cfg_type
        self.seed = int(eval_seed)
        self.layout_id = int(layout_id)
        self.device_id = 0 if device_id is None or str(device_id).strip() == "" else int(device_id)
        self.policy_name = str(policy_name)
        self.additional_info = str(additional_info)
        self._raw_obs = None
        self._last_instruction = ""

        self._env_cfg = self._build_env_cfg()
        self._env = self._create_eval_env(self._env_cfg)
        self._validate_joint_robot()

    def _build_env_cfg(self):
        import env.global_configs as global_configs
        from omegaconf import OmegaConf
        from utils.load_file import load_yaml
        from utils.pipeline_utils import process_config, process_randomization, resolve_random_task_num_envs

        task_registry = importlib.import_module(f"task.{global_configs.BENCHMARK}.task_registry")
        eval_cfg = load_yaml(os.path.join(global_configs.ENV_CONFIG_PATH, self.env_cfg_type + ".yml"))
        eval_cfg["task_name"] = self.task_name
        eval_cfg["num_envs"] = 1
        eval_cfg["device_id"] = self.device_id
        eval_cfg["eval_batch"] = False
        eval_cfg["policy_name"] = self.policy_name
        eval_cfg["additional_info"] = self.additional_info
        eval_cfg["seed"] = self.seed
        eval_cfg["physx_monitor_enabled"] = False

        deploy_cfg = {
            "policy_name": self.policy_name,
            "port": 1,
            "host": "localhost",
            "protocol": "ws",
            "policy_server_url": "ws://localhost:1",
            "evaluation_id": os.environ.get("ROBODOJO_RUN_ID", "lightx2v_ros"),
            "trial_id": f"{self.task_name}-lightx2v_ros",
            "action_case_id": f"{self.task_name}_case",
            "repeat_index": None,
            "action_type": "joint",
        }

        benchmark_path = os.path.join(global_configs.ROOT_DIR, "task", global_configs.BENCHMARK)
        env_cfg = OmegaConf.create(
            {
                "sim": load_yaml(os.path.join(global_configs.ENV_CONFIG_PATH, "sim", eval_cfg["config"]["sim"] + ".yml")),
                "scene": load_yaml(os.path.join(global_configs.ENV_CONFIG_PATH, "scene", eval_cfg["config"]["scene"] + ".yml")),
                "camera": load_yaml(os.path.join(global_configs.ENV_CONFIG_PATH, "camera", eval_cfg["config"]["camera"] + ".yml")),
                "robot": load_yaml(os.path.join(global_configs.ENV_CONFIG_PATH, "robot", eval_cfg["config"]["robot"] + ".yml")),
                "task_env": load_yaml(task_registry.task_config_path(os.path.join(benchmark_path, "config"), self.task_name)),
                "eval_cfg": eval_cfg,
                "deploy_cfg": deploy_cfg,
            }
        )

        num_envs = resolve_random_task_num_envs(self.task_name, 1, env_cfg.sim)
        eval_cfg["num_envs"] = num_envs
        OmegaConf.update(env_cfg, "sim.scene.num_envs", num_envs, force_add=True)
        OmegaConf.update(env_cfg, "eval_cfg.num_envs", num_envs, force_add=True)

        env_cfg = process_randomization(env_cfg)
        env_cfg, eval_num = process_config(env_cfg, task_name=self.task_name)
        OmegaConf.update(env_cfg, "eval_cfg.eval_num", min(int(eval_num), 1), force_add=True)
        OmegaConf.update(
            env_cfg,
            "camera.default_frequency",
            eval_cfg["observation"].get("collect_freq", 0),
            force_add=True,
        )
        env_cfg.sim.seed = [0 for _ in range(num_envs)]
        return env_cfg

    def _create_eval_env(self, env_cfg):
        import src.eval_client.eval_env as eval_env_mod

        old_client = eval_env_mod.WsModelClient
        eval_env_mod.WsModelClient = NoopModelClient
        try:
            return eval_env_mod.create_eval_env(env_cfg, self.simulation_app)
        finally:
            eval_env_mod.WsModelClient = old_client

    def _validate_joint_robot(self):
        info = getattr(self._env, "robot_action_dim_info", {})
        if list(info.get("arm_dim", [])) != [6, 6] or list(info.get("ee_dim", [])) != [1, 1]:
            raise ValueError(f"RoboDojo FastWAM joint path requires dual 6D arms and 1D grippers, got {info}")

    @property
    def task_description(self):
        if self._last_instruction:
            return self._last_instruction
        obs_manager = getattr(self._env, "obs_manager", None)
        return _as_text(getattr(obs_manager, "instruction", ""))

    @property
    def max_steps(self):
        step_lim = getattr(self._env, "step_lim", None)
        return int(step_lim) if step_lim is not None else None

    def reset(self):
        return self._reset_layout(self.layout_id)

    def new_episode(self):
        self.layout_id = self._next_layout_id(self.layout_id)
        return self._reset_layout(self.layout_id)

    def _next_layout_id(self, current):
        seed_info = getattr(getattr(self._env, "seed_manager", None), "seed_info", {})
        layout_ids = sorted(int(value) for value in seed_info.keys())
        if not layout_ids:
            return int(current)
        if int(current) not in layout_ids:
            return layout_ids[0]
        index = layout_ids.index(int(current))
        return layout_ids[(index + 1) % len(layout_ids)]

    def _reset_layout(self, layout_id):
        self.layout_id = int(layout_id)
        self.task_config = f"{self.env_cfg_type}/layout_{self.layout_id}"
        self._env.reset(seed=[self.layout_id])
        self._prepare_official_episode()
        raw_obs = self._env.get_obs()
        self._raw_obs = raw_obs
        return self._convert_observation(raw_obs)

    def _prepare_official_episode(self):
        self._env.run_reward()
        if hasattr(self._env, "get_score"):
            self._env.get_score()
        if getattr(self._env, "interact", False) and hasattr(self._env, "query_support_arm_traj"):
            for env_idx in self._env.get_running_env_idx_list():
                self._env.query_support_arm_traj(env_idx=env_idx)

    def step(self, action):
        action_dict = split_joint_action(action)
        self._env.take_action(action_dict)
        raw_obs = self._env.get_obs()
        self._raw_obs = raw_obs
        obs = self._convert_observation(raw_obs)
        done = _first_bool(self._env.end_flag)
        success = bool(done and _first_bool(self._env.success))
        return obs, success, done

    def _convert_observation(self, raw_obs):
        self._last_instruction = _as_text(raw_obs.get("instruction", ""))
        vision = raw_obs["vision"]
        state_dict = raw_obs["state"]

        images = {}
        for logical_name, (camera_key, obs_key) in self.CAMERA_KEYS.items():
            images[logical_name] = _as_rgb(vision[camera_key][obs_key], camera_key)

        state_parts = []
        for key, dim in self.STATE_KEYS:
            state_parts.append(_as_vector(state_dict[key], dim, key))
        state = np.concatenate(state_parts).astype(np.float32, copy=False)
        return Observation(images=images, state=state)

    def close(self):
        env = getattr(self, "_env", None)
        if env is None:
            return
        try:
            model_client = getattr(env, "model_client", None)
            close_client = getattr(model_client, "close", None)
            if callable(close_client):
                close_client()
        finally:
            try:
                env.close()
            finally:
                self._env = None


def _env_int(name, default):
    value = os.environ.get(name)
    return int(value) if value is not None and str(value).strip().lstrip("-").isdigit() else default


def build_robodojo_env(node, *, simulation_app):
    contract = node.contract
    node.declare_parameter("robodojo_root", str(default_robodojo_root()))
    node.declare_parameter("task_name", DEFAULT_TASK_NAME)
    node.declare_parameter("env_cfg_type", DEFAULT_ENV_CFG_TYPE)
    node.declare_parameter("eval_seed", 0)
    node.declare_parameter("layout_id", 0)
    node.declare_parameter("device_id", _env_int("ROBODOJO_DEVICE_ID", 0))
    node.declare_parameter("additional_info", "lightx2v_ros,action_type=joint")
    device_id = int(node.get_parameter("device_id").value)

    return RoboDojoEnv(
        contract,
        simulation_app=simulation_app,
        robodojo_root=node.get_parameter("robodojo_root").value,
        task_name=node.get_parameter("task_name").value,
        env_cfg_type=node.get_parameter("env_cfg_type").value,
        eval_seed=int(node.get_parameter("eval_seed").value),
        layout_id=int(node.get_parameter("layout_id").value),
        device_id=device_id,
        additional_info=node.get_parameter("additional_info").value,
    )
