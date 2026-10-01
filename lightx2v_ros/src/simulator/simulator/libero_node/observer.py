"""ROS-free LIBERO runtime shared by interactive simulation and benchmarks.

Trial scheduling, horizons, policy inference and result accounting deliberately
live outside this module. LIBERO and LIBERO-plus need separate processes.
"""

import contextlib
import io
import os
import sys
from pathlib import Path

import numpy as np

LIBERO_BENCHMARKS = ("libero_spatial", "libero_object", "libero_goal", "libero_10", "libero_90")
CAMERA_OBS_KEYS = {"agentview": "agentview_image", "wrist": "robot0_eye_in_hand_image", "frontview": "frontview_image", "galleryview": "galleryview_image"}


def default_libero_root():
    return Path(__file__).resolve().parent / "LIBERO"


def add_python_path(path):
    path = str(Path(path).expanduser().resolve())
    if path not in sys.path:
        sys.path.insert(0, path)


def setup_libero_config(libero_root, config_dir=None):
    import yaml

    root = Path(libero_root).expanduser().resolve()
    benchmark_root = root / "libero/libero"
    if not (benchmark_root / "bddl_files").is_dir():
        hint = "Check the optional libero_root override."
        if root.parent == default_libero_root().parent and root.name in ("LIBERO", "LIBERO-plus"):
            hint = f"Run git submodule update --init --recursive lightx2v_ros/src/simulator/simulator/libero_node/{root.name} from the repository root."
        raise FileNotFoundError(f"Incomplete LIBERO source: {root}. {hint}")
    directory = Path(config_dir) if config_dir is not None else Path.home() / ".cache/lightx2v_ros/libero_config"
    directory = directory.expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)
    payload = {
        "benchmark_root": str(benchmark_root),
        "bddl_files": str(benchmark_root / "bddl_files"),
        "init_states": str(benchmark_root / "init_files"),
        "assets": str(benchmark_root / "assets"),
        "datasets": str(root / "libero/datasets"),
    }
    (directory / "config.yaml").write_text(yaml.safe_dump(payload))
    os.environ["LIBERO_CONFIG_PATH"] = str(directory)


def load_libero(libero_root, config_dir=None):
    root = Path(libero_root).expanduser().resolve()
    # The outer 'libero' can be a namespace package with __file__ == None.
    already = sys.modules.get("libero.libero")
    if already and not Path(already.__file__).resolve().is_relative_to(root):
        raise RuntimeError("LIBERO and LIBERO-plus must run in separate processes")
    setup_libero_config(root, config_dir)
    add_python_path(root)
    try:
        from libero.libero import benchmark, get_libero_path
        from libero.libero.envs import OffScreenRenderEnv
    except ModuleNotFoundError as exc:
        if exc.name in {"robosuite", "bddl"}:
            raise ModuleNotFoundError(f"Missing dependency '{exc.name}'. Activate the LIBERO runtime first.") from exc
        raise
    if not Path(sys.modules["libero.libero"].__file__).resolve().is_relative_to(root):
        raise RuntimeError(f"Imported LIBERO from a different installation; expected {root}")
    return benchmark, get_libero_path, OffScreenRenderEnv


def suite_instance(factory):
    with contextlib.redirect_stdout(io.StringIO()):
        return factory()


def load_init_states(get_libero_path, task, init_state_id):
    state, _ = load_init_state(get_libero_path, task, init_state_id)
    return state


def load_init_state(get_libero_path, task, init_state_id):
    import torch

    init_states_path = Path(get_libero_path("init_states")) / task.problem_folder / task.init_states_file
    init_states = torch.load(init_states_path, map_location="cpu", weights_only=False)
    index = int(init_state_id)
    if index < 0 or index >= len(init_states):
        raise ValueError(f"init_state_id {index} is out of range for {task.name!r}; expected 0..{len(init_states) - 1}")
    return init_states[index], len(init_states)


def build_task_catalog(benchmark_module):
    """Return stable UI task ids mapped to their LIBERO suite/task metadata."""
    factories = benchmark_module.get_benchmark_dict()
    catalog = {}
    for benchmark_name in LIBERO_BENCHMARKS:
        factory = factories.get(benchmark_name)
        if factory is None:
            continue
        task_suite = factory()
        for task_id in range(task_suite.get_num_tasks()):
            task = task_suite.get_task(task_id)
            key = f"{benchmark_name}/{task_id}"
            catalog[key] = {
                "benchmark": benchmark_name,
                "task_id": task_id,
                "task_name": task.name,
                "language": task.language,
            }
    return catalog


def load_suite_init_states(suite, task_id):
    from unittest.mock import patch

    import torch

    # Only trusted local benchmark assets. Let the suite resolve Plus-specific
    # perturbation names; do not reconstruct init filenames in either caller.
    original_load = torch.load

    def trusted_load(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        kwargs.setdefault("map_location", "cpu")
        return original_load(*args, **kwargs)

    with patch("torch.load", trusted_load):
        states = suite.get_task_init_states(task_id)
    if not len(states):
        raise ValueError("Empty initial-state collection")
    return states


def quat_to_axis_angle(quat):
    quat = np.asarray(quat, dtype=np.float32).copy()
    w = np.clip(quat[3], -1.0, 1.0)
    denominator = np.sqrt(1.0 - w * w)
    return np.zeros(3, dtype=np.float32) if np.isclose(denominator, 0) else quat[:3] * 2 * np.arccos(w) / denominator


def observation_state(raw):
    state = np.concatenate([raw["robot0_eef_pos"], quat_to_axis_angle(raw["robot0_eef_quat"]), raw["robot0_gripper_qpos"]]).astype(np.float32)
    if state.shape != (8,):
        raise ValueError(f"Expected LIBERO 8-D state; got {state.shape}")
    return state


def observation_components(raw, cameras=("agentview", "wrist")):
    # Match the policy/training convention; never flip again in a caller.
    images = {cam: np.ascontiguousarray(raw[CAMERA_OBS_KEYS[cam]][::-1, ::-1]) for cam in cameras}
    return images, observation_state(raw)


class LiberoActionObserver:
    def __init__(
        self,
        benchmark_name="libero_spatial",
        task_id=0,
        init_state_id=0,
        image_size=224,
        seed=0,
        libero_root=None,
        *,
        config_dir=None,
        camera_names=("robot0_eye_in_hand", "agentview", "frontview", "galleryview"),
        eager_reset=True,
    ):
        self.libero_root = Path(libero_root or default_libero_root()).expanduser().resolve()
        benchmark, get_path, env_cls = load_libero(self.libero_root, config_dir)
        self.benchmark_module = benchmark
        self.benchmark_name = str(benchmark_name).strip().lower()
        factories = benchmark.get_benchmark_dict()
        if self.benchmark_name not in factories or self.benchmark_name not in LIBERO_BENCHMARKS:
            raise ValueError(f"unknown LIBERO benchmark {benchmark_name!r}; available: {', '.join(LIBERO_BENCHMARKS)}")
        suite = suite_instance(factories[self.benchmark_name])
        self.task_id = int(task_id)
        if not 0 <= self.task_id < suite.get_num_tasks():
            raise ValueError(f"task_id {self.task_id} is out of range for {self.benchmark_name!r}")
        self.task = suite.get_task(self.task_id)
        self.task_description = self.task.language
        self.initial_states = load_suite_init_states(suite, self.task_id)
        self.num_init_states = len(self.initial_states)
        self._select_init_state(init_state_id)
        self.image_size, self.seed = int(image_size), int(seed)
        kwargs = {} if camera_names is None else {"camera_names": list(camera_names)}
        self.env = env_cls(
            bddl_file_name=str(Path(get_path("bddl_files")) / self.task.problem_folder / self.task.bddl_file),
            camera_heights=self.image_size,
            camera_widths=self.image_size,
            **kwargs,
        )
        self.env.seed(self.seed)
        self.obs = None
        if eager_reset:
            try:
                self.reset()
            except Exception:
                self.close()
                raise

    @property
    def task_key(self):
        return f"{self.benchmark_name}/{self.task_id}"

    def _select_init_state(self, index):
        index = int(index)
        if not 0 <= index < self.num_init_states:
            raise ValueError(f"init_state_id {index} is out of range for {self.task.name!r}; expected 0..{self.num_init_states - 1}")
        state = self.initial_states[index]
        self.init_state = (state.cpu().numpy() if hasattr(state, "cpu") else np.asarray(state)).copy()
        self.init_state_id = index

    def reset(self, init_state_id=None, *, settle_steps=0, settle_action=None):
        if settle_steps < 0 or (settle_steps and settle_action is None):
            raise ValueError("Settling requires nonnegative steps and an explicit action")
        if init_state_id is not None:
            self._select_init_state(init_state_id)
        self.env.reset()
        self.obs = self.env.set_init_state(self.init_state.copy())
        for _ in range(settle_steps):
            self.obs, _, _, _ = self.env.step(settle_action)
        return self.obs

    def step(self, action):
        self.obs, reward, success, info = self.env.step(np.asarray(action, dtype=np.float32))
        return self.obs, reward, success, info

    def close(self):
        self.env.close()
