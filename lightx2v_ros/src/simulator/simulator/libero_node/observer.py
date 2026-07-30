import os
import sys
from pathlib import Path

import numpy as np

LIBERO_BENCHMARKS = (
    "libero_spatial",
    "libero_object",
    "libero_goal",
    "libero_10",
    "libero_90",
)


def default_libero_root():
    return Path(__file__).resolve().parent / "LIBERO"


def add_python_path(path):
    path = str(Path(path).expanduser())
    if path not in sys.path:
        sys.path.insert(0, path)


def setup_libero_config(libero_root):
    benchmark_root = libero_root / "libero" / "libero"
    if not (benchmark_root / "bddl_files").exists():
        raise FileNotFoundError(f"LIBERO submodule is incomplete: {libero_root}")

    config_dir = Path.home() / ".cache" / "lightx2v_ros" / "libero_config"
    config_file = config_dir / "config.yaml"
    config_dir.mkdir(parents=True, exist_ok=True)
    config_file.write_text(
        "\n".join(
            [
                f"benchmark_root: {benchmark_root}",
                f"bddl_files: {benchmark_root / 'bddl_files'}",
                f"init_states: {benchmark_root / 'init_files'}",
                f"datasets: {libero_root / 'libero' / 'datasets'}",
                f"assets: {benchmark_root / 'assets'}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    os.environ["LIBERO_CONFIG_PATH"] = str(config_dir)


def load_libero(libero_root):
    add_python_path(libero_root)
    setup_libero_config(libero_root)

    try:
        from libero.libero import benchmark, get_libero_path
        from libero.libero.envs import OffScreenRenderEnv
    except ModuleNotFoundError as exc:
        if exc.name in {"robosuite", "bddl"}:
            raise ModuleNotFoundError(f"Missing dependency '{exc.name}'. Activate the LIBERO runtime first.") from exc
        raise

    return benchmark, get_libero_path, OffScreenRenderEnv


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


class LiberoActionObserver:
    def __init__(
        self,
        benchmark_name="libero_spatial",
        task_id=0,
        init_state_id=0,
        image_size=224,
        seed=0,
        libero_root=None,
    ):
        self.libero_root = Path(libero_root or default_libero_root()).expanduser()
        benchmark, get_libero_path, env_cls = load_libero(self.libero_root)

        self.benchmark_module = benchmark
        self.benchmark_name = str(benchmark_name).strip().lower()
        factories = benchmark.get_benchmark_dict()
        if self.benchmark_name not in factories or self.benchmark_name not in LIBERO_BENCHMARKS:
            raise ValueError(f"unknown LIBERO benchmark {benchmark_name!r}; available: {', '.join(LIBERO_BENCHMARKS)}")
        task_suite = factories[self.benchmark_name]()
        self.task_id = int(task_id)
        if self.task_id < 0 or self.task_id >= task_suite.get_num_tasks():
            raise ValueError(f"task_id {self.task_id} is out of range for {self.benchmark_name!r}; expected 0..{task_suite.get_num_tasks() - 1}")
        task = task_suite.get_task(self.task_id)
        self.task = task
        self.task_description = task.language
        bddl_file = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
        # Keep an owned copy: every restart must restore this exact MuJoCo state,
        # rather than returning the observation cached after the last action.
        self.init_state_id = int(init_state_id)
        init_state, self.num_init_states = load_init_state(get_libero_path, task, self.init_state_id)
        self.init_state = np.asarray(init_state).copy()
        self.image_size = int(image_size)
        self.seed = int(seed)

        self.env = env_cls(
            bddl_file_name=str(bddl_file),
            camera_heights=self.image_size,
            camera_widths=self.image_size,
            camera_names=["robot0_eye_in_hand", "agentview", "frontview", "galleryview"],
        )
        self.env.seed(self.seed)
        self.reset()

    @property
    def task_key(self):
        return f"{self.benchmark_name}/{self.task_id}"

    def reset(self):
        """Reset simulator internals and restore the configured initial state."""
        self.env.reset()
        self.obs = self.env.set_init_state(self.init_state.copy())
        return self.obs

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        self.obs, reward, success, info = self.env.step(action)
        return self.obs, reward, success, info

    def close(self):
        self.env.close()
