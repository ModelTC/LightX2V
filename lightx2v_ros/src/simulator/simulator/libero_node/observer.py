import os
import sys
from pathlib import Path

import numpy as np


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
    import torch

    init_states_path = Path(get_libero_path("init_states")) / task.problem_folder / task.init_states_file
    return torch.load(init_states_path, weights_only=False)[init_state_id]


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

        task_suite = benchmark.get_benchmark_dict()[benchmark_name.lower()]()
        task = task_suite.get_task(task_id)
        self.task_description = task.language
        bddl_file = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
        init_state = load_init_states(get_libero_path, task, init_state_id)

        self.env = env_cls(
            bddl_file_name=str(bddl_file),
            camera_heights=image_size,
            camera_widths=image_size,
            camera_names=["robot0_eye_in_hand", "agentview", "frontview", "galleryview"],
        )
        self.env.seed(seed)
        self.env.reset()
        self.obs = self.env.set_init_state(init_state)

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        self.obs, reward, success, info = self.env.step(action)
        return self.obs, reward, success, info

    def close(self):
        self.env.close()
