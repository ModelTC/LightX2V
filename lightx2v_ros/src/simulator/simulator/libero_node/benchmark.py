"""LIBERO and LIBERO-plus evaluation over the shared ROS-free runtime."""

import contextlib
import io
import json
import os
from pathlib import Path

import numpy as np

from simulator.libero_node.env import LiberoEnv, quat_to_axis_angle
from simulator.libero_node.observer import load_libero as load_runtime
from simulator.sim.evaluator import Observation

POLICY_PROFILE = "libero"
ACTION_SPACE = "libero_delta_eef"


TRIALS_FIELD = "num_trials"


PATH_FIELDS = ("EVALUATION.libero_root",)


def defaults(benchmark):
    plus = benchmark == "libero_plus"
    root = Path(__file__).resolve().parent / ("LIBERO-plus" if plus else "LIBERO")
    return {
        "EVALUATION": {
            "libero_root": os.environ.get("LIBERO_PLUS_SOURCE_DIR" if plus else "LIBERO_SOURCE_DIR") or str(root),
            "num_trials": 1 if plus else 50,
            "num_steps_wait": 5,
            "settle_action": [0, 0, 0, 0, 0, 0, -1],
            "render_size": 256,
        },
        "MULTIRUN": {
            "task_suite_names": ["libero_spatial", "libero_object", "libero_goal", "libero_10"],
            "task_ids": None,
        },
    }


def worker_environment(env):
    env.setdefault("MUJOCO_GL", "egl")
    env.setdefault("PYOPENGL_PLATFORM", env["MUJOCO_GL"])
    # robosuite selects the physical device from CUDA_VISIBLE_DEVICES.
    env.pop("MUJOCO_EGL_DEVICE_ID", None)


def load_libero(cfg):
    root = Path(cfg["EVALUATION"]["libero_root"])
    config_dir = Path(cfg["EVALUATION"]["output_dir"]) / "runtime" / f"libero-{os.getpid()}"
    return load_runtime(root, config_dir)


def suite_instance(factory):
    with contextlib.redirect_stdout(io.StringIO()):
        return factory()


def load_init_states(suite, task_id):
    from unittest.mock import patch

    import torch

    # Let the suite resolve Plus perturbation filenames. Assets are trusted local files.
    original_load = torch.load

    def trusted_load(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        kwargs.setdefault("map_location", "cpu")
        return original_load(*args, **kwargs)

    with patch("torch.load", trusted_load):
        states = suite.get_task_init_states(task_id)
    return states


def discover_tasks(cfg):
    benchmark, _, _ = load_libero(cfg)
    factories = benchmark.get_benchmark_dict()
    root = Path(cfg["EVALUATION"]["libero_root"])
    classification = root / "libero/libero/benchmark/task_classification.json"
    categories = json.loads(classification.read_text()) if classification.is_file() else {}
    tasks = []
    selected = cfg["MULTIRUN"]["task_ids"]
    for suite in cfg["MULTIRUN"]["task_suite_names"]:
        instance = suite_instance(factories[suite])
        ids = list(range(instance.get_num_tasks())) if selected is None else selected
        for task_id in ids:
            task = instance.get_task(task_id)
            meta = categories.get(suite, [])
            entry = meta[task_id] if meta else {}
            tasks.append({"key": f"{suite}/{task_id}", "suite": suite, "task_id": task_id, "task_name": task.name, "category": entry.get("category"), "phase": None, "prompt": task.language})
    return tasks


def observation(raw, prompt, step):
    images = {cam: np.ascontiguousarray(raw[LiberoEnv.CAMERA_OBS_KEYS[cam]][::-1, ::-1]) for cam in ("agentview", "wrist")}
    state = np.concatenate([raw["robot0_eef_pos"], quat_to_axis_angle(raw["robot0_eef_quat"]), raw["robot0_gripper_qpos"]]).astype(np.float32)
    return Observation(images, state, prompt, step)


class LiberoAdapter:
    action_dim = 7
    action_space = ACTION_SPACE

    def __init__(self, cfg, task):
        self.cfg, self.task = cfg, task
        evaluation = cfg["EVALUATION"]
        benchmark, get_path, env_cls = load_libero(cfg)
        suite = suite_instance(benchmark.get_benchmark_dict()[task["suite"]])
        self.initial_states = load_init_states(suite, task["task_id"])
        task_info = suite.get_task(task["task_id"])
        self.env = env_cls(
            bddl_file_name=str(Path(get_path("bddl_files")) / task_info.problem_folder / task_info.bddl_file),
            camera_heights=evaluation["render_size"],
            camera_widths=evaluation["render_size"],
        )
        self.env.seed(cfg["seed"])
        self.max_steps = evaluation["max_steps"] or (700 if task["suite"] in ("libero_10", "libero_90") else 400)

    def reset(self, episode_index):
        self.step_index = 0
        self.init_index = episode_index % len(self.initial_states)
        self.env.reset()
        state = self.initial_states[self.init_index]
        raw = self.env.set_init_state(np.asarray(state).copy())
        for _ in range(self.cfg["EVALUATION"]["num_steps_wait"]):
            raw, _, _, _ = self.env.step(self.cfg["EVALUATION"]["settle_action"])
        return observation(raw, self.task["prompt"], 0)

    def step(self, action):
        raw, _, success, _ = self.env.step(np.asarray(action, dtype=np.float32))
        self.step_index += 1
        return observation(raw, self.task["prompt"], self.step_index), bool(success), bool(success) or self.step_index >= self.max_steps

    def episode_metadata(self):
        return {"environment_seed": self.cfg["seed"], "init_state_index": self.init_index}

    def close(self):
        self.env.close()


def create_adapter(cfg, task):
    return LiberoAdapter(cfg, task)
