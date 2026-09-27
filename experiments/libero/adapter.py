"""LIBERO environment adapter; imports no model package and requires no ROS."""

import contextlib
import io
import json
import os
import sys
from pathlib import Path

import numpy as np

from experiments.common.interfaces import Observation


def load_libero(cfg):
    import yaml

    root = Path(cfg["EVALUATION"]["libero_root"])
    benchmark_root = root / "libero/libero"
    if not (benchmark_root / "bddl_files").is_dir():
        from experiments.common.config import ROOT, default_libero_root

        hint = "Check the optional EVALUATION.libero_root override."
        if root.resolve() == default_libero_root(cfg["benchmark"]).resolve():
            hint = f"From the repository root run: git submodule update --init --recursive {root.relative_to(ROOT)}"
        raise FileNotFoundError(f"Incomplete LIBERO source: {root}. {hint}")
    # The outer 'libero' may be a namespace package (__file__ is None).
    # Check the concrete inner package that owns benchmark implementations.
    already = sys.modules.get("libero.libero")
    if already and not Path(already.__file__).resolve().is_relative_to(root.resolve()):
        raise RuntimeError("LIBERO and LIBERO-plus must run in separate processes")
    directory = Path(cfg["EVALUATION"]["output_dir"]) / "runtime" / f"libero-{os.getpid()}"
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
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    if not Path(sys.modules["libero.libero"].__file__).resolve().is_relative_to(root.resolve()):
        raise RuntimeError(f"Imported LIBERO from a different installation; expected {root}")
    return benchmark, get_libero_path, OffScreenRenderEnv


def suite_instance(factory):
    with contextlib.redirect_stdout(io.StringIO()):
        return factory()


def discover_tasks(cfg):
    benchmark, _, _ = load_libero(cfg)
    factories = benchmark.get_benchmark_dict()
    root = Path(cfg["EVALUATION"]["libero_root"])
    classification = root / "libero/libero/benchmark/task_classification.json"
    categories = json.loads(classification.read_text()) if classification.is_file() else {}
    if cfg["benchmark"] == "libero_plus" and not categories:
        raise FileNotFoundError("LIBERO-plus requires task_classification.json for category accounting")
    tasks = []
    selected = cfg["MULTIRUN"]["task_ids"]
    for suite in cfg["MULTIRUN"]["task_suite_names"]:
        instance = suite_instance(factories[suite])
        ids = list(range(instance.get_num_tasks())) if selected is None else selected
        for task_id in ids:
            task = instance.get_task(task_id)
            meta = categories.get(suite, [])
            entry = meta[task_id] if meta else {}
            if entry and (int(entry["id"]) != task_id + 1 or entry["name"] != task.name):
                raise ValueError(f"Classification/task mismatch at {suite}/{task_id}")
            tasks.append({"key": f"{suite}/{task_id}", "suite": suite, "task_id": task_id, "task_name": task.name, "category": entry.get("category"), "phase": None, "prompt": task.language})
    return tasks


def observation(raw, prompt, step):
    quat = np.asarray(raw["robot0_eef_quat"], dtype=np.float32).copy()
    w = np.clip(quat[3], -1.0, 1.0)
    denominator = np.sqrt(1.0 - w * w)
    axis = np.zeros(3, dtype=np.float32) if np.isclose(denominator, 0) else quat[:3] * 2 * np.arccos(w) / denominator
    state = np.concatenate([raw["robot0_eef_pos"], axis, raw["robot0_gripper_qpos"]]).astype(np.float32)
    if state.shape != (8,):
        raise ValueError(f"Expected LIBERO 8-D state; got {state.shape}")
    return Observation({"agentview": np.ascontiguousarray(raw["agentview_image"][::-1, ::-1]), "wrist": np.ascontiguousarray(raw["robot0_eye_in_hand_image"][::-1, ::-1])}, state, prompt, step)


class LiberoAdapter:
    action_dim = 7
    action_space = "libero_delta_eef"

    def __init__(self, cfg, task):
        from unittest.mock import patch

        import torch

        self.cfg, self.task = cfg, task
        benchmark, path, env_cls = load_libero(cfg)
        suite = suite_instance(benchmark.get_benchmark_dict()[task["suite"]])
        sim_task = suite.get_task(task["task_id"])
        # Benchmark owns perturbation -> initial-state resolution. The supplied
        # benchmark assets are trusted local files; never patch torch globally.
        original_load = torch.load

        def trusted_load(*args, **kwargs):
            kwargs.setdefault("weights_only", False)
            kwargs.setdefault("map_location", "cpu")
            return original_load(*args, **kwargs)

        with patch("torch.load", trusted_load):
            self.initial_states = suite.get_task_init_states(task["task_id"])
        if not len(self.initial_states):
            raise ValueError("Empty initial-state collection")
        self.env = env_cls(
            bddl_file_name=str(Path(path("bddl_files")) / sim_task.problem_folder / sim_task.bddl_file), camera_heights=cfg["EVALUATION"]["render_size"], camera_widths=cfg["EVALUATION"]["render_size"]
        )
        self.env.seed(cfg["seed"])
        self.max_steps = cfg["EVALUATION"]["max_steps"] or (700 if task["suite"] in ("libero_10", "libero_90") else 400)

    def reset(self, episode_index):
        self.step_index = 0
        self.init_index = episode_index % len(self.initial_states)
        self.env.reset()
        state = self.initial_states[self.init_index]
        state = state.cpu().numpy() if hasattr(state, "cpu") else np.asarray(state)
        raw = self.env.set_init_state(state.copy())
        for _ in range(self.cfg["EVALUATION"]["num_steps_wait"]):
            raw, _, _, _ = self.env.step(self.cfg["EVALUATION"]["settle_action"])
        return observation(raw, self.task["prompt"], 0)

    def step(self, action):
        raw, _, success, _ = self.env.step(action.tolist())
        self.step_index += 1
        return observation(raw, self.task["prompt"], self.step_index), bool(success), bool(success) or self.step_index >= self.max_steps

    def episode_metadata(self):
        return {"environment_seed": self.cfg["seed"], "init_state_index": self.init_index}

    def close(self):
        self.env.close()
