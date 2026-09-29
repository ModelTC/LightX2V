"""Benchmark protocol over the shared LIBERO runtime; no ROS required."""

import json
import os
from pathlib import Path

from simulator.libero_node.runtime import LiberoRuntime, observation_components, suite_instance
from simulator.libero_node.runtime import load_libero as load_runtime
from simulator.sim.bench.interfaces import Observation


def runtime_directory(cfg):
    return Path(cfg["EVALUATION"]["output_dir"]) / "runtime" / f"libero-{os.getpid()}"


def load_libero(cfg):
    return load_runtime(cfg["EVALUATION"]["libero_root"], runtime_directory(cfg))


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
    images, state = observation_components(raw)
    return Observation(images, state, prompt, step)


class LiberoAdapter:
    action_dim = 7
    action_space = "libero_delta_eef"

    def __init__(self, cfg, task):
        self.cfg, self.task = cfg, task
        evaluation = cfg["EVALUATION"]
        self.runtime = LiberoRuntime(
            benchmark_name=task["suite"],
            task_id=task["task_id"],
            image_size=evaluation["render_size"],
            seed=cfg["seed"],
            libero_root=evaluation["libero_root"],
            config_dir=runtime_directory(cfg),
        )
        self.env, self.initial_states = self.runtime.env, self.runtime.initial_states
        self.max_steps = evaluation["max_steps"] or (700 if task["suite"] in ("libero_10", "libero_90") else 400)

    def reset(self, episode_index):
        self.step_index = 0
        self.init_index = episode_index % len(self.initial_states)
        raw = self.runtime.reset(self.init_index, settle_steps=self.cfg["EVALUATION"]["num_steps_wait"], settle_action=self.cfg["EVALUATION"]["settle_action"])
        return observation(raw, self.task["prompt"], 0)

    def step(self, action):
        raw, _, success, _ = self.runtime.step(action.tolist())
        self.step_index += 1
        return observation(raw, self.task["prompt"], self.step_index), bool(success), bool(success) or self.step_index >= self.max_steps

    def episode_metadata(self):
        return {"environment_seed": self.cfg["seed"], "init_state_index": self.init_index}

    def close(self):
        self.runtime.close()
