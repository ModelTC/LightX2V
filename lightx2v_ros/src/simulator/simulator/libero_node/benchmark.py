"""LIBERO and LIBERO-plus evaluation over the shared ROS-free runtime."""

import json
import os
from pathlib import Path

from simulator.libero_node.observer import LiberoActionObserver, observation_components, suite_instance
from simulator.libero_node.observer import load_libero as load_runtime
from simulator.sim.evaluator import Observation

POLICY_PROFILE = "libero"


LEGACY_TASK = "libero_uncond_2cam224_1e-4"


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


def validate(cfg):
    if cfg.EVALUATION.num_trials < 1 or cfg.EVALUATION.num_steps_wait < 0:
        raise ValueError("LIBERO num_trials must be positive and num_steps_wait nonnegative")


def required_paths(cfg):
    return [cfg["EVALUATION"]["libero_root"]]


def num_trials(cfg):
    return cfg["EVALUATION"]["num_trials"]


def worker_environment(env):
    env.setdefault("MUJOCO_GL", "egl")
    env.setdefault("PYOPENGL_PLATFORM", env["MUJOCO_GL"])
    # robosuite selects the physical device from CUDA_VISIBLE_DEVICES.
    env.pop("MUJOCO_EGL_DEVICE_ID", None)


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
        if cfg["benchmark"] == "libero_plus" and not task.get("category"):
            raise ValueError("Every LIBERO-plus task must have a perturbation category")
        self.cfg, self.task = cfg, task
        evaluation = cfg["EVALUATION"]
        self.runtime = LiberoActionObserver(
            benchmark_name=task["suite"],
            task_id=task["task_id"],
            image_size=evaluation["render_size"],
            seed=cfg["seed"],
            libero_root=evaluation["libero_root"],
            config_dir=runtime_directory(cfg),
            camera_names=None,
            eager_reset=False,
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


def create_adapter(cfg, task):
    return LiberoAdapter(cfg, task)
