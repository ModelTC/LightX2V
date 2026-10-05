import os
from pathlib import Path

POLICY_PROFILE = "libero"
DEFAULT_BACKBONE = "fasterwam"
LEGACY_TASK = "libero_uncond_2cam224_1e-4"
PATH_FIELDS = ("EVALUATION.libero_root",)


def defaults(benchmark):
    plus = benchmark == "libero_plus"
    root = Path(__file__).resolve().parents[1] / ("LIBERO-plus" if plus else "LIBERO")
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


def discover_tasks(cfg):
    from simulator.libero_node.bench.adapter import discover_tasks as discover

    return discover(cfg)


def create_adapter(cfg, task):
    if cfg["benchmark"] == "libero_plus":
        from simulator.libero_node.bench.plus import LiberoPlusAdapter as Adapter
    else:
        from simulator.libero_node.bench.adapter import LiberoAdapter as Adapter
    return Adapter(cfg, task)
