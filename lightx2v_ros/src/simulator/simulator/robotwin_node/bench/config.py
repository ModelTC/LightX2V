import os
from pathlib import Path

POLICY_PROFILE = "robotwin"
DEFAULT_BACKBONE = "fastwam"
LEGACY_TASK = "robotwin_uncond_3cam_384_distilled_1step"
ROBOTWIN_ROOT = Path(__file__).resolve().parents[1] / "RoboTwin"
PATH_FIELDS = ("EVALUATION.robotwin_root", "EVALUATION.seed_cache_dir")


def defaults(benchmark):
    return {
        "EVALUATION": {
            "robotwin_root": os.environ.get("ROBOTWIN_ROOT") or str(ROBOTWIN_ROOT),
            "eval_num_episodes": 100,
            "skip_get_obs_within_replan": True,
            "instruction_type": "unseen",
            "embodiment": "aloha-agilex",
            "max_seed_attempts": 1000,
            "reuse_seed_cache": False,
            "seed_cache_dir": None,
        },
        "MULTIRUN": {"chunk_size": 1, "task_names": None, "phases": ["clean", "random"]},
    }


def validate(cfg):
    evaluation = cfg.EVALUATION
    if evaluation.eval_num_episodes < 1 or evaluation.max_seed_attempts < 1:
        raise ValueError("RoboTwin eval_num_episodes and max_seed_attempts must be positive")
    if not isinstance(evaluation.skip_get_obs_within_replan, bool):
        raise TypeError("skip_get_obs_within_replan must be a boolean")
    if set(cfg.MULTIRUN.phases) - {"clean", "random"}:
        raise ValueError("RoboTwin phases must be clean/random")
    if evaluation.reuse_seed_cache and not evaluation.seed_cache_dir:
        evaluation.seed_cache_dir = os.environ.get("ROBOTWIN_SEED_DIR") or str(Path(evaluation.robotwin_root) / "my_seeds")


def required_paths(cfg):
    return [cfg["EVALUATION"]["robotwin_root"]]


def num_trials(cfg):
    return cfg["EVALUATION"]["eval_num_episodes"]


def worker_environment(env):
    if not env.get("ROBOTWIN_NVIDIA_GL_ROOT"):
        return
    root = Path(env["ROBOTWIN_NVIDIA_GL_ROOT"])
    libgl = next((p for p in (root / "libGL.so.1", root / "libGL.so.1.7.0") if p.is_file()), None)
    if libgl:
        env["LD_PRELOAD"] = ":".join(filter(None, (str(libgl), env.get("LD_PRELOAD"))))
    icd = root / "nvidia_icd_abs.json"
    if icd.is_file():
        env["VK_ICD_FILENAMES"] = str(icd)


def discover_tasks(cfg):
    from simulator.robotwin_node.bench.adapter import discover_tasks as discover

    return discover(cfg)


def create_adapter(cfg, task):
    from simulator.robotwin_node.bench.adapter import RoboTwinAdapter

    return RoboTwinAdapter(cfg, task)
