import os
from pathlib import Path

from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[4]
LIBERO_NODE = ROOT / "lightx2v_ros/src/simulator/simulator/libero_node"
ROBOTWIN_ROOT = ROOT / "lightx2v_ros/src/simulator/simulator/robotwin_node/RoboTwin"


def default_libero_root(benchmark):
    """Match the ROS LIBERO adapter's repo-local submodule convention."""
    return LIBERO_NODE / ("LIBERO-plus" if benchmark == "libero_plus" else "LIBERO")


def load_config(benchmark, argv):
    cfg = OmegaConf.load(ROOT / "configs/bench/robotics/eval.yaml")
    cfg.benchmark = benchmark
    if benchmark == "libero_plus":
        cfg.EVALUATION.num_trials = 1
    if benchmark == "robotwin":
        cfg.model.backbone = "fastwam"
        cfg.EVALUATION.action_infer_mode = "first_frame"
        cfg.EVALUATION.replan_steps = 8
        cfg.EVALUATION.skip_get_obs_within_replan = True
        cfg.MULTIRUN.chunk_size = 1
    cfg.base_ckpt = os.environ.get("BASE_CKPT")
    cfg.lora_path = os.environ.get("LORA_PATH")
    cfg.ckpt = os.environ.get("CKPT_PATH")
    cfg.model.model_path = os.environ.get("WAN_MODEL_PATH")
    cfg.EVALUATION.dataset_stats_path = os.environ.get("DATASET_STATS_PATH")
    cfg.EVALUATION.output_dir = os.environ.get("OUT")
    cfg.EVALUATION.libero_root = os.environ.get("LIBERO_PLUS_SOURCE_DIR" if benchmark == "libero_plus" else "LIBERO_SOURCE_DIR")
    cfg.EVALUATION.robotwin_root = os.environ.get("ROBOTWIN_ROOT")
    OmegaConf.set_struct(cfg, True)
    OmegaConf.set_struct(cfg.model.options, False)
    for argument in argv:
        if "=" not in argument or argument.startswith("--"):
            raise ValueError(f"Expected key=value override (not --key=value), got {argument!r}")
        key, value = argument.split("=", 1)
        if key == "model":
            key = "model.name"
        if key == "task":
            expected = "robotwin_uncond_3cam_384_distilled_1step" if benchmark == "robotwin" else "libero_uncond_2cam224_1e-4"
            if value != expected:
                raise ValueError(f"Unsupported legacy task alias: {value}")
            continue
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist([f"{key}={value}"]))
    if benchmark in ("libero", "libero_plus") and not cfg.EVALUATION.libero_root:
        cfg.EVALUATION.libero_root = str(default_libero_root(benchmark))
    if benchmark == "robotwin" and not cfg.EVALUATION.robotwin_root:
        cfg.EVALUATION.robotwin_root = str(ROBOTWIN_ROOT)
    if cfg.ckpt and (cfg.base_ckpt or cfg.lora_path):
        raise ValueError("Use base_ckpt + lora_path OR a merged ckpt; do not combine them")
    if cfg.lora_path and not cfg.base_ckpt:
        raise ValueError("lora_path requires base_ckpt")
    if not cfg.model.factory and cfg.model.sampler != "teacher_flow":
        raise ValueError("RealtimeWAM teacher distillation uses teacher_flow; baseline preconditioning is not enabled")
    if cfg.model.backbone not in ("fastwam", "fasterwam"):
        raise ValueError("model.backbone must be fastwam or fasterwam")
    if cfg.EVALUATION.action_infer_mode not in ("first_frame", "one_pass_future_cache"):
        raise ValueError("Unsupported action_infer_mode")
    if cfg.model.name == "fastwam" and (cfg.lora_path or cfg.model.backbone != "fastwam" or cfg.EVALUATION.action_infer_mode != "first_frame"):
        raise ValueError("model=fastwam requires dense backbone + first_frame + merged ckpt")
    if cfg.EVALUATION.compile_action_infer:
        raise ValueError("compile_action_infer is not implemented; use false")
    if cfg.EVALUATION.reuse_seed_cache and benchmark != "robotwin":
        raise ValueError("reuse_seed_cache is supported only for RoboTwin")
    if not isinstance(cfg.EVALUATION.skip_get_obs_within_replan, bool):
        raise ValueError("skip_get_obs_within_replan must be a boolean")
    if cfg.EVALUATION.skip_get_obs_within_replan and benchmark != "robotwin":
        raise ValueError("skip_get_obs_within_replan is supported only for RoboTwin")
    for field in ("num_gpus", "max_tasks_per_gpu", "chunk_size"):
        if cfg.MULTIRUN[field] < 1:
            raise ValueError(f"MULTIRUN.{field} must be positive")
    for field in ("num_trials", "eval_num_episodes", "num_inference_steps", "replan_steps", "max_seed_attempts"):
        if cfg.EVALUATION[field] < 1:
            raise ValueError(f"EVALUATION.{field} must be positive")
    if cfg.EVALUATION.replan_steps > cfg.model.action_horizon:
        raise ValueError("replan_steps must not exceed model.action_horizon")
    if cfg.EVALUATION.num_steps_wait < 0 or (cfg.EVALUATION.max_steps is not None and cfg.EVALUATION.max_steps < 1):
        raise ValueError("Invalid episode horizon or settle steps")
    if cfg.seed < 0 or cfg.model.lora_weights not in ("ema", "student"):
        raise ValueError("seed must be nonnegative; lora_weights must be ema/student")
    if not 0 < cfg.MULTIRUN.task_sample_ratio <= 1:
        raise ValueError("task_sample_ratio must be in (0,1]")
    if benchmark == "robotwin" and cfg.EVALUATION.reuse_seed_cache and not cfg.EVALUATION.seed_cache_dir:
        cfg.EVALUATION.seed_cache_dir = os.environ.get("ROBOTWIN_SEED_DIR") or str(Path(cfg.EVALUATION.robotwin_root) / "my_seeds")
    # Resolve paths before any simulator changes the worker's working directory.
    for key in (
        "base_ckpt",
        "lora_path",
        "ckpt",
        "model.model_path",
        "EVALUATION.output_dir",
        "EVALUATION.dataset_stats_path",
        "EVALUATION.libero_root",
        "EVALUATION.robotwin_root",
        "EVALUATION.seed_cache_dir",
    ):
        value = OmegaConf.select(cfg, key)
        if value:
            OmegaConf.update(cfg, key, str(Path(os.path.expandvars(value)).expanduser().resolve()))
    return OmegaConf.to_container(cfg, resolve=True)
