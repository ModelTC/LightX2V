import os
from pathlib import Path

from omegaconf import OmegaConf

from simulator.sim.bench.backends import get_benchmark_backend
from simulator.sim.bench.policies.config import resolve_policy_config

ROOT = Path(__file__).resolve().parents[6]


def load_config(benchmark, argv):
    backend = get_benchmark_backend(benchmark)
    cfg = OmegaConf.merge(OmegaConf.load(ROOT / "configs/bench/robotics/eval.yaml"), backend.defaults(benchmark))
    cfg.benchmark = benchmark
    cfg.base_ckpt = os.environ.get("BASE_CKPT")
    cfg.lora_path = os.environ.get("LORA_PATH")
    cfg.ckpt = os.environ.get("CKPT_PATH")
    cfg.model.model_path = os.environ.get("WAN_MODEL_PATH")
    cfg.EVALUATION.dataset_stats_path = os.environ.get("DATASET_STATS_PATH")
    cfg.EVALUATION.output_dir = os.environ.get("OUT")
    OmegaConf.set_struct(cfg, True)
    OmegaConf.set_struct(cfg.model.options, False)
    for argument in argv:
        if "=" not in argument or argument.startswith("--"):
            raise ValueError(f"Expected key=value override (not --key=value), got {argument!r}")
        key, value = argument.split("=", 1)
        if key == "model":
            key = "model.name"
        if key == "task":
            if value != backend.LEGACY_TASK:
                raise ValueError(f"Unsupported legacy task alias: {value}")
            continue
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist([f"{key}={value}"]))

    backend.validate(cfg)
    # Resolve paths before any simulator changes the worker's working directory.
    for key in (
        "config_json",
        "base_ckpt",
        "lora_path",
        "ckpt",
        "model.model_path",
        "EVALUATION.output_dir",
        "EVALUATION.dataset_stats_path",
        *backend.PATH_FIELDS,
    ):
        value = OmegaConf.select(cfg, key)
        if value:
            OmegaConf.update(cfg, key, str(Path(os.path.expandvars(value)).expanduser().resolve()))
    resolve_policy_config(cfg, ROOT, backend)
    for field in ("num_gpus", "max_tasks_per_gpu", "chunk_size"):
        if cfg.MULTIRUN[field] < 1:
            raise ValueError(f"MULTIRUN.{field} must be positive")
    if cfg.EVALUATION.replan_steps < 1 or cfg.EVALUATION.replan_steps > cfg.model.action_horizon:
        raise ValueError("replan_steps must be positive and not exceed model.action_horizon")
    if cfg.EVALUATION.max_steps is not None and cfg.EVALUATION.max_steps < 1:
        raise ValueError("max_steps must be positive")
    if cfg.seed < 0:
        raise ValueError("seed must be nonnegative")
    if not 0 < cfg.MULTIRUN.task_sample_ratio <= 1:
        raise ValueError("task_sample_ratio must be in (0,1]")
    return OmegaConf.to_container(cfg, resolve=True)
