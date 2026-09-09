"""Run one local pi0.5 policy episode in the selected LIBERO checkout."""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np

from .artifacts import atomic_save_numpy, atomic_write_json, atomic_write_video
from .libero_protocol import (
    LIBERO_BENCHMARKS,
    MAX_STEPS_BY_BENCHMARK,
    TaskSpec,
    configure_libero,
    create_environment,
    load_policy_config,
    load_task_init_states,
    run_episode,
)
from .openpi_runner import OpenPIPolicy

LOGGER = logging.getLogger(__name__)


def _output_path(path: Path, suffix: str, label: str) -> Path:
    path = path.expanduser().resolve()
    if path.suffix.lower() != suffix:
        raise ValueError(f"{label} must end in {suffix}: {path}")
    return path


def _task_spec(runtime, benchmark_name: str, task_id: int) -> TaskSpec:
    factories = runtime.benchmark_module.get_benchmark_dict()
    if benchmark_name not in factories:
        raise KeyError(f"LIBERO checkout does not provide benchmark {benchmark_name!r}")
    suite = factories[benchmark_name]()
    task_count = int(suite.n_tasks)
    if not 0 <= task_id < task_count:
        raise ValueError(f"task_id must be in [0, {task_count}), got {task_id}")
    task = suite.get_task(task_id)
    return TaskSpec(
        benchmark=benchmark_name,
        task_id=task_id,
        suite=suite,
        task=task,
        bddl_path=(Path(runtime.get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file).resolve(),
        init_states_path=(Path(runtime.get_libero_path("init_states")) / task.problem_folder / task.init_states_file).resolve(),
    )


def run_rollout(args: argparse.Namespace) -> dict:
    if args.actions_per_plan < 1:
        raise ValueError("--actions-per-plan must be positive")
    if args.num_steps_wait < 0:
        raise ValueError("--num-steps-wait must be non-negative")
    if args.render_size < 1 or args.fps < 1:
        raise ValueError("--render-size and --fps must be positive")
    if args.env_seed < 0 or args.policy_seed < 0:
        raise ValueError("--env-seed and --policy-seed must be non-negative")
    max_steps = int(args.max_steps if args.max_steps is not None else MAX_STEPS_BY_BENCHMARK[args.benchmark])
    if max_steps < 1:
        raise ValueError("--max-steps must be positive")

    video_path = _output_path(args.save_video_path, ".mp4", "save_video_path")
    action_path = _output_path(args.save_action_path, ".npy", "save_action_path")
    metrics_path = _output_path(args.save_metrics_path, ".json", "save_metrics_path")
    np.random.seed(args.env_seed)
    runtime = configure_libero(args.libero_root, args.libero_config_dir)
    spec = _task_spec(runtime, args.benchmark, args.task_id)
    if not spec.bddl_path.is_file():
        raise FileNotFoundError(f"LIBERO BDDL file is missing: {spec.bddl_path}")
    initial_states, init_states_loader = load_task_init_states(spec)
    if not 0 <= args.init_state_id < len(initial_states):
        raise ValueError(f"init_state_id must be in [0, {len(initial_states)}), got {args.init_state_id}")

    task_description = args.task_description.strip() or str(spec.task.language)
    env = create_environment(runtime, spec, args.render_size, args.env_seed)
    started = time.perf_counter()
    try:
        LOGGER.info("Loading local PyTorch OpenPI policy")
        policy = OpenPIPolicy(
            load_policy_config(
                args.config_json,
                args.model_path,
                seed=args.policy_seed,
            )
        )
        policy.reset()
        episode, frames, actions = run_episode(
            policy=policy,
            env=env,
            initial_state=initial_states[args.init_state_id],
            task_description=task_description,
            max_steps=max_steps,
            num_steps_wait=args.num_steps_wait,
            actions_per_plan=args.actions_per_plan,
            collect_frames=True,
        )
        rng_state_after = policy.export_rng_state()
    finally:
        env.close()

    artifact_errors: list[str] = []
    try:
        atomic_save_numpy(action_path, actions)
    except Exception as exc:
        artifact_errors.append(f"actions: {type(exc).__name__}: {exc}")
        LOGGER.exception("Failed to save single-rollout actions")
    try:
        atomic_write_video(video_path, frames, args.fps)
    except Exception as exc:
        artifact_errors.append(f"video: {type(exc).__name__}: {exc}")
        LOGGER.exception("Failed to save single-rollout video")

    metrics = {
        "schema_version": 1,
        "mode": "single_rollout",
        "benchmark": args.benchmark,
        "task_id": args.task_id,
        "task_name": str(spec.task.name),
        "task_description": task_description,
        "bddl_file": str(spec.bddl_path),
        "init_states_file": str(spec.init_states_path),
        "init_states_loader": init_states_loader,
        "init_state_id": args.init_state_id,
        "init_state_count": len(initial_states),
        "env_seed": args.env_seed,
        "policy_seed": args.policy_seed,
        "policy_rng_state_after": rng_state_after,
        "max_policy_steps": max_steps,
        "actions_per_plan": args.actions_per_plan,
        "video_frames": len(frames),
        "video_fps": args.fps,
        "video_path": str(video_path) if video_path.is_file() else None,
        "action_path": str(action_path) if action_path.is_file() else None,
        "artifact_errors": artifact_errors,
        "total_elapsed_seconds": time.perf_counter() - started,
        **episode,
    }
    # metrics.json is the final, atomic completion marker for the rollout.
    atomic_write_json(metrics_path, metrics)
    LOGGER.info("Single LIBERO rollout complete: success=%s metrics=%s", metrics["success"], metrics_path)
    return metrics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run one local pi0.5-LIBERO rollout")
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--config-json", type=Path, required=True)
    parser.add_argument("--libero-root", type=Path, required=True)
    parser.add_argument("--libero-config-dir", type=Path, required=True)
    parser.add_argument("--benchmark", choices=LIBERO_BENCHMARKS, default="libero_spatial")
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--init-state-id", type=int, default=0)
    parser.add_argument("--env-seed", type=int, default=7)
    parser.add_argument("--policy-seed", type=int, default=0)
    parser.add_argument("--task-description", default="")
    parser.add_argument("--actions-per-plan", type=int, default=5)
    parser.add_argument("--num-steps-wait", type=int, default=10)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--render-size", type=int, default=256)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--save-video-path", type=Path, required=True)
    parser.add_argument("--save-action-path", type=Path, required=True)
    parser.add_argument("--save-metrics-path", type=Path, required=True)
    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    metrics = run_rollout(build_parser().parse_args())
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
