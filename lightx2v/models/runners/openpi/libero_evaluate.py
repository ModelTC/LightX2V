"""Evaluate the native PyTorch pi0.5 policy on LIBERO locally."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import logging
import os
import subprocess
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .libero_rollout import (
    LIBERO_BENCHMARKS,
    LIBERO_DUMMY_ACTION,
    MAX_STEPS_BY_BENCHMARK,
    _configure_libero,
    _rotate_rgb,
    _state_from_observation,
    _write_video,
)
from .openpi_runner import OpenPIPolicy

LOGGER = logging.getLogger(__name__)

OFFICIAL_BENCHMARKS = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
OFFICIAL_RESULTS = {
    "libero_spatial": {"successes": 494, "episodes": 500, "success_rate": 98.8},
    "libero_object": {"successes": 491, "episodes": 500, "success_rate": 98.2},
    "libero_goal": {"successes": 490, "episodes": 500, "success_rate": 98.0},
    "libero_10": {"successes": 462, "episodes": 500, "success_rate": 92.4},
    "average": 96.85,
}
VIDEO_POLICIES = ("all", "failures", "none")


@dataclass(frozen=True)
class EvaluationConfig:
    benchmarks: tuple[str, ...]
    task_ids: dict[str, tuple[int, ...] | None]
    num_trials_per_task: int
    env_seed: int
    policy_seed: int
    actions_per_plan: int
    num_steps_wait: int
    render_size: int
    video_fps: int
    video_policy: str
    save_actions: bool
    fail_fast: bool
    resume: bool
    max_steps: dict[str, int]
    libero_root: Path
    libero_config_dir: Path


@dataclass(frozen=True)
class TaskSpec:
    benchmark: str
    task_id: int
    task: Any
    bddl_path: Path
    init_states_path: Path


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    _atomic_write_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _atomic_save_actions(path: Path, actions: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp.npy")
    np.save(temporary, actions)
    temporary.replace(path)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_tree(root: Path, pattern: str = "*.py") -> dict[str, Any]:
    root = root.expanduser().resolve()
    files = sorted(path for path in root.rglob(pattern) if path.is_file())
    digest = hashlib.sha256()
    for path in files:
        relative = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
    return {"root": str(root), "file_count": len(files), "sha256": digest.hexdigest()}


def _model_artifact_fingerprints(args: argparse.Namespace) -> dict[str, dict[str, Any]]:
    config_path = args.config_json.expanduser().resolve()
    checkpoint_dir = args.model_path.expanduser().resolve()
    paths = {
        "checkpoint": checkpoint_dir / "model.safetensors",
        "model_config": config_path,
        "norm_stats": checkpoint_dir / "assets/physical-intelligence/libero/norm_stats.json",
        "tokenizer": checkpoint_dir / "assets/paligemma_tokenizer.model",
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Required OpenPI model artifacts are missing: {missing}")

    fingerprints = {}
    for label, path in paths.items():
        if label == "checkpoint":
            LOGGER.info("Computing checkpoint SHA-256 for reproducible suite-boundary resume: %s", path)
        fingerprints[label] = {
            "path": str(path),
            "size_bytes": path.stat().st_size,
            "sha256": _sha256_file(path),
        }
    return fingerprints


def _source_fingerprints(config: EvaluationConfig) -> dict[str, Any]:
    project_root = Path(__file__).resolve().parents[4]
    transformers_spec = importlib.util.find_spec("transformers")
    transformers_root = Path(transformers_spec.origin).resolve().parent
    return {
        "lightx2v_openpi": _sha256_tree(project_root / "lightx2v/models/networks/openpi"),
        "lightx2v_openpi_runners": _sha256_tree(project_root / "lightx2v/models/runners/openpi"),
        "libero_commit": _git_commit(config.libero_root),
        "transformers_runtime": _sha256_tree(transformers_root),
    }


def _task_input_fingerprints(task_specs: dict[str, list[TaskSpec]]) -> dict[str, dict[str, Any]]:
    fingerprints = {}
    for benchmark_name, specs in task_specs.items():
        for spec in specs:
            for kind, path in (("bddl", spec.bddl_path), ("init_states", spec.init_states_path)):
                if not path.is_file():
                    raise FileNotFoundError(f"LIBERO {kind} file is missing: {path}")
                key = f"{benchmark_name}/task_{spec.task_id:02d}/{kind}"
                fingerprints[key] = {
                    "path": str(path),
                    "size_bytes": path.stat().st_size,
                    "sha256": _sha256_file(path),
                }
    return fingerprints


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _numeric_runtime_fingerprint() -> dict[str, Any]:
    cuda_available = torch.cuda.is_available()
    gpu = None
    if cuda_available:
        properties = torch.cuda.get_device_properties(0)
        gpu = {
            "name": properties.name,
            "capability": list(torch.cuda.get_device_capability(0)),
            "total_memory_bytes": properties.total_memory,
        }
    try:
        driver_result = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            check=True,
            capture_output=True,
            text=True,
        )
        driver_version = driver_result.stdout.splitlines()[0].strip() or None
    except (OSError, subprocess.CalledProcessError, IndexError):
        driver_version = None
    return {
        "python": os.sys.version,
        "numpy": np.__version__,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "transformers": _package_version("transformers"),
        "mujoco": _package_version("mujoco"),
        "robosuite": _package_version("robosuite"),
        "pillow": _package_version("Pillow"),
        "imageio": _package_version("imageio"),
        "cuda_available": cuda_available,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "nvidia_driver": driver_version,
        "gpu": gpu,
    }


def _parse_csv(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


def _parse_integer_selection(value: str, label: str) -> tuple[int, ...] | None:
    if value == "all":
        return None

    values = []
    for segment in _parse_csv(value):
        if "-" in segment:
            start_text, end_text = segment.split("-", 1)
            start, end = int(start_text), int(end_text)
            if start > end:
                raise ValueError(f"{label} range starts after it ends: {segment!r}")
            values.extend(range(start, end + 1))
        else:
            values.append(int(segment))

    if not values:
        raise ValueError(f"{label} cannot be empty")
    if any(item < 0 for item in values):
        raise ValueError(f"{label} must contain only non-negative integers")
    if len(set(values)) != len(values):
        raise ValueError(f"{label} contains duplicate task IDs: {values}")
    return tuple(values)


def _load_evaluation_config(args: argparse.Namespace) -> EvaluationConfig:
    with args.eval_config.expanduser().resolve().open("r", encoding="utf-8") as handle:
        values = json.load(handle)

    benchmarks = _parse_csv(args.benchmarks) if args.benchmarks is not None else tuple(values["benchmarks"])
    if not benchmarks:
        raise ValueError("At least one LIBERO benchmark is required")
    unknown = [name for name in benchmarks if name not in LIBERO_BENCHMARKS]
    if unknown:
        raise ValueError(f"Unknown LIBERO benchmarks: {unknown}; expected one of {LIBERO_BENCHMARKS}")
    if len(set(benchmarks)) != len(benchmarks):
        raise ValueError(f"Duplicate LIBERO benchmarks are not allowed: {benchmarks}")

    task_value = args.task_ids if args.task_ids is not None else values["task_ids"]
    selection = _parse_integer_selection(task_value, "task_ids")
    task_ids = {name: selection for name in benchmarks}

    max_steps_value = values["max_steps"]
    max_steps = {name: max_steps_value[name] for name in benchmarks}
    if args.max_steps is not None:
        max_steps = {name: args.max_steps for name in benchmarks}

    num_trials = args.num_trials_per_task if args.num_trials_per_task is not None else values["num_trials_per_task"]
    video_policy = args.video_policy if args.video_policy is not None else values["video_policy"]
    resume = args.resume if args.resume is not None else values["resume"]
    fail_fast = args.fail_fast if args.fail_fast is not None else values["fail_fast"]
    save_actions = args.save_actions if args.save_actions is not None else values["save_actions"]

    config = EvaluationConfig(
        benchmarks=benchmarks,
        task_ids=task_ids,
        num_trials_per_task=num_trials,
        env_seed=values["env_seed"],
        policy_seed=values["policy_seed"],
        actions_per_plan=values["actions_per_plan"],
        num_steps_wait=values["num_steps_wait"],
        render_size=values["render_size"],
        video_fps=values["video_fps"],
        video_policy=video_policy,
        save_actions=save_actions,
        fail_fast=fail_fast,
        resume=resume,
        max_steps=max_steps,
        libero_root=args.libero_root.expanduser().resolve(),
        libero_config_dir=args.libero_config_dir.expanduser().resolve(),
    )
    if config.num_trials_per_task < 1:
        raise ValueError("num_trials_per_task must be positive")
    if config.actions_per_plan < 1:
        raise ValueError("actions_per_plan must be positive")
    if config.num_steps_wait < 0:
        raise ValueError("num_steps_wait must be non-negative")
    if config.policy_seed < 0:
        raise ValueError("policy_seed must be non-negative for reproducible per-suite sampling")
    if config.render_size < 1 or config.video_fps < 1:
        raise ValueError("render_size and video_fps must be positive")
    if config.video_policy not in VIDEO_POLICIES:
        raise ValueError(f"video_policy must be one of {VIDEO_POLICIES}, got {config.video_policy!r}")
    if any(limit < 1 for limit in config.max_steps.values()):
        raise ValueError("Every max_steps value must be positive")
    return config


def _is_official_protocol(config: EvaluationConfig) -> bool:
    return (
        all(name in OFFICIAL_BENCHMARKS for name in config.benchmarks)
        and all(config.task_ids[name] is None for name in config.benchmarks)
        and config.num_trials_per_task == 50
        and config.env_seed == 7
        and config.policy_seed == 0
        and config.actions_per_plan == 5
        and config.num_steps_wait == 10
        and config.render_size == 256
        and all(config.max_steps[name] == MAX_STEPS_BY_BENCHMARK[name] for name in config.benchmarks)
    )


def _resolved_config_payload(
    config: EvaluationConfig,
    args: argparse.Namespace,
    artifact_fingerprints: dict[str, dict[str, Any]],
    task_input_fingerprints: dict[str, dict[str, Any]],
    source_fingerprints: dict[str, Any],
    numeric_runtime: dict[str, Any],
) -> dict[str, Any]:
    official_protocol = _is_official_protocol(config)
    return {
        "schema_version": 2,
        "protocol_name": "official_pi05_libero" if official_protocol else "custom_pi05_libero",
        "official_protocol": official_protocol,
        "model_path": str(args.model_path.expanduser().resolve()),
        "model_config": str(args.config_json.expanduser().resolve()),
        "model_artifacts": artifact_fingerprints,
        "task_inputs": task_input_fingerprints,
        "source_fingerprints": source_fingerprints,
        "numeric_runtime": numeric_runtime,
        "benchmarks": list(config.benchmarks),
        "task_ids": {name: "all" if ids is None else list(ids) for name, ids in config.task_ids.items()},
        "num_trials_per_task": config.num_trials_per_task,
        "env_seed": config.env_seed,
        "policy_seed": config.policy_seed,
        "policy_rng_scope": "per_suite_continuous",
        "actions_per_plan": config.actions_per_plan,
        "num_steps_wait": config.num_steps_wait,
        "render_size": config.render_size,
        "video_fps": config.video_fps,
        "video_policy": config.video_policy,
        "save_actions": config.save_actions,
        "fail_fast": config.fail_fast,
        "max_steps": config.max_steps,
        "libero_root": str(config.libero_root),
        "libero_config_dir": str(config.libero_config_dir),
    }


def _protocol_hash(payload: dict[str, Any]) -> str:
    # Paths are useful provenance in resolved_eval_config.json, but they do not
    # change the evaluation protocol.  Excluding only known location fields
    # keeps suite-boundary resume valid after relocating identical artifacts.
    identity = json.loads(json.dumps(payload))
    for key in ("model_path", "model_config", "libero_root", "libero_config_dir"):
        identity.pop(key, None)
    for section in ("model_artifacts", "task_inputs"):
        for fingerprint in identity.get(section, {}).values():
            fingerprint.pop("path", None)
    for key in ("lightx2v_openpi", "lightx2v_openpi_runners", "transformers_runtime"):
        identity["source_fingerprints"][key].pop("root", None)
    encoded = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _load_policy_config(args: argparse.Namespace, config: EvaluationConfig) -> dict[str, Any]:
    config_path = args.config_json.expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        values = json.load(handle)
    values.update(
        {
            "model_cls": "openpi",
            "task": "i2va",
            "model_path": str(args.model_path.expanduser().resolve()),
            "config_json": str(config_path),
            "seed": config.policy_seed,
            "actions_per_plan": config.actions_per_plan,
        }
    )
    return values


def _task_specs(benchmark_module: Any, get_libero_path: Any, config: EvaluationConfig) -> dict[str, list[TaskSpec]]:
    factories = benchmark_module.get_benchmark_dict()
    specs: dict[str, list[TaskSpec]] = {}
    for benchmark_name in config.benchmarks:
        suite = factories[benchmark_name]()
        task_count = suite.get_num_tasks()
        selected_ids = config.task_ids[benchmark_name]
        task_ids = tuple(range(task_count)) if selected_ids is None else selected_ids
        invalid = [task_id for task_id in task_ids if not 0 <= task_id < task_count]
        if invalid:
            raise ValueError(f"Task IDs {invalid} are outside [0, {task_count}) for {benchmark_name}")
        benchmark_specs = []
        for task_id in task_ids:
            task = suite.get_task(task_id)
            benchmark_specs.append(
                TaskSpec(
                    benchmark=benchmark_name,
                    task_id=task_id,
                    task=task,
                    bddl_path=Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file,
                    init_states_path=Path(get_libero_path("init_states")) / task.problem_folder / task.init_states_file,
                )
            )
        specs[benchmark_name] = benchmark_specs
    return specs


def _episode_directory(output_dir: Path, benchmark: str, task_id: int, init_state_id: int) -> Path:
    return output_dir / "episodes" / benchmark / f"task_{task_id:02d}" / f"init_{init_state_id:02d}"


def _episode_metrics_path(output_dir: Path, benchmark: str, task_id: int, init_state_id: int) -> Path:
    return _episode_directory(output_dir, benchmark, task_id, init_state_id) / "metrics.json"


def _read_existing_records(
    output_dir: Path,
    task_specs: dict[str, list[TaskSpec]],
    config: EvaluationConfig,
    protocol_hash: str,
) -> tuple[list[dict[str, Any]], set[str]]:
    records = []
    complete_benchmarks = set()
    for benchmark_name in config.benchmarks:
        benchmark_records = []
        expected = len(task_specs[benchmark_name]) * config.num_trials_per_task
        for spec in task_specs[benchmark_name]:
            for init_state_id in range(config.num_trials_per_task):
                path = _episode_metrics_path(output_dir, benchmark_name, spec.task_id, init_state_id)
                if not path.is_file():
                    continue
                with path.open("r", encoding="utf-8") as handle:
                    record = json.load(handle)
                if record.get("protocol_hash") != protocol_hash:
                    raise RuntimeError(f"Existing episode uses a different evaluation protocol: {path}")
                benchmark_records.append(record)
        if benchmark_records and len(benchmark_records) != expected:
            raise RuntimeError(
                f"Strict resume cannot continue a partially completed suite: {benchmark_name} has {len(benchmark_records)}/{expected} episode records. Run that suite again in a new output directory."
            )
        if len(benchmark_records) == expected:
            complete_benchmarks.add(benchmark_name)
        records.extend(benchmark_records)
    return records, complete_benchmarks


def _run_episode(
    policy: OpenPIPolicy,
    env: Any,
    initial_state: Any,
    task_description: str,
    max_steps: int,
    num_steps_wait: int,
    collect_frames: bool,
) -> tuple[dict[str, Any], list[np.ndarray], np.ndarray]:
    frames: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    success = False
    warmup_steps = 0
    action_chunk_calls = 0
    error_type = None
    error_message = None
    error_traceback = None
    started = time.perf_counter()

    policy.clear_action_queue()
    # Environment reset/init failures are infrastructure errors, not failed
    # policy rollouts, so they intentionally abort instead of polluting the
    # benchmark denominator with 50 repeated setup failures.
    env.reset()
    observation = env.set_init_state(initial_state)
    try:
        # Match OpenPI's official evaluator: warmup done values are ignored and
        # only policy-phase done=True is considered task success.
        for _ in range(num_steps_wait):
            observation, _reward, _done, _info = env.step(LIBERO_DUMMY_ACTION.tolist())
            warmup_steps += 1

        for _step in range(max_steps):
            agentview = _rotate_rgb(observation, "agentview_image")
            if collect_frames:
                frames.append(agentview)
            images = {
                "agentview": agentview,
                "wrist": _rotate_rgb(observation, "robot0_eye_in_hand_image"),
            }
            if not policy.pending_actions:
                action_chunk_calls += 1
            action = policy.next_action(
                images=images,
                state=_state_from_observation(observation),
                task_description=task_description,
            )
            actions.append(action.copy())
            observation, _reward, done, _info = env.step(action.tolist())
            if done:
                success = True
                break
    except Exception as exc:  # Official evaluation counts rollout exceptions as failed episodes.
        if isinstance(exc, (MemoryError, torch.cuda.OutOfMemoryError)) or "CUDA out of memory" in str(exc):
            raise
        error_type = type(exc).__name__
        error_message = str(exc)
        error_traceback = traceback.format_exc(limit=20)
        LOGGER.exception("LIBERO episode failed with an exception")

    pending_actions_discarded = len(policy.pending_actions)
    policy.clear_action_queue()
    action_array = np.asarray(actions, dtype=np.float32).reshape(-1, 7)
    if error_type is not None:
        termination_reason = "exception"
    elif success:
        termination_reason = "success"
    else:
        termination_reason = "step_limit"
    episode = {
        "success": success,
        "termination_reason": termination_reason,
        "warmup_steps_requested": num_steps_wait,
        "warmup_steps_executed": warmup_steps,
        "policy_steps": int(action_array.shape[0]),
        "total_env_steps": warmup_steps + int(action_array.shape[0]),
        "action_chunk_calls": action_chunk_calls,
        "pending_actions_discarded": pending_actions_discarded,
        "elapsed_seconds": time.perf_counter() - started,
        "exception_type": error_type,
        "exception_message": error_message,
        "exception_traceback": error_traceback,
    }
    return episode, frames, action_array


def _save_episode_artifacts(
    output_dir: Path,
    record: dict[str, Any],
    frames: list[np.ndarray],
    actions: np.ndarray,
    config: EvaluationConfig,
) -> dict[str, Any]:
    episode_dir = _episode_directory(output_dir, record["benchmark"], record["task_id"], record["init_state_id"])
    episode_dir.mkdir(parents=True, exist_ok=True)
    artifact_errors = []

    action_path = None
    if config.save_actions:
        action_path = episode_dir / "actions.npy"
        try:
            _atomic_save_actions(action_path, actions)
        except Exception as exc:
            artifact_errors.append(f"actions: {type(exc).__name__}: {exc}")
            LOGGER.exception("Failed to save episode actions")

    save_video = config.video_policy == "all" or (config.video_policy == "failures" and not record["success"])
    video_path = None
    if save_video and frames:
        video_path = episode_dir / "rollout.mp4"
        try:
            _write_video(video_path, frames, config.video_fps)
        except Exception as exc:
            artifact_errors.append(f"video: {type(exc).__name__}: {exc}")
            LOGGER.exception("Failed to save episode video")
    elif save_video:
        artifact_errors.append("video: no policy frames were produced")

    record["action_path"] = str(action_path) if action_path is not None and action_path.is_file() else None
    record["video_path"] = str(video_path) if video_path is not None and video_path.is_file() else None
    record["artifact_errors"] = artifact_errors
    _atomic_write_json(episode_dir / "metrics.json", record)
    return record


def _aggregate(records: list[dict[str, Any]], task_specs: dict[str, list[TaskSpec]], config: EvaluationConfig, protocol_hash: str) -> dict[str, Any]:
    per_benchmark = {}
    per_task = {}
    expected_total = 0
    official_protocol = _is_official_protocol(config)
    for benchmark_name in config.benchmarks:
        specs = task_specs[benchmark_name]
        expected = len(specs) * config.num_trials_per_task
        expected_total += expected
        suite_records = [record for record in records if record["benchmark"] == benchmark_name]
        successes = sum(bool(record["success"]) for record in suite_records)
        errors = sum(record["termination_reason"] == "exception" for record in suite_records)
        completed = len(suite_records)
        rate = successes / expected * 100.0 if completed == expected else None
        per_benchmark[benchmark_name] = {
            "expected": expected,
            "completed": completed,
            "successes": successes,
            "failures": completed - successes,
            "errors": errors,
            "success_rate": rate,
            "success_rate_completed": successes / completed * 100.0 if completed else None,
            "official_reference": OFFICIAL_RESULTS.get(benchmark_name),
            "official_comparable": official_protocol,
            "delta_percentage_points": rate - OFFICIAL_RESULTS[benchmark_name]["success_rate"] if official_protocol and rate is not None and benchmark_name in OFFICIAL_RESULTS else None,
        }
        for spec in specs:
            task_records = [record for record in suite_records if record["task_id"] == spec.task_id]
            task_successes = sum(bool(record["success"]) for record in task_records)
            task_completed = len(task_records)
            key = f"{benchmark_name}/task_{spec.task_id:02d}"
            per_task[key] = {
                "task_name": str(spec.task.name),
                "task_description": str(spec.task.language),
                "expected": config.num_trials_per_task,
                "completed": task_completed,
                "successes": task_successes,
                "failures": task_completed - task_successes,
                "errors": sum(record["termination_reason"] == "exception" for record in task_records),
                "success_rate": task_successes / config.num_trials_per_task * 100.0 if task_completed == config.num_trials_per_task else None,
            }

    completed_total = len(records)
    success_total = sum(bool(record["success"]) for record in records)
    errors_total = sum(record["termination_reason"] == "exception" for record in records)
    artifact_errors_total = sum(len(record.get("artifact_errors", [])) for record in records)
    complete = completed_total == expected_total
    suite_rates = [entry["success_rate"] for entry in per_benchmark.values()]
    mean_suite_rate = sum(suite_rates) / len(suite_rates) if complete and suite_rates and all(rate is not None for rate in suite_rates) else None
    return {
        "schema_version": 1,
        "status": "complete_with_errors" if complete and (errors_total or artifact_errors_total) else "complete" if complete else "in_progress",
        "protocol_hash": protocol_hash,
        "official_protocol": official_protocol,
        "full_official_table": official_protocol and config.benchmarks == OFFICIAL_BENCHMARKS,
        "expected_episodes": expected_total,
        "completed_episodes": completed_total,
        "successes": success_total,
        "failures": completed_total - success_total,
        "errors": errors_total,
        "artifact_errors": artifact_errors_total,
        "success_rate": success_total / expected_total * 100.0 if complete else None,
        "success_rate_completed": success_total / completed_total * 100.0 if completed_total else None,
        "mean_suite_success_rate": mean_suite_rate,
        "official_reference": {"comparable": official_protocol, "results": OFFICIAL_RESULTS},
        "per_benchmark": per_benchmark,
        "per_task": per_task,
        "updated_at_unix": time.time(),
    }


def _write_aggregate_outputs(output_dir: Path, records: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    ordered = sorted(records, key=lambda item: (item["benchmark_index"], item["task_id"], item["init_state_id"]))
    lines = "".join(json.dumps(record, sort_keys=True) + "\n" for record in ordered)
    _atomic_write_text(output_dir / "episodes.jsonl", lines)
    _atomic_write_json(output_dir / "summary.json", summary)


def _git_commit(path: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def run_evaluation(args: argparse.Namespace) -> dict[str, Any]:
    config = _load_evaluation_config(args)
    output_dir = args.output_dir.expanduser().resolve()
    artifact_fingerprints = _model_artifact_fingerprints(args)
    np.random.seed(config.env_seed)
    benchmark_module, get_libero_path, env_type = _configure_libero(config.libero_root, config.libero_config_dir)
    task_specs = _task_specs(benchmark_module, get_libero_path, config)
    task_input_fingerprints = _task_input_fingerprints(task_specs)
    source_fingerprints = _source_fingerprints(config)
    numeric_runtime = _numeric_runtime_fingerprint()
    resolved_payload = _resolved_config_payload(config, args, artifact_fingerprints, task_input_fingerprints, source_fingerprints, numeric_runtime)
    protocol_hash = _protocol_hash(resolved_payload)
    resolved_payload["protocol_hash"] = protocol_hash

    output_dir.mkdir(parents=True, exist_ok=True)
    resolved_path = output_dir / "resolved_eval_config.json"
    if resolved_path.is_file():
        with resolved_path.open("r", encoding="utf-8") as handle:
            existing_config = json.load(handle)
        if existing_config.get("protocol_hash") != protocol_hash:
            existing_episode = next((output_dir / "episodes").rglob("metrics.json"), None) if (output_dir / "episodes").is_dir() else None
            if existing_episode is not None:
                raise RuntimeError(f"Evaluation output contains episode records from a different protocol: {existing_episode}")
            LOGGER.warning("Replacing an unused resolved evaluation config with no episode records: %s", resolved_path)
            _atomic_write_json(resolved_path, resolved_payload)
    else:
        _atomic_write_json(resolved_path, resolved_payload)

    records, complete_benchmarks = _read_existing_records(output_dir, task_specs, config, protocol_hash)
    if records and not config.resume:
        raise FileExistsError(f"Evaluation output already contains {len(records)} episode records. Use a new OPENPI_EVAL_OUTPUT_DIR, or enable resume after a complete suite boundary.")

    summary = _aggregate(records, task_specs, config, protocol_hash)
    _write_aggregate_outputs(output_dir, records, summary)
    pending_benchmarks = [name for name in config.benchmarks if name not in complete_benchmarks]
    if not pending_benchmarks:
        LOGGER.info("All requested LIBERO suites are already complete: %s", output_dir)
        return summary

    LOGGER.info("Loading the local PyTorch pi0.5-LIBERO policy once for %d pending suites", len(pending_benchmarks))
    policy = OpenPIPolicy(_load_policy_config(args, config))
    global_index = len(records)
    try:
        for benchmark_index, benchmark_name in enumerate(config.benchmarks):
            if benchmark_name in complete_benchmarks:
                LOGGER.info("Strict resume: skipping complete suite %s", benchmark_name)
                continue

            # Official results are produced by four independent suite runs.
            # Reset once at the suite boundary, never at episode boundaries.
            np.random.seed(config.env_seed)
            policy.clear_action_queue()
            policy.reset_rng()
            LOGGER.info("Starting suite %s", benchmark_name)
            for spec in task_specs[benchmark_name]:
                initial_states = torch.load(spec.init_states_path, map_location="cpu", weights_only=False)
                if config.num_trials_per_task > len(initial_states):
                    raise ValueError(f"{benchmark_name} task {spec.task_id} has {len(initial_states)} init states, but {config.num_trials_per_task} trials were requested")
                env = env_type(
                    bddl_file_name=str(spec.bddl_path),
                    camera_heights=config.render_size,
                    camera_widths=config.render_size,
                )
                try:
                    env.seed(config.env_seed)
                    for init_state_id in range(config.num_trials_per_task):
                        collect_frames = config.video_policy != "none"
                        episode, frames, actions = _run_episode(
                            policy=policy,
                            env=env,
                            initial_state=initial_states[init_state_id],
                            task_description=str(spec.task.language),
                            max_steps=config.max_steps[benchmark_name],
                            num_steps_wait=config.num_steps_wait,
                            collect_frames=collect_frames,
                        )
                        record = {
                            "schema_version": 1,
                            "protocol_hash": protocol_hash,
                            "global_episode_index": global_index,
                            "benchmark_index": benchmark_index,
                            "benchmark": benchmark_name,
                            "task_id": spec.task_id,
                            "task_name": str(spec.task.name),
                            "task_description": str(spec.task.language),
                            "bddl_file": str(spec.bddl_path),
                            "init_states_file": str(spec.init_states_path),
                            "init_state_id": init_state_id,
                            "env_seed": config.env_seed,
                            "policy_seed": config.policy_seed,
                            "policy_rng_scope": "per_suite_continuous",
                            "max_policy_steps": config.max_steps[benchmark_name],
                            "actions_per_plan": config.actions_per_plan,
                            **episode,
                        }
                        record = _save_episode_artifacts(output_dir, record, frames, actions, config)
                        records.append(record)
                        global_index += 1
                        summary = _aggregate(records, task_specs, config, protocol_hash)
                        _write_aggregate_outputs(output_dir, records, summary)
                        LOGGER.info(
                            "suite=%s task=%02d init=%02d success=%s steps=%d completed=%d/%d",
                            benchmark_name,
                            spec.task_id,
                            init_state_id,
                            record["success"],
                            record["policy_steps"],
                            summary["completed_episodes"],
                            summary["expected_episodes"],
                        )
                        if record["termination_reason"] == "exception" and config.fail_fast:
                            raise RuntimeError(f"Fail-fast: {benchmark_name} task {spec.task_id} init {init_state_id} failed: {record['exception_type']}: {record['exception_message']}")
                finally:
                    env.close()
    finally:
        policy.close()

    summary = _aggregate(records, task_specs, config, protocol_hash)
    _write_aggregate_outputs(output_dir, records, summary)
    LOGGER.info(
        "LIBERO evaluation finished: successes=%d/%d, success_rate=%s, mean_suite_success_rate=%s",
        summary["successes"],
        summary["expected_episodes"],
        summary["success_rate"],
        summary["mean_suite_success_rate"],
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run local quantitative pi0.5-LIBERO evaluation")
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--config-json", type=Path, required=True)
    parser.add_argument("--eval-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--libero-root", type=Path, required=True)
    parser.add_argument("--libero-config-dir", type=Path, required=True)
    parser.add_argument("--benchmarks")
    parser.add_argument("--task-ids")
    parser.add_argument("--num-trials-per-task", type=int)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--video-policy", choices=VIDEO_POLICIES)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--fail-fast", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--save-actions", action=argparse.BooleanOptionalAction, default=None)
    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    args = build_parser().parse_args()
    summary = run_evaluation(args)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
