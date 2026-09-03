"""Official pi0.5-LIBERO protocol shared by rollout and quantitative evaluation."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import logging
import math
import os
import sys
import time
import traceback
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .artifacts import EpisodeKey, atomic_write_json, atomic_write_text

LOGGER = logging.getLogger(__name__)

LIBERO_BENCHMARKS = ("libero_spatial", "libero_object", "libero_goal", "libero_10", "libero_90")
OFFICIAL_BENCHMARKS = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
MAX_STEPS_BY_BENCHMARK = {
    "libero_spatial": 220,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
    "libero_90": 400,
}
OFFICIAL_RESULTS = {
    "libero_spatial": {"successes": 494, "episodes": 500, "success_rate": 98.8},
    "libero_object": {"successes": 491, "episodes": 500, "success_rate": 98.2},
    "libero_goal": {"successes": 490, "episodes": 500, "success_rate": 98.0},
    "libero_10": {"successes": 462, "episodes": 500, "success_rate": 92.4},
    "average": 96.85,
}
VIDEO_POLICIES = ("all", "failures", "none")
LIBERO_DUMMY_ACTION = np.asarray([0.0] * 6 + [-1.0], dtype=np.float32)
POLICY_IMAGE_SIZE = 224
TASK_INPUTS_MANIFEST_FILENAME = "task_inputs_manifest.json"


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
class LiberoRuntime:
    benchmark_module: Any
    get_libero_path: Any
    env_type: Any


@dataclass(frozen=True)
class TaskSpec:
    benchmark: str
    task_id: int
    suite: Any
    task: Any
    bddl_path: Path
    init_states_path: Path


def _parse_csv(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in str(value).split(",") if item.strip())


def parse_integer_selection(value: Any, label: str) -> tuple[int, ...] | None:
    if value is None or value == "all":
        return None
    if isinstance(value, (list, tuple)):
        values = [int(item) for item in value]
    else:
        values = []
        for segment in _parse_csv(str(value)):
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
        raise ValueError(f"{label} contains duplicate IDs: {values}")
    return tuple(values)


def load_evaluation_config(args: argparse.Namespace) -> EvaluationConfig:
    eval_config_path = Path(args.eval_config).expanduser().resolve()
    with eval_config_path.open("r", encoding="utf-8") as handle:
        values = json.load(handle)
    if not isinstance(values, dict):
        raise ValueError(f"OpenPI evaluation config must be a JSON object: {eval_config_path}")

    benchmark_value = args.benchmarks if args.benchmarks is not None else values.get("benchmarks", OFFICIAL_BENCHMARKS)
    benchmarks = _parse_csv(benchmark_value) if isinstance(benchmark_value, str) else tuple(str(item) for item in benchmark_value)
    if not benchmarks:
        raise ValueError("At least one LIBERO benchmark is required")
    unknown = [name for name in benchmarks if name not in LIBERO_BENCHMARKS]
    if unknown:
        raise ValueError(f"Unknown LIBERO benchmarks: {unknown}; expected one of {LIBERO_BENCHMARKS}")
    if len(set(benchmarks)) != len(benchmarks):
        raise ValueError(f"Duplicate LIBERO benchmarks are not allowed: {benchmarks}")

    raw_task_ids: Any = args.task_ids if args.task_ids is not None else values.get("task_ids", "all")
    if isinstance(raw_task_ids, dict):
        task_ids = {name: parse_integer_selection(raw_task_ids.get(name, "all"), f"task_ids.{name}") for name in benchmarks}
    else:
        selection = parse_integer_selection(raw_task_ids, "task_ids")
        task_ids = {name: selection for name in benchmarks}

    raw_max_steps = values.get("max_steps", MAX_STEPS_BY_BENCHMARK)
    if args.max_steps is not None:
        max_steps = {name: int(args.max_steps) for name in benchmarks}
    elif isinstance(raw_max_steps, dict):
        max_steps = {name: int(raw_max_steps[name]) for name in benchmarks}
    else:
        max_steps = {name: int(raw_max_steps) for name in benchmarks}

    def overridden(argument_name: str, config_name: str, default: Any) -> Any:
        argument = getattr(args, argument_name)
        return argument if argument is not None else values.get(config_name, default)

    config = EvaluationConfig(
        benchmarks=benchmarks,
        task_ids=task_ids,
        num_trials_per_task=int(overridden("num_trials_per_task", "num_trials_per_task", 50)),
        env_seed=int(values.get("env_seed", 7)),
        policy_seed=int(values.get("policy_seed", 0)),
        actions_per_plan=int(values.get("actions_per_plan", 5)),
        num_steps_wait=int(values.get("num_steps_wait", 10)),
        render_size=int(values.get("render_size", 256)),
        video_fps=int(values.get("video_fps", 10)),
        video_policy=str(overridden("video_policy", "video_policy", "none")),
        save_actions=bool(overridden("save_actions", "save_actions", False)),
        fail_fast=bool(overridden("fail_fast", "fail_fast", False)),
        resume=bool(overridden("resume", "resume", True)),
        max_steps=max_steps,
        libero_root=Path(args.libero_root).expanduser().resolve(),
        libero_config_dir=Path(args.libero_config_dir).expanduser().resolve(),
    )
    if config.num_trials_per_task < 1:
        raise ValueError("num_trials_per_task must be positive")
    if config.actions_per_plan < 1:
        raise ValueError("actions_per_plan must be positive")
    if config.num_steps_wait < 0:
        raise ValueError("num_steps_wait must be non-negative")
    if config.env_seed < 0 or config.policy_seed < 0:
        raise ValueError("env_seed and policy_seed must be non-negative")
    if config.render_size < 1 or config.video_fps < 1:
        raise ValueError("render_size and video_fps must be positive")
    if config.video_policy not in VIDEO_POLICIES:
        raise ValueError(f"video_policy must be one of {VIDEO_POLICIES}, got {config.video_policy!r}")
    if any(limit < 1 for limit in config.max_steps.values()):
        raise ValueError("Every max_steps value must be positive")
    return config


def is_official_protocol(config: EvaluationConfig) -> bool:
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


def _file_descriptor(path: Path) -> dict[str, Any]:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    stat = path.stat()
    return {"path": str(path), "size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(dict(payload), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _sha256_file(path: Path) -> tuple[str, int]:
    """Hash one immutable view of a file and reject concurrent replacement."""
    path = path.expanduser().resolve()
    before = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    after = path.stat()
    before_identity = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    after_identity = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    if before_identity != after_identity:
        raise RuntimeError(f"LIBERO task input changed while it was being hashed: {path}")
    return digest.hexdigest(), after.st_size


def _manifest_path(path: Path, libero_root: Path) -> str:
    path = path.expanduser().resolve()
    try:
        return str(path.relative_to(libero_root.expanduser().resolve()))
    except ValueError:
        return str(path)


def _file_manifest_entry(spec: TaskSpec, kind: str, path: Path, libero_root: Path) -> dict[str, Any]:
    content_sha256, size_bytes = _sha256_file(path)
    return {
        "benchmark": spec.benchmark,
        "task_id": spec.task_id,
        "kind": kind,
        "source": "file",
        "path": _manifest_path(path, libero_root),
        "size_bytes": size_bytes,
        "content_sha256": content_sha256,
    }


def _semantic_init_states_manifest_entry(spec: TaskSpec, libero_root: Path) -> dict[str, Any]:
    states, loader = load_task_init_states(spec)
    metadata = {
        "dtype": states.dtype.str,
        "shape": list(states.shape),
    }
    digest = hashlib.sha256()
    digest.update(json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode("utf-8"))
    digest.update(states.tobytes(order="C"))
    return {
        "benchmark": spec.benchmark,
        "task_id": spec.task_id,
        "kind": "init_states",
        "source": loader,
        "path": _manifest_path(spec.init_states_path, libero_root),
        "size_bytes": states.nbytes,
        "dtype": metadata["dtype"],
        "shape": metadata["shape"],
        "content_sha256": digest.hexdigest(),
    }


def build_task_inputs_manifest(
    task_specs: Mapping[str, list[TaskSpec]],
    config: EvaluationConfig,
) -> dict[str, Any]:
    """Fingerprint the selected BDDL and initial-state content."""
    entries: list[dict[str, Any]] = []
    for benchmark in config.benchmarks:
        for spec in task_specs[benchmark]:
            entries.append(_file_manifest_entry(spec, "bddl", spec.bddl_path, config.libero_root))
            if spec.init_states_path.is_file():
                entries.append(_file_manifest_entry(spec, "init_states", spec.init_states_path, config.libero_root))
            else:
                entries.append(_semantic_init_states_manifest_entry(spec, config.libero_root))

    entries.sort(key=lambda entry: (str(entry["benchmark"]), int(entry["task_id"]), str(entry["kind"])))
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "hash_algorithm": "sha256",
        "scope": "selected_libero_task_inputs",
        "task_count": sum(len(specs) for specs in task_specs.values()),
        "input_count": len(entries),
        "entries": entries,
    }
    manifest["manifest_sha256"] = _canonical_sha256(manifest)
    return manifest


def _validate_legacy_record_input_paths(
    records: Mapping[EpisodeKey, Mapping[str, Any]],
    task_specs: Mapping[str, list[TaskSpec]],
) -> None:
    specs_by_task = {(spec.benchmark, spec.task_id): spec for specs in task_specs.values() for spec in specs}
    for key, record in records.items():
        if record.get("schema_version") != 1:
            raise RuntimeError(f"Cannot adopt a missing task-input manifest for non-legacy episode {key}: schema_version={record.get('schema_version')!r}")
        spec = specs_by_task.get(key[:2])
        if spec is None:
            raise RuntimeError(f"Cannot match legacy episode {key} to a selected LIBERO task")
        for field, expected in (("bddl_file", spec.bddl_path), ("init_states_file", spec.init_states_path)):
            recorded = record.get(field)
            if not recorded or Path(str(recorded)).expanduser().resolve() != expected:
                raise RuntimeError(f"Cannot adopt task-input manifest: legacy episode {key} has {field}={recorded!r}, expected {expected}")
        expected_loader = "direct_file" if spec.init_states_path.is_file() else "benchmark_api"
        if record.get("init_states_loader") != expected_loader:
            raise RuntimeError(f"Cannot adopt task-input manifest: legacy episode {key} has init_states_loader={record.get('init_states_loader')!r}, expected {expected_loader!r}")


def ensure_task_inputs_manifest(
    output_dir: Path,
    expected_manifest: Mapping[str, Any],
    records: Mapping[EpisodeKey, Mapping[str, Any]],
    task_specs: Mapping[str, list[TaskSpec]],
) -> Path:
    """Create, verify, or safely adopt the task-input sidecar.

    The sidecar deliberately does not participate in ``protocol_id`` so that
    schema-1 evaluations started before content fingerprints existed remain
    resumable. Once present, the complete canonical manifest is compared on
    every startup.
    """
    manifest_path = Path(output_dir) / TASK_INPUTS_MANIFEST_FILENAME
    expected = dict(expected_manifest)
    if manifest_path.exists():
        if not manifest_path.is_file():
            raise RuntimeError(f"LIBERO task-input manifest is not a file: {manifest_path}")
        try:
            existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"Cannot read LIBERO task-input manifest: {manifest_path}") from exc
        if not isinstance(existing, dict):
            raise RuntimeError(f"LIBERO task-input manifest must contain a JSON object: {manifest_path}")
        if existing != expected:
            raise RuntimeError(
                "LIBERO BDDL/init-state content differs from the committed evaluation manifest: "
                f"{manifest_path} (existing={existing.get('manifest_sha256')!r}, "
                f"current={expected.get('manifest_sha256')!r})"
            )
        return manifest_path

    if records:
        _validate_legacy_record_input_paths(records, task_specs)
        LOGGER.warning(
            "Adopting %s for %d legacy schema-1 episodes after validating their recorded BDDL/init-state paths. "
            "The historical episodes predate content fingerprints and therefore cannot be verified retrospectively.",
            manifest_path,
            len(records),
        )
    atomic_write_json(manifest_path, expected)
    return manifest_path


def resolved_protocol(
    config: EvaluationConfig,
    *,
    model_path: Path,
    config_json: Path,
) -> tuple[dict[str, Any], str]:
    """Build a cheap protocol identity without hashing model or source trees."""
    model_path = model_path.expanduser().resolve()
    config_json = config_json.expanduser().resolve()
    artifacts = {
        "checkpoint": _file_descriptor(model_path / "model.safetensors"),
        "model_config": _file_descriptor(config_json),
        "norm_stats": _file_descriptor(model_path / "assets/physical-intelligence/libero/norm_stats.json"),
        "tokenizer": _file_descriptor(model_path / "assets/paligemma_tokenizer.model"),
    }
    protocol_fields = {
        "schema_version": 1,
        "model_artifacts": artifacts,
        "benchmarks": list(config.benchmarks),
        "task_ids": {name: "all" if ids is None else list(ids) for name, ids in config.task_ids.items()},
        "num_trials_per_task": config.num_trials_per_task,
        "env_seed": config.env_seed,
        "policy_seed": config.policy_seed,
        "policy_rng_scope": "per_suite_continuous",
        "actions_per_plan": config.actions_per_plan,
        "num_steps_wait": config.num_steps_wait,
        "render_size": config.render_size,
        "max_steps": config.max_steps,
        "libero_root": str(config.libero_root),
    }
    encoded = json.dumps(protocol_fields, sort_keys=True, separators=(",", ":")).encode("utf-8")
    protocol_id = hashlib.sha256(encoded).hexdigest()
    resolved = {
        **protocol_fields,
        "protocol_id": protocol_id,
        "protocol_name": "official_pi05_libero" if is_official_protocol(config) else "custom_pi05_libero",
        "official_protocol": is_official_protocol(config),
        "libero_config_dir": str(config.libero_config_dir),
        "video_fps": config.video_fps,
        "video_policy": config.video_policy,
        "save_actions": config.save_actions,
        "fail_fast": config.fail_fast,
        "resume": config.resume,
    }
    return resolved, protocol_id


def load_policy_config(
    config_json: Path,
    model_path: Path,
    *,
    seed: int,
    actions_per_plan: int,
) -> dict[str, Any]:
    config_json = config_json.expanduser().resolve()
    model_path = model_path.expanduser().resolve()
    with config_json.open("r", encoding="utf-8") as handle:
        values = json.load(handle)
    if not isinstance(values, dict):
        raise ValueError(f"OpenPI model config must be a JSON object: {config_json}")
    values.update(
        {
            "model_cls": "openpi",
            "task": "i2va",
            "model_path": str(model_path),
            "config_json": str(config_json),
            "seed": int(seed),
            "actions_per_plan": int(actions_per_plan),
        }
    )
    return values


def _assert_module_source(module: Any, libero_root: Path) -> None:
    origin_value = getattr(module, "__file__", None)
    if origin_value:
        origins = (Path(origin_value).resolve(),)
    else:
        search_locations = getattr(module, "__path__", None)
        if search_locations is None:
            raise RuntimeError(f"Cannot verify LIBERO module source for {module.__name__!r}")
        origins = tuple(Path(location).resolve() for location in search_locations)
        if not origins:
            raise RuntimeError(f"LIBERO namespace {module.__name__!r} has no search locations")
    outside = [origin for origin in origins if not origin.is_relative_to(libero_root)]
    if outside:
        raise RuntimeError(
            f"LIBERO module {module.__name__!r} resolves outside requested root {libero_root}: {outside}. Start a fresh worker and remove the conflicting LIBERO package from PYTHONPATH."
        )


def _constrain_libero_namespace(package: Any, libero_root: Path) -> None:
    """Restrict the top-level namespace before resolving ``libero.libero``."""
    if getattr(package, "__file__", None):
        _assert_module_source(package, libero_root)
        return

    requested = (libero_root / "libero").resolve()
    locations = tuple(Path(location).resolve() for location in getattr(package, "__path__", ()))
    if requested not in locations:
        raise RuntimeError(f"Requested LIBERO namespace root {requested} is absent; discovered search locations: {locations}")
    package.__path__ = [str(requested)]
    spec = getattr(package, "__spec__", None)
    if spec is not None:
        spec.submodule_search_locations = package.__path__
    _assert_module_source(package, libero_root)


def configure_libero(libero_root: Path, config_dir: Path) -> LiberoRuntime:
    """Configure and import exactly the LIBERO checkout selected by the caller."""
    libero_root = libero_root.expanduser().resolve()
    benchmark_root = libero_root / "libero" / "libero"
    required = (benchmark_root / "bddl_files", benchmark_root / "init_files", benchmark_root / "assets")
    missing = [str(path) for path in required if not path.is_dir()]
    if missing:
        raise FileNotFoundError(f"LIBERO checkout is incomplete under {libero_root}: {missing}")

    config_dir = config_dir.expanduser().resolve()
    config_text = "\n".join(
        (
            f"benchmark_root: {benchmark_root}",
            f"bddl_files: {benchmark_root / 'bddl_files'}",
            f"init_states: {benchmark_root / 'init_files'}",
            f"datasets: {libero_root / 'libero' / 'datasets'}",
            f"assets: {benchmark_root / 'assets'}",
            "",
        )
    )
    atomic_write_text(config_dir / "config.yaml", config_text)
    os.environ["LIBERO_CONFIG_PATH"] = str(config_dir)

    root_text = str(libero_root)
    if root_text in sys.path:
        sys.path.remove(root_text)
    sys.path.insert(0, root_text)
    importlib.invalidate_caches()

    package = importlib.import_module("libero")
    _constrain_libero_namespace(package, libero_root)
    nested_package = importlib.import_module("libero.libero")
    benchmark_module = importlib.import_module("libero.libero.benchmark")
    envs_module = importlib.import_module("libero.libero.envs")
    for module in (package, nested_package, benchmark_module, envs_module):
        _assert_module_source(module, libero_root)

    return LiberoRuntime(
        benchmark_module=benchmark_module,
        get_libero_path=nested_package.get_libero_path,
        env_type=envs_module.OffScreenRenderEnv,
    )


def build_task_specs(runtime: LiberoRuntime, config: EvaluationConfig) -> dict[str, list[TaskSpec]]:
    factories = runtime.benchmark_module.get_benchmark_dict()
    specs: dict[str, list[TaskSpec]] = {}
    for benchmark_name in config.benchmarks:
        if benchmark_name not in factories:
            raise KeyError(f"LIBERO checkout does not provide benchmark {benchmark_name!r}")
        suite = factories[benchmark_name]()
        task_count = int(suite.n_tasks)
        selected = config.task_ids[benchmark_name]
        task_ids = tuple(range(task_count)) if selected is None else selected
        invalid = [task_id for task_id in task_ids if not 0 <= task_id < task_count]
        if invalid:
            raise ValueError(f"Task IDs {invalid} are outside [0, {task_count}) for {benchmark_name}")

        suite_specs = []
        for task_id in task_ids:
            task = suite.get_task(task_id)
            bddl_path = Path(runtime.get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
            init_states_path = Path(runtime.get_libero_path("init_states")) / task.problem_folder / task.init_states_file
            if not bddl_path.is_file():
                raise FileNotFoundError(f"LIBERO BDDL file is missing: {bddl_path}")
            suite_specs.append(
                TaskSpec(
                    benchmark=benchmark_name,
                    task_id=task_id,
                    suite=suite,
                    task=task,
                    bddl_path=bddl_path.resolve(),
                    init_states_path=init_states_path.resolve(),
                )
            )
        specs[benchmark_name] = suite_specs
    return specs


def load_task_init_states(spec: TaskSpec) -> tuple[np.ndarray, str]:
    if spec.init_states_path.is_file():
        import torch

        states = torch.load(spec.init_states_path, map_location="cpu", weights_only=False)
        loader = "direct_file"
    else:
        states = spec.suite.get_task_init_states(spec.task_id)
        loader = "benchmark_api"
    array = np.asarray(states)
    if array.ndim != 2 or len(array) == 0:
        raise ValueError(f"LIBERO task {spec.benchmark}/{spec.task_id} returned invalid init states with shape {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"LIBERO task {spec.benchmark}/{spec.task_id} returned non-finite init states")
    return np.ascontiguousarray(array), loader


def create_environment(runtime: LiberoRuntime, spec: TaskSpec, render_size: int, seed: int) -> Any:
    env = runtime.env_type(
        bddl_file_name=str(spec.bddl_path),
        camera_heights=render_size,
        camera_widths=render_size,
    )
    env.seed(seed)
    return env


def _quat_to_axis_angle(quaternion: Any) -> np.ndarray:
    # Preserve the simulator dtype exactly as the official helper does.
    quat = np.asarray(quaternion).copy()
    if quat.shape != (4,):
        raise ValueError(f"Expected LIBERO quaternion shape (4,), got {quat.shape}")
    quat[3] = np.clip(quat[3], -1.0, 1.0)
    denominator = math.sqrt(max(0.0, 1.0 - float(quat[3]) ** 2))
    if math.isclose(denominator, 0.0):
        return np.zeros(3)
    return quat[:3] * (2.0 * math.acos(float(quat[3])) / denominator)


def rotate_rgb(observation: dict[str, Any], key: str) -> np.ndarray:
    image = np.asarray(observation[key])
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"LIBERO camera {key!r} must be HxWx3, got {image.shape}")
    return np.ascontiguousarray(image[::-1, ::-1], dtype=np.uint8)


def policy_rgb(observation: dict[str, Any], key: str) -> np.ndarray:
    """Apply the official client-side rotate, PIL resize, and uint8 conversion."""
    image = rotate_rgb(observation, key)
    if image.shape[:2] == (POLICY_IMAGE_SIZE, POLICY_IMAGE_SIZE):
        return image

    from PIL import Image

    height, width = image.shape[:2]
    ratio = max(width / POLICY_IMAGE_SIZE, height / POLICY_IMAGE_SIZE)
    resized_height = int(height / ratio)
    resized_width = int(width / ratio)
    resized = Image.fromarray(image, mode="RGB").resize(
        (resized_width, resized_height),
        resample=Image.BILINEAR,
    )
    canvas = Image.new("RGB", (POLICY_IMAGE_SIZE, POLICY_IMAGE_SIZE), 0)
    canvas.paste(
        resized,
        (
            max(0, int((POLICY_IMAGE_SIZE - resized_width) / 2)),
            max(0, int((POLICY_IMAGE_SIZE - resized_height) / 2)),
        ),
    )
    return np.asarray(canvas, dtype=np.uint8).copy()


def state_from_observation(observation: dict[str, Any]) -> np.ndarray:
    return np.concatenate(
        (
            np.asarray(observation["robot0_eef_pos"]),
            _quat_to_axis_angle(observation["robot0_eef_quat"]),
            np.asarray(observation["robot0_gripper_qpos"]),
        )
    )


def run_episode(
    *,
    policy: Any,
    env: Any,
    initial_state: Any,
    task_description: str,
    max_steps: int,
    num_steps_wait: int,
    collect_frames: bool,
) -> tuple[dict[str, Any], list[np.ndarray], np.ndarray]:
    """Run one episode with the exact official warmup/policy done semantics."""
    frames: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    success = False
    warmup_steps = 0
    warmup_done_observed = 0
    action_chunk_calls = 0
    error_type: str | None = None
    error_message: str | None = None
    error_traceback: str | None = None
    started = time.perf_counter()

    policy.clear_action_queue()
    # Match the official evaluator: reset/init failures are infrastructure errors.
    env.reset()
    observation = env.set_init_state(initial_state)
    try:
        # Official examples/libero/main.py ignores done during settling. Both the
        # one-off rollout and quantitative path use this same branch.
        for _ in range(num_steps_wait):
            observation, _reward, done, _info = env.step(LIBERO_DUMMY_ACTION.tolist())
            warmup_steps += 1
            warmup_done_observed += int(bool(done))

        for _step in range(max_steps):
            agentview = policy_rgb(observation, "agentview_image")
            if collect_frames:
                frames.append(agentview)
            images = {
                "agentview": agentview,
                "wrist": policy_rgb(observation, "robot0_eye_in_hand_image"),
            }
            if policy.pending_action_count == 0:
                action_chunk_calls += 1
            action = policy.next_action(
                images=images,
                state=state_from_observation(observation),
                task_description=task_description,
            )
            action = np.asarray(action).reshape(-1)
            actions.append(action.copy())
            observation, _reward, done, _info = env.step(action.tolist())
            if bool(done):
                success = True
                break
    except Exception as exc:  # The official loop counts ordinary rollout errors as failures.
        message = str(exc)
        if isinstance(exc, MemoryError) or "out of memory" in message.lower():
            raise
        error_type = type(exc).__name__
        error_message = message
        error_traceback = traceback.format_exc(limit=20)
        LOGGER.exception("LIBERO episode failed with an exception")
    finally:
        pending_actions_discarded = policy.pending_action_count
        policy.clear_action_queue()

    action_dim = int(policy.output_action_dim)
    action_array = np.stack(actions).reshape(-1, action_dim) if actions else np.empty((0, action_dim), dtype=np.float64)
    termination_reason = "exception" if error_type is not None else "success" if success else "step_limit"
    episode = {
        "success": success,
        "termination_reason": termination_reason,
        "warmup_steps_requested": num_steps_wait,
        "warmup_steps_executed": warmup_steps,
        "warmup_done_observed": warmup_done_observed,
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


def expected_episode_keys(task_specs: dict[str, list[TaskSpec]], config: EvaluationConfig) -> tuple[EpisodeKey, ...]:
    return tuple((benchmark, spec.task_id, init_state_id) for benchmark in config.benchmarks for spec in task_specs[benchmark] for init_state_id in range(config.num_trials_per_task))
