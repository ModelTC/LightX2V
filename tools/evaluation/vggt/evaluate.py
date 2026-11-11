"""VGGT single-file evaluator: run LightX2V and score against GSO ground truth.

Host: python evaluate.py [--smoke] [--card 2]
Saved outputs: python evaluate.py --reuse-runs LIGHTX2V
Inference runs in an isolated worker; postprocessing uses CPU.
"""

import argparse
import csv
import datetime
import functools
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import random
import re
import resource
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import traceback
from contextlib import contextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = Path("/data")
MODEL = "vggt"
TITLE = "VGGT"
PROTOCOL = "深度逐视图中位数尺度对齐后计算AbsRel、RMSE（米）、δ1；相机使用相对位姿AUC@5/10/20度；几何使用归一化对称Chamfer-L1。"
COMMON_CONFIGS = {"suite.json": "common/suite.json", "weight-lock.json": "common/weight-lock.json"}
PLATFORM_CONFIGS = ("paths.json", "source-lock.json", "evaluation-origin.json", "runtime.json", "model.json")
ACTIVE_PLATFORM = os.getenv("EVAL_PLATFORM", "mlu")
CONFIG_LOCATIONS = {**COMMON_CONFIGS, **{name: f"{ACTIVE_PLATFORM}/{name}" for name in PLATFORM_CONFIGS}}


def config_path(name):
    relative = COMMON_CONFIGS.get(name, f"{ACTIVE_PLATFORM}/{name}")
    return ROOT / "configs" / relative


def select_platform(name):
    global ACTIVE_PLATFORM, CONFIG_LOCATIONS
    if not name or Path(name).name != name:
        raise ValueError(f"Invalid platform name: {name!r}")
    ACTIVE_PLATFORM = name
    CONFIG_LOCATIONS = {**COMMON_CONFIGS, **{key: f"{name}/{key}" for key in PLATFORM_CONFIGS}}
    if not config_path("runtime.json").is_file():
        raise ValueError(f"No platform profile exists at configs/{name}/runtime.json")


def load_platform_config():
    profile = json.loads(config_path("runtime.json").read_text())
    required = ("platform", "device_type", "torch_namespace", "container_image", "python_executable", "environment", "visibility_environment", "monitor_command", "monitor_parser")
    missing = [key for key in required if key not in profile]
    if missing or profile["platform"] != ACTIVE_PLATFORM:
        raise ValueError(f"Invalid configs/{ACTIVE_PLATFORM}/runtime.json; missing={missing}, platform={profile.get('platform')!r}")
    return profile


def device_api(torch_module=None):
    import torch as torch_module_import
    module = torch_module or torch_module_import
    profile = load_platform_config()
    return getattr(module, profile["torch_namespace"])


def device_name(index=0):
    profile = load_platform_config()
    return f"{profile['device_type']}:{index}"


def autocast_context():
    profile = load_platform_config()
    dtype_name = profile.get("autocast_dtype", "bfloat16")
    return torch.autocast(profile["device_type"], dtype=getattr(torch, dtype_name))


def platform_fingerprints():
    profile = load_platform_config()
    return {"platform": profile["platform"], "device_type": profile["device_type"], "runtime_sha256": sha256(config_path("runtime.json")), "model_sha256": sha256(config_path("model.json"))}


def copy_active_configs(destination):
    sources = [config_path(name) for name in (*COMMON_CONFIGS, *PLATFORM_CONFIGS)]
    requirements = load_platform_config().get("requirements_file")
    if requirements:
        sources.append(ROOT / requirements)
    for source in sources:
        relative = source.relative_to(ROOT / "configs")
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)


# Model invocation and external measurements


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def seed_all(seed):
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device_api().manual_seed_all(seed)


@contextmanager
def record_model_events(targets, device):
    """Attach temporary timing hooks and restore every method on success or error."""
    events = []
    originals = []
    try:
        for target, method, label in targets:
            original = getattr(target, method)
            had_instance_attribute = method in vars(target)
            instance_attribute = vars(target).get(method)

            @functools.wraps(original)
            def measured(*args, _original=original, _label=label, **kwargs):
                start, end = device.Event(enable_timing=True), device.Event(enable_timing=True)
                start.record()
                result = _original(*args, **kwargs)
                end.record()
                events.append((_label, start, end))
                return result

            setattr(target, method, measured)
            originals.append((target, method, had_instance_attribute, instance_attribute))
        yield events
    finally:
        for target, method, had_instance_attribute, instance_attribute in reversed(originals):
            if had_instance_attribute:
                setattr(target, method, instance_attribute)
            else:
                delattr(target, method)


class LightX2VAdapter:
    def __init__(self, paths, workdir):
        from lightx2v.models.runners.runner_factory import build_runner
        from lightx2v.utils.set_config import build_startup_config

        source = ROOT / "sources" / "lightx2v-vggt"
        settings = {"model_cls": "vggt", "task": "recon", "model_path": str(paths["vggt"]), "config_json": str(config_path("model.json"))}
        config = build_startup_config(settings)
        self.runner = build_runner(config)
        self.model = "vggt"
        self.last_stage_timings = {}
        self.metadata = {"configuration": dict(config), "source": str(source)}

    def predict(self, image_paths, seed):
        prepare_start = time.perf_counter()
        fields = {"seed": seed}
        fields["input_path"] = [str(DATA / p) for p in image_paths]
        request = self.runner.prepare_request(fields)
        prepare_s = time.perf_counter() - prepare_start
        model_start = time.perf_counter()
        result = self.runner.run_request(request)
        self.last_stage_timings = {
            "request_preparation_s": prepare_s,
            "runner_inference_and_postprocess_s": time.perf_counter() - model_start,
        }
        return result


def measurement_targets(adapter):
    """Return model timing points for the sole LightX2V adapter."""
    return [(adapter.runner.model.transformer_infer, "infer", "vggt_forward")]


def run_benchmark_job(args):
    global torch
    args.run_dir.mkdir(parents=True, exist_ok=True)
    if (args.run_dir / "requests.jsonl").exists():
        raise RuntimeError("Use a new run directory; never mix or overwrite prior requests.")
    suite_path = config_path("suite.json")
    suite = json.loads(suite_path.read_text())
    assert suite["model"] == "vggt"
    samples = suite["samples"][: args.limit] if args.limit else suite["samples"]
    repeats = args.repeats or suite["repeats"]
    paths = {k: DATA / v for (k, v) in json.loads((config_path("paths.json")).read_text())["weights"].items()}
    source = ROOT / "sources" / "lightx2v-vggt"
    sys.path.insert(0, str(source))
    profile = load_platform_config()
    os.environ.update(profile["environment"])
    for variable, key in profile.get("model_path_environment", {}).items():
        os.environ[variable] = str(paths[key])
    import numpy as np
    import torch
    for module_name in profile.get("registration_modules", []):
        __import__(module_name)

    torch.set_grad_enabled(False)
    torch.set_num_threads(8)
    torch.use_deterministic_algorithms(True, warn_only=True)
    backend = device_api(torch)
    assert backend.is_available() and backend.device_count() == 1
    versions = {}
    packages = profile.get("version_packages", ["torch", "transformers", "numpy", "scipy", "Pillow", "timm", "kornia", "trimesh", "diffusers"])
    for name in packages:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    env = {
        "model": "vggt",
        "implementation": args.implementation,
        "platform_validation_status": profile.get("validation_status", "unmarked"),
        "source": str(source),
        "versions": versions,
        "python": platform.python_version(),
        "device": str(backend.get_device_name(0)),
        "visible_device_count": backend.device_count(),
        "physical_card": os.getenv("EVAL_PHYSICAL_CARD"),
        "suite_sha256": hashlib.sha256(suite_path.read_bytes()).hexdigest(),
        "source_lock_sha256": hashlib.sha256((config_path("source-lock.json")).read_bytes()).hexdigest(),
        "sample_count": len(samples),
        "repeats": repeats,
        "diagnostic_subset": args.limit is not None,
        "determinism": {"enabled": torch.are_deterministic_algorithms_enabled(), "warn_only": torch.is_deterministic_algorithms_warn_only_enabled(), "pythonhashseed": os.getenv("PYTHONHASHSEED")},
        "evaluation_driver": {"role": "developer_benchmark", "script": Path(__file__).name, "sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
        "evaluation_code": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in ("evaluate.py", "run_eval.sh")},
        "timing_boundary": "wall time from adapter.predict entry through CPU NumPy output materialization; includes image read/preprocessing, inference, model postprocessing and device-to-host transfer; excludes model load, warmup, artifact write and offline evaluation",
    }
    env["platform_configuration"] = platform_fingerprints()
    write_json(args.run_dir / "environment.json", env)
    backend.synchronize()
    start = time.perf_counter()
    args.implementation = "lightx2v"
    adapter = LightX2VAdapter(paths, args.run_dir)
    backend.synchronize()
    env.update(model_load_s=time.perf_counter() - start, adapter=adapter.metadata)
    write_json(args.run_dir / "environment.json", env)
    targets = measurement_targets(adapter)
    schedule = ([("warmup", samples[0], -1)] if not args.no_warmup else []) + [("measured", sample, repeat) for sample in samples for repeat in range(repeats)]
    for kind, sample, repeat in schedule:
        row = {"id": sample["id"], "seed": sample["seed"], "repeat": repeat, "warmup": kind == "warmup", "status": "error"}
        seed_all(sample["seed"])
        backend.synchronize()
        backend.reset_peak_memory_stats()
        row["memory_before_bytes"] = backend.memory_allocated()
        row["started_at_unix"] = time.time()
        start = time.perf_counter()
        try:
            predict_start = time.perf_counter()
            with record_model_events(targets, backend) as events, torch.inference_mode():
                result = adapter.predict(sample["views"], sample["seed"])
            backend.synchronize()
            predict_call_s = time.perf_counter() - predict_start
            row["events"] = [{"stage": name, "seconds": a.elapsed_time(b) / 1000} for (name, a, b) in events]
            row["peak_allocated_bytes"] = backend.max_memory_allocated()
            row["peak_reserved_bytes"] = backend.max_memory_reserved()
            row["peak_incremental_bytes"] = row["peak_allocated_bytes"] - row["memory_before_bytes"]
            row["cpu_process_peak_rss_kib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            conversion_start = time.perf_counter()
            arrays = {k: v.detach().cpu().float().numpy() if v.is_floating_point() else v.detach().cpu().numpy() for (k, v) in result.items()}
            backend.synchronize()
            conversion_s = time.perf_counter() - conversion_start
            row["pipeline_s"] = time.perf_counter() - start
            row["timing_breakdown_s"] = {
                "adapter_predict_call": predict_call_s,
                "result_cpu_numpy_conversion": conversion_s,
                "evaluation_harness_overhead": max(0.0, row["pipeline_s"] - predict_call_s - conversion_s),
            }
            row["adapter_stage_s"] = getattr(adapter, "last_stage_timings", {})
            row["tensors"] = {k: {"shape": list(v.shape), "dtype": str(v.dtype), "finite_ratio": float(np.isfinite(v).mean())} for (k, v) in arrays.items()}
            if not all((np.isfinite(v).all() for v in arrays.values())):
                raise ValueError("Non-finite output")
            row["status"] = "success"
            if kind != "warmup":
                destination = args.run_dir / "outputs" / f"{sample['id']}_r{repeat}.npz"
                destination.parent.mkdir(exist_ok=True)
                save_start = time.perf_counter()
                np.savez(destination, **arrays)
                row.update(
                    output=str(destination.relative_to(args.run_dir)),
                    artifact_write_s=time.perf_counter() - save_start,
                    output_bytes=destination.stat().st_size,
                    output_sha256=hashlib.sha256(destination.read_bytes()).hexdigest(),
                )
            del result, arrays
        except Exception:
            row.update(elapsed_until_error_s=time.perf_counter() - start, error=traceback.format_exc())
            try:
                device_api().empty_cache()
            except Exception:
                pass
        with (args.run_dir / "requests.jsonl").open("a") as stream:
            row["finished_at_unix"] = time.time()
            stream.write(json.dumps(row, allow_nan=False) + "\n")
        print(json.dumps({k: v for (k, v) in row.items() if k not in ("tensors", "events")}), flush=True)
    write_json(args.run_dir / "finished.json", {"scheduled_requests": len(schedule), "completed": True})
    rows = load_rows(args.run_dir)
    return 0 if all(r["status"] == "success" for r in rows) else 1


# Offline accuracy, output differences and provenance


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_environment(run, environment=None):
    """Verify saved outputs against their frozen run configs and the current model pin."""
    run = Path(run)
    environment = environment or json.loads((run / "environment.json").read_text())
    assert environment["model"] == MODEL, "Run belongs to another model"
    profile = environment.get("platform_configuration") or {}
    platform = profile.get("platform")
    snapshot = run.parent / "configs"
    platform_snapshot = snapshot / str(platform)
    suite_snapshot = snapshot / "common" / "suite.json"
    weights_snapshot = snapshot / "common" / "weight-lock.json"
    source_snapshot = platform_snapshot / "source-lock.json"
    runtime_snapshot = platform_snapshot / "runtime.json"
    model_snapshot = platform_snapshot / "model.json"
    if all(path.is_file() for path in (suite_snapshot, weights_snapshot, source_snapshot, runtime_snapshot, model_snapshot)):
        assert environment["suite_sha256"] == sha256(suite_snapshot), "Saved suite snapshot differs from the run"
        assert environment["source_lock_sha256"] == sha256(source_snapshot), "Saved source-lock snapshot differs from the run"
        assert sha256(config_path("suite.json")) == sha256(suite_snapshot), "Current evaluation suite differs from the run"
        assert sha256(config_path("weight-lock.json")) == sha256(weights_snapshot), "Current weight inventory differs from the run"
        saved_runtime = json.loads(runtime_snapshot.read_text())
        current_runtime = json.loads(config_path("runtime.json").read_text())
        # These fields only configured the removed official-reference adapter.
        for config in (saved_runtime, current_runtime):
            config.pop("reference_weight_dtype", None)
            config.pop("cast_reference_matrices", None)
        assert saved_runtime == current_runtime, "Platform runtime changed beyond removed reference-only settings"
        assert sha256(model_snapshot) == sha256(config_path("model.json")), "LightX2V model configuration differs from the run"
        saved_locks = json.loads(source_snapshot.read_text())
        current_locks = json.loads(config_path("source-lock.json").read_text())
        saved_model_locks = {key: value for key, value in saved_locks.items() if key.startswith("lightx2v-")}
        current_model_locks = {key: value for key, value in current_locks.items() if key.startswith("lightx2v-")}
        assert saved_model_locks == current_model_locks, "Pinned LightX2V source differs from the run"
        assert profile == platform_fingerprints_from_snapshot(platform_snapshot), "Saved platform fingerprint differs from its config snapshot"
        return "frozen_run_snapshot"
    if profile:
        assert profile == platform_fingerprints(), "Platform runtime or model configuration differs"
    assert environment["suite_sha256"] == sha256(config_path("suite.json")), "Suite differs"
    assert environment["source_lock_sha256"] == sha256(config_path("source-lock.json")), "Sources differ"
    return "current_project"


def platform_fingerprints_from_snapshot(directory):
    runtime = json.loads((directory / "runtime.json").read_text())
    return {
        "platform": runtime["platform"],
        "device_type": runtime["device_type"],
        "runtime_sha256": sha256(directory / "runtime.json"),
        "model_sha256": sha256(directory / "model.json"),
    }



def load_rows(run):
    path = run / "requests.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def stats(values):
    array = np.asarray(values, dtype=np.float64)
    if not len(array):
        return None
    return {"count": len(array), "mean": float(array.mean()), "median": float(np.median(array)), "min": float(array.min()), "max": float(array.max()), "p95": float(np.percentile(array, 95))}


def difference(left, right):
    if left.shape != right.shape:
        return {"same_shape": False, "left_shape": list(left.shape), "right_shape": list(right.shape)}
    a, b = left.reshape(-1), right.reshape(-1)
    absolute_sum = squared_sum = ref_squared = 0.0
    maximum = 0.0
    exact = True
    for offset in range(0, a.size, 1_000_000):
        x = a[offset : offset + 1_000_000].astype(np.float64)
        y = b[offset : offset + 1_000_000].astype(np.float64)
        error = x - y
        absolute_sum += np.abs(error).sum()
        squared_sum += (error * error).sum()
        ref_squared += (x * x).sum()
        maximum = max(maximum, float(np.abs(error).max(initial=0)))
        exact &= np.array_equal(x, y)
    return {
        "same_shape": True,
        "elements": int(a.size),
        "exact": bool(exact),
        "max_abs_error": maximum,
        "mae": float(absolute_sum / max(a.size, 1)),
        "rmse": float(math.sqrt(squared_sum / max(a.size, 1))),
        "relative_l2": float(math.sqrt(squared_sum / max(ref_squared, 1e-30))),
    }


def compare(left, right):
    shared = sorted(set(left.files) & set(right.files))
    report = {
        "missing_left": sorted(set(right.files) - set(left.files)),
        "missing_right": sorted(set(left.files) - set(right.files)),
        "tensors": {key: difference(left[key], right[key]) for key in shared},
    }
    return report


def read_matrix(path):
    return np.fromstring(Path(path).read_text(), sep=" ").reshape(4, 4)


def resize_depth(path, size):
    with Image.open(path) as image:
        depth = np.asarray(image).astype(np.float32) / 1000.0
    h, w = depth.shape
    side = max(h, w)
    square = np.zeros((side, side), dtype=np.float32)
    top, left = (side - h) // 2, (side - w) // 2
    square[top : top + h, left : left + w] = depth
    return np.asarray(Image.fromarray(square).resize((size, size), Image.Resampling.NEAREST))


def normalize_cloud(points):
    points = points[np.isfinite(points).all(axis=1)]
    if len(points) < 2:
        raise ValueError("Point cloud has fewer than two finite points")
    centered = points - points.mean(axis=0)
    rms = np.sqrt(np.mean(np.sum(centered * centered, axis=1)))
    if rms < 1e-12:
        raise ValueError("Degenerate point cloud")
    return centered / rms


def gso_geometry_cloud(gt_dir, size, grid_size=64):
    """Reconstruct a deterministic sparse world-space cloud from GSO depth."""
    clouds = []
    axis = np.linspace(0, size - 1, grid_size, dtype=int)
    u, v = np.meshgrid(axis, axis)
    u, v = u.reshape(-1), v.reshape(-1)
    for i in range(5):
        depth_path = gt_dir / f"depth_{i:02d}.png"
        with Image.open(depth_path) as image:
            raw = np.asarray(image).astype(np.float64) / 1000.0
        h, w = raw.shape
        side = max(h, w)
        top, left = (side - h) // 2, (side - w) // 2
        depth = resize_depth(depth_path, size).astype(np.float64)
        z = depth[v, u]
        valid = np.isfinite(z) & (z > 1e-6)
        px, py, z = u[valid], v[valid], z[valid]
        if not len(z):
            continue
        intr = read_matrix(gt_dir / f"intrinsics_{i:02d}.txt")[:3, :3].copy()
        scale = size / side
        fx, fy = intr[0, 0] * scale, intr[1, 1] * scale
        cx = (intr[0, 2] + left + 0.5) * scale - 0.5
        cy = (intr[1, 2] + top + 0.5) * scale - 0.5
        cam = np.stack(((px - cx) * z / fx, (py - cy) * z / fy, z), axis=1)
        pose = read_matrix(gt_dir / f"pose_{i:02d}.txt")
        clouds.append(cam @ pose[:3, :3].T + pose[:3, 3])
    if not clouds:
        raise ValueError("GSO depth produced no valid geometry points")
    return np.concatenate(clouds)


def chamfer_l1_aligned(predicted, target, iterations=20):
    source, target = normalize_cloud(predicted), normalize_cloud(target)
    target_tree = cKDTree(target)
    for _ in range(iterations):
        nearest = target_tree.query(source, k=1)[1]
        source_mean, target_mean = source.mean(axis=0), target[nearest].mean(axis=0)
        u, _, vt = np.linalg.svd((source - source_mean).T @ (target[nearest] - target_mean))
        rotation = vt.T @ u.T
        if np.linalg.det(rotation) < 0:
            vt[-1] *= -1
            rotation = vt.T @ u.T
        translation = target_mean - source_mean @ rotation.T
        source = source @ rotation.T + translation
    dp = target_tree.query(source, k=1)[0]
    dg = cKDTree(source).query(target, k=1)[0]
    return float((dp.mean() + dg.mean()) / 2)


def rotation_angle(matrix):
    return float(np.degrees(np.arccos(np.clip((np.trace(matrix) - 1) / 2, -1, 1))))


def vggt_metrics(output, sample, data):
    gt_dir = data / sample["gt_directory"]
    predicted = output["depth"][0, ..., 0].astype(np.float64)
    predicted_points = output["world_points"][0].astype(np.float64)
    views = []
    for i, pred in enumerate(predicted):
        gt = resize_depth(gt_dir / f"depth_{i:02d}.png", pred.shape[0])
        valid = np.isfinite(gt) & (gt > 1e-6)
        if not valid.any():
            raise ValueError("No valid GT depth pixels")
        p, g = pred[valid], gt[valid].astype(np.float64)
        scale = float(np.median(g) / max(float(np.median(p)), 1e-12))
        p = p * scale
        ratio = np.maximum(g / np.maximum(p, 1e-12), p / g)
        views.append(
            {
                "valid_gt_pixels": int(valid.sum()),
                "pred_positive_ratio": float((p > 0).mean()),
                "median_scale": scale,
                "abs_rel": float((np.abs(p - g) / g).mean()),
                "rmse_m": float(np.sqrt(((p - g) ** 2).mean())),
                "delta1": float((ratio < 1.25).mean()),
            }
        )
    extr = output["extrinsics"][0].astype(np.float64)
    poses = [read_matrix(gt_dir / f"pose_{i:02d}.txt") for i in range(len(extr))]
    # GSO stores camera-to-world. VGGT returns world-to-camera.
    gt = [np.linalg.inv(p) for p in poses]
    pairs = []
    for i in range(len(extr)):
        for j in range(i + 1, len(extr)):
            rp = extr[j, :3, :3] @ extr[i, :3, :3].T
            rg = gt[j][:3, :3] @ gt[i][:3, :3].T
            tp = extr[j, :3, 3] - rp @ extr[i, :3, 3]
            tg = gt[j][:3, 3] - rg @ gt[i][:3, 3]
            row = {"i": i, "j": j, "rotation_deg": rotation_angle(rp @ rg.T)}
            if np.linalg.norm(tp) > 1e-10 and np.linalg.norm(tg) > 1e-10:
                row["translation_direction_deg"] = float(np.degrees(np.arccos(np.clip(np.dot(tp, tg) / (np.linalg.norm(tp) * np.linalg.norm(tg)), -1, 1))))
            pairs.append(row)
    size = predicted_points.shape[1]
    axis = np.linspace(0, size - 1, 64, dtype=int)
    u, v = np.meshgrid(axis, axis)
    sampled_predictions = []
    for i in range(predicted_points.shape[0]):
        depth = resize_depth(gt_dir / f"depth_{i:02d}.png", size).astype(np.float64)
        z = depth[v, u]
        valid = np.isfinite(z) & (z > 1e-6)
        sampled_predictions.append(predicted_points[i, v[valid], u[valid]])
    # Both clouds use the same 64x64 pixel grid and GT-validity mask.
    gt_cloud = gso_geometry_cloud(gt_dir, size)
    pred_cloud = np.concatenate([p for p in sampled_predictions if len(p)], axis=0)
    geometry = {"chamfer_l1": chamfer_l1_aligned(pred_cloud, gt_cloud)}
    return {"depth_views": views, "camera_pairs": pairs, "depth_macro": {key: float(np.mean([v[key] for v in views])) for key in ("abs_rel", "rmse_m", "delta1")}, "geometry": geometry}


def auc(errors, threshold):
    errors = np.sort(np.asarray(errors))
    if not len(errors):
        return None
    recalls = np.arange(1, len(errors) + 1) / len(errors)
    n = np.searchsorted(errors, threshold, side="right")
    x = np.r_[0.0, errors[:n], threshold]
    y = np.r_[0.0, recalls[:n], recalls[n - 1] if n else 0.0]
    integrate = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    return float(integrate(y, x) / threshold)


def evaluate_run(run, suite, data):
    rows = load_rows(run)
    environment = json.loads((run / "environment.json").read_text())
    expected = environment["sample_count"] * environment["repeats"]
    measured = [r for r in rows if not r["warmup"]]
    success = [r for r in measured if r["status"] == "success"]
    metrics = []
    metric_cache = {}
    errors = []
    samples = {s["id"]: s for s in suite["samples"]}
    for row in success:
        path = run / row["output"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == row["output_sha256"], path
        key = (row["id"], row["output_sha256"])
        if key in metric_cache:
            metrics.append({"id": row["id"], "repeat": row["repeat"], "metrics": metric_cache[key], "reused_verified_identical_output": True})
            continue
        with np.load(path, allow_pickle=False) as output:
            try:
                quality = vggt_metrics(output, samples[row["id"]], data)
                metrics.append({"id": row["id"], "repeat": row["repeat"], "metrics": quality})
                metric_cache[key] = quality
            except Exception as exc:
                errors.append({"id": row["id"], "repeat": row["repeat"], "error": str(exc)})
    first = [r["metrics"] for r in metrics if r["repeat"] == 0]
    quality = {}
    if first:
        quality["depth_macro_per_object"] = {key: stats([r["depth_macro"][key] for r in first]) for key in ("abs_rel", "rmse_m", "delta1")}
        quality["geometry_per_object"] = {"chamfer_l1": stats([r["geometry"]["chamfer_l1"] for r in first])}
        rotations = [p["rotation_deg"] for r in first for p in r["camera_pairs"]]
        translations = [p["translation_direction_deg"] for r in first for p in r["camera_pairs"] if "translation_direction_deg" in p]
        joint = [max(p["rotation_deg"], p["translation_direction_deg"]) for r in first for p in r["camera_pairs"] if "translation_direction_deg" in p]
        quality.update(rotation_deg=stats(rotations), translation_direction_deg=stats(translations), relative_pose_auc={str(t): auc(joint, t) for t in (5, 10, 20)})
    times = [r["pipeline_s"] for r in success]
    performance = {
        "pipeline_s": stats(times),
        "throughput_requests_per_s": len(times) / sum(times) if times else None,
        "peak_allocated_gib": max((r["peak_allocated_bytes"] for r in success), default=0) / 2**30,
        "peak_reserved_gib": max((r["peak_reserved_bytes"] for r in success), default=0) / 2**30,
        "peak_incremental_gib": max((r["peak_incremental_bytes"] for r in success), default=0) / 2**30,
        "cpu_peak_rss_gib": max((r["cpu_process_peak_rss_kib"] for r in success), default=0) / 2**20,
        "artifact_write_s": stats([r["artifact_write_s"] for r in success]),
    }
    stages = sorted({e["stage"] for r in success for e in r["events"]})
    performance["event_stage_total_s_per_request"] = {stage: stats([sum((e["seconds"] for e in r["events"] if e["stage"] == stage)) for r in success]) for stage in stages}
    timing_keys = sorted({key for r in success for key in r.get("timing_breakdown_s", {})})
    performance["request_timing_breakdown_s"] = {key: stats([r["timing_breakdown_s"][key] for r in success if key in r.get("timing_breakdown_s", {})]) for key in timing_keys}
    adapter_keys = sorted({key for r in success for key in r.get("adapter_stage_s", {})})
    performance["adapter_stage_s"] = {key: stats([r["adapter_stage_s"][key] for r in success if key in r.get("adapter_stage_s", {})]) for key in adapter_keys}
    performance["model_load_s"] = environment.get("model_load_s")
    telemetry_path = run / "telemetry.jsonl"
    if telemetry_path.exists():
        telemetry = [json.loads(line) for line in telemetry_path.read_text().splitlines()]
        windows = [(r.get("started_at_unix", 0), r.get("started_at_unix", 0) + r["pipeline_s"]) for r in success]
        active = [s for s in telemetry if any((start <= s["timestamp"] <= end for (start, end) in windows))]
        performance["sampled_inference_telemetry"] = {key: stats([s[key] for s in active if key in s]) for key in ("util_percent", "memory_mib", "temperature_c", "power_w")}
        performance["telemetry_samples_total"] = len(telemetry)
        performance["telemetry_samples_in_inference"] = len(active)
        performance["telemetry_multi_process_samples"] = [s for s in active if len(s.get("pids", [])) > 1]
    report = {
        "run": str(run),
        "expected_measured_count": expected,
        "missing_measured_count": max(expected - len(measured), 0),
        "finished_marker": (run / "finished.json").exists(),
        "diagnostic_subset": environment["diagnostic_subset"],
        "success_rate": len(success) / max(expected, 1),
        "measured_count": len(measured),
        "success_count": len(success),
        "failed_count": len(measured) - len(success),
        "warmup": [r for r in rows if r["warmup"]],
        "inference_errors": [r for r in measured if r["status"] != "success"],
        "evaluation_errors": errors,
        "quality": quality,
        "performance": performance,
        "per_output_metrics": metrics,
    }
    return report


def pair_runs(left, right, repeat_check=False):
    a = {(r["id"], r["repeat"]): r for r in load_rows(left) if not r["warmup"] and r["status"] == "success"}
    b = {(r["id"], r["repeat"]): r for r in load_rows(right) if not r["warmup"] and r["status"] == "success"}
    pairs = []
    for key, row in a.items():
        if key[1] != 0:
            continue
        other_key = (key[0], 1 if repeat_check else 0)
        if other_key not in b:
            continue
        with np.load(left / row["output"], allow_pickle=False) as x, np.load(right / b[other_key]["output"], allow_pickle=False) as y:
            pairs.append({"id": key[0], "comparison": compare(x, y)})
    return pairs


# Report formatting and optional artifact export


def number(value):
    return f"{value:.8g}" if value is not None else "缺失"


def field_summary(pairs):
    names = sorted({name for p in pairs for name in p["comparison"]["tensors"]})
    rows = []
    for name in names:
        all_values = [p["comparison"]["tensors"][name] for p in pairs]
        values = [v for v in all_values if v.get("same_shape")]
        count = sum(v["elements"] for v in values)
        if count:
            mae = sum(v["mae"] * v["elements"] for v in values) / count
            rmse = math.sqrt(sum(v["rmse"] ** 2 * v["elements"] for v in values) / count)
            maximum = max(v["max_abs_error"] for v in values)
            rows.append(f"| {name} | {len(values)}/{len(all_values)} | {number(mae)} | {number(rmse)} | {number(maximum)} |")
        else:
            rows.append(f"| {name} | 0/{len(all_values)} | 形状不同 | 形状不同 | 形状不同 |")
    return rows


def repeat_status(pairs):
    exact = sum(not p["comparison"]["missing_left"] and not p["comparison"]["missing_right"] and all(v.get("exact", False) for v in p["comparison"]["tensors"].values()) for p in pairs)
    return f"{exact}/{len(pairs)} 对象的全部保存字段逐元素一致"


def export_additional_outputs(runs, samples, data, destination):
    """Export optional model-specific artifacts from LightX2V predictions."""
    return []


def parse_cnmon(text):
    cards = {}
    current = None
    for line in text.splitlines():
        header = re.match(r"\|\s*(\d+)\s+/\s+MLU", line)
        if header:
            current = int(header.group(1))
            util = re.search(r"\|\s*(\d+)%", line)
            cards[current] = {"util_percent": int(util.group(1)) if util else None, "pids": []}
            continue
        if current is not None:
            memory = re.search(r"(\d+)\s+MiB/\s*(\d+)\s+MiB", line)
            if memory:
                cards[current].update(memory_mib=int(memory.group(1)), capacity_mib=int(memory.group(2)))
                power = re.search(r"(\d+)C\s+(\d+)\s*W/\s*(\d+)\s*W", line)
                if power:
                    cards[current].update(temperature_c=int(power.group(1)), power_w=int(power.group(2)))
                current = None
        process = re.match(r"\|\s*(\d+)\s+/\s+(\d+)\s+/", line)
        if process and int(process.group(1)) in cards:
            cards[int(process.group(1))]["pids"].append(int(process.group(2)))
    return cards


def parse_device_monitor(text):
    profile = load_platform_config()
    parser = profile["monitor_parser"]
    if parser == "cnmon":
        return parse_cnmon(text)
    if parser == "json":
        value = json.loads(text)
        if isinstance(value, dict) and "card" in value:
            card = int(value.pop("card"))
            return {card: value}
        if isinstance(value, dict):
            return {int(card): record for card, record in value.items()}
    raise ValueError(f"Unsupported monitor_parser {parser!r}; provide normalized JSON or cnmon output")


def monitor_snapshot(card):
    profile = load_platform_config()
    command = [part.format(card=card, project=str(ROOT)) for part in profile["monitor_command"]]
    raw = subprocess.check_output(command, text=True, timeout=10)
    return raw, parse_device_monitor(raw)


# One-command orchestration and command line


# The content of this file is appended to each model's self-contained evaluator.


def load_numeric_dependencies():
    """Keep host orchestration and --help usable without model/scientific packages."""
    global np, Image, cKDTree
    import numpy as np
    from PIL import Image
    from scipy.spatial import cKDTree


def flatten_numbers(value, prefix=""):
    result = {}
    if isinstance(value, dict):
        for key, item in value.items():
            result.update(flatten_numbers(item, f"{prefix}.{key}" if prefix else key))
    elif isinstance(value, (int, float, bool)) or value is None:
        result[prefix] = value
    return result


def write_csv(path, rows):
    columns = list(dict.fromkeys(key for row in rows for key in row))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns or ["status"])
        writer.writeheader()
        writer.writerows(rows)


def empty_evaluation(run, expected, error):
    return {
        "run": str(run),
        "expected_measured_count": expected,
        "measured_count": 0,
        "missing_measured_count": expected,
        "finished_marker": False,
        "success_count": 0,
        "failed_count": 0,
        "success_rate": 0,
        "diagnostic_subset": True,
        "warmup": [],
        "inference_errors": [],
        "evaluation_errors": [{"error": error}],
        "quality": {},
        "performance": {},
        "per_output_metrics": [],
    }


def make_markdown(report, summary):
    result = report["lightx2v"]
    lines = [
        f"# {TITLE} 评测报告",
        "",
        f"状态：{summary['status']}；范围：{summary['scope']}；对象数：{summary['sample_count']}；重复次数：{summary['repeats']}。",
        "",
        "本项目只运行 LightX2V 推理；精度指标按固定 GSO 真值计算。",
        "",
        "## 结果",
        "",
        "| 指标 | LightX2V |",
        "|---|---:|",
    ]
    for key in ("success_count", "failed_count", "missing_measured_count", "success_rate"):
        lines.append(f"| {key} | {number(result.get(key))} |")
    for key, value in (flatten_numbers(result["quality"], "quality") | flatten_numbers(result["performance"], "performance")).items():
        lines.append(f"| {key} | {number(value)} |")
    lines += [
        "",
        "## 重复运行差异",
        "",
        repeat_status(report["lightx2v_repeat_differences"]),
        "",
        "## 数据与计时边界",
        "",
        PROTOCOL,
        "",
        f"固定GSO对象、输入顺序、seed和精度策略；平台为 {load_platform_config()['platform']}。",
        "端到端同步墙钟计时覆盖读图、预处理、模型执行和CPU输出，排除模型加载、预热、NPZ写盘及离线指标。",
        "小样本检查仅代表所选对象，不代表完整1023对象GSO评测。",
        "",
        "## 产物",
        "",
        "- summary.json / report.md：完成状态、质量和性能汇总。",
        "- requests.csv / metrics.csv：逐请求时延、显存和逐对象质量指标。",
        "- lightx2v-metrics.json：LightX2V 对 GSO 真值的精度与性能。",
        "- lightx2v/：本项目运行产生的NPZ、环境、请求和遥测记录。",
        "- extra-output-manifest.json：额外产物清单。",
    ]
    if summary["errors"]:
        lines += ["", "## 未完成或失败项目", ""]
        lines += ["- " + row["stage"] + ": " + row["error"].splitlines()[-1] for row in summary["errors"]]
    return "\n".join(lines) + "\n"


def postprocess(args):
    """Compute LightX2V quality against GSO ground truth after inference exits."""
    load_numeric_dependencies()
    destination = args.output
    manifest = json.loads((destination / "manifest.json").read_text())
    suite = json.loads(config_path("suite.json").read_text())
    selected = suite["samples"][: manifest["sample_count"]]
    suite = dict(suite, samples=selected)
    run = args.reuse_runs[0] if args.reuse_runs else destination / "lightx2v"
    runs = {"lightx2v": run}
    expected = manifest["sample_count"] * manifest["repeats"]
    report = {"model": MODEL, "evaluation_code_sha256": sha256(Path(__file__)), "suite_sha256": sha256(config_path("suite.json"))}
    errors = [{"stage": stage, "error": f"Worker exited with code {code}"} for stage, code in manifest.get("stage_exit_codes", {}).items() if code]
    request_table, metric_table = [], []
    try:
        env = json.loads((run / "environment.json").read_text())
        verify_environment(run, env)
        if env["implementation"] != "lightx2v" or env["sample_count"] != len(selected) or env["repeats"] != manifest["repeats"]:
            raise ValueError("Saved LightX2V run/sample count/repeats differ from this evaluation")
        report["lightx2v"] = evaluate_run(run, suite, args.data_root)
    except Exception:
        error = traceback.format_exc()
        errors.append({"stage": "lightx2v", "error": error})
        report["lightx2v"] = empty_evaluation(run, expected, error)
    side = report["lightx2v"]
    write_json(destination / "lightx2v-metrics.json", side)
    try:
        request_rows = load_rows(run)
    except Exception:
        request_rows = []
        errors.append({"stage": "lightx2v_request_log", "error": traceback.format_exc()})
    for row in request_rows:
        request_table.append({"implementation": "lightx2v", **{key: row.get(key) for key in ("id", "seed", "repeat", "warmup", "status", "pipeline_s", "peak_allocated_bytes", "peak_reserved_bytes", "peak_incremental_bytes", "cpu_process_peak_rss_kib", "artifact_write_s", "output", "output_sha256", "error")}, "events_json": json.dumps(row.get("events", []))})
    for row in side["per_output_metrics"]:
        metric_table.append({"implementation": "lightx2v", "id": row["id"], "repeat": row["repeat"], **flatten_numbers(row["metrics"])})
    try:
        report["lightx2v_repeat_differences"] = pair_runs(run, run, True)
    except Exception:
        report["lightx2v_repeat_differences"] = []
        errors.append({"stage": "lightx2v_repeat_differences", "error": traceback.format_exc()})
    if len(report["lightx2v_repeat_differences"]) != len(selected):
        errors.append({"stage": "lightx2v_repeat_differences", "error": f"Expected {len(selected)} object pairs, got {len(report['lightx2v_repeat_differences'])}"})
    errors += [{"stage": "lightx2v_inference", "error": r["error"]} for r in side["inference_errors"]]
    errors += [{"stage": "lightx2v_metric", "error": r["error"]} for r in side["evaluation_errors"]]
    if side["success_count"] != expected or side["missing_measured_count"] or not side["finished_marker"]:
        errors.append({"stage": "lightx2v", "error": f"Success={side['success_count']}, expected={expected}, missing={side['missing_measured_count']}, finished={side['finished_marker']}"})
    write_json(destination / "extra-output-manifest.json", [])
    write_json(destination / "comparison.json", report)
    write_csv(destination / "requests.csv", request_table)
    write_csv(destination / "metrics.csv", metric_table)
    write_csv(destination / "summary.csv", [{"implementation": "lightx2v", **{k: side[k] for k in ("success_count", "failed_count", "missing_measured_count", "success_rate")}, **flatten_numbers(side["quality"], "quality"), **flatten_numbers(side["performance"], "performance")}])
    summary = {"status": "complete" if not errors else "incomplete", "model": MODEL, "scope": "full_gso_1023" if len(selected) == 1023 and args.limit is None else "diagnostic_subset", "sample_count": len(selected), "repeats": manifest["repeats"], "expected_per_implementation": expected, "reused_predictions": manifest["reused_predictions"], "runs": {"lightx2v": str(run)}, "results": {"lightx2v": {k: side[k] for k in ("success_count", "failed_count", "missing_measured_count", "quality", "performance")}}, "errors": errors}
    write_json(destination / "failures.json", errors)
    write_json(destination / "summary.json", summary)
    (destination / "report.md").write_text(make_markdown(report, summary))
    files = [{"path": str(p.relative_to(destination)), "bytes": p.stat().st_size, "sha256": sha256(p)} for p in sorted(destination.rglob("*")) if p.is_file() and p.name not in ("provenance.json", "postprocess.log")]
    write_json(destination / "provenance.json", {"model": MODEL, "evaluation_code_sha256": sha256(Path(__file__)), "suite_sha256": sha256(config_path("suite.json")), "source_lock_sha256": sha256(config_path("source-lock.json")), "weight_lock_sha256": sha256(config_path("weight-lock.json")), "platform_configuration": platform_fingerprints(), "files": files})
    print(json.dumps({"status": summary["status"], "sample_count": len(selected), "output": str(destination)}), flush=True)
    return 0 if summary["status"] == "complete" else 1


def telemetry_loop(card, destination, stop):
    while not stop.is_set():
        row = {"timestamp": time.time(), "card": card}
        try:
            _, cards = monitor_snapshot(card)
            row.update(cards[card])
        except Exception as exc:
            row["error"] = str(exc)
        with destination.open("a") as stream:
            stream.write(json.dumps(row) + "\n")
        stop.wait(2)


def verify_source_files(directory, record):
    for file in record["files"]:
        path = directory / file["path"]
        if not path.is_file() or path.stat().st_size != file["bytes"] or sha256(path) != file["sha256"]:
            raise ValueError(f"Pinned source file differs or is missing: {path}")


def prepare_sources():
    """Rebuild ignored source dependencies from public, immutable Git revisions."""
    locks = json.loads(config_path("source-lock.json").read_text())
    source_root = ROOT / "sources"
    source_root.mkdir(exist_ok=True)
    for name, record in locks.items():
        target = source_root / name
        if target.exists():
            verify_source_files(target, record)
            print(f"{name}: existing pinned files verified", flush=True)
            continue
        if not re.fullmatch(r"[0-9a-f]{40}", record["revision"]):
            raise ValueError("Source revision must be a complete immutable Git SHA")
        temporary = Path(tempfile.mkdtemp(prefix=f".prepare-{name}-", dir=source_root))
        try:
            env = dict(os.environ, GIT_TERMINAL_PROMPT="0", GIT_LFS_SKIP_SMUDGE="1")
            subprocess.run(["git", "init", "--quiet", str(temporary)], check=True, env=env)
            subprocess.run(["git", "-C", str(temporary), "remote", "add", "origin", record["repository"]], check=True, env=env)
            subprocess.run(["git", "-C", str(temporary), "fetch", "--quiet", "--depth", "1", "origin", record["revision"]], check=True, env=env)
            subprocess.run(["git", "-C", str(temporary), "checkout", "--quiet", "--detach", "FETCH_HEAD"], check=True, env=env)
            verify_source_files(temporary, record)
            temporary.rename(target)
            print(f"{name}: pinned revision and file hashes verified", flush=True)
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)


def verify_inputs(args, destination):
    suite = json.loads((config_path("suite.json")).read_text())
    if suite["model"] != MODEL:
        raise ValueError("Suite belongs to another model")
    samples = suite["samples"][: args.limit] if args.limit else suite["samples"]
    for sample in samples:
        for file in sample["files"]:
            path = args.data_root / file["path"]
            if path.stat().st_size != file["bytes"] or sha256(path) != file["sha256"]:
                raise ValueError(f"Input or GT changed: {path}")
    sources = json.loads((config_path("source-lock.json")).read_text())
    for name, lock in sources.items():
        for file in lock["files"]:
            if sha256(ROOT / "sources" / name / file["path"]) != file["sha256"]:
                raise ValueError(f"Source changed: {name}/{file['path']}")
    weights = json.loads((config_path("weight-lock.json")).read_text())
    for file in weights:
        if (args.data_root / file["path"]).stat().st_size != file["bytes"]:
            raise ValueError(f"Weight size changed: {file['path']}")
        if getattr(args, "rehash_weights", False) and sha256(args.data_root / file["path"]) != file["sha256"]:
            raise ValueError(f"Weight hash changed: {file['path']}")
    write_json(
        destination / "input-audit.json",
        {
            "input_and_gt_hashes": "verified",
            "sources": "SHA256 verified",
            "weights": "SHA256 verified" if getattr(args, "rehash_weights", False) else "sizes checked; frozen full SHA256 inventory retained",
            "weight_files": len(weights),
        },
    )
    return suite, samples


def container_command(args, destination, stage, run=None):
    name = f"eval-{MODEL}-{destination.name[-24:]}-{stage}"
    profile = load_platform_config()
    command = [
        "docker",
        "run",
        "--rm",
        "--name",
        name,
        "--network",
        "none",
        "--read-only",
        "--tmpfs",
        "/tmp:rw,size=2g",
        "--shm-size=16g",
        "--mount",
        f"type=bind,src={ROOT},dst=/eval",
        "--mount",
        f"type=bind,src={args.data_root},dst=/data,readonly",
        "--mount",
        f"type=bind,src={destination},dst=/results",
        "-e",
        "PYTHONPATH=/eval/runtime",
        "-e",
        "PYTHONDONTWRITEBYTECODE=1",
        "-e",
        "PYTHONHASHSEED=42",
        "-e",
        "TZ=UTC",
        "-e",
        "HF_HOME=/eval/cache/huggingface",
        "-e",
        "HF_MODULES_CACHE=/eval/cache/hf-modules",
        "-e",
        "TRITON_CACHE_DIR=/eval/cache/triton",
        "-e",
        "TORCH_EXTENSIONS_DIR=/eval/cache/torch-extensions",
        "-e",
        "XDG_CACHE_HOME=/eval/cache",
        "-e",
        "MPLCONFIGDIR=/eval/cache/matplotlib",
    ]
    for key, value in profile["environment"].items():
        command += ["-e", f"{key}={value}"]
    extra = []
    if stage != "postprocess":
        if "docker_device_args" in profile:
            command.extend(part.format(card=args.card, project=str(ROOT)) for part in profile["docker_device_args"])
        else:
            for device in profile.get("docker_devices", []):
                command += ["--device", device.format(card=args.card)]
        for key, value in profile["visibility_environment"].items():
            command += ["-e", f"{key}={value.format(card=args.card)}"]
        extra = ["--run-dir", f"/results/{stage}", "--repeats", str(args.repeats)]
        if args.limit:
            extra += ["--limit", str(args.limit)]
    else:
        run_paths = []
        path = args.reuse_runs[0] if args.reuse_runs else None
        if path is None:
            run_paths.append("/results/lightx2v")
        else:
            if (path.parent / "configs").is_dir():
                target_root = "/saved/lightx2v-output"
                command += ["--mount", f"type=bind,src={path.parent},dst={target_root},readonly"]
                run_paths.append(f"{target_root}/{path.name}")
            else:
                target = f"/saved/lightx2v/{path.name}"
                command += ["--mount", f"type=bind,src={path},dst={target},readonly"]
                run_paths.append(target)
        extra = ["--reuse-runs", *run_paths]
    command += [
        "-w",
        "/eval",
        "--entrypoint",
        load_platform_config()["python_executable"],
        args.image,
        "-u",
        "/eval/evaluate.py",
        "--worker",
        stage,
        "--platform",
        ACTIVE_PLATFORM,
        "--output",
        "/results",
        "--data-root",
        "/data",
        *extra,
    ]
    return name, command


def execute_stage(args, destination, stage):
    run = destination / stage
    stop, monitor = threading.Event(), None
    if stage != "postprocess":
        run.mkdir()
        snapshot, cards = monitor_snapshot(args.card)
        (run / "device-before.txt").write_text(snapshot)
        card = cards.get(args.card)
        idle_memory_max = load_platform_config().get("idle_memory_mib_max", 0)
        if card is None or not 0 <= card.get("memory_mib", -1) <= idle_memory_max or card.get("util_percent") != 0 or card.get("pids"):
            raise RuntimeError(f"Physical card {args.card} is not idle: {card}")
        monitor = threading.Thread(target=telemetry_loop, args=(args.card, run / "telemetry.jsonl", stop), daemon=True)
        monitor.start()
    name, command = container_command(args, destination, stage)
    process = None
    print(f"[{TITLE}] {stage}: running; log={destination / (stage + '.log')}", flush=True)
    try:
        with (destination / f"{stage}.log").open("w") as log:
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
            previous = None
            while process.poll() is None:
                try:
                    rows = load_rows(run)
                except json.JSONDecodeError:
                    # The running worker may be in the middle of appending a row.
                    time.sleep(1)
                    continue
                measured = [row for row in rows if not row.get("warmup")]
                progress = {"stage": stage, "measured": len(measured), "success": sum(r["status"] == "success" for r in measured)}
                write_json(destination / "progress.json", progress)
                if measured and progress != previous:
                    print(f"[{TITLE}] {stage}: {len(measured)} measured, {progress['success']} success", flush=True)
                    previous = progress
                time.sleep(1)
            return process.returncode
    finally:
        stop.set()
        if monitor:
            monitor.join(timeout=12)
        if process is not None and process.poll() is None:
            subprocess.run(["docker", "stop", "--time", "10", name], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            process.terminate()
            process.wait(timeout=15)


def run_campaign(args):
    destination = args.output.resolve()
    destination.mkdir(parents=True, exist_ok=False)
    manifest = {
        "model": MODEL,
        "started_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "evaluation_code_sha256": sha256(Path(__file__)),
        "card": None if args.reuse_runs else args.card,
        "image": args.image,
        "reused_predictions": bool(args.reuse_runs),
        "command_line": sys.argv,
        "host_data_root": str(args.data_root),
        "sample_count": 0,
        "repeats": args.repeats,
        "stage_exit_codes": {},
        "platform_configuration": platform_fingerprints(),
    }
    try:
        suite, samples = verify_inputs(args, destination)
        if args.reuse_runs:
            environments = [json.loads((run / "environment.json").read_text()) for run in args.reuse_runs]
            if len({(env["sample_count"], env["repeats"]) for env in environments}) != 1:
                raise ValueError("Reused runs have different sample counts or repeats")
            manifest["sample_count"] = environments[0]["sample_count"]
            manifest["repeats"] = environments[0]["repeats"]
        else:
            manifest["sample_count"] = len(samples)
        manifest["image_id"] = subprocess.check_output(["docker", "image", "inspect", args.image, "--format", "{{.Id}}"], text=True).strip()
        write_json(destination / "manifest.json", manifest)
        copy_active_configs(destination / "configs")
        if args.install_deps:
            with (destination / "install.log").open("w") as log:
                subprocess.run(
                    [
                        "docker",
                        "run",
                        "--rm",
                        "--mount",
                        f"type=bind,src={ROOT},dst=/eval",
                        "--entrypoint",
                        load_platform_config()["python_executable"],
                        args.image,
                        "-m",
                        "pip",
                        "install",
                        "--no-deps",
                        "--target",
                        "/eval/runtime",
                        "-r",
                        "/eval/" + load_platform_config().get("requirements_file", "runtime-requirements.txt"),
                    ],
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=True,
                )
        if not args.reuse_runs:
            for stage in ("lightx2v",):
                try:
                    manifest["stage_exit_codes"][stage] = execute_stage(args, destination, stage)
                except Exception:
                    manifest["stage_exit_codes"][stage] = -1
                    (destination / f"{stage}-host-error.txt").write_text(traceback.format_exc())
                write_json(destination / "manifest.json", manifest)
        code = execute_stage(args, destination, "postprocess")
        manifest["stage_exit_codes"]["postprocess"] = code
        manifest["finished_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        write_json(destination / "manifest.json", manifest)
        if not (destination / "summary.json").exists():
            raise RuntimeError("Postprocessing failed before writing summary.json; see postprocess.log")
        summary = json.loads((destination / "summary.json").read_text())
        summary["host_output_directory"] = str(destination)
        summary["runs"] = {key: str(path) for key, path in zip(("lightx2v",), args.reuse_runs or [destination / "lightx2v"])}
        summary["stage_exit_codes"] = manifest["stage_exit_codes"]
        write_json(destination / "summary.json", summary)
        write_json(destination / "progress.json", {"stage": "finished", "status": summary["status"]})
        # Final manifest hashes cover the final host-side status and finished metadata.
        provenance = json.loads((destination / "provenance.json").read_text())
        provenance["files"] = [
            {"path": str(p.relative_to(destination)), "bytes": p.stat().st_size, "sha256": sha256(p)} for p in sorted(destination.rglob("*")) if p.is_file() and p.name != "provenance.json"
        ]
        write_json(destination / "provenance.json", provenance)
        print(f"[{TITLE}] {summary['status']}: {destination / 'report.md'}", flush=True)
        return code
    except BaseException:
        error = traceback.format_exc()
        write_json(destination / "fatal-error.json", {"error": error})
        if not (destination / "summary.json").exists():
            write_json(destination / "summary.json", {"status": "failed", "model": MODEL, "error": error})
        print(f"[{TITLE}] failed; evidence saved at {destination}", file=sys.stderr)
        raise


def main():
    bootstrap = argparse.ArgumentParser(add_help=False)
    bootstrap.add_argument("--platform", default=os.getenv("EVAL_PLATFORM", "mlu"))
    bootstrap_args, _ = bootstrap.parse_known_args()
    select_platform(bootstrap_args.platform)
    profile = load_platform_config()
    paths = json.loads(config_path("paths.json").read_text())
    default_data_root = Path(paths["host_data_root"])
    if not default_data_root.is_absolute():
        default_data_root = ROOT / default_data_root
    parser = argparse.ArgumentParser(description=f"{TITLE}: LightX2V inference and GSO quality/performance evaluation.")
    parser.add_argument("--platform", default=ACTIVE_PLATFORM, help="Platform profile directory under configs/ (for example mlu, nvidia or metax)")
    card_env = profile.get("card_index_environment", "EVAL_CARD")
    parser.add_argument("--card", type=int, default=int(os.getenv(card_env, str(profile.get("default_card", 0)))))
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts" / ("evaluation-" + datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")))
    parser.add_argument("--data-root", type=Path, default=Path(os.getenv("EVAL_DATA_ROOT", str(default_data_root))))
    parser.add_argument("--image", default=os.getenv("EVAL_IMAGE", profile["container_image"]))
    parser.add_argument("--smoke", action="store_true", help="One object, two repeats, all metric/report stages")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--reuse-runs", type=Path, nargs="+", metavar="LIGHTX2V_RUN", help="Reuse a saved LightX2V run; for older multi-path invocations, only the final path is read")
    parser.add_argument("--install-deps", action="store_true", help="Install pinned extra dependencies before running")
    parser.add_argument("--prepare-sources", action="store_true", help="Fetch the pinned LightX2V source revision; no inference")
    parser.add_argument("--check-inputs", action="store_true", help="Validate prepared data, sources and checkpoint inventory without model execution")
    parser.add_argument("--rehash-weights", action="store_true", help="Verify checkpoint SHA256 in addition to sizes")
    parser.add_argument("--worker", choices=("lightx2v", "postprocess"), help=argparse.SUPPRESS)
    parser.add_argument("--run-dir", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    select_platform(args.platform)
    if args.prepare_sources:
        prepare_sources()
        return 0
    if not 0 <= args.card <= int(profile.get("max_card_index", 7)) or args.repeats < 2 or (args.limit is not None and args.limit <= 0):
        parser.error(f"Require card 0..{profile.get('max_card_index', 7)}, repeats >= 2 and a positive limit")
    if args.smoke:
        args.limit = 1
    if args.reuse_runs and not args.worker and args.limit:
        parser.error("--reuse-runs reads the selected saved runs; do not combine with --smoke/--limit")
    args.data_root = args.data_root.resolve()
    args.output = args.output.resolve()
    if args.reuse_runs:
        args.reuse_runs = [args.reuse_runs[-1].resolve()]
    if args.check_inputs:
        args.output.mkdir(parents=True, exist_ok=False)
        verify_inputs(args, args.output)
        print(f"Inputs verified; audit: {args.output / 'input-audit.json'}")
        return 0
    if not args.image:
        parser.error(f"Set EVAL_IMAGE or configs/{ACTIVE_PLATFORM}/runtime.json:container_image")
    if args.worker == "postprocess":
        return postprocess(args)
    if args.worker:
        args.implementation = args.worker
        args.no_warmup = False
        return run_benchmark_job(args)
    return run_campaign(args)


if __name__ == "__main__":
    raise SystemExit(main())
