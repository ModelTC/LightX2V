"""VGGT single-file evaluator: execute models, measure, score, compare and report.

Host: python evaluate.py [--smoke] [--card 2]
Saved outputs: python evaluate.py --reuse-runs REFERENCE LIGHTX2V
Each model implementation runs in a fresh interpreter; postprocessing uses CPU.
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
PROTOCOL = "深度逐视图中位数尺度对齐后计算AbsRel、RMSE（米）、δ1；相机使用相对位姿AUC@5/10/20度。"
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


def cast_matrices(module):
    """Apply the reference precision policy declared by the selected profile."""
    import torch

    classes = (torch.nn.Linear, torch.nn.Conv1d, torch.nn.Conv2d, torch.nn.Conv3d, torch.nn.ConvTranspose1d, torch.nn.ConvTranspose2d, torch.nn.ConvTranspose3d, torch.nn.Embedding)
    dtype = getattr(torch, load_platform_config().get("reference_weight_dtype", "bfloat16"))
    count = 0
    for child in module.modules():
        if isinstance(child, classes) or type(child).__name__ in ("SparseConv3d", "SparseInverseConv3d"):
            child.to(dtype=dtype)
            count += 1
    return count


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


class ReferenceAdapter:
    def __init__(self, paths, workdir):
        from vggt.models.vggt import VGGT

        self.model = VGGT()
        state = torch.load(paths["vggt"] / "model.pt", map_location="cpu", weights_only=True)
        loaded = self.model.load_state_dict(state, strict=True)
        del state
        self.model.eval().to(device_name())
        converted = cast_matrices(self.model) if load_platform_config().get("cast_reference_matrices", True) else 0
        self.metadata = {
            "strict_checkpoint_load": str(loaded),
            "matrix_modules_cast": converted,
            "source": str(ROOT / "sources/reference-vggt"),
            "track_head": "loaded but not invoked: no query points, matching reconstruction scope",
        }

    def predict(self, image_paths, seed):
        from vggt.utils.load_fn import load_and_preprocess_images_square
        from vggt.utils.pose_enc import pose_encoding_to_extri_intri

        input_start = time.perf_counter()
        images, coords = load_and_preprocess_images_square([str(DATA / p) for p in image_paths], 518)
        input_s = time.perf_counter() - input_start
        forward_start = time.perf_counter()
        with autocast_context():
            pred = self.model(images.to(device_name()))
        device_api().synchronize()
        forward_s = time.perf_counter() - forward_start
        postprocess_start = time.perf_counter()
        extrinsics, intrinsics = pose_encoding_to_extri_intri(pred["pose_enc"].float(), (518, 518))
        pred.update(extrinsics=extrinsics, intrinsics=intrinsics, original_coords=coords)
        names = ("pose_enc", "depth", "depth_conf", "world_points", "world_points_conf", "extrinsics", "intrinsics", "original_coords")
        outputs = {name: pred[name].float().cpu() for name in names}
        device_api().synchronize()
        self.last_stage_timings = {
            "image_read_and_preprocess_s": input_s,
            "model_forward_s": forward_s,
            "prediction_postprocess_and_cpu_transfer_s": time.perf_counter() - postprocess_start,
        }
        return outputs


def measurement_targets(adapter, implementation):
    """Choose optional developer timing points outside the inference adapter."""
    if implementation == "lightx2v":
        return [(adapter.runner.model.transformer_infer, "infer", "vggt_forward")]
    return [(adapter.model, "forward", "vggt_forward")]


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
    source = ROOT / "sources" / f"{args.implementation}-vggt"
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
    if args.implementation == "lightx2v":
        adapter = LightX2VAdapter(paths, args.run_dir)
    else:
        adapter = ReferenceAdapter(paths, args.run_dir)
    backend.synchronize()
    env.update(model_load_s=time.perf_counter() - start, adapter=adapter.metadata)
    write_json(args.run_dir / "environment.json", env)
    targets = measurement_targets(adapter, args.implementation)
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
    run = Path(run)
    environment = environment or json.loads((run / "environment.json").read_text())
    assert environment["model"] == MODEL, "Run belongs to another model"
    if "platform_configuration" in environment:
        assert environment["platform_configuration"] == platform_fingerprints(), "Platform runtime or model configuration differs"
    origin = json.loads((config_path("evaluation-origin.json")).read_text())
    historical = origin["runs"].get(run.name)
    if historical and sha256(run / "environment.json") == historical["environment_sha256"]:
        assert sha256(run / "requests.jsonl") == historical["requests_sha256"], "Historical requests changed"
        assert sha256(config_path("suite.json")) == origin["project_suite_sha256"], "Frozen protocol changed"
        assert sha256(config_path("source-lock.json")) == origin["project_source_lock_sha256"], "Frozen sources changed"
        assert environment["suite_sha256"] == origin["historical_suite_sha256"]
        assert environment["source_lock_sha256"] == origin["historical_source_lock_sha256"]
        return "historical_byte_preserved"
    pair = {"suite_sha256": environment["suite_sha256"], "source_lock_sha256": environment["source_lock_sha256"]}
    if pair in origin.get("configuration_aliases", []):
        assert sha256(config_path("suite.json")) == origin["project_suite_sha256"], "Frozen protocol changed"
        assert sha256(config_path("source-lock.json")) == origin["project_source_lock_sha256"], "Frozen sources changed"
        return "compatible_frozen_source_inventory"
    assert environment["suite_sha256"] == sha256(config_path("suite.json")), "Suite differs"
    assert environment["source_lock_sha256"] == sha256(config_path("source-lock.json")), "Sources differ"
    return "current_project"


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


def rotation_angle(matrix):
    return float(np.degrees(np.arccos(np.clip((np.trace(matrix) - 1) / 2, -1, 1))))


def vggt_metrics(output, sample, data):
    gt_dir = data / sample["gt_directory"]
    predicted = output["depth"][0, ..., 0].astype(np.float64)
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
    return {"depth_views": views, "camera_pairs": pairs, "depth_macro": {key: float(np.mean([v[key] for v in views])) for key in ("abs_rel", "rmse_m", "delta1")}}


def auc(errors, threshold):
    errors = np.sort(np.asarray(errors))
    if not len(errors):
        return None
    recalls = np.arange(1, len(errors) + 1) / len(errors)
    n = np.searchsorted(errors, threshold, side="right")
    x = np.r_[0.0, errors[:n], threshold]
    y = np.r_[0.0, recalls[:n], recalls[n - 1] if n else 0.0]
    return float(np.trapezoid(y, x) / threshold)


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
    """Export a representative input/GT/prediction figure from saved tensors."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    left = {(r["id"], r["repeat"]): r for r in load_rows(runs["reference"]) if r["status"] == "success" and not r["warmup"]}
    right = {(r["id"], r["repeat"]): r for r in load_rows(runs["lightx2v"]) if r["status"] == "success" and not r["warmup"]}
    sample = next((s for s in samples if (s["id"], 0) in left and (s["id"], 0) in right), None)
    if sample is None:
        return []
    key = (sample["id"], 0)
    gt = resize_depth(data / sample["gt_directory"] / "depth_00.png", 518)
    valid = gt > 1e-6
    predictions = []
    for name, rows in (("reference", left), ("lightx2v", right)):
        with np.load(runs[name] / rows[key]["output"]) as output:
            depth = output["depth"][0, 0, ..., 0].copy()
        depth *= np.median(gt[valid]) / np.median(depth[valid])
        predictions.append(depth)
    with Image.open(data / sample["views"][0]) as image:
        rgb = image.convert("RGB").resize((518, 518), Image.Resampling.BICUBIC)
    fig, axes = plt.subplots(2, 3, figsize=(12, 7), constrained_layout=True)
    axes[0, 0].imshow(rgb)
    axes[0, 0].set_title("Input")
    low, high = np.percentile(gt[valid], [1, 99])
    label = load_platform_config()["platform"].upper()
    for axis, depth, title in ((axes[0, 1], gt, "GT (m)"), (axes[0, 2], predictions[0], f"Original {label}"), (axes[1, 0], predictions[1], f"LightX2V {label}")):
        artist = axis.imshow(np.ma.array(depth, mask=~valid), vmin=low, vmax=high, cmap="viridis")
        axis.set_title(title)
        fig.colorbar(artist, ax=axis, shrink=0.7)
    limit = max(float(np.abs(depth[valid] - gt[valid]).max()) for depth in predictions)
    for axis, depth, title in ((axes[1, 1], predictions[0], "Original - GT"), (axes[1, 2], predictions[1], "LightX2V - GT")):
        artist = axis.imshow(np.ma.array(depth - gt, mask=~valid), vmin=-limit, vmax=limit, cmap="coolwarm")
        axis.set_title(title)
        fig.colorbar(artist, ax=axis, shrink=0.7)
    for axis in axes.flat:
        axis.axis("off")
    path = destination / "depth-example.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return [{"kind": "depth_figure", "sample": sample["id"], "file": path.name, "sha256": sha256(path)}]


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
    lines = [
        f"# {TITLE} 评测报告",
        "",
        f"状态：{summary['status']}；范围：{summary['scope']}；每实现计划请求：{summary['expected_per_implementation']}。",
        "",
        "本次方式：" + ("复用已有预测重新计算指标，时间/显存来自原请求记录。" if summary["reused_predictions"] else "本次实际运行两种实现并计算指标。"),
        "",
        f"| 指标 | 原项目 {load_platform_config()['platform'].upper()} | LightX2V {load_platform_config()['platform'].upper()} |",
        "|---|---:|---:|",
    ]
    sides = [report[key] for key in ("reference", "lightx2v")]
    for key in ("success_count", "failed_count", "missing_measured_count", "success_rate"):
        lines.append(f"| {key} | {number(sides[0].get(key))} | {number(sides[1].get(key))} |")
    metrics = [flatten_numbers(side["quality"], "quality") | flatten_numbers(side["performance"], "performance") for side in sides]
    for key in dict.fromkeys(k for row in metrics for k in row):
        lines.append(f"| {key} | {number(metrics[0].get(key))} | {number(metrics[1].get(key))} |")
    lines += [
        "",
        "## 固定随机性与输出差值",
        "",
        "- 原项目重复：" + repeat_status(report["reference_repeat_differences"]),
        "- LightX2V重复：" + repeat_status(report["lightx2v_repeat_differences"]),
        "",
        "| 字段 | 可逐元素比较对象/配对对象 | 全元素MAE | 全元素RMSE | 最大绝对差 |",
        "|---|---:|---:|---:|---:|",
    ]
    lines += field_summary(report["implementation_differences"])
    if MODEL == "trellis2":
        comparisons = [row["comparison"] for row in report["implementation_differences"]]
        distances = [r["raw_vertex_set_distance"]["symmetric_mean_nearest_distance"] for r in comparisons if "raw_vertex_set_distance" in r]
        overlaps = [r["texture_alignment"]["coordinate_iou"] for r in comparisons if "coordinate_iou" in r.get("texture_alignment", {})]
        lines += [
            "",
            "原始顶点集合距离宏均值：" + number(sum(distances) / len(distances) if distances else None),
            "纹理坐标IoU宏均值：" + number(sum(overlaps) / len(overlaps) if overlaps else None),
            "坐标交集上的完整纹理属性差值见 comparison.json；原始纹理体素保存在NPZ。",
        ]
    lines += [
        "",
        "## 数据与计时边界",
        "",
        PROTOCOL,
        "",
        f"固定GSO对象、输入顺序、seed和精度策略；平台为 {load_platform_config()['platform']}。",
        "同步墙钟计时覆盖读图、预处理、模型执行和CPU输出，排除模型加载、预热、NPZ写盘及离线指标。",
        "原始输出差值与尺度/坐标对齐后的数据集精度分别报告；形状不一致不记零差。",
        "小样本检查仅代表所选对象，不代表完整1023对象GSO评测。",
        "",
        "## 产物",
        "",
        "- comparison.json：全部逐对象精度、性能统计、实现间及重复差值。",
        "- requests.csv / metrics.csv / summary.csv：逐请求和指标表。",
        "- summary.json / failures.json / provenance.json：完成状态、错误和来源。",
        "- reference/、lightx2v/：新运行的NPZ、环境、请求、遥测；复用模式的数据位置见 summary.json。",
        "- extra-output-manifest.json：图或网格的路径、哈希及导出检查。",
    ]
    if summary["errors"]:
        lines += ["", "## 未完成或失败项目", ""]
        lines += ["- " + row["stage"] + ": " + row["error"].splitlines()[-1] for row in summary["errors"]]
    return "\n".join(lines) + "\n"


def postprocess(args):
    """Compute every metric and report in a CPU process after model processes exit."""
    load_numeric_dependencies()
    destination = args.output
    manifest = json.loads((destination / "manifest.json").read_text())
    suite = json.loads((config_path("suite.json")).read_text())
    selected = suite["samples"][: manifest["sample_count"]]
    suite = dict(suite, samples=selected)
    runs = dict(zip(("reference", "lightx2v"), args.reuse_runs))
    expected = manifest["sample_count"] * manifest["repeats"]
    report = {"model": MODEL, "evaluation_code_sha256": sha256(Path(__file__)), "suite_sha256": sha256(config_path("suite.json"))}
    errors = [{"stage": stage, "error": f"Worker exited with code {code}"} for stage, code in manifest.get("stage_exit_codes", {}).items() if code]
    request_table, metric_table = [], []
    environments = {}
    for implementation, run in runs.items():
        try:
            env = json.loads((run / "environment.json").read_text())
            verify_environment(run, env)
            if env["implementation"] != implementation or env["sample_count"] != len(selected) or env["repeats"] != manifest["repeats"]:
                raise ValueError("Run implementation/sample count/repeats differ from this evaluation")
            environments[implementation] = env
            report[implementation] = evaluate_run(run, suite, args.data_root)
        except Exception:
            error = traceback.format_exc()
            errors.append({"stage": implementation, "error": error})
            report[implementation] = empty_evaluation(run, expected, error)
        write_json(destination / f"{implementation}-metrics.json", report[implementation])
        try:
            request_rows = load_rows(run)
        except Exception:
            request_rows = []
            errors.append({"stage": implementation + "_request_log", "error": traceback.format_exc()})
        for row in request_rows:
            request_table.append(
                {
                    "implementation": implementation,
                    **{
                        key: row.get(key)
                        for key in (
                            "id",
                            "seed",
                            "repeat",
                            "warmup",
                            "status",
                            "pipeline_s",
                            "peak_allocated_bytes",
                            "peak_reserved_bytes",
                            "peak_incremental_bytes",
                            "cpu_process_peak_rss_kib",
                            "artifact_write_s",
                            "output",
                            "output_sha256",
                            "error",
                        )
                    },
                    "events_json": json.dumps(row.get("events", [])),
                }
            )
        for row in report[implementation]["per_output_metrics"]:
            metric_table.append({"implementation": implementation, "id": row["id"], "repeat": row["repeat"], **flatten_numbers(row["metrics"])})
    if len(environments) == 2 and environments["reference"]["versions"] != environments["lightx2v"]["versions"]:
        errors.append({"stage": "environment", "error": "Runtime dependency versions differ between implementations"})
    for name, left, right, repeat in (
        ("implementation_differences", runs["reference"], runs["lightx2v"], False),
        ("reference_repeat_differences", runs["reference"], runs["reference"], True),
        ("lightx2v_repeat_differences", runs["lightx2v"], runs["lightx2v"], True),
    ):
        try:
            report[name] = pair_runs(left, right, repeat)
        except Exception:
            report[name] = []
            errors.append({"stage": name, "error": traceback.format_exc()})
        if len(report[name]) != len(selected):
            errors.append({"stage": name, "error": f"Expected {len(selected)} object pairs, got {len(report[name])}"})
    for implementation in runs:
        side = report[implementation]
        errors += [{"stage": implementation + "_inference", "error": r["error"]} for r in side["inference_errors"]]
        errors += [{"stage": implementation + "_metric", "error": r["error"]} for r in side["evaluation_errors"]]
        if side["success_count"] != expected or side["missing_measured_count"] or not side["finished_marker"]:
            errors.append({"stage": implementation, "error": f"Success={side['success_count']}, expected={expected}, missing={side['missing_measured_count']}, finished={side['finished_marker']}"})
    extras = []
    try:
        extras = export_additional_outputs(runs, selected, args.data_root, destination)
    except Exception:
        errors.append({"stage": "extra_outputs", "error": traceback.format_exc()})
    write_json(destination / "extra-output-manifest.json", extras)
    write_json(destination / "comparison.json", report)
    write_csv(destination / "requests.csv", request_table)
    write_csv(destination / "metrics.csv", metric_table)
    write_csv(
        destination / "summary.csv",
        [
            {
                "implementation": key,
                **{k: report[key][k] for k in ("success_count", "failed_count", "missing_measured_count", "success_rate")},
                **flatten_numbers(report[key]["quality"], "quality"),
                **flatten_numbers(report[key]["performance"], "performance"),
            }
            for key in runs
        ],
    )
    summary = {
        "status": "complete" if not errors else "incomplete",
        "model": MODEL,
        "scope": "full_gso_1023" if len(selected) == 1023 and args.limit is None else "diagnostic_subset",
        "sample_count": len(selected),
        "repeats": manifest["repeats"],
        "expected_per_implementation": expected,
        "reused_predictions": manifest["reused_predictions"],
        "runs": {key: str(path) for key, path in runs.items()},
        "results": {key: {k: report[key][k] for k in ("success_count", "failed_count", "missing_measured_count", "quality", "performance")} for key in runs},
        "errors": errors,
    }
    write_json(destination / "failures.json", errors)
    write_json(destination / "summary.json", summary)
    (destination / "report.md").write_text(make_markdown(report, summary))
    files = [
        {"path": str(p.relative_to(destination)), "bytes": p.stat().st_size, "sha256": sha256(p)}
        for p in sorted(destination.rglob("*"))
        if p.is_file() and p.name not in ("provenance.json", "postprocess.log")
    ]
    write_json(
        destination / "provenance.json",
        {
            "model": MODEL,
            "evaluation_code_sha256": sha256(Path(__file__)),
            "suite_sha256": sha256(config_path("suite.json")),
            "source_lock_sha256": sha256(config_path("source-lock.json")),
            "weight_lock_sha256": sha256(config_path("weight-lock.json")),
            "platform_configuration": platform_fingerprints(),
            "files": files,
        },
    )
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


def apply_reference_edits(directory, patch_path, expected_sha256):
    if sha256(patch_path) != expected_sha256:
        raise ValueError(f"Compatibility patch hash differs: {patch_path}")
    edits = json.loads(patch_path.read_text())
    if edits["format"] != "line_edits_v1":
        raise ValueError("Unknown reference compatibility patch format")
    for file in edits["files"]:
        relative = Path(file["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Reference patch path escapes its source directory")
        path = directory / relative
        if file["base_sha256"] is None:
            if path.exists():
                raise ValueError(f"Expected a new reference file: {relative}")
            lines = []
        else:
            if sha256(path) != file["base_sha256"]:
                raise ValueError(f"Official source does not match pinned patch base: {relative}")
            lines = path.read_text().splitlines(keepends=True)
        for hunk in reversed(file["hunks"]):
            start, count = hunk["start"], hunk["delete"]
            if start < 0 or count < 0 or start + count > len(lines):
                raise ValueError("Reference patch line range is invalid")
            lines[start : start + count] = hunk["insert"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(lines))
        if sha256(path) != file["result_sha256"]:
            raise ValueError(f"Reference compatibility result differs: {relative}")


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
            if record.get("patches"):
                apply_reference_edits(temporary, ROOT / record["patches"], record["patches_sha256"])
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
        for implementation, path in zip(("reference", "lightx2v"), args.reuse_runs or [destination / "reference", destination / "lightx2v"]):
            if not args.reuse_runs:
                run_paths.append(f"/results/{implementation}")
            else:
                target = f"/saved/{implementation}/{path.name}"
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
            for stage in ("reference", "lightx2v"):
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
        summary["runs"] = {key: str(path) for key, path in zip(("reference", "lightx2v"), args.reuse_runs or [destination / "reference", destination / "lightx2v"])}
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
    parser = argparse.ArgumentParser(description=f"{TITLE}: model execution, performance, accuracy, differences and reports in one command.")
    parser.add_argument("--platform", default=ACTIVE_PLATFORM, help="Platform profile directory under configs/ (for example mlu, nvidia or metax)")
    card_env = profile.get("card_index_environment", "EVAL_CARD")
    parser.add_argument("--card", type=int, default=int(os.getenv(card_env, str(profile.get("default_card", 0)))))
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts" / ("evaluation-" + datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")))
    parser.add_argument("--data-root", type=Path, default=Path(os.getenv("EVAL_DATA_ROOT", str(default_data_root))))
    parser.add_argument("--image", default=os.getenv("EVAL_IMAGE", profile["container_image"]))
    parser.add_argument("--smoke", action="store_true", help="One object, two repeats, all metric/report stages")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--reuse-runs", type=Path, nargs=2, metavar=("REFERENCE", "LIGHTX2V"), help="Reuse saved predictions; do not run models or remeasure time")
    parser.add_argument("--install-deps", action="store_true", help="Install pinned extra dependencies before running")
    parser.add_argument("--prepare-sources", action="store_true", help="Fetch pinned model implementations and apply the selected profile's compatibility edits; no inference")
    parser.add_argument("--check-inputs", action="store_true", help="Validate prepared data, sources and checkpoint inventory without model execution")
    parser.add_argument("--rehash-weights", action="store_true", help="Verify checkpoint SHA256 in addition to sizes")
    parser.add_argument("--worker", choices=("reference", "lightx2v", "postprocess"), help=argparse.SUPPRESS)
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
        args.reuse_runs = [path.resolve() for path in args.reuse_runs]
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
