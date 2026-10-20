"""Single-device paired A/B latency and output comparison for reconstruction."""

import argparse
import functools
import json
import platform
import time
import traceback
from contextlib import contextmanager, nullcontext
from pathlib import Path

import numpy as np
import torch
from scripts.bench.reconstruction.fixtures import load_requests, sha256, unique_images
from scripts.bench.reconstruction.precision import BF16Audit

from lightx2v.models.runners.runner_factory import build_runner
from lightx2v.utils.set_config import build_startup_config
from lightx2v_platform.base.global_var import AI_DEVICE, PLATFORM


@contextmanager
def record_calls(targets, device):
    events = []
    originals = []
    for label, target, method in targets:
        original = getattr(target, method)

        def timed(*args, _original=original, _label=label, **kwargs):
            start, end = device.Event(enable_timing=True), device.Event(enable_timing=True)
            start.record()
            result = _original(*args, **kwargs)
            end.record()
            events.append((_label, start, end))
            return result

        originals.append((target, method, original))
        setattr(target, method, functools.wraps(original)(timed))
    try:
        yield events
    finally:
        for target, method, original in originals:
            setattr(target, method, original)


def compare_outputs(left, right):
    comparison = {}
    for name in left:
        if left[name].shape != right[name].shape:
            comparison[name] = {"same_shape": False, "left": list(left[name].shape), "right": list(right[name].shape)}
        else:
            error = (left[name].double() - right[name].double()).abs()
            comparison[name] = {"same_shape": True, "max_abs_error": error.max().item() if error.numel() else 0, "exact": torch.equal(left[name], right[name])}
    return comparison


def run_benchmark(model, task, manifest, call_targets, set_variant, request_fields):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config-json")
    parser.add_argument("--requests", type=Path)
    parser.add_argument("--limit", type=int, help="Diagnostic subset; omit for the complete frozen suite")
    parser.add_argument("--audit-bf16", action="store_true", help="Record matrix operand dtypes during excluded warmup")
    args = parser.parse_args()
    device = getattr(torch, AI_DEVICE)
    if device.device_count() != 1:
        raise ValueError("Expose exactly one accelerator device per benchmark process")
    manifest = args.requests or manifest
    requests = load_requests(manifest)
    if args.limit is not None:
        requests = requests[: args.limit]
    inputs = unique_images(requests)
    for relative, image in inputs.items():
        path = args.input_root / relative
        if path.stat().st_size != image["size_bytes"] or sha256(path) != image["sha256"]:
            raise ValueError(f"Input hash mismatch: {relative}")
    config = build_startup_config({"model_cls": model, "task": task, "model_path": args.model_path, "config_json": args.config_json})
    runner = build_runner(config)
    targets = call_targets(runner)
    args.output.mkdir(parents=True, exist_ok=True)
    environment = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "platform": PLATFORM,
        "device": device.get_device_name(0),
        "config": dict(config),
        "manifest_sha256": sha256(manifest),
    }
    (args.output / "environment.json").write_text(json.dumps(environment, indent=2) + "\n")
    records = []
    with (args.output / "requests.jsonl").open("w") as stream:
        for index, request in enumerate(requests):
            outputs = {}
            variants = ["baseline", "optimized"] if index % 2 == 0 else ["optimized", "baseline"]
            for position, variant in enumerate(variants):
                set_variant(runner, variant)
                fields = {"seed": request["seed"]}
                fields.update(request_fields(request, args.input_root, args.output, variant))
                context = runner.prepare_request(fields)
                record = {"id": request["id"], "pair": index, "position": position, "variant": variant, "warmup": index == 0, "seed": request["seed"]}
                try:
                    device.synchronize()
                    start = time.perf_counter()
                    audit = BF16Audit(AI_DEVICE) if args.audit_bf16 and index == 0 else None
                    audit_context = audit if audit is not None else nullcontext()
                    with record_calls(targets, device) as events, audit_context:
                        outputs[variant] = runner.run_request(context)
                    device.synchronize()
                    pipeline = time.perf_counter() - start
                    calls = [{"stage": label, "seconds": begin.elapsed_time(end) / 1000} for label, begin, end in events]
                    finite = all(torch.isfinite(value).all().item() for value in outputs[variant].values())
                    record.update(
                        status="success" if finite else "nonfinite", pipeline_cost_s=pipeline, main_call_seconds=calls, main_call_median_s=float(np.median([call["seconds"] for call in calls]))
                    )
                    if audit is not None:
                        record["bf16_audit"] = audit.report()
                except Exception:
                    record.update(status="failed", error=traceback.format_exc())
                records.append(record)
                stream.write(json.dumps(record) + "\n")
                stream.flush()
                print(json.dumps(record), flush=True)
            if len(outputs) == 2:
                (args.output / f"{request['id']}_comparison.json").write_text(json.dumps(compare_outputs(outputs["baseline"], outputs["optimized"]), indent=2) + "\n")
    effective = [record for record in records if not record["warmup"] and record["status"] == "success"]
    pairs = {record["pair"]: {} for record in effective}
    for record in effective:
        pairs[record["pair"]][record["variant"]] = record["pipeline_cost_s"]
    paired_ids = {index for index, pair in pairs.items() if len(pair) == 2}
    summary = {}
    for variant in ("baseline", "optimized"):
        subset = [record for record in effective if record["variant"] == variant and record["pair"] in paired_ids]
        summary[variant] = {"successful_requests": len(subset), "failed_requests": sum(record["status"] != "success" and record["variant"] == variant for record in records)}
        if subset:
            summary[variant].update(
                pipeline_p50_s=float(np.median([record["pipeline_cost_s"] for record in subset])), main_call_request_median_s=float(np.median([record["main_call_median_s"] for record in subset]))
            )
    changes = [100 * (pair["optimized"] / pair["baseline"] - 1) for pair in pairs.values() if len(pair) == 2]
    summary.update(effective_pairs=len(changes), paired_pipeline_change_median_percent=float(np.median(changes)) if changes else None, grade="formal_candidate" if len(changes) >= 31 else "diagnostic")
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

