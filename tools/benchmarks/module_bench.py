"""Synchronized complete-layer timing; the caller owns model and process groups."""

from __future__ import annotations

import json
import os
import statistics
import sys
import time
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, Callable


@dataclass
class PreparedModule:
    call: Callable[[], Any]
    reset: Callable[[], None]
    synchronize: Callable[[], None]
    identity: dict[str, Any]
    check: Callable[[Any], dict[str, Any]]
    device: str
    process_group: Any = None
    tp_group: Any = None
    sp_group: Any = None


def _gather(value, group):
    import torch.distributed as dist

    if not dist.is_initialized():
        return [value]
    rows = [None] * dist.get_world_size(group)
    dist.all_gather_object(rows, value, group=group)
    return rows


def _phase(call, group):
    """Coordinate Python errors; the launcher must bound hangs inside target collectives."""
    try:
        result = {"value": call(), "error": None}
    except Exception as exc:
        result = {"value": None, "error": f"{type(exc).__name__}: {exc}"}
    rows = _gather(result, group)
    errors = [row["error"] for row in rows if row["error"]]
    if errors:
        raise RuntimeError("; ".join(errors))
    return [row["value"] for row in rows]


def _memory(device, *, peak=False):
    import torch

    if device.type != "cuda":
        return None
    return {
        "allocated_bytes": (torch.cuda.max_memory_allocated if peak else torch.cuda.memory_allocated)(device),
        "reserved_bytes": (torch.cuda.max_memory_reserved if peak else torch.cuda.memory_reserved)(device),
    }


def benchmark(prepared: PreparedModule, *, warmup=10, iterations=30, repeats=3):
    """Time synchronized completion and take the slowest rank; exclude reset/check/barrier."""
    import torch
    import torch.distributed as dist

    device = torch.device(prepared.device)
    context = torch.cuda.device(device) if device.type == "cuda" else nullcontext()
    with context:

        def validate():
            if any(type(v) is not int or v < low for v, low in ((warmup, 0), (iterations, 1), (repeats, 1))):
                raise ValueError("warmup >= 0, iterations/repeats >= 1 must be integers")
            if device.type not in {"cuda", "cpu"} or (device.type == "cuda" and not torch.cuda.is_available()):
                raise ValueError("device must be an available CUDA device or CPU")
            if not dist.is_initialized() and (any(g is not None for g in (prepared.process_group, prepared.tp_group, prepared.sp_group)) or int(os.environ.get("WORLD_SIZE", "1")) > 1):
                raise ValueError("factory must initialize the production process groups")
            if any(not callable(getattr(prepared, name)) for name in ("call", "reset", "synchronize", "check")):
                raise ValueError("call/reset/synchronize/check must be callable")
            identity = prepared.identity
            if not isinstance(identity, dict) or any(not isinstance(identity.get(k), dict) or not identity[k] for k in ("target", "input", "source")):
                raise ValueError("identity requires nonempty target/input/source objects")
            if identity["input"].get("kind") not in {"real", "synthetic"}:
                raise ValueError("identity.input.kind must be real or synthetic")
            json.dumps(identity, allow_nan=False)
            rank = dist.get_rank() if dist.is_initialized() else 0
            groups = {"measurement_ranks": [rank], "tp_ranks": [rank], "sp_ranks": [rank]}
            if dist.is_initialized():
                for name, group in (("measurement_ranks", prepared.process_group), ("tp_ranks", prepared.tp_group), ("sp_ranks", prepared.sp_group)):
                    if group is None and name != "measurement_ranks":
                        continue  # This parallel dimension is disabled; do not infer it from WORLD_SIZE.
                    groups[name] = dist.get_process_group_ranks(group if group is not None else dist.group.WORLD)
                    if rank not in groups[name]:
                        raise ValueError(f"rank {rank} is not a member of {name}")
                if any(not set(groups[k]) <= set(groups["measurement_ranks"]) for k in ("tp_ranks", "sp_ranks")):
                    raise ValueError("TP/SP group extends outside the measurement group")
            return {"rank": rank, "identity": identity, "groups": groups, "parameters": {"warmup": warmup, "iterations": iterations, "repeats": repeats}}

        metadata = _phase(validate, prepared.process_group)
        if any(row["identity"] != metadata[0]["identity"] or row["parameters"] != metadata[0]["parameters"] for row in metadata):
            raise ValueError("all ranks must declare the same target/input/source and measurement parameters")
        rank = dist.get_rank() if dist.is_initialized() else 0
        local = next(row for row in metadata if row["rank"] == rank)

        def reset():
            prepared.reset()
            prepared.synchronize()

        preflight_output = None

        def preflight_call():
            nonlocal preflight_output
            preflight_output = prepared.call()
            prepared.synchronize()

        def preflight_check():
            check = prepared.check(preflight_output)
            if not isinstance(check, dict) or check.get("status") not in {"strict_equal", "pending_review", "unavailable"}:
                raise ValueError("check.status must be strict_equal, pending_review or unavailable; raise on invalid outputs")
            if not isinstance(check.get("scope"), str) or not check["scope"]:
                raise ValueError("check.scope must describe the compared output")
            json.dumps(check, allow_nan=False)
            return check

        def prepare_iteration():
            _phase(reset, prepared.process_group)
            if dist.is_initialized():
                dist.barrier(group=prepared.process_group)
                _phase(prepared.synchronize, prepared.process_group)

        def invoke():
            prepared.call()  # Release outputs each iteration; check/export only in preflight.
            prepared.synchronize()

        samples = []
        peak = end = None

        def measure_one(repeat, iteration):
            nonlocal peak, end
            # Read the target allocator peak before tool-side object collectives.
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            begin = time.perf_counter()
            invoke()
            samples.append({"repeat": repeat, "iteration": iteration, "latency_ms": (time.perf_counter() - begin) * 1000})
            current = _memory(device, peak=True)
            if current is not None:
                peak = {key: max(current[key], (peak or current)[key]) for key in current}
            end = _memory(device)

        try:
            _phase(reset, prepared.process_group)
            _phase(preflight_call, prepared.process_group)
            checks = _phase(preflight_check, prepared.process_group)
            preflight_output = None
            for _ in range(warmup):
                prepare_iteration()
                _phase(invoke, prepared.process_group)
            _phase(reset, prepared.process_group)
            baseline = _memory(device)
            for repeat in range(repeats):
                for iteration in range(iterations):
                    prepare_iteration()
                    _phase(lambda: measure_one(repeat, iteration), prepared.process_group)
        finally:
            preflight_output = None
            error = sys.exc_info()[1]
            try:
                _phase(reset, prepared.process_group)
            except Exception as cleanup:
                if error is None:
                    raise
                print(f"cleanup also failed: {cleanup}", file=sys.stderr)
        increment = {k: peak[k] - baseline[k] for k in peak} if peak else None
        record = {**local, "correctness": checks[metadata.index(local)], "memory": {"warmed_baseline": baseline, "measurement_peak": peak, "end": end, "peak_increment": increment}, "samples": samples}
        raw = _gather(record, prepared.process_group)

    maxima = [max(row["samples"][i]["latency_ms"] for row in raw) for i in range(iterations * repeats)]
    report = {
        "kind": "module_benchmark_v1",
        "identity": prepared.identity,
        "parameters": local["parameters"],
        "timing": {"primary": "synchronized_host_wall", "rank_reduction": "max_per_iteration", "barrier": "outside_each_iteration"},
        "latency_ms": {"mean": statistics.mean(maxima), "median": statistics.median(maxima), "min": min(maxima), "max": max(maxima), "stdev": statistics.pstdev(maxima)},
        "max_rank_latency_ms": maxima,
        "ranks": [{k: row[k] for k in ("rank", "groups", "correctness", "memory")} for row in raw],
    }
    return report, raw


def compare(reports):
    """Compare identical global target/input/source identities without choosing defaults."""
    if not reports or any(r.get("kind") != "module_benchmark_v1" for r in reports):
        raise ValueError("module_benchmark_v1 reports required")
    if any(r["identity"] != reports[0]["identity"] for r in reports):
        raise ValueError("comparison requires identical global target/input/source/hardware identity")
    rows = []
    for report in reports:
        ranks = report["ranks"]
        degrees = {(len(r["groups"]["tp_ranks"]), len(r["groups"]["sp_ranks"])) for r in ranks}
        if len(degrees) != 1:
            raise ValueError("comparison requires uniform declared TP/SP degrees per case")
        tp, sp = degrees.pop()

        def max_memory(field):
            values = [r["memory"][field]["allocated_bytes"] for r in ranks if r["memory"][field] is not None]
            return max(values, default=None)

        errors = [r["correctness"].get("normalized_rmse") for r in ranks]
        rows.append(
            {
                "tp": tp,
                "sp": sp,
                "rank_count": len(ranks),
                "ranks": ranks,
                "factory": report.get("factory"),
                "screen_exceeded": any(r["correctness"].get("screen_exceeded", False) for r in ranks),
                "latency_ms": report["latency_ms"],
                "max_allocated_bytes": max_memory("measurement_peak"),
                "max_end_allocated_bytes": max_memory("end"),
                "max_peak_increment_bytes": max_memory("peak_increment"),
                "max_normalized_rmse": max((e for e in errors if e is not None), default=None),
                "statuses": sorted({r["correctness"]["status"] for r in ranks}),
                "parameters": report["parameters"],
            }
        )
    return {"kind": "module_comparison_v1", "identity": reports[0]["identity"], "cases": rows}
