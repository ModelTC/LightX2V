"""Shape-driven single-GPU operator benchmark."""

from __future__ import annotations

import argparse
import itertools
import json
import os
import platform
import re
import statistics
import subprocess
import time
import traceback
from collections import Counter, defaultdict
from datetime import datetime, timezone
from hashlib import sha1
from pathlib import Path
from typing import Any, Iterable

from tools.benchmarks.operator_backends import BackendAdapter, load_registry, probe_backend_catalog

FAMILIES = {"gemm", "dense_attention", "sparse_attention", "moe", "convolution"}
ATTENTION_SHAPE_FIELDS = {"batch", "seq_q", "seq_kv", "heads", "kv_heads", "head_dim", "causal"}
SHAPE_FIELDS = {
    "gemm": {"m", "n", "k", "bias"},
    "convolution": {"input", "weight", "stride", "padding", "dilation", "groups", "bias"},
    "dense_attention": ATTENTION_SHAPE_FIELDS,
    "sparse_attention": ATTENTION_SHAPE_FIELDS,
    "moe": {
        "tokens",
        "hidden_size",
        "intermediate_size",
        "num_experts",
        "top_k",
        "activation",
        "expert_bias",
    },
}
TIMING_MODE = "event_list_single_sync"
RATE_SCHEMA = "precision_units_v2"
RATE_UNITS = {
    "tflops": "TFLOPS",
    "actual_tflops": "TFLOPS",
    "dense_equivalent_tflops": "TFLOPS",
    "effective_tflops": "TFLOPS",
    "tops": "TOPS",
    "effective_tops": "TOPS",
}
DTYPE_ALIASES = {
    "bfloat16": "bf16",
    "torch.bfloat16": "bf16",
    "float16": "fp16",
    "torch.float16": "fp16",
    "half": "fp16",
    "float32": "fp32",
    "torch.float32": "fp32",
}
CASE_FIELDS = {
    "case_id",
    "operator_family",
    "operator_name",
    "shape",
    "precision",
    "source",
    "tags",
    "call_count",
    "observed_backend",
    "routing",
    "replay",
    "sparse",
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def normalize_dtype(value: Any, default: str = "bf16") -> str:
    dtype = str(value or default).lower()
    return DTYPE_ALIASES.get(dtype, dtype)


def _positive_int(value: Any, label: str) -> int:
    require(not isinstance(value, bool), f"{label} must be a positive integer")
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        raise ValueError(f"{label} must be a positive integer") from None
    require(parsed > 0, f"{label} must be a positive integer")
    if isinstance(value, float):
        require(value.is_integer(), f"{label} must be a positive integer")
    return parsed


def _boolean(value: Any, label: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in {0, 1}:
        return bool(value)
    if isinstance(value, str) and value.lower() in {"true", "false"}:
        return value.lower() == "true"
    raise ValueError(f"{label} must be a boolean")


def normalize_shape(family: str, raw: dict[str, Any]) -> dict[str, Any]:
    require(isinstance(raw, dict), f"{family} shape must be an object")
    if family == "gemm":
        return {
            "m": _positive_int(raw.get("m"), "gemm.m"),
            "n": _positive_int(raw.get("n"), "gemm.n"),
            "k": _positive_int(raw.get("k"), "gemm.k"),
            "bias": _boolean(raw.get("bias", False), "gemm.bias"),
        }
    if family in {"dense_attention", "sparse_attention"}:
        heads = _positive_int(raw.get("heads"), "attention.heads")
        kv_heads = _positive_int(raw.get("kv_heads", heads), "attention.kv_heads")
        seq_q = _positive_int(raw.get("seq_q"), "attention.seq_q")
        seq_kv = _positive_int(raw.get("seq_kv"), "attention.seq_kv")
        causal = _boolean(raw.get("causal", False), "attention.causal")
        require(heads % kv_heads == 0, "attention.heads must be divisible by kv_heads")
        require(not causal or seq_q == seq_kv, "causal attention requires seq_q == seq_kv")
        return {
            "batch": _positive_int(raw.get("batch", 1), "attention.batch"),
            "seq_q": seq_q,
            "seq_kv": seq_kv,
            "heads": heads,
            "kv_heads": kv_heads,
            "head_dim": _positive_int(raw.get("head_dim"), "attention.head_dim"),
            "causal": causal,
        }
    if family == "convolution":
        x, weight = raw.get("input"), raw.get("weight")
        require(isinstance(x, list) and len(x) in {4, 5}, "convolution.input must be NCHW or NCDHW")
        require(isinstance(weight, list) and len(weight) == len(x), "convolution.weight rank mismatch")
        x = [_positive_int(value, "convolution.input") for value in x]
        weight = [_positive_int(value, "convolution.weight") for value in weight]
        groups = _positive_int(raw.get("groups", 1), "convolution.groups")
        require(x[1] == weight[1] * groups and weight[0] % groups == 0, "convolution channel/group mismatch")
        result = {"input": x, "weight": weight, "groups": groups, "bias": _boolean(raw.get("bias", False), "convolution.bias")}
        for name, default in (("stride", 1), ("padding", 0), ("dilation", 1)):
            values = raw.get(name, [default] * (len(x) - 2))
            require(isinstance(values, list) and len(values) == len(x) - 2, f"convolution.{name} rank mismatch")
            require(all(type(value) is int and value >= (0 if name == "padding" else 1) for value in values), f"invalid convolution.{name}")
            result[name] = values
        require(all(x[i + 2] + 2 * result["padding"][i] >= result["dilation"][i] * (weight[i + 2] - 1) + 1 for i in range(len(x) - 2)), "convolution output must be positive")
        return result
    if family == "moe":
        experts = _positive_int(raw.get("num_experts"), "moe.num_experts")
        top_k = _positive_int(raw.get("top_k"), "moe.top_k")
        activation = str(raw.get("activation") or "swiglu")
        require(top_k <= experts, "moe.top_k cannot exceed num_experts")
        require(activation in {"gelu", "swiglu"}, f"unsupported MoE activation: {activation}")
        return {
            "tokens": _positive_int(raw.get("tokens"), "moe.tokens"),
            "hidden_size": _positive_int(raw.get("hidden_size"), "moe.hidden_size"),
            "intermediate_size": _positive_int(raw.get("intermediate_size"), "moe.intermediate_size"),
            "num_experts": experts,
            "top_k": top_k,
            "activation": activation,
            "expert_bias": _boolean(raw.get("expert_bias", False), "moe.expert_bias"),
        }
    raise ValueError(f"unsupported operator family: {family}")


def _validate_routing(case: dict[str, Any], required: bool = False) -> None:
    routing = case.get("routing")
    if routing is None:
        require(not required, f"observed MoE routing is missing: {case['case_id']}")
        return
    counts = routing.get("expert_counts") if isinstance(routing, dict) else None
    shape = case["shape"]
    valid = (
        isinstance(counts, list)
        and len(counts) == shape["num_experts"]
        and all(type(value) is int and 0 <= value <= shape["tokens"] for value in counts)
        and sum(counts) == shape["tokens"] * shape["top_k"]
    )
    require(valid, f"invalid observed MoE expert_counts: {case['case_id']}")


def _validate_replay(case: dict[str, Any], label: str) -> None:
    replay = case.get("replay")
    require(isinstance(replay, dict), f"{label} replay is required: {case['case_id']}")
    require(set(replay) == {"manifest", "sha256"}, f"invalid {label} replay fields: {case['case_id']}")
    require(isinstance(replay["manifest"], str) and replay["manifest"], f"{label} replay manifest is required: {case['case_id']}")
    require(bool(re.fullmatch(r"[0-9a-f]{64}", str(replay["sha256"]))), f"invalid {label} replay sha256: {case['case_id']}")


def _validate_sparse_replay(case: dict[str, Any]) -> None:
    _validate_replay(case, "sparse")
    sparse = case.get("sparse")
    require(isinstance(sparse, dict) and set(sparse) == {"keep_ratio"}, f"sparse.keep_ratio is required: {case['case_id']}")
    keep_ratio = sparse["keep_ratio"]
    require(isinstance(keep_ratio, (int, float)) and not isinstance(keep_ratio, bool), f"invalid sparse keep_ratio: {case['case_id']}")
    require(0 < float(keep_ratio) <= 1, f"sparse keep_ratio must be in (0, 1]: {case['case_id']}")
    require(case["shape"]["batch"] == 1, f"sparse replay currently requires batch=1: {case['case_id']}")


def validate_suite(value: dict[str, Any]) -> dict[str, Any]:
    require(isinstance(value, dict), "shape suite must be an object")
    require(value.get("schema_version") == 1, "shape suite schema_version must be 1")
    require(value.get("kind") == "operator_benchmark_shape_suite_v1", "unsupported shape suite kind")
    require(isinstance(value.get("suite_id"), str) and value["suite_id"], "suite_id is required")
    cases = value.get("cases")
    require(isinstance(cases, list) and cases, "shape suite must contain cases")
    ids = []
    for case in cases:
        require(isinstance(case, dict), "suite case must be an object")
        unknown = sorted(set(case) - CASE_FIELDS)
        require(not unknown, f"suite case contains unknown fields: {unknown}")
        require(isinstance(case.get("case_id"), str) and case["case_id"], "case_id is required")
        family = case.get("operator_family")
        require(family in FAMILIES, f"unsupported operator family: {family}")
        require(case.get("shape") == normalize_shape(family, case.get("shape") or {}), f"non-canonical shape: {case['case_id']}")
        precision = case.get("precision")
        require(isinstance(precision, dict) and precision.get("input_dtype"), f"input_dtype is required: {case['case_id']}")
        require(normalize_dtype(precision["input_dtype"]) == precision["input_dtype"], f"non-canonical dtype: {case['case_id']}")
        if case.get("call_count") is not None:
            require(_positive_int(case["call_count"], "call_count") == case["call_count"], "invalid call_count")
        if family == "moe" and case.get("routing") is not None:
            _validate_routing(case)
        if family == "sparse_attention":
            _validate_sparse_replay(case)
        elif "replay" in case:
            require(family in {"dense_attention", "moe"}, f"{family} does not support replay inputs")
            _validate_replay(case, "dense" if family == "dense_attention" else "MoE")
        ids.append(case["case_id"])
    require(len(ids) == len(set(ids)), "case_id values must be unique")
    return value


def load_suite(path: Path) -> dict[str, Any]:
    return validate_suite(json.loads(path.read_text(encoding="utf-8")))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def inspect_suite(value: dict[str, Any]) -> dict[str, Any]:
    validate_suite(value)
    cases = value["cases"]
    return {
        "suite_id": value["suite_id"],
        "case_count": len(cases),
        "family_case_counts": dict(sorted(Counter(case["operator_family"] for case in cases).items())),
        "dtype_case_counts": dict(sorted(Counter(case["precision"]["input_dtype"] for case in cases).items())),
        "missing_call_count": [case["case_id"] for case in cases if case.get("call_count") is None],
        "missing_observed_backend": [case["case_id"] for case in cases if case.get("observed_backend") is None],
    }


def generate_sweep(family: str, axes: dict[str, list[Any]], suite_id: str, dtype: str, max_cases: int = 10000) -> dict[str, Any]:
    require(family in FAMILIES, f"unsupported sweep family: {family}")
    require(family != "sparse_attention", "sparse attention requires a real replay artifact and cannot be swept")
    require(bool(axes), "sweep requires axes")
    unknown = set(axes) - SHAPE_FIELDS[family]
    require(not unknown, f"unknown {family} sweep axes: {sorted(unknown)}")
    names = sorted(axes)
    combinations = list(itertools.product(*(axes[name] for name in names)))
    require(len(combinations) <= max_cases, f"sweep expands to {len(combinations)} cases")
    cases = []
    for values in combinations:
        shape = normalize_shape(family, dict(zip(names, values)))
        identity = json.dumps([family, shape, normalize_dtype(dtype)], sort_keys=True, separators=(",", ":"))
        cases.append(
            {
                "case_id": f"{suite_id}.{family}.{sha1(identity.encode()).hexdigest()[:12]}",
                "operator_family": family,
                "operator_name": family,
                "shape": shape,
                "precision": {"input_dtype": normalize_dtype(dtype)},
                "source": {"kind": "synthetic_sweep"},
                "tags": ["synthetic_sweep"],
            }
        )
    return validate_suite(
        {
            "schema_version": 1,
            "kind": "operator_benchmark_shape_suite_v1",
            "suite_id": suite_id,
            "source_kind": "synthetic_sweep",
            "cases": cases,
        }
    )


def parse_backend_assignments(values: list[str]) -> dict[str, list[str]]:
    result = {}
    for value in values:
        family, separator, names = value.partition("=")
        backends = [name.strip() for name in names.split(",") if name.strip()]
        if not separator or family not in FAMILIES or not backends:
            raise ValueError(f"backend must use FAMILY=NAME1,NAME2: {value}")
        require(family not in result, f"duplicate backend family: {family}")
        result[family] = backends
    return result


def _command(argv: list[str]) -> subprocess.CompletedProcess[str] | None:
    try:
        return subprocess.run(argv, check=False, capture_output=True, text=True, timeout=10)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None


def environment(device: str) -> dict[str, Any]:
    import torch

    value: dict[str, Any] = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "device": device,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    if torch.cuda.is_available():
        properties = torch.cuda.get_device_properties(torch.device(device))
        value["gpu"] = {
            "name": properties.name,
            "major": properties.major,
            "minor": properties.minor,
            "total_memory_bytes": properties.total_memory,
        }
    smi = _command(["nvidia-smi", "--query-gpu=index,uuid,name,memory.used,utilization.gpu", "--format=csv,noheader,nounits"])
    value["nvidia_smi"] = smi.stdout.splitlines() if smi is not None and smi.returncode == 0 else []
    commit = _command(["git", "rev-parse", "HEAD"])
    value["git_commit"] = commit.stdout.strip() if commit is not None and commit.returncode == 0 else None
    return value


def _environment_identity(value: dict[str, Any]) -> dict[str, Any]:
    return {
        "python": value.get("python"),
        "torch": value.get("torch"),
        "cuda_runtime": value.get("cuda_runtime"),
        "device": value.get("device"),
        "cuda_visible_devices": value.get("cuda_visible_devices"),
        "gpu": value.get("gpu"),
        "git_commit": value.get("git_commit"),
    }


def _measure(fn: Any, warmup: int, iterations: int, prewarm_seconds: float, device: str) -> tuple[list[float], Any]:
    import torch

    result = None
    for _ in range(warmup):
        result = fn()
    deadline = time.monotonic() + prewarm_seconds
    while time.monotonic() < deadline:
        result = fn()
    torch.cuda.synchronize(torch.device(device))
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
    for start, end in zip(starts, ends):
        start.record()
        result = fn()
        end.record()
    ends[-1].synchronize()
    latencies = [float(start.elapsed_time(end)) for start, end in zip(starts, ends)]
    return latencies, result


def _input_identity(case: dict[str, Any]) -> dict[str, Any]:
    """实际输入身份；adapter精度另存，标签和调用权重不改变输入。"""
    requested_dtype = case.get("requested_input_dtype")
    if requested_dtype is None:
        requested_dtype = (case.get("precision") or {}).get("input_dtype")
    return {
        **{field: case.get(field) for field in ("operator_family", "shape", "routing", "replay", "sparse")},
        "precision": {"input_dtype": normalize_dtype(requested_dtype, "")},
    }


def _runtime_case(case: dict[str, Any], adapter: BackendAdapter, options: dict[str, Any]) -> dict[str, Any]:
    runtime_case = {
        **_input_identity(case),
        "case_id": f"{case['case_id']}.{adapter.descriptor.name}",
        "backend": adapter.descriptor.name,
        "backend_plugin": adapter.descriptor.plugin,
        "requested_input_dtype": case["precision"]["input_dtype"],
        "precision": adapter.precision(case),
        "source": {**(case.get("source") or {}), "canonical_case_id": case["case_id"]},
        "run": {
            "warmup": options["warmup"],
            "iterations": options["iterations"],
            "prewarm_seconds": options["prewarm_seconds"],
            "device": options["device"],
            "moe_routing": options["moe_routing"],
            "seed": options["seed"],
            "timing_mode": options["timing_mode"],
            "rate_schema": options["rate_schema"],
            "catalog_fingerprint": options.get("catalog_fingerprint"),
        },
    }
    return runtime_case


def _error_record(
    run_id: str,
    case: dict[str, Any],
    env: dict[str, Any],
    error_type: str,
    message: str,
    stage: str,
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "case": case,
        "environment": env,
        "status": "error",
        "error": {"type": error_type, "message": message, "stage": stage, "traceback": traceback.format_exc(limit=6)},
    }


def _fatal_cuda_error(error: dict[str, Any]) -> bool:
    """Recognize context-corrupting CUDA diagnostics, including historical raw rows."""
    message = re.sub(r"[^a-z]", "", str(error.get("message", "")).lower())
    return "cuda" in message and any(
        token in message
        for token in (
            "illegalmemoryaccess",
            "illegaladdress",
            "devicesideassert",
            "cudaerrorassert",
            "illegalinstruction",
            "misalignedaddress",
            "invalidaddressspace",
            "invalidpc",
            "hardwarestackerror",
            "unspecifiedlaunchfailure",
            "cudaerrorlaunchfailure",
            "launchfailed",
            "launchtimedout",
            "cudaerrorlaunchtimeout",
        )
    )


def _check_output(output, expected_shape=None):
    """Check execution structure, without a numerical reference or product-quality claim."""
    import torch

    require(isinstance(output, torch.Tensor), "non_tensor_output")
    require(bool(torch.isfinite(output).all()), "nonfinite_output")
    require(expected_shape is None or tuple(output.shape) == tuple(expected_shape), "output_shape_mismatch")
    return tuple(output.shape)


def _sparse_work(prepared):
    """Read production compute facts outside timing; retain no QKV or mask."""
    import tempfile

    from lightx2v.utils.workload_stats import WorkloadStats

    with tempfile.TemporaryDirectory(prefix="lightx2v-bench-stats-") as directory:
        with WorkloadStats(Path(directory) / "stats", families=["attention"], max_calls=2):
            output = prepared.fn()
        routes = [json.loads(line) for line in (Path(directory) / "stats/workloads.jsonl").read_text().splitlines()]
    metrics = prepared.extra_metrics
    metrics.update(actual_compute_count_available=False, block_density_actual=None)
    if len(routes) == 1:
        work, routing = routes[0]["work"], routes[0]["routing"]
        if work.get("selected_logical_flops") is not None:
            prepared.work = work["selected_logical_flops"]
            prepared.work_definition = "actual_production_selected_regions_QK_PV; excludes router/softmax/quantization"
            metrics.update(actual_compute_count_available=True, scheduled_tile_flops=work.get("scheduled_tile_flops"))
        if routing.get("total_tiles"):
            metrics["block_density_actual"] = routing["selected_tiles"] / routing["total_tiles"]
    return output


def run_case(case: dict[str, Any], adapter: BackendAdapter, run_id: str, env: dict[str, Any], options: dict[str, Any]) -> dict[str, Any]:
    import torch

    runtime_case = _runtime_case(case, adapter, options)
    support_error = adapter.support_error(case)
    if support_error:
        return _error_record(run_id, runtime_case, env, "BackendCapabilityError", support_error, "support")
    if not torch.cuda.is_available():
        return _error_record(run_id, runtime_case, env, "BackendCapabilityError", "CUDA is unavailable", "support")
    stage = "prepare"
    try:
        torch.manual_seed(options["measurement_seed"])
        torch.cuda.manual_seed_all(options["measurement_seed"])
        target = torch.device(options["device"])
        with torch.cuda.device(target):
            setup_started = time.perf_counter()
            prepared = adapter.prepare(case, options["device"], options)
            torch.cuda.synchronize(target)
            setup_ms = (time.perf_counter() - setup_started) * 1000
            stage = "preflight"
            require((prepared.correctness or {}).get("passed") is not False, "backend declared an invalid result")
            output = _sparse_work(prepared) if case["operator_family"] == "sparse_attention" else prepared.fn()
            expected_shape = _check_output(output, prepared.expected_output_shape)
            del output
            stage = "measure"
            latencies, result = _measure(
                prepared.fn,
                options["warmup"],
                options["iterations"],
                options["prewarm_seconds"],
                options["device"],
            )
            _check_output(result, expected_shape)
        mean_ms = statistics.mean(latencies)
        count_available = case["operator_family"] != "sparse_attention" or prepared.extra_metrics.get("actual_compute_count_available", False)
        measured_work = prepared.work if count_available else None
        rates = {prepared.rate_metric: measured_work / (mean_ms / 1000) / 1e12 if measured_work is not None else None}
        dense_equivalent = {}
        if prepared.dense_equivalent_work is not None:
            rates["dense_equivalent_tflops"] = prepared.dense_equivalent_work / (mean_ms / 1000) / 1e12
            dense_equivalent = {
                "dense_equivalent_work": prepared.dense_equivalent_work,
                "dense_equivalent_work_definition": prepared.dense_equivalent_work_definition,
            }
        return {
            "run_id": run_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "case": runtime_case,
            "environment": env,
            "status": "ok",
            "metrics": {
                "latency_ms_mean": mean_ms,
                "latency_ms_median": statistics.median(latencies),
                "setup_ms": setup_ms,
                **rates,
                "work": measured_work,
                "work_definition": prepared.work_definition if count_available else "unknown_actual_selected_regions; no logical FLOPs inferred from block density",
                **dense_equivalent,
                "measurement_scope": "kernel_cuda_event_list_single_sync",
                "output_shape": list(result.shape) if hasattr(result, "shape") else None,
                **prepared.extra_metrics,
            },
            "correctness": {"checked": False, "passed": None, "status": "not_checked"},
        }
    except Exception as exc:
        return _error_record(run_id, runtime_case, env, type(exc).__name__, str(exc), stage)


def _record_key(record: dict[str, Any]) -> tuple[str, str, str] | None:
    case = record.get("case") if isinstance(record.get("case"), dict) else {}
    source = case.get("source") if isinstance(case.get("source"), dict) else {}
    if not record.get("run_id") or not source.get("canonical_case_id") or not case.get("backend"):
        return None
    return str(record["run_id"]), str(source["canonical_case_id"]), str(case["backend"])


def load_records(paths: Iterable[Path]) -> list[dict[str, Any]]:
    records = []
    for path in paths:
        require(path.is_file(), f"raw result does not exist: {path}")
        for line_no, line in enumerate(path.read_text(encoding="utf-8").split("\n"), start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            require(isinstance(record, dict), f"raw record must be an object: {path}:{line_no}")
            records.append({**record, "_source_path": str(path), "_source_line": line_no})
    return records


def run_suite(
    shape_suite: dict[str, Any],
    output: Path,
    assignments: dict[str, list[str]],
    plugins: list[str],
    *,
    repeat_runs: int = 3,
    warmup: int = 10,
    iterations: int = 30,
    prewarm_seconds: float = 0.0,
    device: str = "cuda:0",
    moe_routing: str = "balanced",
    seed: int = 42,
    append: bool = False,
    candidate_catalog: dict[str, Any] | None = None,
) -> dict[str, Any]:
    validate_suite(shape_suite)
    require(repeat_runs > 0 and iterations > 0, "repeat_runs and iterations must be positive")
    require(warmup >= 0 and prewarm_seconds >= 0, "warmup and prewarm_seconds must be non-negative")
    registry = load_registry(plugins)
    if candidate_catalog is None and any(case["operator_family"] not in assignments for case in shape_suite["cases"]):
        candidate_catalog = probe_backend_catalog(registry, device)
    selections = {}
    for case in shape_suite["cases"]:
        family = case["operator_family"]
        names = assignments.get(family)
        if names is None:
            names = [item["name"] for item in candidate_catalog["candidates"] if item["family"] == family and item["status"] == "eligible"]
        adapters = [registry.require(name) for name in names]
        require(all(adapter.descriptor.family == family for adapter in adapters), f"backend family mismatch: {case['case_id']}")
        if family == "moe" and moe_routing == "observed" and not case.get("replay"):
            _validate_routing(case, required=True)
        selections[case["case_id"]] = adapters
    if output.exists() and output.stat().st_size and not append:
        raise ValueError(f"raw output already exists; use --append: {output}")
    existing_records = load_records([output]) if append and output.exists() else []
    existing = set()
    expected_run_ids = {f"repeat-{index:03d}" for index in range(repeat_runs)}
    run_contract = {
        "warmup": warmup,
        "iterations": iterations,
        "prewarm_seconds": prewarm_seconds,
        "device": device,
        "moe_routing": moe_routing,
        "seed": seed,
        "timing_mode": TIMING_MODE,
        "rate_schema": RATE_SCHEMA,
        "catalog_fingerprint": (candidate_catalog or {}).get("catalog_fingerprint"),
    }
    env = environment(device)
    expected_environment = _environment_identity(env)
    cases = {case["case_id"]: case for case in shape_suite["cases"]}
    for record in existing_records:
        key = _record_key(record)
        require(key is not None and key not in existing, "existing raw has missing or duplicate identity")
        run_id, case_id, backend = key
        require(run_id in expected_run_ids and case_id in cases, "existing raw suite/repeat drift")
        raw_case = record["case"]
        actual_input = _input_identity(raw_case)
        for field, expected in _input_identity(cases[case_id]).items():
            require(actual_input[field] == expected, f"existing raw {field} drift: {case_id}")
        require(raw_case.get("run") == run_contract, "existing raw measurement arguments differ from this run")
        require(
            _environment_identity(record.get("environment") or {}) == expected_environment,
            "existing raw hardware or software environment differs from this run",
        )
        require(backend in {item.descriptor.name for item in selections[case_id]}, f"existing raw backend drift: {backend}")
        require(raw_case.get("precision") == registry.require(backend).precision(cases[case_id]), f"existing raw backend precision drift: {case_id}")
        existing.add(key)
    output.parent.mkdir(parents=True, exist_ok=True)
    if candidate_catalog is not None:
        write_json(output.parent / "backend_catalog.json", candidate_catalog)
    written = 0
    statuses: Counter[str] = Counter()
    options = {**run_contract, "measurement_seed": seed}
    needs_separator = False
    if append and output.exists() and output.stat().st_size:
        with output.open("rb") as previous:
            previous.seek(-1, os.SEEK_END)
            needs_separator = previous.read(1) not in (b"\n", b"\r")
    with output.open("a", encoding="utf-8") as handle:
        if needs_separator:
            handle.write("\n")
        for repeat in range(repeat_runs):
            run_id = f"repeat-{repeat:03d}"
            options["measurement_seed"] = seed + repeat
            for case in shape_suite["cases"]:
                adapters = selections[case["case_id"]]
                if not adapters:
                    continue  # Keep dependency/architecture exclusions in the catalog when no candidate can run.
                adapters = adapters[repeat % len(adapters) :] + adapters[: repeat % len(adapters)]
                for adapter in adapters:
                    key = (run_id, case["case_id"], adapter.descriptor.name)
                    if key in existing:
                        continue
                    record = run_case(case, adapter, run_id, env, options)
                    handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
                    handle.flush()
                    written += 1
                    statuses[record["status"]] += 1
                    if _fatal_cuda_error(record.get("error") or {}):
                        raise ValueError(f"fatal CUDA context error; raw saved to {output}; stop and restart in a fresh process")
    return {"written": written, "skipped_existing": len(existing), "status_counts": dict(statuses), "raw": str(output)}


def _matches(record: dict[str, Any], case: dict[str, Any]) -> bool:
    raw = record.get("case") if isinstance(record.get("case"), dict) else {}
    source = raw.get("source") if isinstance(raw.get("source"), dict) else {}
    return source.get("canonical_case_id") == case["case_id"] and _input_identity(raw) == _input_identity(case)


def _peak_family(precision: dict[str, Any]) -> str | None:
    for value in (precision.get("peak_family"), precision.get("quant"), precision.get("weight_dtype"), precision.get("input_dtype")):
        lowered = normalize_dtype(value, "")
        if "e4m3" in lowered or "e5m2" in lowered:
            return "fp8"
        for name in ("nvfp4", "mxfp4", "mxfp8", "fp8", "fp4", "int8", "bf16", "fp16", "tf32"):
            if name in lowered:
                return name
    return None


def _validate_peak_rate(family: str, label: str, entry: Any) -> None:
    require(isinstance(entry, dict), f"peak entry must be an object: {label}")
    rate = entry.get("dense_rate")
    require(isinstance(rate, (int, float)) and not isinstance(rate, bool) and rate > 0, f"invalid dense peak rate: {label}")
    expected_unit = "TOPS" if family.startswith("int") else "TFLOPS"
    require(entry.get("unit") == expected_unit, f"peak unit for {label} must be {expected_unit}")


def _select_peak(
    platform_value: dict[str, Any] | None,
    family: str | None,
    precision: dict[str, Any],
) -> tuple[dict[str, Any] | None, str | None, str]:
    if platform_value is None:
        return None, None, "not_requested"
    if family is None:
        return None, None, "precision_unknown"
    family_entry = (platform_value.get("peaks") or {}).get(family)
    if family_entry is None:
        return None, None, "peak_missing"
    variants = family_entry.get("variants")
    if variants is None:
        return family_entry, None, "available"
    accumulator = normalize_dtype(precision.get("accum_dtype"), "")
    if accumulator in {"", "backend_default", "production_wrapper_default"}:
        return None, None, "precision_variant_unknown"
    variant_entry = variants.get(accumulator)
    if variant_entry is None:
        return None, accumulator, "peak_variant_missing"
    return variant_entry, accumulator, "available"


def _hardware_platform(peaks: dict[str, Any] | None, platform_id: str | None, records: list[dict[str, Any]]) -> dict[str, Any] | None:
    if peaks is None:
        require(platform_id is None, "--platform requires --peaks")
        return None
    require(bool(platform_id), "--platform is required with --peaks")
    platform_value = (peaks.get("platforms") or {}).get(platform_id)
    require(isinstance(platform_value, dict), f"unknown hardware platform: {platform_id}")
    for family, entry in (platform_value.get("peaks") or {}).items():
        require(isinstance(entry, dict), f"peak entry must be an object: {family}")
        variants = entry.get("variants")
        if variants is None:
            _validate_peak_rate(family, family, entry)
            continue
        require(isinstance(variants, dict) and variants, f"peak variants must be a non-empty object: {family}")
        require("dense_rate" not in entry and "unit" not in entry, f"peak family cannot mix direct rate and variants: {family}")
        for accumulator, variant_entry in variants.items():
            require(normalize_dtype(accumulator, "") == accumulator, f"non-canonical accumulator dtype: {family}.{accumulator}")
            _validate_peak_rate(family, f"{family}.{accumulator}", variant_entry)
    identification = platform_value.get("identification") or {}
    gpus = [(record.get("environment") or {}).get("gpu") for record in records]
    gpus = [gpu for gpu in gpus if isinstance(gpu, dict)]
    if identification.get("gpu_name_regex"):
        require(gpus and all(re.search(identification["gpu_name_regex"], str(gpu.get("name"))) for gpu in gpus), "raw GPU name does not match peak profile")
    if identification.get("cuda_capability"):
        capabilities = {f"{gpu.get('major')}.{gpu.get('minor')}" for gpu in gpus}
        require(capabilities == {str(identification["cuda_capability"])}, "raw CUDA capability does not match peak profile")
    return platform_value


def build_recommendation_report(
    shape_suite: dict[str, Any],
    records: list[dict[str, Any]],
    *,
    required_runs: int = 3,
    max_spread_pct: float = 5.0,
    max_spread_ms: float = 0.005,
    peaks: dict[str, Any] | None = None,
    platform_id: str | None = None,
    candidate_catalog: dict[str, Any] | None = None,
) -> dict[str, Any]:
    validate_suite(shape_suite)
    known_candidates = {item["name"]: item for item in (candidate_catalog or {}).get("candidates", [])}
    if candidate_catalog is not None:
        require(candidate_catalog.get("kind") == "operator_benchmark_backend_catalog_v1", "unsupported backend catalog kind")
        fingerprint = candidate_catalog.get("catalog_fingerprint")
        require(isinstance(fingerprint, str) and fingerprint, "backend catalog fingerprint is required")
        for record in records:
            run = (record.get("case") or {}).get("run") or {}
            require(run.get("catalog_fingerprint") == fingerprint, "raw backend catalog differs from report catalog")
    matched: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    unmatched = 0
    for record in records:
        cases = [case for case in shape_suite["cases"] if _matches(record, case)]
        if len(cases) == 1:
            matched[cases[0]["case_id"]].append(record)
        else:
            unmatched += 1
    matched_records = [record for items in matched.values() for record in items]
    if matched_records:
        environments = {json.dumps(_environment_identity(record.get("environment") or {}), sort_keys=True) for record in matched_records}
        contracts = {json.dumps((record.get("case") or {}).get("run") or {}, sort_keys=True) for record in matched_records}
        require(len(environments) == 1, "mixed raw hardware or software environments")
        require(len(contracts) == 1, "mixed raw measurement contracts")
    platform_value = _hardware_platform(peaks, platform_id, matched_records)
    workloads = []
    for case in shape_suite["cases"]:
        family = case["operator_family"]
        attention_scope = (candidate_catalog or {}).get("attention_scopes", {}).get(family)
        conv_scope = (candidate_catalog or {}).get("convolution_scopes", {}).get(f"conv{len(case['shape']['input']) - 2}d") if family == "convolution" else None
        catalog_candidates = {item["name"]: item for item in (candidate_catalog or {}).get("candidates", []) if item.get("family") == family}
        if family == "sparse_attention":
            require(
                not any(item.get("production_variant") == "dense_replay_baseline" for item in catalog_candidates.values()),
                "sparse catalog contains retired dense controls; regenerate the backend catalog",
            )
        expected_backends = {name for name, item in catalog_candidates.items() if item.get("status") == "eligible"}
        dependency_blockers = sorted(name for name, item in catalog_candidates.items() if item.get("status") in {"cuda_unavailable", "dependency_missing"})
        if family == "gemm":
            unmapped = sorted((candidate_catalog or {}).get("unmapped_mm_backends", []))
        elif family in {"dense_attention", "sparse_attention"}:
            unmapped = sorted(attention_scope["unmapped_production_backends"] if attention_scope is not None else (candidate_catalog or {}).get("unmapped_attention_backends", []))
        elif family == "moe":
            unmapped = sorted((candidate_catalog or {}).get("unmapped_moe_backends", []))
        elif family == "convolution":
            unmapped = (conv_scope or {}).get("unmapped_production_backends", ["convolution_inventory_missing"])
        else:
            unmapped = []
        exclusion_key = {"gemm": "excluded_mm_backends", "moe": "excluded_moe_backends", "dense_attention": "excluded_attention_backends", "sparse_attention": "excluded_attention_backends"}.get(
            family
        )
        case_exclusions = attention_scope["excluded_production_backends"] if attention_scope is not None else (candidate_catalog or {}).get(exclusion_key, {})
        if family == "sparse_attention":
            case_exclusions = {**case_exclusions, **(candidate_catalog or {}).get("mask_input_requirements", {})}
        if conv_scope is not None:
            case_exclusions = conv_scope["excluded_production_backends"]
        grouped: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
        for record in matched[case["case_id"]]:
            descriptor = known_candidates.get(record["case"]["backend"])
            require(descriptor is None or descriptor["family"] == family, f"raw backend family mismatch: {case['case_id']}")
            grouped[str(record["case"]["backend"])].append(record)
        assessments = {}
        for backend, items in sorted(grouped.items()):
            run_ids = [str(item.get("run_id")) for item in items]
            require(len(run_ids) == len(set(run_ids)), f"duplicate run id: {case['case_id']}.{backend}")
            contracts = {json.dumps(item["case"].get("run") or {}, sort_keys=True) for item in items}
            precisions = {json.dumps(item["case"].get("precision") or {}, sort_keys=True) for item in items}
            require(len(contracts) == 1 and len(precisions) == 1, f"mixed raw contract: {case['case_id']}.{backend}")
            precision = json.loads(next(iter(precisions)))
            ok = [item for item in items if item.get("status") == "ok"]
            errors = [item for item in items if item.get("status") != "ok"]
            autotune_configs = {json.dumps(item["metrics"].get("autotune"), sort_keys=True) for item in ok}
            require(len(autotune_configs) <= 1, f"mixed autotune contract: {case['case_id']}.{backend}")
            latencies = [float(item["metrics"]["latency_ms_mean"]) for item in ok]
            median = statistics.median(latencies) if latencies else None
            spread_ms = max(latencies) - min(latencies) if latencies else None
            spread_pct = spread_ms / median * 100 if median and spread_ms is not None else None
            correctness_failed = any((item.get("correctness") or {}).get("passed") is False for item in items)
            if correctness_failed:
                status = "correctness_failed"
            elif any(_fatal_cuda_error(item.get("error") or {}) for item in errors):
                status = "measurement_error"
            elif errors and not ok:
                stages = {(item.get("error") or {}).get("stage") for item in errors}
                status = "unavailable" if stages <= {"support", "prepare"} else "measurement_error"
            elif errors:
                status = "measurement_error"
            elif len(ok) < required_runs:
                status = "insufficient_runs"
            elif spread_pct is not None and spread_pct > max_spread_pct and spread_ms > max_spread_ms:
                status = "unstable"
            else:
                status = "accepted"
            rate_values = []
            rate_metric = None
            for item in ok:
                for name in RATE_UNITS:
                    if name != "dense_equivalent_tflops" and item["metrics"].get(name) is not None:
                        rate_metric = name
                        rate_values.append(float(item["metrics"][name]))
                        break
            rate = statistics.median(rate_values) if rate_values else None
            rate_unit = RATE_UNITS.get(rate_metric)
            dense_equivalent_values = [float(item["metrics"]["dense_equivalent_tflops"]) for item in ok if item["metrics"].get("dense_equivalent_tflops") is not None]
            dense_equivalent_rate = statistics.median(dense_equivalent_values) if dense_equivalent_values else None
            # 只汇总raw中实际记录的计算量/块密度，不反推，也不掩盖缺失的repeat。
            compute_ranges = {}
            for field in ("work", "dense_equivalent_work", "block_density_actual"):
                counts = [item["metrics"].get(field) for item in ok]
                compute_ranges[field] = [min(counts), max(counts)] if counts and all(count is not None for count in counts) else None
            sparse = case["operator_family"] == "sparse_attention"
            if sparse and ok:
                require(
                    rate_metric == "actual_tflops"
                    or rate_metric is None
                    and all("work" in item["metrics"] and item["metrics"]["work"] is None and item["metrics"].get("actual_compute_count_available") is False for item in ok),
                    f"sparse raw lacks actual rate: {case['case_id']}.{backend}",
                )
                require(
                    len(dense_equivalent_values) == len(ok),
                    f"sparse raw lacks dense-equivalent rate: {case['case_id']}.{backend}",
                )
            family = _peak_family(precision)
            peak_entry, peak_variant, peak_status = _select_peak(platform_value, family, precision)
            peak = peak_entry.get("dense_rate") if peak_entry else None
            if peak_status == "available" and rate_unit is None:
                peak_status = "rate_unavailable"
            elif peak_status == "available" and peak_entry.get("unit") != rate_unit:
                peak_status = "unit_mismatch"
            efficiency = rate / float(peak) if rate is not None and peak and peak_status == "available" else None
            dense_equivalent_efficiency = dense_equivalent_rate / float(peak) if dense_equivalent_rate is not None and peak and peak_status == "available" else None
            assessments[backend] = {
                "status": status,
                "accepted": status == "accepted",
                "correctness": [{"run_id": item.get("run_id"), **(item.get("correctness") or {"checked": False})} for item in items],
                "autotune": json.loads(next(iter(autotune_configs))) if autotune_configs else None,
                "latency_ms": median,
                "spread_pct": spread_pct,
                "spread_ms": spread_ms,
                "rate": rate,
                "rate_metric": rate_metric,
                "rate_unit": rate_unit,
                "work_range": compute_ranges["work"],
                "dense_equivalent_work_range": compute_ranges["dense_equivalent_work"],
                "block_density_actual_range": compute_ranges["block_density_actual"],
                "peak_family": family,
                "peak_variant": peak_variant,
                "peak_status": peak_status,
                "nominal_peak_rate": peak,
                "nominal_efficiency": efficiency,
                "actual_rate": rate if sparse else None,
                "actual_rate_metric": rate_metric if sparse else None,
                "actual_peak_efficiency": efficiency if sparse else None,
                "dense_equivalent_rate": dense_equivalent_rate,
                "dense_equivalent_rate_metric": "dense_equivalent_tflops" if dense_equivalent_rate is not None else None,
                "dense_equivalent_peak_efficiency": dense_equivalent_efficiency,
                "precision": precision,
                "errors": [
                    {
                        "type": item.get("error", {}).get("type"),
                        "message": item.get("error", {}).get("message"),
                        "stage": item.get("error", {}).get("stage"),
                    }
                    for item in errors
                ],
            }
        ranking = sorted(
            (name for name, item in assessments.items() if item["accepted"]),
            key=lambda name: assessments[name]["latency_ms"],
        )
        measured_winner = ranking[0] if ranking else None
        terminal_statuses = {"accepted", "unavailable", "correctness_failed"}
        non_terminal = {name: item["status"] for name, item in assessments.items() if item["status"] not in terminal_statuses}
        candidate_coverage_complete = (
            candidate_catalog is not None
            and bool(expected_backends)
            and not dependency_blockers
            and not unmapped
            and unmatched == 0
            and expected_backends == set(assessments)
            and all(assessments[name]["status"] in terminal_statuses for name in expected_backends)
        )
        observed_name = case.get("observed_backend")
        observed = None
        if observed_name:
            if observed_name not in assessments:
                observed = {"status": "not_measured", "backend": observed_name}
            elif not assessments[observed_name]["accepted"]:
                observed = {"status": "not_accepted", "backend": observed_name}
            elif measured_winner:
                observed = {
                    "status": "comparable",
                    "backend": observed_name,
                    "measured_winner": measured_winner,
                    "speedup": assessments[observed_name]["latency_ms"] / assessments[measured_winner]["latency_ms"],
                }
        workloads.append(
            {
                "case_id": case["case_id"],
                "operator_family": case["operator_family"],
                "shape": case["shape"],
                "source": case.get("source"),
                "replay": case.get("replay"),
                "sparse": case.get("sparse"),
                "call_count": case.get("call_count"),
                "measured_winner": measured_winner,
                "candidate_coverage_complete": candidate_coverage_complete,
                "recommendation_scope": "complete_candidate_space" if candidate_coverage_complete else "measured_subset",
                "coverage_blockers": {
                    "not_measured": sorted(expected_backends - set(assessments)),
                    "dependency_missing": dependency_blockers,
                    "unmapped_production_backends": unmapped,
                    "catalog_missing": candidate_catalog is None,
                    "unmatched_raw_records": unmatched,
                    "non_terminal_candidates": non_terminal,
                },
                "catalog_candidates": catalog_candidates,
                "excluded_production_backends": case_exclusions,
                "integration_gaps": {
                    name: reason
                    for name, reason in (
                        (conv_scope or {}).get("integration_gaps", {}) if case["operator_family"] == "convolution" else (candidate_catalog or {}).get("attention_integration_gaps", {})
                    ).items()
                    if name in unmapped
                },
                "production_mappings": (conv_scope or {}).get("mapped_backends", {}),
                "ranking": ranking,
                "observed_backend": observed,
                "backends": assessments,
            }
        )
    complete = bool(workloads) and all(item["call_count"] is not None and item["measured_winner"] for item in workloads)
    weighted = sum(item["call_count"] * item["backends"][item["measured_winner"]]["latency_ms"] for item in workloads) if complete else None
    return {
        "kind": "operator_benchmark_report_v1",
        "suite_id": shape_suite["suite_id"],
        "catalog_fingerprint": (candidate_catalog or {}).get("catalog_fingerprint"),
        "hardware": {"platform_id": platform_id, "gpus": [record["environment"]["gpu"] for record in matched_records[:1] if (record.get("environment") or {}).get("gpu")], "profile": platform_value},
        "policy": {"required_runs": required_runs, "max_spread_pct": max_spread_pct, "max_spread_ms": max_spread_ms},
        "summary": {
            "workload_count": len(workloads),
            "measured_winner_count": sum(item["measured_winner"] is not None for item in workloads),
            "coverage_complete": bool(workloads) and all(item["candidate_coverage_complete"] for item in workloads),
            "matched_record_count": sum(len(items) for items in matched.values()),
            "unmatched_record_count": unmatched,
            "weighted_measured_complete": complete,
            "weighted_per_shape_measured_winner_ms": weighted,
        },
        "workloads": workloads,
    }


def hardware_markdown(hardware):
    """从本次设备身份及显式峰值文件生成简短平台说明，不维护内置硬件表。"""
    lines = []
    if not hardware.get("gpus"):
        lines.append("当前分析范围没有可核实的GPU身份。")
    for gpu in hardware.get("gpus", []):
        name = gpu.get("name", "未知")
        capability = f"{gpu['major']}.{gpu['minor']}" if gpu.get("major") is not None and gpu.get("minor") is not None else "未知"
        label = f"设备 {gpu['id']}" if gpu.get("id") is not None else "硬件"
        lines.append(f"{label}：**{name}**，SM {capability}。")
    profile = hardware.get("profile") or {}
    peaks = []
    for dtype, entry in profile.get("peaks", {}).items():
        variants = entry.get("variants") or {"": entry}
        for accum, value in variants.items():
            label = dtype.upper() + (f" / {accum.upper()}累加" if accum else "")
            peaks.append(f"{label} {value['dense_rate']:g} {value['unit']}")
    lines += ["", "名义 dense 峰值：" + "；".join(peaks) + "。" if peaks else "未提供匹配的硬件峰值；只报告绝对时间和计算速率。"]
    source = profile.get("source", {}).get("specification")
    if source:
        lines.append(f"峰值来源：[厂商规格]({source})；已去除规格中的结构化稀疏倍率，名义峰值不等于实测持续性能。")
    return lines


def markdown_report(value: dict[str, Any]) -> str:
    """Render a concise facts table; the Agent builds the final report from report.json."""
    sp = value["kind"] == "sp_attention_benchmark_report_v1"

    def number(value):
        return "—" if value is None else f"{value:.4g}"

    def cell(value):
        return str(value).replace("|", "\\|").replace("\n", " ")

    lines = ["# Benchmark 简表", "", *hardware_markdown(value.get("hardware") or {}), "", "本页仅供快速查看；完整事实见 [report.json](report.json)，最终中文报告由Agent按正式REPORT_TEMPLATE.md生成。"]
    for group in value["workloads"]:
        lines += [
            "",
            "## " + cell(group["case_id"]),
            "",
            "输入：`" + json.dumps(group["shape"], ensure_ascii=False) + "`",
            "",
            "| 后端 / 配置 | 延迟/ms | 计算效率 | 轮间极差/ms | 状态 |",
            "|---|---:|---:|---:|---|",
        ]
        items = group["candidates"] if sp else group["backends"]
        for name, item in sorted(items.items(), key=lambda row: (row[1]["latency_ms"] is None, row[1]["latency_ms"] or 0, row[0])):
            if sp and item["latency_ms"] is None:
                continue
            if sp:
                rate = item.get("actual_tflops") if group.get("sparse") else item.get("aggregate_effective_tflops")
                equivalent, unit = item.get("dense_equivalent_tflops"), "TFLOPS"
            else:
                rate, equivalent, unit = item.get("rate"), item.get("dense_equivalent_rate"), item.get("rate_unit") or ""
            if rate is None:
                equivalent = None
            efficiency = number(rate) + (" / " + number(equivalent) + "*" if equivalent is not None else "")
            lines.append("| " + " | ".join([cell(name), number(item["latency_ms"]), efficiency + " " + unit, number(item.get("spread_ms")), cell(item["status"])]) + " |")
    lines += ["", "耗时为各轮均值的中位数；*为dense等价效率。状态、误差、覆盖与具体失败原因均保存在JSON；本表不作质量或采用决定。"]
    return "\n".join(lines) + "\n"


def _axis_value(value: str) -> Any:
    if value.lower() in {"true", "false"}:
        return value.lower() == "true"
    try:
        return int(value)
    except ValueError:
        return value


def _axes(values: list[str]) -> dict[str, list[Any]]:
    result = {}
    for value in values:
        name, separator, raw = value.partition("=")
        parsed = [_axis_value(item.strip()) for item in raw.split(",") if item.strip()]
        require(bool(separator and name and parsed), f"axis must use NAME=value1,value2: {value}")
        result[name] = parsed
    return result


def _report_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--peaks", type=Path)
    parser.add_argument("--platform")
    parser.add_argument("--required-runs", type=int, default=3)
    parser.add_argument("--max-spread-pct", type=float, default=5.0)
    parser.add_argument("--max-spread-ms", type=float, default=0.005)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    inspect_parser = commands.add_parser("inspect", help="Validate and summarize a shape suite.")
    inspect_parser.add_argument("--suite", type=Path, required=True)
    sweep_parser = commands.add_parser("sweep", help="Generate a theoretical shape grid.")
    sweep_parser.add_argument("--family", choices=sorted(FAMILIES - {"sparse_attention", "convolution"}), required=True)
    sweep_parser.add_argument("--axis", action="append", required=True)
    sweep_parser.add_argument("--dtype", default="bf16")
    sweep_parser.add_argument("--suite-id", required=True)
    sweep_parser.add_argument("--output", type=Path, required=True)
    backend_parser = commands.add_parser("backends", help="List registered backends.")
    backend_parser.add_argument("--plugin", action="append", default=[])
    backend_parser.add_argument("--probe", action="store_true")
    backend_parser.add_argument("--device", default="cuda:0")
    run_parser = commands.add_parser("run", help="Run a suite and write raw plus report files.")
    run_parser.add_argument("--suite", type=Path, required=True)
    run_parser.add_argument("--output-dir", type=Path, required=True)
    run_parser.add_argument("--backend", action="append", default=[])
    run_parser.add_argument("--plugin", action="append", default=[])
    run_parser.add_argument("--repeat-runs", type=int, default=3)
    run_parser.add_argument("--warmup", type=int, default=10)
    run_parser.add_argument("--iterations", type=int, default=30)
    run_parser.add_argument("--prewarm-seconds", type=float, default=0.0)
    run_parser.add_argument("--device", default="cuda:0")
    run_parser.add_argument("--seed", type=int, default=42)
    run_parser.add_argument("--moe-routing", choices=["balanced", "uniform", "skewed", "observed"], default="balanced")
    run_parser.add_argument("--append", action="store_true")
    _report_args(run_parser)
    report_parser = commands.add_parser("report", help="Rebuild a report from raw JSONL.")
    report_parser.add_argument("--suite", type=Path, required=True)
    report_parser.add_argument("--raw", nargs="+", type=Path, required=True)
    report_parser.add_argument("--output-dir", type=Path, required=True)
    report_parser.add_argument("--backend-catalog", type=Path)
    _report_args(report_parser)
    return parser


def _make_report(
    args: argparse.Namespace,
    shape_suite: dict[str, Any],
    raw_paths: list[Path],
    candidate_catalog: dict[str, Any] | None = None,
) -> dict[str, Any]:
    require(bool(args.peaks) == bool(args.platform), "--peaks and --platform must be provided together")
    peaks = json.loads(args.peaks.read_text(encoding="utf-8")) if args.peaks else None
    backend_catalog_path = getattr(args, "backend_catalog", None)
    if candidate_catalog is None and backend_catalog_path:
        candidate_catalog = json.loads(backend_catalog_path.read_text(encoding="utf-8"))
    value = build_recommendation_report(
        shape_suite,
        load_records(raw_paths),
        required_runs=args.required_runs,
        max_spread_pct=args.max_spread_pct,
        max_spread_ms=args.max_spread_ms,
        peaks=peaks,
        platform_id=args.platform,
        candidate_catalog=candidate_catalog,
    )
    write_json(args.output_dir / "report.json", value)
    (args.output_dir / "report.md").write_text(markdown_report(value), encoding="utf-8")
    return value


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "inspect":
            print(json.dumps(inspect_suite(load_suite(args.suite)), ensure_ascii=False, sort_keys=True))
            return 0
        if args.command == "sweep":
            value = generate_sweep(args.family, _axes(args.axis), args.suite_id, args.dtype)
            write_json(args.output, value)
            print(json.dumps({"case_count": len(value["cases"]), "output": str(args.output)}, sort_keys=True))
            return 0
        if args.command == "backends":
            registry = load_registry(args.plugin)
            value = probe_backend_catalog(registry, args.device) if args.probe else registry.descriptors()
            print(json.dumps(value, ensure_ascii=False, indent=2))
            return 0
        if args.command == "run":
            shape_suite = load_suite(args.suite)
            raw_path = args.output_dir / "raw.jsonl"
            registry = load_registry(args.plugin)
            candidate_catalog = probe_backend_catalog(registry, args.device)
            run_summary = run_suite(
                shape_suite,
                raw_path,
                parse_backend_assignments(args.backend),
                args.plugin,
                repeat_runs=args.repeat_runs,
                warmup=args.warmup,
                iterations=args.iterations,
                prewarm_seconds=args.prewarm_seconds,
                device=args.device,
                moe_routing=args.moe_routing,
                seed=args.seed,
                append=args.append,
                candidate_catalog=candidate_catalog,
            )
            write_json(args.output_dir / "run_summary.json", run_summary)
            value = _make_report(args, shape_suite, [raw_path], candidate_catalog)
            print(json.dumps({**run_summary, **value["summary"]}, ensure_ascii=False, sort_keys=True))
            return 0
        if args.command == "report":
            value = _make_report(args, load_suite(args.suite), args.raw)
            print(json.dumps(value["summary"], ensure_ascii=False, sort_keys=True))
            return 0
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    raise AssertionError(f"unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
