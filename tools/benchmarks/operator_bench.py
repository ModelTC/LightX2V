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

FAMILIES = {"gemm", "dense_attention", "sparse_attention", "moe"}
ATTENTION_SHAPE_FIELDS = {"batch", "seq_q", "seq_kv", "heads", "kv_heads", "head_dim", "causal"}
SHAPE_FIELDS = {
    "gemm": {"m", "n", "k", "bias"},
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
DEFAULT_BACKENDS = {
    "gemm": ["torch_linear"],
    "dense_attention": ["torch_sdpa"],
    "sparse_attention": ["dynamic_sparse_sage2_replay"],
    "moe": ["torch_expert_loop"],
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


def _validate_sparse_replay(case: dict[str, Any]) -> None:
    replay = case.get("replay")
    require(isinstance(replay, dict), f"sparse replay is required: {case['case_id']}")
    require(set(replay) == {"manifest", "sha256"}, f"invalid sparse replay fields: {case['case_id']}")
    require(isinstance(replay["manifest"], str) and replay["manifest"], f"sparse replay manifest is required: {case['case_id']}")
    require(bool(re.fullmatch(r"[0-9a-f]{64}", str(replay["sha256"]))), f"invalid sparse replay sha256: {case['case_id']}")
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


def _runtime_case(case: dict[str, Any], adapter: BackendAdapter, options: dict[str, Any]) -> dict[str, Any]:
    runtime_case = {
        "case_id": f"{case['case_id']}.{adapter.descriptor.name}",
        "operator_family": case["operator_family"],
        "backend": adapter.descriptor.name,
        "backend_plugin": adapter.descriptor.plugin,
        "shape": case["shape"],
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
    for field in ("replay", "sparse"):
        if field in case:
            runtime_case[field] = case[field]
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
            stage = "measure"
            latencies, result = _measure(
                prepared.fn,
                options["warmup"],
                options["iterations"],
                options["prewarm_seconds"],
                options["device"],
            )
        mean_ms = statistics.mean(latencies)
        rates = {prepared.rate_metric: prepared.work / (mean_ms / 1000) / 1e12}
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
                "work": prepared.work,
                "work_definition": prepared.work_definition,
                **dense_equivalent,
                "measurement_scope": "kernel_cuda_event_list_single_sync",
                "output_shape": list(result.shape) if hasattr(result, "shape") else None,
                **prepared.extra_metrics,
            },
            "correctness": prepared.correctness or {"checked": False, "passed": None},
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
        for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
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
    selections = {}
    for case in shape_suite["cases"]:
        family = case["operator_family"]
        adapters = [registry.require(name) for name in assignments.get(family, DEFAULT_BACKENDS[family])]
        require(all(adapter.descriptor.family == family for adapter in adapters), f"backend family mismatch: {case['case_id']}")
        if family == "moe" and moe_routing == "observed":
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
        require(raw_case.get("shape") == cases[case_id]["shape"], f"existing raw shape drift: {case_id}")
        require(raw_case.get("replay") == cases[case_id].get("replay"), f"existing raw replay drift: {case_id}")
        require(raw_case.get("sparse") == cases[case_id].get("sparse"), f"existing raw sparse config drift: {case_id}")
        require(raw_case.get("run") == run_contract, "existing raw measurement arguments differ from this run")
        require(
            _environment_identity(record.get("environment") or {}) == expected_environment,
            "existing raw hardware or software environment differs from this run",
        )
        require(backend in {item.descriptor.name for item in selections[case_id]}, f"existing raw backend drift: {backend}")
        existing.add(key)
    output.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    statuses: Counter[str] = Counter()
    options = {**run_contract, "measurement_seed": seed}
    with output.open("a", encoding="utf-8") as handle:
        for repeat in range(repeat_runs):
            run_id = f"repeat-{repeat:03d}"
            options["measurement_seed"] = seed + repeat
            for case in shape_suite["cases"]:
                adapters = selections[case["case_id"]]
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
    return {"written": written, "skipped_existing": len(existing), "status_counts": dict(statuses), "raw": str(output)}


def _matches(record: dict[str, Any], case: dict[str, Any]) -> bool:
    raw = record.get("case") if isinstance(record.get("case"), dict) else {}
    source = raw.get("source") if isinstance(raw.get("source"), dict) else {}
    precision = raw.get("precision") if isinstance(raw.get("precision"), dict) else {}
    return (
        source.get("canonical_case_id") == case["case_id"]
        and raw.get("operator_family") == case["operator_family"]
        and raw.get("shape") == case["shape"]
        and raw.get("replay") == case.get("replay")
        and raw.get("sparse") == case.get("sparse")
        and normalize_dtype(precision.get("input_dtype"), "") == case["precision"]["input_dtype"]
    )


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
    if candidate_catalog is not None:
        require(candidate_catalog.get("kind") == "operator_benchmark_backend_catalog_v1", "unsupported backend catalog kind")
        fingerprint = candidate_catalog.get("catalog_fingerprint")
        require(isinstance(fingerprint, str) and fingerprint, "backend catalog fingerprint is required")
        for record in records:
            run = (record.get("case") or {}).get("run") or {}
            require(run.get("catalog_fingerprint") == fingerprint, "raw backend catalog differs from report catalog")
    platform_value = _hardware_platform(peaks, platform_id, records)
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
        environments = {
            json.dumps(_environment_identity(record.get("environment") or {}), sort_keys=True)
            for record in matched_records
        }
        contracts = {
            json.dumps((record.get("case") or {}).get("run") or {}, sort_keys=True)
            for record in matched_records
        }
        require(len(environments) == 1, "mixed raw hardware or software environments")
        require(len(contracts) == 1, "mixed raw measurement contracts")
    workloads = []
    for case in shape_suite["cases"]:
        family = case["operator_family"]
        catalog_candidates = {
            item["name"]: item
            for item in (candidate_catalog or {}).get("candidates", [])
            if item.get("family") == family
        }
        expected_backends = {name for name, item in catalog_candidates.items() if item.get("status") == "eligible"}
        dependency_blockers = sorted(
            name
            for name, item in catalog_candidates.items()
            if item.get("status") in {"cuda_unavailable", "dependency_missing"}
        )
        if family == "gemm":
            unmapped = sorted((candidate_catalog or {}).get("unmapped_mm_backends", []))
        elif family in {"dense_attention", "sparse_attention"}:
            unmapped = sorted((candidate_catalog or {}).get("unmapped_attention_backends", []))
        else:
            unmapped = []
        grouped: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
        for record in matched[case["case_id"]]:
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
            latencies = [float(item["metrics"]["latency_ms_mean"]) for item in ok]
            median = statistics.median(latencies) if latencies else None
            spread_ms = max(latencies) - min(latencies) if latencies else None
            spread_pct = spread_ms / median * 100 if median and spread_ms is not None else None
            correctness_failed = any((item.get("correctness") or {}).get("passed") is False for item in ok)
            if errors and not ok:
                stages = {(item.get("error") or {}).get("stage") for item in errors}
                status = "unavailable" if stages <= {"support", "prepare"} else "measurement_error"
            elif errors:
                status = "measurement_error"
            elif correctness_failed:
                status = "correctness_failed"
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
                    if item["metrics"].get(name) is not None:
                        rate_metric = name
                        rate_values.append(float(item["metrics"][name]))
                        break
            rate = statistics.median(rate_values) if rate_values else None
            rate_unit = RATE_UNITS.get(rate_metric)
            dense_equivalent_values = [
                float(item["metrics"]["dense_equivalent_tflops"])
                for item in ok
                if item["metrics"].get("dense_equivalent_tflops") is not None
            ]
            dense_equivalent_rate = statistics.median(dense_equivalent_values) if dense_equivalent_values else None
            sparse = case["operator_family"] == "sparse_attention"
            if sparse and ok:
                require(rate_metric == "actual_tflops", f"sparse raw lacks actual rate: {case['case_id']}.{backend}")
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
            dense_equivalent_efficiency = (
                dense_equivalent_rate / float(peak)
                if dense_equivalent_rate is not None and peak and peak_status == "available"
                else None
            )
            assessments[backend] = {
                "status": status,
                "accepted": status == "accepted",
                "latency_ms": median,
                "spread_pct": spread_pct,
                "spread_ms": spread_ms,
                "rate": rate,
                "rate_metric": rate_metric,
                "rate_unit": rate_unit,
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
        non_terminal = {
            name: item["status"]
            for name, item in assessments.items()
            if item["status"] not in terminal_statuses
        }
        candidate_coverage_complete = (
            candidate_catalog is not None
            and bool(expected_backends)
            and not dependency_blockers
            and not unmapped
            and unmatched == 0
            and expected_backends == set(assessments)
            and all(assessments[name]["status"] in terminal_statuses for name in expected_backends)
        )
        winner = measured_winner if candidate_coverage_complete else None
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
                "call_count": case.get("call_count"),
                "winner": winner,
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
                "ranking": ranking,
                "observed_backend": observed,
                "backends": assessments,
            }
        )
    complete = bool(workloads) and all(item["call_count"] is not None and item["measured_winner"] for item in workloads)
    weighted = (
        sum(item["call_count"] * item["backends"][item["measured_winner"]]["latency_ms"] for item in workloads)
        if complete
        else None
    )
    return {
        "kind": "operator_benchmark_report_v1",
        "suite_id": shape_suite["suite_id"],
        "catalog_fingerprint": (candidate_catalog or {}).get("catalog_fingerprint"),
        "hardware": {"platform_id": platform_id},
        "policy": {"required_runs": required_runs, "max_spread_pct": max_spread_pct, "max_spread_ms": max_spread_ms},
        "summary": {
            "workload_count": len(workloads),
            "recommended_count": sum(item["winner"] is not None for item in workloads),
            "measured_winner_count": sum(item["measured_winner"] is not None for item in workloads),
            "coverage_complete": bool(workloads) and all(item["candidate_coverage_complete"] for item in workloads),
            "matched_record_count": sum(len(items) for items in matched.values()),
            "unmatched_record_count": unmatched,
            "weighted_measured_complete": complete,
            "weighted_per_shape_measured_winner_ms": weighted,
        },
        "workloads": workloads,
    }


def markdown_report(value: dict[str, Any]) -> str:
    lines = [
        "# 核心算子 Benchmark 报告",
        "",
        f"- Suite：`{value['suite_id']}`",
        f"- 已推荐：{value['summary']['recommended_count']} / {value['summary']['workload_count']}",
        f"- 候选覆盖完整：{'是' if value['summary']['coverage_complete'] else '否'}",
        f"- 指定硬件：`{value['hardware']['platform_id'] or '未提供'}`",
        "",
        "| Case | Family | Winner | Latency ms | Actual/primary rate | Actual/primary efficiency | Dense-equivalent rate | Dense-equivalent efficiency | Observed |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for workload in value["workloads"]:
        winner = workload["winner"]
        measured_winner = workload["measured_winner"]
        item = workload["backends"].get(measured_winner) if measured_winner else None
        latency = f"{item['latency_ms']:.6f}" if item else "-"
        rate = f"{item['rate']:.3f} {item['rate_unit']}" if item and item["rate"] is not None else "-"
        if item and item["nominal_efficiency"] is not None:
            efficiency = f"{item['nominal_efficiency'] * 100:.2f}%"
        else:
            efficiency = item["peak_status"] if item else "-"
        dense_rate = (
            f"{item['dense_equivalent_rate']:.3f} TFLOPS"
            if item and item["dense_equivalent_rate"] is not None
            else "-"
        )
        dense_efficiency = (
            f"{item['dense_equivalent_peak_efficiency'] * 100:.2f}%"
            if item and item["dense_equivalent_peak_efficiency"] is not None
            else "-"
        )
        observed = workload["observed_backend"]
        observed_text = "-" if observed is None else observed["status"]
        if observed and observed["status"] == "comparable":
            observed_text = f"{observed['backend']} -> {measured_winner}: {observed['speedup']:.3f}x"
        recommendation = winner or (f"{measured_winner} (measured)" if measured_winner else "-")
        lines.append(
            f"| `{workload['case_id']}` | `{workload['operator_family']}` | `{recommendation}` | {latency} | {rate} | {efficiency} | {dense_rate} | {dense_efficiency} | {observed_text} |"
        )
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
    sweep_parser.add_argument("--family", choices=sorted(FAMILIES - {"sparse_attention"}), required=True)
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
            write_json(args.output_dir / "backend_catalog.json", candidate_catalog)
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
