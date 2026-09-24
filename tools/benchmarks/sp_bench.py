"""Shape-driven distributed sequence-parallel attention benchmark."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import re
import statistics
import subprocess
import traceback
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tools.benchmarks.operator_backends import load_registry, probe_backend_catalog
from tools.benchmarks.operator_bench import environment, normalize_dtype, require, write_json

SUITE_KIND = "sp_attention_benchmark_shape_suite_v1"
INPUT_DTYPES = {"bf16", "fp16"}
CASE_FIELDS = {"case_id", "shape", "precision", "call_count", "observed_candidate", "source", "tags"}
SHAPE_FIELDS = {"sequence", "heads", "kv_heads", "head_dim", "sp_size", "aux_tokens", "aux_q", "aux_first", "causal"}
QUANT_SCHEMES = (None, "fp8", "fp4")
NCCL_ENV_NAMES = (
    "NCCL_ALGO",
    "NCCL_PROTO",
    "NCCL_MIN_NCHANNELS",
    "NCCL_MAX_NCHANNELS",
    "NCCL_P2P_DISABLE",
    "NCCL_IB_DISABLE",
    "NCCL_NVLS_ENABLE",
)


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


def _non_negative_int(value: Any, label: str) -> int:
    require(not isinstance(value, bool), f"{label} must be a non-negative integer")
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        raise ValueError(f"{label} must be a non-negative integer") from None
    require(parsed >= 0, f"{label} must be a non-negative integer")
    if isinstance(value, float):
        require(value.is_integer(), f"{label} must be a non-negative integer")
    return parsed


def _boolean(value: Any, label: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in {0, 1}:
        return bool(value)
    if isinstance(value, str) and value.lower() in {"true", "false"}:
        return value.lower() == "true"
    raise ValueError(f"{label} must be a boolean")


def normalize_shape(raw: dict[str, Any]) -> dict[str, Any]:
    require(isinstance(raw, dict), "SP attention shape must be an object")
    require(not (set(raw) - SHAPE_FIELDS), f"SP attention shape contains unknown fields: {sorted(set(raw) - SHAPE_FIELDS)}")
    shape = {
        "sequence": _positive_int(raw.get("sequence"), "shape.sequence"),
        "heads": _positive_int(raw.get("heads"), "shape.heads"),
        "kv_heads": _positive_int(raw.get("kv_heads", raw.get("heads")), "shape.kv_heads"),
        "head_dim": _positive_int(raw.get("head_dim"), "shape.head_dim"),
        "sp_size": _positive_int(raw.get("sp_size"), "shape.sp_size"),
        "aux_tokens": _non_negative_int(raw.get("aux_tokens", 0), "shape.aux_tokens"),
        "aux_q": _boolean(raw.get("aux_q", False), "shape.aux_q"),
        "aux_first": _boolean(raw.get("aux_first", False), "shape.aux_first"),
        "causal": _boolean(raw.get("causal", False), "shape.causal"),
    }
    require(shape["heads"] % shape["kv_heads"] == 0, "shape.heads must be divisible by shape.kv_heads")
    require(shape["sequence"] % shape["sp_size"] == 0, "shape.sequence must be divisible by shape.sp_size")
    require(not shape["aux_q"] or shape["aux_tokens"] > 0, "shape.aux_q requires positive shape.aux_tokens")
    require(not shape["causal"], "SP benchmark v1 only supports causal=false")
    return shape


def validate_suite(value: dict[str, Any]) -> dict[str, Any]:
    require(isinstance(value, dict), "SP shape suite must be an object")
    require(value.get("schema_version") == 1, "SP shape suite schema_version must be 1")
    require(value.get("kind") == SUITE_KIND, f"SP shape suite kind must be {SUITE_KIND}")
    require(isinstance(value.get("suite_id"), str) and value["suite_id"], "suite_id is required")
    cases = value.get("cases")
    require(isinstance(cases, list) and cases, "SP shape suite must contain cases")
    case_ids = []
    for case in cases:
        require(isinstance(case, dict), "SP suite case must be an object")
        require(not (set(case) - CASE_FIELDS), f"SP suite case contains unknown fields: {sorted(set(case) - CASE_FIELDS)}")
        require(isinstance(case.get("case_id"), str) and case["case_id"], "case_id is required")
        require(case.get("shape") == normalize_shape(case.get("shape") or {}), f"non-canonical SP shape: {case['case_id']}")
        precision = case.get("precision")
        require(isinstance(precision, dict) and set(precision) == {"input_dtype"}, f"input_dtype is required: {case['case_id']}")
        dtype = normalize_dtype(precision["input_dtype"])
        require(dtype == precision["input_dtype"] and dtype in INPUT_DTYPES, f"unsupported SP input dtype: {precision['input_dtype']}")
        if case.get("call_count") is not None:
            require(_positive_int(case["call_count"], "call_count") == case["call_count"], "invalid call_count")
        case_ids.append(case["case_id"])
    require(len(case_ids) == len(set(case_ids)), "SP case_id values must be unique")
    return value


def load_suite(path: Path) -> dict[str, Any]:
    return validate_suite(json.loads(path.read_text(encoding="utf-8")))


def inspect_suite(value: dict[str, Any]) -> dict[str, Any]:
    validate_suite(value)
    return {
        "suite_id": value["suite_id"],
        "case_count": len(value["cases"]),
        "sp_sizes": dict(sorted(Counter(case["shape"]["sp_size"] for case in value["cases"]).items())),
        "dtype_case_counts": dict(sorted(Counter(case["precision"]["input_dtype"] for case in value["cases"]).items())),
        "missing_call_count": [case["case_id"] for case in value["cases"] if case.get("call_count") is None],
        "missing_observed_candidate": [case["case_id"] for case in value["cases"] if case.get("observed_candidate") is None],
    }


def _candidate_id(candidate: dict[str, Any]) -> str:
    quant = candidate["quant_scheme"] or "none"
    fields = [candidate["algorithm"], f"attn={candidate['dense_backend']}", f"comm={quant}", f"fusion={int(candidate['tensor_fusion'])}"]
    if candidate["algorithm"] == "ulysses":
        fields.extend(
            (
                f"prepost={candidate['prepost_backend']}",
                f"a2a={candidate['a2a_backend']}",
                f"head={int(candidate['head_parallel'])}",
            )
        )
    return "__".join(fields)


def _support_error(
    case: dict[str, Any],
    candidate: dict[str, Any],
    *,
    ring_lse_backends: set[str],
    fp4_available: bool,
) -> str | None:
    shape = case["shape"]
    algorithm = candidate["algorithm"]
    quant = candidate["quant_scheme"]
    if quant == "fp4" and not fp4_available:
        return "FP4 communication dependency is unavailable"
    if quant == "fp4" and shape["head_dim"] % 16:
        return "FP4 communication requires head_dim divisible by 16"
    if algorithm == "ring":
        if candidate["dense_backend"] not in ring_lse_backends:
            return "Ring requires a dense backend with native apply_with_lse()"
        if shape["heads"] != shape["kv_heads"]:
            return "Ring requires equal Q/K/V head counts"
        if shape["aux_tokens"]:
            return "Ring auxiliary-token semantics are outside the v1 comparable subset"
        return None

    world_size = shape["sp_size"]
    if shape["heads"] % world_size or shape["kv_heads"] % world_size:
        return "Ulysses requires Q and KV head counts divisible by sp_size"
    is_gqa = shape["heads"] != shape["kv_heads"]
    if candidate["tensor_fusion"] and is_gqa:
        return "Ulysses tensor_fusion does not support GQA"
    if candidate["head_parallel"] and is_gqa:
        return "Ulysses head_parallel does not support GQA"
    if candidate["prepost_backend"] == "triton":
        if is_gqa:
            return "Triton pre/post does not support GQA"
        if quant == "fp4":
            return "Triton pre/post does not support FP4 communication"
        if not candidate["tensor_fusion"]:
            return "Triton pre/post requires tensor_fusion"
    if candidate["a2a_backend"] == "round_robin":
        if world_size % 2:
            return "round_robin requires an even sp_size"
        if candidate["head_parallel"]:
            return "round_robin does not support head_parallel"
    return None


def enumerate_candidates(
    case: dict[str, Any],
    dense_backends: list[str],
    *,
    a2a_backends: list[str] | None = None,
    ring_lse_backends: set[str] | None = None,
    fp4_available: bool = True,
) -> dict[str, Any]:
    a2a_backends = a2a_backends or ["torch", "round_robin"]
    ring_lse_backends = ring_lse_backends or set()
    raw_candidates = []
    for dense_backend, prepost, a2a, quant, fusion, head_parallel in itertools.product(
        dense_backends,
        ("torch", "triton"),
        a2a_backends,
        QUANT_SCHEMES,
        (False, True),
        (False, True),
    ):
        raw_candidates.append(
            {
                "algorithm": "ulysses",
                "dense_backend": dense_backend,
                "prepost_backend": prepost,
                "a2a_backend": a2a,
                "quant_scheme": quant,
                "tensor_fusion": fusion,
                "head_parallel": head_parallel,
            }
        )
    for dense_backend, quant, fusion in itertools.product(dense_backends, QUANT_SCHEMES, (False, True)):
        raw_candidates.append(
            {
                "algorithm": "ring",
                "dense_backend": dense_backend,
                "prepost_backend": "torch",
                "a2a_backend": "torch",
                "quant_scheme": quant,
                "tensor_fusion": fusion,
                "head_parallel": False,
            }
        )

    candidates = []
    exclusions = []
    for candidate in raw_candidates:
        candidate = {"candidate_id": _candidate_id(candidate), **candidate}
        reason = _support_error(
            case,
            candidate,
            ring_lse_backends=ring_lse_backends,
            fp4_available=fp4_available,
        )
        if reason:
            exclusions.append({"candidate_id": candidate["candidate_id"], "reason": reason})
        else:
            candidates.append(candidate)
    return {"case_id": case["case_id"], "candidates": candidates, "exclusions": exclusions}


def build_candidate_catalog(
    suite: dict[str, Any],
    runtime: dict[str, Any],
    dense_backends: list[str] | None = None,
) -> dict[str, Any]:
    validate_suite(suite)
    leaf_family = "dense_attention"
    backend_catalog = runtime["catalog"]
    dependency_blockers = sorted(
        item["name"]
        for item in backend_catalog.get("candidates", [])
        if item.get("family") == leaf_family and item.get("status") in {"cuda_unavailable", "dependency_missing"}
    )
    unmapped_backends = sorted(backend_catalog.get("unmapped_attention_backends", []))
    backend_scope_complete = not dependency_blockers and not unmapped_backends
    backend_scope_blockers = {
        "dependency_missing": dependency_blockers,
        "unmapped_production_backends": unmapped_backends,
    }
    all_dense_backends = list(runtime["dense_backends"])
    dense_backends = list(dense_backends or all_dense_backends)
    require(not (set(dense_backends) - set(all_dense_backends)), "requested dense backend is not eligible on this device")
    cases = [
        enumerate_candidates(
            case,
            dense_backends,
            a2a_backends=runtime["a2a_backends"],
            ring_lse_backends=runtime["ring_lse_backends"],
            fp4_available=runtime["fp4_available"],
        )
        for case in suite["cases"]
    ]
    identity = {
        "backend_catalog_fingerprint": runtime["catalog"]["catalog_fingerprint"],
        "leaf_family": leaf_family,
        "backend_scope_complete": backend_scope_complete,
        "backend_scope_blockers": backend_scope_blockers,
        "dense_backends": dense_backends,
        "all_eligible_dense_backends": all_dense_backends,
        "a2a_backends": runtime["a2a_backends"],
        "ring_lse_backends": sorted(runtime["ring_lse_backends"]),
        "fp4_available": runtime["fp4_available"],
        "cases": cases,
    }
    fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return {
        "kind": "sp_attention_benchmark_candidate_catalog_v1",
        "catalog_fingerprint": fingerprint,
        "scope_complete": backend_scope_complete and set(dense_backends) == set(all_dense_backends),
        **identity,
    }


def _discover_runtime(device: str) -> dict[str, Any]:
    catalog = probe_backend_catalog(load_registry([]), device)
    dense_backends = sorted(
        {
            item["production_backend"]
            for item in catalog["candidates"]
            if item["family"] == "dense_attention" and item["status"] == "eligible" and item["production_backend"]
        }
    )
    import lightx2v.common.ops.attn  # noqa: F401
    from lightx2v.common.ops.attn.utils import seq_p
    from lightx2v.utils.registry_factory import A2A_BACKEND_REGISTER, ATTN_WEIGHT_REGISTER

    ring_lse_backends = set()
    for name in dense_backends:
        backend_type = ATTN_WEIGHT_REGISTER.get(name)
        if backend_type is not None and callable(getattr(backend_type, "apply_with_lse", None)):
            ring_lse_backends.add(name)
    return {
        "catalog": catalog,
        "dense_backends": dense_backends,
        "ring_lse_backends": ring_lse_backends,
        "a2a_backends": list(dict.fromkeys(("torch", "round_robin", *sorted(A2A_BACKEND_REGISTER.keys())))),
        "fp4_available": seq_p.quant_fp4_sage3 is not None and seq_p.dequant_fp4_sage3 is not None,
    }


def _selected_candidates(
    suite: dict[str, Any],
    runtime: dict[str, Any],
    candidate_ids: set[str],
    algorithms: set[str],
) -> dict[str, list[dict[str, Any]]]:
    selections = {}
    known_ids = set()
    for case in suite["cases"]:
        result = enumerate_candidates(
            case,
            runtime["dense_backends"],
            a2a_backends=runtime["a2a_backends"],
            ring_lse_backends=runtime["ring_lse_backends"],
            fp4_available=runtime["fp4_available"],
        )
        known_ids.update(item["candidate_id"] for item in result["candidates"])
        selected = [
            item
            for item in result["candidates"]
            if item["algorithm"] in algorithms and (not candidate_ids or item["candidate_id"] in candidate_ids)
        ]
        require(selected, f"no eligible SP candidates selected for case {case['case_id']}")
        selections[case["case_id"]] = selected
    missing = sorted(candidate_ids - known_ids)
    require(not missing, f"unknown or ineligible candidate ids: {missing}")
    return selections


def _make_inputs(case: dict[str, Any], device: Any, seed: int, rank: int) -> tuple[Any, Any, Any, Any, Any, Any]:
    import torch

    shape = case["shape"]
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}[case["precision"]["input_dtype"]]
    local_sequence = shape["sequence"] // shape["sp_size"]
    generator = torch.Generator(device=device).manual_seed(seed + rank)
    q = torch.randn((local_sequence, shape["heads"], shape["head_dim"]), device=device, dtype=dtype, generator=generator)
    k = torch.randn((local_sequence, shape["kv_heads"], shape["head_dim"]), device=device, dtype=dtype, generator=generator)
    v = torch.randn((local_sequence, shape["kv_heads"], shape["head_dim"]), device=device, dtype=dtype, generator=generator)
    aux_q = aux_k = aux_v = None
    if shape["aux_tokens"]:
        aux_k = torch.randn((shape["aux_tokens"], shape["kv_heads"], shape["head_dim"]), device=device, dtype=dtype, generator=generator)
        aux_v = torch.randn((shape["aux_tokens"], shape["kv_heads"], shape["head_dim"]), device=device, dtype=dtype, generator=generator)
        if shape["aux_q"]:
            aux_q = torch.randn((shape["aux_tokens"], shape["heads"], shape["head_dim"]), device=device, dtype=dtype, generator=generator)
    return q, k, v, aux_q, aux_k, aux_v


def _prepare_operation(case: dict[str, Any], candidate: dict[str, Any], device: Any, seed: int, rank: int) -> tuple[Any, int]:
    import lightx2v.common.ops.attn  # noqa: F401
    from lightx2v.common.ops.attn.ring_attn import RingAttnWeight
    from lightx2v.common.ops.attn.ulysses_attn import UlyssesAttnWeight
    from lightx2v.utils.registry_factory import ATTN_WEIGHT_REGISTER

    shape = case["shape"]
    q, k, v, aux_q, aux_k, aux_v = _make_inputs(case, device, seed, rank)

    dense_type = ATTN_WEIGHT_REGISTER.get(candidate["dense_backend"])
    require(dense_type is not None, f"dense production backend is not registered: {candidate['dense_backend']}")
    dense_operator = dense_type()
    sp_operator = UlyssesAttnWeight() if candidate["algorithm"] == "ulysses" else RingAttnWeight()

    def fn() -> Any:
        return sp_operator.apply(
            q,
            k,
            v,
            aux_q=aux_q,
            aux_k=aux_k,
            aux_v=aux_v,
            attention_module=dense_operator,
            seq_p_group=None,
            prepost_backend=candidate["prepost_backend"],
            a2a_backend=candidate["a2a_backend"],
            quant_scheme=candidate["quant_scheme"],
            tensor_fusion=candidate["tensor_fusion"],
            head_parallel=candidate["head_parallel"],
            aux_first=shape["aux_first"],
            attention_kwargs={"causal": shape["causal"]},
        )

    query_tokens = shape["sequence"] + (shape["aux_tokens"] if shape["aux_q"] else 0)
    kv_tokens = shape["sequence"] + shape["aux_tokens"]
    work = 4 * shape["heads"] * query_tokens * kv_tokens * shape["head_dim"]
    return fn, work


def _measure_distributed(
    fn: Any,
    warmup: int,
    iterations: int,
    device: Any,
    *,
    check_finite: bool = True,
) -> tuple[list[float], list[list[float]], Any, int, bool]:
    import torch
    import torch.distributed as dist

    torch.cuda.reset_peak_memory_stats(device)
    for _ in range(warmup):
        result = fn()
    torch.cuda.synchronize(device)
    dist.barrier()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
    for start, end in zip(starts, ends):
        start.record()
        result = fn()
        end.record()
    ends[-1].synchronize()
    local = torch.tensor([start.elapsed_time(end) for start, end in zip(starts, ends)], device=device, dtype=torch.float64)
    gathered = [torch.empty_like(local) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, local)
    maximum = local.clone()
    dist.all_reduce(maximum, op=dist.ReduceOp.MAX)
    memory = torch.tensor(torch.cuda.max_memory_allocated(device), device=device, dtype=torch.int64)
    dist.all_reduce(memory, op=dist.ReduceOp.MAX)
    finite_value = True
    if check_finite:
        finite = torch.tensor(int(_output_tensor(result).isfinite().all().item()), device=device, dtype=torch.int32)
        dist.all_reduce(finite, op=dist.ReduceOp.MIN)
        finite_value = bool(finite.item())
    return maximum.cpu().tolist(), [item.cpu().tolist() for item in gathered], result, int(memory.item()), finite_value


def _output_tensor(result: Any) -> Any:
    if isinstance(result, tuple):
        return result[0]
    return result


def _tensor_bytes(value: Any) -> int:
    if hasattr(value, "numel") and hasattr(value, "element_size"):
        return int(value.numel() * value.element_size())
    if isinstance(value, (list, tuple)):
        return sum(_tensor_bytes(item) for item in value)
    return 0


def _diagnostic_measure(fn: Any, warmup: int, iterations: int, device: Any) -> dict[str, Any]:
    latencies, per_rank, _, peak_memory, _ = _measure_distributed(fn, warmup, iterations, device, check_finite=False)
    return {
        "latency_ms_mean": statistics.mean(latencies),
        "latency_ms_median": statistics.median(latencies),
        "per_rank_latency_ms_mean": [statistics.mean(values) for values in per_rank],
        "max_memory_allocated_bytes": peak_memory,
    }


def _communication_rates(logical_bytes: int, network_bytes: int, latency_ms: float, world_size: int) -> dict[str, Any]:
    return {
        "logical_payload_bytes_per_rank": logical_bytes,
        "network_bytes_per_rank": network_bytes,
        "aggregate_network_bytes": network_bytes * world_size,
        "algorithmic_bandwidth_gbps": logical_bytes / latency_ms / 1e6,
        "bus_bandwidth_gbps": network_bytes / latency_ms / 1e6,
        "bandwidth_unit": "GB/s",
        "bus_bandwidth_definition": "per_rank_bytes_crossing_gpu_links/latency",
    }


def _overlap_estimate(full_l1_ms: float, compute_only_ms: float, layer2_ms: float) -> dict[str, Any]:
    serial_reference = compute_only_ms + layer2_ms
    residual = full_l1_ms - serial_reference
    tolerance = max(0.005, serial_reference * 0.05)
    estimate_status = "estimated"
    if residual > tolerance:
        estimate_status = "unresolved_positive_residual"
    elif full_l1_ms < compute_only_ms - tolerance:
        estimate_status = "unresolved_cross_measurement_inversion"
    exposed = hidden = ratio = None
    if estimate_status == "estimated":
        exposed = min(layer2_ms, max(0.0, full_l1_ms - compute_only_ms))
        hidden = layer2_ms - exposed
        ratio = hidden / layer2_ms if layer2_ms else None
    return {
        "full_l1_latency_ms": full_l1_ms,
        "compute_only_latency_ms": compute_only_ms,
        "layer2_layout_communication_latency_ms": layer2_ms,
        "serial_reference_latency_ms": serial_reference,
        "full_minus_serial_reference_ms": residual,
        "estimation_tolerance_ms": tolerance,
        "estimate_status": estimate_status,
        "observed_full_minus_compute_ms": full_l1_ms - compute_only_ms,
        "estimated_exposed_layout_communication_ms": exposed,
        "estimated_hidden_layout_communication_ms": hidden,
        "estimated_overlap_ratio": ratio,
        "definition": "1-(full_l1-compute_only)/layer2 when independently measured components are additive within tolerance",
        "interpretation": "diagnostic estimate only; null means unmeasured composition or cross-measurement effects prevent attribution",
    }


def _ulysses_diagnostics(
    case: dict[str, Any],
    candidate: dict[str, Any],
    device: Any,
    seed: int,
    rank: int,
    warmup: int,
    iterations: int,
) -> dict[str, Any]:
    import torch.distributed as dist

    from lightx2v.common.ops.attn.ulysses_a2a import create_ulysses_a2a_backend
    from lightx2v.common.ops.attn.ulysses_attn import UlyssesAttnWeight
    from lightx2v.common.ops.attn.ulysses_prepost import create_ulysses_prepost_backend
    from lightx2v.utils.registry_factory import ATTN_WEIGHT_REGISTER

    shape = case["shape"]
    q, k, v, _, _, _ = _make_inputs(case, device, seed, rank)
    world_size = dist.get_world_size()
    local_len, _, hidden_dims = q.shape
    shard_heads = shape["heads"] // world_size
    quant = candidate["quant_scheme"]
    fusion = candidate["tensor_fusion"]
    prepost = create_ulysses_prepost_backend(candidate["prepost_backend"])
    a2a = create_ulysses_a2a_backend(candidate["a2a_backend"])

    def pack_qkv() -> Any:
        return prepost.pack_qkv(q, k, v, world_size, quant, fusion)

    packed_qkv = pack_qkv()

    def exchange_qkv() -> Any:
        return UlyssesAttnWeight._exchange_packed(packed_qkv, a2a, None)

    exchanged_qkv = exchange_qkv()

    def unpack_qkv() -> Any:
        return prepost.unpack_qkv(exchanged_qkv, q, k, v, None, None, None, rank, world_size, False)

    attn_q, attn_k, attn_v = unpack_qkv()
    fake_output = attn_q.reshape(shape["sequence"], -1)

    def pack_output() -> Any:
        return prepost.pack_attn(fake_output, local_len, world_size, shard_heads, hidden_dims, quant)

    packed_output = pack_output()

    def exchange_output() -> Any:
        return UlyssesAttnWeight._exchange_packed(packed_output, a2a, None)

    exchanged_output = exchange_output()

    def unpack_output() -> Any:
        return prepost.unpack_attn(exchanged_output, q.dtype, hidden_dims)

    def total_path() -> Any:
        current_qkv = prepost.pack_qkv(q, k, v, world_size, quant, fusion)
        current_qkv = UlyssesAttnWeight._exchange_packed(current_qkv, a2a, None)
        current_q, _, _ = prepost.unpack_qkv(current_qkv, q, k, v, None, None, None, rank, world_size, False)
        current_output = current_q.reshape(shape["sequence"], -1)
        current_output = prepost.pack_attn(current_output, local_len, world_size, shard_heads, hidden_dims, quant)
        current_output = UlyssesAttnWeight._exchange_packed(current_output, a2a, None)
        return prepost.unpack_attn(current_output, q.dtype, hidden_dims)

    def communication_path() -> Any:
        first = UlyssesAttnWeight._exchange_packed(packed_qkv, a2a, None)
        second = UlyssesAttnWeight._exchange_packed(packed_output, a2a, None)
        return first, second

    segment_fns = {
        "pack_qkv": pack_qkv,
        "exchange_qkv": exchange_qkv,
        "unpack_qkv": unpack_qkv,
        "pack_output": pack_output,
        "exchange_output": exchange_output,
        "unpack_output": unpack_output,
    }
    segments = {name: {**_diagnostic_measure(fn, warmup, iterations, device), "calls_per_attention": 1} for name, fn in segment_fns.items()}
    total = _diagnostic_measure(total_path, warmup, iterations, device)
    communication = _diagnostic_measure(communication_path, warmup, iterations, device)
    dense_operator = ATTN_WEIGHT_REGISTER[candidate["dense_backend"]]()
    dense_kwargs = UlyssesAttnWeight._dense_attention_kwargs({"causal": shape["causal"]}, attn_q, attn_k)

    def compute_only() -> Any:
        return dense_operator.apply(q=attn_q, k=attn_k, v=attn_v, **dense_kwargs)

    compute = _diagnostic_measure(compute_only, warmup, iterations, device)
    full_fn, _ = _prepare_operation(case, candidate, device, seed, rank)
    full = _diagnostic_measure(full_fn, warmup, iterations, device)
    logical_bytes = _tensor_bytes(packed_qkv) + _tensor_bytes(packed_output)
    network_bytes = logical_bytes * (world_size - 1) // world_size
    collective_calls = sum(1 + int(scale is not None) for payload, scale in (*packed_qkv, *packed_output))
    serial_sum = sum(item["latency_ms_mean"] for item in segments.values())
    return {
        "layer2": {
            "total": total,
            "segments": segments,
            "segment_serial_sum_ms": serial_sum,
            "total_minus_segment_sum_ms": total["latency_ms_mean"] - serial_sum,
        },
        "layer3": {
            **communication,
            **_communication_rates(logical_bytes, network_bytes, communication["latency_ms_mean"], world_size),
            "communication_pattern": "all_to_all",
            "primitive": candidate["a2a_backend"],
            "collective_calls_per_attention": collective_calls,
        },
        "overlap": _overlap_estimate(full["latency_ms_mean"], compute["latency_ms_mean"], total["latency_ms_mean"]),
    }


def _ring_diagnostics(
    case: dict[str, Any],
    candidate: dict[str, Any],
    device: Any,
    seed: int,
    rank: int,
    warmup: int,
    iterations: int,
) -> dict[str, Any]:
    import torch
    import torch.distributed as dist

    from lightx2v.common.ops.attn.ring_attn import RingAttnWeight, _merge_attention_blocks
    from lightx2v.common.ops.attn.utils.ring_comm import RingComm
    from lightx2v.common.ops.attn.utils.seq_p import pack_seq_p_tensor, unpack_seq_p_tensor
    from lightx2v.utils.registry_factory import ATTN_WEIGHT_REGISTER

    q, k, v, _, _, _ = _make_inputs(case, device, seed, rank)
    world_size = dist.get_world_size()
    hidden_dims = q.shape[-1]
    quant = candidate["quant_scheme"]
    fusion = candidate["tensor_fusion"]

    def pack_kv() -> Any:
        if fusion:
            return (pack_seq_p_tensor(torch.cat((k, v), dim=0), quant),)
        return pack_seq_p_tensor(k, quant), pack_seq_p_tensor(v, quant)

    packed_kv = pack_kv()

    def unpack_kv(packed: Any = packed_kv) -> Any:
        if fusion:
            current_kv = unpack_seq_p_tensor(packed[0], q.dtype, hidden_dims)
            return current_kv.chunk(2, dim=0)
        return unpack_seq_p_tensor(packed[0], q.dtype, hidden_dims), unpack_seq_p_tensor(packed[1], q.dtype, hidden_dims)

    def rotate_step(comm: Any, packed: Any) -> Any:
        next_packed = []
        for payload, scale in packed:
            next_payload = comm.enqueue_send_recv(payload)
            next_scale = None if scale is None else comm.enqueue_send_recv(scale)
            next_packed.append((next_payload, next_scale))
        comm.commit()
        comm.wait()
        return tuple(next_packed)

    def rotate_path() -> Any:
        current = packed_kv
        comm = RingComm(None)
        for _ in range(world_size - 1):
            current = rotate_step(comm, current)
        return current

    def total_path() -> Any:
        current = pack_kv()
        comm = RingComm(None)
        current_k = None
        for step in range(world_size):
            current_k, _ = unpack_kv(current)
            if step + 1 < world_size:
                current = rotate_step(comm, current)
        return current_k

    segments = {
        "pack_kv": {**_diagnostic_measure(pack_kv, warmup, iterations, device), "calls_per_attention": 1},
        "unpack_kv_block": {**_diagnostic_measure(unpack_kv, warmup, iterations, device), "calls_per_attention": world_size},
        "ring_rotate": {**_diagnostic_measure(rotate_path, warmup, iterations, device), "calls_per_attention": 1},
    }
    total = _diagnostic_measure(total_path, warmup, iterations, device)
    communication = _diagnostic_measure(rotate_path, warmup, iterations, device)
    gathered_k = [torch.empty_like(k) for _ in range(world_size)]
    gathered_v = [torch.empty_like(v) for _ in range(world_size)]
    dist.all_gather(gathered_k, k)
    dist.all_gather(gathered_v, v)
    dense_operator = ATTN_WEIGHT_REGISTER[candidate["dense_backend"]]()

    def compute_only() -> Any:
        output = lse = None
        for block_k, block_v in zip(gathered_k, gathered_v):
            block_output, block_lse = RingAttnWeight._apply_attention_block(dense_operator, q, block_k, block_v, {})
            output, lse = _merge_attention_blocks(output, lse, block_output, block_lse)
        return output

    compute = _diagnostic_measure(compute_only, warmup, iterations, device)
    full_fn, _ = _prepare_operation(case, candidate, device, seed, rank)
    full = _diagnostic_measure(full_fn, warmup, iterations, device)
    logical_bytes = _tensor_bytes(packed_kv) * (world_size - 1)
    serial_sum = sum(item["latency_ms_mean"] * item["calls_per_attention"] for item in segments.values())
    return {
        "layer2": {
            "total": total,
            "segments": segments,
            "segment_serial_sum_ms": serial_sum,
            "total_minus_segment_sum_ms": total["latency_ms_mean"] - serial_sum,
        },
        "layer3": {
            **communication,
            **_communication_rates(logical_bytes, logical_bytes, communication["latency_ms_mean"], world_size),
            "communication_pattern": "ring_p2p",
            "primitive": "ring_p2p",
            "p2p_steps_per_attention": world_size - 1,
        },
        "overlap": _overlap_estimate(full["latency_ms_mean"], compute["latency_ms_mean"], total["latency_ms_mean"]),
    }


def prepare_diagnostics(
    case: dict[str, Any],
    candidate: dict[str, Any],
    device: Any,
    seed: int,
    rank: int,
    warmup: int,
    iterations: int,
) -> dict[str, Any]:
    require(not candidate["head_parallel"], "L2/L3 diagnostics do not support head_parallel; use L1 to measure the production pipeline")
    require(case["shape"]["aux_tokens"] == 0, "L2/L3 diagnostics currently require aux_tokens=0")
    if candidate["algorithm"] == "ulysses":
        return _ulysses_diagnostics(case, candidate, device, seed, rank, warmup, iterations)
    return _ring_diagnostics(case, candidate, device, seed, rank, warmup, iterations)


def _git_commit() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True, timeout=5).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def _command_output(argv: list[str]) -> list[str]:
    try:
        result = subprocess.run(argv, check=False, capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return []
    return result.stdout.splitlines() if result.returncode == 0 else []


def _distributed_environment(device: Any, rank: int, local_rank: int) -> list[dict[str, Any]]:
    import torch
    import torch.distributed as dist

    value = environment(str(device))
    properties = torch.cuda.get_device_properties(device)
    value["gpu"].update(
        {
            "uuid": str(properties.uuid),
            "pci_bus_id": properties.pci_bus_id,
            "pci_device_id": properties.pci_device_id,
            "pci_domain_id": properties.pci_domain_id,
        }
    )
    value.update(
        {
            "rank": rank,
            "local_rank": local_rank,
            "world_size": dist.get_world_size(),
            "git_commit": _git_commit(),
            "nccl_version": torch.cuda.nccl.version(),
            "nccl_environment": {name: os.environ[name] for name in NCCL_ENV_NAMES if name in os.environ},
        }
    )
    if rank == 0:
        value["nvidia_smi_topology"] = _command_output(["nvidia-smi", "topo", "-m"])
    else:
        value.pop("nvidia_smi", None)
    gathered: list[dict[str, Any] | None] = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, value)
    return [item for item in gathered if item is not None]


def _raw_record(
    case: dict[str, Any],
    candidate: dict[str, Any],
    repeat: int,
    latencies: list[float],
    per_rank_latencies: list[list[float]],
    result: Any,
    work: int,
    peak_memory: int,
    environments: list[dict[str, Any]],
    run_contract: dict[str, Any],
    catalog_fingerprint: str,
    finite: bool,
) -> dict[str, Any]:
    output = _output_tensor(result)
    mean_ms = statistics.mean(latencies)
    per_rank_means = [statistics.mean(values) for values in per_rank_latencies]
    return {
        "kind": "sp_attention_benchmark_raw_v1",
        "run_id": f"repeat-{repeat:03d}",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "case": case,
        "candidate": candidate,
        "catalog_fingerprint": catalog_fingerprint,
        "environment": {"ranks": environments},
        "status": "ok",
        "metrics": {
            "latency_ms_mean": mean_ms,
            "latency_ms_median": statistics.median(latencies),
            "per_rank_latency_ms_mean": per_rank_means,
            "aggregate_effective_tflops": work / (mean_ms / 1000) / 1e12,
            "global_dense_work": work,
            "work_definition": "4*heads*(sequence+aux_q)*(sequence+aux_kv)*head_dim",
            "max_memory_allocated_bytes": peak_memory,
            "measurement_scope": "production_sp_attention_max_rank_cuda_event_list_single_sync",
            "output_shape": list(output.shape),
        },
        "correctness": {"checked": True, "passed": finite, "check": "finite_output_all_ranks"},
        "run": run_contract,
    }


def _error_record(
    case: dict[str, Any],
    candidate: dict[str, Any],
    repeat: int,
    environments: list[dict[str, Any]],
    run_contract: dict[str, Any],
    catalog_fingerprint: str,
    errors: list[dict[str, Any] | None],
    stage: str,
) -> dict[str, Any]:
    return {
        "kind": "sp_attention_benchmark_raw_v1",
        "run_id": f"repeat-{repeat:03d}",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "case": case,
        "candidate": candidate,
        "catalog_fingerprint": catalog_fingerprint,
        "environment": {"ranks": environments},
        "status": "error",
        "error": {"stage": stage, "ranks": errors},
        "run": run_contract,
    }


def load_raw_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        record = json.loads(line)
        require(isinstance(record, dict), f"SP raw record must be an object: {path}:{line_no}")
        records.append(record)
    return records


def _record_key(record: dict[str, Any]) -> tuple[str, str, str]:
    return record["run_id"], record["case"]["case_id"], record["candidate"]["candidate_id"]


def _validate_report_record_identities(
    suite: dict[str, Any],
    records: list[dict[str, Any]],
    *,
    expected_kind: str,
    candidate_catalog: dict[str, Any] | None = None,
) -> None:
    cases = {case["case_id"]: case for case in suite["cases"]}
    catalog_candidates = {
        (case["case_id"], candidate["candidate_id"]): candidate
        for case in (candidate_catalog or {}).get("cases", [])
        for candidate in case["candidates"]
    }
    expected_fingerprint = (candidate_catalog or {}).get("catalog_fingerprint")
    keys = set()
    seen_candidates = {}
    run_contracts = set()
    environments = set()
    for record in records:
        require(record.get("kind") in {None, expected_kind}, f"raw kind must be {expected_kind}")
        try:
            key = _record_key(record)
        except (KeyError, TypeError):
            raise ValueError("raw record lacks run/case/candidate identity") from None
        require(key not in keys, f"duplicate raw identity: {key}")
        keys.add(key)
        _, case_id, candidate_id = key
        require(case_id in cases and record["case"] == cases[case_id], f"raw suite drift: {case_id}")
        candidate_key = (case_id, candidate_id)
        candidate = record["candidate"]
        if candidate_key in catalog_candidates:
            require(candidate == catalog_candidates[candidate_key], f"raw candidate contract drift: {candidate_id}")
        elif candidate_catalog is not None:
            raise ValueError(f"raw candidate is absent from the catalog: {case_id}.{candidate_id}")
        previous = seen_candidates.setdefault(candidate_key, candidate)
        require(candidate == previous, f"raw candidate differs across repeats: {candidate_id}")
        run_contracts.add(json.dumps(record.get("run") or {}, sort_keys=True))
        ranks = (record.get("environment") or {}).get("ranks") or []
        environments.add(json.dumps(_environment_identity(ranks), sort_keys=True))
        if expected_fingerprint is not None:
            require(record.get("catalog_fingerprint") == expected_fingerprint, "raw candidate catalog differs")
    require(len(run_contracts) <= 1, "mixed SP raw measurement contracts")
    require(len(environments) <= 1, "mixed SP raw hardware or software environments")


def _environment_identity(environments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "rank": value.get("rank"),
            "local_rank": value.get("local_rank"),
            "world_size": value.get("world_size"),
            "gpu": value.get("gpu"),
            "torch": value.get("torch"),
            "cuda_runtime": value.get("cuda_runtime"),
            "cuda_visible_devices": value.get("cuda_visible_devices"),
            "git_commit": value.get("git_commit"),
            "nccl_version": list(value.get("nccl_version") or []),
            "nccl_environment": value.get("nccl_environment"),
        }
        for value in environments
    ]


def validate_existing_records(
    suite: dict[str, Any],
    candidate_catalog: dict[str, Any],
    records: list[dict[str, Any]],
    run_contract: dict[str, Any],
    environments: list[dict[str, Any]],
    repeat_runs: int,
) -> set[tuple[str, str, str]]:
    cases = {case["case_id"]: case for case in suite["cases"]}
    candidates = {
        (case["case_id"], candidate["candidate_id"]): candidate
        for case in candidate_catalog["cases"]
        for candidate in case["candidates"]
    }
    expected_environment = _environment_identity(environments)
    keys = set()
    for record in records:
        require(record.get("kind") == "sp_attention_benchmark_raw_v1", "existing SP raw kind differs")
        require(record.get("catalog_fingerprint") == candidate_catalog["catalog_fingerprint"], "existing SP candidate catalog differs")
        require(record.get("run") == run_contract, "existing SP measurement arguments differ")
        key = _record_key(record)
        require(key not in keys, f"duplicate SP raw identity: {key}")
        run_id, case_id, candidate_id = key
        require(run_id.startswith("repeat-") and int(run_id.removeprefix("repeat-")) < repeat_runs, f"existing SP repeat is outside requested range: {run_id}")
        require(case_id in cases and record["case"] == cases[case_id], f"existing SP suite drift: {case_id}")
        require((case_id, candidate_id) in candidates, f"existing SP candidate is no longer eligible: {case_id}.{candidate_id}")
        require(record["candidate"] == candidates[(case_id, candidate_id)], f"existing SP candidate contract drift: {candidate_id}")
        require(_environment_identity(record["environment"]["ranks"]) == expected_environment, "existing SP hardware or software environment differs")
        keys.add(key)
    return keys


def build_report(
    suite: dict[str, Any],
    records: list[dict[str, Any]],
    *,
    candidate_catalog: dict[str, Any] | None = None,
    required_runs: int = 3,
    max_spread_pct: float = 5.0,
    max_spread_ms: float = 0.005,
) -> dict[str, Any]:
    validate_suite(suite)
    _validate_report_record_identities(
        suite,
        records,
        expected_kind="sp_attention_benchmark_raw_v1",
        candidate_catalog=candidate_catalog,
    )
    grouped: defaultdict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[(record["case"]["case_id"], record["candidate"]["candidate_id"])].append(record)
    catalog_cases = {case["case_id"]: case for case in (candidate_catalog or {}).get("cases", [])}
    workloads = []
    for case in suite["cases"]:
        expected = {
            candidate["candidate_id"]: candidate
            for candidate in catalog_cases.get(case["case_id"], {}).get("candidates", [])
        }
        if not expected:
            expected = {
                candidate_id: items[0]["candidate"]
                for (case_id, candidate_id), items in grouped.items()
                if case_id == case["case_id"]
            }
        candidates = {}
        for candidate_id, candidate in sorted(expected.items()):
            items = grouped.get((case["case_id"], candidate_id), [])
            errors = [item for item in items if item.get("status") != "ok"]
            correctness_failed = any((item.get("correctness") or {}).get("passed") is False for item in items)
            accepted_items = [
                item
                for item in items
                if item.get("status") == "ok" and (item.get("correctness") or {}).get("passed") is not False
            ]
            latencies = [float(item["metrics"]["latency_ms_mean"]) for item in accepted_items]
            median = statistics.median(latencies) if latencies else None
            spread_ms = max(latencies) - min(latencies) if len(latencies) > 1 else None
            spread_pct = spread_ms / median * 100 if spread_ms is not None and median else None
            if not items:
                status = "not_measured"
            elif errors and accepted_items:
                status = "measurement_error"
            elif errors:
                status = "unavailable" if all((item.get("error") or {}).get("stage") == "prepare" for item in errors) else "measurement_error"
            elif correctness_failed:
                status = "correctness_failed"
            elif len(latencies) < required_runs:
                status = "insufficient_runs"
            elif spread_pct is not None and spread_pct > max_spread_pct and spread_ms > max_spread_ms:
                status = "unstable"
            else:
                status = "accepted"
            candidates[candidate_id] = {
                "status": status,
                "runs": len(latencies),
                "latency_ms": median,
                "spread_pct": spread_pct,
                "spread_ms": spread_ms,
                "aggregate_effective_tflops": statistics.median(item["metrics"]["aggregate_effective_tflops"] for item in accepted_items) if latencies else None,
                "candidate": candidate,
                "errors": [item.get("error") for item in errors],
            }
        ranking = sorted((name for name, item in candidates.items() if item["status"] == "accepted"), key=lambda name: candidates[name]["latency_ms"])
        terminal_statuses = {"accepted", "unavailable", "correctness_failed"}
        coverage_complete = (
            candidate_catalog is not None
            and candidate_catalog.get("scope_complete") is True
            and bool(candidates)
            and all(item["status"] in terminal_statuses for item in candidates.values())
        )
        measured_winner = ranking[0] if ranking else None
        winner = measured_winner if coverage_complete else None
        observed = case.get("observed_candidate")
        observed_gap = None
        if measured_winner and observed in candidates and candidates[observed]["status"] == "accepted":
            observed_gap = candidates[observed]["latency_ms"] / candidates[measured_winner]["latency_ms"]
        workloads.append(
            {
                "case_id": case["case_id"],
                "winner": winner,
                "measured_winner": measured_winner,
                "candidate_coverage_complete": coverage_complete,
                "recommendation_scope": "complete_candidate_space" if coverage_complete else "measured_subset",
                "ranking": ranking,
                "observed_candidate": observed,
                "observed_to_winner_ratio": observed_gap,
                "candidates": candidates,
            }
        )
    return {
        "kind": "sp_attention_benchmark_report_v1",
        "suite_id": suite["suite_id"],
        "catalog_fingerprint": (candidate_catalog or {}).get("catalog_fingerprint"),
        "catalog_scope_complete": (candidate_catalog or {}).get("scope_complete", False),
        "catalog_scope_blockers": (candidate_catalog or {}).get("backend_scope_blockers", {}),
        "policy": {"required_runs": required_runs, "max_spread_pct": max_spread_pct, "max_spread_ms": max_spread_ms},
        "summary": {
            "workload_count": len(workloads),
            "recommended_count": sum(item["winner"] is not None for item in workloads),
            "coverage_complete": bool(workloads) and all(item["candidate_coverage_complete"] for item in workloads),
        },
        "workloads": workloads,
    }


def _local_error(exc: Exception) -> dict[str, Any]:
    return {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc(limit=8)}


def _gather_rank_errors(local_error: dict[str, Any] | None) -> list[dict[str, Any] | None]:
    import torch.distributed as dist

    errors: list[dict[str, Any] | None] = [None] * dist.get_world_size()
    dist.all_gather_object(errors, local_error)
    return errors


def run_suite(
    suite: dict[str, Any],
    output_dir: Path,
    *,
    candidate_ids: set[str],
    algorithms: set[str],
    dense_backends: list[str],
    repeat_runs: int,
    warmup: int,
    iterations: int,
    seed: int,
    append: bool,
    required_runs: int,
    max_spread_pct: float,
    max_spread_ms: float,
) -> dict[str, Any]:
    import torch
    import torch.distributed as dist

    validate_suite(suite)
    require(repeat_runs > 0 and iterations > 0 and warmup >= 0, "repeat-runs and iterations must be positive; warmup must be non-negative")
    require("RANK" in os.environ and "LOCAL_RANK" in os.environ, "SP run must be launched with torchrun")
    raw_path = output_dir / "raw.jsonl"
    require(append or not raw_path.exists(), f"SP raw output already exists; use --append: {raw_path}")
    dist.init_process_group(backend="nccl")
    try:
        rank = dist.get_rank()
        local_rank = int(os.environ["LOCAL_RANK"])
        device = torch.device("cuda", local_rank)
        torch.cuda.set_device(device)
        world_size = dist.get_world_size()
        require(all(case["shape"]["sp_size"] == world_size for case in suite["cases"]), "every case sp_size must match torchrun world size")
        runtime = _discover_runtime(str(device))
        candidate_catalog = build_candidate_catalog(suite, runtime)
        selection_runtime = dict(runtime)
        if dense_backends:
            unavailable = sorted(set(dense_backends) - set(runtime["dense_backends"]))
            require(not unavailable, f"dense backends are not eligible on this device: {unavailable}")
            selection_runtime["dense_backends"] = dense_backends
            selection_runtime["ring_lse_backends"] = runtime["ring_lse_backends"] & set(dense_backends)
        selections = _selected_candidates(suite, selection_runtime, candidate_ids, algorithms)
        environments = _distributed_environment(device, rank, local_rank)
        run_contract = {
            "warmup": warmup,
            "iterations": iterations,
            "seed": seed,
            "world_size": world_size,
            "timing_mode": "max_rank_event_list_single_sync",
        }
        state: list[Any] = [None, None]
        if rank == 0:
            try:
                existing_records = load_raw_records(raw_path) if append else []
                existing_keys = validate_existing_records(
                    suite,
                    candidate_catalog,
                    existing_records,
                    run_contract,
                    environments,
                    repeat_runs,
                )
                state = [None, sorted(existing_keys)]
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                state = [str(exc), None]
        dist.broadcast_object_list(state, src=0)
        require(state[0] is None, str(state[0]))
        existing = {tuple(key) for key in state[1]}
        handle = None
        if rank == 0:
            output_dir.mkdir(parents=True, exist_ok=True)
            write_json(output_dir / "candidate_catalog.json", candidate_catalog)
            handle = raw_path.open("a", encoding="utf-8")
        written = 0
        for repeat in range(repeat_runs):
            for case in suite["cases"]:
                candidates = selections[case["case_id"]]
                candidates = candidates[repeat % len(candidates) :] + candidates[: repeat % len(candidates)]
                for candidate in candidates:
                    key = (f"repeat-{repeat:03d}", case["case_id"], candidate["candidate_id"])
                    if key in existing:
                        continue
                    try:
                        fn, work = _prepare_operation(case, candidate, device, seed + repeat * 1000, rank)
                        local_error = None
                    except Exception as exc:
                        local_error = _local_error(exc)
                    errors = _gather_rank_errors(local_error)
                    if any(error is not None for error in errors):
                        record = _error_record(
                            case,
                            candidate,
                            repeat,
                            environments,
                            run_contract,
                            candidate_catalog["catalog_fingerprint"],
                            errors,
                            "prepare",
                        )
                    else:
                        try:
                            latencies, per_rank, result, peak_memory, finite = _measure_distributed(fn, warmup, iterations, device)
                            local_error = None
                        except Exception as exc:
                            local_error = _local_error(exc)
                        errors = _gather_rank_errors(local_error)
                        if any(error is not None for error in errors):
                            record = _error_record(
                                case,
                                candidate,
                                repeat,
                                environments,
                                run_contract,
                                candidate_catalog["catalog_fingerprint"],
                                errors,
                                "measure",
                            )
                        else:
                            record = _raw_record(
                                case,
                                candidate,
                                repeat,
                                latencies,
                                per_rank,
                                result,
                                work,
                                peak_memory,
                                environments,
                                run_contract,
                                candidate_catalog["catalog_fingerprint"],
                                finite,
                            )
                    if rank == 0:
                        handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
                        handle.flush()
                        written += 1
        dist.barrier()
        if rank == 0:
            handle.close()
            records = load_raw_records(raw_path)
            report = build_report(
                suite,
                records,
                candidate_catalog=candidate_catalog,
                required_runs=required_runs,
                max_spread_pct=max_spread_pct,
                max_spread_ms=max_spread_ms,
            )
            write_json(output_dir / "report.json", report)
            summary = {
                "raw": str(raw_path),
                "written": written,
                "skipped_existing": len(existing),
                "record_count": len(records),
                **report["summary"],
            }
            write_json(output_dir / "run_summary.json", summary)
            print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
            return summary
        return {"rank": rank, "record_count": 0}
    finally:
        dist.destroy_process_group()


def _topology_fingerprint(records: list[dict[str, Any]]) -> str | None:
    environments = [record.get("environment", {}).get("ranks") for record in records]
    environments = [value for value in environments if value]
    if not environments:
        return None
    identities = []
    for ranks in environments:
        rank_zero = next((item for item in ranks if item.get("rank") == 0), ranks[0])
        raw_topology = rank_zero.get("nvidia_smi_topology") or []
        if not raw_topology:
            return None
        topology = [re.sub(r"\x1b\[[0-9;]*m", "", line).strip() for line in raw_topology]
        topology = [" ".join(line.split()) for line in topology if line.strip()]
        if any(
            item.get("gpu", {}).get(field) is None
            for item in ranks
            for field in ("name", "major", "minor", "pci_domain_id", "pci_bus_id", "pci_device_id")
        ):
            return None
        rank_map = [
            {
                "rank": item.get("rank"),
                "local_rank": item.get("local_rank"),
                "gpu_name": item.get("gpu", {}).get("name"),
                "cuda_capability": f"{item.get('gpu', {}).get('major')}.{item.get('gpu', {}).get('minor')}",
                "pci": [
                    item.get("gpu", {}).get("pci_domain_id"),
                    item.get("gpu", {}).get("pci_bus_id"),
                    item.get("gpu", {}).get("pci_device_id"),
                ],
            }
            for item in sorted(ranks, key=lambda value: value.get("rank", -1))
        ]
        identities.append({"rank_map": rank_map, "nvidia_smi_topology": topology})
    canonical = json.dumps(identities[0], sort_keys=True, separators=(",", ":"))
    require(all(json.dumps(item, sort_keys=True, separators=(",", ":")) == canonical for item in identities), "diagnostic topology differs across records")
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _interconnect_platform(
    profiles: dict[str, Any] | None,
    platform_id: str | None,
    records: list[dict[str, Any]],
) -> tuple[dict[str, Any] | None, str | None]:
    fingerprint = _topology_fingerprint(records)
    if profiles is None:
        require(platform_id is None, "--platform requires --interconnect-peaks")
        return None, fingerprint
    require(bool(platform_id), "--platform is required with --interconnect-peaks")
    require(profiles.get("schema_version") == 1, "interconnect profile schema_version must be 1")
    platform_value = (profiles.get("platforms") or {}).get(platform_id)
    require(isinstance(platform_value, dict), f"unknown interconnect platform: {platform_id}")
    identification = platform_value.get("identification") or {}
    require(isinstance(identification, dict), "interconnect identification must be an object")
    require(fingerprint is not None, "diagnostic raw does not contain topology identity")
    ranks = next(record["environment"]["ranks"] for record in records if record.get("environment", {}).get("ranks"))
    gpus = [item.get("gpu", {}) for item in ranks]
    if identification.get("gpu_name_regex"):
        require(all(re.search(str(identification["gpu_name_regex"]), str(gpu.get("name"))) for gpu in gpus), "raw GPU name does not match interconnect profile")
    if identification.get("cuda_capability"):
        capabilities = {f"{gpu.get('major')}.{gpu.get('minor')}" for gpu in gpus}
        require(capabilities == {str(identification["cuda_capability"])}, "raw CUDA capability does not match interconnect profile")
    if identification.get("world_size") is not None:
        world_size = _positive_int(identification["world_size"], "identification.world_size")
        require(identification["world_size"] == world_size, "identification.world_size must be a canonical integer")
        require(world_size == len(ranks), "raw world size does not match interconnect profile")
    require(identification.get("topology_fingerprint") == fingerprint, "raw topology does not match interconnect profile")
    peaks = platform_value.get("interconnect_peaks")
    require(isinstance(peaks, dict) and peaks, "interconnect_peaks must be a non-empty object")
    for pattern, entry in peaks.items():
        require(pattern in {"all_to_all", "ring_p2p"}, f"unsupported communication pattern: {pattern}")
        require(isinstance(entry, dict), f"interconnect peak must be an object: {pattern}")
        rate = entry.get("bus_bandwidth")
        require(isinstance(rate, (int, float)) and not isinstance(rate, bool) and rate > 0, f"invalid interconnect peak: {pattern}")
        require(entry.get("unit") == "GB/s", f"interconnect peak unit must be GB/s: {pattern}")
        require(entry.get("kind") in {"theoretical", "empirical_envelope"}, f"invalid interconnect peak kind: {pattern}")
        require(isinstance(entry.get("source"), str) and entry["source"], f"interconnect peak source is required: {pattern}")
    return platform_value, fingerprint


def build_diagnostic_report(
    suite: dict[str, Any],
    records: list[dict[str, Any]],
    required_runs: int = 3,
    max_spread_pct: float = 5.0,
    max_spread_ms: float = 0.005,
    interconnect_profiles: dict[str, Any] | None = None,
    platform_id: str | None = None,
) -> dict[str, Any]:
    platform_value, topology_fingerprint = _interconnect_platform(interconnect_profiles, platform_id, records)
    grouped: defaultdict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[(record["case"]["case_id"], record["candidate"]["candidate_id"])].append(record)
    results = []
    for (case_id, candidate_id), items in sorted(grouped.items()):
        ok = [item for item in items if item.get("status") == "ok"]
        errors = [item.get("error") for item in items if item.get("status") != "ok"]
        layer2_values = [float(item["metrics"]["layer2"]["total"]["latency_ms_mean"]) for item in ok]
        layer3_values = [float(item["metrics"]["layer3"]["latency_ms_mean"]) for item in ok]
        layer2_spread_ms = max(layer2_values) - min(layer2_values) if len(layer2_values) > 1 else None
        layer3_spread_ms = max(layer3_values) - min(layer3_values) if len(layer3_values) > 1 else None
        layer2_median = statistics.median(layer2_values) if layer2_values else None
        layer3_median = statistics.median(layer3_values) if layer3_values else None
        layer2_spread_pct = layer2_spread_ms / layer2_median * 100 if layer2_spread_ms is not None and layer2_median else None
        layer3_spread_pct = layer3_spread_ms / layer3_median * 100 if layer3_spread_ms is not None and layer3_median else None
        unstable = any(
            spread_pct is not None and spread_pct > max_spread_pct and spread_ms > max_spread_ms
            for spread_pct, spread_ms in ((layer2_spread_pct, layer2_spread_ms), (layer3_spread_pct, layer3_spread_ms))
        )
        segment_names = sorted({name for item in ok for name in item["metrics"]["layer2"]["segments"]})
        segments = {
            name: {
                "latency_ms": statistics.median(item["metrics"]["layer2"]["segments"][name]["latency_ms_mean"] for item in ok),
                "calls_per_attention": ok[0]["metrics"]["layer2"]["segments"][name]["calls_per_attention"],
            }
            for name in segment_names
        }
        if errors and not ok:
            status = "unavailable"
        elif errors:
            status = "measurement_error"
        elif len(ok) < required_runs:
            status = "insufficient_runs"
        elif unstable:
            status = "unstable"
        else:
            status = "accepted"
        layer3 = None
        overlap = None
        if ok:
            first = ok[0]["metrics"]["layer3"]
            communication_pattern = first.get("communication_pattern") or ("ring_p2p" if first["primitive"] == "ring_p2p" else "all_to_all")
            peak_entry = (platform_value or {}).get("interconnect_peaks", {}).get(communication_pattern)
            peak_status = "available" if peak_entry else ("profile_not_provided" if platform_value is None else "peak_missing")
            nominal_peak = float(peak_entry["bus_bandwidth"]) if peak_entry else None
            bus_bandwidth = statistics.median(item["metrics"]["layer3"]["bus_bandwidth_gbps"] for item in ok)
            layer3 = {
                "latency_ms": layer3_median,
                "spread_pct": layer3_spread_pct,
                "spread_ms": layer3_spread_ms,
                "logical_payload_bytes_per_rank": first["logical_payload_bytes_per_rank"],
                "network_bytes_per_rank": first["network_bytes_per_rank"],
                "aggregate_network_bytes": first["aggregate_network_bytes"],
                "algorithmic_bandwidth_gbps": statistics.median(item["metrics"]["layer3"]["algorithmic_bandwidth_gbps"] for item in ok),
                "bus_bandwidth_gbps": bus_bandwidth,
                "communication_pattern": communication_pattern,
                "primitive": first["primitive"],
                "peak_status": peak_status,
                "nominal_bus_peak_gbps": nominal_peak,
                "peak_kind": peak_entry.get("kind") if peak_entry else None,
                "peak_source": peak_entry.get("source") if peak_entry else None,
                "bus_peak_efficiency": bus_bandwidth / nominal_peak if status == "accepted" and nominal_peak else None,
                "efficiency_status": "available" if status == "accepted" and nominal_peak else ("measurement_not_accepted" if status != "accepted" else peak_status),
            }
            overlap = {
                name: statistics.median(item["metrics"]["overlap"][name] for item in ok)
                for name in (
                    "full_l1_latency_ms",
                    "compute_only_latency_ms",
                    "layer2_layout_communication_latency_ms",
                    "serial_reference_latency_ms",
                    "full_minus_serial_reference_ms",
                    "estimation_tolerance_ms",
                    "observed_full_minus_compute_ms",
                )
            }
            estimate_statuses = sorted({item["metrics"]["overlap"]["estimate_status"] for item in ok})
            overlap["estimate_status"] = estimate_statuses[0] if len(estimate_statuses) == 1 else "inconsistent_across_runs"
            for name in (
                "estimated_exposed_layout_communication_ms",
                "estimated_hidden_layout_communication_ms",
                "estimated_overlap_ratio",
            ):
                values = [item["metrics"]["overlap"][name] for item in ok]
                overlap[name] = statistics.median(values) if all(value is not None for value in values) else None
            overlap["definition"] = ok[0]["metrics"]["overlap"]["definition"]
            overlap["interpretation"] = ok[0]["metrics"]["overlap"]["interpretation"]
        results.append(
            {
                "case_id": case_id,
                "candidate_id": candidate_id,
                "status": status,
                "runs": len(ok),
                "layer2_latency_ms": layer2_median,
                "layer2_spread_pct": layer2_spread_pct,
                "layer2_spread_ms": layer2_spread_ms,
                "layer2_segments": segments,
                "layer3": layer3,
                "overlap": overlap,
                "errors": errors,
            }
        )
    return {
        "kind": "sp_attention_diagnostic_report_v1",
        "suite_id": suite["suite_id"],
        "hardware": {"platform_id": platform_id, "topology_fingerprint": topology_fingerprint},
        "policy": {"required_runs": required_runs, "max_spread_pct": max_spread_pct, "max_spread_ms": max_spread_ms},
        "summary": {"result_count": len(results), "accepted_count": sum(item["status"] == "accepted" for item in results)},
        "results": results,
    }


def run_diagnostics(
    suite: dict[str, Any],
    output_dir: Path,
    *,
    candidate_ids: set[str],
    dense_backends: list[str],
    repeat_runs: int,
    warmup: int,
    iterations: int,
    seed: int,
    required_runs: int,
    max_spread_pct: float,
    max_spread_ms: float,
    interconnect_profiles: dict[str, Any] | None,
    platform_id: str | None,
) -> dict[str, Any]:
    import torch
    import torch.distributed as dist

    validate_suite(suite)
    require(candidate_ids, "diagnose requires at least one explicit --candidate")
    require(repeat_runs > 0 and iterations > 0 and warmup >= 0, "repeat-runs and iterations must be positive; warmup must be non-negative")
    require("RANK" in os.environ and "LOCAL_RANK" in os.environ, "SP diagnostics must be launched with torchrun")
    raw_path = output_dir / "diagnostic_raw.jsonl"
    require(not raw_path.exists(), f"SP diagnostic output already exists: {raw_path}")
    dist.init_process_group(backend="nccl")
    try:
        rank = dist.get_rank()
        local_rank = int(os.environ["LOCAL_RANK"])
        device = torch.device("cuda", local_rank)
        torch.cuda.set_device(device)
        world_size = dist.get_world_size()
        require(all(case["shape"]["sp_size"] == world_size for case in suite["cases"]), "every case sp_size must match torchrun world size")
        runtime = _discover_runtime(str(device))
        candidate_catalog = build_candidate_catalog(suite, runtime)
        selection_runtime = dict(runtime)
        if dense_backends:
            unavailable = sorted(set(dense_backends) - set(runtime["dense_backends"]))
            require(not unavailable, f"dense backends are not eligible on this device: {unavailable}")
            selection_runtime["dense_backends"] = dense_backends
            selection_runtime["ring_lse_backends"] = runtime["ring_lse_backends"] & set(dense_backends)
        selections = _selected_candidates(suite, selection_runtime, candidate_ids, {"ulysses", "ring"})
        environments = _distributed_environment(device, rank, local_rank)
        _interconnect_platform(interconnect_profiles, platform_id, [{"environment": {"ranks": environments}}])
        handle = None
        if rank == 0:
            output_dir.mkdir(parents=True, exist_ok=True)
            write_json(output_dir / "candidate_catalog.json", candidate_catalog)
            handle = raw_path.open("w", encoding="utf-8")
        written = 0
        for repeat in range(repeat_runs):
            for case in suite["cases"]:
                candidates = selections[case["case_id"]]
                candidates = candidates[repeat % len(candidates) :] + candidates[: repeat % len(candidates)]
                for candidate in candidates:
                    try:
                        metrics = prepare_diagnostics(case, candidate, device, seed + repeat * 1000, rank, warmup, iterations)
                        local_error = None
                    except Exception as exc:
                        local_error = _local_error(exc)
                    errors = _gather_rank_errors(local_error)
                    if any(error is not None for error in errors):
                        record = {
                            "kind": "sp_attention_diagnostic_raw_v1",
                            "run_id": f"repeat-{repeat:03d}",
                            "case": case,
                            "candidate": candidate,
                            "catalog_fingerprint": candidate_catalog["catalog_fingerprint"],
                            "environment": {"ranks": environments},
                            "status": "error",
                            "error": {"ranks": errors},
                        }
                    else:
                        record = {
                            "kind": "sp_attention_diagnostic_raw_v1",
                            "run_id": f"repeat-{repeat:03d}",
                            "case": case,
                            "candidate": candidate,
                            "catalog_fingerprint": candidate_catalog["catalog_fingerprint"],
                            "environment": {"ranks": environments},
                            "status": "ok",
                            "metrics": metrics,
                            "run": {"warmup": warmup, "iterations": iterations, "timing_mode": "max_rank_event_list_single_sync"},
                        }
                    if rank == 0:
                        handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
                        handle.flush()
                        written += 1
        dist.barrier()
        if rank == 0:
            handle.close()
            records = load_raw_records(raw_path)
            report = build_diagnostic_report(
                suite,
                records,
                required_runs=required_runs,
                max_spread_pct=max_spread_pct,
                max_spread_ms=max_spread_ms,
                interconnect_profiles=interconnect_profiles,
                platform_id=platform_id,
            )
            write_json(output_dir / "diagnostic_report.json", report)
            summary = {"raw": str(raw_path), "written": written, **report["summary"]}
            write_json(output_dir / "diagnostic_summary.json", summary)
            print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
            return summary
        return {"rank": rank, "written": 0}
    finally:
        dist.destroy_process_group()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    inspect_parser = commands.add_parser("inspect", help="Validate and summarize an SP shape suite.")
    inspect_parser.add_argument("--suite", type=Path, required=True)
    candidate_parser = commands.add_parser("candidates", help="Probe and enumerate valid SP candidates.")
    candidate_parser.add_argument("--suite", type=Path, required=True)
    candidate_parser.add_argument("--device", default="cuda:0")
    candidate_parser.add_argument("--dense-backend", action="append", default=[])
    run_parser = commands.add_parser("run", help="Run production SP attention under torchrun.")
    run_parser.add_argument("--suite", type=Path, required=True)
    run_parser.add_argument("--output-dir", type=Path, required=True)
    run_parser.add_argument("--candidate", action="append", default=[])
    run_parser.add_argument("--algorithm", action="append", choices=("ulysses", "ring"), default=[])
    run_parser.add_argument("--dense-backend", action="append", default=[])
    run_parser.add_argument("--repeat-runs", type=int, default=3)
    run_parser.add_argument("--warmup", type=int, default=10)
    run_parser.add_argument("--iterations", type=int, default=30)
    run_parser.add_argument("--seed", type=int, default=42)
    run_parser.add_argument("--append", action="store_true")
    run_parser.add_argument("--required-runs", type=int, default=3)
    run_parser.add_argument("--max-spread-pct", type=float, default=5.0)
    run_parser.add_argument("--max-spread-ms", type=float, default=0.005)
    diagnose_parser = commands.add_parser("diagnose", help="Measure SP layout/communication and raw communication paths.")
    diagnose_parser.add_argument("--suite", type=Path, required=True)
    diagnose_parser.add_argument("--output-dir", type=Path, required=True)
    diagnose_parser.add_argument("--candidate", action="append", required=True)
    diagnose_parser.add_argument("--dense-backend", action="append", default=[])
    diagnose_parser.add_argument("--repeat-runs", type=int, default=3)
    diagnose_parser.add_argument("--warmup", type=int, default=10)
    diagnose_parser.add_argument("--iterations", type=int, default=30)
    diagnose_parser.add_argument("--seed", type=int, default=42)
    diagnose_parser.add_argument("--required-runs", type=int, default=3)
    diagnose_parser.add_argument("--max-spread-pct", type=float, default=5.0)
    diagnose_parser.add_argument("--max-spread-ms", type=float, default=0.005)
    diagnose_parser.add_argument("--interconnect-peaks", type=Path)
    diagnose_parser.add_argument("--platform")
    report_diagnostics_parser = commands.add_parser("report-diagnostics", help="Rebuild an SP diagnostic report from raw records.")
    report_diagnostics_parser.add_argument("--suite", type=Path, required=True)
    report_diagnostics_parser.add_argument("--raw", type=Path, required=True)
    report_diagnostics_parser.add_argument("--output-dir", type=Path, required=True)
    report_diagnostics_parser.add_argument("--required-runs", type=int, default=3)
    report_diagnostics_parser.add_argument("--max-spread-pct", type=float, default=5.0)
    report_diagnostics_parser.add_argument("--max-spread-ms", type=float, default=0.005)
    report_diagnostics_parser.add_argument("--interconnect-peaks", type=Path)
    report_diagnostics_parser.add_argument("--platform")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        suite = load_suite(args.suite)
        if args.command == "inspect":
            print(json.dumps(inspect_suite(suite), ensure_ascii=False, sort_keys=True))
            return 0
        if args.command == "candidates":
            runtime = _discover_runtime(args.device)
            dense_backends = args.dense_backend or runtime["dense_backends"]
            require(dense_backends, "no eligible dense attention backend was discovered")
            value = build_candidate_catalog(suite, runtime, dense_backends)
            print(json.dumps(value, ensure_ascii=False, indent=2))
            return 0
        if args.command == "run":
            run_suite(
                suite,
                args.output_dir,
                candidate_ids=set(args.candidate),
                algorithms=set(args.algorithm or ("ulysses", "ring")),
                dense_backends=args.dense_backend,
                repeat_runs=args.repeat_runs,
                warmup=args.warmup,
                iterations=args.iterations,
                seed=args.seed,
                append=args.append,
                required_runs=args.required_runs,
                max_spread_pct=args.max_spread_pct,
                max_spread_ms=args.max_spread_ms,
            )
            return 0
        if args.command == "diagnose":
            require(bool(args.interconnect_peaks) == bool(args.platform), "--interconnect-peaks and --platform must be provided together")
            profiles = json.loads(args.interconnect_peaks.read_text(encoding="utf-8")) if args.interconnect_peaks else None
            run_diagnostics(
                suite,
                args.output_dir,
                candidate_ids=set(args.candidate),
                dense_backends=args.dense_backend,
                repeat_runs=args.repeat_runs,
                warmup=args.warmup,
                iterations=args.iterations,
                seed=args.seed,
                required_runs=args.required_runs,
                max_spread_pct=args.max_spread_pct,
                max_spread_ms=args.max_spread_ms,
                interconnect_profiles=profiles,
                platform_id=args.platform,
            )
            return 0
        if args.command == "report-diagnostics":
            require(bool(args.interconnect_peaks) == bool(args.platform), "--interconnect-peaks and --platform must be provided together")
            profiles = json.loads(args.interconnect_peaks.read_text(encoding="utf-8")) if args.interconnect_peaks else None
            report = build_diagnostic_report(
                suite,
                load_raw_records(args.raw),
                required_runs=args.required_runs,
                max_spread_pct=args.max_spread_pct,
                max_spread_ms=args.max_spread_ms,
                interconnect_profiles=profiles,
                platform_id=args.platform,
            )
            args.output_dir.mkdir(parents=True, exist_ok=True)
            write_json(args.output_dir / "diagnostic_report.json", report)
            summary = {"raw": str(args.raw), **report["summary"]}
            write_json(args.output_dir / "diagnostic_summary.json", summary)
            print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
            return 0
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    raise AssertionError(f"unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
