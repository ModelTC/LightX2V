"""Shape-driven distributed sequence-parallel attention benchmark."""

from __future__ import annotations

import argparse
import hashlib
import importlib
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
CASE_FIELDS = {"case_id", "shape", "precision", "call_count", "observed_candidate", "source", "tags", "replay", "sparse"}
SHAPE_FIELDS = {"sequence", "heads", "kv_heads", "head_dim", "sp_size", "aux_tokens", "aux_q", "aux_first", "causal"}
QUANT_SCHEMES = (None, "fp8", "fp4")
SP_REFERENCE_TOLERANCES = {
    "flash_attn2": {"bf16": 0.02, "fp16": 0.01, "kind": "exact"},
    "flash_attn3": {"bf16": 0.02, "fp16": 0.01, "kind": "exact"},
    "sage_attn2": {"bf16": 0.10, "fp16": 0.10, "kind": "approximate"},
    "torch_sdpa": {"bf16": 0.02, "fp16": 0.01, "kind": "exact"},
}
SPARSE_SP_BACKENDS = {
    "dynamic_sparse_triton_replay": {"factory": "dynamic", "operator": "triton"},
    "dynamic_sparse_sage2_replay": {"factory": "dynamic", "operator": "sage2"},
    "dynamic_sparse_sage3_replay": {"factory": "dynamic", "operator": "sage3"},
    "dynamic_sparse_fa4_replay": {"factory": "dynamic", "operator": "fa4"},
    "sparge_sage2_replay": {"factory": "sparge"},
    "spas_sage2_replay": {
        "factory": "class",
        "symbol": "lightx2v.common.ops.attn.sage_attn:SparseSageAttn2Weight",
    },
    "spas_sage3_replay": {
        "factory": "class",
        "symbol": "lightx2v.common.ops.attn.sage_attn:SparseSageAttn3Weight",
    },
    "spas_fa4_replay": {
        "factory": "class",
        "symbol": "lightx2v.common.ops.attn.flash_attn:SparseFlashAttn4Weight",
    },
}
SPARSE_SP_EXCLUSIONS = {
    "flash_attn3_replay": "dense replay baseline; use a dense SP suite for distributed backend comparison",
    "flash_attn4_replay": "dense replay baseline; use a dense SP suite for distributed backend comparison",
    "sage_attn3_replay": "dense replay baseline; use a dense SP suite for distributed backend comparison",
}
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
        replay = case.get("replay")
        sparse = case.get("sparse")
        require((replay is None) == (sparse is None), f"SP replay and sparse settings must be provided together: {case['case_id']}")
        if replay is not None:
            require(isinstance(replay, dict) and set(replay) == {"manifest", "sha256"}, f"invalid SP replay fields: {case['case_id']}")
            require(isinstance(replay["manifest"], str) and replay["manifest"], f"SP replay manifest is required: {case['case_id']}")
            require(bool(re.fullmatch(r"[0-9a-f]{64}", str(replay["sha256"]))), f"invalid SP replay sha256: {case['case_id']}")
            require(isinstance(sparse, dict) and set(sparse) == {"keep_ratio"}, f"sparse.keep_ratio is required: {case['case_id']}")
            keep_ratio = sparse["keep_ratio"]
            require(isinstance(keep_ratio, (int, float)) and not isinstance(keep_ratio, bool), f"invalid sparse keep_ratio: {case['case_id']}")
            require(0 < float(keep_ratio) <= 1, f"sparse keep_ratio must be in (0, 1]: {case['case_id']}")
            if case["shape"]["aux_tokens"]:
                require(case["shape"]["aux_q"] and case["shape"]["aux_first"], f"sparse SP replay aux tokens must be a replicated Q/K/V prefix: {case['case_id']}")
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


def derive_replicated_prefix_split(total_sequence: int, conditioner_tokens: int, sp_size: int) -> tuple[int, int]:
    total_sequence = _positive_int(total_sequence, "total_sequence")
    conditioner_tokens = _non_negative_int(conditioner_tokens, "conditioner_tokens")
    sp_size = _positive_int(sp_size, "sp_size")
    require(conditioner_tokens <= total_sequence, "conditioner_tokens cannot exceed total_sequence")
    remainder = total_sequence % sp_size
    require(conditioner_tokens >= remainder, f"cannot split sequence={total_sequence} over sp_size={sp_size} with conditioner_tokens={conditioner_tokens}")
    aux_tokens = conditioner_tokens - ((conditioner_tokens - remainder) % sp_size)
    main_sequence = total_sequence - aux_tokens
    require(main_sequence > 0 and main_sequence % sp_size == 0, "derived SP main sequence is invalid")
    return main_sequence, aux_tokens


def derive_replay_suites(
    manifest_path: Path,
    output_dir: Path,
    *,
    suite_id: str,
    conditioner_tokens: int,
    sp_sizes: list[int],
    keep_ratio: float,
) -> list[Path]:
    require(manifest_path.is_file(), f"SP replay manifest does not exist: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    require(manifest.get("schema_version") == 1 and manifest.get("kind") == "operator_benchmark_qkv_replay_v1", "unsupported SP replay manifest")
    require(manifest.get("layout") == "BSHD", "SP replay layout must be BSHD")
    tensors = manifest.get("tensors")
    require(isinstance(tensors, dict) and set(tensors) == {"q", "k", "v"}, "SP replay manifest requires exactly q/k/v tensors")
    q_shape = tensors["q"].get("shape")
    k_shape = tensors["k"].get("shape")
    v_shape = tensors["v"].get("shape")
    require(
        isinstance(q_shape, list) and len(q_shape) == 4 and isinstance(k_shape, list) and len(k_shape) == 4 and v_shape == k_shape and q_shape[:2] == k_shape[:2] and q_shape[3] == k_shape[3],
        "SP replay Q/K/V shapes are incompatible",
    )
    dtype = tensors["q"].get("dtype")
    require(dtype in INPUT_DTYPES and tensors["k"].get("dtype") == dtype and tensors["v"].get("dtype") == dtype, "SP replay Q/K/V dtypes are incompatible")
    require(isinstance(keep_ratio, (int, float)) and not isinstance(keep_ratio, bool) and 0 < float(keep_ratio) <= 1, "keep_ratio must be in (0, 1]")
    sizes = list(dict.fromkeys(_positive_int(value, "sp_size") for value in sp_sizes))
    require(sizes, "at least one sp_size is required")
    manifest_sha256 = _file_sha256(manifest_path)
    written = []
    for sp_size in sizes:
        main_sequence, aux_tokens = derive_replicated_prefix_split(q_shape[1], conditioner_tokens, sp_size)
        require(q_shape[2] % sp_size == 0 and k_shape[2] % sp_size == 0, f"attention heads are not divisible by sp_size={sp_size}")
        case_id = f"{suite_id}.sp{sp_size}"
        suite = {
            "schema_version": 1,
            "kind": SUITE_KIND,
            "suite_id": case_id,
            "cases": [
                {
                    "case_id": case_id,
                    "shape": {
                        "sequence": main_sequence,
                        "heads": q_shape[2],
                        "kv_heads": k_shape[2],
                        "head_dim": q_shape[3],
                        "sp_size": sp_size,
                        "aux_tokens": aux_tokens,
                        "aux_q": bool(aux_tokens),
                        "aux_first": bool(aux_tokens),
                        "causal": False,
                    },
                    "precision": {"input_dtype": dtype},
                    "replay": {"manifest": str(manifest_path.resolve()), "sha256": manifest_sha256},
                    "sparse": {"keep_ratio": float(keep_ratio)},
                    "source": {
                        "kind": "real_qkv_replay",
                        "split_policy": "largest_replicated_conditioner_prefix_without_padding",
                        "conditioner_tokens": conditioner_tokens,
                        "replay_provenance": manifest.get("provenance") or {},
                    },
                    "tags": ["real_qkv_replay", "sparse_sp", f"sp{sp_size}"],
                }
            ],
        }
        validate_suite(suite)
        output = output_dir / f"{suite_id}_sp{sp_size}.json"
        require(not output.exists(), f"derived SP suite already exists: {output}")
        write_json(output, suite)
        written.append(output)
    return written


def _candidate_id(candidate: dict[str, Any]) -> str:
    quant = candidate["quant_scheme"] or "none"
    fields = [candidate["algorithm"], f"attn={_candidate_backend(candidate)}", f"comm={quant}", f"fusion={int(candidate['tensor_fusion'])}"]
    if candidate["algorithm"] == "ulysses":
        fields.extend(
            (
                f"prepost={candidate['prepost_backend']}",
                f"a2a={candidate['a2a_backend']}",
                f"head={int(candidate['head_parallel'])}",
            )
        )
        if candidate["head_parallel"]:
            fields.append(f"head_group={candidate['head_parallel_group_size']}")
    return "__".join(fields)


def _candidate_backend(candidate: dict[str, Any]) -> str:
    return candidate.get("attention_backend") or candidate["dense_backend"]


def _head_parallel_options(case: dict[str, Any]) -> tuple[tuple[bool, int], ...]:
    shape = case["shape"]
    world_size = shape["sp_size"]
    local_heads = shape["heads"] // world_size if shape["heads"] % world_size == 0 else 0
    return ((False, 1), *((True, group_size) for group_size in range(1, local_heads + 1)))


def _packed_tensor_description(name: str, shape: list[int], quant_scheme: str | None) -> dict[str, Any]:
    payload_shape = list(shape)
    if quant_scheme == "fp4":
        payload_shape[-1] //= 2
    value = {
        "name": name,
        "logical_shape": list(shape),
        "payload_shape": payload_shape,
        "payload_dtype": {None: "input_dtype", "fp8": "fp8_e4m3fn", "fp4": "packed_fp4"}[quant_scheme],
        "scale_shape": None,
        "scale_dtype": None,
    }
    if quant_scheme == "fp8":
        value.update(scale_shape=[*shape[:-1], 1], scale_dtype="fp32")
    elif quant_scheme == "fp4":
        value.update(scale_shape=[*shape[:-1], shape[-1] // 16], scale_dtype="backend_fp4_scale")
    return value


def _sp_shape_description(case: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    shape = case["shape"]
    world_size = shape["sp_size"]
    local_main = shape["sequence"] // world_size
    aux_tokens = shape["aux_tokens"]
    q_heads = shape["heads"]
    kv_heads = shape["kv_heads"]
    head_dim = shape["head_dim"]
    q_shard_heads = q_heads // world_size
    kv_shard_heads = kv_heads // world_size
    quant = candidate["quant_scheme"]
    q_tokens = shape["sequence"] + (aux_tokens if shape["aux_q"] else 0)
    kv_tokens = shape["sequence"] + aux_tokens
    common = {
        "logical_attention": {
            "q_shape": [1, q_tokens, q_heads, head_dim],
            "k_shape": [1, kv_tokens, kv_heads, head_dim],
            "v_shape": [1, kv_tokens, kv_heads, head_dim],
            "input_dtype": case["precision"]["input_dtype"],
            "causal": shape["causal"],
        },
        "partition": {
            "sp_size": world_size,
            "global_main_tokens": shape["sequence"],
            "local_main_tokens": local_main,
            "replicated_aux_tokens": aux_tokens,
            "aux_first": shape["aux_first"],
        },
        "local_input": {
            "q_shape": [local_main, q_heads, head_dim],
            "k_shape": [local_main, kv_heads, head_dim],
            "v_shape": [local_main, kv_heads, head_dim],
            "aux_q_shape": [aux_tokens, q_heads, head_dim] if shape["aux_q"] else None,
            "aux_kv_shape": [aux_tokens, kv_heads, head_dim] if aux_tokens else None,
        },
    }

    if candidate["algorithm"] == "ring":
        if candidate["tensor_fusion"]:
            packed = [_packed_tensor_description("fused_kv", [2 * local_main, kv_heads, head_dim], quant)]
        else:
            packed = [
                _packed_tensor_description("k", [local_main, kv_heads, head_dim], quant),
                _packed_tensor_description("v", [local_main, kv_heads, head_dim], quant),
            ]
        common["per_rank_attention"] = {
            "q_shape": [local_main + (aux_tokens if shape["aux_q"] else 0), q_heads, head_dim],
            "kv_block_shape": [local_main, kv_heads, head_dim],
        }
        common["communication"] = {
            "algorithm": "ring",
            "kv_rotation_steps": world_size - 1,
            "kv_rotation": packed,
            "aux_qkv": "replicated; bypasses Ring rotation",
        }
        return common

    if candidate["head_parallel"]:
        group_size = candidate["head_parallel_group_size"]
        head_groups = [min(group_size, q_shard_heads - begin) for begin in range(0, q_shard_heads, group_size)]

        def qkv_payloads(heads: int) -> list[dict[str, Any]]:
            if candidate["tensor_fusion"]:
                return [_packed_tensor_description("fused_qkv", [world_size, local_main, 3, heads, head_dim], quant)]
            return [
                _packed_tensor_description("q", [world_size, local_main, heads, head_dim], quant),
                _packed_tensor_description("k", [world_size, local_main, heads, head_dim], quant),
                _packed_tensor_description("v", [world_size, local_main, heads, head_dim], quant),
            ]

        common["per_rank_attention"] = {
            "head_group_sizes": head_groups,
            "q_shapes_per_call": [[q_tokens, heads, head_dim] for heads in head_groups],
            "k_shapes_per_call": [[kv_tokens, heads, head_dim] for heads in head_groups],
            "v_shapes_per_call": [[kv_tokens, heads, head_dim] for heads in head_groups],
            "calls_per_attention": len(head_groups),
        }
        common["communication"] = {
            "algorithm": "ulysses",
            "head_group_sizes": head_groups,
            "qkv_all_to_all": {
                "calls_per_attention": len(head_groups),
                "packed_tensors_per_call": [qkv_payloads(heads) for heads in head_groups],
            },
            "output_all_to_all": {
                "calls_per_attention": len(head_groups),
                "packed_tensors_per_call": [[_packed_tensor_description("attention_output", [world_size, heads, local_main, head_dim], quant)] for heads in head_groups],
            },
            "auxiliary": {
                "qkv": "replicated; bypasses QKV all-to-all",
                "output_all_gather_input_shape": [aux_tokens, q_shard_heads, head_dim] if shape["aux_q"] else None,
            },
        }
        return common

    if candidate["tensor_fusion"]:
        qkv_packed = [_packed_tensor_description("fused_qkv", [world_size, local_main, 3, q_shard_heads, head_dim], quant)]
    else:
        qkv_packed = [
            _packed_tensor_description("q", [world_size, local_main, q_shard_heads, head_dim], quant),
            _packed_tensor_description("k", [world_size, local_main, kv_shard_heads, head_dim], quant),
            _packed_tensor_description("v", [world_size, local_main, kv_shard_heads, head_dim], quant),
        ]
    output_packed = [_packed_tensor_description("attention_output", [world_size, q_shard_heads, local_main, head_dim], quant)]
    common["per_rank_attention"] = {
        "q_shape": [q_tokens, q_shard_heads, head_dim],
        "k_shape": [kv_tokens, kv_shard_heads, head_dim],
        "v_shape": [kv_tokens, kv_shard_heads, head_dim],
        "calls_per_attention": 1,
    }
    common["communication"] = {
        "algorithm": "ulysses",
        "qkv_all_to_all": {"calls_per_attention": 1, "packed_tensors": qkv_packed},
        "output_all_to_all": {"calls_per_attention": 1, "packed_tensors": output_packed},
        "auxiliary": {
            "qkv": "replicated; bypasses QKV all-to-all",
            "output_all_gather_input_shape": [aux_tokens, q_shard_heads, head_dim] if shape["aux_q"] else None,
        },
    }
    return common


def _is_sparse_case(case: dict[str, Any]) -> bool:
    return case.get("replay") is not None


def _support_error(
    case: dict[str, Any],
    candidate: dict[str, Any],
    *,
    ring_lse_backends: set[str],
    fp4_available: bool,
    allow_unvalidated: bool = False,
) -> str | None:
    shape = case["shape"]
    algorithm = candidate["algorithm"]
    quant = candidate["quant_scheme"]
    backend = _candidate_backend(candidate)
    if quant == "fp4" and not fp4_available:
        return "FP4 communication dependency is unavailable"
    if quant == "fp4" and shape["head_dim"] % 16:
        return "FP4 communication requires head_dim divisible by 16"
    if algorithm == "ring":
        if candidate.get("leaf_family") == "sparse_attention":
            return "Sparse Ring requires a leaf backend with native per-block output and LSE; no current sparse candidate provides this contract"
        if backend not in ring_lse_backends:
            return "Ring requires a dense backend with native apply_with_lse()"
        if shape["heads"] != shape["kv_heads"]:
            return "Ring requires equal Q/K/V head counts"
        if backend not in SP_REFERENCE_TOLERANCES and not allow_unvalidated:
            return f"SP dense backend {backend} is not validated"
        if shape["aux_tokens"] and quant == "fp4" and not allow_unvalidated:
            return "SP auxiliary-token FP4 communication is not validated"
        return None

    world_size = shape["sp_size"]
    if shape["heads"] % world_size or shape["kv_heads"] % world_size:
        return "Ulysses requires Q and KV head counts divisible by sp_size"
    head_group_size = candidate.get("head_parallel_group_size", 1)
    if not isinstance(head_group_size, int) or isinstance(head_group_size, bool):
        return "Ulysses head_parallel_group_size must be an integer"
    if not candidate["head_parallel"] and head_group_size != 1:
        return "Ulysses head_parallel_group_size requires head_parallel"
    if candidate["head_parallel"] and not 1 <= head_group_size <= shape["heads"] // world_size:
        return "Ulysses head_parallel_group_size must be within the local head count"
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
    if candidate.get("leaf_family") != "sparse_attention" and backend not in SP_REFERENCE_TOLERANCES and not allow_unvalidated:
        return f"SP dense backend {backend} is not validated"
    if shape["aux_tokens"] and quant == "fp4" and not allow_unvalidated:
        return "SP auxiliary-token FP4 communication is not validated"
    return None


def enumerate_candidates(
    case: dict[str, Any],
    dense_backends: list[str],
    *,
    a2a_backends: list[str] | None = None,
    ring_lse_backends: set[str] | None = None,
    fp4_available: bool = True,
    allow_unvalidated: bool = False,
) -> dict[str, Any]:
    a2a_backends = a2a_backends or ["torch", "round_robin"]
    ring_lse_backends = ring_lse_backends or set()
    raw_candidates = []
    for dense_backend, prepost, a2a, quant, fusion, head_option in itertools.product(
        dense_backends,
        ("torch", "triton"),
        a2a_backends,
        QUANT_SCHEMES,
        (False, True),
        _head_parallel_options(case),
    ):
        head_parallel, head_parallel_group_size = head_option
        raw_candidates.append(
            {
                "algorithm": "ulysses",
                "dense_backend": dense_backend,
                "prepost_backend": prepost,
                "a2a_backend": a2a,
                "quant_scheme": quant,
                "tensor_fusion": fusion,
                "head_parallel": head_parallel,
                "head_parallel_group_size": head_parallel_group_size,
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
                "head_parallel_group_size": 1,
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
            allow_unvalidated=allow_unvalidated,
        )
        if reason:
            exclusions.append({"candidate_id": candidate["candidate_id"], "reason": reason})
        else:
            candidates.append(candidate)
    return {"case_id": case["case_id"], "candidates": candidates, "exclusions": exclusions}


def enumerate_sparse_candidates(
    case: dict[str, Any],
    sparse_backends: list[str],
    *,
    a2a_backends: list[str] | None = None,
    fp4_available: bool = True,
) -> dict[str, Any]:
    a2a_backends = a2a_backends or ["torch", "round_robin"]
    raw_candidates = []
    for backend, prepost, a2a, quant, fusion, head_option in itertools.product(
        sparse_backends,
        ("torch", "triton"),
        a2a_backends,
        QUANT_SCHEMES,
        (False, True),
        _head_parallel_options(case),
    ):
        head_parallel, head_parallel_group_size = head_option
        raw_candidates.append(
            {
                "algorithm": "ulysses",
                "attention_backend": backend,
                "leaf_family": "sparse_attention",
                "prepost_backend": prepost,
                "a2a_backend": a2a,
                "quant_scheme": quant,
                "tensor_fusion": fusion,
                "head_parallel": head_parallel,
                "head_parallel_group_size": head_parallel_group_size,
            }
        )

    candidates = []
    exclusions = []
    for candidate in raw_candidates:
        candidate = {"candidate_id": _candidate_id(candidate), **candidate}
        reason = _support_error(case, candidate, ring_lse_backends=set(), fp4_available=fp4_available)
        if reason:
            exclusions.append({"candidate_id": candidate["candidate_id"], "reason": reason})
        else:
            candidates.append(candidate)
    for backend in sparse_backends:
        exclusions.append(
            {
                "candidate_id": f"ring__attn={backend}",
                "reason": "Sparse Ring requires a leaf backend with native per-block output and LSE; no current sparse candidate provides this contract",
            }
        )
    return {"case_id": case["case_id"], "candidates": candidates, "exclusions": exclusions}


def _sparse_sp_backend_scope(backend_catalog: dict[str, Any]) -> tuple[list[str], dict[str, str], list[str]]:
    eligible = {item["name"] for item in backend_catalog.get("candidates", []) if item.get("family") == "sparse_attention" and item.get("status") == "eligible"}
    supported = sorted(eligible & set(SPARSE_SP_BACKENDS))
    exclusions = {name: reason for name, reason in SPARSE_SP_EXCLUSIONS.items() if name in eligible}
    unmapped = sorted(eligible - set(SPARSE_SP_BACKENDS) - set(exclusions))
    return supported, exclusions, unmapped


def build_candidate_catalog(
    suite: dict[str, Any],
    runtime: dict[str, Any],
    dense_backends: list[str] | None = None,
    *,
    sparse_backends: list[str] | None = None,
    allow_unvalidated: bool = False,
) -> dict[str, Any]:
    validate_suite(suite)
    sparse_mode = all(_is_sparse_case(case) for case in suite["cases"])
    require(sparse_mode or not any(_is_sparse_case(case) for case in suite["cases"]), "SP suite cannot mix replay-backed sparse and synthetic dense cases")
    leaf_family = "sparse_attention" if sparse_mode else "dense_attention"
    backend_catalog = runtime["catalog"]
    dependency_blockers = sorted(
        item["name"] for item in backend_catalog.get("candidates", []) if item.get("family") == leaf_family and item.get("status") in {"cuda_unavailable", "dependency_missing"}
    )
    unmapped_backends = sorted(backend_catalog.get("unmapped_attention_backends", []))
    sparse_sp_unmapped = sorted(runtime.get("unmapped_sparse_sp_backends", [])) if sparse_mode else []
    backend_scope_complete = not dependency_blockers and not unmapped_backends and not sparse_sp_unmapped
    backend_scope_blockers = {
        "dependency_missing": dependency_blockers,
        "unmapped_production_backends": unmapped_backends,
        "unmapped_sparse_sp_backends": sparse_sp_unmapped,
    }
    if sparse_mode:
        all_sparse_backends = list(runtime["sparse_backends"])
        sparse_backends = list(sparse_backends or all_sparse_backends)
        require(not (set(sparse_backends) - set(all_sparse_backends)), "requested sparse backend is not eligible on this device")
        cases = [
            enumerate_sparse_candidates(
                case,
                sparse_backends,
                a2a_backends=runtime["a2a_backends"],
                fp4_available=runtime["fp4_available"],
            )
            for case in suite["cases"]
        ]
        identity = {
            "backend_catalog_fingerprint": runtime["catalog"]["catalog_fingerprint"],
            "leaf_family": leaf_family,
            "backend_scope_complete": backend_scope_complete,
            "backend_scope_blockers": backend_scope_blockers,
            "sparse_backends": sparse_backends,
            "sparse_sp_exclusions": runtime.get("sparse_sp_exclusions", {}),
            "all_eligible_sparse_backends": all_sparse_backends,
            "a2a_backends": runtime["a2a_backends"],
            "fp4_available": runtime["fp4_available"],
            "cases": cases,
        }
        fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        return {
            "kind": "sp_attention_benchmark_candidate_catalog_v1",
            "catalog_fingerprint": fingerprint,
            "scope_complete": backend_scope_complete and set(sparse_backends) == set(all_sparse_backends),
            **identity,
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
            allow_unvalidated=allow_unvalidated,
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
        "allow_unvalidated": allow_unvalidated,
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
    dense_backends = sorted({item["production_backend"] for item in catalog["candidates"] if item["family"] == "dense_attention" and item["status"] == "eligible" and item["production_backend"]})
    sparse_backends, sparse_sp_exclusions, unmapped_sparse_sp_backends = _sparse_sp_backend_scope(catalog)
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
        "sparse_backends": sparse_backends,
        "sparse_sp_exclusions": sparse_sp_exclusions,
        "unmapped_sparse_sp_backends": unmapped_sparse_sp_backends,
        "ring_lse_backends": ring_lse_backends,
        "a2a_backends": list(dict.fromkeys(("torch", "round_robin", *sorted(A2A_BACKEND_REGISTER.keys())))),
        "fp4_available": seq_p.quant_fp4_sage3 is not None and seq_p.dequant_fp4_sage3 is not None,
    }


def _selected_candidates(
    suite: dict[str, Any],
    runtime: dict[str, Any],
    candidate_ids: set[str],
    algorithms: set[str],
    *,
    allow_unvalidated: bool = False,
) -> dict[str, list[dict[str, Any]]]:
    selections = {}
    known_ids = set()
    for case in suite["cases"]:
        if _is_sparse_case(case):
            result = enumerate_sparse_candidates(
                case,
                runtime["sparse_backends"],
                a2a_backends=runtime["a2a_backends"],
                fp4_available=runtime["fp4_available"],
            )
        else:
            result = enumerate_candidates(
                case,
                runtime["dense_backends"],
                a2a_backends=runtime["a2a_backends"],
                ring_lse_backends=runtime["ring_lse_backends"],
                fp4_available=runtime["fp4_available"],
                allow_unvalidated=allow_unvalidated,
            )
        known_ids.update(item["candidate_id"] for item in result["candidates"])
        selected = [item for item in result["candidates"] if item["algorithm"] in algorithms and (not candidate_ids or item["candidate_id"] in candidate_ids)]
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
    local_generator = torch.Generator(device=device).manual_seed(seed + rank)
    q = torch.randn((local_sequence, shape["heads"], shape["head_dim"]), device=device, dtype=dtype, generator=local_generator)
    k = torch.randn((local_sequence, shape["kv_heads"], shape["head_dim"]), device=device, dtype=dtype, generator=local_generator)
    v = torch.randn((local_sequence, shape["kv_heads"], shape["head_dim"]), device=device, dtype=dtype, generator=local_generator)
    aux_q = aux_k = aux_v = None
    if shape["aux_tokens"]:
        aux_generator = torch.Generator(device=device).manual_seed(seed + 1_000_003)
        aux_k = torch.randn((shape["aux_tokens"], shape["kv_heads"], shape["head_dim"]), device=device, dtype=dtype, generator=aux_generator)
        aux_v = torch.randn((shape["aux_tokens"], shape["kv_heads"], shape["head_dim"]), device=device, dtype=dtype, generator=aux_generator)
        if shape["aux_q"]:
            aux_q = torch.randn((shape["aux_tokens"], shape["heads"], shape["head_dim"]), device=device, dtype=dtype, generator=aux_generator)
    return q, k, v, aux_q, aux_k, aux_v


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_replay_range(manifest_path: Path, spec: dict[str, Any], start: int, end: int, device: Any, torch: Any) -> Any:
    require(0 <= start <= end <= spec["shape"][1], "SP replay range is outside the tensor sequence")
    chunks = []
    cursor = start
    for index, shard in enumerate(spec["shards"]):
        shard_start = shard.get("sequence_start")
        shard_end = shard.get("sequence_end")
        require(type(shard_start) is int and type(shard_end) is int and shard_end > shard_start, f"invalid SP replay shard range at {index}")
        if shard_end <= start or shard_start >= end:
            continue
        overlap_start = max(start, shard_start)
        overlap_end = min(end, shard_end)
        require(overlap_start == cursor, f"SP replay shard coverage drift at {index}")
        shard_path = Path(str(shard.get("path")))
        if not shard_path.is_absolute():
            shard_path = manifest_path.parent / shard_path
        require(shard_path.is_file() and _file_sha256(shard_path) == shard.get("sha256"), f"SP replay shard file mismatch at {index}: {shard_path}")
        payload = torch.load(shard_path, map_location="cpu", weights_only=True)
        tensor = payload.get(shard.get("tensor_key")) if isinstance(payload, dict) else None
        expected_shape = [spec["shape"][0], shard_end - shard_start, *spec["shape"][2:]]
        tensor_matches = isinstance(tensor, torch.Tensor) and list(tensor.shape) == expected_shape and normalize_dtype(getattr(tensor, "dtype", None), "") == spec.get("dtype")
        require(tensor_matches, f"SP replay shard tensor mismatch at {index}")
        chunks.append(tensor[:, overlap_start - shard_start : overlap_end - shard_start])
        cursor = overlap_end
    require(cursor == end, "SP replay shards do not cover the requested sequence range")
    if not chunks:
        return None
    return torch.cat(chunks, dim=1).squeeze(0).to(device=device).contiguous()


def _load_sp_replay_inputs(case: dict[str, Any], device: Any, rank: int) -> tuple[Any, Any, Any, Any, Any, Any]:
    import torch

    replay = case["replay"]
    manifest_path = Path(replay["manifest"]).resolve()
    require(manifest_path.is_file(), f"SP replay manifest does not exist: {manifest_path}")
    require(_file_sha256(manifest_path) == replay["sha256"], f"SP replay manifest sha256 mismatch: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    require(manifest.get("schema_version") == 1 and manifest.get("kind") == "operator_benchmark_qkv_replay_v1", f"unsupported SP replay manifest: {manifest_path}")
    require(manifest.get("layout") == "BSHD", "SP replay layout must be BSHD")
    specs = manifest.get("tensors")
    require(isinstance(specs, dict) and set(specs) == {"q", "k", "v"}, "SP replay manifest requires exactly q/k/v tensors")

    shape = case["shape"]
    total_sequence = shape["sequence"] + shape["aux_tokens"]
    expected = {
        "q": [1, total_sequence, shape["heads"], shape["head_dim"]],
        "k": [1, total_sequence, shape["kv_heads"], shape["head_dim"]],
        "v": [1, total_sequence, shape["kv_heads"], shape["head_dim"]],
    }
    dtype = case["precision"]["input_dtype"]
    for name, spec in specs.items():
        require(spec.get("shape") == expected[name] and spec.get("dtype") == dtype, f"SP replay {name} contract does not match the case")
        require(isinstance(spec.get("shards"), list) and spec["shards"], f"SP replay {name} requires shards")

    aux_end = shape["aux_tokens"]
    local_length = shape["sequence"] // shape["sp_size"]
    main_start = aux_end + rank * local_length
    main_end = main_start + local_length
    main = [_load_replay_range(manifest_path, specs[name], main_start, main_end, device, torch) for name in ("q", "k", "v")]
    aux = [_load_replay_range(manifest_path, specs[name], 0, aux_end, device, torch) if aux_end else None for name in ("q", "k", "v")]
    return (*main, *aux)


def _sparse_attention_operator(backend: str, keep_ratio: float) -> Any:
    spec = SPARSE_SP_BACKENDS.get(backend)
    require(spec is not None, f"unsupported sparse SP backend: {backend}")
    if spec["factory"] == "dynamic":
        from lightx2v.common.ops.attn.dynamic_sparse_attn import DynamicSparseAttnWeight

        return DynamicSparseAttnWeight({"sparsity_ratio": 1.0 - keep_ratio, "operator": spec["operator"]})
    if spec["factory"] == "sparge":
        from lightx2v.common.ops.attn.sparge_attn import SpargeAttnWeight

        operator = SpargeAttnWeight()
    else:
        module_name, separator, symbol_name = spec["symbol"].partition(":")
        require(bool(separator), f"invalid sparse SP backend symbol: {spec['symbol']}")
        operator_type = getattr(importlib.import_module(module_name), symbol_name)
        operator = operator_type()
    operator.topk = keep_ratio
    return operator


def _prepare_operation(
    case: dict[str, Any],
    candidate: dict[str, Any],
    device: Any,
    seed: int,
    rank: int,
    inputs: tuple[Any, Any, Any, Any, Any, Any] | None = None,
) -> tuple[Any, int, dict[str, Any] | None]:
    import lightx2v.common.ops.attn  # noqa: F401
    from lightx2v.common.ops.attn.ring_attn import RingAttnWeight
    from lightx2v.common.ops.attn.ulysses_attn import UlyssesAttnWeight
    from lightx2v.utils.registry_factory import ATTN_WEIGHT_REGISTER

    shape = case["shape"]
    q, k, v, aux_q, aux_k, aux_v = inputs or (_load_sp_replay_inputs(case, device, rank) if _is_sparse_case(case) else _make_inputs(case, device, seed, rank))

    operation_metrics = None
    if candidate.get("leaf_family") == "sparse_attention":
        dense_operator = _sparse_attention_operator(_candidate_backend(candidate), float(case["sparse"]["keep_ratio"]))
        keep_ratio = float(case["sparse"]["keep_ratio"])
        key_blocks = None
        selected_key_blocks = None
        block_k = getattr(dense_operator, "BLKK", None)
        if block_k is not None:
            key_blocks = (shape["sequence"] + shape["aux_tokens"] + block_k - 1) // block_k
            selected_key_blocks = max(1, min(key_blocks, int(keep_ratio * key_blocks)))
            block_density = selected_key_blocks / key_blocks
            density_source = "fixed_topk_block_contract"
        else:
            block_density = keep_ratio
            density_source = "backend_topk_contract"
        operation_metrics = {
            "keep_ratio": keep_ratio,
            "block_density_actual": block_density,
            "block_density_source": density_source,
            "q_block_size": getattr(dense_operator, "BLKQ", None),
            "k_block_size": block_k,
            "key_blocks": key_blocks,
            "selected_key_blocks_per_row": selected_key_blocks,
        }
    else:
        dense_type = ATTN_WEIGHT_REGISTER.get(_candidate_backend(candidate))
        require(dense_type is not None, f"dense production backend is not registered: {_candidate_backend(candidate)}")
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
            **({"head_parallel_group_size": candidate.get("head_parallel_group_size", 1)} if candidate["algorithm"] == "ulysses" else {}),
        )

    query_tokens = shape["sequence"] + (shape["aux_tokens"] if shape["aux_q"] else 0)
    kv_tokens = shape["sequence"] + shape["aux_tokens"]
    work = 4 * shape["heads"] * query_tokens * kv_tokens * shape["head_dim"]
    return fn, work, operation_metrics


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
        finite = torch.tensor(int(all(tensor.isfinite().all().item() for tensor in _output_tensors(result))), device=device, dtype=torch.int32)
        dist.all_reduce(finite, op=dist.ReduceOp.MIN)
        finite_value = bool(finite.item())
    return maximum.cpu().tolist(), [item.cpu().tolist() for item in gathered], result, int(memory.item()), finite_value


def _output_tensors(result: Any) -> tuple[Any, ...]:
    if isinstance(result, tuple):
        return tuple(item for item in result if item is not None)
    return (result,)


def _output_tensor(result: Any) -> Any:
    if isinstance(result, tuple):
        return result[0]
    return result


def _result_correctness_policy(case: dict[str, Any], candidate: dict[str, Any], *, allow_missing: bool = False) -> dict[str, Any]:
    if candidate.get("leaf_family") == "sparse_attention":
        return {"policy": "sparse_replay_replication_v1", "atol": 0.10, "rtol": 0.10}
    backend = _candidate_backend(candidate)
    if backend not in SP_REFERENCE_TOLERANCES:
        require(allow_missing, f"missing SP correctness contract for dense backend: {backend}")
        return {"policy": "exploratory_unregistered_backend_v1", "atol": 0.10, "rtol": 0.10}
    contract = SP_REFERENCE_TOLERANCES[backend]
    dtype = case["precision"]["input_dtype"]
    tolerance = 0.10 if candidate["quant_scheme"] is not None else contract[dtype]
    return {
        "policy": "quantized_communication_v1" if candidate["quant_scheme"] is not None else f"{contract['kind']}_dense_v1",
        "atol": tolerance,
        "rtol": tolerance,
    }


def _result_correctness_tolerance(case: dict[str, Any], candidate: dict[str, Any]) -> float:
    return _result_correctness_policy(case, candidate)["atol"]


def _result_correctness(case: dict[str, Any], candidate: dict[str, Any], result: Any, device: Any, finite: bool) -> dict[str, Any]:
    import torch
    import torch.distributed as dist

    shape = case["shape"]
    expected_main = (shape["sequence"] // shape["sp_size"], shape["heads"] * shape["head_dim"])
    expected_aux = (shape["aux_tokens"], shape["heads"] * shape["head_dim"]) if shape["aux_q"] else None
    main = result[0] if isinstance(result, tuple) and len(result) == 2 else None
    aux = result[1] if isinstance(result, tuple) and len(result) == 2 else None
    local_shape_ok = isinstance(main, torch.Tensor) and tuple(main.shape) == expected_main
    local_shape_ok = local_shape_ok and ((aux is None) if expected_aux is None else isinstance(aux, torch.Tensor) and tuple(aux.shape) == expected_aux)
    shape_ok = torch.tensor(int(local_shape_ok), device=device, dtype=torch.int32)
    dist.all_reduce(shape_ok, op=dist.ReduceOp.MIN)
    shape_passed = bool(shape_ok.item())

    replicated_aux_checked = expected_aux is not None and shape_passed
    replicated_aux_passed = None
    replicated_aux_max_abs_diff = None
    policy = _result_correctness_policy(case, candidate)
    atol = policy["atol"]
    rtol = policy["rtol"]
    if replicated_aux_checked:
        gathered = [torch.empty_like(aux) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered, aux)
        reference = gathered[0].float()
        replicated_aux_max_abs_diff = max(float((item.float() - reference).abs().max().item()) for item in gathered)
        local_consistent = all(torch.allclose(item.float(), reference, atol=atol, rtol=rtol) for item in gathered)
        consistent = torch.tensor(int(local_consistent), device=device, dtype=torch.int32)
        dist.all_reduce(consistent, op=dist.ReduceOp.MIN)
        replicated_aux_passed = bool(consistent.item())

    passed = finite and shape_passed and replicated_aux_passed is not False
    return {
        "checked": True,
        "passed": passed,
        "check": "main_aux_shape_finite_and_replicated_aux_all_ranks",
        "finite": finite,
        "shape_passed": shape_passed,
        "expected_main_output_shape": list(expected_main),
        "expected_aux_output_shape": list(expected_aux) if expected_aux is not None else None,
        "replicated_aux_checked": replicated_aux_checked,
        "replicated_aux_passed": replicated_aux_passed,
        "replicated_aux_max_abs_diff": replicated_aux_max_abs_diff,
        "replicated_aux_atol": atol if replicated_aux_checked else None,
        "replicated_aux_rtol": rtol if replicated_aux_checked else None,
        "tolerance_policy": policy,
    }


def _gather_sequence(tensor: Any) -> Any:
    import torch
    import torch.distributed as dist

    gathered = [torch.empty_like(tensor) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, tensor)
    return torch.cat(gathered, dim=0)


def _dense_reference_outputs(case: dict[str, Any], device: Any, seed: int, rank: int) -> tuple[Any, Any]:
    import torch
    import torch.nn.functional as F

    shape = case["shape"]
    q, k, v, aux_q, aux_k, aux_v = _make_inputs(case, device, seed, rank)
    global_q = _gather_sequence(q)
    global_k = _gather_sequence(k)
    global_v = _gather_sequence(v)
    full_q = global_q if aux_q is None else torch.cat((aux_q, global_q), dim=0) if shape["aux_first"] else torch.cat((global_q, aux_q), dim=0)
    full_k = torch.cat((aux_k, global_k), dim=0) if shape["aux_first"] and aux_k is not None else global_k
    full_v = torch.cat((aux_v, global_v), dim=0) if shape["aux_first"] and aux_v is not None else global_v
    if aux_k is not None and not shape["aux_first"]:
        full_k = torch.cat((global_k, aux_k), dim=0)
        full_v = torch.cat((global_v, aux_v), dim=0)
    output = (
        F.scaled_dot_product_attention(
            full_q.transpose(0, 1).unsqueeze(0),
            full_k.transpose(0, 1).unsqueeze(0),
            full_v.transpose(0, 1).unsqueeze(0),
            is_causal=shape["causal"],
            enable_gqa=shape["heads"] != shape["kv_heads"],
        )
        .squeeze(0)
        .transpose(0, 1)
    )
    aux_length = 0 if aux_q is None else aux_q.shape[0]
    if aux_length and shape["aux_first"]:
        global_main, aux_output = output[aux_length:], output[:aux_length]
    elif aux_length:
        global_main, aux_output = output[:-aux_length], output[-aux_length:]
    else:
        global_main, aux_output = output, None
    local_length = q.shape[0]
    local_main = global_main[rank * local_length : (rank + 1) * local_length]
    return local_main.reshape(local_length, -1), None if aux_output is None else aux_output.reshape(aux_length, -1)


def _reference_correctness(
    case: dict[str, Any],
    candidate: dict[str, Any],
    actual: Any,
    expected: tuple[Any, Any],
    device: Any,
    *,
    check: str = "production_sp_attention_vs_torch_sdpa_dense_reference_all_ranks",
) -> dict[str, Any]:
    import torch
    import torch.distributed as dist

    actual_main = actual[0] if isinstance(actual, tuple) and len(actual) == 2 else None
    actual_aux = actual[1] if isinstance(actual, tuple) and len(actual) == 2 else None
    expected_main, expected_aux = expected
    local_shape = isinstance(actual_main, torch.Tensor) and tuple(actual_main.shape) == tuple(expected_main.shape)
    local_shape = local_shape and ((actual_aux is None) if expected_aux is None else isinstance(actual_aux, torch.Tensor) and tuple(actual_aux.shape) == tuple(expected_aux.shape))
    local_finite = local_shape and all(tensor.isfinite().all().item() for tensor in _output_tensors(actual))
    main_max = float((actual_main.float() - expected_main.float()).abs().max().item()) if local_shape else float("inf")
    aux_max = float((actual_aux.float() - expected_aux.float()).abs().max().item()) if expected_aux is not None and local_shape else None
    policy = _result_correctness_policy(case, candidate, allow_missing=True)
    local_close = local_shape and torch.allclose(actual_main.float(), expected_main.float(), atol=policy["atol"], rtol=policy["rtol"])
    if expected_aux is not None and local_shape:
        local_close = local_close and torch.allclose(actual_aux.float(), expected_aux.float(), atol=policy["atol"], rtol=policy["rtol"])

    flags = torch.tensor((int(local_shape), int(local_finite), int(local_close)), device=device, dtype=torch.int32)
    dist.all_reduce(flags, op=dist.ReduceOp.MIN)
    maxima = torch.tensor((main_max, 0.0 if aux_max is None else aux_max), device=device, dtype=torch.float64)
    dist.all_reduce(maxima, op=dist.ReduceOp.MAX)
    return {
        "checked": True,
        "passed": bool(flags[0].item() and flags[1].item() and flags[2].item()),
        "check": check,
        "shape_passed": bool(flags[0].item()),
        "finite": bool(flags[1].item()),
        "allclose_passed": bool(flags[2].item()),
        "main_max_abs_diff": float(maxima[0].item()),
        "aux_max_abs_diff": None if expected_aux is None else float(maxima[1].item()),
        "tolerance_policy": policy,
    }


def _bulk_reference_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
    reference = {**candidate, "head_parallel": False, "head_parallel_group_size": 1}
    reference["candidate_id"] = _candidate_id(reference)
    return reference


def _reference_validation_candidates(case: dict[str, Any], candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not _is_sparse_case(case):
        return candidates
    grouped = [candidate for candidate in candidates if candidate["algorithm"] == "ulysses" and candidate["head_parallel"]]
    require(grouped, "sparse reference validation requires a grouped Ulysses candidate")
    return grouped


def validate_reference_suite(
    suite: dict[str, Any],
    output: Path,
    *,
    candidate_ids: set[str],
    algorithms: set[str],
    dense_backends: list[str],
    sparse_backends: list[str],
    seed: int,
) -> dict[str, Any]:
    import torch
    import torch.distributed as dist

    validate_suite(suite)
    require("RANK" in os.environ and "LOCAL_RANK" in os.environ, "SP reference validation must be launched with torchrun")
    require(not output.exists(), f"SP reference validation output already exists: {output}")
    dist.init_process_group(backend="nccl")
    try:
        rank = dist.get_rank()
        local_rank = int(os.environ["LOCAL_RANK"])
        device = torch.device("cuda", local_rank)
        torch.cuda.set_device(device)
        world_size = dist.get_world_size()
        require(all(case["shape"]["sp_size"] == world_size for case in suite["cases"]), "every case sp_size must match torchrun world size")
        runtime = _discover_runtime(str(device))
        selected_runtime = dict(runtime)
        if dense_backends:
            unavailable = sorted(set(dense_backends) - set(runtime["dense_backends"]))
            require(not unavailable, f"dense backends are not eligible on this device: {unavailable}")
            selected_runtime["dense_backends"] = dense_backends
            selected_runtime["ring_lse_backends"] = runtime["ring_lse_backends"] & set(dense_backends)
        if sparse_backends:
            unavailable = sorted(set(sparse_backends) - set(runtime["sparse_backends"]))
            require(not unavailable, f"sparse backends are not eligible on this device: {unavailable}")
            selected_runtime["sparse_backends"] = sparse_backends
        catalog = build_candidate_catalog(suite, selected_runtime, allow_unvalidated=True)
        selections = _selected_candidates(
            suite,
            selected_runtime,
            candidate_ids,
            algorithms,
            allow_unvalidated=True,
        )
        selections = {case["case_id"]: _reference_validation_candidates(case, selections[case["case_id"]]) for case in suite["cases"]}
        environments = _distributed_environment(device, rank, local_rank)
        records = []
        for case in suite["cases"]:
            try:
                inputs = _load_sp_replay_inputs(case, device, rank) if _is_sparse_case(case) else None
                expected = None if inputs is not None else _dense_reference_outputs(case, device, seed, rank)
                reference_error = None
            except Exception as exc:
                reference_error = _local_error(exc)
            reference_errors = _gather_rank_errors(reference_error)
            for candidate in selections[case["case_id"]]:
                reference_candidate = None
                if any(error is not None for error in reference_errors):
                    errors = reference_errors
                    correctness = None
                else:
                    correctness = None
                    try:
                        if inputs is not None:
                            reference_candidate = _bulk_reference_candidate(candidate)
                            reference_fn, _, _ = _prepare_operation(case, reference_candidate, device, seed, rank, inputs=inputs)
                            expected = reference_fn()
                        fn, _, _ = _prepare_operation(case, candidate, device, seed, rank, inputs=inputs)
                        actual = fn()
                        torch.cuda.synchronize(device)
                        correctness = _reference_correctness(
                            case,
                            candidate,
                            actual,
                            expected,
                            device,
                            check=(
                                "production_grouped_sparse_sp_attention_vs_same_leaf_bulk_reference_all_ranks"
                                if inputs is not None
                                else "production_sp_attention_vs_torch_sdpa_dense_reference_all_ranks"
                            ),
                        )
                        local_error = None
                    except Exception as exc:
                        local_error = _local_error(exc)
                    errors = _gather_rank_errors(local_error)
                if rank == 0:
                    records.append(
                        {
                            "case_id": case["case_id"],
                            "candidate": candidate,
                            "reference_candidate": reference_candidate,
                            "status": "error" if any(error is not None for error in errors) else "ok",
                            "correctness": correctness,
                            "errors": errors if any(error is not None for error in errors) else None,
                        }
                    )
        dist.barrier()
        if rank == 0:
            value = {
                "kind": "sp_attention_reference_validation_v1",
                "suite_id": suite["suite_id"],
                "catalog_fingerprint": catalog["catalog_fingerprint"],
                "seed": seed,
                "environment": {"ranks": environments},
                "summary": {
                    "record_count": len(records),
                    "passed_count": sum(item["status"] == "ok" and item["correctness"]["passed"] for item in records),
                    "failed_count": sum(item["status"] != "ok" or not item["correctness"]["passed"] for item in records),
                },
                "records": records,
            }
            output.parent.mkdir(parents=True, exist_ok=True)
            write_json(output, value)
            print(json.dumps({"output": str(output), **value["summary"]}, ensure_ascii=False, sort_keys=True))
            return value
        return {"rank": rank, "record_count": 0}
    finally:
        dist.destroy_process_group()


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


def _communication_component(
    logical_bytes: int,
    network_bytes: int,
    measurement: dict[str, Any],
    world_size: int,
    peak_profile_pattern: str,
) -> dict[str, Any]:
    latency_ms = float(measurement["latency_ms_mean"])
    return {
        **_communication_rates(logical_bytes, network_bytes, latency_ms, world_size),
        "latency_ms_mean": latency_ms,
        "peak_profile_pattern": peak_profile_pattern,
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
    inputs: tuple[Any, Any, Any, Any, Any, Any] | None = None,
) -> dict[str, Any]:
    import torch.distributed as dist

    from lightx2v.common.ops.attn.ulysses_a2a import create_ulysses_a2a_backend
    from lightx2v.common.ops.attn.ulysses_attn import UlyssesAttnWeight
    from lightx2v.common.ops.attn.ulysses_prepost import create_ulysses_prepost_backend
    from lightx2v.common.ops.attn.utils.seq_p import split_main_aux_output
    from lightx2v.utils.registry_factory import ATTN_WEIGHT_REGISTER

    shape = case["shape"]
    q, k, v, aux_q, aux_k, aux_v = inputs or _make_inputs(case, device, seed, rank)
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
        return prepost.unpack_qkv(exchanged_qkv, q, k, v, aux_q, aux_k, aux_v, rank, world_size, shape["aux_first"])

    attn_q, attn_k, attn_v = unpack_qkv()
    fake_attention_output = attn_q.reshape(attn_q.shape[0], -1)
    fake_output, fake_aux_output = split_main_aux_output(
        fake_attention_output,
        shape["sequence"],
        shape["aux_tokens"] if shape["aux_q"] else 0,
        shape["aux_first"],
    )

    def pack_output() -> Any:
        return prepost.pack_attn(fake_output, local_len, world_size, shard_heads, hidden_dims, quant)

    packed_output = pack_output()

    def exchange_output() -> Any:
        return UlyssesAttnWeight._exchange_packed(packed_output, a2a, None)

    exchanged_output = exchange_output()

    def unpack_output() -> Any:
        return prepost.unpack_attn(exchanged_output, q.dtype, hidden_dims)

    def gather_aux_output() -> Any:
        return UlyssesAttnWeight._gather_aux(fake_aux_output, world_size, None)

    def total_path() -> Any:
        current_qkv = prepost.pack_qkv(q, k, v, world_size, quant, fusion)
        current_qkv = UlyssesAttnWeight._exchange_packed(current_qkv, a2a, None)
        current_q, _, _ = prepost.unpack_qkv(current_qkv, q, k, v, aux_q, aux_k, aux_v, rank, world_size, shape["aux_first"])
        current_attention_output = current_q.reshape(current_q.shape[0], -1)
        current_output, current_aux_output = split_main_aux_output(
            current_attention_output,
            shape["sequence"],
            shape["aux_tokens"] if shape["aux_q"] else 0,
            shape["aux_first"],
        )
        current_output = prepost.pack_attn(current_output, local_len, world_size, shard_heads, hidden_dims, quant)
        current_output = UlyssesAttnWeight._exchange_packed(current_output, a2a, None)
        current_output = prepost.unpack_attn(current_output, q.dtype, hidden_dims)
        current_aux_output = UlyssesAttnWeight._gather_aux(current_aux_output, world_size, None)
        return current_output, current_aux_output

    def communication_path() -> Any:
        first = UlyssesAttnWeight._exchange_packed(packed_qkv, a2a, None)
        second = UlyssesAttnWeight._exchange_packed(packed_output, a2a, None)
        gathered_aux = UlyssesAttnWeight._gather_aux(fake_aux_output, world_size, None)
        return first, second, gathered_aux

    segment_fns = {
        "pack_qkv": pack_qkv,
        "exchange_qkv": exchange_qkv,
        "unpack_qkv": unpack_qkv,
        "pack_output": pack_output,
        "exchange_output": exchange_output,
        "unpack_output": unpack_output,
    }
    if fake_aux_output is not None:
        segment_fns["gather_aux_output"] = gather_aux_output
    segments = {name: {**_diagnostic_measure(fn, warmup, iterations, device), "calls_per_attention": 1} for name, fn in segment_fns.items()}
    total = _diagnostic_measure(total_path, warmup, iterations, device)
    communication = _diagnostic_measure(communication_path, warmup, iterations, device)
    if candidate.get("leaf_family") == "sparse_attention":
        dense_operator = _sparse_attention_operator(_candidate_backend(candidate), float(case["sparse"]["keep_ratio"]))
    else:
        dense_operator = ATTN_WEIGHT_REGISTER[_candidate_backend(candidate)]()
    dense_kwargs = UlyssesAttnWeight._dense_attention_kwargs({"causal": shape["causal"]}, attn_q, attn_k)

    def compute_only() -> Any:
        return dense_operator.apply(q=attn_q, k=attn_k, v=attn_v, **dense_kwargs)

    compute = _diagnostic_measure(compute_only, warmup, iterations, device)
    full_fn, _, _ = _prepare_operation(case, candidate, device, seed, rank, inputs=(q, k, v, aux_q, aux_k, aux_v))
    full = _diagnostic_measure(full_fn, warmup, iterations, device)
    qkv_bytes = _tensor_bytes(packed_qkv)
    output_bytes = _tensor_bytes(packed_output)
    aux_output_bytes = _tensor_bytes(fake_aux_output)
    components = {
        "main_qkv_all_to_all": _communication_component(
            qkv_bytes,
            qkv_bytes * (world_size - 1) // world_size,
            segments["exchange_qkv"],
            world_size,
            "all_to_all",
        ),
        "main_output_all_to_all": _communication_component(
            output_bytes,
            output_bytes * (world_size - 1) // world_size,
            segments["exchange_output"],
            world_size,
            "all_to_all",
        ),
    }
    if fake_aux_output is not None:
        components["aux_output_all_gather"] = _communication_component(
            aux_output_bytes,
            aux_output_bytes * (world_size - 1),
            segments["gather_aux_output"],
            world_size,
            "all_gather",
        )
    logical_bytes = sum(item["logical_payload_bytes_per_rank"] for item in components.values())
    network_bytes = sum(item["network_bytes_per_rank"] for item in components.values())
    collective_calls = sum(1 + int(scale is not None) for payload, scale in (*packed_qkv, *packed_output)) + int(fake_aux_output is not None)
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
            "communication_pattern": "all_to_all_with_aux_all_gather" if fake_aux_output is not None else "all_to_all",
            "peak_profile_pattern": "all_to_all",
            "primitive": candidate["a2a_backend"],
            "collective_calls_per_attention": collective_calls,
            "communication_components": components,
            "aux_qkv_input": "replicated_bypass_qkv_all_to_all" if shape["aux_tokens"] else None,
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
    inputs: tuple[Any, Any, Any, Any, Any, Any] | None = None,
) -> dict[str, Any]:
    import torch
    import torch.distributed as dist

    from lightx2v.common.ops.attn.ring_attn import RingAttnWeight, _merge_attention_blocks
    from lightx2v.common.ops.attn.utils.ring_comm import RingComm
    from lightx2v.common.ops.attn.utils.seq_p import pack_seq_p_tensor, split_main_aux_output, unpack_seq_p_tensor
    from lightx2v.utils.registry_factory import ATTN_WEIGHT_REGISTER

    shape = case["shape"]
    q, k, v, aux_q, aux_k, aux_v = inputs or _make_inputs(case, device, seed, rank)
    world_size = dist.get_world_size()
    hidden_dims = q.shape[-1]
    quant = candidate["quant_scheme"]
    fusion = candidate["tensor_fusion"]

    def pack_kv() -> Any:
        if fusion:
            return (pack_seq_p_tensor(torch.cat((k, v), dim=0), quant),)
        return pack_seq_p_tensor(k, quant), pack_seq_p_tensor(v, quant)

    packed_kv = pack_kv()

    def prepare_query() -> Any:
        if aux_q is None:
            return q
        return torch.cat((aux_q, q), dim=0) if shape["aux_first"] else torch.cat((q, aux_q), dim=0)

    attention_q = prepare_query()

    def append_aux_kv(block_k: Any, block_v: Any) -> tuple[Any, Any]:
        if aux_k is None:
            return block_k, block_v
        return torch.cat((block_k, aux_k), dim=0), torch.cat((block_v, aux_v), dim=0)

    def split_fake_output() -> Any:
        fake = attention_q.reshape(attention_q.shape[0], -1)
        return split_main_aux_output(fake, q.shape[0], shape["aux_tokens"] if shape["aux_q"] else 0, shape["aux_first"])

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
        current_q = prepare_query()
        current = pack_kv()
        comm = RingComm(None)
        current_k = None
        for step in range(world_size):
            current_k, current_v = unpack_kv(current)
            if step + 1 == world_size:
                current_k, current_v = append_aux_kv(current_k, current_v)
            if step + 1 < world_size:
                current = rotate_step(comm, current)
        fake = current_q.reshape(current_q.shape[0], -1)
        main, aux = split_main_aux_output(fake, q.shape[0], shape["aux_tokens"] if shape["aux_q"] else 0, shape["aux_first"])
        return current_k, current_v, main, aux

    segments = {
        "pack_kv": {**_diagnostic_measure(pack_kv, warmup, iterations, device), "calls_per_attention": 1},
        "unpack_kv_block": {**_diagnostic_measure(unpack_kv, warmup, iterations, device), "calls_per_attention": world_size},
        "ring_rotate": {**_diagnostic_measure(rotate_path, warmup, iterations, device), "calls_per_attention": 1},
    }
    if aux_q is not None:
        segments["prepare_aux_query"] = {**_diagnostic_measure(prepare_query, warmup, iterations, device), "calls_per_attention": 1}
        segments["split_aux_output"] = {**_diagnostic_measure(split_fake_output, warmup, iterations, device), "calls_per_attention": 1}
    if aux_k is not None:
        segments["append_aux_kv"] = {**_diagnostic_measure(lambda: append_aux_kv(k, v), warmup, iterations, device), "calls_per_attention": 1}
    total = _diagnostic_measure(total_path, warmup, iterations, device)
    communication = _diagnostic_measure(rotate_path, warmup, iterations, device)
    gathered_k = [torch.empty_like(k) for _ in range(world_size)]
    gathered_v = [torch.empty_like(v) for _ in range(world_size)]
    dist.all_gather(gathered_k, k)
    dist.all_gather(gathered_v, v)
    dense_operator = ATTN_WEIGHT_REGISTER[_candidate_backend(candidate)]()
    compute_blocks = list(zip(gathered_k, gathered_v))
    if aux_k is not None:
        compute_blocks[-1] = append_aux_kv(*compute_blocks[-1])

    def compute_only() -> Any:
        output = lse = None
        for block_k, block_v in compute_blocks:
            block_output, block_lse = RingAttnWeight._apply_attention_block(dense_operator, attention_q, block_k, block_v, {})
            output, lse = _merge_attention_blocks(output, lse, block_output, block_lse)
        return output

    compute = _diagnostic_measure(compute_only, warmup, iterations, device)
    full_fn, _, _ = _prepare_operation(case, candidate, device, seed, rank, inputs=(q, k, v, aux_q, aux_k, aux_v))
    full = _diagnostic_measure(full_fn, warmup, iterations, device)
    logical_bytes = _tensor_bytes(packed_kv) * (world_size - 1)
    components = {
        "main_kv_ring_rotation": _communication_component(
            logical_bytes,
            logical_bytes,
            communication,
            world_size,
            "ring_p2p",
        )
    }
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
            "peak_profile_pattern": "ring_p2p",
            "primitive": "ring_p2p",
            "p2p_steps_per_attention": world_size - 1,
            "communication_components": components,
            "aux_qkv_input": "replicated_bypass_ring_rotation" if shape["aux_tokens"] else None,
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
    inputs: tuple[Any, Any, Any, Any, Any, Any] | None = None,
) -> dict[str, Any]:
    require(not candidate["head_parallel"], "L2/L3 diagnostics do not support head_parallel; use L1 to measure the production pipeline")
    if candidate["algorithm"] == "ulysses":
        return _ulysses_diagnostics(case, candidate, device, seed, rank, warmup, iterations, inputs=inputs)
    return _ring_diagnostics(case, candidate, device, seed, rank, warmup, iterations, inputs=inputs)


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
    correctness: dict[str, Any],
    operation_metrics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    output = _output_tensor(result)
    aux_output = result[1] if isinstance(result, tuple) and len(result) == 2 else None
    mean_ms = statistics.mean(latencies)
    per_rank_means = [statistics.mean(values) for values in per_rank_latencies]
    metrics = {
        "latency_ms_mean": mean_ms,
        "latency_ms_median": statistics.median(latencies),
        "per_rank_latency_ms_mean": per_rank_means,
        "aggregate_effective_tflops": work / (mean_ms / 1000) / 1e12,
        "global_dense_work": work,
        "work_definition": "4*heads*(sequence+aux_q)*(sequence+aux_kv)*head_dim",
        "max_memory_allocated_bytes": peak_memory,
        "measurement_scope": "production_sp_attention_max_rank_cuda_event_list_single_sync",
        "output_shape": list(output.shape),
        "aux_output_shape": list(aux_output.shape) if aux_output is not None else None,
    }
    if operation_metrics is not None:
        density = float(operation_metrics["block_density_actual"])
        metrics.update(
            {
                "actual_tflops": work * density / (mean_ms / 1000) / 1e12,
                "dense_equivalent_tflops": work / (mean_ms / 1000) / 1e12,
                "global_actual_sparse_work": work * density,
                **operation_metrics,
            }
        )
    return {
        "kind": "sp_attention_benchmark_raw_v1",
        "run_id": f"repeat-{repeat:03d}",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "case": case,
        "candidate": candidate,
        "catalog_fingerprint": catalog_fingerprint,
        "environment": {"ranks": environments},
        "status": "ok",
        "metrics": metrics,
        "correctness": correctness,
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
    catalog_candidates = {(case["case_id"], candidate["candidate_id"]): candidate for case in (candidate_catalog or {}).get("cases", []) for candidate in case["candidates"]}
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
    candidates = {(case["case_id"], candidate["candidate_id"]): candidate for case in candidate_catalog["cases"] for candidate in case["candidates"]}
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
        expected = {candidate["candidate_id"]: candidate for candidate in catalog_cases.get(case["case_id"], {}).get("candidates", [])}
        if not expected:
            expected = {candidate_id: items[0]["candidate"] for (case_id, candidate_id), items in grouped.items() if case_id == case["case_id"]}
        candidates = {}
        for candidate_id, candidate in sorted(expected.items()):
            items = grouped.get((case["case_id"], candidate_id), [])
            errors = [item for item in items if item.get("status") != "ok"]
            correctness_failed = any((item.get("correctness") or {}).get("passed") is False for item in items)
            accepted_items = [item for item in items if item.get("status") == "ok" and (item.get("correctness") or {}).get("passed") is not False]
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
                "actual_tflops": statistics.median(item["metrics"]["actual_tflops"] for item in accepted_items if item["metrics"].get("actual_tflops") is not None)
                if any(item["metrics"].get("actual_tflops") is not None for item in accepted_items)
                else None,
                "dense_equivalent_tflops": statistics.median(item["metrics"]["dense_equivalent_tflops"] for item in accepted_items if item["metrics"].get("dense_equivalent_tflops") is not None)
                if any(item["metrics"].get("dense_equivalent_tflops") is not None for item in accepted_items)
                else None,
                "candidate": candidate,
                "errors": [item.get("error") for item in errors],
            }
        ranking = sorted((name for name, item in candidates.items() if item["status"] == "accepted"), key=lambda name: candidates[name]["latency_ms"])
        terminal_statuses = {"accepted", "unavailable", "correctness_failed"}
        coverage_complete = (
            candidate_catalog is not None and candidate_catalog.get("scope_complete") is True and bool(candidates) and all(item["status"] in terminal_statuses for item in candidates.values())
        )
        measured_winner = ranking[0] if ranking else None
        winner = measured_winner if coverage_complete else None
        observed = case.get("observed_candidate")
        observed_gap = None
        if measured_winner and observed in candidates and candidates[observed]["status"] == "accepted":
            observed_gap = candidates[observed]["latency_ms"] / candidates[measured_winner]["latency_ms"]
        recommendation = None
        if winner is not None:
            winner_result = candidates[winner]
            recommendation = {
                "candidate_id": winner,
                "configuration": winner_result["candidate"],
                "workload": case,
                "latency_ms": winner_result["latency_ms"],
                "spread_pct": winner_result["spread_pct"],
                "actual_tflops": winner_result["actual_tflops"],
                "dense_equivalent_tflops": winner_result["dense_equivalent_tflops"],
                **_sp_shape_description(case, winner_result["candidate"]),
            }
        workloads.append(
            {
                "case_id": case["case_id"],
                "winner": winner,
                "recommendation": recommendation,
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
    sparse_backends: list[str],
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
        if sparse_backends:
            unavailable = sorted(set(sparse_backends) - set(runtime["sparse_backends"]))
            require(not unavailable, f"sparse backends are not eligible on this device: {unavailable}")
            selection_runtime["sparse_backends"] = sparse_backends
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
        input_cache = {}
        for repeat in range(repeat_runs):
            for case in suite["cases"]:
                candidates = selections[case["case_id"]]
                candidates = candidates[repeat % len(candidates) :] + candidates[: repeat % len(candidates)]
                for candidate in candidates:
                    key = (f"repeat-{repeat:03d}", case["case_id"], candidate["candidate_id"])
                    if key in existing:
                        continue
                    try:
                        if _is_sparse_case(case) and case["case_id"] not in input_cache:
                            input_cache[case["case_id"]] = _load_sp_replay_inputs(case, device, rank)
                        fn, work, operation_metrics = _prepare_operation(
                            case,
                            candidate,
                            device,
                            seed + repeat * 1000,
                            rank,
                            inputs=input_cache.get(case["case_id"]),
                        )
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
                            correctness = _result_correctness(case, candidate, result, device, finite)
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
                                correctness,
                                operation_metrics,
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
        if any(item.get("gpu", {}).get(field) is None for item in ranks for field in ("name", "major", "minor", "pci_domain_id", "pci_bus_id", "pci_device_id")):
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
        require(pattern in {"all_to_all", "all_gather", "ring_p2p"}, f"unsupported communication pattern: {pattern}")
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
    validate_suite(suite)
    _validate_report_record_identities(
        suite,
        records,
        expected_kind="sp_attention_diagnostic_raw_v1",
    )
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
            peak_profile_pattern = first.get("peak_profile_pattern", communication_pattern)
            peak_entry = (platform_value or {}).get("interconnect_peaks", {}).get(peak_profile_pattern)
            peak_status = "available" if peak_entry else ("profile_not_provided" if platform_value is None else "peak_missing")
            nominal_peak = float(peak_entry["bus_bandwidth"]) if peak_entry else None
            bus_bandwidth = statistics.median(item["metrics"]["layer3"]["bus_bandwidth_gbps"] for item in ok)
            component_names = sorted({name for item in ok for name in item["metrics"]["layer3"].get("communication_components", {})})
            communication_components = {}
            for name in component_names:
                component_values = [item["metrics"]["layer3"].get("communication_components", {}).get(name) for item in ok]
                component_values = [value for value in component_values if value is not None]
                first_component = component_values[0]
                component_pattern = first_component.get("peak_profile_pattern")
                component_peak = (platform_value or {}).get("interconnect_peaks", {}).get(component_pattern)
                component_peak_status = "available" if component_peak else ("profile_not_provided" if platform_value is None else "peak_missing")
                component_latencies = [float(value["latency_ms_mean"]) for value in component_values if value.get("latency_ms_mean") is not None]
                component_spread_ms = max(component_latencies) - min(component_latencies) if len(component_latencies) > 1 else None
                component_median = statistics.median(component_latencies) if component_latencies else None
                component_spread_pct = component_spread_ms / component_median * 100 if component_spread_ms is not None and component_median else None
                component_unstable = component_spread_pct is not None and component_spread_pct > max_spread_pct and component_spread_ms > max_spread_ms
                if len(component_latencies) != len(ok):
                    component_status = "not_measured"
                elif errors:
                    component_status = "measurement_error"
                elif len(component_latencies) < required_runs:
                    component_status = "insufficient_runs"
                elif component_unstable:
                    component_status = "unstable"
                else:
                    component_status = "accepted"
                component_bus_bandwidth = statistics.median(float(value["bus_bandwidth_gbps"]) for value in component_values) if component_latencies else None
                component_nominal_peak = float(component_peak["bus_bandwidth"]) if component_peak else None
                observed_component_efficiency = component_bus_bandwidth / component_nominal_peak if component_bus_bandwidth is not None and component_nominal_peak else None
                communication_components[name] = {
                    "logical_payload_bytes_per_rank": first_component["logical_payload_bytes_per_rank"],
                    "network_bytes_per_rank": first_component["network_bytes_per_rank"],
                    "aggregate_network_bytes": first_component.get("aggregate_network_bytes"),
                    "latency_ms": component_median,
                    "spread_pct": component_spread_pct,
                    "spread_ms": component_spread_ms,
                    "algorithmic_bandwidth_gbps": (statistics.median(float(value["algorithmic_bandwidth_gbps"]) for value in component_values) if component_latencies else None),
                    "bus_bandwidth_gbps": component_bus_bandwidth,
                    "peak_profile_pattern": component_pattern,
                    "peak_status": component_peak_status,
                    "nominal_bus_peak_gbps": component_nominal_peak,
                    "peak_kind": component_peak.get("kind") if component_peak else None,
                    "peak_source": component_peak.get("source") if component_peak else None,
                    "observed_bus_peak_efficiency": observed_component_efficiency,
                    "bus_peak_efficiency": (observed_component_efficiency if component_status == "accepted" else None),
                    "efficiency_status": (
                        "available" if component_status == "accepted" and component_nominal_peak else ("measurement_not_accepted" if component_status != "accepted" else component_peak_status)
                    ),
                    "status": component_status,
                }
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
                "peak_profile_pattern": peak_profile_pattern,
                "primitive": first["primitive"],
                "communication_components": communication_components,
                "aux_qkv_input": first.get("aux_qkv_input"),
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
                "candidate": items[0]["candidate"],
                "workload": items[0]["case"],
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


def _format_bytes(value: int | float | None) -> str:
    if value is None:
        return "-"
    return f"{float(value) / 2**20:.2f} MiB"


def render_diagnostic_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# SP Attention L2/L3 诊断报告",
        "",
        f"- Suite：`{report['suite_id']}`",
        f"- 接受结果：`{report['summary']['accepted_count']}/{report['summary']['result_count']}`",
        f"- 硬件 profile：`{report['hardware'].get('platform_id') or '未提供'}`",
        "",
    ]
    for result in report["results"]:
        shape = result["workload"]["shape"]
        precision = result["workload"]["precision"]
        candidate = result["candidate"]
        lines.extend(
            (
                f"## {result['case_id']}",
                "",
                f"- Candidate：`{result['candidate_id']}`",
                f"- 状态：`{result['status']}`，repeats=`{result['runs']}`",
                f"- 配置：algorithm=`{candidate['algorithm']}`，attention=`{_candidate_backend(candidate)}`，communication=`{candidate['quant_scheme'] or 'none'}`，fusion=`{candidate['tensor_fusion']}`，pre/post=`{candidate['prepost_backend']}`，A2A=`{candidate['a2a_backend']}`，head pipeline=`{candidate['head_parallel']}`",
                f"- 输入精度：`{precision['input_dtype']}`",
                f"- Attention：main sequence=`{shape['sequence']}`，aux=`{shape['aux_tokens']}`，heads=`{shape['heads']}`，kv_heads=`{shape['kv_heads']}`，head_dim=`{shape['head_dim']}`，SP=`{shape['sp_size']}`",
                f"- L2 layout + communication：`{result['layer2_latency_ms']:.4f} ms`" if result["layer2_latency_ms"] is not None else "- L2 layout + communication：不可用",
                "",
                "### L2 分段",
                "",
                "| 阶段 | 单次延迟 | 每次 attention 调用数 |",
                "|---|---:|---:|",
            )
        )
        for name, segment in sorted(result["layer2_segments"].items()):
            lines.append(f"| `{name}` | {segment['latency_ms']:.4f} ms | {segment['calls_per_attention']} |")
        layer3 = result["layer3"]
        lines.extend(("", "### L3 实际通信", ""))
        if layer3 is None:
            lines.append("L3 不可用。")
        else:
            efficiency = layer3["bus_peak_efficiency"]
            lines.extend(
                (
                    f"- 路径：`{layer3['communication_pattern']}` / `{layer3['primitive']}`",
                    f"- 延迟：`{layer3['latency_ms']:.4f} ms`",
                    f"- 整体 Bus bandwidth：`{layer3['bus_bandwidth_gbps']:.2f} GB/s`",
                    f"- 峰值口径：`{layer3['peak_profile_pattern']}`，nominal=`{layer3['nominal_bus_peak_gbps']:.2f} GB/s`"
                    if layer3["nominal_bus_peak_gbps"] is not None
                    else f"- 峰值口径：`{layer3['peak_profile_pattern']}`，nominal 不可用",
                    f"- 理论峰值效率：`{efficiency * 100:.2f}%`" if efficiency is not None else f"- 理论峰值效率：不可用（{layer3['efficiency_status']}）",
                    f"- Aux 输入：`{layer3['aux_qkv_input'] or '无'}`",
                    "",
                    "| 通信分量 | 独立延迟 | 逻辑 payload/rank | 跨链路字节/rank | Bus bandwidth | 峰值效率（口径） | 状态 |",
                    "|---|---:|---:|---:|---:|---:|---|",
                )
            )
            for name, component in sorted(layer3["communication_components"].items()):
                latency = f"{component['latency_ms']:.4f} ms" if component.get("latency_ms") is not None else "-"
                bandwidth = f"{component['bus_bandwidth_gbps']:.2f} GB/s" if component.get("bus_bandwidth_gbps") is not None else "-"
                component_efficiency = component.get("bus_peak_efficiency")
                observed_efficiency = component.get("observed_bus_peak_efficiency")
                if component_efficiency is not None:
                    efficiency_text = f"{component_efficiency * 100:.2f}% (`{component.get('peak_profile_pattern')}`)"
                elif observed_efficiency is not None:
                    efficiency_text = f"~{observed_efficiency * 100:.2f}% (`{component.get('peak_profile_pattern')}`; {component.get('status')})"
                else:
                    efficiency_text = "-"
                lines.append(
                    f"| `{name}` | {latency} | {_format_bytes(component['logical_payload_bytes_per_rank'])} | {_format_bytes(component['network_bytes_per_rank'])} | {bandwidth} | {efficiency_text} | `{component.get('status', 'not_measured')}` |"
                )
        overlap = result["overlap"]
        lines.extend(("", "### L1/L2 关系", ""))
        if overlap is None:
            lines.append("Overlap 估算不可用。")
        else:
            ratio = overlap["estimated_overlap_ratio"]
            lines.extend(
                (
                    f"- L1 完整路径：`{overlap['full_l1_latency_ms']:.4f} ms`",
                    f"- Compute-only：`{overlap['compute_only_latency_ms']:.4f} ms`",
                    f"- 估算状态：`{overlap['estimate_status']}`",
                    f"- 估算通信隐藏比例：`{ratio * 100:.2f}%`" if ratio is not None else "- 估算通信隐藏比例：不可用",
                )
            )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def diagnostic_candidates_from_report(path: Path, suite: dict[str, Any]) -> dict[str, str]:
    require(path.is_file(), f"recommendation report does not exist: {path}")
    report = json.loads(path.read_text(encoding="utf-8"))
    require(report.get("kind") == "sp_attention_benchmark_report_v1", "unsupported SP recommendation report")
    require(report.get("suite_id") == suite["suite_id"], "SP recommendation report suite_id differs from diagnostic suite")
    candidates = {
        item["case_id"]: item["recommendation"]["candidate_id"] for item in report.get("workloads", []) if isinstance(item.get("recommendation"), dict) and item["recommendation"].get("candidate_id")
    }
    missing = sorted({case["case_id"] for case in suite["cases"]} - set(candidates))
    require(not missing, f"SP recommendation report does not contain a formal winner for cases: {missing}")
    return candidates


def run_diagnostics(
    suite: dict[str, Any],
    output_dir: Path,
    *,
    candidate_ids: set[str],
    recommended_candidate_ids: dict[str, str] | None,
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
    recommended_candidate_ids = recommended_candidate_ids or {}
    requested_candidate_ids = candidate_ids | set(recommended_candidate_ids.values())
    require(requested_candidate_ids, "diagnose requires --candidate or --recommendation-report")
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
        selections = _selected_candidates(suite, selection_runtime, requested_candidate_ids, {"ulysses", "ring"})
        for case_id, candidates in selections.items():
            allowed = candidate_ids | ({recommended_candidate_ids[case_id]} if case_id in recommended_candidate_ids else set())
            selections[case_id] = [candidate for candidate in candidates if candidate["candidate_id"] in allowed]
            require(selections[case_id], f"no requested diagnostic candidate is eligible for case {case_id}")
        environments = _distributed_environment(device, rank, local_rank)
        _interconnect_platform(interconnect_profiles, platform_id, [{"environment": {"ranks": environments}}])
        run_contract = {
            "warmup": warmup,
            "iterations": iterations,
            "seed": seed,
            "world_size": world_size,
            "timing_mode": "max_rank_event_list_single_sync",
        }
        handle = None
        if rank == 0:
            output_dir.mkdir(parents=True, exist_ok=True)
            write_json(output_dir / "candidate_catalog.json", candidate_catalog)
            handle = raw_path.open("w", encoding="utf-8")
        written = 0
        replay_input_cache = {}
        for repeat in range(repeat_runs):
            for case in suite["cases"]:
                if _is_sparse_case(case) and case["case_id"] not in replay_input_cache:
                    replay_input_cache[case["case_id"]] = _load_sp_replay_inputs(case, device, rank)
                candidates = selections[case["case_id"]]
                candidates = candidates[repeat % len(candidates) :] + candidates[: repeat % len(candidates)]
                for candidate in candidates:
                    try:
                        metrics = prepare_diagnostics(
                            case,
                            candidate,
                            device,
                            seed + repeat * 1000,
                            rank,
                            warmup,
                            iterations,
                            inputs=replay_input_cache.get(case["case_id"]),
                        )
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
                            "run": run_contract,
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
                            "run": run_contract,
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
            report_path = output_dir / "diagnostic_report.md"
            report_path.write_text(render_diagnostic_markdown(report), encoding="utf-8")
            summary = {"raw": str(raw_path), "report": str(report_path), "written": written, **report["summary"]}
            write_json(output_dir / "diagnostic_summary.json", summary)
            print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
            return summary
        return {"rank": rank, "written": 0}
    finally:
        dist.destroy_process_group()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    derive_parser = commands.add_parser("derive-replay-suites", help="Derive per-SP suites from one complete QKV replay.")
    derive_parser.add_argument("--manifest", type=Path, required=True)
    derive_parser.add_argument("--output-dir", type=Path, required=True)
    derive_parser.add_argument("--suite-id", required=True)
    derive_parser.add_argument("--conditioner-tokens", type=int, required=True)
    derive_parser.add_argument("--sp-size", type=int, action="append", default=[])
    derive_parser.add_argument("--keep-ratio", type=float, default=0.15)
    inspect_parser = commands.add_parser("inspect", help="Validate and summarize an SP shape suite.")
    inspect_parser.add_argument("--suite", type=Path, required=True)
    candidate_parser = commands.add_parser("candidates", help="Probe and enumerate valid SP candidates.")
    candidate_parser.add_argument("--suite", type=Path, required=True)
    candidate_parser.add_argument("--device", default="cuda:0")
    candidate_parser.add_argument("--dense-backend", action="append", default=[])
    candidate_parser.add_argument("--sparse-backend", action="append", default=[])
    reference_parser = commands.add_parser("validate-reference", help="Compare production SP candidates with a dense or sparse bulk reference.")
    reference_parser.add_argument("--suite", type=Path, required=True)
    reference_parser.add_argument("--output", type=Path, required=True)
    reference_parser.add_argument("--candidate", action="append", default=[])
    reference_parser.add_argument("--algorithm", action="append", choices=("ulysses", "ring"), default=[])
    reference_parser.add_argument("--dense-backend", action="append", default=[])
    reference_parser.add_argument("--sparse-backend", action="append", default=[])
    reference_parser.add_argument("--seed", type=int, default=42)
    run_parser = commands.add_parser("run", help="Run production SP attention under torchrun.")
    run_parser.add_argument("--suite", type=Path, required=True)
    run_parser.add_argument("--output-dir", type=Path, required=True)
    run_parser.add_argument("--candidate", action="append", default=[])
    run_parser.add_argument("--algorithm", action="append", choices=("ulysses", "ring"), default=[])
    run_parser.add_argument("--dense-backend", action="append", default=[])
    run_parser.add_argument("--sparse-backend", action="append", default=[])
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
    diagnose_parser.add_argument("--candidate", action="append", default=[])
    diagnose_parser.add_argument("--recommendation-report", type=Path)
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
        if args.command == "derive-replay-suites":
            outputs = derive_replay_suites(
                args.manifest,
                args.output_dir,
                suite_id=args.suite_id,
                conditioner_tokens=args.conditioner_tokens,
                sp_sizes=args.sp_size or [2, 4, 8],
                keep_ratio=args.keep_ratio,
            )
            print(json.dumps({"outputs": [str(path) for path in outputs]}, ensure_ascii=False, sort_keys=True))
            return 0
        suite = load_suite(args.suite)
        if args.command == "inspect":
            print(json.dumps(inspect_suite(suite), ensure_ascii=False, sort_keys=True))
            return 0
        if args.command == "candidates":
            runtime = _discover_runtime(args.device)
            if all(_is_sparse_case(case) for case in suite["cases"]):
                sparse_backends = args.sparse_backend or runtime["sparse_backends"]
                require(sparse_backends, "no eligible sparse attention backend was discovered")
                value = build_candidate_catalog(suite, runtime, sparse_backends=sparse_backends)
            else:
                dense_backends = args.dense_backend or runtime["dense_backends"]
                require(dense_backends, "no eligible dense attention backend was discovered")
                value = build_candidate_catalog(suite, runtime, dense_backends)
            print(json.dumps(value, ensure_ascii=False, indent=2))
            return 0
        if args.command == "validate-reference":
            validate_reference_suite(
                suite,
                args.output,
                candidate_ids=set(args.candidate),
                algorithms=set(args.algorithm or ("ulysses", "ring")),
                dense_backends=args.dense_backend,
                sparse_backends=args.sparse_backend,
                seed=args.seed,
            )
            return 0
        if args.command == "run":
            run_suite(
                suite,
                args.output_dir,
                candidate_ids=set(args.candidate),
                algorithms=set(args.algorithm or ("ulysses", "ring")),
                dense_backends=args.dense_backend,
                sparse_backends=args.sparse_backend,
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
            candidate_ids = set(args.candidate)
            recommended_candidate_ids = {}
            if args.recommendation_report:
                recommended_candidate_ids = diagnostic_candidates_from_report(args.recommendation_report, suite)
            run_diagnostics(
                suite,
                args.output_dir,
                candidate_ids=candidate_ids,
                recommended_candidate_ids=recommended_candidate_ids,
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
            report_path = args.output_dir / "diagnostic_report.md"
            report_path.write_text(render_diagnostic_markdown(report), encoding="utf-8")
            summary = {"raw": str(args.raw), "report": str(report_path), **report["summary"]}
            write_json(args.output_dir / "diagnostic_summary.json", summary)
            print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
            return 0
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    raise AssertionError(f"unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
