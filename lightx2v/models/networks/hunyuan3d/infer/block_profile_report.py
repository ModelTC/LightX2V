"""Hunyuan3D-specific block profile report config and MoE op expansion."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from tools.profile.region_event_trace import (
    GpuEvent,
    OpEntry,
    RegionBlock,
    RegionTraceConfig,
    RegionTraceHooks,
    analyze_region_trace,
    infer_peak_tflops_from_device,
)

if TYPE_CHECKING:
    from lightx2v.utils.transformer_profile import ProfileRunMeta

_PEAK_TFLOPS_BF16, _PEAK_TFLOPS_FP8 = infer_peak_tflops_from_device()

_HUNYUAN3D_CONFIG = RegionTraceConfig(
    region_order=(
        "skip_connection",
        "self_attn",
        "cross_attn",
        "dense_ffn",
        "moe",
    ),
    skip_regions=frozenset({"ProfilerStep", "moe_ffn"}),
    gpu_skip_prefixes=("ProfilerStep#", "block_"),
    cpu_skip_prefixes=("ProfilerStep", "block_"),
    peak_tflops_bf16=_PEAK_TFLOPS_BF16,
    peak_tflops_fp8=_PEAK_TFLOPS_FP8,
)


def _moe_stage_specs(fc_schema: str, hidden: int, intermediate: int) -> tuple[tuple[str, int, int], ...]:
    if fc_schema == "gate_up_down":
        return (
            ("gate_proj", intermediate, hidden),
            ("up_proj", intermediate, hidden),
            ("down_proj", hidden, intermediate),
        )
    if fc_schema == "flashinfer_swiglu":
        return (
            ("fc1", 2 * intermediate, hidden),
            ("fc2", hidden, intermediate),
        )
    return (
        ("fc1", intermediate, hidden),
        ("fc2", hidden, intermediate),
    )


def _gemm_op(op: OpEntry, offset: int, tag: str, m: int, n: int, k: int, extra: dict) -> OpEntry:
    return OpEntry(
        seq=op.seq * 1000 + offset,
        region=op.region,
        kind="GEMM",
        tag=tag,
        flops=float(2 * m * n * k),
        extra={**op.extra, "M": int(m), "N": int(n), "K": int(k), **extra},
    )


def _expand_moe_routed(op: OpEntry) -> list[OpEntry]:
    if op.kind != "MOE" or op.tag != "moe_routed":
        return [op]

    backend = str(op.extra.get("backend", "pytorch_loop"))
    fc_schema = str(op.extra.get("fc_schema", "fc1_fc2"))
    hidden = int(op.extra["hidden"])
    intermediate = int(op.extra["intermediate"])
    routed_tokens = int(op.extra.get("routed_tokens", int(op.extra["tokens"]) * int(op.extra["top_k"])))
    stages = _moe_stage_specs(fc_schema, hidden, intermediate)

    expanded: list[OpEntry] = []
    offset = 0
    expert_tokens = op.extra.get("expert_tokens")
    if backend == "pytorch_loop" and expert_tokens:
        for expert_idx, count in enumerate(expert_tokens):
            count = int(count)
            if count <= 0:
                continue
            for stage, n, k in stages:
                offset += 1
                expanded.append(
                    _gemm_op(
                        op,
                        offset,
                        f"moe_routed.e{expert_idx}.{stage}",
                        count,
                        n,
                        k,
                        {
                            "expert": expert_idx,
                            "stage": stage,
                            "backend": backend,
                            "fc_schema": fc_schema,
                            "routed_tokens": routed_tokens,
                        },
                    )
                )
        return expanded

    for stage, n, k in stages:
        offset += 1
        expanded.append(
            _gemm_op(
                op,
                offset,
                f"moe_routed.{stage}",
                routed_tokens,
                n,
                k,
                {
                    "stage": stage,
                    "backend": backend,
                    "fc_schema": fc_schema,
                    "routed_tokens": routed_tokens,
                },
            )
        )
    return expanded


def analyze(
    trace_path: Path,
    op_trace_path: Path,
    *,
    block_idx: int,
    step_id: int | None,
    profile_meta: ProfileRunMeta | None = None,
) -> tuple[str, list[GpuEvent], list[RegionBlock]]:
    hooks = RegionTraceHooks(
        subtitle_builder=lambda step, n_events, _win: (f"Layer index: {block_idx}   ProfilerStep#{step}   GPU events: {n_events}"),
        expand_op=_expand_moe_routed,
    )
    result = analyze_region_trace(
        trace_path,
        op_trace_path,
        window_annotation=f"block_{block_idx}",
        config=_HUNYUAN3D_CONFIG,
        hooks=hooks,
        step_id=step_id,
        profile_meta=profile_meta,
    )
    return result.report, result.gpu_events, result.regions
