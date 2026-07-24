"""SekoTalk / Wan-specific block profile report config."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from tools.profile.region_event_trace import (
    GpuEvent,
    RegionBlock,
    RegionTraceConfig,
    RegionTraceHooks,
    analyze_region_trace,
    infer_peak_tflops_from_device,
)

if TYPE_CHECKING:
    from lightx2v.utils.transformer_profile import ProfileRunMeta

try:
    _PEAK_TFLOPS_BF16, _PEAK_TFLOPS_FP8 = infer_peak_tflops_from_device()
except RuntimeError:
    _PEAK_TFLOPS_BF16 = _PEAK_TFLOPS_FP8 = None

_WAN_CONFIG = RegionTraceConfig(
    region_order=(
        "self_attn",
        "cross_attn",
        "dense_ffn",
        "audio_adapter",
    ),
    skip_regions=frozenset({"ProfilerStep"}),
    gpu_skip_prefixes=("ProfilerStep#", "block_"),
    cpu_skip_prefixes=("ProfilerStep", "block_"),
    peak_tflops_bf16=_PEAK_TFLOPS_BF16,
    peak_tflops_fp8=_PEAK_TFLOPS_FP8,
)


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
    )
    result = analyze_region_trace(
        trace_path,
        op_trace_path,
        window_annotation=f"block_{block_idx}",
        config=_WAN_CONFIG,
        hooks=hooks,
        step_id=step_id,
        profile_meta=profile_meta,
    )
    return result.report, result.gpu_events, result.regions
