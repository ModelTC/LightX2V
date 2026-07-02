#!/usr/bin/env python3
"""Chrome trace report grouped by ``record_function`` regions."""

from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from tools.profile.trace_correlation import (
    build_correlation_indexes,
    related_events_for_kernel,
    simplify_kernel_name,
)

_GPU_CATS = frozenset({"kernel", "gpu_memcpy", "gpu_memset"})

# Core-kernel patterns (extend when new GEMM/ATTN/MoE backends appear in traces).
_CORE_GEMM = re.compile(
    r"nvjet|cutlass.*GemmUniversal|cutlass_3x_gemm|cutlass::device_kernel<.*gemm|_grouped_mm",
    re.I,
)
_CORE_ATTN = re.compile(r"cudnn.*sdpa|flash.*attn|flash_fwd_kernel|fmha|sv_f8_attn|qk_int8.*attn", re.I)
_CORE_MOE = re.compile(r"cutlass.*GemmUniversal", re.I)

OpExpander = Callable[["OpEntry"], list["OpEntry"]]
SubtitleBuilder = Callable[[int, int, tuple[float, float]], str]


def _default_expand_op(op: "OpEntry") -> list["OpEntry"]:
    return [op]


def infer_peak_tflops_from_device(device: int | None = None) -> tuple[float, float]:
    """Return BF16/FP8 dense Tensor Core peaks for the current CUDA device."""
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("Cannot infer peak TFLOPS: PyTorch is not available.") from exc

    if not torch.cuda.is_available():
        raise RuntimeError("Cannot infer peak TFLOPS: CUDA is not available. Please add hardware peak TFLOPS for this environment.")

    device = torch.cuda.current_device() if device is None else device
    props = torch.cuda.get_device_properties(device)
    name = props.name
    if "H100" in name and props.major == 9 and props.minor == 0:
        return 989.0, 1979.0

    raise RuntimeError(f"Cannot infer peak TFLOPS for {name} (sm_{props.major}{props.minor}). Please add this hardware's BF16/FP8 peak TFLOPS to infer_peak_tflops_from_device().")


@dataclass(frozen=True)
class RegionTraceConfig:
    """Per-model region layout and skip rules for trace parsing."""

    region_order: tuple[str, ...]
    peak_tflops_bf16: float
    peak_tflops_fp8: float
    skip_regions: frozenset[str] = frozenset({"ProfilerStep"})
    gpu_skip_prefixes: tuple[str, ...] = ("ProfilerStep#",)
    cpu_skip_prefixes: tuple[str, ...] = ("ProfilerStep",)


@dataclass(frozen=True)
class RegionTraceHooks:
    """Per-model report callbacks."""

    subtitle_builder: SubtitleBuilder
    expand_op: OpExpander = _default_expand_op


@dataclass
class GpuEvent:
    idx: int
    ts: float
    dur_ms: float
    cat: str
    kernel: str
    kernel_raw: str
    region: str
    is_core: bool
    core_kind: str  # GEMM | ATTN | MOE | ""


@dataclass
class OpEntry:
    seq: int
    region: str
    kind: str
    tag: str
    flops: float
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class RegionBlock:
    name: str
    gpu_ann_ms: float | None
    kernel_sum_ms: float
    events: list[GpuEvent] = field(default_factory=list)
    ops: list[OpEntry] = field(default_factory=list)


@dataclass
class RegionTraceResult:
    report: str
    gpu_events: list[GpuEvent]
    regions: list[RegionBlock]
    step_id: int
    window: tuple[float, float]


@dataclass(frozen=True)
class _Interval:
    start: float
    end: float

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


@dataclass
class ProfilerStepStats:
    step_id: int
    wall_us: float
    gpu_active_us: float
    gpu_event_count: int

    @property
    def gap_us(self) -> float:
        return max(0.0, self.wall_us - self.gpu_active_us)


def parse_step_id(name: str) -> int | None:
    """Parse ``ProfilerStep#N`` annotation names."""
    m = re.match(r"^ProfilerStep#(\d+)$", name)
    return int(m.group(1)) if m else None


def load_events(path: Path) -> list[dict]:
    """Load Chrome / TensorBoard ``traceEvents`` list from a ``.pt.trace.json`` file."""
    with path.open() as f:
        payload = json.load(f)
    return payload.get("traceEvents", payload)


def _merge_intervals(intervals: list[_Interval]) -> list[_Interval]:
    if not intervals:
        return []
    ordered = sorted(intervals, key=lambda item: item.start)
    merged = [ordered[0]]
    for cur in ordered[1:]:
        prev = merged[-1]
        if cur.start <= prev.end:
            merged[-1] = _Interval(prev.start, max(prev.end, cur.end))
        else:
            merged.append(cur)
    return merged


def _event_interval(ev: dict) -> _Interval:
    start = float(ev["ts"])
    return _Interval(start, start + float(ev.get("dur", 0)))


def _is_contained(window: _Interval, event: _Interval) -> bool:
    return event.start >= window.start and event.end <= window.end


def analyze_trace_events(events: list[dict]) -> list[ProfilerStepStats]:
    steps: list[tuple[int, _Interval]] = []
    gpu_events: list[_Interval] = []
    for ev in events:
        if ev.get("ph") != "X":
            continue
        cat = ev.get("cat")
        if cat == "user_annotation":
            step_id = parse_step_id(ev.get("name", ""))
            if step_id is not None:
                steps.append((step_id, _event_interval(ev)))
        elif cat in _GPU_CATS and float(ev.get("dur", 0)) > 0:
            gpu_events.append(_event_interval(ev))

    stats: list[ProfilerStepStats] = []
    for step_id, window in sorted(steps, key=lambda item: item[0]):
        contained = [event for event in gpu_events if _is_contained(window, event)]
        gpu_active_us = sum(interval.duration for interval in _merge_intervals(contained))
        stats.append(
            ProfilerStepStats(
                step_id=step_id,
                wall_us=window.duration,
                gpu_active_us=gpu_active_us,
                gpu_event_count=len(contained),
            )
        )
    return stats


def load_op_trace(path: Path) -> list[OpEntry]:
    """Load logical-op JSONL produced by model-side ``op_shape_trace`` emitters."""
    ops: list[OpEntry] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            flops = float(row.pop("flops", 0))
            seq = int(row.pop("seq"))
            region = row.pop("region")
            kind = row.pop("kind")
            tag = row.pop("tag")
            ops.append(OpEntry(seq=seq, region=region, kind=kind, tag=tag, flops=flops, extra=row))
    return ops


def expand_ops(ops: list[OpEntry], hooks: RegionTraceHooks) -> list[OpEntry]:
    expanded: list[OpEntry] = []
    for op in ops:
        expanded.extend(hooks.expand_op(op))
    return expanded


def is_core_kernel(raw: str, cat: str) -> tuple[bool, str]:
    """Heuristic: map a GPU kernel name to a compute class (GEMM / ATTN / MOE).

    Extend this regex table when profiling on new stacks.
    """
    if cat != "kernel":
        return False, ""
    if _CORE_ATTN.search(raw):
        return True, "ATTN"
    if _CORE_MOE.search(raw):
        return True, "MOE"
    if _CORE_GEMM.search(raw):
        return True, "GEMM"
    return False, ""


def _launch_ts(ev: dict, by_corr: dict, by_ext: dict) -> float:
    related = related_events_for_kernel(ev, by_corr, by_ext)
    for rel in related:
        if rel.get("cat") != "cuda_runtime":
            continue
        name = rel.get("name", "")
        if "cudaLaunchKernel" in name or "LaunchKernel" in name:
            return float(rel["ts"])
    return float(ev["ts"])


def assign_parent_region(
    ts: float,
    te: float,
    launch_ts: float,
    cpu_anns: list[tuple[float, float, str]],
    gpu_anns: list[tuple[float, float, str]],
    config: RegionTraceConfig,
) -> str:
    """Assign a GPU event to a ``record_function`` region name.

    1. Largest overlap with ``gpu_user_annotation`` intervals (preferred).
    2. CPU ``user_annotation`` half-open interval containing the launch timestamp.
    3. ``"-"`` if no match.
    """
    best = ("-", 0.0)
    for a0, a1, name in gpu_anns:
        if name.startswith(config.gpu_skip_prefixes):
            continue
        ov = max(0.0, min(te, a1) - max(ts, a0))
        if ov > best[1]:
            best = (name, ov)
    if best[1] > 0:
        return best[0]

    if cpu_anns:
        for i, (start, _end, name) in enumerate(cpu_anns):
            next_start = cpu_anns[i + 1][0] if i + 1 < len(cpu_anns) else float("inf")
            if start <= launch_ts < next_start:
                return name
        last = cpu_anns[-1]
        if launch_ts >= last[0]:
            return last[2]

    cand = [a for a in cpu_anns if a[0] <= launch_ts < a[1]]
    if not cand:
        cand = [a for a in cpu_anns if a[0] < te and a[1] > ts]
    if not cand and cpu_anns and launch_ts < cpu_anns[0][0]:
        return cpu_anns[0][2]
    if not cand:
        return "-"
    return min(cand, key=lambda a: a[1] - a[0])[2]


def collect_regions(
    events: list[dict],
    t0: float,
    t1: float,
    config: RegionTraceConfig,
) -> tuple[list[tuple[float, float, str]], list[tuple[float, float, str]], dict[str, float]]:
    """Collect CPU/GPU ``record_function`` intervals inside ``[t0, t1]``."""
    cpu_regions: list[tuple[float, float, str]] = []
    gpu_ann_intervals: list[tuple[float, float, str]] = []
    gpu_ann_ms: dict[str, float] = defaultdict(float)
    for ev in events:
        if ev.get("ph") != "X":
            continue
        name = ev.get("name", "")
        cat = ev.get("cat", "")
        ts, te = float(ev["ts"]), float(ev["ts"]) + float(ev["dur"])
        if ts < t0 or te > t1:
            continue
        if cat == "user_annotation":
            if name.startswith(config.cpu_skip_prefixes) or name in config.skip_regions:
                continue
            cpu_regions.append((ts, te, name))
        elif cat == "gpu_user_annotation":
            if name.startswith(config.gpu_skip_prefixes) or name in config.skip_regions:
                continue
            gpu_ann_intervals.append((ts, te, name))
            gpu_ann_ms[name] += float(ev["dur"]) / 1000.0
    cpu_regions.sort(key=lambda x: x[0])
    return cpu_regions, gpu_ann_intervals, dict(gpu_ann_ms)


def collect_gpu_events(
    events: list[dict],
    bt0: float,
    bt1: float,
    cpu_regions: list[tuple[float, float, str]],
    gpu_ann_intervals: list[tuple[float, float, str]],
    config: RegionTraceConfig,
) -> list[GpuEvent]:
    """Collect GPU timeline events in ``[bt0, bt1)`` and assign each to a region."""
    by_corr, by_ext = build_correlation_indexes(events)
    out: list[GpuEvent] = []
    idx = 0
    for ev in events:
        if ev.get("ph") != "X" or ev.get("cat") not in _GPU_CATS:
            continue
        ks = float(ev["ts"])
        if not (bt0 <= ks < bt1):
            continue
        idx += 1
        raw = ev.get("name", "")
        cat = ev.get("cat", "")
        ke = ks + float(ev.get("dur", 0))
        launch = _launch_ts(ev, by_corr, by_ext)
        region = assign_parent_region(ks, ke, launch, cpu_regions, gpu_ann_intervals, config)
        is_core, core_kind = is_core_kernel(raw, cat)
        out.append(
            GpuEvent(
                idx=idx,
                ts=ks,
                dur_ms=float(ev.get("dur", 0)) / 1000.0,
                cat=cat,
                kernel=simplify_kernel_name(raw),
                kernel_raw=raw,
                region=region,
                is_core=is_core,
                core_kind=core_kind,
            )
        )
    out.sort(key=lambda e: (e.ts, e.idx))
    return out


def group_by_region(
    gpu_events: list[GpuEvent],
    ops: list[OpEntry],
    gpu_ann_ms: dict[str, float],
    config: RegionTraceConfig,
) -> list[RegionBlock]:
    """Bucket events and logical ops by region, ordered per ``config.region_order``."""
    by_name: dict[str, RegionBlock] = {name: RegionBlock(name=name, gpu_ann_ms=gpu_ann_ms.get(name), kernel_sum_ms=0.0) for name in config.region_order}
    extra: dict[str, RegionBlock] = {}

    def _block(region: str) -> RegionBlock:
        if region in by_name:
            return by_name[region]
        if region not in extra:
            extra[region] = RegionBlock(name=region, gpu_ann_ms=gpu_ann_ms.get(region), kernel_sum_ms=0.0)
        return extra[region]

    for ev in gpu_events:
        block = _block(ev.region)
        block.events.append(ev)
        block.kernel_sum_ms += ev.dur_ms

    for op in ops:
        _block(op.region).ops.append(op)

    ordered: list[RegionBlock] = []
    for name in config.region_order:
        blk = by_name[name]
        if blk.events or blk.ops:
            ordered.append(blk)
    for name in sorted(extra):
        if name not in config.region_order and not name.startswith(config.gpu_skip_prefixes):
            ordered.append(extra[name])
    return ordered


def peak_tflops_for_kernel(kernel_raw: str, config: RegionTraceConfig) -> float:
    """Pick H100 roofline by kernel dtype (BF16 vs FP8/quant)."""
    if re.search(r"fp8|e4m3|int8", kernel_raw, re.I):
        return config.peak_tflops_fp8
    return config.peak_tflops_bf16


def tflops_str(flops: float, dur_ms: float, peak_tflops: float) -> str:
    if dur_ms <= 0:
        return "unknown"
    t = flops / (dur_ms / 1000.0) / 1e12
    pct = 100.0 * t / peak_tflops
    return f"{t:.0f} TFLOPS ({pct:.0f}% peak)"


def format_op_shape(op: OpEntry) -> str:
    if op.kind == "GEMM":
        return f"M={op.extra.get('M')} N={op.extra.get('N')} K={op.extra.get('K')}"
    if op.kind == "ATTN":
        return f"B={op.extra.get('B')} H={op.extra.get('H')} Sq={op.extra.get('Sq')} Sk={op.extra.get('Sk')} D={op.extra.get('D')}"
    if op.kind == "MOE":
        return f"tokens={op.extra.get('tokens')} top_k={op.extra.get('top_k')} H={op.extra.get('hidden')} I={op.extra.get('intermediate')}"
    return ""


def region_wall_ms(block: RegionBlock) -> float:
    if block.gpu_ann_ms is not None and block.gpu_ann_ms > 0:
        return block.gpu_ann_ms
    if block.kernel_sum_ms > 0:
        return block.kernel_sum_ms
    return 0.0


def format_evt_line(
    ev: GpuEvent,
    op: OpEntry | None,
    *,
    peak_tflops: float,
) -> str:
    base = f"    evt {ev.idx:03d}  {ev.dur_ms:7.3f} ms  {ev.kernel}"
    if op is None:
        return base
    shape = format_op_shape(op)
    flops_g = op.flops / 1e9
    eff = tflops_str(op.flops, ev.dur_ms, peak_tflops)
    return f"{base}  {op.kind} {op.tag}  {shape}  {flops_g:.1f} GF  {eff}"


def format_unmatched_op_line(op: OpEntry) -> str:
    shape = format_op_shape(op)
    flops_g = op.flops / 1e9
    return f"    {op.kind} {op.tag}  {shape}  {'—':>7s}  {flops_g:.1f} GF  (no core kernel match)"


def render_region_chronological(
    block: RegionBlock,
    wall: float,
    config: RegionTraceConfig,
    hooks: RegionTraceHooks,
) -> list[str]:
    """Chronological lines for one region."""
    lines: list[str] = []
    gemm_queue = [o for o in block.ops if o.kind == "GEMM"]
    attn_queue = [o for o in block.ops if o.kind == "ATTN"]

    for ev in sorted(block.events, key=lambda e: (e.ts, e.idx)):
        op = None
        if ev.is_core:
            if ev.core_kind in {"GEMM", "MOE"} and gemm_queue:
                op = gemm_queue.pop(0)
            elif ev.core_kind == "ATTN" and attn_queue:
                op = attn_queue.pop(0)
        lines.append(format_evt_line(ev, op, peak_tflops=peak_tflops_for_kernel(ev.kernel_raw, config)))

    for op in gemm_queue:
        lines.append(format_unmatched_op_line(op))
    for op in attn_queue:
        lines.append(format_unmatched_op_line(op))

    return lines


def format_step_summary(step: ProfilerStepStats) -> str:
    wall_ms = step.wall_us / 1000.0
    compute_ms = step.gpu_active_us / 1000.0
    gap_ms = step.gap_us / 1000.0
    return f"ProfilerStep  compute={compute_ms:.3f} ms  (baseline)  wall={wall_ms:.3f} ms  gap={gap_ms:.3f} ms  (profiler overhead)"


def block_kernel_split(gpu_events: list[GpuEvent]) -> tuple[float, float, int, int]:
    core_ms = prim_ms = 0.0
    core_n = prim_n = 0
    for ev in gpu_events:
        if ev.is_core:
            core_ms += ev.dur_ms
            core_n += 1
        else:
            prim_ms += ev.dur_ms
            prim_n += 1
    return core_ms, prim_ms, core_n, prim_n


def find_profiler_steps(events: list[dict]) -> list[tuple[int, float, float]]:
    steps: list[tuple[int, float, float]] = []
    for ev in events:
        if ev.get("ph") == "X" and ev.get("cat") == "user_annotation":
            sid = parse_step_id(ev.get("name", ""))
            if sid is not None:
                steps.append((sid, float(ev["ts"]), float(ev["ts"]) + float(ev["dur"])))
    return steps


def find_annotation_window(
    events: list[dict],
    annotation: str,
    step_t0: float,
    step_t1: float,
) -> tuple[float, float] | None:
    """First ``user_annotation`` named ``annotation`` at or after ``step_t0``."""
    for ev in events:
        if ev.get("ph") == "X" and ev.get("cat") == "user_annotation" and ev.get("name") == annotation:
            if float(ev["ts"]) >= step_t0:
                return float(ev["ts"]), float(ev["ts"]) + float(ev["dur"])
    return None


def render_report(
    trace_path: Path,
    subtitle: str,
    step_stats: ProfilerStepStats,
    regions: list[RegionBlock],
    gpu_events: list[GpuEvent],
    config: RegionTraceConfig,
    hooks: RegionTraceHooks,
    *,
    profile_meta: Any | None = None,
    window_label: str = "Block",
) -> str:
    lines: list[str] = []
    lines.append("=" * 100)
    lines.append(f"Trace: {trace_path}")
    if profile_meta is not None:
        lines.append(profile_meta.format_header())
    lines.append(subtitle)
    unassigned = sum(1 for e in gpu_events if e.region == "-")
    lines.append(f"Region assigned: {len(gpu_events) - unassigned}/{len(gpu_events)}")
    lines.append(format_step_summary(step_stats))
    core_ms, prim_ms, core_n, prim_n = block_kernel_split(gpu_events)
    total_ms = core_ms + prim_ms
    if total_ms > 0:
        lines.append(
            f"{window_label} kernels  core={core_ms:.3f} ms ({100 * core_ms / total_ms:.1f}%, n={core_n})  "
            f"primitive={prim_ms:.3f} ms ({100 * prim_ms / total_ms:.1f}%, n={prim_n})  "
            f"(self-time sum in block window)"
        )
    lines.append("=" * 100)

    for block in regions:
        wall = region_wall_ms(block)
        lines.append("")
        lines.append(f"── {block.name} ── kernel_sum={block.kernel_sum_ms:.3f} ms")
        lines.extend(render_region_chronological(block, wall, config, hooks))

    lines.append("")
    return "\n".join(lines)


def analyze_region_trace(
    trace_path: Path,
    op_trace_path: Path,
    *,
    window_annotation: str,
    config: RegionTraceConfig,
    hooks: RegionTraceHooks,
    step_id: int | None = None,
    profile_meta: Any | None = None,
    window_label: str = "Block",
) -> RegionTraceResult:
    """End-to-end: trace + op JSONL → text report and structured buckets."""
    events = load_events(trace_path)
    ops = expand_ops(load_op_trace(op_trace_path), hooks)

    steps = find_profiler_steps(events)
    if not steps:
        raise SystemExit(f"No ProfilerStep in {trace_path}")
    if step_id is None:
        step_id = steps[-1][0]
    step_stats_list = analyze_trace_events(events)
    step_stats = next(s for s in step_stats_list if s.step_id == step_id)
    _, t0, t1 = next(s for s in steps if s[0] == step_id)

    block_win = find_annotation_window(events, window_annotation, t0, t1)
    if block_win is None:
        block_win = (t0, t1)

    cpu_regions, gpu_ann_intervals, gpu_ann_ms = collect_regions(events, block_win[0], block_win[1], config)
    # GPU kernels may finish after the CPU annotation returns; extend to ProfilerStep end.
    gpu_events = collect_gpu_events(events, block_win[0], t1, cpu_regions, gpu_ann_intervals, config)
    regions = group_by_region(gpu_events, ops, gpu_ann_ms, config)
    subtitle = hooks.subtitle_builder(step_id, len(gpu_events), block_win)
    report = render_report(
        trace_path,
        subtitle,
        step_stats,
        regions,
        gpu_events,
        config,
        hooks,
        profile_meta=profile_meta,
        window_label=window_label,
    )
    return RegionTraceResult(
        report=report,
        gpu_events=gpu_events,
        regions=regions,
        step_id=step_id,
        window=block_win,
    )


__all__ = [
    "GpuEvent",
    "OpEntry",
    "ProfilerStepStats",
    "RegionBlock",
    "RegionTraceConfig",
    "RegionTraceHooks",
    "RegionTraceResult",
    "SubtitleBuilder",
    "analyze_trace_events",
    "analyze_region_trace",
    "assign_parent_region",
    "collect_gpu_events",
    "collect_regions",
    "expand_ops",
    "format_op_shape",
    "format_step_summary",
    "group_by_region",
    "infer_peak_tflops_from_device",
    "is_core_kernel",
    "peak_tflops_for_kernel",
    "load_events",
    "load_op_trace",
    "render_report",
    "tflops_str",
]
