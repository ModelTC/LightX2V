#!/usr/bin/env python3
"""Summarize ProfilerStep gaps from GPU timeline annotations."""

from __future__ import annotations

import argparse
import glob
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

_GPU_CATS = frozenset({"kernel", "gpu_memcpy", "gpu_memset"})
_STEP_CAT = "gpu_user_annotation"
_CPU_STEP_CAT = "user_annotation"
_SYNC_RUNTIME_NAMES = ("cudaStreamSynchronize", "cudaDeviceSynchronize")


@dataclass(frozen=True)
class Interval:
    start: float
    end: float

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


@dataclass(frozen=True)
class GpuActivity:
    interval: Interval
    cat: str


@dataclass(frozen=True)
class RuntimeSync:
    interval: Interval
    name: str


@dataclass(frozen=True)
class ProfilerStepGap:
    step_id: int
    window_category: str
    window_name: str
    wall_us: float
    gpu_active_us: float
    raw_gpu_sum_us: float
    gpu_event_count: int
    gpu_event_counts: dict[str, int]
    sync_event_counts: dict[str, int]

    @property
    def gap_us(self) -> float:
        return max(0.0, self.wall_us - self.gpu_active_us)

    @property
    def gap_ratio(self) -> float:
        if self.wall_us <= 0:
            return 0.0
        return self.gap_us / self.wall_us


def parse_step_id(name: str) -> int | None:
    m = re.match(r"^ProfilerStep#(\d+)$", name)
    return int(m.group(1)) if m else None


def load_events(path: Path) -> list[dict]:
    with path.open() as f:
        payload = json.load(f)
    return payload.get("traceEvents", payload)


def event_interval(ev: dict) -> Interval:
    start = float(ev["ts"])
    return Interval(start, start + float(ev.get("dur", 0)))


def is_contained(window: Interval, event: Interval) -> bool:
    return event.start >= window.start and event.end <= window.end


def is_started_in(window: Interval, event: Interval) -> bool:
    return window.start <= event.start < window.end


def merge_intervals(intervals: list[Interval]) -> list[Interval]:
    if not intervals:
        return []
    ordered = sorted(intervals, key=lambda item: item.start)
    merged = [ordered[0]]
    for cur in ordered[1:]:
        prev = merged[-1]
        if cur.start <= prev.end:
            merged[-1] = Interval(prev.start, max(prev.end, cur.end))
        else:
            merged.append(cur)
    return merged


def analyze_events(events: list[dict]) -> list[ProfilerStepGap]:
    steps: list[tuple[int, Interval]] = []
    cpu_steps: dict[int, Interval] = {}
    gpu_annotations: list[tuple[str, Interval]] = []
    gpu_events: list[GpuActivity] = []
    sync_events: list[RuntimeSync] = []
    for ev in events:
        if ev.get("ph") != "X":
            continue
        cat = ev.get("cat")
        name = ev.get("name", "")
        if cat == _STEP_CAT:
            gpu_annotations.append((name, event_interval(ev)))
            step_id = parse_step_id(name)
            if step_id is not None:
                steps.append((step_id, event_interval(ev)))
        elif cat == _CPU_STEP_CAT:
            step_id = parse_step_id(name)
            if step_id is not None:
                cpu_steps[step_id] = event_interval(ev)
        elif cat in _GPU_CATS and float(ev.get("dur", 0)) > 0:
            gpu_events.append(GpuActivity(event_interval(ev), cat))
        elif cat == "cuda_runtime" and name in _SYNC_RUNTIME_NAMES:
            sync_events.append(RuntimeSync(event_interval(ev), name))

    step_windows: list[tuple[int, Interval, str, str]] = [(step_id, window, _STEP_CAT, f"ProfilerStep#{step_id}") for step_id, window in steps]
    if not step_windows:
        for step_id, cpu_window in sorted(cpu_steps.items()):
            candidates = [(name, interval) for name, interval in gpu_annotations if is_started_in(cpu_window, interval)]
            if candidates:
                name, window = max(candidates, key=lambda item: item[1].duration)
                step_windows.append((step_id, window, _STEP_CAT, name))
            else:
                step_windows.append((step_id, cpu_window, _CPU_STEP_CAT, f"ProfilerStep#{step_id}"))

    stats: list[ProfilerStepGap] = []
    for step_id, window, window_category, window_name in sorted(step_windows, key=lambda item: item[0]):
        contained = [event for event in gpu_events if is_contained(window, event.interval)]
        intervals = [event.interval for event in contained]
        counts = {cat: 0 for cat in sorted(_GPU_CATS)}
        for event in contained:
            counts[event.cat] += 1
        sync_counts = {name: 0 for name in _SYNC_RUNTIME_NAMES}
        sync_window = cpu_steps.get(step_id)
        if sync_window is not None:
            for event in sync_events:
                if is_started_in(sync_window, event.interval):
                    sync_counts[event.name] += 1
        stats.append(
            ProfilerStepGap(
                step_id=step_id,
                window_category=window_category,
                window_name=window_name,
                wall_us=window.duration,
                gpu_active_us=sum(interval.duration for interval in merge_intervals(intervals)),
                raw_gpu_sum_us=sum(event.interval.duration for event in contained),
                gpu_event_count=len(contained),
                gpu_event_counts=counts,
                sync_event_counts=sync_counts,
            )
        )
    return stats


def format_event_counts(counts: dict[str, int]) -> str:
    return ", ".join(f"{key}={counts.get(key, 0)}" for key in sorted(_GPU_CATS))


def format_sync_counts(counts: dict[str, int]) -> str:
    return ", ".join(f"{key}={counts.get(key, 0)}" for key in _SYNC_RUNTIME_NAMES)


def render_summary(trace_path: Path, stats: list[ProfilerStepGap], *, brief: bool = False) -> str:
    if not stats:
        return f"{trace_path}: no ProfilerStep#* events found\n"

    window_sources = ", ".join(dict.fromkeys(f"{step.window_category}:{step.window_name}" for step in stats))

    if brief:
        lines = [f"Trace: {trace_path}", f"Step windows: {window_sources}"]
        for step in stats:
            lines.append(
                f"ProfilerStep#{step.step_id}: "
                f"wall={step.wall_us / 1000:.3f} ms, "
                f"gpu_active={step.gpu_active_us / 1000:.3f} ms, "
                f"self_gap={step.gap_us / 1000:.3f} ms, "
                f"raw_gpu_sum={step.raw_gpu_sum_us / 1000:.3f} ms, "
                f"{format_event_counts(step.gpu_event_counts)}, "
                f"{format_sync_counts(step.sync_event_counts)}"
            )
        return "\n".join(lines) + "\n"

    lines = [
        f"Trace: {trace_path}",
        f"Step windows: {window_sources}",
        f"{'Step':>6}  {'Wall':>10}  {'GPU act':>10}  {'Raw GPU':>10}  {'#ev':>6}  {'Gap':>10}  {'Gap%':>7}  GPU event counts  Sync API counts",
    ]
    for step in stats:
        lines.append(
            f"{step.step_id:>6}  "
            f"{step.wall_us / 1000:9.3f}ms  "
            f"{step.gpu_active_us / 1000:9.3f}ms  "
            f"{step.raw_gpu_sum_us / 1000:9.3f}ms  "
            f"{step.gpu_event_count:>6}  "
            f"{step.gap_us / 1000:9.3f}ms  "
            f"{100 * step.gap_ratio:6.1f}%  "
            f"{format_event_counts(step.gpu_event_counts)}  "
            f"{format_sync_counts(step.sync_event_counts)}"
        )
    return "\n".join(lines) + "\n"


def resolve_trace_paths(traces: Sequence[str]) -> list[Path]:
    paths: list[Path] = []
    for trace in traces:
        if any(ch in trace for ch in "*?["):
            matches = sorted(glob.glob(trace, recursive=True))
            if not matches:
                print(f"warning: no files match {trace}", file=sys.stderr)
            paths.extend(Path(match) for match in matches)
        else:
            paths.append(Path(trace))
    return paths


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Summarize ProfilerStep GPU timeline gaps from .pt.trace.json files.",
    )
    parser.add_argument("traces", nargs="+", help="Path(s) to .pt.trace.json")
    parser.add_argument("--brief", action="store_true", help="Print compact wall/gpu_active/self_gap/event-count lines.")
    parser.add_argument("--step-id", type=int, default=None, help="Only print one ProfilerStep#N.")
    args = parser.parse_args(argv)

    paths = resolve_trace_paths(args.traces)
    if not paths:
        print("error: no trace files to analyze", file=sys.stderr)
        return 0

    for path in paths:
        stats = analyze_events(load_events(path))
        if args.step_id is not None:
            stats = [step for step in stats if step.step_id == args.step_id]
        print(render_summary(path, stats, brief=args.brief), end="")

    return 0


if __name__ == "__main__":
    sys.exit(main())
