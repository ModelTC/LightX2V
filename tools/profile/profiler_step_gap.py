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
class ProfilerStepGap:
    step_id: int
    wall_us: float
    gpu_active_us: float
    raw_gpu_sum_us: float
    gpu_event_count: int
    gpu_event_counts: dict[str, int]

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
    gpu_events: list[GpuActivity] = []
    for ev in events:
        if ev.get("ph") != "X":
            continue
        cat = ev.get("cat")
        if cat == _STEP_CAT:
            step_id = parse_step_id(ev.get("name", ""))
            if step_id is not None:
                steps.append((step_id, event_interval(ev)))
        elif cat in _GPU_CATS and float(ev.get("dur", 0)) > 0:
            gpu_events.append(GpuActivity(event_interval(ev), cat))

    stats: list[ProfilerStepGap] = []
    for step_id, window in sorted(steps, key=lambda item: item[0]):
        contained = [event for event in gpu_events if is_contained(window, event.interval)]
        intervals = [event.interval for event in contained]
        counts = {cat: 0 for cat in sorted(_GPU_CATS)}
        for event in contained:
            counts[event.cat] += 1
        stats.append(
            ProfilerStepGap(
                step_id=step_id,
                wall_us=window.duration,
                gpu_active_us=sum(interval.duration for interval in merge_intervals(intervals)),
                raw_gpu_sum_us=sum(event.interval.duration for event in contained),
                gpu_event_count=len(contained),
                gpu_event_counts=counts,
            )
        )
    return stats


def format_event_counts(counts: dict[str, int]) -> str:
    return ", ".join(f"{key}={counts.get(key, 0)}" for key in sorted(_GPU_CATS))


def render_summary(trace_path: Path, stats: list[ProfilerStepGap], *, brief: bool = False) -> str:
    if not stats:
        return f"{trace_path}: no ProfilerStep#* events found in {_STEP_CAT}\n"

    if brief:
        lines = [f"Trace: {trace_path}", f"Step annotation category: {_STEP_CAT}"]
        for step in stats:
            lines.append(
                f"ProfilerStep#{step.step_id}: "
                f"wall={step.wall_us / 1000:.3f} ms, "
                f"gpu_active={step.gpu_active_us / 1000:.3f} ms, "
                f"self_gap={step.gap_us / 1000:.3f} ms, "
                f"raw_gpu_sum={step.raw_gpu_sum_us / 1000:.3f} ms, "
                f"{format_event_counts(step.gpu_event_counts)}"
            )
        return "\n".join(lines) + "\n"

    lines = [
        f"Trace: {trace_path}",
        f"Step annotation category: {_STEP_CAT}",
        f"{'Step':>6}  {'Wall':>10}  {'GPU act':>10}  {'Raw GPU':>10}  {'#ev':>6}  {'Gap':>10}  {'Gap%':>7}  GPU event counts",
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
            f"{format_event_counts(step.gpu_event_counts)}"
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
