#!/usr/bin/env python3
"""Check ProfilerStep idle gaps in PyTorch Profiler TensorBoard traces.

Reads ``*.pt.trace.json`` and compares each ``ProfilerStep#N`` wall time against
device-timeline GPU activity (``kernel``, ``gpu_memcpy``, ``gpu_memset`` events
that lie **fully inside** the step window).

ALERT means: the gap looks worth a manual look in TensorBoard (PYTORCH PROFILER).
It is **not** a verdict that something is broken — micro-benchmarks and CPU-heavy
paths can trigger it legitimately.

Heuristics (active / last step):
  - gap_ratio > 10%% (``--gap-threshold``), or
  - gap_us > gpu_event_count × 10us (``--gap-per-event-us``)

If ALERT fires, try more profiler warmup / pre-steps first; if gap stays high,
inspect CPU-side stalls (autotune, sync, Python) in the timeline.

Usage:
  python tools/profile/analyze_torch_trace_gap.py path/to/trace.pt.trace.json
  python tools/profile/analyze_torch_trace_gap.py prof_results/**/*.pt.trace.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

_PROFILER_STEP_RE = re.compile(r"^ProfilerStep#(\d+)$")
_GPU_ACTIVITY_CATS = frozenset({"kernel", "gpu_memcpy", "gpu_memset"})
_DEFAULT_GAP_PER_EVENT_US = 10.0


@dataclass(frozen=True)
class Interval:
    start: float
    end: float

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


@dataclass
class ProfilerStepStats:
    step_id: int
    wall_us: float
    gpu_span_us: float
    gpu_active_us: float
    gpu_event_count: int

    @property
    def gap_us(self) -> float:
        return max(0.0, self.wall_us - self.gpu_active_us)

    @property
    def gap_ratio(self) -> float:
        if self.wall_us <= 0:
            return 0.0
        return self.gap_us / self.wall_us

    @property
    def gpu_coverage(self) -> float:
        if self.wall_us <= 0:
            return 0.0
        return min(1.0, self.gpu_active_us / self.wall_us)

    def gap_budget_us(self, gap_per_event_us: float) -> float:
        return self.gpu_event_count * gap_per_event_us

    def should_alert(self, gap_ratio_threshold: float, gap_per_event_us: float) -> tuple[bool, list[str]]:
        reasons: list[str] = []
        if self.gap_ratio > gap_ratio_threshold:
            reasons.append(f"gap_ratio {self.gap_ratio:.1%} > {gap_ratio_threshold:.1%}")
        budget = self.gap_budget_us(gap_per_event_us)
        if self.gap_us > budget:
            reasons.append(
                f"gap {_format_us(self.gap_us)} > budget {_format_us(budget)} "
                f"({self.gpu_event_count} events × {gap_per_event_us:g}us)"
            )
        return bool(reasons), reasons


def _merge_intervals(intervals: Sequence[Interval]) -> list[Interval]:
    if not intervals:
        return []
    ordered = sorted(intervals, key=lambda x: x.start)
    merged: list[Interval] = [ordered[0]]
    for cur in ordered[1:]:
        prev = merged[-1]
        if cur.start <= prev.end:
            merged[-1] = Interval(prev.start, max(prev.end, cur.end))
        else:
            merged.append(cur)
    return merged


def _is_contained_in(window: Interval, event: Interval) -> bool:
    return event.start >= window.start and event.end <= window.end


def _parse_step_id(name: str) -> int | None:
    match = _PROFILER_STEP_RE.match(name)
    return int(match.group(1)) if match else None


def _collect_profiler_steps(events: Iterable[dict]) -> list[tuple[int, Interval, str]]:
    steps: list[tuple[int, Interval, str]] = []
    for ev in events:
        if ev.get("ph") != "X":
            continue
        if ev.get("cat") != "user_annotation":
            continue
        name = ev.get("name", "")
        step_id = _parse_step_id(name)
        if step_id is None:
            continue
        ts = float(ev["ts"])
        dur = float(ev["dur"])
        steps.append((step_id, Interval(ts, ts + dur), name))
    return sorted(steps, key=lambda x: x[0])


def _collect_gpu_activity_intervals(events: Iterable[dict]) -> list[Interval]:
    intervals: list[Interval] = []
    for ev in events:
        if ev.get("ph") != "X":
            continue
        if ev.get("cat") not in _GPU_ACTIVITY_CATS:
            continue
        ts = float(ev["ts"])
        dur = float(ev["dur"])
        if dur <= 0:
            continue
        intervals.append(Interval(ts, ts + dur))
    return intervals


def analyze_trace_events(events: Sequence[dict]) -> list[ProfilerStepStats]:
    steps = _collect_profiler_steps(events)
    gpu_events = _collect_gpu_activity_intervals(events)
    stats: list[ProfilerStepStats] = []

    for step_id, window, _name in steps:
        contained = [ev for ev in gpu_events if _is_contained_in(window, ev)]
        merged = _merge_intervals(contained)
        gpu_active = sum(x.duration for x in merged)
        if merged:
            gpu_span = max(x.end for x in merged) - min(x.start for x in merged)
        else:
            gpu_span = 0.0
        stats.append(
            ProfilerStepStats(
                step_id=step_id,
                wall_us=window.duration,
                gpu_span_us=gpu_span,
                gpu_active_us=gpu_active,
                gpu_event_count=len(contained),
            )
        )
    return stats


def _format_us(us: float) -> str:
    if us >= 1_000_000:
        return f"{us / 1_000_000:.3f}s"
    if us >= 1_000:
        return f"{us / 1_000:.3f}ms"
    return f"{us:.1f}us"


def analyze_trace_file(path: Path) -> list[ProfilerStepStats]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    events = payload.get("traceEvents", payload)
    if not isinstance(events, list):
        raise ValueError(f"Unexpected trace format in {path}")
    return analyze_trace_events(events)


def render_report(
    path: Path,
    stats: Sequence[ProfilerStepStats],
    *,
    gap_threshold: float,
    gap_per_event_us: float,
) -> str:
    if not stats:
        return f"{path}: no ProfilerStep#* events found.\n"

    lines = [
        f"Trace: {path}",
        f"Profiler steps: {len(stats)}",
        f"Gap budget: gpu_event_count × {gap_per_event_us:g}us",
        "",
        f"{'Step':>6}  {'Wall':>10}  {'GPU act':>10}  {'#ev':>6}  {'Budget':>10}  {'Gap':>10}  {'Gap%':>7}  {'Cov%':>7}",
    ]
    for s in stats:
        budget = s.gap_budget_us(gap_per_event_us)
        lines.append(
            f"{s.step_id:>6}  "
            f"{_format_us(s.wall_us):>10}  "
            f"{_format_us(s.gpu_active_us):>10}  "
            f"{s.gpu_event_count:>6}  "
            f"{_format_us(budget):>10}  "
            f"{_format_us(s.gap_us):>10}  "
            f"{100 * s.gap_ratio:6.1f}%  "
            f"{100 * s.gpu_coverage:6.1f}%"
        )

    active = stats[-1]
    alert, reasons = active.should_alert(gap_threshold, gap_per_event_us)
    budget = active.gap_budget_us(gap_per_event_us)
    lines.append("")
    lines.append(
        f"Active step (last): ProfilerStep#{active.step_id}  "
        f"gap={_format_us(active.gap_us)}  budget={_format_us(budget)}  "
        f"gap_ratio={active.gap_ratio:.1%}  gpu_coverage={active.gpu_coverage:.1%}  "
        f"gpu_events={active.gpu_event_count}"
    )

    if len(stats) >= 2:
        first = stats[0]
        delta = first.gap_ratio - active.gap_ratio
        lines.append(
            f"Warmup trend: ProfilerStep#{first.step_id} gap {first.gap_ratio:.1%} "
            f"-> #{active.step_id} gap {active.gap_ratio:.1%} (delta {delta:+.1%})"
        )

    if alert:
        lines.append("")
        lines.append(
            "ALERT: gap on the active step looks high — open TensorBoard "
            "(PYTORCH PROFILER) and check the GPU timeline."
        )
        lines.append("       This is a hint to inspect, not proof of a bug.")
        for reason in reasons:
            lines.append(f"  - {reason}")
        lines.append("Suggested checks (in order):")
        lines.append("  1. Increase profiler schedule wait/warmup (e.g. wait=1, warmup=3 or higher).")
        lines.append("  2. Add pre-steps outside the profiler session (JIT / autotune / cache build).")
        lines.append("  3. If gap remains high after (1)(2), inspect CPU-side gaps in the timeline.")
    else:
        lines.append("")
        lines.append(
            f"OK: gap_ratio {active.gap_ratio:.1%} <= {gap_threshold:.1%} "
            f"and gap {_format_us(active.gap_us)} <= budget {_format_us(budget)}."
        )

    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Analyze ProfilerStep idle gaps in .pt.trace.json files.")
    parser.add_argument("traces", nargs="+", type=Path, help="Path(s) to .pt.trace.json")
    parser.add_argument(
        "--gap-threshold",
        type=float,
        default=0.10,
        help="Flag active step when gap_ratio exceeds this value (default: 0.10).",
    )
    parser.add_argument(
        "--gap-per-event-us",
        type=float,
        default=_DEFAULT_GAP_PER_EVENT_US,
        help="Gap budget per contained GPU event in microseconds (default: 10).",
    )
    args = parser.parse_args(argv)

    for path in args.traces:
        stats = analyze_trace_file(path)
        print(render_report(path, stats, gap_threshold=args.gap_threshold, gap_per_event_us=args.gap_per_event_us), end="")

    return 0


if __name__ == "__main__":
    sys.exit(main())
