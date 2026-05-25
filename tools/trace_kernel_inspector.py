#!/usr/bin/env python3
"""
Inspect PyTorch Profiler Chrome Trace JSON (.json / .pt.trace.json).

Subcommands:
  stat   - Aggregate GPU kernel stats (count / total / avg self-time).
  trace  - Chronological GPU events with correlation / External id handles.
  stack  - Full call chain for a correlation or External id.
  list   - List ProfilerStep names and gpu_user_annotation windows in a trace.

Examples:
  python tools/trace_kernel_inspector.py stat save_results/trace.json
  python tools/trace_kernel_inspector.py stat save_results/trace.json --short-name --sort total
  python tools/trace_kernel_inspector.py trace save_results/trace.json --output events.csv
  python tools/trace_kernel_inspector.py stack save_results/trace.json --id 386475
  python tools/trace_kernel_inspector.py list save_results/trace.json
  python tools/trace_kernel_inspector.py stat save_results/trace.json --window my_region
    # --window filters by torch.profiler.record_function name (gpu_user_annotation bounds)
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from typing import Dict, List, Optional, Tuple


def simplify_kernel_name(name: str) -> str:
    """Heuristically shorten long CUDA kernel names while keeping them distinguishable."""
    original = name

    if name.startswith("void "):
        name = name[5:]

    if name.startswith("at::native::(anonymous namespace)::"):
        name = name[len("at::native::(anonymous namespace)::") :]
    elif name.startswith("at::native::"):
        name = name[len("at::native::") :]

    for prefix in ("flashinfer::", "cutlass::"):
        if name.startswith(prefix):
            name = name[len(prefix) :]
            break

    m = re.search(r"vectorized_elementwise_kernel<\d+,\s*([^,]+)", original)
    if m:
        functor = m.group(1)
        functor = (
            functor.replace("at::native::", "")
            .replace("CUDAFunctor_", "")
            .replace("BinaryFunctor", "Bin")
            .replace("c10::BFloat16, c10::BFloat16, c10::BFloat16, ", "")
            .replace("binary_internal::", "")
            .replace("MulFunctor<float>", "Mul")
            .replace("GeluCUDAKernelImpl(...", "Gelu")
        )
        if "Gelu" in original:
            return "vectorized_elementwise_kernel(Gelu)"
        if "CUDAFunctor_add" in original:
            return "vectorized_elementwise_kernel(Add)"
        if "MulFunctor" in original:
            return "vectorized_elementwise_kernel(Mul)"
        return f"vectorized_elementwise_kernel({functor})"

    if "elementwise_kernel" in original and "gpu_kernel_impl_nocast" in original:
        if "MulFunctor" in original:
            return "elementwise_kernel(Mul)"
        if "CUDAFunctor_add" in original:
            return "elementwise_kernel(add)"
        if "GeluCUDAKernelImpl" in original:
            return "elementwise_kernel(Gelu)"
        return "elementwise_kernel"

    m = re.search(r"FlashAttn\w+", original)
    if m:
        return f"cutlass::{m.group(0)}"

    if "<" in name:
        name = name[: name.index("<")]

    if len(name) > 70:
        name = name[:67] + "..."

    return name


class KernelInspector:
    GPU_CATS = {"kernel", "gpu_memcpy", "gpu_memset", "gpu_user_annotation"}

    def __init__(self, path: str):
        self.path = path
        with open(path, "r", encoding="utf-8") as f:
            self.data = json.load(f)
        self.events = self.data.get("traceEvents", [])

        self._by_external_id: Dict[int, List[dict]] = defaultdict(list)
        self._by_correlation: Dict[int, List[dict]] = defaultdict(list)
        self._gpu_x_events: List[dict] = []
        self._pf_by_id: Dict[int, dict] = {}
        self._all_pfs: List[dict] = []

        for ev in self.events:
            args = ev.get("args", {})
            ext_id = args.get("External id")
            corr = args.get("correlation")
            if ext_id is not None:
                self._by_external_id[int(ext_id)].append(ev)
            if corr is not None:
                self._by_correlation[int(corr)].append(ev)
            if ev.get("cat") in self.GPU_CATS and ev.get("ph") == "X":
                self._gpu_x_events.append(ev)
            if ev.get("cat") == "python_function" and ev.get("ph") == "X":
                pid = args.get("Python id")
                if pid is not None:
                    self._pf_by_id[pid] = ev
                    self._all_pfs.append(ev)

        self._compute_self_durs()

    def _compute_self_durs(self) -> None:
        grouped: Dict[tuple, List[dict]] = defaultdict(list)
        for ev in self._gpu_x_events:
            grouped[(ev["pid"], ev["tid"])].append(ev)
        for evs in grouped.values():
            evs.sort(key=lambda e: e["ts"])
            stack: List[list] = []
            for ev in evs:
                while stack and stack[-1][0]["ts"] + stack[-1][0]["dur"] <= ev["ts"]:
                    parent, children_dur = stack.pop()
                    parent["self_dur"] = parent["dur"] - children_dur
                if stack:
                    stack[-1][1] += ev["dur"]
                stack.append([ev, 0.0])
            while stack:
                parent, children_dur = stack.pop()
                parent["self_dur"] = parent["dur"] - children_dur

    def list_profiler_steps(self) -> List[str]:
        names = {
            ev["name"]
            for ev in self.events
            if ev.get("ph") == "X" and "ProfilerStep" in str(ev.get("name", ""))
        }
        return sorted(names, key=lambda n: (len(n), n))

    def list_gpu_windows(self) -> List[str]:
        names = {
            ev["name"]
            for ev in self.events
            if ev.get("ph") == "X" and ev.get("cat") == "gpu_user_annotation"
        }
        return sorted(names)

    def find_profiler_step_window(self, step_name: str) -> Optional[Tuple[float, float]]:
        for cat in ("gpu_user_annotation", "user_annotation"):
            for ev in self.events:
                if ev.get("ph") == "X" and ev.get("name") == step_name and ev.get("cat") == cat:
                    return (ev["ts"], ev["ts"] + ev["dur"])
        return None

    def resolve_step_name(self, step_name: Optional[str]) -> Optional[str]:
        if step_name:
            return step_name
        steps = self.list_profiler_steps()
        for candidate in reversed(steps):
            if self.find_profiler_step_window(candidate):
                return candidate
        return None

    def _find_gpu_annotation_window(
        self,
        anno_name: str,
        step_bounds: Optional[Tuple[float, float]] = None,
    ) -> Optional[Tuple[float, float]]:
        """Use gpu_user_annotation bounds only (not CPU user_annotation)."""
        candidates = []
        for ev in self.events:
            if ev.get("ph") != "X" or ev.get("name") != anno_name:
                continue
            if ev.get("cat") != "gpu_user_annotation":
                continue
            if step_bounds is not None and (
                ev["ts"] < step_bounds[0] or ev["ts"] > step_bounds[1]
            ):
                continue
            candidates.append(ev)
        if not candidates:
            return None
        ev = max(candidates, key=lambda e: e["dur"])
        return (ev["ts"], ev["ts"] + ev["dur"])

    def _get_window_events(
        self,
        step_name: Optional[str] = None,
        window_name: Optional[str] = None,
    ) -> List[dict]:
        resolved_step = self.resolve_step_name(step_name)
        if window_name:
            step_bounds = self.find_profiler_step_window(resolved_step) if resolved_step else None
            bounds = self._find_gpu_annotation_window(window_name, step_bounds)
            if bounds is None:
                return []
            start, end = bounds
            return [
                ev
                for ev in self._gpu_x_events
                if ev.get("cat") != "gpu_user_annotation" and start <= ev["ts"] <= end
            ]

        if not resolved_step:
            return []

        step_ev = None
        for ev in self.events:
            if (
                ev.get("ph") == "X"
                and ev.get("name") == resolved_step
                and ev.get("cat") == "gpu_user_annotation"
            ):
                step_ev = ev
                break
        if step_ev is None:
            return []

        pid, tid = step_ev["pid"], step_ev["tid"]
        annos = [
            ev
            for ev in self.events
            if ev.get("cat") == "gpu_user_annotation"
            and ev.get("ph") == "X"
            and ev.get("pid") == pid
            and ev.get("tid") == tid
        ]
        annos.sort(key=lambda e: e["ts"])
        if not annos:
            return []

        start = annos[0]["ts"]
        end = max(ev["ts"] + ev["dur"] for ev in annos)
        return [ev for ev in self._gpu_x_events if start <= ev["ts"] <= end]

    def _window_label(self, step_name: Optional[str], window_name: Optional[str]) -> str:
        if window_name:
            return f"window={window_name}"
        return f"step={step_name or 'auto'}"

    def cmd_list(self) -> None:
        steps = self.list_profiler_steps()
        windows = self.list_gpu_windows()
        print(f"Trace: {self.path}")
        print("\nProfiler steps:")
        for name in steps:
            print(f"  {name}")
        print("\nGPU annotation windows (for --window):")
        for name in windows:
            print(f"  {name}")

    def cmd_stat(
        self,
        sort_by: str = "total",
        short_name: bool = False,
        out_path: Optional[str] = None,
        step_name: Optional[str] = None,
        window_name: Optional[str] = None,
    ) -> None:
        resolved_step = self.resolve_step_name(step_name)
        events = self._get_window_events(step_name=resolved_step, window_name=window_name)
        if not events:
            label = self._window_label(resolved_step, window_name)
            print(f"No GPU events in {label}.", file=sys.stderr)
            self._print_hints()
            return

        stats = defaultdict(lambda: {"count": 0, "total_self_us": 0.0})
        for ev in events:
            name = simplify_kernel_name(ev["name"]) if short_name else ev["name"]
            self_dur = ev.get("self_dur", ev["dur"])
            stats[name]["count"] += 1
            stats[name]["total_self_us"] += self_dur

        rows = []
        for name, s in stats.items():
            rows.append((name, s["count"], s["total_self_us"], s["total_self_us"] / s["count"]))

        if sort_by == "count":
            rows.sort(key=lambda x: -x[1])
        elif sort_by == "avg":
            rows.sort(key=lambda x: -x[3])
        else:
            rows.sort(key=lambda x: -x[2])

        fh = open(out_path, "w", newline="", encoding="utf-8") if out_path else sys.stdout
        writer = csv.writer(fh)
        writer.writerow(["Name", "Count", "TotalSelfTime_us", "AvgSelfTime_us"])
        for name, cnt, tot, avg in rows:
            writer.writerow([name, cnt, f"{tot:.1f}", f"{avg:.1f}"])
        total_cnt = sum(s["count"] for s in stats.values())
        total_self = sum(s["total_self_us"] for s in stats.values())
        writer.writerow(["TOTAL", total_cnt, f"{total_self:.1f}", f"{total_self / total_cnt:.1f}"])
        if out_path:
            fh.close()
            print(f"Wrote CSV to {out_path}")

    def cmd_trace(
        self,
        short_name: bool = False,
        out_path: Optional[str] = None,
        step_name: Optional[str] = None,
        window_name: Optional[str] = None,
    ) -> None:
        resolved_step = self.resolve_step_name(step_name)
        events = self._get_window_events(step_name=resolved_step, window_name=window_name)
        if not events:
            label = self._window_label(resolved_step, window_name)
            print(f"No GPU events in {label}.", file=sys.stderr)
            self._print_hints()
            return

        events.sort(key=lambda e: e["ts"])
        fh = open(out_path, "w", newline="", encoding="utf-8") if out_path else sys.stdout
        writer = csv.writer(fh)
        writer.writerow(
            ["Idx", "Timestamp", "Category", "SelfTime_us", "Correlation", "ExternalId", "Name"]
        )
        for idx, ev in enumerate(events):
            args = ev.get("args", {})
            writer.writerow(
                [
                    idx,
                    f"{ev['ts']:.1f}",
                    ev.get("cat", "N/A"),
                    f"{ev.get('self_dur', ev.get('dur', 0)):.1f}",
                    args.get("correlation", ""),
                    args.get("External id", ""),
                    simplify_kernel_name(ev["name"]) if short_name else ev["name"],
                ]
            )
        if out_path:
            fh.close()
            print(f"Wrote CSV to {out_path}")
        else:
            print(f"\nTotal events: {len(events)}")
            print("Tip: use stack --id <Correlation|ExternalId> for Python call chains.")

    def cmd_stack(self, target_id: int) -> None:
        ext_events = self._by_external_id.get(target_id, [])
        corr_events = self._by_correlation.get(target_id, [])

        all_events: List[dict] = []
        seen = set()
        for ev in ext_events + corr_events:
            uid = id(ev)
            if uid not in seen:
                seen.add(uid)
                all_events.append(ev)

        for ev in list(all_events):
            args = ev.get("args", {})
            corr = args.get("correlation")
            ext = args.get("External id")
            if corr is not None:
                for sib in self._by_correlation.get(int(corr), []):
                    uid = id(sib)
                    if uid not in seen:
                        seen.add(uid)
                        all_events.append(sib)
            if ext is not None:
                for sib in self._by_external_id.get(int(ext), []):
                    uid = id(sib)
                    if uid not in seen:
                        seen.add(uid)
                        all_events.append(sib)

        if not all_events:
            print(f"No events found for id={target_id}")
            print("Collect trace with LIGHTX2V_TORCH_PROFILE_STACK=1 for richer Python stacks.")
            return

        all_events.sort(key=lambda e: (e.get("ts", 0), e.get("pid", 0), e.get("tid", 0)))

        gpu_kernels = [e for e in all_events if e.get("cat") == "kernel"]
        gpu_annos = [e for e in all_events if e.get("cat") == "gpu_user_annotation"]
        cuda_drivers = [e for e in all_events if e.get("cat") in ("cuda_driver", "cuda_runtime")]
        cpu_ops = [e for e in all_events if e.get("cat") == "cpu_op"]

        python_stack: List[dict] = []
        if cuda_drivers:
            cd = cuda_drivers[0]
            candidates = []
            for pf in self._all_pfs:
                if pf["pid"] == cd["pid"] and pf["tid"] == cd["tid"]:
                    if pf["ts"] <= cd["ts"] and pf["ts"] + pf["dur"] >= cd["ts"] + cd["dur"]:
                        candidates.append(pf)
            if candidates:
                innermost = min(candidates, key=lambda pf: pf["dur"])
                path = []
                cur = innermost["args"]["Python id"]
                while cur in self._pf_by_id:
                    path.append(self._pf_by_id[cur])
                    parent = self._pf_by_id[cur]["args"].get("Python parent id")
                    if parent is None or parent not in self._pf_by_id:
                        break
                    cur = parent
                python_stack = path

        print("=" * 70)
        print(f"Call Stack for id={target_id}")
        print("=" * 70)

        if gpu_kernels:
            print(f"[GPU KER] {gpu_kernels[0]['name']}")
        if gpu_annos:
            print(f"  └─[GPU ANN] {gpu_annos[0]['name']}")
        if cuda_drivers:
            print(f"      └─[CUDA DRV] {cuda_drivers[0]['name']}")

        if python_stack:
            for i, node in enumerate(python_stack):
                indent = "          " + "  " * i
                print(f"{indent}└─[PY] {node['name']}")
        elif cpu_ops:
            for op in cpu_ops:
                print(f"      └─[CPU OP] {op['name']}")

        print("\n" + "-" * 70)
        print("Raw related events:")
        print(f"{'Cat':<18s} {'Name':<50s} {'ts':>16s} {'dur(us)':>10s} {'corr':>10s} {'ext_id':>8s}")
        print("-" * 115)
        for ev in all_events:
            args = ev.get("args", {})
            print(
                f"{ev.get('cat', 'N/A')[:18]:<18s} {ev.get('name', 'N/A')[:50]:<50s} "
                f"{ev.get('ts', 0):16.1f} {ev.get('dur', 0):10.1f} "
                f"{str(args.get('correlation', '')):>10s} {str(args.get('External id', '')):>8s}"
            )

    def _print_hints(self) -> None:
        steps = self.list_profiler_steps()
        windows = self.list_gpu_windows()
        if steps:
            print("Available ProfilerStep names:", ", ".join(steps), file=sys.stderr)
        if windows:
            print("Available --window names:", ", ".join(windows), file=sys.stderr)
        print("Run: python tools/trace_kernel_inspector.py list <trace.json>", file=sys.stderr)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect PyTorch Profiler Chrome Trace JSON files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_list = subparsers.add_parser("list", help="List profiler steps and GPU annotation windows")
    p_list.add_argument("trace", help="Path to Chrome trace JSON")

    p_stat = subparsers.add_parser("stat", help="Aggregate statistics by GPU activity name")
    p_stat.add_argument("trace", help="Path to Chrome trace JSON")
    p_stat.add_argument("--sort", choices=["count", "total", "avg"], default="total")
    p_stat.add_argument("--short-name", action="store_true")
    p_stat.add_argument("--output", help="Write CSV output")
    p_stat.add_argument("--step", help="ProfilerStep name (default: auto-detect last active step)")
    p_stat.add_argument(
        "--window",
        help="Filter by torch.profiler.record_function / gpu_user_annotation name",
    )

    p_trace = subparsers.add_parser("trace", help="Chronological GPU activity list")
    p_trace.add_argument("trace", help="Path to Chrome trace JSON")
    p_trace.add_argument("--short-name", action="store_true")
    p_trace.add_argument("--output", help="Write CSV output")
    p_trace.add_argument("--step", help="ProfilerStep name (default: auto-detect)")
    p_trace.add_argument("--window", help="Filter by gpu_user_annotation name")

    p_stack = subparsers.add_parser("stack", help="Query call chain by correlation or External id")
    p_stack.add_argument("trace", help="Path to Chrome trace JSON")
    p_stack.add_argument("--id", type=int, required=True)

    args = parser.parse_args()
    inspector = KernelInspector(args.trace)

    if args.command == "list":
        inspector.cmd_list()
    elif args.command == "stat":
        inspector.cmd_stat(
            args.sort, args.short_name, args.output, getattr(args, "step", None), args.window
        )
    elif args.command == "trace":
        inspector.cmd_trace(
            args.short_name, args.output, getattr(args, "step", None), args.window
        )
    elif args.command == "stack":
        inspector.cmd_stack(args.id)


if __name__ == "__main__":
    main()
