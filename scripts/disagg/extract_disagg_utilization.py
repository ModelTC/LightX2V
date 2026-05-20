#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


def _fmt_float3(value):
    try:
        return f"{float(value):.3f}"
    except (TypeError, ValueError):
        return ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract average GPU utilization rows from disagg_controller_metrics.json")
    parser.add_argument(
        "--metrics",
        default="/root/zht/LightX2V/save_results/disagg_controller_metrics.json",
        help="Input disagg metrics json path",
    )
    parser.add_argument(
        "--output",
        default="/root/zht/LightX2V/save_results/disagg_wan22_i2v_util.csv",
        help="Output csv path",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    metrics_path = Path(args.metrics)
    out_path = Path(args.output)

    if not metrics_path.is_file():
        raise FileNotFoundError(f"metrics file not found: {metrics_path}")

    with metrics_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    samples = payload.get("monitor_samples", [])
    if not isinstance(samples, list):
        raise ValueError(f"invalid metrics format: monitor_samples must be a list, got {type(samples)}")

    grouped_samples = defaultdict(list)
    for item in samples:
        if not isinstance(item, dict):
            continue
        if item.get("status") != "ok":
            continue
        sample_ts = item.get("sample_ts_from_global_start_s")
        if sample_ts is None:
            continue
        try:
            grouped_samples[float(sample_ts)].append(item)
        except (TypeError, ValueError):
            continue

    rows = []
    for sample_ts in sorted(grouped_samples):
        group = grouped_samples[sample_ts]
        gpu_utils = []
        mem_utils = []
        for item in group:
            gpu_util = item.get("gpu_utilization")
            mem_used = item.get("gpu_memory_used_mb")
            mem_total = item.get("gpu_memory_total_mb")
            if gpu_util is not None:
                gpu_utils.append(float(gpu_util))
            if mem_used is not None and mem_total:
                mem_utils.append(float(mem_used) / float(mem_total) * 100.0)

        if not gpu_utils or not mem_utils:
            continue

        rows.append(
            {
                "time_from_start_s": _fmt_float3(sample_ts),
                "avg_gpu_utilization": _fmt_float3(sum(gpu_utils) / len(gpu_utils)),
                "avg_gpu_memory_occupancy_rate": _fmt_float3(sum(mem_utils) / len(mem_utils)),
            }
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "time_from_start_s",
                "avg_gpu_utilization",
                "avg_gpu_memory_occupancy_rate",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(f"wrote {len(rows)} rows to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
