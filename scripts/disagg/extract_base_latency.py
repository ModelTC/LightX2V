#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def _fmt_float3(value):
    try:
        return f"{float(value):.3f}"
    except (TypeError, ValueError):
        return ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract baseline latency rows from baseline_controller_metrics.json")
    parser.add_argument(
        "--metrics",
        default="/root/zht/LightX2V/save_results/baseline_controller_metrics.json",
        help="Input baseline metrics json path",
    )
    parser.add_argument(
        "--output",
        default="/root/zht/LightX2V/save_results/base_wan22_i2v.csv",
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

    requests = payload.get("requests", [])
    if not isinstance(requests, list):
        raise ValueError(f"invalid metrics format: requests must be a list, got {type(requests)}")

    global_start_ts = None
    for item in requests:
        if isinstance(item, dict) and item.get("client_send_ts") is not None:
            ts = float(item["client_send_ts"])
            global_start_ts = ts if global_start_ts is None else min(global_start_ts, ts)

    rows = []
    for item in requests:
        if not isinstance(item, dict):
            continue

        finish_rel = item.get("elapsed_from_global_start_s")
        if finish_rel is None:
            finish_ts = item.get("finish_ts")
            if finish_ts is not None and global_start_ts is not None:
                finish_rel = float(finish_ts) - float(global_start_ts)

        row = {
            "finish_time_from_global_start_s": _fmt_float3(finish_rel),
            "e2e_latency_s": _fmt_float3(item.get("e2e_latency_s")),
        }
        rows.append(row)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["finish_time_from_global_start_s", "e2e_latency_s"])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(f"wrote {len(rows)} rows to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
