#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import csv
import re
from datetime import datetime
from pathlib import Path

WAIT_PATTERNS = [
    re.compile(r"^\[INFO\]\s+(\d{2}\s+\w{3}\s+\d{4}\s+\d{2}:\d{2}:\d{2}).*waiting workload configs on port=", re.IGNORECASE),
    re.compile(r"^\[(?:INFO|WARNING|ERROR|DEBUG|CRITICAL)\]\s+(\d{2}\s+\w{3}\s+\d{4}\s+\d{2}:\d{2}:\d{2}).*waiting workload configs on port=", re.IGNORECASE),
]
LAT_PATTERNS = [
    re.compile(r"^\[INFO\]\s+(\d{2}\s+\w{3}\s+\d{4}\s+\d{2}:\d{2}:\d{2}).*Latency summary room=(\d+) metrics=(\{.*\})"),
    re.compile(r"^\[(?:INFO|WARNING|ERROR|DEBUG|CRITICAL)\]\s+(\d{2}\s+\w{3}\s+\d{4}\s+\d{2}:\d{2}:\d{2}).*Latency summary room=(\d+) metrics=(\{.*\})"),
]
TS_FMT = "%d %b %Y %H:%M:%S"
LOGURU_TS_FMT = "%Y-%m-%d %H:%M:%S"


def _fmt_float3(value):
    try:
        return f"{float(value):.3f}"
    except (TypeError, ValueError):
        return value


def _match_any(patterns, line):
    for pattern in patterns:
        match = pattern.match(line)
        if match:
            return match
    return None


def _parse_timestamp(raw_ts: str):
    for fmt in (TS_FMT, LOGURU_TS_FMT):
        try:
            return datetime.strptime(raw_ts, fmt)
        except ValueError:
            pass
    raise ValueError(f"unsupported timestamp format: {raw_ts}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract latency summary rows relative to waiting workload log time")
    parser.add_argument(
        "--log",
        default="/root/zht/LightX2V/save_results/disagg_wan22_i2v_dynamic_controller.log",
        help="Controller log path",
    )
    parser.add_argument(
        "--output",
        default="/root/zht/LightX2V/save_results/disagg_wan22_i2v_dynamic_results.csv",
        help="Output table path",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    log_path = Path(args.log)
    out_path = Path(args.output)

    if not log_path.is_file():
        raise FileNotFoundError(f"log file not found: {log_path}")

    wait_ts = None
    rows = []
    metric_keys = []

    with log_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if wait_ts is None:
                m_wait = _match_any(WAIT_PATTERNS, line)
                if m_wait:
                    wait_ts = _parse_timestamp(m_wait.group(1))
                    continue

            m_lat = _match_any(LAT_PATTERNS, line)
            if not m_lat:
                continue

            ts = _parse_timestamp(m_lat.group(1))
            room = int(m_lat.group(2))
            metrics = ast.literal_eval(m_lat.group(3))
            if not isinstance(metrics, dict):
                continue

            if wait_ts is None:
                rel_s = "NA"
            else:
                rel_s = f"{int((ts - wait_ts).total_seconds())}s"

            if not metric_keys:
                metric_keys = list(metrics.keys())

            row = {
                "room": room,
                "latency_summary_ts": ts.strftime("%Y-%m-%d %H:%M:%S"),
                "relative_to_waiting_s": rel_s,
            }
            for key in metric_keys:
                value = metrics.get(key)
                row[key] = "" if value is None else _fmt_float3(value)
            rows.append(row)

    out_path.parent.mkdir(parents=True, exist_ok=True)

    header = ["room", "latency_summary_ts", "relative_to_waiting_s", *metric_keys]
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(f"wrote {len(rows)} rows to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
