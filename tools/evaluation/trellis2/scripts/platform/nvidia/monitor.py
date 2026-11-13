#!/usr/bin/env python3
"""Emit the normalized device snapshot consumed by evaluate.py."""

import argparse
import csv
import io
import json
import subprocess


def query(*args):
    return subprocess.check_output(["nvidia-smi", *args], text=True)


def number(value):
    value = value.strip()
    if value in ("", "N/A", "[Not Supported]"):
        return None
    return float(value)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--card", type=int, required=True)
    args = parser.parse_args()
    fields = "index,uuid,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw"
    rows = list(csv.reader(io.StringIO(query(f"--query-gpu={fields}", "--format=csv,noheader,nounits"))))
    row = next((row for row in rows if int(row[0].strip()) == args.card), None)
    if row is None:
        raise SystemExit(f"GPU index {args.card} was not reported by nvidia-smi")
    processes = query("--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader,nounits")
    pids = []
    for item in csv.reader(io.StringIO(processes)):
        if len(item) >= 2 and item[0].strip() == row[1].strip():
            pids.append(int(item[1].strip()))
    values = [number(x) for x in row[2:]]
    result = {
        "card": args.card,
        "util_percent": int(values[0]) if values[0] is not None else None,
        "memory_mib": int(values[1]) if values[1] is not None else None,
        "capacity_mib": int(values[2]) if values[2] is not None else None,
        "temperature_c": int(values[3]) if values[3] is not None else None,
        "power_w": values[4],
        "pids": sorted(set(pids)),
    }
    print(json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    main()
