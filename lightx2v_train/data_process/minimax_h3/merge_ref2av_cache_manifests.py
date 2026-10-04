#!/usr/bin/env python3
"""Merge Ref2AV cache manifests from multiple filesystems.

The source manifests use paths relative to their own cache roots.  The merged
manifest writes absolute condition paths so LatentDataset can consume the two
physical roots as one logical dataset.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

REQUIRED_KEYS = {"condition_path", "target_height", "target_width", "num_frames"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--expected-count", type=int)
    parser.add_argument("manifests", nargs="+", type=Path)
    args = parser.parse_args()
    if args.expected_count is not None and args.expected_count <= 0:
        parser.error("--expected-count must be positive")
    return args


def read_rows(manifest: Path) -> list[dict]:
    manifest = manifest.expanduser().resolve()
    if not manifest.is_file():
        raise FileNotFoundError(f"Ref2AV cache manifest not found: {manifest}")

    rows = []
    with manifest.open("r", encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict) or not REQUIRED_KEYS <= set(row):
                raise ValueError(f"{manifest}:{line_number} must contain at least {sorted(REQUIRED_KEYS)}")
            condition = Path(row["condition_path"]).expanduser()
            if not condition.is_absolute():
                condition = manifest.parent / condition
            condition = condition.resolve()
            if not condition.is_file():
                raise FileNotFoundError(f"Missing condition cache referenced by {manifest}:{line_number}: {condition}")
            rows.append({**row, "condition_path": str(condition)})
    return rows


def main() -> int:
    args = parse_args()
    output = args.output.expanduser().resolve()
    merged = []
    seen = set()
    seen_names = {}
    counts = {}
    for source in args.manifests:
        source = source.expanduser().resolve()
        rows = read_rows(source)
        counts[str(source)] = len(rows)
        for row in rows:
            condition = row["condition_path"]
            if condition in seen:
                raise ValueError(f"Duplicate condition cache across manifests: {condition}")
            seen.add(condition)
            logical_name = Path(condition).name
            if logical_name in seen_names:
                raise ValueError(f"Duplicate logical condition sample across manifests: name={logical_name}, first={seen_names[logical_name]}, duplicate={condition}")
            seen_names[logical_name] = condition
            merged.append(row)

    merged.sort(key=lambda row: Path(row["condition_path"]).name)
    if args.expected_count is not None and len(merged) != args.expected_count:
        raise ValueError(f"Merged row count is {len(merged)}, expected {args.expected_count}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp.{os.getpid()}")
    try:
        with temporary.open("w", encoding="utf-8") as output_file:
            for row in merged:
                output_file.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            output_file.flush()
            os.fsync(output_file.fileno())
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)

    print(f"Merged rows={len(merged)} sources={counts} output={output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
