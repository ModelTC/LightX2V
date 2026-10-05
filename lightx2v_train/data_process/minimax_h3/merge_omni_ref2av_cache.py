#!/usr/bin/env python3
"""Publish a complete Omni condition-cache manifest after all 32 workers finish.

Only manifests, completion receipts and cache file existence are checked here;
the encoder already validated tensor payloads. This avoids rereading every
large .pt file over AFS just to merge a dataset. No torch/GPU dependency.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import tempfile
from collections import Counter
from contextlib import ExitStack
from pathlib import Path


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def records(path):
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"Expected an object at {path}:{number}")
            yield row


def atomic_lines(path, rows):
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False) as handle:
            temporary = Path(handle.name)
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def checked_index(row, total, shard_index, num_shards):
    index = row.get("source_index")
    if type(index) is not int or not 0 <= index < total or index % num_shards != shard_index:
        raise ValueError(f"Wrong/missing source_index={index!r} in shard {shard_index}")
    return index


def merge(output_dir, num_shards=32):
    output_dir = Path(output_dir).expanduser().resolve(strict=True)
    namespace = json.loads((output_dir / "preprocess_config.json").read_text(encoding="utf-8"))
    config = namespace["preprocess_config"]
    fingerprint = namespace["preprocess_fingerprint"]
    required_policy = {
        "dtype": "bf16",
        "reference_latent_dtype": "bf16",
        "reference_image_resize_mode": "match",
        "target_policy": "fixed-768p",
        "prompt_policy": "enhanced-or-original",
        "image_only": True,
    }
    for key, expected in required_policy.items():
        if config.get(key) != expected:
            raise ValueError(f"Omni cache merge requires {key}={expected!r}")
    selection = config["selection"]
    if selection.get("num_shards") != num_shards or selection.get("start_index") != 0 or selection.get("max_samples") is not None:
        raise ValueError("Merge requires all input rows (start=0, no max-samples) and matching num-shards")
    all_rows, all_failures = [], []
    seen, paths = set(), set()
    total = None
    for shard_index in range(num_shards):
        stem = f"metadata.shard-{shard_index:03d}-of-{num_shards:03d}"
        manifest = output_dir / f"{stem}.jsonl"
        failures = output_dir / f"{stem}.failed.jsonl"
        receipt_path = output_dir / f"{stem}.complete.json"
        if not receipt_path.is_file():
            raise FileNotFoundError(f"Shard {shard_index} is not complete: {receipt_path}")
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if receipt.get("preprocess_fingerprint") != fingerprint or receipt.get("shard_index") != shard_index or receipt.get("num_shards") != num_shards:
            raise ValueError(f"Mismatched completion receipt: {receipt_path}")
        if receipt.get("stage") not in {"all", "references"}:
            raise ValueError(f"Text-only shard is not a training cache: {receipt_path}")
        row_total = receipt.get("input_total_rows")
        if type(row_total) is not int or row_total < 1 or (total is not None and total != row_total):
            raise ValueError(f"Inconsistent input row count in {receipt_path}")
        total = row_total
        for path, field in ((manifest, "manifest_sha256"), (failures, "failures_sha256")):
            if digest(path) != receipt.get(field):
                raise ValueError(f"Changed shard file since completion: {path}")
        rows = list(records(manifest))
        failed = list(records(failures))
        expected_selected = len(range(shard_index, total, num_shards))
        if (receipt.get("selected_count"), receipt.get("completed_count"), receipt.get("failed_count")) != (expected_selected, len(rows), len(failed)):
            raise ValueError(f"Receipt counts do not match shard files: {receipt_path}")
        if len(rows) + len(failed) != expected_selected:
            raise ValueError(f"Incomplete selected row accounting: {receipt_path}")
        for row in rows + failed:
            index = checked_index(row, total, shard_index, num_shards)
            if index in seen:
                raise ValueError(f"Duplicate source_index={index}")
            seen.add(index)
        for row in rows:
            relative = Path(row["condition_path"])
            condition = (output_dir / relative).resolve()
            if relative.is_absolute() or output_dir / "conditions" not in condition.parents:
                raise ValueError(f"Unexpected condition path: {relative}")
            if str(relative) in paths:
                raise ValueError(f"Duplicate condition path: {relative}")
            if not condition.is_file():
                raise FileNotFoundError(condition)
            paths.add(str(relative))
            image_count = row.get("reference_image_count")
            if type(image_count) is not int or not 1 <= image_count <= 9 or row.get("ref_image_count") != image_count:
                raise ValueError(f"Invalid image-count metadata for {relative}")
            if row.get("reference_video_count") != 0 or row.get("reference_audio_count") != 0:
                raise ValueError(f"Non-image reference in {relative}")
            if row.get("num_frames") != 124 or (row.get("target_height"), row.get("target_width")) not in {(768, 1344), (1344, 768)}:
                raise ValueError(f"Wrong fixed target geometry for {relative}")
        all_rows.extend(rows)
        all_failures.extend(failed)
        print(f"shard={shard_index}/{num_shards} completed={len(rows)} failed={len(failed)}", flush=True)
    if len(seen) != total:
        raise ValueError(f"Not every source row is accounted for: {len(seen)}/{total}")
    if not all_rows:
        raise ValueError("No valid cache rows; inspect shard failure files")
    all_rows.sort(key=lambda row: row["source_index"])
    all_failures.sort(key=lambda row: row["source_index"])
    manifest = output_dir / "metadata.jsonl"
    completion = output_dir / "metadata.complete.json"
    # Remove the small publication marker first; a failed write cannot leave a
    # stale receipt authorizing a different manifest for training.
    completion.unlink(missing_ok=True)
    atomic_lines(manifest, all_rows)
    atomic_lines(output_dir / "failed.jsonl", all_failures)
    summary = {
        "preprocess_fingerprint": fingerprint,
        "manifest_sha256": digest(manifest),
        "num_shards": num_shards,
        "input_total_rows": total,
        "completed_count": len(all_rows),
        "failed_count": len(all_failures),
        "reference_image_counts": dict(sorted(Counter(row["ref_image_count"] for row in all_rows).items())),
        "prompt_sources": dict(Counter(row.get("prompt_source", "unknown") for row in all_rows)),
    }
    atomic_lines(completion, [summary])
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-shards", type=int, default=32)
    args = parser.parse_args()
    if args.num_shards != 32:
        parser.error("This four-node workflow uses exactly 32 shards")
    # Node launchers retain these locks while their children are alive. Merge
    # and encoding must never publish the same namespace concurrently.
    with ExitStack() as stack:
        for node in range(4):
            handle = stack.enter_context((args.output_dir / f".node-{node}.lock").open("a"))
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise SystemExit(f"Node {node} is still encoding; merge after all four nodes finish")
        merge(args.output_dir, args.num_shards)


if __name__ == "__main__":
    main()
