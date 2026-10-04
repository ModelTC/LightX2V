#!/usr/bin/env python3
"""Rebase absolute paths in MiniMax-H3 JSON/JSONL cache manifests.

The condition ``.pt`` payloads are intentionally left untouched: training
loads their cached tensors, while absolute media/model strings inside
``cache_metadata`` are provenance only. The JSONL ``condition_path`` values,
however, are used by the dataset loader and must point at the new cache root.
"""

from __future__ import annotations

import argparse
import json
import os
import stat
import tempfile
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Rebase absolute paths in MiniMax-H3 JSON and JSONL manifests.")
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--old-prefix", required=True)
    parser.add_argument("--new-prefix", required=True)
    parser.add_argument(
        "--write",
        action="store_true",
        help="Atomically replace changed files. Without this flag, only report a dry run.",
    )
    parser.add_argument(
        "--verify-condition-paths",
        action="store_true",
        help="Fail if any rebased condition_path is missing or empty.",
    )
    return parser.parse_args()


def normalized_prefix(value: str, name: str) -> str:
    value = value.rstrip("/")
    if not value or not value.startswith("/"):
        raise ValueError(f"{name} must be a non-root absolute path, got {value!r}.")
    return value


def rebase_value(value, old_prefix: str, new_prefix: str):
    replacements = 0
    if isinstance(value, str):
        if value == old_prefix or value.startswith(old_prefix + "/"):
            return new_prefix + value[len(old_prefix) :], 1
        return value, 0
    if isinstance(value, list):
        result = []
        for item in value:
            item, changed = rebase_value(item, old_prefix, new_prefix)
            result.append(item)
            replacements += changed
        return result, replacements
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            rebased_key = key
            if isinstance(key, str) and (key == old_prefix or key.startswith(old_prefix + "/")):
                rebased_key = new_prefix + key[len(old_prefix) :]
                replacements += 1
            item, changed = rebase_value(item, old_prefix, new_prefix)
            if rebased_key in result:
                raise ValueError(f"Path rebasing creates duplicate JSON key {rebased_key!r}.")
            result[rebased_key] = item
            replacements += changed
        return result, replacements
    return value, 0


def check_condition_path(value, source: Path, line_number: int):
    if not isinstance(value, dict) or "condition_path" not in value:
        return 0
    path = Path(value["condition_path"])
    if not path.is_absolute():
        path = source.parent / path
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(f"Missing/empty condition_path at {source}:{line_number}: {path}")
    return 1


def temporary_path(path: Path):
    handle = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    )
    return handle, Path(handle.name)


def process_jsonl(path, old_prefix, new_prefix, write, verify):
    replacements = rows = verified = 0
    output = temp_path = None
    if write:
        output, temp_path = temporary_path(path)
    try:
        with path.open("r", encoding="utf-8") as source:
            for line_number, line in enumerate(source, 1):
                if not line.strip():
                    if output is not None:
                        output.write(line)
                    continue
                value = json.loads(line)
                value, changed = rebase_value(value, old_prefix, new_prefix)
                replacements += changed
                rows += 1
                if verify:
                    verified += check_condition_path(value, path, line_number)
                if output is not None:
                    output.write(json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n")
        if output is not None:
            output.flush()
            os.fsync(output.fileno())
            output.close()
            if replacements:
                os.chmod(temp_path, stat.S_IMODE(path.stat().st_mode))
                os.replace(temp_path, path)
                temp_path = None
        return rows, replacements, verified
    finally:
        if output is not None and not output.closed:
            output.close()
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


def process_json(path, old_prefix, new_prefix, write, verify):
    with path.open("r", encoding="utf-8") as source:
        value = json.load(source)
    value, replacements = rebase_value(value, old_prefix, new_prefix)
    verified = check_condition_path(value, path, 1) if verify else 0
    if write and replacements:
        output, temp_path = temporary_path(path)
        try:
            json.dump(value, output, ensure_ascii=False, indent=2)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
            output.close()
            os.chmod(temp_path, stat.S_IMODE(path.stat().st_mode))
            os.replace(temp_path, path)
            temp_path = None
        finally:
            if not output.closed:
                output.close()
            if temp_path is not None:
                temp_path.unlink(missing_ok=True)
    return 1, replacements, verified


def main():
    args = parse_args()
    root = args.root.expanduser().resolve()
    if not root.is_dir():
        raise NotADirectoryError(root)
    old_prefix = normalized_prefix(args.old_prefix, "--old-prefix")
    new_prefix = normalized_prefix(args.new_prefix, "--new-prefix")
    if old_prefix == new_prefix:
        raise ValueError("--old-prefix and --new-prefix must differ.")

    paths = sorted(path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in {".json", ".jsonl"})
    if not paths:
        raise FileNotFoundError(f"No JSON/JSONL files found below {root}")

    files_changed = total_rows = total_replacements = total_verified = 0
    for path in paths:
        processor = process_jsonl if path.suffix.lower() == ".jsonl" else process_json
        rows, replacements, verified = processor(
            path,
            old_prefix,
            new_prefix,
            args.write,
            args.verify_condition_paths,
        )
        total_rows += rows
        total_replacements += replacements
        total_verified += verified
        files_changed += bool(replacements)
        print(
            json.dumps(
                {
                    "file": str(path),
                    "rows": rows,
                    "replacements": replacements,
                },
                ensure_ascii=False,
            )
        )

    print(
        json.dumps(
            {
                "mode": "write" if args.write else "dry-run",
                "root": str(root),
                "files_scanned": len(paths),
                "files_changed": files_changed,
                "rows": total_rows,
                "replacements": total_replacements,
                "condition_paths_verified": total_verified,
                "pt_payloads_modified": 0,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
