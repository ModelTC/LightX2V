#!/usr/bin/env python3
"""Filter a Ref2AV condition-cache manifest by reference media kind.

The Ref2AV training manifest intentionally contains only condition paths and
target geometry.  Reference kinds live inside each ``condition_*.pt`` payload.
This utility uses ``torch.load(..., mmap=True)`` so it can inspect that small
piece of metadata without reading the large cached tensors into memory.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path

import torch

VALID_KINDS = ("image", "video", "audio")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Input cache metadata.jsonl.")
    parser.add_argument("--output", required=True, type=Path, help="Filtered output JSONL.")
    filters = parser.add_mutually_exclusive_group()
    filters.add_argument(
        "--only-kind",
        choices=VALID_KINDS,
        help="Keep rows whose non-empty reference list contains only this kind.",
    )
    filters.add_argument(
        "--exclude-kind",
        choices=VALID_KINDS,
        action="append",
        default=[],
        help="Drop rows containing this kind. Repeat to exclude multiple kinds.",
    )
    parser.add_argument(
        "--expected-input-count",
        type=int,
        help="Fail unless the input contains exactly this many non-empty rows.",
    )
    return parser.parse_args()


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise TypeError(f"{path}:{line_number} must contain a JSON object")
            yield line_number, value


def resolve_condition(row: dict, manifest: Path, line_number: int) -> Path:
    value = row.get("condition_path")
    if not isinstance(value, str) or not value:
        raise ValueError(f"{manifest}:{line_number} has no condition_path")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = manifest.parent / path
    path = path.resolve()
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(f"{manifest}:{line_number} points to a missing/empty condition cache: {path}")
    return path


def reference_signature(condition_path: Path) -> tuple[str, ...]:
    payload = torch.load(
        condition_path,
        map_location="cpu",
        weights_only=False,
        mmap=True,
    )
    positive = payload.get("conditioning", {}).get("positive") if isinstance(payload, dict) else None
    references = positive.get("references") if isinstance(positive, dict) else None
    if not isinstance(references, list) or not references:
        raise ValueError(f"Condition cache has no reference metadata: {condition_path}")

    kinds = []
    for reference_index, reference in enumerate(references, start=1):
        if not isinstance(reference, dict):
            raise TypeError(f"{condition_path} references[{reference_index - 1}] must be a dictionary")
        kind = str(reference.get("kind", "")).lower()
        if kind not in VALID_KINDS:
            raise ValueError(f"{condition_path} references[{reference_index - 1}] has invalid kind={kind!r}")
        kinds.append(kind)
    return tuple(kinds)


def main() -> int:
    args = parse_args()
    if args.only_kind is None and not args.exclude_kind:
        args.only_kind = "image"
    input_path = args.input.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    if input_path == output_path:
        raise ValueError("--output must differ from --input")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.tmp.{os.getpid()}")
    signatures: Counter[tuple[str, ...]] = Counter()
    input_rows = kept_rows = 0
    try:
        with temporary.open("w", encoding="utf-8") as output_file:
            for line_number, row in read_jsonl(input_path):
                input_rows += 1
                condition_path = resolve_condition(row, input_path, line_number)
                signature = reference_signature(condition_path)
                signatures[signature] += 1
                if args.exclude_kind:
                    keep = not any(kind in args.exclude_kind for kind in signature)
                else:
                    keep = all(kind == args.only_kind for kind in signature)
                if keep:
                    output_file.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
                    kept_rows += 1
            output_file.flush()
            os.fsync(output_file.fileno())

        if args.expected_input_count is not None and input_rows != args.expected_input_count:
            raise ValueError(f"Input row count is {input_rows}, expected {args.expected_input_count}")
        if kept_rows == 0:
            filter_description = f"excluding {args.exclude_kind}" if args.exclude_kind else f"with only kind={args.only_kind}"
            raise ValueError(f"No reference rows found {filter_description}")
        os.replace(temporary, output_path)
    finally:
        temporary.unlink(missing_ok=True)

    print(
        json.dumps(
            {
                "input": str(input_path),
                "output": str(output_path),
                "input_rows": input_rows,
                "kept_rows": kept_rows,
                "only_kind": args.only_kind,
                "excluded_kinds": args.exclude_kind,
                "reference_signatures": {"+".join(signature): count for signature, count in sorted(signatures.items())},
                "tensor_payloads_materialized": False,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
