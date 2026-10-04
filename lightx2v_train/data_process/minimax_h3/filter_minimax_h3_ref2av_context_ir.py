#!/usr/bin/env python3
"""Materialize an immutable image-count subset of completed Ref2AV Context-IR.

The condition encoder fingerprints its complete input JSONL.  Feeding it a
Context-IR file that is still being appended makes later resume impossible.
This selector therefore joins the successful and terminal-failure Context-IR
outputs against the fixed VLM story source, verifies terminal coverage for the
requested image counts, and atomically publishes only successful rows.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from pathlib import Path
from typing import Iterable

SCHEMA_VERSION = 1
IMAGE_KIND_ALIASES = {"image", "reference_image", "image_url"}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="Completed VLM raw_stories JSONL.")
    parser.add_argument("--success", type=Path, required=True, help="Context-IR success JSONL.")
    parser.add_argument("--failed", type=Path, required=True, help="Context-IR terminal-failure JSONL.")
    parser.add_argument("--output", type=Path, required=True, help="Immutable selected success JSONL.")
    parser.add_argument("--manifest", type=Path, help="Audit manifest; defaults beside --output.")
    parser.add_argument(
        "--image-counts",
        type=int,
        nargs="+",
        default=(1, 2, 3),
        help="Pure-image reference counts to retain (default: 1 2 3).",
    )
    parser.add_argument(
        "--expected-source-selected-count",
        type=int,
        help="Refuse a changed/incomplete VLM source selection.",
    )
    parser.add_argument(
        "--expected-source-count",
        action="append",
        default=[],
        metavar="N=COUNT",
        help="Expected VLM source count for one image-count bucket; repeatable.",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    args.image_counts = tuple(dict.fromkeys(args.image_counts))
    if not args.image_counts or any(value < 1 or value > 9 for value in args.image_counts):
        parser.error("--image-counts must contain integers in 1..9")
    if args.expected_source_selected_count is not None and args.expected_source_selected_count < 0:
        parser.error("--expected-source-selected-count must be non-negative")
    try:
        args.expected_source_counts = parse_expected_counts(args.expected_source_count)
    except ValueError as error:
        parser.error(str(error))
    unexpected = sorted(set(args.expected_source_counts) - set(args.image_counts))
    if unexpected:
        parser.error(f"--expected-source-count contains unselected image counts: {unexpected}")
    return args


def parse_expected_counts(values: Iterable[str]) -> dict[int, int]:
    counts: dict[int, int] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"expected N=COUNT, got {value!r}")
        raw_image_count, raw_count = value.split("=", 1)
        try:
            image_count = int(raw_image_count)
            count = int(raw_count)
        except ValueError as error:
            raise ValueError(f"expected integer N=COUNT, got {value!r}") from error
        if not 1 <= image_count <= 9 or count < 0:
            raise ValueError(f"invalid expected source count {value!r}")
        if image_count in counts:
            raise ValueError(f"duplicate expected source image count {image_count}")
        counts[image_count] = count
    return counts


def read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(path)
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON at {path}:{line_number}; the producer may still be writing: {error}") from error
            if not isinstance(row, dict):
                raise TypeError(f"{path}:{line_number} must contain a JSON object")
            row = dict(row)
            row["_audit_line_number"] = line_number
            rows.append(row)
    return rows


def sha256_file(path: Path, chunk_size: int = 4 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def file_identity(path: Path) -> dict:
    return {
        "path": str(path.resolve()),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def row_identity(row: dict, *, path: Path) -> str:
    value = row.get("sample_id")
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{path}:{row.get('_audit_line_number', '?')} requires a non-empty sample_id")
    return value.strip()


def reference_signature(row: dict, *, path: Path) -> tuple[int, bool]:
    references = row.get("references")
    if not isinstance(references, list) or not references:
        images = row.get("reference_images")
        videos = row.get("reference_videos", [])
        audios = row.get("reference_audios", [])
        if not isinstance(images, list) or not isinstance(videos, list) or not isinstance(audios, list):
            raise TypeError(f"{path}:{row.get('_audit_line_number', '?')} has invalid reference lists")
        return len(images), not videos and not audios

    image_count = 0
    image_only = True
    for reference_index, reference in enumerate(references, start=1):
        if not isinstance(reference, dict):
            raise TypeError(f"{path}:{row.get('_audit_line_number', '?')} references[{reference_index}] must be an object")
        kind = str(reference.get("kind", reference.get("type", reference.get("role", "")))).strip().lower()
        if kind in IMAGE_KIND_ALIASES:
            image_count += 1
        else:
            image_only = False
    declared_images = row.get("reference_images")
    if isinstance(declared_images, list) and len(declared_images) != image_count:
        raise ValueError(f"{path}:{row.get('_audit_line_number', '?')} reference_images count {len(declared_images)} disagrees with ordered references count {image_count}")
    return image_count, image_only


def orientation(row: dict) -> str:
    value = str(row.get("target_orientation", "")).strip().lower()
    if value in {"landscape", "portrait"}:
        return value
    height = int(row.get("target_height", 0) or 0)
    width = int(row.get("target_width", 0) or 0)
    if width > height:
        return "landscape"
    if height > width:
        return "portrait"
    return "unknown"


def clean_row(row: dict) -> dict:
    return {key: value for key, value in row.items() if key != "_audit_line_number"}


def index_selected_rows(
    rows: list[dict],
    *,
    path: Path,
    image_counts: set[int],
    require_enhanced_prompt: bool,
) -> tuple[dict[str, dict], Counter, Counter]:
    selected: dict[str, dict] = {}
    counts = Counter()
    orientations = Counter()
    for row in rows:
        image_count, image_only = reference_signature(row, path=path)
        if not image_only or image_count not in image_counts:
            continue
        identity = row_identity(row, path=path)
        if identity in selected:
            raise ValueError(f"Duplicate selected sample_id={identity!r} in {path}")
        if require_enhanced_prompt:
            prompt = row.get("enhanced_prompt")
            if not isinstance(prompt, str) or not prompt.strip():
                raise ValueError(f"Selected Context-IR success {identity!r} has no enhanced_prompt")
        selected[identity] = clean_row(row)
        counts[image_count] += 1
        orientations[orientation(row)] += 1
    return selected, counts, orientations


def atomic_write_jsonl(path: Path, rows: Iterable[dict], *, overwrite: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        digest = hashlib.sha256()
        with temporary.open("x", encoding="utf-8") as handle:
            for row in rows:
                encoded = (json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")
                handle.write(encoded.decode("utf-8"))
                digest.update(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        if path.exists() and not overwrite:
            if sha256_file(path) == digest.hexdigest():
                return
            raise FileExistsError(f"Output already exists with different content: {path}")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_write_json(path: Path, value: dict, *, overwrite: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        if path.exists() and not overwrite:
            if sha256_file(path) == sha256_file(temporary):
                return
            existing = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(existing, dict) and existing.get("selection_fingerprint") == value.get("selection_fingerprint"):
                # The aggregate Context-IR success/failure files may keep
                # growing with unrelated 4-9-image rows after this selected
                # subset is already terminal.  The immutable output and its
                # selected terminal identities are unchanged, so preserve the
                # first audit snapshot and let encoder resume remain idempotent.
                return
            raise FileExistsError(f"Manifest already exists with different content: {path}")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def build_selection(args: argparse.Namespace) -> dict:
    source_path = args.source.expanduser().resolve()
    success_path = args.success.expanduser().resolve()
    failed_path = args.failed.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    manifest_path = args.manifest.expanduser().resolve() if args.manifest is not None else output_path.with_suffix(output_path.suffix + ".manifest.json")
    image_counts = set(args.image_counts)

    source_rows = read_jsonl(source_path)
    success_rows = read_jsonl(success_path)
    failed_rows = read_jsonl(failed_path)
    source, source_counts, source_orientations = index_selected_rows(
        source_rows,
        path=source_path,
        image_counts=image_counts,
        require_enhanced_prompt=False,
    )
    successes, success_counts, success_orientations = index_selected_rows(
        success_rows,
        path=success_path,
        image_counts=image_counts,
        require_enhanced_prompt=True,
    )
    failures, failure_counts, failure_orientations = index_selected_rows(
        failed_rows,
        path=failed_path,
        image_counts=image_counts,
        require_enhanced_prompt=False,
    )

    expected_total = args.expected_source_selected_count
    if expected_total is not None and len(source) != expected_total:
        raise RuntimeError(f"VLM source selection is not the expected immutable set: actual={len(source)} expected={expected_total} counts={dict(sorted(source_counts.items()))}")
    for image_count, expected in args.expected_source_counts.items():
        actual = source_counts[image_count]
        if actual != expected:
            raise RuntimeError(f"VLM source image_count={image_count} has actual={actual}, expected={expected}")

    overlap = sorted(set(successes) & set(failures))
    if overlap:
        raise RuntimeError(f"Context-IR IDs appear in both success and failure outputs: {overlap[:10]}")
    unexpected = sorted((set(successes) | set(failures)) - set(source))
    if unexpected:
        raise RuntimeError(f"Context-IR selected IDs are absent from the fixed VLM source: {unexpected[:10]}")
    missing = sorted(set(source) - set(successes) - set(failures))
    if missing:
        raise RuntimeError(
            f"Context-IR is not terminal for every selected 1/2/3-image story: source={len(source)} succeeded={len(successes)} failed={len(failures)} missing={len(missing)} examples={missing[:10]}"
        )

    ordered_successes = [successes[identity] for identity in source if identity in successes]
    atomic_write_jsonl(output_path, ordered_successes, overwrite=args.overwrite)
    output_identity = file_identity(output_path)
    selection_fingerprint = hashlib.sha256(
        json.dumps(
            {
                "source_ids": list(source),
                "successful_output_sha256": output_identity["sha256"],
                "terminal_failed_sample_ids": sorted(failures),
                "image_counts": list(args.image_counts),
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "selection_fingerprint": selection_fingerprint,
        "selection": {
            "image_counts": list(args.image_counts),
            "require_image_only": True,
            "terminal_coverage_required": True,
        },
        "source": file_identity(source_path),
        "context_ir_success": file_identity(success_path),
        "context_ir_failed": file_identity(failed_path),
        "source_selected_rows": len(source),
        "successful_selected_rows": len(successes),
        "terminal_failed_selected_rows": len(failures),
        "source_image_counts": dict(sorted(source_counts.items())),
        "success_image_counts": dict(sorted(success_counts.items())),
        "failure_image_counts": dict(sorted(failure_counts.items())),
        "source_orientations": dict(sorted(source_orientations.items())),
        "success_orientations": dict(sorted(success_orientations.items())),
        "failure_orientations": dict(sorted(failure_orientations.items())),
        "terminal_failed_sample_ids": sorted(failures),
        "output": output_identity,
    }
    atomic_write_json(manifest_path, manifest, overwrite=args.overwrite)
    return manifest


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    manifest = build_selection(args)
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
