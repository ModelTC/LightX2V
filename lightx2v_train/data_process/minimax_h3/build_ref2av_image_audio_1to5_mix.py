#!/usr/bin/env python3
"""Build one immutable Ref2AV Context-IR mix for match-cache encoding.

The mix contains every successful pure-image row with 1--5 images from the
20k image-only source, plus every successful image+audio row (and no reference
video) from the 10k multimodal source.  It does not copy media or rewrite
prompts; absolute/local reference paths already carried by each row remain the
source of truth for the condition encoder.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from pathlib import Path

IMAGE_ALIASES = {"image", "reference_image", "image_url"}
VIDEO_ALIASES = {"video", "reference_video", "video_url"}
AUDIO_ALIASES = {"audio", "reference_audio", "audio_url"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--image-only",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--image-audio",
        type=Path,
        required=True,
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--image-counts", type=int, nargs="+", default=(1, 2, 3, 4, 5))
    parser.add_argument("--expected-image-only", type=int)
    parser.add_argument("--expected-image-audio", type=int)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    args.image_counts = tuple(dict.fromkeys(args.image_counts))
    if not args.image_counts or any(not 1 <= value <= 9 for value in args.image_counts):
        parser.error("--image-counts must contain values in 1..9")
    return args


def read_jsonl(path: Path):
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON at {path}:{line_number}: {error}") from error
            if not isinstance(row, dict):
                raise TypeError(f"{path}:{line_number} must contain an object")
            yield row


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(4 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def reference_counts(row: dict) -> tuple[int, int, int]:
    references = row.get("references")
    if not isinstance(references, list) or not references:
        raise ValueError(f"sample {row.get('sample_id')!r} has no ordered references")
    counts = Counter()
    for index, reference in enumerate(references, start=1):
        if not isinstance(reference, dict):
            raise TypeError(f"sample {row.get('sample_id')!r} reference {index} is not an object")
        kind = str(reference.get("kind", reference.get("type", reference.get("role", "")))).strip().lower()
        if kind in IMAGE_ALIASES:
            counts["image"] += 1
        elif kind in VIDEO_ALIASES:
            counts["video"] += 1
        elif kind in AUDIO_ALIASES:
            counts["audio"] += 1
        else:
            raise ValueError(f"sample {row.get('sample_id')!r} reference {index} has unknown kind={kind!r}")
    return counts["image"], counts["video"], counts["audio"]


def validate_selected(row: dict, path: Path) -> str:
    sample_id = row.get("sample_id")
    if not isinstance(sample_id, str) or not sample_id.strip():
        raise ValueError(f"Selected row in {path} has no sample_id")
    if str(row.get("task", "")).strip().lower() not in {"ref2av", "ref2va"}:
        raise ValueError(f"Selected sample {sample_id!r} is not Ref2AV")
    if not isinstance(row.get("enhanced_prompt"), str) or not row["enhanced_prompt"].strip():
        raise ValueError(f"Selected sample {sample_id!r} has no enhanced_prompt")
    return sample_id.strip()


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
    raise ValueError(f"sample {row.get('sample_id')!r} has invalid target geometry")


def atomic_write(path: Path, rows: list[dict], overwrite: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        if path.exists() and not overwrite:
            if sha256_file(path) == sha256_file(temporary):
                return
            raise FileExistsError(f"Output exists with different content: {path}")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> int:
    args = parse_args()
    image_only_path = args.image_only.expanduser().resolve()
    image_audio_path = args.image_audio.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    allowed_counts = set(args.image_counts)

    selected: list[dict] = []
    seen_ids: set[str] = set()
    composition = Counter()
    by_image_count = Counter()
    by_orientation = Counter()

    for source_name, source_path in (
        ("image_only", image_only_path),
        ("image_audio", image_audio_path),
    ):
        for row in read_jsonl(source_path):
            images, videos, audios = reference_counts(row)
            keep = (source_name == "image_only" and images in allowed_counts and videos == 0 and audios == 0) or (
                source_name == "image_audio" and images in allowed_counts and videos == 0 and audios > 0
            )
            if not keep:
                continue
            sample_id = validate_selected(row, source_path)
            if sample_id in seen_ids:
                raise ValueError(f"Duplicate selected sample_id={sample_id!r} across sources")
            seen_ids.add(sample_id)
            selected.append(row)
            composition[source_name] += 1
            by_image_count[images] += 1
            by_orientation[orientation(row)] += 1

    expected = {
        "image_only": args.expected_image_only,
        "image_audio": args.expected_image_audio,
    }
    if any(count is not None and composition[name] != count for name, count in expected.items()):
        raise RuntimeError(f"Changed/incomplete input selection: actual={dict(composition)} expected={expected}")
    if set(by_image_count) != allowed_counts:
        raise RuntimeError(f"Selected image-count buckets differ from request: {dict(sorted(by_image_count.items()))}")

    atomic_write(output_path, selected, args.overwrite)
    manifest = {
        "schema_version": 1,
        "output": str(output_path),
        "output_rows": len(selected),
        "output_sha256": sha256_file(output_path),
        "sources": {
            "image_only": {"path": str(image_only_path), "sha256": sha256_file(image_only_path)},
            "image_audio": {"path": str(image_audio_path), "sha256": sha256_file(image_audio_path)},
        },
        "composition": dict(sorted(composition.items())),
        "image_counts": dict(sorted(by_image_count.items())),
        "orientations": dict(sorted(by_orientation.items())),
    }
    manifest_path = output_path.with_suffix(output_path.suffix + ".manifest.json")
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
