#!/usr/bin/env python3
"""Build 5-second H3 keyframe-task inputs for cache/API stages.

For each of I2AV, L2AV and FL2AV, this utility selects disjoint source rows
for original and Context-IR captions, keeps the selected caption byte-for-byte
apart from surrounding whitespace, and assigns the target canvas from the
actual, EXIF-corrected keyframe dimensions.  It never relabels a landscape
keyframe as portrait merely to balance bucket counts.

The output JSONL files still point at the original keyframes.  Both the prompt
cache builder and the teacher-video requester apply the same H3 keyframe image
preparation for the assigned canvas.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from collections import Counter
from pathlib import Path
from typing import Iterable

TASKS = ("i2av", "l2av", "fl2av")
TASK_ALIASES = {
    "i2v": "i2av",
    "i2va": "i2av",
    "i2av": "i2av",
    "l2v": "l2av",
    "l2va": "l2av",
    "l2av": "l2av",
    "flf2v": "fl2av",
    "flf2va": "fl2av",
    "flf2av": "fl2av",
    "fl2v": "fl2av",
    "fl2va": "fl2av",
    "fl2av": "fl2av",
}
IMAGE_FIELDS = {
    "i2av": ("first_frame",),
    "l2av": ("last_frame",),
    "fl2av": ("first_frame", "last_frame"),
}
GEOMETRY = {
    "landscape": {"target_height": 768, "target_width": 1344, "target_resolution": "16:9"},
    "portrait": {"target_height": 1344, "target_width": 768, "target_resolution": "9:16"},
}
NUM_FRAMES = 124
FPS = 24
GENERATION_DURATION = 5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--samples-per-task", type=int, default=9600)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--max-prompt-characters",
        type=int,
        default=7000,
        help="Reject condition captions longer than the H3 V2 video API text limit.",
    )
    parser.add_argument(
        "--square-policy",
        choices=("error", "landscape", "portrait", "hash"),
        default="error",
        help=("How to map genuinely square keyframes onto the two supported canvases. The default fails instead of silently changing their composition."),
    )
    parser.add_argument(
        "--task",
        action="append",
        choices=TASKS,
        dest="tasks",
        help="Build only this task; repeat as needed. Defaults to all three.",
    )
    parser.add_argument(
        "--source",
        action="append",
        default=[],
        metavar="TASK=JSONL",
        help="Source JSONL for one task; required for each selected task.",
    )
    parser.add_argument(
        "--base-mixed",
        action="append",
        default=[],
        metavar="TASK=JSONL",
        help=("Preserve the selected rows/variants from an existing DMD JSONL, replacing only API-incompatible captions. Required for each selected task."),
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.samples_per_task <= 0 or args.samples_per_task % 4:
        parser.error("--samples-per-task must be a positive multiple of four")
    if args.max_prompt_characters <= 0:
        parser.error("--max-prompt-characters must be positive")
    args.tasks = tuple(dict.fromkeys(args.tasks or TASKS))
    return args


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON at {path}:{line_number}: {error}") from error
            if not isinstance(row, dict):
                raise TypeError(f"Expected a JSON object at {path}:{line_number}")
            row["_source_row_number"] = line_number
            rows.append(row)
    return rows


def parse_sources(values: Iterable[str]) -> dict[str, Path]:
    sources = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"--source must be TASK=JSONL, got {value!r}")
        task, raw_path = value.split("=", 1)
        task = TASK_ALIASES.get(task.strip().lower(), task.strip().lower())
        if task not in TASKS:
            raise ValueError(f"Unsupported task {task!r}; expected one of {TASKS}")
        sources[task] = Path(raw_path).expanduser()
    return {task: path.expanduser().resolve() for task, path in sources.items()}


def parse_base_mixed(values: Iterable[str]) -> dict[str, Path]:
    paths = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"--base-mixed must be TASK=JSONL, got {value!r}")
        task, raw_path = value.split("=", 1)
        task = TASK_ALIASES.get(task.strip().lower(), task.strip().lower())
        if task not in TASKS:
            raise ValueError(f"Unsupported task {task!r}; expected one of {TASKS}")
        paths[task] = Path(raw_path).expanduser()
    return {task: path.expanduser().resolve() for task, path in paths.items()}


def normalized_prompt(value: str) -> str:
    return " ".join(value.split()).casefold()


def source_identity(task: str, row: dict) -> str:
    keys = ("metadata_id", "id", "source_line")
    for key in keys:
        value = row.get(key)
        if value not in (None, ""):
            return f"{task}:{key}:{value}"
    return f"{task}:row:{row['_source_row_number']}"


def resolve_image(value: object, source_path: Path, *, location: str, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{location} is missing non-empty {field}")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = source_path.parent / path
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{location} references missing {field}: {path}")
    return str(path)


def image_geometry(path: str) -> tuple[int, int]:
    try:
        from PIL import Image, ImageOps
    except ImportError as error:
        raise ImportError("Keyframe orientation inspection requires Pillow.") from error
    with Image.open(path) as opened:
        image = ImageOps.exif_transpose(opened)
        return int(image.width), int(image.height)


def classify_geometry(
    width: int,
    height: int,
    *,
    identity: str,
    square_policy: str,
) -> str:
    if width > height:
        return "landscape"
    if height > width:
        return "portrait"
    if square_policy == "error":
        raise ValueError(f"source_id={identity!r} has a genuinely square {width}x{height} keyframe; choose --square-policy explicitly instead of silently changing its composition")
    if square_policy in GEOMETRY:
        return square_policy
    digest = hashlib.sha256(identity.encode("utf-8")).digest()
    return "landscape" if digest[0] % 2 == 0 else "portrait"


def validate_source_rows(
    task: str,
    source_path: Path,
    max_chars: int,
    square_policy: str,
) -> list[dict]:
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    validated = []
    seen_ids = set()
    seen_prompts = set()
    for row in read_jsonl(source_path):
        location = f"{source_path}:{row['_source_row_number']}"
        declared = TASK_ALIASES.get(str(row.get("task", task)).strip().lower())
        if declared != task:
            raise ValueError(f"{location} declares task={row.get('task')!r}, expected {task}")
        prompt = row.get("prompt")
        enhanced = row.get("enhanced_prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError(f"{location} has no non-empty prompt")
        if not isinstance(enhanced, str) or not enhanced.strip():
            raise ValueError(f"{location} has no non-empty enhanced_prompt")
        prompt, enhanced = prompt.strip(), enhanced.strip()
        identity = source_identity(task, row)
        prompt_key = normalized_prompt(prompt)
        if identity in seen_ids or prompt_key in seen_prompts:
            continue
        seen_ids.add(identity)
        seen_prompts.add(prompt_key)
        normalized = dict(row)
        normalized.pop("_source_row_number", None)
        normalized.update(
            {
                "source_id": identity,
                "prompt": prompt,
                "enhanced_prompt": enhanced,
                "_original_fits_api": len(prompt) <= max_chars,
                "_enhanced_fits_api": len(enhanced) <= max_chars,
            }
        )
        dimensions = []
        for field in IMAGE_FIELDS[task]:
            normalized[field] = resolve_image(row.get(field), source_path, location=location, field=field)
            dimensions.append((field, *image_geometry(normalized[field])))
        orientations = {
            classify_geometry(
                width,
                height,
                identity=identity,
                square_policy=square_policy,
            )
            for _field, width, height in dimensions
        }
        if len(orientations) != 1:
            raise ValueError(f"{location} has keyframes with conflicting orientations: {dimensions}")
        if len({(width, height) for _field, width, height in dimensions}) != 1:
            raise ValueError(f"{location} has keyframes with different dimensions: {dimensions}")
        normalized["_source_image_width"] = dimensions[0][1]
        normalized["_source_image_height"] = dimensions[0][2]
        normalized["_source_orientation"] = orientations.pop()
        validated.append(normalized)
    return validated


def select_disjoint_variants(
    task: str,
    rows: list[dict],
    count: int,
    seed: int,
) -> list[tuple[dict, str]]:
    half = count // 2
    rng = random.Random(f"{seed}:{task}:keyframe-dmd")
    enhanced_candidates = [row for row in rows if row["_enhanced_fits_api"]]
    rng.shuffle(enhanced_candidates)
    enhanced_rows = enhanced_candidates[:half]
    used_ids = {row["source_id"] for row in enhanced_rows}
    original_candidates = [row for row in rows if row["source_id"] not in used_ids and row["_original_fits_api"]]
    rng.shuffle(original_candidates)
    original_rows = original_candidates[:half]
    if len(enhanced_rows) < half or len(original_rows) < half:
        raise ValueError(
            f"{task} cannot supply {half} enhanced + {half} original API-compatible, source-disjoint rows; available enhanced={len(enhanced_candidates)}, remaining original={len(original_candidates)}"
        )
    return [(row, "enhanced") for row in enhanced_rows] + [(row, "original") for row in original_rows]


def preserve_existing_selection(
    task: str,
    source_rows: list[dict],
    base_path: Path,
    count: int,
    seed: int,
    max_chars: int,
) -> tuple[list[tuple[dict, str]], Counter]:
    if not base_path.is_file():
        raise FileNotFoundError(base_path)
    by_id = {row["source_id"]: row for row in source_rows}
    base_rows = read_jsonl(base_path)
    if len(base_rows) != count:
        raise ValueError(f"{base_path} has {len(base_rows)} rows, expected {count}")
    selected: list[tuple[dict, str]] = []
    used_ids = set()
    replacements = Counter()
    invalid_by_variant = Counter()
    for base_row in base_rows:
        variant = str(base_row.get("prompt_variant", "")).strip().lower()
        if variant not in {"original", "enhanced"}:
            raise ValueError(f"{base_path} has invalid prompt_variant={variant!r}")
        identity = str(base_row.get("source_id", "")).strip()
        if identity not in by_id:
            raise KeyError(f"{base_path} source_id={identity!r} is absent from the current source JSONL")
        if identity in used_ids:
            raise ValueError(f"{base_path} repeats source_id={identity!r}")
        source = by_id[identity]
        expected = source["prompt"] if variant == "original" else source["enhanced_prompt"]
        actual = base_row.get("caption")
        if not isinstance(actual, str) or actual.strip() != expected:
            raise ValueError(f"{base_path} caption does not match {variant} text for source_id={identity!r}")
        if len(expected) > max_chars:
            invalid_by_variant[variant] += 1
            continue
        selected.append((source, variant))
        used_ids.add(identity)

    rng = random.Random(f"{seed}:{task}:api-compatible-replacements")
    for variant in ("original", "enhanced"):
        needed = invalid_by_variant[variant]
        if not needed:
            continue
        fits_key = "_original_fits_api" if variant == "original" else "_enhanced_fits_api"
        candidates = [row for row in source_rows if row["source_id"] not in used_ids and row[fits_key]]
        rng.shuffle(candidates)
        if len(candidates) < needed:
            raise ValueError(f"{task} needs {needed} replacement {variant} rows but only {len(candidates)} are available")
        for row in candidates[:needed]:
            selected.append((row, variant))
            used_ids.add(row["source_id"])
            replacements[variant] += 1

    variants = Counter(variant for _row, variant in selected)
    expected_half = count // 2
    if variants != Counter({"original": expected_half, "enhanced": expected_half}):
        raise ValueError(f"{base_path} does not preserve the required 1:1 variant balance after replacements: {variants}")
    if len(selected) != count or len(used_ids) != count:
        raise AssertionError(f"{task} selection is not source-disjoint: rows={len(selected)} ids={len(used_ids)}")
    return selected, replacements


def build_output_rows(task: str, selected: list[tuple[dict, str]], seed: int) -> dict[str, list[dict]]:
    rng = random.Random(f"{seed}:{task}:orientation")
    buckets = {"landscape": [], "portrait": []}
    for row, variant in selected:
        bucket = row["_source_orientation"]
        caption = row["prompt"] if variant == "original" else row["enhanced_prompt"]
        output = {
            "source_id": row["source_id"],
            "metadata_id": row.get("metadata_id"),
            "source_line": row.get("source_line"),
            "task": task,
            "prompt_variant": variant,
            "caption": caption,
            "prompt": row["prompt"],
            "enhanced_prompt": row["enhanced_prompt"],
            "source_duration": row.get("duration"),
            "source_resolution": row.get("resolution", "adaptive"),
            "source_image_width": row["_source_image_width"],
            "source_image_height": row["_source_image_height"],
            "source_orientation": row["_source_orientation"],
            "duration": row.get("duration"),
            "generation_duration": GENERATION_DURATION,
            "target_num_frames": NUM_FRAMES,
            "num_frames": NUM_FRAMES,
            "fps": FPS,
            "effective_target_duration": NUM_FRAMES / FPS,
            "aspect_bucket": bucket,
            "target_orientation": bucket,
            **GEOMETRY[bucket],
        }
        for field in IMAGE_FIELDS[task]:
            output[field] = row[field]
        buckets[bucket].append({key: value for key, value in output.items() if value is not None})
    for bucket, rows in buckets.items():
        rng.shuffle(rows)
        for index, row in enumerate(rows):
            row["id"] = index
            row["bucket_id"] = index
    if sum(len(rows) for rows in buckets.values()) != len(selected):
        raise AssertionError(f"{task} lost rows while assigning media-orientation buckets")
    return buckets


def atomic_write_jsonl(path: Path, rows: list[dict], overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Output exists: {path}; pass --overwrite")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    sources = parse_sources(args.source)
    base_mixed = parse_base_mixed(args.base_mixed)
    missing = set(args.tasks) - (sources.keys() & base_mixed.keys())
    if missing:
        raise ValueError(f"Pass both --source TASK=JSONL and --base-mixed TASK=JSONL for {sorted(missing)}.")
    output_root = args.output_root.expanduser().resolve()
    summary = {
        "seed": args.seed,
        "samples_per_task": args.samples_per_task,
        "max_prompt_characters": args.max_prompt_characters,
        "orientation_policy": "actual_exif_corrected_keyframe_dimensions",
        "square_policy": args.square_policy,
        "generation_duration": GENERATION_DURATION,
        "num_frames": NUM_FRAMES,
        "fps": FPS,
        "tasks": {},
    }
    for task in args.tasks:
        source_path = sources[task]
        source_rows = validate_source_rows(
            task,
            source_path,
            args.max_prompt_characters,
            args.square_policy,
        )
        selected, replacements = preserve_existing_selection(
            task,
            source_rows,
            base_mixed[task],
            args.samples_per_task,
            args.seed,
            args.max_prompt_characters,
        )
        buckets = build_output_rows(task, selected, args.seed)
        task_dir = output_root / task
        task_rows = []
        bucket_summary = {}
        for bucket in ("landscape", "portrait"):
            path = task_dir / f"{bucket}.jsonl"
            atomic_write_jsonl(path, buckets[bucket], args.overwrite)
            task_rows.extend(buckets[bucket])
            bucket_summary[bucket] = {
                "path": str(path),
                "records": len(buckets[bucket]),
                "variants": dict(Counter(row["prompt_variant"] for row in buckets[bucket])),
                "source_dimensions": dict(Counter(f"{row['source_image_width']}x{row['source_image_height']}" for row in buckets[bucket])),
            }
        random.Random(f"{args.seed}:{task}:combined").shuffle(task_rows)
        for index, row in enumerate(task_rows):
            row["id"] = index
        combined_path = task_dir / "metadata.jsonl"
        atomic_write_jsonl(combined_path, task_rows, args.overwrite)
        identities = [row["source_id"] for row in task_rows]
        if len(set(identities)) != len(identities):
            raise AssertionError(f"{task} output contains duplicate source_id values")
        summary["tasks"][task] = {
            "source": str(source_path),
            "source_sha256": sha256_file(source_path),
            "base_mixed": str(base_mixed[task]),
            "base_mixed_sha256": sha256_file(base_mixed[task]),
            "api_incompatible_rows_replaced": dict(replacements),
            "combined": str(combined_path),
            "records": len(task_rows),
            "variants": dict(Counter(row["prompt_variant"] for row in task_rows)),
            "buckets": bucket_summary,
        }
        print(json.dumps({task: summary["tasks"][task]}, ensure_ascii=False), flush=True)
    manifest_path = output_root / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote manifest: {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
