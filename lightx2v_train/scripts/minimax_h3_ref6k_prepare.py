#!/usr/bin/env python3
"""Publish 5k original-teacher multi-image targets + 1k self-generated single-image targets.

CPU metadata only: reuse existing condition/AV files without copying tensors.
The distributed launcher calls this once before starting its training workers.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("_ref6k_teacher_import", ROOT / "data_process/minimax_h3/import_lightx2v_ref_teacher_latents.py")
teacher_import = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(teacher_import)
require = teacher_import.require
digest = teacher_import.digest
file_digest = teacher_import.file_digest
GEOMETRIES = {(124, 768, 1344), (124, 1344, 768)}


def read_json(path):
    return json.loads(Path(path).read_text())


def rows(path):
    return [row for _, row in teacher_import.jsonl(Path(path))]


def checked_receipt(manifest):
    records = rows(manifest)
    receipt = read_json(manifest.with_suffix(".complete.json"))
    require(receipt.get("manifest_sha256") == file_digest(manifest), f"Manifest differs from completion receipt: {manifest}")
    require(
        receipt.get("completed_count") == receipt.get("input_total_rows") == len(records) and receipt.get("failed_count") == 0 and bool(receipt.get("preprocess_fingerprint")),
        f"Cache is incomplete: {manifest}",
    )
    return records, receipt


def completed_teacher_indexes(directory, per_count):
    indexes, selection = [], None
    for node in (0, 1):
        index = directory / f"index.node{node}.jsonl"
        receipt = read_json(directory / f"complete.node{node}.json")
        records = rows(index)
        expected = 5 * per_count // 2
        require(
            receipt.get("expected") == receipt.get("completed") == len(records) == expected and receipt.get("failed") == 0,
            f"Teacher node{node} must have {expected} successful samples and zero failures",
        )
        require(receipt.get("index_sha256") == file_digest(index), f"Teacher index differs from completion receipt: {index}")
        failures = directory / f"failures.node{node}.jsonl"
        require(receipt.get("failures_sha256") == file_digest(failures) and not rows(failures), f"Teacher node{node} has inconsistent failures")
        identity = receipt.get("selection_sha256")
        require(isinstance(identity, str) and identity and selection in (None, identity), "Teacher nodes must use the same selection")
        selection = identity
        for row in records:
            require(row.get("node_index") == node and row.get("selection_sha256") == selection, f"Wrong node/selection in {index}")
            export = teacher_import.absolute(row.get("export_dir"), directory)
            complete = read_json(export / "completion.json")
            require(
                complete.get("manifest_sha256") == file_digest(export / "manifest.json")
                and complete.get("request_fingerprint") == row.get("request_fingerprint")
                and complete.get("cache_fingerprint") == row.get("cache_fingerprint"),
                f"Teacher sample completion differs: {export}",
            )
        indexes.append(index)
    return indexes


def check_rows(records):
    for row in records:
        geometry = tuple(row[key] for key in ("target_num_frames", "target_height", "target_width"))
        require(geometry in GEOMETRIES and row["fps"] == 24, f"Ref6k requires 124 frames at 768p/24fps: {row['source_id']}")
        cost = row.get("packed_sequence_tokens_124")
        require(type(cost) in (int, float) and math.isfinite(cost) and cost > 0, f"Missing Ref token cost: {row['source_id']}")
    require(len({row["condition_path"] for row in records}) == len(records), "Ref6k contains duplicate condition paths")


def prepare(teacher_export_dir, raw_jsonl, media_root, selfgen_metadata, output_dir, *, seed=42, per_count=1000):
    teacher_export_dir, raw_jsonl, media_root, selfgen_metadata, output_dir = (Path(path).expanduser().resolve() for path in (teacher_export_dir, raw_jsonl, media_root, selfgen_metadata, output_dir))
    require(type(per_count) is int and per_count > 0 and per_count % 2 == 0, "per_count must be a positive even integer")
    require(type(seed) is int and seed >= 0, "seed must be a nonnegative integer")
    require(selfgen_metadata != output_dir / "metadata.jsonl", "Output must not overwrite the self-generated source")
    indexes = completed_teacher_indexes(teacher_export_dir, per_count)
    teacher_dir = output_dir / "teacher5k"
    imported = teacher_import.import_exports(raw_jsonl, indexes, teacher_dir, media_root=media_root, expected_count=5 * per_count)
    settings = imported["teacher_settings"]
    require(settings.get("schema") == teacher_import.SCHEMA and settings.get("export_mode") == "final", "Ref6k requires final-only teacher exports")
    require(settings.get("decfg_merged_identity") == [] and not settings.get("lora_configs"), "Ref6k requires original teacher exports without DeCFG or LoRA")
    require(
        settings.get("enable_cfg") is False and settings.get("guidance_scale") == 1.0 and settings.get("reference_image_resize_mode") == "match",
        "Ref6k requires CFG-free teacher exports with match references",
    )
    teacher_rows = rows(teacher_dir / "metadata.jsonl")
    require(Counter(row["reference_image_count"] for row in teacher_rows) == {count: per_count for count in range(2, 7)}, "Teacher data must contain exactly 1000 rows per image-count bucket 2..6")
    _, selfgen_receipt = checked_receipt(selfgen_metadata)
    gan = teacher_import._leaf("lightx2v_train/trainers/dmd/dmd2_gan_data.py", "_ref6k_gan_data")
    _, candidates, _ = gan.load_gan_real_records(selfgen_metadata, task="ref2av", geometry=None)
    candidates = [row for row in candidates if row["reference_image_count"] == 1 and tuple(row[key] for key in ("target_num_frames", "target_height", "target_width")) in GEOMETRIES]
    require(len(candidates) >= per_count, f"Self-generated cache has only {len(candidates)} eligible single-image 768p rows; need {per_count}")
    require(len({str(row["source_id"]) for row in candidates}) == len(candidates), "Duplicate self-generated source IDs")
    candidates.sort(key=lambda row: (digest({"seed": seed, "source_id": str(row["source_id"])}), str(row["source_id"])))
    selected = candidates[:per_count]
    records = [dict(row, ref6k_source="omni_teacher_base") for row in teacher_rows] + [dict(row, ref6k_source="selfgen_single_image") for row in selected]
    check_rows(records)
    sources = {
        "teacher_export_dir": str(teacher_export_dir),
        "teacher_import_fingerprint": imported["preprocess_fingerprint"],
        "teacher_model_path": settings["model_path"],
        "selfgen_metadata": str(selfgen_metadata),
        "selfgen_manifest_sha256": selfgen_receipt["manifest_sha256"],
        "seed": seed,
    }
    fingerprint = digest({"schema": "h3_ref6k_v1", "sources": sources, "rows": records})
    receipt = {
        "schema": "h3_ref6k_v1",
        "schema_version": 1,
        "teacher_weights": "base",
        "completed_count": len(records),
        "input_total_rows": len(records),
        "failed_count": 0,
        "preprocess_fingerprint": fingerprint,
        "reference_image_counts": {str(count): per_count for count in range(1, 7)},
        "source_counts": {"omni_teacher_base": 5 * per_count, "selfgen_single_image": per_count},
        "selfgen_eligible_single_image_rows": len(candidates),
        "tensor_validation_scope": "metadata_and_file_presence_only",
        **sources,
    }
    marker = output_dir / "metadata.complete.json"
    if marker.exists():
        previous = read_json(marker)
        require(previous.get("preprocess_fingerprint") == fingerprint, "Ref6k inputs/selection changed; choose a new output directory")
        checked_receipt(output_dir / "metadata.jsonl")
        checked_receipt(output_dir / "condition_metadata.jsonl")
        return previous
    condition_rows = [{key: value for key, value in row.items() if key not in {"video_latent_path", "audio_latent_path", "video_latent_shape", "audio_latent_shape"}} for row in records]
    publications = []
    for name, values in (("metadata", records), ("condition_metadata", condition_rows)):
        content = "".join(json.dumps(row, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n" for row in values).encode()
        manifest = output_dir / f"{name}.jsonl"
        teacher_import.atomic_write(manifest, content)
        publications.append((name, {**receipt, "manifest_sha256": file_digest(manifest)}))
    # Publish the training receipt last, after both manifests are complete.
    for name, value in reversed(publications):
        teacher_import.atomic_write(output_dir / f"{name}.complete.json", (json.dumps(value, sort_keys=True, ensure_ascii=False, indent=2) + "\n").encode())
    return read_json(marker)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("teacher-export-dir", "raw-jsonl", "media-root", "selfgen-metadata", "output-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    receipt = prepare(args.teacher_export_dir, args.raw_jsonl, args.media_root, args.selfgen_metadata, args.output_dir, seed=args.seed)
    print(json.dumps(receipt, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
