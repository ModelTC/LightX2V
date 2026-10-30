#!/usr/bin/env python3
"""Download, index and merge the public DiffSynth H3 self-generated AV dataset.

The metadata stages use the standard library only. GPU encoding is performed
by encode_modelscope_h3_dataset.py, once per independently assigned GPU.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from collections import Counter
from contextlib import contextmanager
from pathlib import Path

DATASET_ID = "DiffSynth-Studio/MiniMax-H3-Self-Generated-Dataset"
SCHEMA_VERSION = 1
SUBSETS = {"t2av": "metadata.jsonl", "ref2av": "ref2va_metadata.jsonl"}
FIXED_RESOLUTIONS = {(768, 1344), (1344, 768)}


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path):
    rows = []
    with Path(path).open(encoding="utf-8") as handle:
        for index, line in enumerate(handle, 1):
            if line.strip():
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError(f"Expected a JSON object: {path}:{index}")
                rows.append(row)
    return rows


def atomic_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def jsonl_bytes(rows):
    return "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows).encode("utf-8")


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def download_dataset(args):
    try:
        from modelscope.hub.snapshot_download import dataset_snapshot_download
    except ImportError as error:
        raise RuntimeError("Install the downloader with this Python: python -m pip install modelscope") from error
    destination = args.dataset_dir.expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    # ModelScope owns individual file verification and interrupted-download reuse.
    for attempt in range(1, 4):
        try:
            result = dataset_snapshot_download(
                dataset_id=DATASET_ID,
                revision=args.revision,
                local_dir=str(destination),
                max_workers=args.download_workers,
            )
            break
        except Exception:
            if attempt == 3:
                raise
            print(f"[download] attempt {attempt} failed; retrying with the existing partial download", flush=True)
            time.sleep(3)
    metadata = {}
    for name in SUBSETS.values():
        path = destination / name
        if not path.is_file() or not path.stat().st_size:
            raise RuntimeError(f"Download did not produce {path}; SDK returned {result!r}")
        metadata[name] = file_sha256(path)
    atomic_json(
        destination / "download.complete.json",
        {
            "schema_version": SCHEMA_VERSION,
            "dataset_id": DATASET_ID,
            "revision": args.revision,
            "metadata_sha256": metadata,
        },
    )
    print(f"[download] complete: {destination}", flush=True)


def raw_media_path(root, value):
    if not isinstance(value, str) or not value.strip():
        raise ValueError("A non-empty dataset-relative media path is required")
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Media path must stay inside the dataset: {value!r}")
    path = (root / relative).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise ValueError(f"Media path escapes the dataset: {value!r}") from error
    if not path.is_file() or not path.stat().st_size:
        raise FileNotFoundError(f"Dataset media is missing or empty: {path}")
    return str(path)


def prepared_row(row, subset, root, source_index, metadata_hash):
    prompt = row.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("Dataset row is missing its prompt")
    height, width = row.get("height"), row.get("width")
    if any(type(value) is not int or value <= 0 or value % 32 for value in (height, width)):
        raise ValueError(f"Expected positive native dimensions divisible by 32, got {height}x{width}")
    if row.get("num_frames") != 124:
        raise ValueError(f"Expected 124 frames, got {row.get('num_frames')!r}")
    video = raw_media_path(root, row.get("video_path"))
    identity = f"diffsynth_h3:{subset}:{row['video_path']}"
    result = {
        **row,
        "source_id": identity,
        "sample_id": identity,
        "source_index": source_index,
        "source_dataset": DATASET_ID,
        "source_metadata_sha256": metadata_hash,
        "task": subset,
        "caption": prompt,
        "prompt": prompt,
        "video_path": video,
        "target_height": height,
        "target_width": width,
        "target_num_frames": 124,
        "num_frames": 124,
        "fps": 24,
        "target_orientation": "landscape" if width >= height else "portrait",
        "references": [],
    }
    if subset == "ref2av":
        kind = row.get("reference_kind")
        labels = row.get("reference_labels")
        normalized_labels = " ".join(str(labels).replace("<", "").replace(">", "").split())
        if kind == "image" and normalized_labels == "Picture 1":
            reference = {"kind": "image"}
        elif kind == "video" and normalized_labels in {"Video 1", "Audio 1 + Video 1"}:
            reference = {"kind": "video", "include_embedded_audio": normalized_labels == "Audio 1 + Video 1"}
        else:
            raise ValueError(f"Unsupported reference declaration: kind={kind!r}, labels={labels!r}")
        reference["path"] = raw_media_path(root, row.get("reference_path"))
        result["references"] = [reference]
        result["reference_path"] = reference["path"]
    return result


def prepare_dataset(args):
    root = args.dataset_dir.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    selected = tuple(SUBSETS) if args.subsets == "all" else (args.subsets,)
    rows, missing = [], []
    source_index = 0
    source_counts = {}
    hashes = {}
    seen = set()
    for subset in selected:
        manifest = root / SUBSETS[subset]
        hashes[subset] = file_sha256(manifest)
        source_rows = read_jsonl(manifest)
        if args.max_samples_per_subset is not None:
            source_rows = source_rows[: args.max_samples_per_subset]
        source_counts[subset] = len(source_rows)
        for index, row in enumerate(source_rows):
            current_index = source_index
            source_index += 1
            identity = (subset, row.get("video_path"))
            if identity in seen:
                raise ValueError(f"Duplicate dataset video identity: {identity}")
            seen.add(identity)
            try:
                rows.append(prepared_row(row, subset, root, current_index, hashes[subset]))
            except FileNotFoundError as error:
                if not args.allow_missing_media:
                    raise
                missing.append({"subset": subset, "metadata_row": index, "source_index": current_index, "video_path": row.get("video_path"), "error": str(error)})
    if not rows:
        raise ValueError("No usable input rows remain after dataset preparation")
    previous = output / "prepared.jsonl"
    if previous.is_file() and previous.read_bytes() != jsonl_bytes(rows):
        if (output / "cache.complete.json").exists() or any(output.glob("receipt.shard-*.json")):
            raise ValueError("The selected dataset/metadata changed in an encoded output directory; choose a new --output-dir")
    atomic_jsonl(output / "prepared.jsonl", rows)
    atomic_jsonl(output / "prepare_excluded_missing_media.jsonl", missing)
    counts = Counter(row["task"] for row in rows)
    compatible = Counter(row["task"] for row in rows if (row["target_height"], row["target_width"]) in FIXED_RESOLUTIONS)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "dataset_id": DATASET_ID,
        "dataset_dir": str(root),
        "subsets": list(selected),
        "source_counts": source_counts,
        "prepared_count": len(rows),
        "prepared_by_task": dict(counts),
        "native_768p_by_task": dict(compatible),
        "excluded_missing_media": len(missing),
        "source_metadata_sha256": hashes,
        "prepared_sha256": file_sha256(output / "prepared.jsonl"),
        "resolution_policy": "native",
        "fps": 24,
        "num_frames": 124,
    }
    atomic_json(output / "prepare.complete.json", summary)
    print(f"[prepare] {json.dumps(summary, ensure_ascii=False)}", flush=True)


def condition_row(row):
    return {
        key: value
        for key, value in row.items()
        if key
        not in {
            "video_path",
            "audio_path",
            "video_latent_path",
            "audio_latent_path",
            "video_latent_shape",
            "audio_latent_shape",
        }
    }


def is_native_768p(row):
    return (row["target_height"], row["target_width"]) in FIXED_RESOLUTIONS


def is_image_only(row):
    kinds = Counter(entry.get("kind", entry.get("type")) for entry in row.get("references", []))
    return (
        row.get("reference_image_count", row.get("ref_image_count", kinds["image"])) > 0
        and row.get("reference_video_count", kinds["video"]) == 0
        and row.get("reference_audio_count", kinds["audio"]) == 0
    )


def in_view(row, name):
    if name == "all":
        return True
    if name.startswith("t2av"):
        return row["task"] == "t2av" and (name == "t2av" or is_native_768p(row))
    return row["task"] == "ref2av" and (name == "ref2av" or (is_image_only(row) and (name == "ref2av_imageonly" or is_native_768p(row))))


@contextmanager
def _imageonly_view_lock(output):
    """Serialize the four launchers using shared-directory atomic mkdir."""
    reservation = output / ".ref2av_imageonly.publish-lock"
    deadline = time.monotonic() + 60
    while True:
        try:
            reservation.mkdir()
            break
        except FileExistsError:
            if time.monotonic() >= deadline:
                raise RuntimeError(f"Timed out waiting for image-only view publication: {reservation}")
            time.sleep(0.05)
    try:
        yield
    finally:
        reservation.rmdir()


def build_imageonly_view(output_dir):
    """Publish pure-image Ref targets/conditions from an existing full AV cache.

    This is a standard-library-only metadata operation. Target AV latents and
    image conditioning are reused byte-for-byte, with every native geometry.
    """
    output = Path(output_dir).expanduser().resolve()
    source = output / "ref2av/metadata.jsonl"
    source_bytes = source.read_bytes()
    source_sha256 = hashlib.sha256(source_bytes).hexdigest()
    source_receipt = json.loads(source.with_suffix(".complete.json").read_text())
    rows = [json.loads(line) for line in source_bytes.splitlines() if line.strip()]
    if (
        source_receipt.get("schema_version") != SCHEMA_VERSION
        or source_receipt.get("manifest_sha256") != source_sha256
        or source_receipt.get("completed_count") != len(rows)
        or source_receipt.get("failed_count") != 0
        or source_receipt.get("input_total_rows") != len(rows)
        or not source_receipt.get("preprocess_fingerprint")
    ):
        raise ValueError(f"Ref source is not a fully verified complete cache: {source}")
    identities = set()
    selected = []
    for row in rows:
        if not isinstance(row, dict) or row.get("task") != "ref2av" or not row.get("source_id") or row["source_id"] in identities:
            raise ValueError(f"Invalid or duplicate Ref source identity in {source}")
        identities.add(row["source_id"])
        counts = (row.get("reference_image_count", row.get("ref_image_count", 0)), row.get("reference_video_count", 0), row.get("reference_audio_count", 0))
        if any(type(value) is not int or value < 0 for value in counts):
            raise ValueError(f"Invalid Ref media counts: {row['source_id']}")
        references = row.get("references")
        if references is not None and (not isinstance(references, list) or not references or any(not isinstance(reference, dict) for reference in references)):
            raise ValueError(f"Invalid Ref reference declarations: {row['source_id']}")
        if counts[0] <= 0 or counts[1] != 0 or counts[2] != 0:
            continue
        kinds = [str(reference.get("kind", reference.get("type", ""))).lower() for reference in references or []]
        if any(kind != "image" for kind in kinds):
            raise ValueError(f"Image-only counts disagree with reference declarations: {row['source_id']}")
        if int(row.get("reference_audio_rows", 0)) != 0:
            raise ValueError(f"Image-only condition contains audio reference rows: {row['source_id']}")
        height, width = row.get("target_height"), row.get("target_width")
        frames = row.get("target_num_frames", row.get("num_frames"))
        if frames != 124 or any(type(value) is not int or value <= 0 or value % 32 for value in (height, width)):
            raise ValueError(f"Expected native 32-aligned 124-frame image-only target: {row['source_id']}")
        for key in ("condition_path", "video_latent_path", "audio_latent_path"):
            value = row.get(key)
            if not isinstance(value, str) or not value:
                raise ValueError(f"Missing image-only {key}: {row['source_id']}")
            path = Path(value)
            if not path.is_absolute() or not path.is_file() or path.stat().st_size == 0:
                raise ValueError(f"Missing absolute image-only {key}: {value!r}")
        selected.append(row)
    if not selected:
        raise ValueError(f"Ref source contains no pure-image examples: {source}")
    fingerprint = hashlib.sha256(
        json.dumps({"view": "ref2av_imageonly", "source_sha256": source_sha256, "source_fingerprint": source_receipt["preprocess_fingerprint"]}, sort_keys=True).encode()
    ).hexdigest()
    directory = output / "ref2av_imageonly"
    publications = ((directory / "metadata.jsonl", selected), (directory / "condition_metadata.jsonl", [condition_row(row) for row in selected]))
    with _imageonly_view_lock(output):
        for path, records in publications:
            content = jsonl_bytes(records)
            receipt = {
                "schema_version": SCHEMA_VERSION,
                "prepared_sha256": source_receipt.get("prepared_sha256"),
                "preprocess_fingerprint": fingerprint,
                "manifest_sha256": hashlib.sha256(content).hexdigest(),
                "completed_count": len(selected),
                "failed_count": 0,
                "input_total_rows": len(selected),
                "source_manifest_sha256": source_sha256,
                "source_total_rows": len(rows),
                "excluded_non_image_rows": len(rows) - len(selected),
                "reference_image_counts": dict(Counter(row.get("reference_image_count", row.get("ref_image_count", 0)) for row in selected)),
            }
            receipt_path = path.with_suffix(".complete.json")
            if path.is_file() and path.read_bytes() == content and receipt_path.is_file() and json.loads(receipt_path.read_text()) == json.loads(json.dumps(receipt)):
                continue
            receipt_path.unlink(missing_ok=True)
            atomic_jsonl(path, records)
            atomic_json(receipt_path, receipt)
    summary = {"input_rows": len(rows), "kept_rows": len(selected), "excluded_rows": len(rows) - len(selected), "resolution_policy": "native", "output_manifest": str(directory / "metadata.jsonl")}
    print(f"[imageonly-view] {json.dumps(summary, ensure_ascii=False)}", flush=True)
    return summary


def merge_dataset(args):
    output = args.output_dir.expanduser().resolve()
    prepared_path = output / "prepared.jsonl"
    expected_hash = file_sha256(prepared_path)
    prepared = read_jsonl(prepared_path)
    expected = {row["source_id"]: row for row in prepared}
    if len(expected) != len(prepared):
        raise ValueError("Duplicate source_id in prepared manifest")
    successes, failures, accounted, contracts = [], [], set(), []
    for shard_index in range(args.num_shards):
        suffix = f"{shard_index:05d}-of-{args.num_shards:05d}"
        receipt = json.loads((output / f"receipt.shard-{suffix}.json").read_text())
        shard_expected = {row["source_id"] for index, row in enumerate(prepared) if index % args.num_shards == shard_index}
        checks = {"schema_version": SCHEMA_VERSION, "prepared_sha256": expected_hash, "shard_index": shard_index, "num_shards": args.num_shards, "expected_count": len(shard_expected)}
        if any(receipt.get(key) != value for key, value in checks.items()):
            raise ValueError(f"Stale/incompatible encoder receipt for shard {shard_index}: expected {checks}")
        contracts.append({key: value for key, value in receipt.items() if key not in {"expected_count", "completed_count", "failed_count"}})
        rows = read_jsonl(output / f"metadata.shard-{suffix}.jsonl")
        failed = read_jsonl(output / f"failed.shard-{suffix}.jsonl")
        if len(rows) != receipt.get("completed_count") or len(failed) != receipt.get("failed_count"):
            raise ValueError(f"Encoder receipt/count mismatch for shard {shard_index}")
        local = set()
        for row in rows + failed:
            identity = row.get("source_id")
            if identity not in shard_expected or identity in local or identity in accounted:
                raise ValueError(f"Wrong/duplicate source identity in shard {shard_index}: {identity}")
            local.add(identity)
        if local != shard_expected:
            raise ValueError(f"Encoder shard {shard_index} does not account for every input row")
        accounted.update(local)
        for row in rows:
            original = expected[row["source_id"]]
            for key in ("task", "target_height", "target_width", "num_frames"):
                if row.get(key) != original.get(key):
                    raise ValueError(f"Encoded {key} differs from prepared row {row['source_id']}")
            if not row.get("cache_fingerprint"):
                raise ValueError(f"Encoded row is missing its condition identity: {row['source_id']}")
            for key in ("condition_path", "video_latent_path", "audio_latent_path"):
                path = Path(row.get(key, ""))
                if not path.is_absolute() or not path.is_file() or not path.stat().st_size:
                    raise ValueError(f"Missing absolute encoded {key}: {row.get(key)!r}")
        successes.extend(rows)
        failures.extend(failed)
    if failures and not args.allow_failures:
        raise RuntimeError(f"Encoding failed for {len(failures)} rows; inspect failed.shard-*.jsonl and rerun to retry")
    if not successes:
        raise RuntimeError("Encoding produced no complete AV+condition samples")
    successes.sort(key=lambda row: row["source_index"])
    views = {name: [row for row in successes if in_view(row, name)] for name in ("t2av", "t2av_768p", "ref2av", "ref2av_imageonly", "ref2av_imageonly_768p")}
    t2av_768p = views["t2av_768p"]
    # Only publish a completion marker after every manifest has been replaced.
    (output / "cache.complete.json").unlink(missing_ok=True)
    atomic_jsonl(output / "all_metadata.jsonl", successes)
    atomic_jsonl(output / "metadata.jsonl", t2av_768p)
    atomic_jsonl(output / "condition_metadata.jsonl", [condition_row(row) for row in t2av_768p])
    atomic_jsonl(output / "failed.jsonl", failures)
    published = [(output / "all_metadata.jsonl", "all", successes), (output / "metadata.jsonl", "t2av_768p", t2av_768p), (output / "condition_metadata.jsonl", "t2av_768p", t2av_768p)]
    for name, rows in views.items():
        directory = output / name
        atomic_jsonl(directory / "metadata.jsonl", rows)
        atomic_jsonl(directory / "condition_metadata.jsonl", [condition_row(row) for row in rows])
        published.extend([(directory / "metadata.jsonl", name, rows), (directory / "condition_metadata.jsonl", name, rows)])
    for path, name, rows in published:
        input_count = sum(in_view(row, name) for row in prepared)
        fingerprint = hashlib.sha256(json.dumps({"contracts": contracts, "view": name}, sort_keys=True).encode()).hexdigest()
        atomic_json(
            path.with_suffix(".complete.json"),
            {
                "schema_version": SCHEMA_VERSION,
                "prepared_sha256": expected_hash,
                "preprocess_fingerprint": fingerprint,
                "manifest_sha256": file_sha256(path),
                "completed_count": len(rows),
                "failed_count": input_count - len(rows),
                "input_total_rows": input_count,
                "reference_image_counts": dict(Counter(row.get("reference_image_count", row.get("ref_image_count", 0)) for row in rows if row["task"] == "ref2av")),
            },
        )
    summary = {
        "schema_version": SCHEMA_VERSION,
        "dataset_id": DATASET_ID,
        "prepared_sha256": expected_hash,
        "prepared_count": len(prepared),
        "completed_count": len(successes),
        "failed_count": len(failures),
        "num_shards": args.num_shards,
        "views": {key: len(value) for key, value in views.items()},
        "default_cm_metadata": str(output / "metadata.jsonl"),
    }
    preparation = output / "prepare.complete.json"
    if preparation.is_file():
        source = json.loads(preparation.read_text())
        summary.update(source_counts=source.get("source_counts"), excluded_missing_media=source.get("excluded_missing_media", 0))
    atomic_json(output / "cache.complete.json", summary)
    print(f"[merge] {json.dumps(summary, ensure_ascii=False)}", flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    download = subparsers.add_parser("download")
    download.add_argument("--dataset-dir", type=Path, required=True)
    download.add_argument("--revision", default="master")
    download.add_argument("--download-workers", type=int, default=8)
    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--dataset-dir", type=Path, required=True)
    prepare.add_argument("--output-dir", type=Path, required=True)
    prepare.add_argument("--subsets", choices=("all", *SUBSETS), default="all")
    prepare.add_argument("--max-samples-per-subset", type=int)
    prepare.add_argument("--allow-missing-media", action="store_true")
    merge = subparsers.add_parser("merge")
    merge.add_argument("--output-dir", type=Path, required=True)
    merge.add_argument("--num-shards", type=int, default=4)
    merge.add_argument("--allow-failures", action="store_true")
    imageonly = subparsers.add_parser("imageonly-view", help="Build native pure-image Ref views from an existing verified full AV cache, without encoding")
    imageonly.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    for option in ("download_workers", "num_shards", "max_samples_per_subset"):
        value = getattr(args, option, None)
        if value is not None and value <= 0:
            parser.error(f"--{option.replace('_', '-')} must be positive")
    try:
        if args.command == "imageonly-view":
            build_imageonly_view(args.output_dir)
        else:
            {"download": download_dataset, "prepare": prepare_dataset, "merge": merge_dataset}[args.command](args)
    except (OSError, ValueError, RuntimeError, KeyError, TypeError) as error:
        parser.exit(1, f"H3 dataset {args.command} failed: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
