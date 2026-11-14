#!/usr/bin/env python3
"""Select 1,000 enhanced-prompt image-only samples per count, split 500/500."""

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from common import atomic_json, atomic_jsonl, digest, file_digest, records

DEFAULT_RAW = "/mnt/lm_data_afs/gushiqiao/datasets/omni_r2v_image_only_100k_20261004/omni_r2v_image_only_100000_ir.jsonl"
DEFAULT_CACHE = str(Path(DEFAULT_RAW).parent / "latent_match124_bf16/metadata.jsonl")
DEFAULT_MEDIA = "/mnt/lm_data_afs/gushiqiao/datasets/omini-r2v"
DEFAULT_OUTPUT = "/mnt/devsft_afs_2/gushiqiao/outputs/data_new2/omni_ref_teacher29_decfg_5k"


def source_identity(row, source_index=None):
    for key in ("sample_id", "metadata_id", "id"):
        value = row.get(key)
        if value is not None and str(value).strip():
            return str(value)
    if source_index is not None:
        return str(source_index)
    raise ValueError("missing stable sample_id/metadata_id/id")


def ordered_images(row, media_root):
    entries = None
    for key in ("references", "actual_ordered_references", "ordered_references"):
        if row.get(key) is not None:
            entries = row[key]
            break
    if entries is None:
        if row.get("reference_videos") or row.get("reference_audios"):
            raise ValueError("non-image references")
        entries = row.get("reference_images", [])
        if isinstance(entries, str):
            entries = [entries]
        entries = [{"kind": "image", "path": value} if isinstance(value, str) else {"kind": "image", **value} for value in entries]
    if not isinstance(entries, list) or not entries or not all(isinstance(entry, dict) for entry in entries):
        raise ValueError("references must be a non-empty ordered list")
    orders = [entry.get("order", entry.get("reference_index")) for entry in entries]
    if any(order is not None for order in orders):
        if any(type(order) is not int for order in orders) or orders not in (list(range(len(entries))), list(range(1, len(entries) + 1))):
            raise ValueError("reference order must already be contiguous zero- or one-based; refusing to reorder")
    images = []
    for entry in entries:
        kind = str(entry.get("kind", entry.get("modality", entry.get("type", entry.get("role", ""))))).strip().lower()
        if kind not in {"image", "reference_image", "image_url"}:
            raise ValueError("non-image or unknown reference modality")
        value = next((entry[key] for key in ("local_path", "path", "media_path", "url", "rel_path") if entry.get(key)), None)
        if not isinstance(value, str) or not value.strip() or "," in value or value.startswith(("http://", "https://", "data:", "mm_file://")):
            raise ValueError("reference needs a local image path without commas")
        path = Path(value.strip()).expanduser()
        images.append(str((path if path.is_absolute() else Path(media_root) / path).resolve()))
    return images


def load_cache(path):
    cache = {}
    for _, row in records(path):
        key = (row.get("source_index"), str(row.get("source_id")))
        if type(key[0]) is not int or row.get("source_id") is None:
            raise ValueError("Cache metadata requires source_index and source_id on every row")
        if key in cache:
            raise ValueError(f"Duplicate condition cache identity: {key}")
        cache[key] = row
    return cache


def prepare(raw_path, cache_path, media_root, output_dir, node_index, quota=1000, seed=42, check_images=True):
    if quota <= 0 or quota % 2:
        raise ValueError("Per-count quota must be a positive even number")
    raw_path, cache_path, output_dir = Path(raw_path).resolve(), Path(cache_path).resolve(), Path(output_dir)
    cache = load_cache(cache_path)
    buckets, rejected = defaultdict(list), Counter()
    seen_ids, total = set(), 0
    for source_index, row in records(raw_path):
        total += 1
        prompt = row.get("enhanced_prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            rejected["no_enhanced_prompt"] += 1
            continue
        try:
            source_id = source_identity(row, source_index)
            images = ordered_images(row, media_root)
        except (ValueError, TypeError) as error:
            rejected[str(error)] += 1
            continue
        if len(images) not in range(2, 7):
            rejected["image_count_outside_2_to_6"] += 1
            continue
        cached = cache.get((source_index, source_id))
        if cached is None:
            rejected["no_exact_condition_cache_identity"] += 1
            continue
        if str(row.get("task", "")).lower() not in {"ref2av", "ref2va", "ref2v"} or cached.get("task") != "ref2av":
            raise ValueError(f"Raw/cache task mismatch for {(source_index, source_id)}")
        for count_key in ("reference_image_count", "ref_image_count"):
            if count_key in cached and cached[count_key] != len(images):
                raise ValueError(f"Cache {count_key} differs from actual ordered references for {(source_index, source_id)}")
        height, width = cached.get("target_height"), cached.get("target_width")
        if (height, width) not in {(768, 1344), (1344, 768)} or cached.get("num_frames") != 124:
            raise ValueError(f"Invalid cached geometry for {(source_index, source_id)}: {height}x{width}, frames={cached.get('num_frames')}")
        if cached.get("prompt_source", "enhanced_prompt") != "enhanced_prompt":
            raise ValueError(f"Cache prompt differs from enhanced_prompt for {(source_index, source_id)}")
        prompt_hash = hashlib.sha256(prompt.strip().encode()).hexdigest()
        for prompt_key in ("prompt", "caption", "enhanced_prompt"):
            if prompt_key in cached and cached[prompt_key] != prompt.strip():
                raise ValueError(f"Cache {prompt_key} differs from enhanced_prompt for {(source_index, source_id)}")
        for hash_key in ("prompt_sha256", "enhanced_prompt_sha256"):
            if hash_key in cached and cached[hash_key] != prompt_hash:
                raise ValueError(f"Cache {hash_key} differs from enhanced_prompt for {(source_index, source_id)}")
        if cached.get("reference_video_count", 0) or cached.get("reference_audio_count", 0):
            raise ValueError(f"Cache is not image-only for {(source_index, source_id)}")
        if source_id in seen_ids:
            raise ValueError(f"Duplicate stable source identity: {source_id}")
        seen_ids.add(source_id)
        condition = Path(cached["condition_path"])
        if not condition.is_absolute():
            condition = cache_path.parent / condition
        sample = {
            "source_jsonl": str(raw_path),
            "source_index": source_index,
            "source_id": source_id,
            "sample_key": digest(source_id)[:24],
            "prompt": prompt.strip(),
            "prompt_source": "enhanced_prompt",
            "reference_images": images,
            "ref_image_count": len(images),
            "target_height": height,
            "target_width": width,
            "num_frames": 124,
            "fps": 24,
            "seed": seed,
            "cache_metadata_path": str(cache_path),
            "cache_condition_path": str(condition),
            "condition_cache_fingerprint": cached.get("cache_fingerprint"),
        }
        buckets[len(images)].append(sample)
    counts = {str(count): len(buckets[count]) for count in range(2, 7)}
    if any(len(buckets[count]) < quota for count in range(2, 7)):
        raise ValueError(f"Insufficient eligible samples: need {quota} each, have {counts}; rejected={dict(rejected)}")
    selected = []
    for count in range(2, 7):
        # Hash ranking is stable across Python versions and raw row iteration order.
        ranked = sorted(buckets[count], key=lambda row: (digest([seed, count, row["source_id"]]), row["source_id"]))[:quota]
        for position, row in enumerate(ranked):
            row["node_index"] = position % 2
            if check_images:
                for image in row["reference_images"]:
                    if not Path(image).is_file() or Path(image).stat().st_size == 0:
                        raise FileNotFoundError(f"Selected source {row['source_id']} has missing/empty image: {image}")
            selected.append(row)
    selection_hash = digest(selected)
    summary = {
        "schema": "h3_teacher_selection_v1",
        "selection_sha256": selection_hash,
        "source_jsonl": str(raw_path),
        "source_sha256": file_digest(raw_path),
        "source_rows": total,
        "cache_metadata_path": str(cache_path),
        "cache_metadata_sha256": file_digest(cache_path),
        "seed": seed,
        "per_count": quota,
        "total_selected": len(selected),
        "eligible_counts": counts,
        "rejected_counts": dict(rejected),
        "geometry_source": "condition_cache",
        "image_paths_checked": check_images,
    }
    directory = output_dir / "selection" / f"node{node_index}"
    other_manifest = output_dir / "selection" / f"node{1 - node_index}" / "manifest.json"
    if other_manifest.exists():
        other = json.loads(other_manifest.read_text())
        for key in ("selection_sha256", "source_sha256", "cache_metadata_sha256"):
            if other[key] != summary[key]:
                raise ValueError(f"Nodes disagree on {key}; do not collect into this output root")
    own_manifest = directory / "manifest.json"
    if own_manifest.exists() and json.loads(own_manifest.read_text()) != summary:
        raise ValueError(f"Selection changed at {directory}; use a new output root")
    atomic_jsonl(directory / "selected.jsonl", selected)
    for shard in (0, 1):
        atomic_jsonl(directory / f"node{shard}.jsonl", [row for row in selected if row["node_index"] == shard])
    atomic_json(own_manifest, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-jsonl", default=DEFAULT_RAW)
    parser.add_argument("--cache-metadata", default=DEFAULT_CACHE)
    parser.add_argument("--media-root", default=DEFAULT_MEDIA)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT)
    parser.add_argument("--node-index", type=int, choices=(0, 1), required=True)
    parser.add_argument("--per-count", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    prepare(args.raw_jsonl, args.cache_metadata, args.media_root, args.output_dir, args.node_index, args.per_count, args.seed)


if __name__ == "__main__":
    main()
