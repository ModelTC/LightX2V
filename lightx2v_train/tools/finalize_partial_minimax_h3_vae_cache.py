#!/usr/bin/env python
"""Build a MiniMax-H3 VAE cache manifest from completed latent files."""

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path

import torch

from lightx2v_train.data.utils import record_value
from lightx2v_train.data.video_dataset import VideoDataset
from lightx2v_train.model_zoo.native.minimax_h3.vae_geometry import (
    align_num_frames,
    validate_source_num_frames,
    video_latent_num_frames,
)
from lightx2v_train.model_zoo.native.minimax_h3.video_vae import resolve_video_vae_dir


CACHE_SCHEMA_VERSION = 1
LATENT_CHANNELS = 24


def parse_args():
    parser = argparse.ArgumentParser(description="Finalize an interrupted MiniMax-H3 VAE latent cache.")
    parser.add_argument("--input", required=True, help="Source metadata.jsonl used by the cache job.")
    parser.add_argument("--output-dir", required=True, help="Directory containing latents/.")
    parser.add_argument("--model-path", required=True, help="MiniMax-H3 model root or video-VAE directory.")
    parser.add_argument("--num-frames", type=int, default=362)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--height", type=int, default=768)
    parser.add_argument("--width", type=int, default=1344)
    parser.add_argument("--fixed-geometry", action="store_true")
    parser.add_argument("--posterior", choices=("mode", "sample"), default="mode")
    parser.add_argument("--storage-dtype", choices=("bf16", "fp16", "fp32"), default="fp32")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--expected-count", type=int, help="Fail instead of writing if this many valid rows are not found.")
    parser.add_argument("--metadata-output", help="Manifest path; defaults to <output-dir>/metadata.jsonl.")
    parser.add_argument(
        "--lock-file",
        help="Writer lock used by the cache launcher. Defaults to <output-dir>/../../.vae-latent-cache.lock.",
    )
    return parser.parse_args()


def _file_identity(path):
    path = Path(path).resolve()
    stat = path.stat()
    return {"path": str(path), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def _model_identity(vae_dir):
    files = []
    for path in sorted(vae_dir.iterdir()):
        if path.is_file() and (path.name == "config.json" or "safetensors" in path.name):
            stat = path.stat()
            identity = {"name": path.name, "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
            if path.name == "config.json":
                identity["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
            files.append(identity)
    encoded = json.dumps(files, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _cache_fingerprint(args, meta, vae_identity_sha256, index):
    payload = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "index": index,
        "video": _file_identity(meta["video_path"]),
        "video_start_time": meta["video_start_time"],
        "source_num_frames": meta["source_num_frames"],
        "num_frames": meta["num_frames"],
        "target_height": meta["target_height"],
        "target_width": meta["target_width"],
        "fps": args.fps,
        "posterior": args.posterior,
        "posterior_seed": args.seed + index,
        "storage_dtype": args.storage_dtype,
        "vae_identity_sha256": vae_identity_sha256,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _write_records(path, records):
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
    os.replace(temporary, path)


def _source_geometry(args, dataset, index):
    source_record = dataset.cache_source_record(index)
    source_frames = record_value(source_record, "num_frames", "frames")
    if source_frames is None:
        raise ValueError(f"Source row {index} has no num_frames field; its cache fingerprint cannot be recovered.")
    source_num_frames = min(int(source_frames), args.num_frames)
    validate_source_num_frames(source_num_frames)

    meta = dataset.samples[index]["meta"]
    if args.fixed_geometry:
        target_height, target_width = args.height, args.width
    else:
        target_height = int(meta["target_height"])
        target_width = int(meta["target_width"])
    if target_height % 32 or target_width % 32:
        raise ValueError(f"Source row {index} has invalid H3 geometry {target_height}x{target_width}.")

    return source_record, {
        "video_path": meta["video_path"],
        "video_start_time": 0.0,
        "source_num_frames": source_num_frames,
        "num_frames": align_num_frames(source_num_frames),
        "target_height": target_height,
        "target_width": target_width,
    }


def _validate_latent(path, expected_shape, expected_dtype):
    latent = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    if not torch.is_tensor(latent):
        return f"expected Tensor, got {type(latent).__name__}"
    if tuple(latent.shape) != expected_shape:
        return f"expected shape {expected_shape}, got {tuple(latent.shape)}"
    if latent.dtype != expected_dtype:
        return f"expected dtype {expected_dtype}, got {latent.dtype}"
    if not torch.isfinite(latent).all().item():
        return "contains non-finite values"
    return None


def _finalize(args, output_dir):
    dataset = VideoDataset(
        metadata_paths=args.input,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        frame_rate=args.fps,
        fix_frame_rate=False,
        time_division_factor=1,
        time_division_remainder=0,
        geometry_from_metadata=not args.fixed_geometry,
        random_start=False,
        decode_retries=1,
        preserve_records=True,
        max_samples=args.max_samples,
    )
    latent_dir = output_dir / "latents"
    latent_files = set(latent_dir.glob("latent_*.pt"))
    vae_identity_sha256 = _model_identity(resolve_video_vae_dir(args.model_path))
    expected_dtype = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }[args.storage_dtype]

    records = []
    matched_paths = set()
    invalid = []
    for index in range(len(dataset)):
        source_record, meta = _source_geometry(args, dataset, index)
        fingerprint = _cache_fingerprint(args, meta, vae_identity_sha256, index)
        latent_path = latent_dir / f"latent_{index:08d}_{fingerprint[:16]}.pt"
        if not latent_path.is_file():
            continue

        expected_shape = (
            LATENT_CHANNELS,
            video_latent_num_frames(meta["num_frames"]),
            meta["target_height"] // 16,
            meta["target_width"] // 16,
        )
        validation_error = _validate_latent(latent_path, expected_shape, expected_dtype)
        if validation_error:
            invalid.append((latent_path, validation_error))
            continue

        source_record.update(
            {
                "vae_cache_index": index,
                "vae_cache_schema_version": CACHE_SCHEMA_VERSION,
                "vae_cache_fingerprint": fingerprint,
                "video_path": meta["video_path"],
                "latent_path": str(latent_path.relative_to(output_dir)),
                "source_num_frames": meta["source_num_frames"],
                "num_frames": meta["num_frames"],
                "target_height": meta["target_height"],
                "target_width": meta["target_width"],
                "latent_dtype": args.storage_dtype,
                "latent_posterior": args.posterior,
            }
        )
        records.append(source_record)
        matched_paths.add(latent_path)

    orphan_paths = sorted(latent_files - matched_paths - {path for path, _ in invalid})
    if invalid:
        details = "\n".join(f"  {path}: {error}" for path, error in invalid[:10])
        raise RuntimeError(f"Found {len(invalid)} invalid latent files:\n{details}")
    if orphan_paths:
        details = "\n".join(f"  {path}" for path in orphan_paths[:10])
        raise RuntimeError(f"Found {len(orphan_paths)} latent files that do not match this cache configuration:\n{details}")
    if args.expected_count is not None and len(records) != args.expected_count:
        raise RuntimeError(f"Expected {args.expected_count} valid latent files, found {len(records)}.")

    cache_indices = [record["vae_cache_index"] for record in records]
    latent_paths = [record["latent_path"] for record in records]
    if cache_indices != sorted(set(cache_indices)) or len(latent_paths) != len(set(latent_paths)):
        raise RuntimeError("Finalized records contain duplicate or unsorted cache identities.")

    metadata_path = Path(args.metadata_output).resolve() if args.metadata_output else output_dir / "metadata.jsonl"
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    _write_records(metadata_path, records)
    print(
        f"Wrote {len(records)} partial rows from {len(dataset)} source rows to {metadata_path}; "
        f"missing={len(dataset) - len(records)}",
        flush=True,
    )


def main():
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    lock_path = Path(args.lock_file).resolve() if args.lock_file else output_dir.parents[1] / ".vae-latent-cache.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_mode = "r" if lock_path.exists() else "a"
    with lock_path.open(lock_mode, encoding="utf-8") as lock_handle:
        try:
            fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"VAE cache writer still holds {lock_path}; stop it before finalizing.") from error
        _finalize(args, output_dir)


if __name__ == "__main__":
    main()
