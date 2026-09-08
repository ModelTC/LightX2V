#!/usr/bin/env python
"""Cache normalized MiniMax-H3 latents for decoder distillation."""

import argparse
import hashlib
import json
import os
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist

from lightx2v_train.data.video_dataset import VideoDataset
from lightx2v_train.model_zoo.minimax_h3.vae_data_process import MiniMaxH3VAEDistillationProcessor
from lightx2v_train.model_zoo.native.minimax_h3.vae_geometry import video_latent_num_frames
from lightx2v_train.model_zoo.native.minimax_h3.video_vae import (
    imagenet_preprocess,
    load_minimax_h3_video_vae,
    normalize_video_latents,
    resolve_video_vae_dir,
)


CACHE_SCHEMA_VERSION = 1


def parse_args():
    parser = argparse.ArgumentParser(description="Encode API videos with the released MiniMax-H3 video VAE.")
    parser.add_argument("--input", required=True, help="Source metadata.jsonl.")
    parser.add_argument("--output-dir", required=True, help="Directory for metadata.jsonl and latents/.")
    parser.add_argument("--model-path", required=True, help="MiniMax-H3 model root or video-VAE directory.")
    parser.add_argument("--num-frames", type=int, default=362, help="Maximum sampled frames before H3 alignment.")
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--height", type=int, default=768, help="Fallback height when metadata geometry is disabled.")
    parser.add_argument("--width", type=int, default=1344, help="Fallback width when metadata geometry is disabled.")
    parser.add_argument("--fixed-geometry", action="store_true", help="Ignore target_height/target_width in metadata.")
    parser.add_argument("--posterior", choices=("mode", "sample"), default="mode")
    parser.add_argument("--storage-dtype", choices=("bf16", "fp16", "fp32"), default="fp32")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--batch-size", type=int, default=1, help="Videos encoded together on each GPU.")
    parser.add_argument("--decode-workers", type=int, default=0, help="Background video decoder threads per GPU.")
    parser.add_argument("--decode-prefetch", type=int, default=2, help="Maximum decoded samples queued per GPU.")
    parser.add_argument("--pin-memory", action="store_true", help="Use pinned CPU batches for asynchronous H2D copies.")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _distributed_context():
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    if world_size > 1:
        dist.init_process_group("nccl", timeout=timedelta(hours=24))
    return rank, world_size, torch.device("cuda", local_rank)


def _storage_dtype(name):
    return {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }[name]


def _save_tensor(path, tensor, rank):
    temporary = path.with_name(f".{path.name}.rank{rank}.tmp")
    torch.save(tensor, temporary)
    os.replace(temporary, path)


def _write_records(path, records):
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
    os.replace(temporary, path)


def _file_identity(path):
    path = Path(path).resolve()
    stat = path.stat()
    return {
        "path": str(path),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


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


def _cache_fingerprint(args, sample, vae_identity_sha256, index):
    payload = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "index": index,
        "video": _file_identity(sample["meta"]["video_path"]),
        "video_start_time": sample["meta"]["video_start_time"],
        "source_num_frames": sample["meta"]["source_num_frames"],
        "num_frames": sample["meta"]["num_frames"],
        "target_height": sample["meta"]["target_height"],
        "target_width": sample["meta"]["target_width"],
        "fps": args.fps,
        "posterior": args.posterior,
        "posterior_seed": args.seed + index,
        "storage_dtype": args.storage_dtype,
        "vae_identity_sha256": vae_identity_sha256,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _merge_metadata(output_dir, world_size, expected_rows):
    records = []
    for rank in range(world_size):
        shard_path = output_dir / f"metadata.world{world_size:05d}.rank{rank:05d}.jsonl"
        with shard_path.open("r", encoding="utf-8") as handle:
            records.extend(json.loads(line) for line in handle if line.strip())
    records.sort(key=lambda record: record["vae_cache_index"])
    if len(records) != expected_rows or [record["vae_cache_index"] for record in records] != list(range(expected_rows)):
        raise RuntimeError(f"Expected {expected_rows} cached rows, got {len(records)} unique ordered rows.")
    _write_records(output_dir / "metadata.jsonl", records)


def _prefetched_samples(dataset, indices, workers, prefetch):
    if workers == 0:
        for index in indices:
            yield index, dataset[index]
        return

    index_iterator = iter(indices)
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="video-decode") as executor:
        pending = {}
        for _ in range(prefetch):
            try:
                index = next(index_iterator)
            except StopIteration:
                break
            pending[executor.submit(dataset.__getitem__, index)] = index

        while pending:
            completed, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in completed:
                index = pending.pop(future)
                yield index, future.result()
                try:
                    next_index = next(index_iterator)
                except StopIteration:
                    continue
                pending[executor.submit(dataset.__getitem__, next_index)] = next_index


def _build_cache_item(args, dataset, index, sample, latent_dir, output_dir, vae_identity_sha256):
    cache_fingerprint = _cache_fingerprint(args, sample, vae_identity_sha256, index)
    latent_path = latent_dir / f"latent_{index:08d}_{cache_fingerprint[:16]}.pt"
    expected_shape = (
        24,
        video_latent_num_frames(int(sample["meta"]["num_frames"])),
        int(sample["meta"]["target_height"]) // 16,
        int(sample["meta"]["target_width"]) // 16,
    )
    use_existing = latent_path.is_file() and not args.overwrite
    if use_existing:
        cached = torch.load(latent_path, map_location="cpu", weights_only=True)
        use_existing = torch.is_tensor(cached) and tuple(cached.shape) == expected_shape

    record = dataset.cache_source_record(index)
    record.update(
        {
            "vae_cache_index": index,
            "vae_cache_schema_version": CACHE_SCHEMA_VERSION,
            "vae_cache_fingerprint": cache_fingerprint,
            "video_path": sample["meta"]["video_path"],
            "latent_path": str(latent_path.relative_to(output_dir)),
            "source_num_frames": int(sample["meta"]["source_num_frames"]),
            "num_frames": int(sample["meta"]["num_frames"]),
            "target_height": int(sample["meta"]["target_height"]),
            "target_width": int(sample["meta"]["target_width"]),
            "latent_dtype": args.storage_dtype,
            "latent_posterior": args.posterior,
        }
    )
    return {
        "index": index,
        "video": sample["inputs"]["video"],
        "latent_path": latent_path,
        "expected_shape": expected_shape,
        "use_existing": use_existing,
        "record": record,
    }


def _encode_batch(args, vae, items, device, storage_dtype, rank):
    videos = torch.stack([item["video"] for item in items])
    if args.pin_memory:
        videos = videos.pin_memory()
    videos = videos.to(device=device, dtype=torch.float32, non_blocking=args.pin_memory)

    with torch.inference_mode():
        posterior = vae.encode(imagenet_preprocess(videos)).latent_dist
        if args.posterior == "mode":
            latents = posterior.mode()
        else:
            samples = []
            for batch_index, item in enumerate(items):
                generator = torch.Generator(device=device).manual_seed(args.seed + item["index"])
                noise = torch.randn(
                    posterior.mean[batch_index : batch_index + 1].shape,
                    generator=generator,
                    device=device,
                    dtype=posterior.mean.dtype,
                )
                samples.append(
                    posterior.mean[batch_index : batch_index + 1]
                    + posterior.std[batch_index : batch_index + 1] * noise
                )
            latents = torch.cat(samples)
        latents = normalize_video_latents(vae, latents.float()).to(device="cpu", dtype=storage_dtype)

    for item, latent in zip(items, latents):
        if tuple(latent.shape) != item["expected_shape"]:
            raise RuntimeError(
                f"Encoded latent shape {tuple(latent.shape)} does not match {item['expected_shape']} "
                f"at row {item['index']}."
            )
        _save_tensor(item["latent_path"], latent, rank)


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("MiniMax-H3 latent caching requires CUDA.")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1.")
    if args.decode_workers < 0:
        raise ValueError("--decode-workers cannot be negative.")
    if args.decode_prefetch < 1:
        raise ValueError("--decode-prefetch must be at least 1.")

    rank, world_size, device = _distributed_context()
    output_dir = Path(args.output_dir).resolve()
    latent_dir = output_dir / "latents"
    output_dir.mkdir(parents=True, exist_ok=True)
    latent_dir.mkdir(parents=True, exist_ok=True)

    processor = MiniMaxH3VAEDistillationProcessor()
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
        sample_processor=processor,
        max_samples=args.max_samples,
    )

    vae_dir = resolve_video_vae_dir(args.model_path)
    vae_identity_sha256 = _model_identity(vae_dir)
    vae = load_minimax_h3_video_vae(args.model_path, torch_dtype=torch.float32, local_files_only=True)
    vae.decoder = None
    vae.post_quant_conv = None
    vae.requires_grad_(False).eval().to(device)
    storage_dtype = _storage_dtype(args.storage_dtype)
    records = []
    pending_batch = []
    processed = 0
    encoded = 0
    reused = 0
    next_report = 25

    def finish_batch():
        nonlocal encoded, processed, next_report
        if not pending_batch:
            return
        _encode_batch(args, vae, pending_batch, device, storage_dtype, rank)
        records.extend(item["record"] for item in pending_batch)
        encoded += len(pending_batch)
        processed += len(pending_batch)
        while processed >= next_report:
            print(
                f"rank={rank} processed={processed} encoded={encoded} reused={reused} "
                f"last_index={pending_batch[-1]['index']}",
                flush=True,
            )
            next_report += 25
        pending_batch.clear()

    indices = range(rank, len(dataset), world_size)
    for index, sample in _prefetched_samples(dataset, indices, args.decode_workers, args.decode_prefetch):
        item = _build_cache_item(args, dataset, index, sample, latent_dir, output_dir, vae_identity_sha256)
        if item["use_existing"]:
            records.append(item["record"])
            reused += 1
            processed += 1
            while processed >= next_report:
                print(
                    f"rank={rank} processed={processed} encoded={encoded} reused={reused} last_index={index}",
                    flush=True,
                )
                next_report += 25
            continue

        if pending_batch and tuple(item["video"].shape) != tuple(pending_batch[0]["video"].shape):
            finish_batch()
        pending_batch.append(item)
        if len(pending_batch) == args.batch_size:
            finish_batch()
    finish_batch()

    shard_path = output_dir / f"metadata.world{world_size:05d}.rank{rank:05d}.jsonl"
    _write_records(shard_path, records)
    if world_size > 1:
        dist.barrier()
    if rank == 0:
        _merge_metadata(output_dir, world_size, len(dataset))
        print(f"Wrote {len(dataset)} rows to {output_dir / 'metadata.jsonl'}", flush=True)
    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
