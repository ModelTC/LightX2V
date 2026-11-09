#!/usr/bin/env python3
"""Encode a shared T2AV prompt cache once across all torchrun global ranks."""

import argparse
import hashlib
import json
import os
import traceback
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timedelta, timezone
from pathlib import Path

import build_minimax_h3_prompt_latents as builder
import torch
import torch.distributed as dist


def prompt_samples(args):
    samples = builder.read_samples(Path(args.prompts), "prompt", "t2av", 0, None)
    if len(samples) != args.expected_samples:
        raise ValueError(f"Expected {args.expected_samples} usable T2AV prompts, got {len(samples)}")
    for sample in samples:
        row = sample["record"]
        if int(row.get("num_frames", 124)) != 124 or float(row.get("fps", 24)) != 24:
            raise ValueError(f"Expected 124 frames at 24 fps: source_index={sample['source_index']}")
    return samples


def merge_shards(output_dir, samples, encode_args, model_descriptor, world_size):
    """Publish only an exact, disjoint cover of the current input namespace."""
    expected = {sample["source_index"]: sample for sample in samples}
    rows = {}
    counts = []
    for rank in range(world_size):
        manifest = output_dir / f"metadata.shard-{rank:05d}-of-{world_size:05d}.jsonl"
        shard = builder._read_json_rows(manifest)
        counts.append(len(shard))
        assigned = {index for index in expected if index % world_size == rank}
        seen = set()
        for row in shard:
            index = row.get("source_index")
            if type(index) is not int or index not in assigned or index in seen or index in rows:
                raise ValueError(f"Unexpected or duplicated source_index={index!r}: {manifest}")
            sample = expected[index]
            fingerprint = builder.descriptor_fingerprint(builder.cache_descriptor(sample, encode_args, model_descriptor))
            relative = f"conditions/condition_{index:08d}.pt"
            if (
                row.get("condition_path") != relative
                or row.get("cache_fingerprint") != fingerprint
                or row.get("source_id") != sample["source_id"]
                or row.get("caption") != sample["prompt"]
                or row.get("task") != "t2av"
                or (row.get("num_frames"), row.get("target_height"), row.get("target_width")) != (124, 768, 1344)
            ):
                raise ValueError(f"Stale or mismatched shard record source_index={index}: {manifest}")
            if not (output_dir / relative).is_file():
                raise FileNotFoundError(output_dir / relative)
            seen.add(index)
            rows[index] = row
        if seen != assigned:
            raise ValueError(f"Incomplete shard {rank}: expected={len(assigned)} actual={len(seen)}")
    if set(rows) != set(expected):
        raise ValueError("Prompt cache shards do not exactly cover this input")
    manifest = output_dir / "metadata.jsonl"
    builder.atomic_write_jsonl(manifest, [rows[index] for index in sorted(rows)])
    # The training loader discovers task/bucket manifests, not the flat
    # preprocessing index. Resolve paths before publishing the nested view.
    training_manifest = output_dir / "t2av" / "landscape" / "metadata.jsonl"
    training_manifest.parent.mkdir(parents=True, exist_ok=True)
    training_rows = [{**rows[index], "condition_path": str((output_dir / rows[index]["condition_path"]).resolve())} for index in sorted(rows)]
    builder.atomic_write_jsonl(training_manifest, training_rows)
    receipt = {
        "schema_version": 1,
        "prompt_count": len(rows),
        "world_size": world_size,
        "shard_counts": counts,
        "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "training_manifest_sha256": hashlib.sha256(training_manifest.read_bytes()).hexdigest(),
        "preprocess_fingerprint": json.loads((output_dir / "preprocess_config.json").read_text())["preprocess_fingerprint"],
    }
    builder.atomic_write_json(output_dir / "prompt_cache.complete.json", receipt)
    return receipt


def run_encoder_logged(encode_argv, log_path):
    """Keep verbose encoding output and tracebacks in this rank's own log."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("x", encoding="utf-8", buffering=1) as handle, redirect_stdout(handle), redirect_stderr(handle):
        try:
            builder.main(encode_argv)
        except Exception:
            traceback.print_exc()
            raise


def initialize_shared_namespace(output_dir, descriptor, rank):
    """Publish once, and propagate failures before any rank starts encoding."""
    status = [None]
    if rank == 0:
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
            builder.initialize_cache_namespace(output_dir, descriptor, allow_replace=False)
        except Exception as exception:
            status[0] = f"{type(exception).__name__}: {exception}"
    dist.broadcast_object_list(status, src=0)
    if status[0]:
        raise RuntimeError(f"Prompt cache namespace initialization failed: {status[0]}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompts", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--expected-samples", type=int, default=2000)
    parser.add_argument("--expected-world-size", type=int, default=32)
    args = parser.parse_args(argv)
    rank, world_size, local_rank = (int(os.environ[key]) for key in ("RANK", "WORLD_SIZE", "LOCAL_RANK"))
    if world_size != args.expected_world_size:
        raise ValueError(f"Expected {args.expected_world_size} global encoders, got {world_size}")
    output_dir = Path(args.output_dir).expanduser().resolve()
    samples = prompt_samples(args)
    encode_argv = [
        args.prompts,
        "--prompt-column",
        "prompt",
        "--task",
        "t2av",
        "--model-path",
        args.model_path,
        "--output-dir",
        str(output_dir),
        "--height",
        "768",
        "--width",
        "1344",
        "--num-frames",
        "124",
        "--dtype",
        "bf16",
        "--num-shards",
        str(world_size),
        "--shard-index",
        str(rank),
        "--device",
        f"cuda:{local_rank}",
    ]
    encode_args = builder.parse_args(encode_argv)
    model_descriptor = builder.model_identity(args.model_path)
    torch.cuda.set_device(local_rank)
    dist.init_process_group("gloo", timeout=timedelta(hours=3))
    try:
        # Every rank validates the same recipe before any expensive Qwen load.
        descriptor = builder.preprocessing_descriptor(encode_args, "t2av", model_descriptor)
        identities = [None] * world_size
        dist.all_gather_object(identities, builder.descriptor_fingerprint(descriptor))
        if len(set(identities)) != 1:
            raise ValueError("Workers disagree on prompt/model/cache preprocessing identity")
        initialize_shared_namespace(output_dir, descriptor, rank)
        count = sum(sample["source_index"] % world_size == rank for sample in samples)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
        log_path = output_dir / "logs" / f"rank-{rank:05d}.{stamp}.{os.getpid()}.log"
        print(f"[prompt-cache] rank={rank}/{world_size} local_rank={local_rank} assigned={count} device=cuda:{local_rank} log={log_path}", flush=True)
        error = None
        try:
            run_encoder_logged(encode_argv, log_path)
        except Exception as exception:
            error = f"rank={rank}: {type(exception).__name__}: {exception}; log={log_path}"
            print(f"[prompt-cache] failed {error}", flush=True)
        else:
            print(f"[prompt-cache] rank={rank}/{world_size} completed={count} log={log_path}", flush=True)
        errors = [None] * world_size
        dist.all_gather_object(errors, error)
        if any(errors):
            raise RuntimeError("Prompt encoding failed: " + "; ".join(value for value in errors if value))
        status = [None]
        if rank == 0:
            try:
                receipt = merge_shards(output_dir, samples, encode_args, model_descriptor, world_size)
                print(f"[prompt-cache] complete={json.dumps(receipt, sort_keys=True)} path={output_dir}", flush=True)
            except Exception as exception:
                status[0] = f"{type(exception).__name__}: {exception}"
        dist.broadcast_object_list(status, src=0)
        if status[0]:
            raise RuntimeError(f"Prompt manifest merge failed: {status[0]}")
        dist.barrier()
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
