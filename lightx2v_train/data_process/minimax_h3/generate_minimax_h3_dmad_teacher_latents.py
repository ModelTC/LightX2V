#!/usr/bin/env python3
"""Offline, conditioned H3 teacher targets for Ref2AV DMAD (no paid API).

torchrun --standalone --nproc_per_node=8 generate_minimax_h3_dmad_teacher_latents.py
  --conditions /path/to/condition/metadata.jsonl --model-path /path/to/H3
  --output-dir /path/to/teacher

Uses one FSDP group, different conditions per rank, and exactly the same number
of model forwards on all ranks. Last-round padding is never written. Resume
skips only fully cached rounds, to avoid divergent FSDP collectives.
Teacher: 31 Euler evaluations, video/audio shifts 12/3, no CFG/LoRA, BF16 rows.
DMAD student training independently uses stochastic re-noise and shifts 12/2.
"""

import argparse
import fcntl
import hashlib
import json
import math
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def teacher_filename(fingerprint):
    return "teacher_" + hashlib.sha256(fingerprint.encode("utf-8")).hexdigest() + ".pt"


def teacher_seed(fingerprint, base_seed):
    # Stable under subset selection, reordering and a changed world size.
    digest = hashlib.sha256(f"{base_seed}:{fingerprint}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % (2**63 - 1)


def merge_shards(output_dir, world, records):
    """Empty padded ranks are valid; duplicates/missing identities are not."""
    joined = {}
    for rank in range(world):
        with (output_dir / f"metadata.rank{rank}.jsonl").open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                fingerprint = row["cache_fingerprint"]
                if fingerprint in joined:
                    raise ValueError(f"Duplicate teacher output fingerprint: {fingerprint}")
                joined[fingerprint] = row
    if set(joined) != {row["cache_fingerprint"] for row in records}:
        raise ValueError("Teacher output shards do not cover exactly the requested conditions.")
    for row in records:
        actual = joined[row["cache_fingerprint"]]
        if actual["condition_path"] != row["condition_path"]:
            raise ValueError("Teacher output shard condition path mismatch.")
    return [joined[row["cache_fingerprint"]] for row in records]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--conditions", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=31)
    parser.add_argument("--video-shift", type=float, default=12.0)
    parser.add_argument("--audio-shift", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--attention-backend", default="_flash_3_hub")
    args = parser.parse_args()
    if args.steps < 1 or any(not math.isfinite(x) or x <= 0 for x in (args.video_shift, args.audio_shift)):
        parser.error("steps and flow shifts must be positive")
    if args.max_samples is not None and args.max_samples < 1:
        parser.error("--max-samples must be positive")
    return args


def main():
    args = parse_args()
    import torch
    import torch.distributed as dist

    from lightx2v_train.data.minimax_h3_cache_dataset import LatentDataset, _condition_payload
    from lightx2v_train.data.minimax_h3_dmad_dataset import _finite_tensors, pack_target_payload
    from lightx2v_train.data.minimax_h3_dmad_manifest import _unique_rows, read_manifest, require_matching_identity, resolve_manifest_path, target_geometry
    from lightx2v_train.model_capabilities import DistributionMatchingCapability, ParallelCapability
    from lightx2v_train.model_zoo import build_loaded_model
    from lightx2v_train.runtime.distributed import cleanup_distributed, init_distributed
    from lightx2v_train.schedulers import DMDFlowMatchingScheduler

    source, records = read_manifest(args.conditions)
    records = records[: args.max_samples] if args.max_samples else records
    for row in records:
        row["condition_path"] = resolve_manifest_path(row.get("condition_path"), source.parent, "condition_path")
        target_geometry(row)
        if not row.get("cache_fingerprint"):
            raise ValueError("Teacher generation needs exact condition cache_fingerprint.")
    _unique_rows(records, str(source))
    world = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))
    config = {
        "model": {
            "name": "minimax_h3_ref2av",
            "pretrained_model_name_or_path": str(args.model_path),
            "running_dtype": "bf16",
            "transformer_param_dtype": "bf16",
            "local_files_only": True,
            "use_autocast": False,
            "attention_backend": args.attention_backend,
            "capabilities": {"distribution_matching": {"geometry_from_metadata": True, "video_flow_shift": args.video_shift, "audio_flow_shift": args.audio_shift}},
        },
        "distributed": {
            "backend": "nccl",
            "timeout_minutes": 180,
            "sequence_parallel": {"enabled": False, "size": 1},
            "fsdp2": {
                "enabled": world > 1,
                "size": world,
                "stream_load_pretrained": world > 1,
                "reshard_after_forward": {"root_reshard": False, "block_reshard": True},
                "mixed_precision": {"param_dtype": "bf16", "reduce_dtype": "fp32", "cast_forward_inputs": False},
            },
        },
        "scheduler": {"num_train_timesteps": 1000, "time_shift_settings": {"do_time_shift": False}},
        "training": {"dmd": {"num_inference_steps": args.steps}},
    }
    init_distributed(config)
    run_lock = None
    try:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        lock_error = [None]
        if rank == 0:
            run_lock = (args.output_dir / ".teacher.lock").open("a+")
            try:
                fcntl.flock(run_lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                lock_error[0] = f"Another teacher generator owns {args.output_dir}"
        if dist.is_initialized():
            dist.broadcast_object_list(lock_error, src=0)
        if lock_error[0]:
            raise RuntimeError(lock_error[0])
        model = build_loaded_model(config, load_transformer=True, load_vae=False, load_condition_encoder=False)
        capability = model.capabilities.require(DistributionMatchingCapability)
        capability.denoiser().requires_grad_(False)
        model.capabilities.require(ParallelCapability).apply(config)
        capability.set_training(False)
        scheduler = DMDFlowMatchingScheduler(config)
        scheduler.set_timesteps(args.steps, device=capability.device)
        # Reuse the existing condition loader/collator, without real/teacher targets.
        dataset = LatentDataset(data_paths=str(source), defer_latent_loading=True, max_samples=args.max_samples)
        recipe = {
            "model_path": str(args.model_path.resolve()),
            "steps": args.steps,
            "video_shift": args.video_shift,
            "audio_shift": args.audio_shift,
            "base_seed": args.seed,
            "seed_policy": "condition_fingerprint_sha256",
            "sampler": "euler",
            "role": "teacher",
        }
        output_rows = []
        with torch.no_grad():
            for offset in range(0, len(records), world):
                index = min(offset + rank, len(records) - 1)
                valid = offset + rank < len(records)
                row = records[index]
                dest = args.output_dir / teacher_filename(row["cache_fingerprint"])
                cached = dest.is_file()
                if cached:
                    previous = torch.load(dest, map_location="cpu", weights_only=True)
                    pack_target_payload(previous, row, label="teacher", base_dir=dest.parent)
                    if previous.get("teacher_recipe") != recipe:
                        raise ValueError(f"Existing teacher recipe differs: {dest}; choose a new output directory.")
                all_cached = torch.tensor(int(cached), device=capability.device)
                if dist.is_initialized():
                    dist.all_reduce(all_cached, op=dist.ReduceOp.MIN)
                if not bool(all_cached):
                    torch.manual_seed(teacher_seed(row["cache_fingerprint"], args.seed))
                    # Verify that metadata really belongs to the condition PT.
                    condition_item = torch.load(row["condition_path"], map_location="cpu", weights_only=True)
                    require_matching_identity(row, condition_item, "teacher condition")
                    _finite_tensors(condition_item, "teacher condition")
                    positive = _condition_payload(condition_item)
                    if target_geometry(positive) != target_geometry(row):
                        raise ValueError("Teacher condition geometry does not match metadata.")
                    del condition_item, positive
                    sample = next(iter(torch.utils.data.DataLoader(torch.utils.data.Subset(dataset, [index]), batch_size=1)))
                    condition, _ = capability.encode_conditions(sample, "", 1.0, lambda x: x)
                    shape = capability.latent_shape(sample, None, lambda x: x)
                    xt = capability.initial_latents(shape, torch.float32, lambda x: x)
                    for step in range(args.steps):
                        sigma = scheduler.sigma_at(step, device=capability.device, dtype=torch.float32)
                        velocity = capability.predict_velocity(xt, sigma, condition)
                        xt, x0 = capability.step(scheduler, velocity, step, xt)
                    if valid and not cached:
                        height, width, frames = target_geometry(row)
                        payload = {
                            **{k: row[k] for k in ("source_id", "source_row_uid", "sample_id") if k in row},
                            "normalized": True,
                            "cache_fingerprint": row["cache_fingerprint"],
                            "condition_path": row["condition_path"],
                            "target_height": height,
                            "target_width": width,
                            "target_num_frames": frames,
                            "teacher_recipe": recipe,
                            "seed": teacher_seed(row["cache_fingerprint"], args.seed),
                            "video": x0.video[0].cpu().to(torch.bfloat16),
                            "audio": x0.audio[0].cpu().to(torch.bfloat16),
                        }
                        pack_target_payload(payload, row, label="teacher", base_dir=dest.parent)
                        temporary = dest.with_suffix(f".tmp.rank{rank}")
                        torch.save(payload, temporary)
                        os.replace(temporary, dest)
                if valid:
                    output_rows.append({**row, "normalized": True, "teacher_latent_path": str(dest.resolve()), "teacher_recipe": recipe})
                print(f"[teacher rank={rank}] round={offset // world + 1} sample={index} cached={cached}", flush=True)
        shard = args.output_dir / f"metadata.rank{rank}.jsonl"
        tmp = shard.with_suffix(".tmp")
        with tmp.open("w", encoding="utf-8") as handle:
            for row in output_rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        os.replace(tmp, shard)
        if dist.is_initialized():
            dist.barrier()
        if rank == 0:
            joined = merge_shards(args.output_dir, world, records)
            tmp = args.output_dir / "metadata.jsonl.tmp"
            with tmp.open("w", encoding="utf-8") as handle:
                for row in joined:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            os.replace(tmp, args.output_dir / "metadata.jsonl")
        if dist.is_initialized():
            dist.barrier()
    finally:
        if run_lock is not None:
            run_lock.close()
        cleanup_distributed()


if __name__ == "__main__":
    main()
