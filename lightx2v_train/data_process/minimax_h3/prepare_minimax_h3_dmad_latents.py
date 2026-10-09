#!/usr/bin/env python3
"""Encode ground-truth AV into identity-bound H3 DMAD real target latents.

Targets are JSONL/JSON/CSV rows with a unique source_id and target_video_path
(video_path is also accepted); --identity-field selects a different source ID
column. An optional audio_path supplies a separate real soundtrack. No audio,
short media, and orientation mismatches fail: silence is never substituted.

The condition metadata owns the exact prompt/reference fingerprint and target
geometry. Source-ID matching must be one-to-one. Optional target fingerprint,
condition_path, and geometry fields are checked, never silently overwritten.
Only real-data pairing can bind raw sources this way; generated teacher data
must already have been generated from the exact Ref2AV condition.

Outputs are normalized, packed .pt targets and metadata.jsonl, plus the exact
condition_metadata.jsonl subset for the strict DMAD join. Outputs are new only;
no existing files are replaced. --dry-run validates metadata without importing
torch/diffusers or loading models. --rank/--world-size partition rows without
duplication; distributed workers use output-dir/shard_NNNNN directories.
"""

import argparse
import hashlib
import importlib.util
import json
import os
import sys
import tempfile
from pathlib import Path

TRAIN_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_MODULE = TRAIN_ROOT / "lightx2v_train/data/minimax_h3_dmad_manifest.py"
_SPEC = importlib.util.spec_from_file_location("dmad_manifest_helpers", MANIFEST_MODULE)
manifest = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(manifest)


def _source_id(row, key):
    value = row.get(key)
    if isinstance(value, bool) or not isinstance(value, (str, int)) or not str(value).strip():
        raise ValueError(f"Real-target pairing requires an explicit non-empty {key}; row numbers are not identities.")
    return str(value)


def prepare_jobs(conditions, targets, *, identity_field="source_id", max_samples=None, rank=0, world_size=1):
    if world_size <= 0 or not 0 <= rank < world_size:
        raise ValueError("Require world_size > 0 and 0 <= rank < world_size.")
    if max_samples is not None and max_samples <= 0:
        raise ValueError("max_samples must be positive.")
    condition_path, condition_rows = manifest.read_manifest(conditions)
    targets_path, target_rows = manifest.read_manifest(targets)
    targets_by_id = {}
    for target in target_rows:
        source_id = _source_id(target, identity_field)
        if source_id in targets_by_id:
            raise ValueError(f"Duplicate real target source ID: {source_id}")
        targets_by_id[source_id] = target
    seen_sources, seen_fingerprints, seen_conditions = set(), set(), set()
    jobs = []
    for index, original in enumerate(condition_rows):
        condition = dict(original)
        source_id = _source_id(condition, "source_id")
        fingerprint = manifest.condition_fingerprint(condition)
        condition["condition_path"] = manifest.resolve_manifest_path(condition.get("condition_path"), condition_path.parent, "condition_path")
        if source_id in seen_sources or fingerprint in seen_fingerprints or condition["condition_path"] in seen_conditions:
            raise ValueError(f"Ambiguous/duplicate condition identity for real source {source_id}; select one condition per source.")
        seen_sources.add(source_id)
        seen_fingerprints.add(fingerprint)
        seen_conditions.add(condition["condition_path"])
        geometry = manifest.target_geometry(condition)
        condition.update(zip(manifest.GEOMETRY_KEYS, geometry))
        if max_samples is not None and index >= max_samples:
            continue
        if index % world_size != rank:
            continue
        target = targets_by_id.get(source_id)
        if target is None:
            raise ValueError(f"Missing ground-truth AV target for source_id={source_id!r}; reference images are not real target videos.")
        if "cache_fingerprint" in target:
            manifest.require_matching_identity(condition, target, "real source")
        if "condition_path" in target:
            declared = manifest.resolve_manifest_path(target["condition_path"], targets_path.parent, "condition_path")
            if declared != condition["condition_path"]:
                raise ValueError(f"Real source condition_path mismatch for {source_id}.")
        if any(key in target for key in (*manifest.GEOMETRY_KEYS, "num_frames")):
            if manifest.target_geometry(target) != geometry:
                raise ValueError(f"Real source target geometry mismatch for {source_id}.")
        video = manifest.resolve_manifest_path(target.get("target_video_path", target.get("video_path")), targets_path.parent, "target_video_path/video_path")
        audio = manifest.resolve_manifest_path(target["audio_path"], targets_path.parent, "audio_path") if target.get("audio_path") else video
        # Make optional cache paths independent of the new output location.
        for key in ("negative_condition_path", "video_latent_path", "audio_latent_path"):
            if condition.get(key):
                condition[key] = manifest.resolve_manifest_path(condition[key], condition_path.parent, key)
        negative = condition_path.parent / "negative_condition.pt"
        if not condition.get("negative_condition_path") and negative.is_file():
            condition["negative_condition_path"] = str(negative.resolve())
        jobs.append({"condition": condition, "video_path": video, "audio_path": audio})
    if not jobs:
        raise ValueError(f"No real targets assigned to rank {rank}/{world_size}.")
    return jobs


def _atomic_tensor_save(torch, payload, output):
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=output.parent, prefix=f".{output.name}.", suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, output)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _condition_identity(torch, job):
    from lightx2v_train.data.minimax_h3_cache_dataset import _condition_payload
    from lightx2v_train.data.minimax_h3_dmad_dataset import _finite_tensors

    row = job["condition"]
    condition = torch.load(row["condition_path"], map_location="cpu", weights_only=False)
    manifest.require_matching_identity(row, condition, "real source condition cache")
    positive = _condition_payload(condition)
    if manifest.target_geometry(positive) != manifest.target_geometry(row):
        raise ValueError("Real source condition cache geometry does not match its manifest.")
    if positive.get("task") not in ("ref2av", "ref2va"):
        raise ValueError("DMAD real preparation requires an actual Ref2AV cache.")
    _finite_tensors(condition, "real source condition cache")


def encode_jobs(jobs, *, model_path, output_dir, device):
    # All model/media imports remain behind metadata-only preflight.
    sys.path.insert(0, str(TRAIN_ROOT))
    import torch
    from diffusers import AutoencoderKLMiniMaxH3, AutoencoderKLMiniMaxH3Audio

    from lightx2v_train.data.minimax_h3_dmad_dataset import pack_target_payload

    helper_path = Path(__file__).with_name("encode_minimax_h3_teacher_av_latents.py")
    spec = importlib.util.spec_from_file_location("h3_existing_av_encoder", helper_path)
    encoder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(encoder)
    # Refuse any output-dir reuse, including partial output from an earlier
    # failed run. A fresh destination is explicit and cannot mix old identities.
    output_dir.mkdir(parents=True, exist_ok=False)
    latent_dir = output_dir / "latents"
    latent_dir.mkdir()
    for job in jobs:
        _condition_identity(torch, job)
    target_device = torch.device(device)
    video_vae = AutoencoderKLMiniMaxH3.from_pretrained(model_path, subfolder="vae", torch_dtype=torch.float32, local_files_only=True).to(target_device).eval()
    audio_vae = AutoencoderKLMiniMaxH3Audio.from_pretrained(model_path, subfolder="audio_vae", torch_dtype=torch.float32, local_files_only=True).to(target_device).eval()
    records = []
    for index, job in enumerate(jobs):
        row = job["condition"]
        height, width, frames = manifest.target_geometry(row)
        # Check/decode both modalities before spending video-VAE compute.
        audio_frames = int(round(frames / 24 * 40))
        waveform = encoder.decode_audio(Path(job["audio_path"]), audio_frames * 800)
        pixels = encoder.decode_video(Path(job["video_path"]), height, width, frames, 24.0)
        video = encoder.encode_video_latent(video_vae, pixels, target_device, "mode", 0)
        del pixels
        audio = encoder.encode_audio_latent(audio_vae, waveform, target_device).transpose(1, 2).contiguous()
        del waveform
        metadata = {
            "normalized": True,
            "cache_fingerprint": row["cache_fingerprint"],
            "condition_path": row["condition_path"],
            "source_id": row["source_id"],
            "target_height": height,
            "target_width": width,
            "target_num_frames": frames,
            "target_source_video": job["video_path"],
            "target_source_audio": job["audio_path"],
            "source_kind": "real",
            "posterior_mode": "mode",
            "vae_model_path": str(model_path),
        }
        for key in manifest.SOURCE_ID_KEYS:
            if key in row:
                metadata[key] = row[key]
        packed = pack_target_payload({**metadata, "video": video, "audio": audio}, row, label="encoded real", base_dir=output_dir)
        stem = hashlib.sha256(row["cache_fingerprint"].encode("utf-8")).hexdigest()
        destination = latent_dir / f"{stem}.pt"
        _atomic_tensor_save(torch, {**metadata, **packed}, destination)
        records.append({**row, "real_latent_path": str(destination), "normalized": True, "target_source_video": job["video_path"], "target_source_audio": job["audio_path"]})
        print(f"[real {index + 1}/{len(jobs)}] {destination}", flush=True)
        del video, audio, packed
    manifest.write_manifest_atomic(records, output_dir / "metadata.jsonl")
    manifest.write_manifest_atomic([job["condition"] for job in jobs], output_dir / "condition_metadata.jsonl")
    return records


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--conditions", required=True)
    parser.add_argument("--targets", required=True, help="Real source AV rows, not reference image rows.")
    parser.add_argument("--identity-field", default="source_id")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world-size", type=int, default=1)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    try:
        jobs = prepare_jobs(args.conditions, args.targets, identity_field=args.identity_field, max_samples=args.max_samples, rank=args.rank, world_size=args.world_size)
        model_path = Path(args.model_path).expanduser().resolve()
        for subfolder in ("vae", "audio_vae"):
            if not (model_path / subfolder / "config.json").is_file():
                raise FileNotFoundError(f"Missing local H3 {subfolder}/config.json in {model_path}")
        output = Path(args.output_dir).expanduser().resolve()
        if args.world_size > 1:
            output /= f"shard_{args.rank:05d}"
        if output.exists():
            raise FileExistsError(f"Refusing to reuse real-latent output directory: {output}")
        if args.dry_run:
            print(
                json.dumps(
                    {"stage": "real", "rank": args.rank, "world_size": args.world_size, "samples": len(jobs), "output_dir": str(output), "tensor_and_media_validation": "deferred until encoding"},
                    indent=2,
                )
            )
            return 0
        encode_jobs(jobs, model_path=model_path, output_dir=output, device=args.device)
    except (ValueError, TypeError, OSError) as error:
        parser.exit(1, f"DMAD real preparation error: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
