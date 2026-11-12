"""Generate paired GAN positives with H3-Image's released six-step teacher.

Each rank owns independent samples and one frozen BF16 teacher. The cached
FP32 tensors are already VAE-normalized diffusion latents; no VAE, image
download, decoding or additional sigma shift is involved.
"""

import argparse
import json
import os
import sys
import time
from datetime import timedelta
from pathlib import Path

from common import atomic_json, atomic_jsonl, establish_recipe, fingerprint, read_prompts
from prepare_cache import validate_cache


def read_conditions(manifest, expected_samples):
    rows = read_prompts(manifest, expected_samples)
    for row in rows:
        height, width = row["target_height"], row["target_width"]
        if row["task"] != "t2i" or row["num_frames"] != 1 or min(height, width) <= 0 or height % 32 or width % 32 or row["latent_shape"] != [24, 1, height // 16, width // 16]:
            raise ValueError(f"Invalid H3-Image condition geometry: id={row['id']}")
        if not row.get("cache_fingerprint") or not (manifest.parent / row["condition_path"]).is_file():
            raise ValueError(f"Missing H3-Image condition cache: id={row['id']}")
    return rows


def teacher_recipe(model_path, rows, seed, attention_backend):
    root = model_path.parent if model_path.name == "transformer_diffusers" else model_path
    transformer = root / "transformer_diffusers"
    schedule = json.loads((root / "pdd_config.json").read_text())
    if schedule["num_inference_steps"] != 6 or len(schedule["sigmas"]) != 7:
        raise ValueError("GAN positives require the released six-step H3-Image teacher")
    files = {
        "transformer_config": transformer / "config.json",
        "transformer_index": transformer / "diffusion_pytorch_model.safetensors.index.json",
        "pdd_config": root / "pdd_config.json",
    }
    return {
        "schema_version": 1,
        "source": "teacher_generated",
        "model": "minimax_h3_image",
        "image_head_mode": "released",
        "condition_manifest_sha256": fingerprint(rows),
        "component_fingerprints": {key: fingerprint(json.loads(path.read_text())) for key, path in files.items()},
        "teacher_steps": 6,
        "released_sigmas": schedule["sigmas"],
        "sigma_rounding": "float32(1 - float32(1 - sigma))",
        "guidance_scale": 1.0,
        "additional_shift": None,
        "parameter_dtype": "bf16",
        "latent_dtype": "fp32",
        "vae_normalized": True,
        "attention_backend": attention_backend,
        "seed": seed,
        "seed_scheme": "sha256(h3-image-gan-teacher, seed, id, condition_fingerprint) mod 2**63",
    }


def real_record(row, manifest, output_dir, recipe_sha, seed):
    condition_path = (manifest.parent / row["condition_path"]).resolve()
    record = dict(row)
    record.update(
        condition_path=os.path.relpath(condition_path, output_dir),
        image_latent_path=f"image_latents/{row['id']:08d}.pt",
        source="teacher_generated",
        vae_normalized=True,
        teacher_steps=6,
        teacher_seed=int(fingerprint(["h3-image-gan-teacher", seed, row["id"], row["cache_fingerprint"]])[:16], 16) % (2**63),
        real_sample_key=fingerprint(["h3-image-gan-teacher", row["id"], row["cache_fingerprint"]]),
        real_cache_fingerprint=fingerprint([recipe_sha, row]),
    )
    return record


def validate_latent(payload, record):
    import torch

    latent = payload["latents"]
    if (
        payload["real_sample_key"] != record["real_sample_key"]
        or payload["real_cache_fingerprint"] != record["real_cache_fingerprint"]
        or tuple(latent.shape) != tuple(record["latent_shape"])
        or latent.dtype != torch.float32
        or not bool(torch.isfinite(latent).all())
    ):
        raise ValueError(f"Invalid or stale H3-Image GAN latent: id={record['id']}")


def load_teacher(model_path, attention_backend):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from lightx2v_train.model_zoo.minimax_h3_image.model import MiniMaxH3ImageModel

    model = MiniMaxH3ImageModel(
        {
            "model": {
                "name": "minimax_h3_image",
                "pretrained_model_name_or_path": str(model_path),
                "image_head_mode": "released",
                "running_dtype": "bf16",
                "transformer_param_dtype": "bf16",
                "latent_dtype": "fp32",
                "attention_backend": attention_backend,
            }
        }
    )
    model.load_components(load_transformer=True, load_vae=False, load_condition_encoder=False)
    model.transformer.requires_grad_(False).eval()
    return model


def generate_teacher_latent(model, condition, record):
    import torch

    if model.image_head_mode != "released":
        raise ValueError("GAN positives must use the released six-head teacher")
    sigmas = model.pdd_schedule.physical_sigmas(device=model.device)
    if sigmas.numel() != 7:
        raise ValueError("GAN positives require exactly six teacher updates")
    generator = torch.Generator(device=model.device).manual_seed(record["teacher_seed"])
    with torch.inference_mode():
        state = torch.randn((1, *record["latent_shape"]), device=model.device, dtype=torch.float32, generator=generator)
        condition = model.prepare_text_condition(condition)
        for current, following in zip(sigmas[:-1], sigmas[1:]):
            # Adapter returns noise-minus-clean; the official head returns
            # clean-minus-noise. A descending-sigma Euler update is equivalent.
            velocity = model.predict_denoiser_output(state, current, condition)
            state = state + (following - current) * velocity.float()
        return state.squeeze(0).to(device="cpu", dtype=torch.float32)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--condition-cache", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-samples", type=int, default=10000)
    parser.add_argument("--expected-world-size", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--attention-backend", default="torch_sdpa")
    args = parser.parse_args(argv)
    import torch
    import torch.distributed as dist

    rank, world, local_rank = (int(os.environ.get(key, default)) for key, default in (("RANK", 0), ("WORLD_SIZE", 1), ("LOCAL_RANK", 0)))
    if args.expected_world_size is not None and args.expected_world_size != world:
        raise ValueError(f"Expected {args.expected_world_size} teacher ranks, got {world}")
    torch.set_num_threads(4)
    manifest = args.condition_cache.resolve()
    if manifest.is_dir():
        manifest /= "metadata.jsonl"
    root = args.output_dir.resolve()
    rows = read_conditions(manifest, args.expected_samples)
    recipe = teacher_recipe(args.model_path.resolve(), rows, args.seed, args.attention_backend)
    recipe_sha = fingerprint(recipe)
    records = [real_record(row, manifest, root, recipe_sha, args.seed) for row in rows]
    if world > 1:
        dist.init_process_group("gloo", timeout=timedelta(days=3))
    if rank == 0:
        (root / "image_latents").mkdir(parents=True, exist_ok=True)
        establish_recipe(root / "real_cache_recipe.json", recipe)
    if world > 1:
        dist.barrier()
    pending = []
    for index, record in enumerate(records):
        if index % world != rank:
            continue
        validate_cache(root / record["condition_path"], record)
        path = root / record["image_latent_path"]
        if path.is_file():
            validate_latent(torch.load(path, map_location="cpu", weights_only=True), record)
        else:
            pending.append(record)
    print(f"[gan-real] rank={rank}/{world} pending={len(pending)} source=teacher_generated steps=6", flush=True)
    if pending:
        torch.cuda.set_device(local_rank)
        model = load_teacher(args.model_path, args.attention_backend)
        started = time.monotonic()
        for index, record in enumerate(pending):
            condition = torch.load(root / record["condition_path"], map_location="cpu", weights_only=True)["conditioning"]["positive"]
            latent = generate_teacher_latent(model, condition, record)
            payload = {"latents": latent, "real_sample_key": record["real_sample_key"], "real_cache_fingerprint": record["real_cache_fingerprint"]}
            validate_latent(payload, record)
            path = root / record["image_latent_path"]
            temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
            torch.save(payload, temporary)
            temporary.replace(path)
            elapsed = time.monotonic() - started
            print(f"[gan-real] rank={rank}/{world} completed={index + 1}/{len(pending)} id={record['id']} seconds_per_sample={elapsed / (index + 1):.2f}", flush=True)
        del model
        torch.cuda.empty_cache()
    if world > 1:
        dist.barrier()
    if rank == 0:
        missing = [row["image_latent_path"] for row in records if not (root / row["image_latent_path"]).is_file()]
        if missing:
            raise ValueError(f"Missing teacher-generated GAN latents: {missing[:5]}")
        atomic_jsonl(root / "metadata.jsonl", records)
        atomic_json(root / "real.complete.json", {"count": len(records), "recipe_sha256": recipe_sha, "manifest_sha256": fingerprint(records), "source": "teacher_generated"})
        print(f"[gan-real] complete={len(records)} manifest={root / 'metadata.jsonl'}", flush=True)
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
