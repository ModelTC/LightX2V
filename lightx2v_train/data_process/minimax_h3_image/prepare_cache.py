"""Cache H3-Image T2I text conditions; PDMD generates image latents online."""

import argparse
import json
import os
from datetime import timedelta
from pathlib import Path

from common import atomic_json, atomic_jsonl, establish_recipe, fingerprint, read_prompts


def load_conditioner(model_path, text_encoder_path, device):
    import torch
    from transformers import Qwen2TokenizerFast, Qwen3VLForConditionalGeneration, Qwen3VLProcessor

    encoder = (
        Qwen3VLForConditionalGeneration.from_pretrained(
            text_encoder_path,
            dtype=torch.bfloat16,
            device_map={"": device},
            local_files_only=True,
            attn_implementation="sdpa",
        )
        .requires_grad_(False)
        .eval()
    )
    # hidden_states[50] must be an intermediate state, before the final norm.
    if encoder.config.text_config.num_hidden_layers <= 50:
        raise ValueError("H3-Image requires the full text encoder with more than 50 layers")
    tokenizer = Qwen2TokenizerFast.from_pretrained(model_path / "tokenizer", local_files_only=True)
    processor = Qwen3VLProcessor.from_pretrained(model_path / "processor", local_files_only=True)
    return encoder, tokenizer, processor


def encode_prompt(prompt, encoder, tokenizer, processor, device):
    import torch

    token_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"][:4096]
    if not token_ids:
        raise ValueError("The text encoder received an empty token sequence")
    ids = torch.tensor([token_ids], device=device, dtype=torch.long)
    token_types = torch.tensor(processor.create_mm_token_type_ids([token_ids]), device=device, dtype=torch.long)
    with torch.inference_mode():
        output = encoder.model(input_ids=ids, attention_mask=torch.ones_like(ids), mm_token_type_ids=token_types, use_cache=False, output_hidden_states=True)
        embeddings = output.hidden_states[50].to(device="cpu", dtype=torch.bfloat16)
    condition = {"prompt_embeds": embeddings, "text_token_tags": torch.ones(len(token_ids), dtype=torch.long)}
    validate_condition(condition)
    return condition


def validate_condition(condition):
    import torch

    embeds, tags = condition["prompt_embeds"], condition["text_token_tags"]
    if (
        embeds.ndim != 3
        or embeds.shape[0] != 1
        or embeds.shape[-1] != 5120
        or not 0 < embeds.shape[1] <= 4096
        or embeds.dtype != torch.bfloat16
        or tags.shape != (embeds.shape[1],)
        or tags.dtype != torch.long
        or not bool((tags == 1).all())
        or not bool(torch.isfinite(embeds).all())
    ):
        raise ValueError("Invalid H3-Image cache: expected finite BF16 [1,L,5120] and L text tags=1")


def cache_record(row, recipe_sha, height, width):
    return {
        "id": row["id"],
        "prompt": row["prompt"],
        "task": "t2i",
        "num_frames": 1,
        "target_height": height,
        "target_width": width,
        "latent_shape": [24, 1, height // 16, width // 16],
        "condition_path": f"conditions/{row['id']:08d}.pt",
        "cache_fingerprint": fingerprint([recipe_sha, row]),
    }


def make_cache(record, condition):
    return {
        "cache_schema_version": 1,
        "cache_fingerprint": record["cache_fingerprint"],
        "inputs": {},
        "conditioning": {"prompt": record["prompt"], "active": "positive", "positive": condition},
        "meta": {key: record[key] for key in ("id", "task", "num_frames", "target_height", "target_width", "latent_shape")},
    }


def validate_cache(path, record):
    import torch

    cache = torch.load(path, map_location="cpu", weights_only=True)
    if cache["cache_schema_version"] != 1 or cache["cache_fingerprint"] != record["cache_fingerprint"] or cache["conditioning"]["prompt"] != record["prompt"]:
        raise ValueError(f"Stale H3-Image condition cache: {path}")
    validate_condition(cache["conditioning"]["positive"])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--text-encoder-path", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--height", type=int, default=2048)
    parser.add_argument("--width", type=int, default=2048)
    parser.add_argument("--expected-samples", type=int, default=10000)
    parser.add_argument("--expected-world-size", type=int)
    args = parser.parse_args(argv)
    import torch
    import torch.distributed as dist

    rank, world, local_rank = (int(os.environ.get(key, default)) for key, default in (("RANK", 0), ("WORLD_SIZE", 1), ("LOCAL_RANK", 0)))
    if args.expected_world_size is not None and world != args.expected_world_size:
        raise ValueError(f"Expected {args.expected_world_size} encoding ranks, got {world}")
    if min(args.height, args.width) <= 0 or args.height % 32 or args.width % 32:
        raise ValueError("Image height and width must be positive multiples of 32")
    torch.set_num_threads(4)
    if world > 1:
        dist.init_process_group("gloo", timeout=timedelta(days=3))
    rows = read_prompts(args.prompts, args.expected_samples)
    encoder_path = args.text_encoder_path or args.model_path / "text_encoder"
    files = {
        "text_encoder_config": encoder_path / "config.json",
        "text_encoder_index": encoder_path / "model.safetensors.index.json",
        "tokenizer": args.model_path / "tokenizer" / "tokenizer.json",
        "processor": args.model_path / "processor" / "preprocessor_config.json",
    }
    recipe = {
        "schema_version": 1,
        "model": "minimax_h3_image",
        "prompts_sha256": fingerprint(rows),
        "height": args.height,
        "width": args.width,
        "text_encoder_layer": 50,
        "text_encoder_max_length": 4096,
        "dtype": "bf16",
        "text_token_tag": 1,
        "component_fingerprints": {key: fingerprint(json.loads(path.read_text())) for key, path in files.items()},
    }
    root = args.output_dir.resolve()
    if rank == 0:
        (root / "conditions").mkdir(parents=True, exist_ok=True)
        establish_recipe(root / "cache_recipe.json", recipe)
    if world > 1:
        dist.barrier()
    records = [cache_record(row, fingerprint(recipe), args.height, args.width) for row in rows]
    pending = []
    for index, record in enumerate(records):
        if index % world == rank:
            path = root / record["condition_path"]
            if path.exists():
                validate_cache(path, record)
            else:
                pending.append(record)
    print(f"[cache] rank={rank}/{world} pending={len(pending)}", flush=True)
    if pending:
        torch.cuda.set_device(local_rank)
        encoder, tokenizer, processor = load_conditioner(args.model_path, encoder_path, f"cuda:{local_rank}")
        for index, record in enumerate(pending):
            condition = encode_prompt(record["prompt"], encoder, tokenizer, processor, f"cuda:{local_rank}")
            path = root / record["condition_path"]
            temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
            torch.save(make_cache(record, condition), temporary)
            temporary.replace(path)
            print(f"[cache] rank={rank}/{world} completed={index + 1}/{len(pending)} id={record['id']} tokens={condition['prompt_embeds'].shape[1]}", flush=True)
        del encoder, tokenizer, processor
    if world > 1:
        dist.barrier()
    if rank == 0:
        missing = [record["condition_path"] for record in records if not (root / record["condition_path"]).is_file()]
        if missing:
            raise ValueError(f"Missing condition caches: {missing[:5]}")
        atomic_jsonl(root / "metadata.jsonl", records)
        atomic_json(root / "cache.complete.json", {"count": len(records), "recipe_sha256": fingerprint(recipe), "manifest_sha256": fingerprint(records)})
        print(f"[cache] complete={len(records)} path={root / 'metadata.jsonl'}", flush=True)
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
