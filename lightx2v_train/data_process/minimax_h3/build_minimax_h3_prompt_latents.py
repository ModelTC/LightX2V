#!/usr/bin/env python3
"""Encode one-prompt-per-line text into an H3 latent_dataset cache."""

import argparse
import json
import os
from pathlib import Path

import torch

TEXT_ENCODER_LAYER = 50
TEXT_TAG = 1


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prompts", help="UTF-8 txt file; each non-empty line is one prompt.")
    parser.add_argument("--model-path", default="/data/nvme6/gushiqiao/models/MiniMax-H3")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_prompts(path, start_index=0, max_samples=None):
    with Path(path).expanduser().open("r", encoding="utf-8") as handle:
        prompts = [line.strip() for line in handle if line.strip()]
    prompts = prompts[start_index:]
    if max_samples is not None:
        prompts = prompts[:max_samples]
    if not prompts:
        raise RuntimeError(f"No non-empty prompts found in {path}.")
    return prompts


def atomic_write_jsonl(path, rows):
    path = Path(path)
    temporary = path.with_suffix(f"{path.suffix}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def load_conditioner(model_path, device, dtype):
    try:
        from transformers import Qwen2TokenizerFast, Qwen3VLForConditionalGeneration, Qwen3VLProcessor
    except ImportError as exc:
        raise ImportError("MiniMax-H3 prompt encoding requires a Transformers build with Qwen3-VL support. The model's local_diffusers environment contains the required version.") from exc

    root = Path(model_path).expanduser().resolve()
    text_encoder_path = root / "text_encoder"
    tokenizer_path = root / "tokenizer"
    processor_path = root / "processor"
    for path in (text_encoder_path, tokenizer_path, processor_path):
        if not path.exists():
            raise FileNotFoundError(f"Missing MiniMax-H3 component: {path}")

    load_kwargs = {"dtype": dtype, "local_files_only": True}
    if str(device) != "cpu":
        load_kwargs["device_map"] = {"": str(device)}
    text_encoder = Qwen3VLForConditionalGeneration.from_pretrained(str(text_encoder_path), **load_kwargs)
    text_encoder.requires_grad_(False).eval()
    tokenizer = Qwen2TokenizerFast.from_pretrained(str(tokenizer_path), local_files_only=True)
    processor = Qwen3VLProcessor.from_pretrained(str(processor_path), local_files_only=True)
    return text_encoder, tokenizer, processor


@torch.inference_mode()
def encode_prompt(text_encoder, tokenizer, processor, prompt, device, output_dtype):
    token_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    if not token_ids:
        raise ValueError("The tokenizer produced no tokens for a non-empty prompt.")
    input_ids = torch.tensor([token_ids], dtype=torch.long, device=device)
    mm_token_type_ids = torch.tensor(processor.create_mm_token_type_ids([token_ids]), dtype=torch.long, device=device)
    outputs = text_encoder.model(
        input_ids=input_ids,
        attention_mask=torch.ones_like(input_ids),
        mm_token_type_ids=mm_token_type_ids,
        pixel_values=None,
        image_grid_thw=None,
        use_cache=False,
        output_hidden_states=True,
    )
    if len(outputs.hidden_states) <= TEXT_ENCODER_LAYER:
        raise RuntimeError(f"Qwen3-VL returned {len(outputs.hidden_states)} hidden states; H3 requires index {TEXT_ENCODER_LAYER}.")
    prompt_embeds = outputs.hidden_states[TEXT_ENCODER_LAYER].to(dtype=output_dtype).cpu()
    text_token_tags = torch.full((len(token_ids),), TEXT_TAG, dtype=torch.long)
    return {"prompt_embeds": prompt_embeds, "text_token_tags": text_token_tags}


def main():
    args = parse_args()
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    prompts = read_prompts(args.prompts, args.start_index, args.max_samples)
    output_dir = Path(args.output_dir).expanduser().resolve()
    condition_dir = output_dir / "conditions"
    condition_dir.mkdir(parents=True, exist_ok=True)
    text_encoder, tokenizer, processor = load_conditioner(args.model_path, args.device, dtype)

    rows = []
    for local_index, prompt in enumerate(prompts):
        source_index = args.start_index + local_index
        relative_path = Path("conditions") / f"condition_{source_index:08d}.pt"
        output_path = output_dir / relative_path
        if args.overwrite or not output_path.is_file():
            condition = encode_prompt(text_encoder, tokenizer, processor, prompt, args.device, dtype)
            torch.save(
                {
                    "conditioning": {"positive": condition},
                    "prompt": prompt,
                    "source_index": source_index,
                },
                output_path,
            )
        rows.append(
            {
                "id": source_index,
                "caption": prompt,
                "condition_path": str(relative_path),
            }
        )
        print(f"[{local_index + 1}/{len(prompts)}] {output_path}", flush=True)

    manifest = output_dir / "metadata.jsonl"
    atomic_write_jsonl(manifest, rows)
    print(f"Wrote {len(rows)} samples to {manifest}", flush=True)


if __name__ == "__main__":
    main()
