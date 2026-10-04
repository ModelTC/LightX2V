#!/usr/bin/env python3
"""Build MiniMax-H3 text/keyframe conditions for ``minimax_h3_cache_dataset``.

The input may be a one-prompt-per-line text file or JSON/JSONL metadata. JSON
records can describe ``t2av``, ``i2av``, ``l2av`` and ``fl2av`` samples. For
keyframe tasks this script reproduces the released Diffusers presentation,
Qwen3-VL vision encoding and clean keyframe VAE rows. Conditioning noise is
intentionally added later by the DMD trainer for each rollout.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps

TEXT_ENCODER_LAYER = 50
CACHE_SCHEMA_VERSION = 3
VIDEO_TAG = 0
TEXT_TAG = 1
KEYFRAME_ENCODE_SEED = 42
VAE_SPATIAL_SCALE_FACTOR = 16
PATCH_SIZE = (1, 2, 2)
VIDEO_LATENT_CHANNELS = 24
FPS = 24
MIN_DURATION = 5.0
MAX_DURATION = 15.0
PIXEL_MEAN = (0.485, 0.456, 0.406)
PIXEL_STD = (0.229, 0.224, 0.225)
TASK_ALIASES = {
    "t2v": "t2av",
    "t2va": "t2av",
    "t2av": "t2av",
    "i2v": "i2av",
    "i2va": "i2av",
    "i2av": "i2av",
    "l2v": "l2av",
    "l2va": "l2av",
    "l2av": "l2av",
    "flf2v": "fl2av",
    "flf2va": "fl2av",
    "flf2av": "fl2av",
    "fl2v": "fl2av",
    "fl2va": "fl2av",
    "fl2av": "fl2av",
}
TASK_ANCHORS = {
    "t2av": (),
    "i2av": ("first",),
    "l2av": ("last",),
    "fl2av": ("first", "last"),
}
ANCHOR_CODES = {"first": 0, "last": 1}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "prompts",
        nargs="?",
        help="UTF-8 .txt, .jsonl, .json or .csv input. JSON metadata should use caption/prompt fields.",
    )
    parser.add_argument("--metadata", help="Alias for the positional input path.")
    parser.add_argument("--prompt-column", default="caption")
    parser.add_argument("--task", choices=tuple(sorted(TASK_ALIASES)), help="Override/inject one task for all rows.")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda:0", help="Qwen3-VL encoding device.")
    parser.add_argument("--vae-device", help="Keyframe VAE device; defaults to --device after Qwen is unloaded.")
    parser.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--height", type=int, default=768)
    parser.add_argument("--width", type=int, default=1344)
    parser.add_argument("--num-frames", type=int, default=124)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--replace-manifest",
        action="store_true",
        help="Write metadata.jsonl from only this invocation instead of merging chunks.",
    )
    args = parser.parse_args()
    if bool(args.prompts) == bool(args.metadata):
        parser.error("Pass exactly one positional input path or --metadata.")
    args.input_path = Path(args.prompts or args.metadata)
    if args.start_index < 0:
        parser.error("--start-index must be non-negative.")
    if args.max_samples is not None and args.max_samples <= 0:
        parser.error("--max-samples must be positive.")
    if args.height % 32 or args.width % 32:
        parser.error("MiniMax-H3 height and width must be divisible by 32.")
    if args.num_frames % 17 != 5:
        parser.error("MiniMax-H3 num-frames must have the form 17*n+5.")
    duration = args.num_frames / FPS
    if not MIN_DURATION <= duration <= MAX_DURATION:
        parser.error(f"Aligned MiniMax-H3 num-frames must produce {MIN_DURATION:g}-{MAX_DURATION:g}s at {FPS} fps; got {args.num_frames} frames ({duration:g}s).")
    return args


def normalize_task(value: str | None) -> str:
    value = "t2av" if value is None else str(value).strip().lower()
    try:
        return TASK_ALIASES[value]
    except KeyError as error:
        raise ValueError(f"Unsupported MiniMax-H3 task {value!r}; expected one of {sorted(TASK_ALIASES)}.") from error


def _read_json_rows(path: Path) -> list[dict]:
    if path.suffix.lower() == ".jsonl":
        rows = []
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(f"Invalid JSON at {path}:{line_number}: {error}") from error
                if not isinstance(row, dict):
                    raise TypeError(f"Expected a JSON object at {path}:{line_number}.")
                rows.append(row)
        return rows
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            payload = payload.get("data", payload.get("records", [payload]))
        if not isinstance(payload, list) or not all(isinstance(row, dict) for row in payload):
            raise TypeError(f"{path} must contain a JSON object/list of objects.")
        return payload
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def resolve_media_path(value: str, metadata_path: Path, key: str) -> str:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = metadata_path.parent / path
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{key} points to a missing image: {path}")
    return str(path)


def read_samples(
    path: Path,
    prompt_column: str,
    task_override: str | None,
    start_index: int,
    max_samples: int | None,
) -> list[dict]:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Input does not exist: {path}")
    if path.suffix.lower() in {".txt", ".list"}:
        records = [{"caption": line.strip()} for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    elif path.suffix.lower() in {".jsonl", ".json", ".csv"}:
        records = _read_json_rows(path)
    else:
        raise ValueError(f"Unsupported input suffix {path.suffix!r}; use txt/list/json/jsonl/csv.")

    output = []
    for source_index, row in enumerate(records):
        prompt = row.get(prompt_column) or row.get("caption") or row.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            continue
        row_task = row.get("task")
        if task_override and row_task not in (None, "") and normalize_task(row_task) != task_override:
            raise ValueError(f"Row {source_index} declares task={row_task!r}, but --task selects {task_override!r}.")
        task = normalize_task(task_override or row.get("task"))
        sample = {
            "source_index": source_index,
            "source_id": row.get("source_id", row.get("metadata_id", row.get("id", source_index))),
            "task": task,
            "prompt": prompt.strip(),
            "prompt_variant": row.get("prompt_variant"),
            "record": row,
        }
        if task in {"i2av", "fl2av"}:
            value = row.get("first_frame") or row.get("image") or row.get("image_path")
            if not value:
                raise ValueError(f"Row {source_index} task={task} requires first_frame.")
            sample["first_frame"] = resolve_media_path(value, path, "first_frame")
        if task in {"l2av", "fl2av"}:
            value = row.get("last_frame") or row.get("last_image") or row.get("last_image_path")
            if not value:
                raise ValueError(f"Row {source_index} task={task} requires last_frame.")
            sample["last_frame"] = resolve_media_path(value, path, "last_frame")
        output.append(sample)

    output = output[start_index:]
    if max_samples is not None:
        output = output[:max_samples]
    if not output:
        raise RuntimeError(f"No usable samples found in {path}.")
    return output


def atomic_write_jsonl(path: Path, rows: list[dict]) -> None:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":"), default=str) + "\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_write_json(path: Path, payload: dict) -> None:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True, default=str)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_torch_save(payload: dict, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _path_identity(path: Path) -> dict:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def _content_identity(path: Path) -> dict:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return {
        "path": str(path.resolve()),
        "size": path.stat().st_size,
        "sha256": digest.hexdigest(),
    }


def _indexed_weight_files(index_path: Path) -> list[Path]:
    if not index_path.is_file():
        return []
    payload = json.loads(index_path.read_text(encoding="utf-8"))
    weight_map = payload.get("weight_map")
    if not isinstance(weight_map, dict):
        raise ValueError(f"Model index has no weight_map object: {index_path}")
    files = []
    for filename in sorted(set(weight_map.values())):
        path = index_path.parent / filename
        if not path.is_file():
            raise FileNotFoundError(f"Model index {index_path} references a missing shard: {path}")
        files.append(path)
    return files


def model_identity(model_path: str) -> dict:
    root = Path(model_path).expanduser().resolve()
    files: list[Path] = []
    for relative in (
        "text_encoder/config.json",
        "text_encoder/model.safetensors",
        "text_encoder/model.safetensors.index.json",
        "tokenizer/merges.txt",
        "tokenizer/tokenizer.json",
        "tokenizer/tokenizer_config.json",
        "tokenizer/vocab.json",
        "processor/merges.txt",
        "processor/preprocessor_config.json",
        "processor/tokenizer.json",
        "processor/tokenizer_config.json",
        "processor/video_preprocessor_config.json",
        "processor/vocab.json",
        "vae/config.json",
        "vae/diffusion_pytorch_model.safetensors",
        "vae/diffusion_pytorch_model.safetensors.index.json",
    ):
        path = root / relative
        if path.is_file():
            files.append(path)
            if path.name.endswith(".safetensors.index.json"):
                files.extend(_indexed_weight_files(path))

    text_config_path = root / "text_encoder" / "config.json"
    vae_config_path = root / "vae" / "config.json"
    if not text_config_path.is_file() or not vae_config_path.is_file():
        raise FileNotFoundError(f"MiniMax-H3 preprocessing requires text_encoder/config.json and vae/config.json below {root}.")
    text_config = json.loads(text_config_path.read_text(encoding="utf-8"))
    vae_config = json.loads(vae_config_path.read_text(encoding="utf-8"))
    text_hidden_size = int(text_config.get("text_config", text_config).get("hidden_size", 0))
    video_latent_channels = int(vae_config.get("latent_channels", 0))
    if text_hidden_size <= 0:
        raise ValueError(f"Could not determine Qwen hidden_size from {text_config_path}.")
    if video_latent_channels != VIDEO_LATENT_CHANNELS:
        raise ValueError(f"MiniMax-H3 VAE latent_channels={video_latent_channels}, expected {VIDEO_LATENT_CHANNELS}: {vae_config_path}")
    unique_files = {str(path.resolve()): path for path in files}
    return {
        "root": str(root),
        "text_hidden_size": text_hidden_size,
        "video_latent_channels": video_latent_channels,
        # Config/index files and every referenced weight shard are included.
        # Size+mtime is deliberate: hashing ~80 GB of local weights per run is
        # prohibitively expensive while still detecting normal replacements.
        "component_files": [_path_identity(unique_files[key]) for key in sorted(unique_files)],
    }


def preprocessing_descriptor(
    args: argparse.Namespace,
    task_override: str | None,
    model: dict,
) -> dict:
    return {
        "schema_version": CACHE_SCHEMA_VERSION,
        "input": _content_identity(args.input_path.expanduser().resolve()),
        "prompt_column": args.prompt_column,
        "task_override": task_override,
        "target_height": args.height,
        "target_width": args.width,
        "target_num_frames": args.num_frames,
        "dtype": args.dtype,
        "text_encoder_layer": TEXT_ENCODER_LAYER,
        "keyframe_encode_seed": KEYFRAME_ENCODE_SEED,
        "model": model,
    }


def initialize_cache_namespace(
    output_dir: Path,
    descriptor: dict,
    *,
    allow_replace: bool,
) -> None:
    identity_path = output_dir / "preprocess_config.json"
    fingerprint = descriptor_fingerprint(descriptor)
    expected = {
        "cache_schema_version": CACHE_SCHEMA_VERSION,
        "preprocess_fingerprint": fingerprint,
        "preprocess_config": descriptor,
    }
    if identity_path.is_file():
        actual = json.loads(identity_path.read_text(encoding="utf-8"))
        if actual.get("preprocess_fingerprint") == fingerprint:
            return
        if not allow_replace:
            raise RuntimeError(f"Output directory belongs to a different input/model/geometry: {output_dir}. Use a new --output-dir, or intentionally reset it with --overwrite --replace-manifest.")
    elif (output_dir / "metadata.jsonl").exists() or any((output_dir / "conditions").glob("*.pt")):
        if not allow_replace:
            raise RuntimeError(
                f"Output directory contains legacy caches without preprocess_config.json: {output_dir}. Use a new --output-dir, or intentionally adopt/reset it with --overwrite --replace-manifest."
            )
    atomic_write_json(identity_path, expected)


def cache_descriptor(sample: dict, args: argparse.Namespace, model: dict) -> dict:
    media = {}
    for key in ("first_frame", "last_frame"):
        if key in sample:
            media[key] = _path_identity(Path(sample[key]))
    return {
        "schema_version": CACHE_SCHEMA_VERSION,
        "source_index": sample["source_index"],
        "source_id": sample["source_id"],
        "prompt": sample["prompt"],
        "prompt_variant": sample.get("prompt_variant"),
        "task": sample["task"],
        "keyframe_anchors": list(TASK_ANCHORS[sample["task"]]),
        "target_height": args.height,
        "target_width": args.width,
        "target_num_frames": args.num_frames,
        "dtype": args.dtype,
        "text_encoder_layer": TEXT_ENCODER_LAYER,
        "keyframe_encode_seed": KEYFRAME_ENCODE_SEED,
        "model": model,
        "media": media,
    }


def descriptor_fingerprint(descriptor: dict) -> str:
    encoded = json.dumps(
        descriptor,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _anchor_codes(value) -> list[int]:
    if not torch.is_tensor(value):
        raise TypeError("keyframe_anchors must be a tensor in a MiniMax-H3 cache.")
    if value.dtype != torch.long:
        raise TypeError(f"keyframe_anchors must use int64, got {value.dtype}.")
    return [int(item) for item in value.flatten().tolist()]


def validate_cache_payload(
    payload: dict,
    descriptor: dict,
    path: Path,
    *,
    require_keyframe_latents: bool = False,
) -> dict:
    expected_fingerprint = descriptor_fingerprint(descriptor)
    actual_fingerprint = payload.get("cache_fingerprint") if isinstance(payload, dict) else None
    actual_schema = payload.get("cache_schema_version") if isinstance(payload, dict) else None
    if actual_schema != CACHE_SCHEMA_VERSION or actual_fingerprint != expected_fingerprint:
        raise RuntimeError(
            f"Stale or incompatible MiniMax-H3 condition cache: {path}. Its prompt/task/media/geometry/model fingerprint does not match this input. Use a new --output-dir or pass --overwrite."
        )
    positive = payload.get("conditioning", {}).get("positive")
    if not isinstance(positive, dict):
        raise KeyError(f"MiniMax-H3 cache has no conditioning.positive mapping: {path}")
    prompt_embeds = positive.get("prompt_embeds")
    tags = positive.get("text_token_tags")
    if not torch.is_tensor(prompt_embeds) or prompt_embeds.ndim != 3 or prompt_embeds.shape[0] != 1:
        raise ValueError(f"Invalid prompt_embeds in {path}: expected [1, tokens, dim].")
    expected_hidden_size = int(descriptor["model"]["text_hidden_size"])
    if prompt_embeds.shape[-1] != expected_hidden_size:
        raise ValueError(f"Invalid prompt embedding width in {path}: got {prompt_embeds.shape[-1]}, expected {expected_hidden_size}.")
    expected_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[descriptor["dtype"]]
    if prompt_embeds.dtype != expected_dtype:
        raise ValueError(f"Cached prompt dtype is {prompt_embeds.dtype}, expected {expected_dtype}: {path}")
    if not torch.is_tensor(tags) or tags.ndim != 1 or tags.shape[0] != prompt_embeds.shape[1]:
        raise ValueError(f"Invalid text_token_tags in {path}: expected one tag per prompt row.")
    if tags.dtype != torch.long or not bool(torch.isin(tags, torch.tensor([VIDEO_TAG, TEXT_TAG])).all()):
        raise ValueError(f"Invalid text_token_tags in {path}: expected int64 values from {{0, 1}}.")
    expected_codes = [ANCHOR_CODES[value] for value in descriptor["keyframe_anchors"]]
    if _anchor_codes(positive.get("keyframe_anchors")) != expected_codes:
        raise ValueError(f"Cached keyframe anchors do not match task={descriptor['task']}: {path}")
    for name in ("task", "target_height", "target_width", "target_num_frames"):
        expected = descriptor[name]
        actual = positive.get(name)
        if actual != expected:
            raise ValueError(f"Cached {name}={actual!r}, expected {expected!r}: {path}")

    condition_rows = positive.get("condition_video_latents")
    num_keyframes = len(expected_codes)
    if num_keyframes:
        if not bool((tags == VIDEO_TAG).any()):
            raise ValueError(f"Keyframe cache has no Qwen vision-token tags: {path}")
        if condition_rows is None:
            if require_keyframe_latents:
                raise KeyError(f"MiniMax-H3 cache has no condition_video_latents: {path}")
        else:
            rows_per_keyframe = (descriptor["target_height"] // VAE_SPATIAL_SCALE_FACTOR // PATCH_SIZE[1]) * (descriptor["target_width"] // VAE_SPATIAL_SCALE_FACTOR // PATCH_SIZE[2])
            expected_shape = (num_keyframes * rows_per_keyframe, VIDEO_LATENT_CHANNELS * int(np.prod(PATCH_SIZE)))
            if not torch.is_tensor(condition_rows) or tuple(condition_rows.shape) != expected_shape:
                raise ValueError(f"Invalid condition_video_latents in {path}: got {getattr(condition_rows, 'shape', None)}, expected {expected_shape}.")
            if condition_rows.dtype != torch.float32:
                raise ValueError(f"Cached condition_video_latents must be float32, got {condition_rows.dtype}: {path}")
    elif bool((tags == VIDEO_TAG).any()):
        raise ValueError(f"T2AV cache must not contain Qwen vision-token tags: {path}")
    elif condition_rows is not None and (not torch.is_tensor(condition_rows) or condition_rows.numel()):
        raise ValueError(f"T2AV cache must not contain keyframe latents: {path}")
    return payload


def load_validated_cache(path: Path, descriptor: dict, *, require_keyframe_latents: bool = False) -> dict:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    return validate_cache_payload(
        payload,
        descriptor,
        path,
        require_keyframe_latents=require_keyframe_latents,
    )


def merge_manifest(path: Path, current_rows: list[dict], replace: bool) -> list[dict]:
    merged = {}
    if path.is_file() and not replace:
        for row in _read_json_rows(path):
            condition_path = row.get("condition_path")
            if not condition_path:
                raise ValueError(f"Existing manifest row has no condition_path: {row}")
            cache_path = Path(condition_path)
            if not cache_path.is_absolute():
                cache_path = path.parent / cache_path
            cache_path = cache_path.resolve()
            try:
                cache_path.relative_to(path.parent.resolve())
            except ValueError as error:
                raise ValueError(f"Existing condition_path escapes output directory: {condition_path}") from error
            if not cache_path.is_file():
                raise FileNotFoundError(f"Existing manifest references a missing cache: {cache_path}")
            merged[str(condition_path)] = row
    for row in current_rows:
        merged[str(row["condition_path"])] = row
    return sorted(merged.values(), key=lambda row: (int(row.get("id", 0)), str(row["condition_path"])))


def load_conditioner(model_path: str, device: str, dtype: torch.dtype):
    try:
        from transformers import Qwen2TokenizerFast, Qwen3VLForConditionalGeneration, Qwen3VLProcessor
    except ImportError as exc:
        raise ImportError("MiniMax-H3 prompt encoding requires a Transformers build with Qwen3-VL support. Use the model's local_diffusers environment.") from exc

    root = Path(model_path).expanduser().resolve()
    paths = {name: root / name for name in ("text_encoder", "tokenizer", "processor")}
    for path in paths.values():
        if not path.exists():
            raise FileNotFoundError(f"Missing MiniMax-H3 component: {path}")
    load_kwargs = {"dtype": dtype, "local_files_only": True}
    if str(device) != "cpu":
        load_kwargs["device_map"] = {"": str(device)}
    text_encoder = Qwen3VLForConditionalGeneration.from_pretrained(str(paths["text_encoder"]), **load_kwargs)
    text_encoder.requires_grad_(False).eval()
    tokenizer = Qwen2TokenizerFast.from_pretrained(str(paths["tokenizer"]), local_files_only=True)
    processor = Qwen3VLProcessor.from_pretrained(str(paths["processor"]), local_files_only=True)
    return text_encoder, tokenizer, processor


def prepare_keyframe_image(image: Image.Image, height: int, width: int, stretch: bool) -> Image.Image:
    image = ImageOps.exif_transpose(image).convert("RGB")
    if image.size == (width, height):
        return image
    if stretch:
        return image.resize((width, height), Image.Resampling.LANCZOS)
    scale = max(width / image.size[0], height / image.size[1])
    resized_size = (max(width, round(image.size[0] * scale)), max(height, round(image.size[1] * scale)))
    left = max(0, (resized_size[0] - width) // 2)
    top = max(0, (resized_size[1] - height) // 2)
    return image.resize(resized_size, Image.Resampling.LANCZOS).crop((left, top, left + width, top + height))


def load_prepared_images(sample: dict, height: int, width: int) -> list[Image.Image]:
    paths = []
    if "first_frame" in sample:
        paths.append(sample["first_frame"])
    if "last_frame" in sample:
        paths.append(sample["last_frame"])
    images = []
    for index, path in enumerate(paths):
        with Image.open(path) as image:
            images.append(prepare_keyframe_image(image, height, width, stretch=index == 0))
    return images


@torch.inference_mode()
def encode_prompt(
    text_encoder,
    tokenizer,
    processor,
    prompt: str,
    images: list[Image.Image],
    device: str,
    output_dtype: torch.dtype,
) -> dict[str, torch.Tensor]:
    pixel_values = image_grid_thw = None
    token_ids, token_tags = [], []
    if images:
        vision = processor.image_processor(images=images, return_tensors="pt")
        pixel_values, image_grid_thw = vision["pixel_values"], vision["image_grid_thw"]
        merge_unit = processor.image_processor.merge_size**2
        for index in range(len(images)):
            count = int(image_grid_thw[index].prod()) // merge_unit
            label = tokenizer(f"<Picture {index + 1}>: ", add_special_tokens=False)["input_ids"]
            block = [tokenizer.convert_tokens_to_ids("<|vision_start|>")]
            block += [tokenizer.convert_tokens_to_ids("<|image_pad|>")] * count
            block += [tokenizer.convert_tokens_to_ids("<|vision_end|>")]
            token_ids += label + block
            token_tags += [TEXT_TAG] * len(label) + [VIDEO_TAG] * len(block)
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    token_ids += prompt_ids
    token_tags += [TEXT_TAG] * len(prompt_ids)
    if not token_ids:
        raise ValueError("The tokenizer produced no tokens for a non-empty prompt.")

    input_ids = torch.tensor([token_ids], dtype=torch.long, device=device)
    mm_token_type_ids = torch.tensor(processor.create_mm_token_type_ids([token_ids]), dtype=torch.long, device=device)
    parameter_dtype = next(text_encoder.parameters()).dtype
    outputs = text_encoder.model(
        input_ids=input_ids,
        attention_mask=torch.ones_like(input_ids),
        mm_token_type_ids=mm_token_type_ids,
        pixel_values=None if pixel_values is None else pixel_values.to(device=device, dtype=parameter_dtype),
        image_grid_thw=None if image_grid_thw is None else image_grid_thw.to(device),
        use_cache=False,
        output_hidden_states=True,
    )
    if len(outputs.hidden_states) <= TEXT_ENCODER_LAYER:
        raise RuntimeError(f"Qwen3-VL returned {len(outputs.hidden_states)} hidden states; H3 requires index {TEXT_ENCODER_LAYER}.")
    prompt_embeds = outputs.hidden_states[TEXT_ENCODER_LAYER].to(dtype=output_dtype).cpu()
    return {
        "prompt_embeds": prompt_embeds,
        "text_token_tags": torch.tensor(token_tags, dtype=torch.long),
    }


def release_device_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def load_video_vae(model_path: str, device: str, dtype: torch.dtype):
    try:
        from diffusers import AutoencoderKLMiniMaxH3
    except ImportError as error:
        raise ImportError("Keyframe cache generation requires Diffusers with AutoencoderKLMiniMaxH3.") from error
    path = Path(model_path).expanduser().resolve() / "vae"
    if not path.is_dir():
        raise FileNotFoundError(f"Missing MiniMax-H3 video VAE: {path}")
    vae = AutoencoderKLMiniMaxH3.from_pretrained(str(path), torch_dtype=dtype, local_files_only=True)
    vae.requires_grad_(False).eval().to(device)
    return vae


def patchify_video_latents(latents: torch.Tensor, patch_size: tuple[int, int, int] = (1, 2, 2)) -> torch.Tensor:
    patch_t, patch_h, patch_w = patch_size
    batch, channels, frames, height, width = latents.shape
    if batch != 1 or frames % patch_t or height % patch_h or width % patch_w:
        raise ValueError(f"Cannot patchify keyframe latent shape {tuple(latents.shape)} with patch {patch_size}.")
    latents = latents.reshape(
        batch,
        channels,
        frames // patch_t,
        patch_t,
        height // patch_h,
        patch_h,
        width // patch_w,
        patch_w,
    )
    return latents.permute(0, 2, 4, 6, 1, 3, 5, 7).reshape(-1, channels * patch_t * patch_h * patch_w).contiguous()


@torch.inference_mode()
def encode_keyframes(vae, images: list[Image.Image], device: str) -> torch.Tensor:
    from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution

    latent_mean = torch.tensor(vae.config.latents_mean, dtype=torch.float32).view(1, -1, 1, 1, 1)
    latent_std = torch.tensor(vae.config.latents_std, dtype=torch.float32).view(1, -1, 1, 1, 1)
    pixel_mean = torch.tensor(PIXEL_MEAN, dtype=torch.float32, device=device).view(1, -1, 1, 1, 1)
    pixel_std = torch.tensor(PIXEL_STD, dtype=torch.float32, device=device).view(1, -1, 1, 1, 1)
    rows = []
    for image in images:
        pixels = torch.from_numpy(np.array(image, copy=True)).to(device).permute(2, 0, 1)[None, :, None]
        # Keep the normalized input in float32. AutoencoderKLMiniMaxH3 performs
        # the same internal dtype handling as the released Diffusers pipeline.
        pixels = (pixels.float().div(255.0) - pixel_mean) / pixel_std
        moments = vae._encode_clip(pixels)
        posterior = DiagonalGaussianDistribution(moments)
        latents = posterior.sample(generator=torch.Generator().manual_seed(KEYFRAME_ENCODE_SEED))
        latents = latents.to(torch.float16).float().cpu()
        rows.append(patchify_video_latents((latents - latent_mean) / latent_std))
    return torch.cat(rows).to(torch.float32).contiguous()


def condition_path_for_sample(condition_dir: Path, output_index: int) -> tuple[Path, Path]:
    relative = Path("conditions") / f"condition_{output_index:08d}.pt"
    return relative, condition_dir.parent / relative


def main() -> None:
    args = parse_args()
    args.input_path = args.input_path.expanduser().resolve()
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    task_override = normalize_task(args.task) if args.task else None
    samples = read_samples(
        args.input_path,
        args.prompt_column,
        task_override,
        args.start_index,
        args.max_samples,
    )
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    model_descriptor = model_identity(args.model_path)
    namespace_descriptor = preprocessing_descriptor(args, task_override, model_descriptor)
    initialize_cache_namespace(
        output_dir,
        namespace_descriptor,
        allow_replace=args.overwrite and args.replace_manifest,
    )
    condition_dir = output_dir / "conditions"
    condition_dir.mkdir(parents=True, exist_ok=True)
    entries = []
    text_jobs = []
    manifest_rows = []
    for sample in samples:
        source_record = sample.get("record") or {}
        declared_num_frames = source_record.get("target_num_frames")
        if declared_num_frames not in (None, "") and int(declared_num_frames) != int(args.num_frames):
            raise ValueError(f"source_index={sample['source_index']} declares target_num_frames={declared_num_frames}, but preprocessing was requested with num_frames={args.num_frames}.")
        if sample["task"] != "t2av":
            declared_height = source_record.get("target_height")
            declared_width = source_record.get("target_width")
            if declared_height not in (None, "") and declared_width not in (None, ""):
                declared_geometry = (int(declared_height), int(declared_width))
                requested_geometry = (int(args.height), int(args.width))
                if declared_geometry != requested_geometry:
                    raise ValueError(
                        f"source_index={sample['source_index']} declares target geometry "
                        f"{declared_geometry[0]}x{declared_geometry[1]}, but preprocessing was "
                        f"requested at {requested_geometry[0]}x{requested_geometry[1]}."
                    )
        # The source row index is stable across --start-index chunks. It keeps
        # filenames and manifest IDs deterministic when preprocessing resumes.
        output_index = sample["source_index"]
        relative_path, output_path = condition_path_for_sample(condition_dir, output_index)
        descriptor = cache_descriptor(sample, args, model_descriptor)
        entry = {
            "sample": sample,
            "output_index": output_index,
            "relative_path": relative_path,
            "output_path": output_path,
            "descriptor": descriptor,
        }
        entries.append(entry)
        if args.overwrite or not output_path.is_file():
            text_jobs.append(entry)
        else:
            load_validated_cache(output_path, descriptor)

        row = {
            "id": output_index,
            "source_index": sample["source_index"],
            "source_id": sample["source_id"],
            "task": sample["task"],
            "prompt_variant": sample.get("prompt_variant"),
            "caption": sample["prompt"],
            "condition_path": str(relative_path),
            "cache_fingerprint": descriptor_fingerprint(descriptor),
            "target_height": args.height,
            "target_width": args.width,
            "num_frames": args.num_frames,
        }
        # Keep the source text/provenance needed by the teacher-video stage.
        # ``caption`` remains the only text encoded into the condition cache;
        # these fields are metadata and never alter the cached tensors.
        for key in (
            "prompt",
            "enhanced_prompt",
            "source_record_id",
            "selection_occurrence",
            "target_prompt_variant",
            "ir_source_path",
            "ir_source_row_number",
            "caption_source_path",
            "caption_source_row_number",
            "original_source_path",
            "original_source_row_number",
            "enhanced_source_path",
            "enhanced_source_row_number",
            "raw_prompt_sha256",
            "source_video",
            "source_prompt_path",
            "target_local_id",
            "parent_source_index",
            "donor_task",
            "donor_dataset_id",
            "parent_task",
            "donor_source_id",
            "parent_source_id",
            "donor_source_record_id",
            "parent_source_record_id",
            "parent_source_key",
            "donor_source_line",
            "parent_source_line",
            "donor_metadata_id",
            "parent_metadata_id",
            "source_first_frame",
            "source_last_frame",
            "source_first_frame_sha256",
            "source_last_frame_sha256",
            "first_frame_input_path",
            "last_frame_input_path",
            "first_frame_input_sha256",
            "last_frame_input_sha256",
            "first_frame_strategy",
            "last_frame_strategy",
            "original_first_frame_strategy",
            "first_frame_sha256",
            "last_frame_sha256",
            "output_first_frame_sha256",
            "output_last_frame_sha256",
            "teacher_video_path",
            "teacher_request_fingerprint",
            "teacher_generation_task_id",
            "teacher_frame_extract_path",
            "teacher_first_frame_input_path",
            "teacher_first_frame_sha256",
            "teacher_first_fallback_reason",
            "metadata_id",
            "source_line",
            "source_duration",
            "source_resolution",
            "source_image_width",
            "source_image_height",
            "source_orientation",
            "duration",
            "generation_duration",
            "target_resolution",
            "aspect_bucket",
            "target_orientation",
            "fps",
            "effective_target_duration",
            "image_edit_fingerprint",
            "image_edit_fingerprints",
            "first_frame_image_edit_fingerprint",
            "last_frame_image_edit_fingerprint",
            "first_image_edit_fingerprint",
            "last_image_edit_fingerprint",
            "image_edit_model",
            "image_edit_prompt_sha256",
            "image_edit_request_fingerprint",
            "image_edit_response_id",
            "portrait_expansion_version",
            "portrait_cache_input_fingerprint",
            "context_ir_task_id",
            "context_ir_requested_resolution",
            "context_ir_resolved_resolution",
        ):
            if source_record.get(key) is not None:
                row[key] = source_record[key]
        # ``source_index`` above is the stable row index inside this cache
        # namespace. Preserve an upstream dataset index under a distinct name
        # so it cannot change condition filenames or async API sample keys.
        if source_record.get("source_index") is not None:
            row["source_dataset_index"] = source_record["source_index"]
        for key in ("first_frame", "last_frame"):
            if key in sample:
                row[key] = sample[key]
        manifest_rows.append(row)

    if text_jobs:
        print(
            f"Loading MiniMax-H3 Qwen3-VL conditioner on {args.device}; encode={len(text_jobs)} reuse={len(entries) - len(text_jobs)} ...",
            flush=True,
        )
        text_encoder, tokenizer, processor = load_conditioner(args.model_path, args.device, dtype)
        for job_index, entry in enumerate(text_jobs, start=1):
            sample = entry["sample"]
            images = load_prepared_images(sample, args.height, args.width)
            condition = encode_prompt(
                text_encoder,
                tokenizer,
                processor,
                sample["prompt"],
                images,
                args.device,
                dtype,
            )
            condition.update(
                {
                    "task": sample["task"],
                    "keyframe_anchors": torch.tensor([ANCHOR_CODES[value] for value in TASK_ANCHORS[sample["task"]]], dtype=torch.long),
                    "target_height": args.height,
                    "target_width": args.width,
                    "target_num_frames": args.num_frames,
                }
            )
            atomic_torch_save(
                {
                    "cache_schema_version": CACHE_SCHEMA_VERSION,
                    "cache_fingerprint": descriptor_fingerprint(entry["descriptor"]),
                    "cache_metadata": entry["descriptor"],
                    "conditioning": {"positive": condition},
                    "prompt": sample["prompt"],
                    "source_index": sample["source_index"],
                    "source_id": sample["source_id"],
                    "task": sample["task"],
                },
                entry["output_path"],
            )
            print(f"[text {job_index}/{len(text_jobs)}] {entry['output_path']}", flush=True)
        del text_encoder, tokenizer, processor
        release_device_memory()
    else:
        print(f"Reusing {len(entries)} validated Qwen condition caches.", flush=True)

    vae_jobs = []
    for entry in entries:
        if not TASK_ANCHORS[entry["sample"]["task"]]:
            continue
        payload = load_validated_cache(entry["output_path"], entry["descriptor"])
        if "condition_video_latents" not in payload["conditioning"]["positive"]:
            vae_jobs.append(entry)
    if vae_jobs:
        vae_device = args.vae_device or args.device
        print(
            f"Loading MiniMax-H3 keyframe VAE on {vae_device}; encode={len(vae_jobs)} reuse={sum(bool(TASK_ANCHORS[e['sample']['task']]) for e in entries) - len(vae_jobs)} ...",
            flush=True,
        )
        vae = load_video_vae(args.model_path, vae_device, dtype)
        for job_index, entry in enumerate(vae_jobs, start=1):
            sample = entry["sample"]
            payload = load_validated_cache(entry["output_path"], entry["descriptor"])
            positive = payload["conditioning"]["positive"]
            images = load_prepared_images(sample, args.height, args.width)
            positive["condition_video_latents"] = encode_keyframes(vae, images, vae_device)
            validate_cache_payload(
                payload,
                entry["descriptor"],
                entry["output_path"],
                require_keyframe_latents=True,
            )
            atomic_torch_save(payload, entry["output_path"])
            print(f"[vae {job_index}/{len(vae_jobs)}] {entry['output_path']}", flush=True)
        del vae
        release_device_memory()

    manifest = output_dir / "metadata.jsonl"
    replace_manifest = args.replace_manifest or (args.start_index == 0 and args.max_samples is None)
    merged_rows = merge_manifest(manifest, manifest_rows, replace=replace_manifest)
    atomic_write_jsonl(manifest, merged_rows)
    task_counts = {task: sum(row["task"] == task for row in merged_rows) for task in TASK_ANCHORS}
    print(
        f"Wrote manifest rows={len(merged_rows)} current_chunk={len(manifest_rows)} to {manifest}; task_counts={task_counts}",
        flush=True,
    )


if __name__ == "__main__":
    main()
