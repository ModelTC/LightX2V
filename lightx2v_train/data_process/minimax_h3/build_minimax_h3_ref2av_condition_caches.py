#!/usr/bin/env python3
"""Build condition-only MiniMax-H3 Ref2AV DMD caches.

The default input is completed Context-IR JSONL with ``enhanced_prompt``,
target geometry, and ordered local-media ``references``.  Optional prompt
fallback and fixed-768p target policies also accept raw image-only Ref2V rows.
The output intentionally contains no target video/audio
latents. It is consumed by the ``minimax_h3_ref_cache_dataset`` and
``minimax_h3_ref2av`` model with adaptive video regularization disabled.

Run with a Diffusers build containing the MiniMax-H3 modular pipeline.
Model imports are lazy, so ``--dry-run``
and the schema self-tests do not require loading model weights.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib
import json
import math
import os
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Iterable

import torch

CACHE_SCHEMA_VERSION = 1
TEXT_ENCODER_LAYER = 50
KEYFRAME_ENCODE_SEED = 42
VIDEO_LATENT_CHANNELS = 24
AUDIO_LATENT_CHANNELS = 32
TEXT_HIDDEN_SIZE = 5120
PATCH_SIZE = (1, 2, 2)
VIDEO_ROW_WIDTH = VIDEO_LATENT_CHANNELS * math.prod(PATCH_SIZE)
TASK_ALIASES = {"ref2av": "ref2av", "ref2va": "ref2av", "ref2v": "ref2av"}
REFERENCE_LIMITS = {"image": 9, "video": 3, "audio": 3}
REFERENCE_RESIZE_MODES = ("match", "max", "diffusers")
REFERENCE_IMAGE_SHORT_EDGE = 2048
MANIFEST_REQUIRED_KEYS = {
    "condition_path",
    "target_height",
    "target_width",
    "num_frames",
}
MANIFEST_COST_KEYS = {
    "task",
    "target_orientation",
    "reference_image_count",
    "reference_video_count",
    "reference_audio_count",
    "prompt_token_count",
    "reference_video_rows",
    "reference_audio_rows",
    "reference_compute_cost",
    "packed_sequence_tokens_124",
    "ref_image_count",
    "source_index",
    "source_id",
    "prompt_source",
    "cache_fingerprint",
}
# Runtime image-only DMD fixes the generated target to 124 frames.  At 768p
# this contributes 37,296 video rows and 414 audio rows to the packed sequence.
FIXED_DMD_NUM_FRAMES = 124
FIXED_DMD_TARGET_ROWS = 37 * 24 * 42 + 207 * 2
REFERENCE_KIND_ALIASES = {
    "image": "image",
    "reference_image": "image",
    "image_url": "image",
    "video": "video",
    "reference_video": "video",
    "video_url": "video",
    "audio": "audio",
    "reference_audio": "audio",
    "audio_url": "audio",
}


@dataclass(frozen=True)
class ReferenceSpec:
    kind: str
    path: Path
    sha256: str
    size_bytes: int
    include_embedded_audio: bool | None = None

    def fingerprint_record(self) -> dict:
        record = {
            "kind": self.kind,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
        }
        if self.kind == "video" and self.include_embedded_audio is not None:
            record["include_embedded_audio"] = self.include_embedded_audio
        return record


@dataclass(frozen=True)
class SampleSpec:
    source_index: int
    source_id: str
    prompt: str
    target_height: int
    target_width: int
    target_num_frames: int
    references: tuple[ReferenceSpec, ...]
    descriptor: dict
    fingerprint: str


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "metadata",
        nargs="?",
        type=Path,
        help="Ref2AV/Ref2V JSONL with local references; raw prompts require --prompt-policy enhanced-or-original.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--model-path",
        type=Path,
        help="Converted official Diffusers root (contains vae/ and audio_vae/).",
    )
    parser.add_argument(
        "--source-model-path",
        type=Path,
        help="Original Ref2VA root (text_encoder/tokenizer/processor and provenance).",
    )
    parser.add_argument(
        "--media-root",
        type=Path,
        help="Base for relative reference paths; defaults to the metadata directory.",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument(
        "--reference-latent-dtype",
        choices=("bf16", "fp16", "fp32"),
        default="fp32",
        help="Reference cache storage precision; legacy caches use fp32. Text uses --dtype.",
    )
    parser.add_argument(
        "--prompt-policy",
        choices=("enhanced-only", "enhanced-or-original"),
        default="enhanced-only",
        help="Optional fallback: enhanced_prompt, prompt_en/prompt_en_original, prompt_cn/prompt_cn_original.",
    )
    parser.add_argument(
        "--target-policy",
        choices=("source", "fixed-768p"),
        default="source",
        help="fixed-768p uses source orientation, 768x1344 or 1344x768, and 124 frames unless overridden; no target video is read.",
    )
    parser.add_argument("--image-only", action="store_true", help="Reject rows containing video/audio references.")
    parser.add_argument("--skip-invalid", action="store_true", help="Record invalid/failed rows in the shard failure JSONL and continue; namespace/resume conflicts remain fatal.")
    parser.add_argument(
        "--reference-resize-mode",
        choices=REFERENCE_RESIZE_MODES,
        default="diffusers",
        help=(
            "Reference-image policy. match preserves aspect ratio and caps the image at the "
            "target-canvas area without upscaling; max only downsizes a short edge above 2048; "
            "diffusers always scales the short edge to 2048."
        ),
    )
    parser.add_argument(
        "--target-num-frames",
        type=int,
        help=("Override cached target frame count. Under --target-policy source, validate the original timing; fixed-768p ignores it and defaults to 124 frames."),
    )
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument(
        "--stage",
        choices=("all", "text", "references"),
        default="all",
        help="Two-stage resume boundary; all encodes text, unloads Qwen, then loads the VAEs.",
    )
    parser.add_argument("--manifest-name", help="Override this invocation's JSONL manifest filename.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--replace-manifest", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate rows, media hashes, order, geometry and model identity without loading models or writing.",
    )
    parser.add_argument(
        "--merge-shards",
        action="store_true",
        help="Merge completed metadata.shard-XXX-of-YYY.jsonl files into metadata.jsonl; load no models.",
    )
    args = parser.parse_args(argv)

    if args.start_index < 0:
        parser.error("--start-index must be non-negative")
    if args.max_samples is not None and args.max_samples <= 0:
        parser.error("--max-samples must be positive")
    if args.num_shards <= 0 or not 0 <= args.shard_index < args.num_shards:
        parser.error("Require --num-shards > 0 and 0 <= --shard-index < --num-shards")
    if args.target_num_frames is not None and (args.target_num_frames % 17 != 5 or not 107 <= args.target_num_frames <= 362):
        parser.error("--target-num-frames must be 17*n+5 in [107, 362]")
    if not args.merge_shards and args.metadata is None:
        parser.error("metadata is required unless --merge-shards is used")
    if not args.merge_shards and (args.model_path is None or args.source_model_path is None):
        parser.error("--model-path and --source-model-path are required for encoding or --dry-run")
    return args


def canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")


def digest_object(value: object) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


def sha256_file(path: Path, chunk_size: int = 4 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        while True:
            chunk = input_file.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        with temporary.open("w", encoding="utf-8") as output_file:
            json.dump(value, output_file, ensure_ascii=False, indent=2, sort_keys=True)
            output_file.write("\n")
            output_file.flush()
            os.fsync(output_file.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_create_json(path: Path, value: object) -> None:
    """Publish a complete JSON file only when no concurrent owner exists."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.create.{os.getpid()}")
    try:
        with temporary.open("x", encoding="utf-8") as output_file:
            json.dump(value, output_file, ensure_ascii=False, indent=2, sort_keys=True)
            output_file.write("\n")
            output_file.flush()
            os.fsync(output_file.fileno())
        try:
            # link(2) is an atomic no-clobber publication.  Unlike replace(),
            # two differently configured shards can never overwrite ownership.
            os.link(temporary, path)
        except FileExistsError:
            pass
    finally:
        temporary.unlink(missing_ok=True)


def atomic_write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        with temporary.open("w", encoding="utf-8") as output_file:
            for row in rows:
                output_file.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            output_file.flush()
            os.fsync(output_file.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_torch_save(value: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        torch.save(value, temporary)
        with temporary.open("rb") as saved_file:
            os.fsync(saved_file.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON at {path}:{line_number}: {error}") from error
            if not isinstance(row, dict):
                raise TypeError(f"{path}:{line_number} must contain a JSON object")
            rows.append(row)
    return rows


def _small_file_identity(path: Path) -> dict:
    return {
        "relative_path": path.name,
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _weight_identity(path: Path, root: Path) -> dict:
    stat = path.stat()
    return {
        "relative_path": str(path.resolve().relative_to(root.resolve())),
        "size_bytes": stat.st_size,
        # Hashing ~80 GiB per worker is wasteful.  Config/index content plus
        # each shard's size+mtime detects ordinary local checkpoint changes.
        "mtime_ns": stat.st_mtime_ns,
    }


def _indexed_weights(index_path: Path) -> list[Path]:
    if not index_path.is_file():
        return []
    payload = json.loads(index_path.read_text(encoding="utf-8"))
    weight_map = payload.get("weight_map")
    if not isinstance(weight_map, dict):
        raise ValueError(f"Weight index has no weight_map: {index_path}")
    paths = []
    for filename in sorted(set(weight_map.values())):
        path = index_path.parent / filename
        if not path.is_file():
            raise FileNotFoundError(f"{index_path} references missing weight shard {path}")
        paths.append(path)
    return paths


def _component_identity(root: Path, relative: str) -> dict:
    component = root / relative
    if not component.is_dir():
        raise FileNotFoundError(f"Missing model component: {component}")
    small = []
    weights = []
    for path in sorted(component.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix in {".json", ".yaml", ".txt"}:
            record = _small_file_identity(path)
            record["relative_path"] = str(path.resolve().relative_to(root.resolve()))
            small.append(record)
        elif path.suffix == ".safetensors":
            weights.append(_weight_identity(path, root))
    for index_path in component.glob("*.safetensors.index.json"):
        for path in _indexed_weights(index_path):
            identity = _weight_identity(path, root)
            if identity not in weights:
                weights.append(identity)
    return {
        "relative_path": relative,
        "config_files": small,
        "weight_files": sorted(weights, key=lambda item: item["relative_path"]),
    }


def _read_config(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Missing model config: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Model config must be a JSON object: {path}")
    return value


def _equivalent_config_value(left: object, right: object) -> bool:
    if isinstance(left, (int, float)) and not isinstance(left, bool):
        if not isinstance(right, (int, float)) or isinstance(right, bool):
            return False
        return math.isclose(float(left), float(right), rel_tol=1e-12, abs_tol=1e-12)
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(_equivalent_config_value(a, b) for a, b in zip(left, right))
    return left == right


def _assert_conversion_matches_source(source_root: Path, model_root: Path) -> None:
    source_video = _read_config(source_root / "video_vae" / "config.json")
    converted_video = _read_config(model_root / "vae" / "config.json")
    source_audio = _read_config(source_root / "audio_vae" / "config.json")
    converted_audio = _read_config(model_root / "audio_vae" / "config.json")
    comparisons = (
        ("video latent_channels", source_video.get("latent_channels"), converted_video.get("latent_channels")),
        ("video latents_mean", source_video.get("latents_mean"), converted_video.get("latents_mean")),
        ("video latents_std", source_video.get("latents_std"), converted_video.get("latents_std")),
        ("audio latent_channels", source_audio.get("latent_channels"), converted_audio.get("latent_channels")),
        ("audio latents_mean", source_audio.get("latents_mean"), converted_audio.get("latents_mean")),
        ("audio latents_std", source_audio.get("latents_std"), converted_audio.get("latents_std")),
        ("audio sampling_rate", source_audio.get("sample_rate"), converted_audio.get("sampling_rate")),
    )
    for name, source_value, converted_value in comparisons:
        if not _equivalent_config_value(source_value, converted_value):
            raise RuntimeError(f"Converted Diffusers component does not match Ref2VA source ({name}): source={source_value!r}, converted={converted_value!r}")


def model_identity(source_root: Path, model_root: Path) -> dict:
    source_root = source_root.expanduser().resolve()
    model_root = model_root.expanduser().resolve()
    _assert_conversion_matches_source(source_root, model_root)
    text_config = _read_config(source_root / "text_encoder" / "config.json")
    hidden_size = int(text_config.get("text_config", text_config).get("hidden_size", 0))
    video_config = _read_config(model_root / "vae" / "config.json")
    audio_config = _read_config(model_root / "audio_vae" / "config.json")
    if hidden_size != TEXT_HIDDEN_SIZE:
        raise ValueError(f"Ref2VA text hidden size is {hidden_size}, expected {TEXT_HIDDEN_SIZE}")
    if int(video_config.get("latent_channels", 0)) != VIDEO_LATENT_CHANNELS:
        raise ValueError("Converted Ref2VA video VAE must have 24 latent channels")
    if int(audio_config.get("latent_channels", 0)) != AUDIO_LATENT_CHANNELS:
        raise ValueError("Converted Ref2VA audio VAE must have 32 latent channels")
    return {
        "source_root": str(source_root),
        "converted_root": str(model_root),
        "text_hidden_size": hidden_size,
        "video_latent_channels": VIDEO_LATENT_CHANNELS,
        "audio_latent_channels": AUDIO_LATENT_CHANNELS,
        "source_components": [_component_identity(source_root, name) for name in ("text_encoder", "tokenizer", "processor", "video_vae", "audio_vae")],
        "converted_components": [_component_identity(model_root, name) for name in ("vae", "audio_vae")],
    }


def _normalize_kind(value: object, source_index: int, reference_index: int) -> str:
    normalized = str(value or "").strip().lower()
    try:
        return REFERENCE_KIND_ALIASES[normalized]
    except KeyError as error:
        raise ValueError(f"row {source_index} references[{reference_index}] has unsupported kind/type/role {value!r}") from error


def _resolve_media_path(value: object, media_root: Path, label: str) -> Path:
    if not isinstance(value, (str, os.PathLike)) or not str(value).strip():
        raise ValueError(f"{label} requires a non-empty local path")
    text = str(value).strip()
    if text.startswith(("http://", "https://", "data:", "mm_file://")):
        raise ValueError(f"{label} must be local for offline latent encoding, got {text[:80]!r}")
    path = Path(text).expanduser()
    if not path.is_absolute():
        path = media_root / path
    path = path.resolve()
    if not path.is_file() or path.stat().st_size <= 0:
        raise FileNotFoundError(f"{label} points to a missing/empty file: {path}")
    return path


def _ordered_reference_entries(row: dict, source_index: int) -> list[dict]:
    for key in ("references", "actual_ordered_references", "ordered_references"):
        value = row.get(key)
        if value is not None:
            if not isinstance(value, list) or not value:
                raise ValueError(f"row {source_index} {key} must be a non-empty list")
            if not all(isinstance(entry, dict) for entry in value):
                raise TypeError(f"row {source_index} {key} entries must be objects")
            explicit_orders = [entry.get("order", entry.get("reference_index")) for entry in value]
            if any(order is not None for order in explicit_orders):
                if any(isinstance(order, bool) or not isinstance(order, int) for order in explicit_orders):
                    raise ValueError(f"row {source_index} {key} must give integer order on every reference")
                expected_one_based = list(range(1, len(value) + 1))
                expected_zero_based = list(range(len(value)))
                if explicit_orders not in (expected_zero_based, expected_one_based):
                    raise ValueError(f"row {source_index} {key} order is semantic and must already be contiguous zero- or one-based; got {explicit_orders}. Refusing to silently reorder.")
            return value

    derived = []
    for kind, key in (
        ("image", "reference_images"),
        ("video", "reference_videos"),
        ("audio", "reference_audios"),
    ):
        values = row.get(key, [])
        if isinstance(values, str):
            values = [values]
        if not isinstance(values, list):
            raise TypeError(f"row {source_index} {key} must be a list")
        derived.extend({"kind": kind, "path": value} for value in values)
    if not derived:
        raise ValueError(f"row {source_index} has no Ref2AV references")
    return derived


def _duration_to_num_frames(duration: int) -> int:
    if isinstance(duration, bool) or not isinstance(duration, int) or not 4 <= duration <= 15:
        raise ValueError(f"duration must be an integer from 4 through 15, got {duration!r}")
    return math.ceil((24 * duration - 5) / 17) * 17 + 5


def _validate_target_num_frames(value: object, source_index: int) -> int:
    if isinstance(value, bool):
        raise ValueError(f"row {source_index} target_num_frames cannot be boolean")
    try:
        value = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"row {source_index} requires integer target_num_frames") from error
    if value % 17 != 5 or not 107 <= value <= 362:
        raise ValueError(f"row {source_index} target_num_frames must be 17*n+5 in [107, 362] (MiniMax API 4-15 second assignment), got {value}")
    return value


def resolve_reference_image_size(
    width: int,
    height: int,
    *,
    target_width: int,
    target_height: int,
    mode: str,
    short_edge: int = REFERENCE_IMAGE_SHORT_EDGE,
) -> tuple[int, int]:
    """Return ``(height, width)`` using the MiniMax-H3 Turbo resize policies."""

    if width <= 0 or height <= 0:
        raise ValueError(f"A reference image must have a positive size, got {width}x{height}.")
    if width > 4 * height or height > 4 * width:
        raise ValueError(f"A reference image must be within 1:4 and 4:1, got {width}x{height}.")
    if target_width <= 0 or target_height <= 0:
        raise ValueError(f"The target canvas must have a positive size, got {target_width}x{target_height}.")
    if mode == "match":
        scale = min(1.0, math.sqrt((target_width * target_height) / (width * height)))
    elif mode == "max":
        scale = min(1.0, short_edge / min(width, height))
    elif mode == "diffusers":
        scale = short_edge / min(width, height)
    else:
        raise ValueError(f"Unsupported reference resize mode {mode!r}")
    multiple = 32
    return (
        max(multiple, round(height * scale / multiple) * multiple),
        max(multiple, round(width * scale / multiple) * multiple),
    )


def select_prompt(row: dict, source_index: int, policy: str) -> tuple[str, str]:
    fields = ("enhanced_prompt",)
    if policy == "enhanced-or-original":
        fields += ("prompt_en", "prompt_en_original", "prompt_cn", "prompt_cn_original")
    elif policy != "enhanced-only":
        raise ValueError(f"Unsupported prompt policy {policy!r}")
    for field in fields:
        value = row.get(field)
        if isinstance(value, str) and value.strip():
            prompt = value.strip()
            if field != "enhanced_prompt":
                # Keep identities intact; normalize only explicit image reference
                # markers. Never rewrite ordinary prose or Subject indices.
                pattern = r"<\s*(?:Figure|Image|Picture|图片|图)\s*(\d+)\s*>|@(?:Figure|Image|Picture|图片|图)\s*(\d+)"
                prompt = re.sub(pattern, lambda match: f"<Picture {int(match[1] or match[2])}>", prompt, flags=re.I)
            return prompt, field
    raise ValueError(f"row {source_index} requires a non-empty {' or '.join(fields)}")


def resolve_target_geometry(row: dict, source_index: int, policy: str) -> tuple[int, int]:
    info = row.get("target_video_info") or {}
    try:
        height = int(row.get("target_height", info.get("height")))
        width = int(row.get("target_width", info.get("width")))
    except (TypeError, ValueError) as error:
        if policy == "fixed-768p" and row.get("target_orientation") in {"landscape", "portrait"}:
            return (1344, 768) if row["target_orientation"] == "portrait" else (768, 1344)
        raise ValueError(f"row {source_index} requires integer target_height and target_width or a known target_orientation") from error
    if min(height, width) <= 0:
        raise ValueError(f"row {source_index} target geometry must be positive, got {height}x{width}")
    if policy == "fixed-768p":
        return (1344, 768) if height > width else (768, 1344)
    if policy != "source":
        raise ValueError(f"Unsupported target policy {policy!r}")
    if height % 32 or width % 32:
        raise ValueError(f"row {source_index} target geometry must be positive multiples of 32, got {height}x{width}")
    return height, width


def normalize_sample(
    row: dict,
    source_index: int,
    media_root: Path,
    dtype_name: str,
    model_descriptor: dict,
    hash_cache: dict[Path, str] | None = None,
    reference_resize_mode: str = "diffusers",
    target_num_frames_override: int | None = None,
    prompt_policy: str = "enhanced-only",
    target_policy: str = "source",
    reference_latent_dtype: str = "fp32",
    image_only: bool = False,
) -> SampleSpec:
    task = str(row.get("task", "")).strip().lower()
    if task not in TASK_ALIASES:
        raise ValueError(f"row {source_index} requires task=ref2av/ref2va/ref2v, got {task!r}")
    prompt, prompt_source = select_prompt(row, source_index, prompt_policy)
    target_height, target_width = resolve_target_geometry(row, source_index, target_policy)
    declared_num_frames = row.get("target_num_frames", row.get("num_frames"))
    if target_policy == "fixed-768p":
        source_target_num_frames = declared_num_frames
        target_num_frames = _validate_target_num_frames(target_num_frames_override if target_num_frames_override is not None else FIXED_DMD_NUM_FRAMES, source_index)
    else:
        if declared_num_frames is None:
            declared_num_frames = _duration_to_num_frames(row.get("duration"))
        source_target_num_frames = _validate_target_num_frames(declared_num_frames, source_index)
        if row.get("duration") is not None:
            expected = _duration_to_num_frames(row["duration"])
            if expected != source_target_num_frames:
                raise ValueError(f"row {source_index} duration={row['duration']} maps to {expected} frames, but target_num_frames={source_target_num_frames}")
        target_num_frames = source_target_num_frames if target_num_frames_override is None else _validate_target_num_frames(target_num_frames_override, source_index)
    if reference_latent_dtype not in {"bf16", "fp16", "fp32"}:
        raise ValueError(f"Unsupported reference latent dtype {reference_latent_dtype!r}")
    if reference_resize_mode not in REFERENCE_RESIZE_MODES:
        raise ValueError(f"row {source_index} has unsupported reference_resize_mode={reference_resize_mode!r}")

    hash_cache = hash_cache if hash_cache is not None else {}
    references = []
    for reference_index, entry in enumerate(_ordered_reference_entries(row, source_index), start=1):
        kind_value = entry.get("kind", entry.get("modality", entry.get("type", entry.get("role"))))
        kind = _normalize_kind(kind_value, source_index, reference_index)
        if image_only and kind != "image":
            raise ValueError(f"row {source_index} is not image-only: found {kind} reference")
        media_value = next(
            (entry[key] for key in ("local_path", "path", "media_path", "url", "rel_path") if entry.get(key) not in (None, "")),
            None,
        )
        path = _resolve_media_path(
            media_value,
            media_root,
            f"row {source_index} references[{reference_index}]",
        )
        actual_hash = hash_cache.get(path)
        if actual_hash is None:
            actual_hash = sha256_file(path)
            hash_cache[path] = actual_hash
        declared_hash = entry.get("sha256")
        if declared_hash is not None and str(declared_hash).lower() != actual_hash:
            raise ValueError(f"row {source_index} references[{reference_index}] SHA-256 mismatch for {path}: declared={declared_hash}, actual={actual_hash}")
        embedded = entry.get("include_embedded_audio")
        if embedded is None and kind == "video" and "has_audio" in entry:
            embedded = entry["has_audio"]
        if embedded is not None and not isinstance(embedded, bool):
            raise TypeError(f"row {source_index} references[{reference_index}] include_embedded_audio must be boolean")
        references.append(
            ReferenceSpec(
                kind=kind,
                path=path,
                sha256=actual_hash,
                size_bytes=path.stat().st_size,
                include_embedded_audio=embedded if kind == "video" else None,
            )
        )

    counts = Counter(reference.kind for reference in references)
    if len(references) > 12:
        raise ValueError(f"row {source_index} has {len(references)} references; MiniMax limit is 12")
    for kind, limit in REFERENCE_LIMITS.items():
        if counts[kind] > limit:
            raise ValueError(f"row {source_index} has {counts[kind]} {kind} refs; limit is {limit}")
    if counts["image"] + counts["video"] == 0:
        raise ValueError(f"row {source_index} cannot use audio references alone")

    source_id = str(row.get("sample_id", row.get("metadata_id", row.get("id", source_index))))
    descriptor = {
        "cache_schema_version": CACHE_SCHEMA_VERSION,
        "source_index": source_index,
        "source_id": source_id,
        "task": "ref2av",
        "enhanced_prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "target_height": target_height,
        "target_width": target_width,
        "target_num_frames": target_num_frames,
        "source_target_num_frames": source_target_num_frames,
        "dtype": dtype_name,
        "text_encoder_layer": TEXT_ENCODER_LAYER,
        "reference_video_sample_fps": 2.0,
        "reference_image_resize_mode": reference_resize_mode,
        "reference_image_short_edge": REFERENCE_IMAGE_SHORT_EDGE,
        "reference_encode_seed": KEYFRAME_ENCODE_SEED,
        "patch_size": list(PATCH_SIZE),
        # List order is intentionally part of the fingerprint.
        "references": [reference.fingerprint_record() for reference in references],
        "model": model_descriptor,
    }
    # Leave fingerprints unchanged for the historical strict/fp32 policy.
    if prompt_policy != "enhanced-only" or target_policy != "source" or reference_latent_dtype != "fp32" or image_only:
        descriptor.update(
            prompt_policy=prompt_policy,
            prompt_source=prompt_source,
            prompt_sha256=hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            target_policy=target_policy,
            reference_latent_dtype=reference_latent_dtype,
            image_only=image_only,
            ref_image_count=counts["image"],
            source_target_geometry={key: row.get(key) for key in ("target_height", "target_width", "target_orientation", "target_video_info", "duration")},
        )
    return SampleSpec(
        source_index=source_index,
        source_id=source_id,
        prompt=prompt,
        target_height=target_height,
        target_width=target_width,
        target_num_frames=target_num_frames,
        references=tuple(references),
        descriptor=descriptor,
        fingerprint=digest_object(descriptor),
    )


def cache_path(output_dir: Path, source_index: int) -> tuple[Path, Path]:
    relative = Path("conditions") / f"condition_{source_index:08d}.pt"
    return relative, output_dir / relative


def manifest_row(sample: SampleSpec, relative_path: Path, payload: dict | None = None) -> dict:
    row = {
        "condition_path": str(relative_path),
        "target_height": sample.target_height,
        "target_width": sample.target_width,
        "num_frames": sample.target_num_frames,
    }
    if payload is None:
        return row

    positive = payload.get("conditioning", {}).get("positive")
    if not isinstance(positive, dict):
        raise ValueError("Complete Ref2AV payload has no conditioning.positive.")
    prompt_embeds = positive.get("prompt_embeds")
    references = positive.get("references")
    if not torch.is_tensor(prompt_embeds) or prompt_embeds.ndim != 3:
        raise ValueError("Complete Ref2AV payload has invalid prompt_embeds.")
    if not isinstance(references, list) or not references:
        raise ValueError("Complete Ref2AV payload has no encoded references.")

    counts = Counter(str(reference.get("kind", "")) for reference in references)
    reference_video_rows = sum(int(reference["video_latents"].shape[0]) for reference in references if torch.is_tensor(reference.get("video_latents")))
    reference_audio_rows = sum(int(reference["audio_latents"].shape[0]) for reference in references if torch.is_tensor(reference.get("audio_latents")))
    prompt_token_count = int(prompt_embeds.shape[1])
    reference_compute_cost = prompt_token_count + reference_video_rows + reference_audio_rows
    row.update(
        {
            "task": "ref2av",
            "target_orientation": ("landscape" if sample.target_width > sample.target_height else "portrait"),
            "reference_image_count": counts["image"],
            "ref_image_count": counts["image"],
            "source_index": sample.source_index,
            "source_id": sample.source_id,
            "prompt_source": sample.descriptor.get("prompt_source", "enhanced_prompt"),
            "cache_fingerprint": sample.fingerprint,
            "reference_video_count": counts["video"],
            "reference_audio_count": counts["audio"],
            "prompt_token_count": prompt_token_count,
            "reference_video_rows": reference_video_rows,
            "reference_audio_rows": reference_audio_rows,
            "reference_compute_cost": reference_compute_cost,
            "packed_sequence_tokens_124": (reference_compute_cost + FIXED_DMD_TARGET_ROWS),
        }
    )
    return row


def _expect_tensor(value: object, name: str, ndim: int, dtype: torch.dtype | None = None) -> torch.Tensor:
    if not torch.is_tensor(value) or value.ndim != ndim:
        raise ValueError(f"{name} must be a {ndim}D tensor, got {getattr(value, 'shape', None)}")
    if dtype is not None and value.dtype != dtype:
        raise ValueError(f"{name} dtype is {value.dtype}, expected {dtype}")
    return value


def validate_cache_payload(
    payload: object,
    sample: SampleSpec,
    path: Path,
    expected_dtype: torch.dtype,
    require_complete: bool,
) -> str:
    if not isinstance(payload, dict):
        raise TypeError(f"Cache must be a dict: {path}")
    if payload.get("cache_schema_version") != CACHE_SCHEMA_VERSION:
        raise RuntimeError(f"Incompatible cache schema: {path}")
    if payload.get("cache_fingerprint") != sample.fingerprint:
        raise RuntimeError(f"Stale Ref2AV cache at {path}; prompt/media SHA/order/geometry/model identity changed. Use a new --output-dir or pass --overwrite intentionally.")
    stage = payload.get("cache_stage")
    if stage not in {"text", "complete"}:
        raise ValueError(f"Cache has invalid cache_stage={stage!r}: {path}")
    if require_complete and stage != "complete":
        raise ValueError(f"Cache is only stage={stage}, not complete: {path}")
    positive = payload.get("conditioning", {}).get("positive")
    if not isinstance(positive, dict):
        raise KeyError(f"Cache has no conditioning.positive: {path}")
    prompt_embeds = _expect_tensor(positive.get("prompt_embeds"), "prompt_embeds", 3, expected_dtype)
    if prompt_embeds.shape[0] != 1 or prompt_embeds.shape[2] != TEXT_HIDDEN_SIZE:
        raise ValueError(f"prompt_embeds must be [1, tokens, {TEXT_HIDDEN_SIZE}]: {path}")
    tags = _expect_tensor(positive.get("text_token_tags"), "text_token_tags", 1, torch.long)
    if tags.shape[0] != prompt_embeds.shape[1] or not bool(torch.isin(tags, torch.tensor([0, 1])).all()):
        raise ValueError(f"Invalid text_token_tags: {path}")
    if not bool((tags == 0).any()):
        raise ValueError(f"Ref2AV Qwen presentation has no reference vision rows: {path}")
    expected_scalars = {
        "task": "ref2av",
        "target_height": sample.target_height,
        "target_width": sample.target_width,
        "target_num_frames": sample.target_num_frames,
    }
    for key, expected in expected_scalars.items():
        if positive.get(key) != expected:
            raise ValueError(f"Cached {key}={positive.get(key)!r}, expected {expected!r}: {path}")
    if stage == "text":
        if positive.get("references") not in (None, []):
            raise ValueError(f"Text-stage cache unexpectedly contains encoded references: {path}")
        return stage

    encoded = positive.get("references")
    reference_dtype = _dtype(sample.descriptor.get("reference_latent_dtype", "fp32"))
    if not isinstance(encoded, list) or len(encoded) != len(sample.references):
        raise ValueError(f"Complete cache reference count/order mismatch: {path}")
    for index, (entry, spec) in enumerate(zip(encoded, sample.references)):
        if not isinstance(entry, dict) or entry.get("kind") != spec.kind or entry.get("normalized") is not True:
            raise ValueError(f"Invalid encoded references[{index}] identity: {path}")
        video_rows = entry.get("video_latents")
        audio_rows = entry.get("audio_latents")
        if spec.kind != "audio":
            video_rows = _expect_tensor(video_rows, f"references[{index}].video_latents", 2, reference_dtype)
            if video_rows.shape[1] != VIDEO_ROW_WIDTH:
                raise ValueError(f"references[{index}] visual row width must be {VIDEO_ROW_WIDTH}: {path}")
            frames = int(entry.get("num_latent_frames", 0))
            height = int(entry.get("latent_height", 0))
            width = int(entry.get("latent_width", 0))
            if frames <= 0 or height <= 0 or width <= 0 or height % 2 or width % 2:
                raise ValueError(f"references[{index}] has invalid visual geometry: {path}")
            if spec.kind == "image" and frames != 1:
                raise ValueError(f"references[{index}] image must have one latent frame: {path}")
            expected_rows = frames * (height // 2) * (width // 2)
            if video_rows.shape[0] != expected_rows:
                raise ValueError(f"references[{index}] visual row count mismatch: {path}")
        elif video_rows is not None:
            raise ValueError(f"references[{index}] audio cannot contain video_latents: {path}")

        should_have_audio = spec.kind == "audio" or (spec.kind == "video" and spec.include_embedded_audio is True)
        should_not_have_audio = spec.kind == "image" or (spec.kind == "video" and spec.include_embedded_audio is False)
        if audio_rows is not None:
            audio_rows = _expect_tensor(audio_rows, f"references[{index}].audio_latents", 2, reference_dtype)
            num_audio = int(entry.get("num_audio_latents", 0))
            if num_audio <= 0 or tuple(audio_rows.shape) != (2 * num_audio, AUDIO_LATENT_CHANNELS):
                raise ValueError(f"references[{index}] audio row geometry mismatch: {path}")
        if should_have_audio and audio_rows is None:
            raise ValueError(f"references[{index}] declared embedded/audio content but encoded none: {path}")
        if should_not_have_audio and audio_rows is not None:
            raise ValueError(f"references[{index}] declared no audio but encoded audio rows: {path}")
    return stage


def load_cache(
    path: Path,
    sample: SampleSpec,
    dtype: torch.dtype,
    require_complete: bool = False,
) -> tuple[dict, str]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    stage = validate_cache_payload(payload, sample, path, dtype, require_complete)
    return payload, stage


def _dtype(name: str) -> torch.dtype:
    return {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[name]


def import_official_runtime() -> dict:
    try:
        from diffusers import AutoencoderKLMiniMaxH3, AutoencoderKLMiniMaxH3Audio
        from diffusers.modular_pipelines.minimax_h3 import MiniMaxH3Reference
        from diffusers.modular_pipelines.minimax_h3.before_encoder import MiniMaxH3Ref2VASetupStep
        from diffusers.modular_pipelines.minimax_h3.encoders import (
            MiniMaxH3Ref2VAReferenceEncoderStep,
            MiniMaxH3Ref2VATextEncoderStep,
        )
        from transformers import Qwen2TokenizerFast, Qwen3VLForConditionalGeneration, Qwen3VLProcessor

        before_encoder_module = importlib.import_module("diffusers.modular_pipelines.minimax_h3.before_encoder")
    except (ImportError, AttributeError) as error:
        raise RuntimeError("Ref2AV cache encoding requires a Diffusers installation with the MiniMax-H3 modular pipeline and Qwen3-VL reference encoders.") from error
    return {
        "AutoencoderKLMiniMaxH3": AutoencoderKLMiniMaxH3,
        "AutoencoderKLMiniMaxH3Audio": AutoencoderKLMiniMaxH3Audio,
        "MiniMaxH3Reference": MiniMaxH3Reference,
        "MiniMaxH3Ref2VASetupStep": MiniMaxH3Ref2VASetupStep,
        "MiniMaxH3Ref2VAReferenceEncoderStep": MiniMaxH3Ref2VAReferenceEncoderStep,
        "MiniMaxH3Ref2VATextEncoderStep": MiniMaxH3Ref2VATextEncoderStep,
        "before_encoder_module": before_encoder_module,
        "Qwen2TokenizerFast": Qwen2TokenizerFast,
        "Qwen3VLForConditionalGeneration": Qwen3VLForConditionalGeneration,
        "Qwen3VLProcessor": Qwen3VLProcessor,
    }


def _build_official_references(sample: SampleSpec, runtime: dict) -> list:
    reference_class = runtime["MiniMaxH3Reference"]
    references = []
    for index, spec in enumerate(sample.references):
        reference = reference_class(**{spec.kind: str(spec.path)})
        if spec.kind == "video" and spec.include_embedded_audio is not None:
            if reference.has_audio != spec.include_embedded_audio:
                raise ValueError(
                    f"row {sample.source_index} references[{index}] declared include_embedded_audio={spec.include_embedded_audio}, but decoded media has_audio={reference.has_audio}: {spec.path}"
                )
        references.append(reference)
    return references


def prepare_official_references(sample: SampleSpec, runtime: dict, sampling_rate: int = 32000) -> list:
    references = _build_official_references(sample, runtime)
    components = SimpleNamespace(audio_sampling_rate=sampling_rate)
    before_encoder = runtime["before_encoder_module"]
    original_resolver = before_encoder.resolve_reference_image_size
    resize_mode = sample.descriptor["reference_image_resize_mode"]

    def configured_resolver(width: int, height: int) -> tuple[int, int]:
        return resolve_reference_image_size(
            width,
            height,
            target_width=sample.target_width,
            target_height=sample.target_height,
            mode=resize_mode,
            short_edge=int(sample.descriptor["reference_image_short_edge"]),
        )

    # The pinned upstream SetupStep resolves its image policy through this
    # module-level symbol.  Patch only for the synchronous preparation call and
    # restore it even if media validation fails.  Each cache worker is a
    # separate process, so no cross-worker state is shared.
    before_encoder.resolve_reference_image_size = configured_resolver
    try:
        prepared, resolved_num_frames = runtime["MiniMaxH3Ref2VASetupStep"].prepare_references(
            components,
            references,
            sample.target_num_frames,
        )
    finally:
        before_encoder.resolve_reference_image_size = original_resolver
    if resolved_num_frames != sample.target_num_frames:
        raise RuntimeError(f"Official Ref2VA preparation changed target_num_frames from {sample.target_num_frames} to {resolved_num_frames}")
    return prepared


def load_conditioner(runtime: dict, source_root: Path, device: str, dtype: torch.dtype):
    kwargs = {"dtype": dtype, "local_files_only": True}
    if torch.device(device).type != "cpu":
        kwargs["device_map"] = {"": str(device)}
    text_encoder = runtime["Qwen3VLForConditionalGeneration"].from_pretrained(str(source_root / "text_encoder"), **kwargs)
    text_encoder.requires_grad_(False).eval()
    tokenizer = runtime["Qwen2TokenizerFast"].from_pretrained(str(source_root / "tokenizer"), local_files_only=True)
    processor = runtime["Qwen3VLProcessor"].from_pretrained(str(source_root / "processor"), local_files_only=True)
    return SimpleNamespace(
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        processor=processor,
        _execution_device=torch.device(device),
    )


def load_vaes(runtime: dict, model_root: Path, device: str, dtype: torch.dtype):
    vae = runtime["AutoencoderKLMiniMaxH3"].from_pretrained(str(model_root / "vae"), torch_dtype=dtype, local_files_only=True)
    audio_vae = runtime["AutoencoderKLMiniMaxH3Audio"].from_pretrained(str(model_root / "audio_vae"), torch_dtype=dtype, local_files_only=True)
    vae.requires_grad_(False).eval().to(device)
    audio_vae.requires_grad_(False).eval().to(device)
    if int(audio_vae.config.sampling_rate) != 32000:
        raise ValueError(f"Ref2VA audio VAE must use 32000 Hz, got {audio_vae.config.sampling_rate}")
    return SimpleNamespace(
        vae=vae,
        audio_vae=audio_vae,
        _execution_device=torch.device(device),
        patch_size=PATCH_SIZE,
        audio_latent_channels=AUDIO_LATENT_CHANNELS,
    )


def release_models(*values: object) -> None:
    del values
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


@torch.inference_mode()
def encode_text_stage(
    sample: SampleSpec,
    runtime: dict,
    components,
    output_dtype: torch.dtype,
) -> dict:
    prepared = prepare_official_references(sample, runtime)
    prompt_embeds, text_token_tags = runtime["MiniMaxH3Ref2VATextEncoderStep"].encode_prompt(
        components,
        sample.prompt,
        prepared,
        device=components._execution_device,
        dtype=output_dtype,
    )
    positive = {
        "task": "ref2av",
        "prompt_embeds": prompt_embeds.detach().to(device="cpu", dtype=output_dtype).contiguous(),
        "text_token_tags": text_token_tags.detach().cpu().to(torch.long).contiguous(),
        "target_height": sample.target_height,
        "target_width": sample.target_width,
        "target_num_frames": sample.target_num_frames,
    }
    payload = {
        "cache_schema_version": CACHE_SCHEMA_VERSION,
        "cache_stage": "text",
        "cache_fingerprint": sample.fingerprint,
        "cache_metadata": sample.descriptor,
        "conditioning": {"positive": positive},
        "source_index": sample.source_index,
        "source_id": sample.source_id,
        "task": "ref2av",
    }
    validate_cache_payload(payload, sample, Path("<memory>"), output_dtype, require_complete=False)
    return payload


@torch.inference_mode()
def encode_reference_stage(sample: SampleSpec, payload: dict, runtime: dict, components, dtype: torch.dtype) -> dict:
    reference_dtype = _dtype(sample.descriptor.get("reference_latent_dtype", "fp32"))
    prepared = prepare_official_references(
        sample,
        runtime,
        sampling_rate=int(components.audio_vae.config.sampling_rate),
    )
    encoded_entries = []
    for reference, spec in zip(prepared, sample.references):
        video_rows, audio_rows = runtime["MiniMaxH3Ref2VAReferenceEncoderStep"].encode_references(
            components,
            [reference],
            device=components._execution_device,
        )
        entry = {"kind": spec.kind, "normalized": True}
        if spec.kind != "audio":
            if video_rows is None:
                raise RuntimeError(f"Official encoder returned no visual rows for {spec.kind} reference")
            entry.update(
                {
                    "video_latents": video_rows.to(device="cpu", dtype=reference_dtype).contiguous(),
                    "num_latent_frames": int(reference.num_latent_frames),
                    "latent_height": int(reference.latent_height),
                    "latent_width": int(reference.latent_width),
                }
            )
        if audio_rows is not None:
            entry.update(
                {
                    "audio_latents": audio_rows.to(device="cpu", dtype=reference_dtype).contiguous(),
                    "num_audio_latents": int(reference.num_audio_latents),
                }
            )
        encoded_entries.append(entry)
    payload["conditioning"]["positive"]["references"] = encoded_entries
    payload["cache_stage"] = "complete"
    validate_cache_payload(payload, sample, Path("<memory>"), dtype, require_complete=True)
    return payload


def _selected_rows(rows: list[dict], args: argparse.Namespace) -> list[tuple[int, dict]]:
    indexed = list(enumerate(rows))[args.start_index :]
    if args.max_samples is not None:
        indexed = indexed[: args.max_samples]
    return [(source_index, row) for relative_index, (source_index, row) in enumerate(indexed) if relative_index % args.num_shards == args.shard_index]


def read_selected_rows(path: Path, args: argparse.Namespace) -> tuple[list[tuple[int, dict]], int]:
    """Hold only this worker's assigned rows, not 100k full prompts per GPU."""
    selected = []
    total = 0
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            source_index = total
            total += 1
            relative_index = source_index - args.start_index
            if relative_index < 0 or (args.max_samples is not None and relative_index >= args.max_samples):
                continue
            if relative_index % args.num_shards != args.shard_index:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON at {path}:{line_number}: {error}") from error
            if not isinstance(row, dict):
                raise TypeError(f"Expected JSON object at {path}:{line_number}")
            selected.append((source_index, row))
    return selected, total


def _manifest_path(args: argparse.Namespace, output_dir: Path) -> Path:
    if args.manifest_name:
        name = args.manifest_name
    elif args.num_shards > 1:
        name = f"metadata.shard-{args.shard_index:03d}-of-{args.num_shards:03d}.jsonl"
    else:
        name = "metadata.jsonl"
    path = Path(name)
    if path.is_absolute() or len(path.parts) != 1:
        raise ValueError("--manifest-name must be one filename inside --output-dir")
    return output_dir / path


def _merge_current_manifest(path: Path, rows: list[dict], replace: bool) -> list[dict]:
    if replace or not path.is_file():
        return sorted(rows, key=lambda row: row["condition_path"])
    merged = {row["condition_path"]: row for row in read_jsonl(path)}
    for row in rows:
        merged[row["condition_path"]] = row
    return sorted(merged.values(), key=lambda row: row["condition_path"])


def initialize_namespace(
    args: argparse.Namespace,
    metadata: Path,
    output_dir: Path,
    model_descriptor: dict,
) -> None:
    descriptor = {
        "cache_schema_version": CACHE_SCHEMA_VERSION,
        "input_path": str(metadata),
        "input_size_bytes": metadata.stat().st_size,
        "input_sha256": sha256_file(metadata),
        "media_root": str(args.media_root),
        "dtype": args.dtype,
        "reference_image_resize_mode": args.reference_resize_mode,
        "target_num_frames_override": args.target_num_frames,
        "selection": {
            "start_index": args.start_index,
            "max_samples": args.max_samples,
            "num_shards": args.num_shards,
        },
        "model": model_descriptor,
    }
    if (
        getattr(args, "prompt_policy", "enhanced-only") != "enhanced-only"
        or getattr(args, "target_policy", "source") != "source"
        or getattr(args, "reference_latent_dtype", "fp32") != "fp32"
        or getattr(args, "image_only", False)
    ):
        descriptor.update(
            prompt_policy=args.prompt_policy,
            target_policy=args.target_policy,
            reference_latent_dtype=args.reference_latent_dtype,
            image_only=args.image_only,
        )
    fingerprint = digest_object(descriptor)
    path = output_dir / "preprocess_config.json"
    expected = {
        "cache_schema_version": CACHE_SCHEMA_VERSION,
        "preprocess_fingerprint": fingerprint,
        "preprocess_config": descriptor,
    }
    if path.is_file():
        actual = json.loads(path.read_text(encoding="utf-8"))
        if actual.get("preprocess_fingerprint") != fingerprint:
            raise RuntimeError(
                f"Output directory belongs to a different input/model/media-root/dtype: {output_dir}. "
                "The start/max/shard selection is also part of this identity. "
                "Use a new --output-dir; caches are not silently adopted."
            )
        return
    atomic_create_json(path, expected)
    # A concurrent shard may have won the no-clobber publication.
    actual = json.loads(path.read_text(encoding="utf-8"))
    if actual.get("preprocess_fingerprint") != fingerprint:
        raise RuntimeError(f"Concurrent shard initialized a different cache namespace at {output_dir}")


def merge_shards(args: argparse.Namespace) -> None:
    output_dir = args.output_dir.expanduser().resolve()
    rows = []
    seen = set()
    for shard_index in range(args.num_shards):
        path = output_dir / f"metadata.shard-{shard_index:03d}-of-{args.num_shards:03d}.jsonl"
        if not path.is_file():
            raise FileNotFoundError(f"Missing completed shard manifest: {path}")
        for row in read_jsonl(path):
            row_keys = set(row)
            if not MANIFEST_REQUIRED_KEYS <= row_keys or row_keys - (MANIFEST_REQUIRED_KEYS | MANIFEST_COST_KEYS):
                raise ValueError(f"Shard manifest has invalid Ref2AV fields: required={sorted(MANIFEST_REQUIRED_KEYS)}, path={path}, row={row}")
            condition = output_dir / row["condition_path"]
            if not condition.is_file():
                raise FileNotFoundError(f"Shard manifest references missing condition cache: {condition}")
            if row["condition_path"] in seen:
                raise ValueError(f"Duplicate condition_path across shard manifests: {row['condition_path']}")
            payload = torch.load(condition, map_location="cpu", weights_only=False)
            positive = payload.get("conditioning", {}).get("positive") if isinstance(payload, dict) else None
            if (
                not isinstance(payload, dict)
                or payload.get("cache_schema_version") != CACHE_SCHEMA_VERSION
                or payload.get("cache_stage") != "complete"
                or not isinstance(positive, dict)
                or not isinstance(positive.get("references"), list)
                or not positive["references"]
            ):
                raise ValueError(f"Shard manifest references incomplete cache: {condition}")
            seen.add(row["condition_path"])
            rows.append(row)
    rows.sort(key=lambda row: row["condition_path"])
    atomic_write_jsonl(output_dir / "metadata.jsonl", rows)
    print(f"Merged {len(rows)} complete condition rows into {output_dir / 'metadata.jsonl'}", flush=True)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    args.output_dir = args.output_dir.expanduser().resolve()
    if args.merge_shards:
        merge_shards(args)
        return 0

    args.metadata = args.metadata.expanduser().resolve()
    if not args.metadata.is_file():
        raise FileNotFoundError(f"Metadata does not exist: {args.metadata}")
    args.media_root = args.media_root.expanduser().resolve() if args.media_root is not None else args.metadata.parent
    args.model_path = args.model_path.expanduser().resolve()
    args.source_model_path = args.source_model_path.expanduser().resolve()
    dtype = _dtype(args.dtype)
    model_descriptor = model_identity(args.source_model_path, args.model_path)
    selected, input_total_rows = read_selected_rows(args.metadata, args)
    if input_total_rows == 0:
        raise RuntimeError("Input metadata contains no rows")

    hash_cache: dict[Path, str] = {}
    samples = []
    failures = []
    failed_indices = set()
    manifest = _manifest_path(args, args.output_dir)
    failure_path = manifest.with_suffix(".failed.jsonl")
    receipt_path = manifest.with_suffix(".complete.json")

    def record_failure(source_index, source_id, stage, error, *, persist=False):
        if not args.skip_invalid:
            raise error
        failure = {
            "source_index": source_index,
            "source_id": str(source_id),
            "stage": stage,
            "error_type": type(error).__name__,
            "error": str(error),
        }
        failures.append(failure)
        failed_indices.add(source_index)
        print(f"[skip] {json.dumps(failure, ensure_ascii=False)}", file=sys.stderr, flush=True)
        if persist:
            with failure_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(failure, ensure_ascii=False) + "\n")
                handle.flush()
                os.fsync(handle.fileno())

    for source_index, row in selected:
        try:
            sample = normalize_sample(
                row,
                source_index,
                args.media_root,
                args.dtype,
                model_descriptor,
                hash_cache,
                reference_resize_mode=args.reference_resize_mode,
                target_num_frames_override=args.target_num_frames,
                prompt_policy=args.prompt_policy,
                target_policy=args.target_policy,
                reference_latent_dtype=args.reference_latent_dtype,
                image_only=args.image_only,
            )
        except (OSError, ValueError, TypeError, KeyError) as error:
            record_failure(source_index, row.get("sample_id", row.get("id", source_index)), "normalize", error)
        else:
            samples.append(sample)
    # Raw prompts can be large; retain only normalized sample descriptors.
    selected_count = len(selected)
    del selected
    counts = Counter(tuple(reference.kind for reference in sample.references) for sample in samples)
    geometry = Counter((sample.target_height, sample.target_width, sample.target_num_frames) for sample in samples)
    print(
        f"Validated rows={len(samples)} shard={args.shard_index}/{args.num_shards} "
        f"unique_media={len(hash_cache)} reference_orders={dict(counts)} geometries={len(geometry)} "
        f"reference_resize_mode={args.reference_resize_mode} "
        f"target_num_frames_override={args.target_num_frames}",
        flush=True,
    )
    if args.dry_run:
        print("Dry-run complete: no cache directory or model weights were opened.", flush=True)
        return 0

    args.output_dir.mkdir(parents=True, exist_ok=True)
    initialize_namespace(args, args.metadata, args.output_dir, model_descriptor)
    # A failed/incomplete retry must never leave a previous completion receipt
    # looking current. The caller's per-node/shard lock excludes duplicate jobs.
    receipt_path.unlink(missing_ok=True)
    atomic_write_jsonl(failure_path, failures)
    conditions_dir = args.output_dir / "conditions"
    conditions_dir.mkdir(parents=True, exist_ok=True)

    entries = []
    text_jobs = []
    reference_jobs = []
    for sample in samples:
        relative_path, output_path = cache_path(args.output_dir, sample.source_index)
        stage = None
        if output_path.is_file() and not args.overwrite:
            _, stage = load_cache(output_path, sample, dtype)
        if args.overwrite or stage is None:
            if args.stage == "references":
                raise RuntimeError(f"--stage references requires an existing validated text-stage cache: {output_path}")
            text_jobs.append((sample, output_path))
            stage = "text"
        if stage != "complete" and args.stage in {"all", "references"}:
            reference_jobs.append((sample, output_path))
        entries.append((sample, relative_path, output_path))

    runtime = None
    if text_jobs:
        runtime = import_official_runtime()
        print(
            f"Loading Ref2VA Qwen3-VL from {args.source_model_path} on {args.device}; text_encode={len(text_jobs)} ...",
            flush=True,
        )
        conditioner = load_conditioner(runtime, args.source_model_path, args.device, dtype)
        for index, (sample, output_path) in enumerate(text_jobs, start=1):
            try:
                payload = encode_text_stage(sample, runtime, conditioner, dtype)
            except (OSError, ValueError, TypeError, KeyError) as error:
                record_failure(sample.source_index, sample.source_id, "text", error, persist=True)
                continue
            atomic_torch_save(payload, output_path)
            print(f"[text {index}/{len(text_jobs)}] {output_path}", flush=True)
        del conditioner
        release_models()

    reference_jobs = [(sample, path) for sample, path in reference_jobs if sample.source_index not in failed_indices]
    if reference_jobs:
        runtime = runtime or import_official_runtime()
        print(
            f"Loading converted Ref2VA video/audio VAEs from {args.model_path} on {args.device}; reference_encode={len(reference_jobs)} ...",
            flush=True,
        )
        vaes = load_vaes(runtime, args.model_path, args.device, dtype)
        for index, (sample, output_path) in enumerate(reference_jobs, start=1):
            payload, stage = load_cache(output_path, sample, dtype)
            if stage == "complete" and not args.overwrite:
                continue
            try:
                payload = encode_reference_stage(sample, payload, runtime, vaes, dtype)
            except (OSError, ValueError, TypeError, KeyError) as error:
                record_failure(sample.source_index, sample.source_id, "references", error, persist=True)
                continue
            atomic_torch_save(payload, output_path)
            print(f"[refs {index}/{len(reference_jobs)}] {output_path}", flush=True)
        del vaes
        release_models()

    completed_rows = []
    incomplete = 0
    for sample, relative_path, output_path in entries:
        if sample.source_index in failed_indices:
            continue
        payload, stage = load_cache(output_path, sample, dtype)
        if stage == "complete":
            completed_rows.append(manifest_row(sample, relative_path, payload))
        else:
            incomplete += 1
    # A text-only stage is a valid resumable result but must never be exposed
    # to LatentDataset as a training row.
    if args.stage == "text":
        print(
            f"Text caches ready={len(entries)} pending_reference_stage={incomplete}; no training manifest written.",
            flush=True,
        )
    else:
        if incomplete:
            raise RuntimeError(f"Refusing to write training manifest with {incomplete} incomplete caches")
        merged = _merge_current_manifest(manifest, completed_rows, args.replace_manifest or args.num_shards > 1)
        atomic_write_jsonl(manifest, merged)
        namespace = json.loads((args.output_dir / "preprocess_config.json").read_text(encoding="utf-8"))
        receipt = {
            "cache_schema_version": CACHE_SCHEMA_VERSION,
            "preprocess_fingerprint": namespace["preprocess_fingerprint"],
            "stage": args.stage,
            "manifest": manifest.name,
            "manifest_sha256": sha256_file(manifest),
            "failures": failure_path.name,
            "failures_sha256": sha256_file(failure_path),
            "input_total_rows": input_total_rows,
            "selected_count": selected_count,
            "completed_count": len(completed_rows),
            "failed_count": len(failed_indices),
            "num_shards": args.num_shards,
            "shard_index": args.shard_index,
        }
        atomic_write_json(receipt_path, receipt)
        print(f"Wrote complete rows={len(merged)} current={len(completed_rows)} to {manifest}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
