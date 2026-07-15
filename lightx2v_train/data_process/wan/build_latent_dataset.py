#!/usr/bin/env python3
"""Build a Wan latent_dataset cache from video/caption metadata.

The output is directly consumable by lightx2v_train.data.latent_dataset:

  output_dir/
    metadata.jsonl
    latents/*.pt
    conditions/*.pt

Each manifest row may contain video_latent_path, condition_path, or both,
depending on --cache-components.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lightx2v_train.data.utils import prompt_text, read_records, record_value, resolve_data_path  # noqa: E402
from lightx2v_train.model_zoo.native.wan.modules.t5 import T5EncoderModel  # noqa: E402
from lightx2v_train.model_zoo.native.wan.modules.vae2_2 import Wan2_2_VAE  # noqa: E402
from lightx2v_train.utils.constants import WAN_NEGATIVE_PROMPT  # noqa: E402

TORCH_DTYPES = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}


@dataclass
class ClipEntry:
    path: Path
    prompt: str
    frame_count: int | None
    metadata_path: Path
    row_index: int
    prefix: str
    number: int | None


def parse_args():
    parser = argparse.ArgumentParser(description="Build Wan video/prompt latent_dataset cache from json/jsonl/csv metadata.")
    parser.add_argument("metadata", nargs="+", help="CSV/JSON/JSONL metadata file(s), or video directory.")
    parser.add_argument("--output-dir", required=True, help="Output latent_dataset directory.")
    parser.add_argument("--cache-components", default="all", choices=["all", "video", "prompt"], help="Build video latents, prompt conditions, or both.")
    parser.add_argument("--model-dir", default="/path/to/models/Wan2.2-TI2V-5B", help="Wan2.2 model directory.")
    parser.add_argument("--vae-checkpoint", default=None, help="Override Wan VAE checkpoint path.")
    parser.add_argument("--t5-checkpoint", default=None, help="Override T5 checkpoint path.")
    parser.add_argument("--tokenizer-path", default=None, help="Override tokenizer path.")
    parser.add_argument("--video-root", action="append", default=[], help="Root used to resolve relative video paths. Can be passed more than once.")
    parser.add_argument("--video-column", default="video")
    parser.add_argument("--prompt-column", default="caption")
    parser.add_argument("--height", type=int, default=704)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--raw-frames", type=int, default=None, help="Raw frames per sample. Defaults to 1 + ratio * (latent_frames - 1).")
    parser.add_argument("--latent-frames", type=int, default=96)
    parser.add_argument("--temporal-compression-ratio", type=int, default=4)
    parser.add_argument("--frame-rate", type=float, default=24.0)
    parser.add_argument("--fix-frame-rate", action="store_true")
    parser.add_argument("--prompt-separator", default=" The scene transitions. ")
    parser.add_argument("--video-glob", default="*.mp4", help="Glob used when metadata path is a directory.")
    parser.add_argument("--recursive", action="store_true", help="Recursively scan video directories.")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-groups", type=int, default=None)
    parser.add_argument("--rank", type=int, default=int(os.environ.get("RANK", "0")))
    parser.add_argument("--local-rank", "--local_rank", dest="local_rank", type=int, default=int(os.environ.get("LOCAL_RANK", "0")))
    parser.add_argument("--world-size", type=int, default=int(os.environ.get("WORLD_SIZE", "1")))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--text-device", default=None, help="Defaults to --device.")
    parser.add_argument("--vae-dtype", default="fp32", choices=["fp32", "fp16", "bf16"])
    parser.add_argument("--t5-dtype", default="bf16", choices=["fp32", "fp16", "bf16"])
    parser.add_argument("--save-dtype", default="bf16", choices=["fp32", "fp16", "bf16"])
    parser.add_argument("--max-sequence-length", type=int, default=512)
    parser.add_argument("--vae-chunk-latent-frames", type=int, default=16, help="Use 0 to encode the full video in one VAE call.")
    parser.add_argument("--vae-halo-latents", type=int, default=28)
    parser.add_argument("--skip-negative-cache", action="store_true")
    parser.add_argument("--skip-missing", action="store_true", default=True)
    parser.add_argument("--no-skip-missing", dest="skip_missing", action="store_false")
    parser.add_argument("--strict-decode", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--log-source-limit", type=int, default=0)
    return parser.parse_args()


def need_video(args):
    return args.cache_components in {"all", "video"}


def need_prompt(args):
    return args.cache_components in {"all", "prompt"}


def crop_and_resize(image, target_height, target_width):
    width, height = image.size
    scale = max(target_width / width, target_height / height)
    resized_width = round(width * scale)
    resized_height = round(height * scale)
    image = image.resize((resized_width, resized_height), Image.Resampling.BILINEAR)
    left = max(0, (resized_width - target_width) // 2)
    top = max(0, (resized_height - target_height) // 2)
    return image.crop((left, top, left + target_width, top + target_height))


def frame_to_tensor(frame):
    array = np.array(frame, dtype=np.uint8, copy=True)
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


def numeric_parts(path):
    match = re.search(r"(\d+)$", path.stem)
    if not match:
        return path.stem, None
    return path.stem[: match.start(1)], int(match.group(1))


def load_sidecar_prompt(video_path, prompt_column):
    for sidecar in (video_path.with_suffix(".txt"), video_path.with_suffix(".json")):
        if not sidecar.is_file():
            continue
        try:
            if sidecar.suffix == ".txt":
                return sidecar.read_text(encoding="utf-8").strip()
            payload = json.loads(sidecar.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                return str(payload.get(prompt_column, payload.get("caption", payload.get("prompt", ""))))
        except Exception:
            return ""
    return ""


def iter_rows(metadata_path, args):
    metadata_path = Path(metadata_path)
    if metadata_path.is_dir():
        pattern = f"**/{args.video_glob}" if args.recursive else args.video_glob
        for index, video_path in enumerate(sorted(metadata_path.glob(pattern))):
            yield {"video": str(video_path), "caption": load_sidecar_prompt(video_path, args.prompt_column)}, index
        return
    for index, row in enumerate(read_records(metadata_path, prompt_column=args.prompt_column)):
        yield row, index


def resolve_video(metadata_path, row, args):
    video_value = record_value(row, args.video_column, "video_path", "video")
    if video_value is None:
        return None
    base_dir = metadata_path if metadata_path.is_dir() else metadata_path.parent
    return resolve_data_path(video_value, base_dir, roots=args.video_root, subdirs=("video", "videos"))


def parse_optional_int(value):
    if value is None or value == "":
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def load_entries(args):
    entries = []
    entry_limit = None
    if not need_video(args) and args.max_samples is not None:
        entry_limit = int(args.max_samples) * max(1, int(args.world_size))
    for metadata in args.metadata:
        metadata_path = Path(metadata)
        for row, row_index in iter_rows(metadata_path, args):
            video_path = resolve_video(metadata_path, row, args)
            if need_video(args) and args.skip_missing and (video_path is None or not video_path.is_file()):
                continue
            prompt = prompt_text(row, args.prompt_column)
            if not need_video(args):
                entries.append(ClipEntry(path=video_path or Path(""), prompt=prompt, frame_count=None, metadata_path=metadata_path, row_index=row_index, prefix="", number=None))
                if entry_limit is not None and len(entries) >= entry_limit:
                    return entries
                continue
            prefix, number = numeric_parts(video_path)
            entries.append(
                ClipEntry(
                    path=video_path,
                    prompt=prompt,
                    frame_count=parse_optional_int(record_value(row, "frames", "frame", "num_frames")),
                    metadata_path=metadata_path,
                    row_index=row_index,
                    prefix=prefix,
                    number=number,
                )
            )
    return entries


def group_consecutive(entries):
    entries = sorted(
        entries,
        key=lambda item: (
            str(item.path.parent),
            item.prefix,
            item.number is None,
            item.number if item.number is not None else item.path.stem,
            item.path.name,
        ),
    )
    groups = []
    current = []
    for entry in entries:
        if not current:
            current = [entry]
            continue
        prev = current[-1]
        is_next = prev.path.parent == entry.path.parent and prev.prefix == entry.prefix and prev.number is not None and entry.number is not None and entry.number == prev.number + 1
        if is_next:
            current.append(entry)
        else:
            groups.append(current)
            current = [entry]
    if current:
        groups.append(current)
    return groups


def usable_frame_count(reader, args):
    total_raw_frames = int(reader.count_frames())
    if not args.fix_frame_rate:
        return total_raw_frames
    metadata = reader.get_meta_data()
    duration = metadata.get("duration") or total_raw_frames / metadata.get("fps", args.frame_rate)
    return int(math.floor(duration * args.frame_rate))


def iter_frame_ids(reader, args):
    total_raw_frames = int(reader.count_frames())
    raw_fps = reader.get_meta_data().get("fps", args.frame_rate)
    count = usable_frame_count(reader, args)
    for frame_id in range(count):
        if args.fix_frame_rate:
            target_time = frame_id / args.frame_rate
            yield min(int(round(target_time * raw_fps)), total_raw_frames - 1)
        else:
            yield frame_id


def iter_video_frames(video_path, args):
    reader = imageio.get_reader(video_path)
    try:
        for frame_id in iter_frame_ids(reader, args):
            frame = Image.fromarray(reader.get_data(frame_id)).convert("RGB")
            frame = crop_and_resize(frame, args.height, args.width)
            yield frame_to_tensor(frame)
    finally:
        reader.close()


def concat_prompt(parts, separator):
    cleaned = [part.strip() for part in parts if part and part.strip()]
    return separator.join(cleaned) if cleaned else " "


def iter_group_windows(group, args):
    frames = []
    prompt_parts = []
    source_paths = []
    source_set = set()
    window_index = 0

    def reset_window():
        frames.clear()
        prompt_parts.clear()
        source_paths.clear()
        source_set.clear()

    for clip in group:
        try:
            for frame in iter_video_frames(clip.path, args):
                path_key = str(clip.path)
                if path_key not in source_set:
                    source_set.add(path_key)
                    source_paths.append(path_key)
                    prompt_parts.append(clip.prompt)
                frames.append(frame)
                if len(frames) == args.raw_frames:
                    video = torch.stack(frames, dim=1).contiguous()
                    yield window_index, video, concat_prompt(prompt_parts, args.prompt_separator), list(source_paths)
                    window_index += 1
                    reset_window()
        except Exception as error:
            if args.strict_decode:
                raise
            print(f"[warn] skip bad video {clip.path}: {error}", flush=True)
            reset_window()


def count_group_frames(group, args):
    if all(clip.frame_count is not None for clip in group):
        return sum(clip.frame_count for clip in group)
    total = 0
    for clip in group:
        try:
            reader = imageio.get_reader(clip.path)
            try:
                total += usable_frame_count(reader, args)
            finally:
                reader.close()
        except Exception as error:
            if args.strict_decode:
                raise
            print(f"[warn] cannot count {clip.path}: {error}", flush=True)
    return total


def latent_range_to_raw_window(latent_start, latent_end, ratio):
    if latent_start == 0:
        return 0, 1 + ratio * (latent_end - 1), 0
    return ratio * (latent_start - 1), 1 + ratio * (latent_end - 1), 1


def encode_video_latent(vae, video_cpu, args):
    device = torch.device(args.device)
    vae_dtype = TORCH_DTYPES[args.vae_dtype]
    latent_parts = []

    def prepare_video_chunk(chunk_cpu):
        return chunk_cpu.to(device=device, dtype=vae_dtype).div_(127.5).sub_(1.0)

    with torch.inference_mode():
        if args.vae_chunk_latent_frames <= 0:
            video = prepare_video_chunk(video_cpu)
            return vae.encode([video])[0].cpu()
        chunk = args.vae_chunk_latent_frames
        for keep_start in range(0, args.latent_frames, chunk):
            keep_end = min(args.latent_frames, keep_start + chunk)
            halo_start = max(0, keep_start - args.vae_halo_latents)
            raw_start, raw_end, pseudo_prefix = latent_range_to_raw_window(halo_start, keep_end, args.temporal_compression_ratio)
            drop_latents = pseudo_prefix + (keep_start - halo_start)
            video = prepare_video_chunk(video_cpu[:, raw_start:raw_end].contiguous())
            encoded = vae.encode([video])[0].cpu()
            keep = encoded[:, drop_latents : drop_latents + (keep_end - keep_start)].contiguous()
            if keep.shape[1] != keep_end - keep_start:
                raise RuntimeError(f"VAE chunk produced {keep.shape[1]} latent frames, expected {keep_end - keep_start}.")
            latent_parts.append(keep)
            del video, encoded, keep
            if device.type == "cuda":
                torch.cuda.empty_cache()
        return torch.cat(latent_parts, dim=1)


def encode_prompt_embed(text_encoder, prompt, args):
    device = torch.device(args.text_device or args.device)
    with torch.inference_mode():
        contexts = text_encoder([prompt], device)
        context = contexts[0][: args.max_sequence_length].to(dtype=TORCH_DTYPES[args.save_dtype]).cpu()
        if context.shape[0] < args.max_sequence_length:
            pad = context.new_zeros(args.max_sequence_length - context.shape[0], context.shape[1])
            context = torch.cat([context, pad], dim=0)
        return context


def load_vae(args):
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    model_dir = Path(args.model_dir)
    checkpoint = args.vae_checkpoint or str(model_dir / "Wan2.2_VAE.pth")
    print(f"loading Wan VAE: {checkpoint}", flush=True)
    return Wan2_2_VAE(vae_pth=checkpoint, dtype=TORCH_DTYPES[args.vae_dtype], device=device)


def load_text_encoder(args):
    model_dir = Path(args.model_dir)
    checkpoint = args.t5_checkpoint or str(model_dir / "models_t5_umt5-xxl-enc-bf16.pth")
    tokenizer_path = args.tokenizer_path or str(model_dir / "google" / "umt5-xxl")
    device = torch.device(args.text_device or args.device)
    print(f"loading Wan T5: {checkpoint}", flush=True)
    return T5EncoderModel(
        text_len=args.max_sequence_length,
        dtype=TORCH_DTYPES[args.t5_dtype],
        device=device,
        checkpoint_path=checkpoint,
        tokenizer_path=tokenizer_path,
    )


def save_pt(path, payload, overwrite):
    if path.exists() and not overwrite:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    torch.save(payload, tmp_path)
    tmp_path.replace(path)
    return True


def condition_payload(prompt_embed, prompt):
    return {"prompt": prompt, "conditions": {"prompt_embed": prompt_embed.contiguous()}}


def validate_args(args):
    expected_raw = 1 + args.temporal_compression_ratio * (args.latent_frames - 1)
    if args.raw_frames != expected_raw:
        raise ValueError(f"--raw-frames must be 1 + ratio * (latent_frames - 1) = {expected_raw}, got {args.raw_frames}.")
    if args.height % 16 != 0 or args.width % 16 != 0:
        raise ValueError(f"height/width must be divisible by 16, got {args.height}x{args.width}.")
    if args.rank < 0 or args.rank >= args.world_size:
        raise ValueError(f"rank must be in [0, world_size), got rank={args.rank}, world_size={args.world_size}.")


def manifest_path(output_dir, args):
    if args.world_size == 1:
        return output_dir / "metadata.jsonl"
    return output_dir / f"metadata_rank{args.rank:02d}.jsonl"


def build_prompt_only(entries, args, text_encoder, output_dir):
    rows = []
    conditions_dir = output_dir / "conditions"
    save_dtype = TORCH_DTYPES[args.save_dtype]
    for idx, entry in enumerate(entries):
        if idx % args.world_size != args.rank:
            continue
        if args.max_samples is not None and len(rows) >= args.max_samples:
            break
        condition_rel = Path("conditions") / f"condition_{idx:08d}.pt"
        condition_path = output_dir / condition_rel
        prompt_embed = encode_prompt_embed(text_encoder, entry.prompt, args).to(dtype=save_dtype)
        save_pt(condition_path, condition_payload(prompt_embed, entry.prompt), args.overwrite)
        rows.append({"condition_path": str(condition_rel), "caption": entry.prompt, "row_index": entry.row_index})
    return rows


def main():
    args = parse_args()
    if args.raw_frames is None:
        args.raw_frames = 1 + args.temporal_compression_ratio * (args.latent_frames - 1)
    if args.device == "cuda":
        args.device = f"cuda:{args.local_rank}"
    if args.text_device == "cuda":
        args.text_device = f"cuda:{args.local_rank}"
    validate_args(args)
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    entries = load_entries(args)
    print(f"entries={len(entries)} cache_components={args.cache_components} output_dir={output_dir}", flush=True)

    if args.dry_run and not need_video(args):
        print(f"dry_run_prompt_samples={len(entries)}", flush=True)
        return

    if need_video(args):
        groups = group_consecutive(entries)
        if args.max_groups is not None:
            groups = groups[: args.max_groups]
        local_groups = [(index, group) for index, group in enumerate(groups) if index % args.world_size == args.rank]
        print(f"groups={len(groups)} local_groups={len(local_groups)} raw_frames={args.raw_frames} latent_frames={args.latent_frames}", flush=True)
        if args.dry_run:
            total_windows = 0
            for group_index, group in local_groups:
                frame_count = count_group_frames(group, args)
                windows = frame_count // args.raw_frames
                total_windows += windows
                print(f"group={group_index:06d} clips={len(group)} frames={frame_count} windows={windows} first={group[0].path} last={group[-1].path}", flush=True)
            print(f"dry_run_windows={total_windows}", flush=True)
            return
    else:
        groups = []
        local_groups = []

    vae = load_vae(args) if need_video(args) else None
    text_encoder = load_text_encoder(args) if need_prompt(args) else None
    save_dtype = TORCH_DTYPES[args.save_dtype]
    rows = []

    if need_prompt(args) and not args.skip_negative_cache:
        negative_path = output_dir / "negative_condition.pt"
        prompt_embed = encode_prompt_embed(text_encoder, WAN_NEGATIVE_PROMPT, args).to(dtype=save_dtype)
        save_pt(negative_path, condition_payload(prompt_embed, WAN_NEGATIVE_PROMPT), args.overwrite)

    if not need_video(args):
        rows = build_prompt_only(entries, args, text_encoder, output_dir)
    else:
        saved = 0
        for local_index, (group_index, group) in enumerate(local_groups):
            if args.log_every > 0 and local_index % args.log_every == 0:
                print(f"processing group {local_index + 1}/{len(local_groups)} global={group_index}", flush=True)
            for window_index, video, prompt, source_paths in iter_group_windows(group, args):
                if args.max_samples is not None and saved >= args.max_samples:
                    break
                stem = f"sample_g{group_index:06d}_w{window_index:03d}"
                row = {
                    "caption": prompt,
                    "video_paths": source_paths,
                    "raw_num_frames": args.raw_frames,
                    "latent_num_frames": args.latent_frames,
                    "height": args.height,
                    "width": args.width,
                }
                if need_video(args):
                    latent_rel = Path("latents") / f"{stem}.pt"
                    latent = encode_video_latent(vae, video, args)
                    if latent.shape[1] != args.latent_frames:
                        raise RuntimeError(f"Expected {args.latent_frames} latent frames, got shape={tuple(latent.shape)}")
                    latent_payload = {
                        "latents": latent.to(dtype=save_dtype).contiguous(),
                        "num_frames": int(latent.shape[1]),
                        "height": int(latent.shape[2]),
                        "width": int(latent.shape[3]),
                        "fps": args.frame_rate,
                    }
                    save_pt(output_dir / latent_rel, latent_payload, args.overwrite)
                    row["video_latent_path"] = str(latent_rel)
                if need_prompt(args):
                    condition_rel = Path("conditions") / f"{stem}.pt"
                    prompt_embed = encode_prompt_embed(text_encoder, prompt, args).to(dtype=save_dtype)
                    save_pt(output_dir / condition_rel, condition_payload(prompt_embed, prompt), args.overwrite)
                    row["condition_path"] = str(condition_rel)
                rows.append(row)
                saved += 1
                del video
            if args.max_samples is not None and saved >= args.max_samples:
                break

    out_manifest = manifest_path(output_dir, args)
    tmp_manifest = out_manifest.with_suffix(f"{out_manifest.suffix}.tmp.{os.getpid()}")
    with tmp_manifest.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(tmp_manifest, out_manifest)
    print(f"Wrote manifest: {out_manifest} ({len(rows)} samples)", flush=True)


if __name__ == "__main__":
    main()
