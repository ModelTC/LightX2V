#!/usr/bin/env python3
"""Build Wan2.2 5B teacher-forcing cache from numbered video segments.

The script groups mp4 files whose numeric suffix is consecutive, concatenates
their frames and prompts, encodes 381 raw frames into 96 Wan2.2 latent frames,
and saves .pt files consumable by wan_t2v_cached_dataset.
"""

import argparse
import csv
import json
import math
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


DEFAULT_MODEL_DIR = "/data/nvme0/gushiqiao/models/official_models/wan2.2/Wan2.2-TI2V-5B"
torch = None
imageio = None
Image = None


@dataclass
class ClipEntry:
    path: Path
    prompt: str
    frame_count: Optional[int]
    metadata_path: Path
    row_index: int
    prefix: str
    number: Optional[int]


def parse_args():
    parser = argparse.ArgumentParser(description="Concatenate consecutive numbered videos and cache Wan2.2 5B 96-frame latents.")
    parser.add_argument("--metadata", nargs="+", required=True, help="CSV/JSON/JSONL metadata file(s), or video directory.")
    parser.add_argument("--video-root", default=None, help="Root used to resolve relative video paths from metadata.")
    parser.add_argument("--output-dir", required=True, help="Directory for cached .pt files.")
    parser.add_argument("--model-dir", default=DEFAULT_MODEL_DIR, help="Wan2.2-TI2V-5B model directory.")
    parser.add_argument("--vae-checkpoint", default=None, help="Override Wan2.2_VAE.pth path.")
    parser.add_argument("--t5-checkpoint", default=None, help="Override models_t5_umt5-xxl-enc-bf16.pth path.")
    parser.add_argument("--tokenizer-path", default=None, help="Override google/umt5-xxl tokenizer path.")
    parser.add_argument("--video-column", default="video")
    parser.add_argument("--prompt-column", default="caption")
    parser.add_argument("--height", type=int, default=704)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--raw-frames", type=int, default=381)
    parser.add_argument("--latent-frames", type=int, default=96)
    parser.add_argument("--temporal-compression-ratio", type=int, default=4)
    parser.add_argument("--frame-rate", type=float, default=24.0)
    parser.add_argument("--fix-frame-rate", action="store_true")
    parser.add_argument("--prompt-separator", default=" The scene transitions. ")
    parser.add_argument("--video-glob", default="*.mp4", help="Glob used when --metadata points to a directory.")
    parser.add_argument("--recursive", action="store_true", help="Recursively collect videos from metadata directories.")
    parser.set_defaults(skip_missing=True)
    parser.add_argument("--skip-missing", dest="skip_missing", action="store_true")
    parser.add_argument("--no-skip-missing", dest="skip_missing", action="store_false")
    parser.add_argument("--strict-decode", action="store_true", help="Fail on the first bad video instead of skipping it.")
    parser.add_argument("--max-groups", type=int, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--rank", type=int, default=int(os.environ.get("RANK", "0")))
    parser.add_argument("--local-rank", "--local_rank", dest="local_rank", type=int, default=int(os.environ.get("LOCAL_RANK", "0")))
    parser.add_argument("--world-size", type=int, default=int(os.environ.get("WORLD_SIZE", "1")))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--text-device", default=None, help="Defaults to --device.")
    parser.add_argument("--vae-dtype", default="fp32", choices=["fp32", "fp16", "bf16"])
    parser.add_argument("--t5-dtype", default="bf16", choices=["fp32", "fp16", "bf16"])
    parser.add_argument("--save-dtype", default="bf16", choices=["fp32", "fp16", "bf16"])
    parser.add_argument("--max-sequence-length", type=int, default=512)
    parser.add_argument(
        "--vae-chunk-latent-frames",
        type=int,
        default=16,
        help="Encode in latent chunks to lower peak memory. Use 0 for one full VAE encode.",
    )
    parser.add_argument("--vae-halo-latents", type=int, default=28)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Only group videos and count possible samples.")
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument(
        "--log-source-limit",
        type=int,
        default=0,
        help="Maximum source videos to print per cached sample. 0 prints all source videos.",
    )
    return parser.parse_args()


def torch_dtype(name):
    ensure_torch()
    return {
        "fp32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }[name]


def ensure_torch():
    global torch
    if torch is not None:
        return
    try:
        import torch as torch_module
    except ModuleNotFoundError as error:
        raise ModuleNotFoundError("This cache builder must run in an environment with PyTorch installed.") from error
    torch = torch_module


def resolve_torchrun_device(device, local_rank):
    if device is None:
        return None
    device = str(device)
    if device == "cuda":
        return f"cuda:{local_rank}"
    return device


def normalize_runtime_args(args):
    args.device = resolve_torchrun_device(args.device, args.local_rank)
    if args.text_device is not None:
        args.text_device = resolve_torchrun_device(args.text_device, args.local_rank)
    return args


def ensure_video_imports():
    global imageio, Image
    if imageio is None:
        try:
            import imageio.v2 as imageio_module
        except ModuleNotFoundError:
            try:
                import imageio as imageio_module
            except ModuleNotFoundError as error:
                raise ModuleNotFoundError(
                    "This cache builder needs imageio to decode videos. Install imageio/imageio-ffmpeg or run it in the same environment that can use wan_t2v_video_dataset."
                ) from error
        imageio = imageio_module
    if Image is None:
        try:
            from PIL import Image as image_module
        except ModuleNotFoundError as error:
            raise ModuleNotFoundError("This cache builder needs pillow to resize video frames.") from error
        Image = image_module


def pil_resampling(name):
    if hasattr(Image, "Resampling"):
        return getattr(Image.Resampling, name)
    return getattr(Image, name)


def crop_and_resize(image, target_height, target_width):
    width, height = image.size
    scale = max(target_width / width, target_height / height)
    resized_width = round(width * scale)
    resized_height = round(height * scale)
    image = image.resize((resized_width, resized_height), pil_resampling("BILINEAR"))

    left = max(0, (resized_width - target_width) // 2)
    top = max(0, (resized_height - target_height) // 2)
    return image.crop((left, top, left + target_width, top + target_height))


def frame_to_tensor(frame):
    ensure_torch()
    import numpy as np

    array = np.array(frame, dtype=np.uint8, copy=True)
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


def numeric_parts(path):
    match = re.search(r"(\d+)$", path.stem)
    if not match:
        return path.stem, None
    return path.stem[: match.start(1)], int(match.group(1))


def iter_metadata_rows(metadata_path, args):
    if metadata_path.is_dir():
        pattern = f"**/{args.video_glob}" if args.recursive else args.video_glob
        for index, video_path in enumerate(sorted(metadata_path.glob(pattern))):
            yield {"video": str(video_path), "caption": load_sidecar_prompt(video_path, args.prompt_column)}, index
        return

    if metadata_path.suffix == ".jsonl":
        with metadata_path.open("r", encoding="utf-8") as handle:
            for index, line in enumerate(handle):
                if line.strip():
                    yield json.loads(line), index
        return

    if metadata_path.suffix == ".json":
        with metadata_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if isinstance(payload, dict):
            for key in ("data", "samples", "items"):
                if isinstance(payload.get(key), list):
                    payload = payload[key]
                    break
        if not isinstance(payload, list):
            raise ValueError(f"JSON metadata must be a list, got {type(payload).__name__}: {metadata_path}")
        for index, row in enumerate(payload):
            yield row, index
        return

    csv.field_size_limit(sys.maxsize)
    with metadata_path.open("r", encoding="utf-8-sig", newline="") as handle:
        for index, row in enumerate(csv.DictReader(handle)):
            yield row, index


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


def resolve_video_path(metadata_path, raw_video, video_root):
    if raw_video is None or not str(raw_video).strip():
        return None

    video_path = Path(str(raw_video).strip())
    if video_path.is_absolute():
        candidates = [video_path]
    else:
        metadata_dir = metadata_path if metadata_path.is_dir() else metadata_path.parent
        candidates = [
            metadata_dir / video_path,
            metadata_dir / "video" / video_path.name,
        ]
        if video_root is not None:
            root = Path(video_root)
            candidates.extend([root / video_path, root / video_path.name])

    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return candidates[0]


def load_entries(args):
    entries = []
    for metadata in args.metadata:
        metadata_path = Path(metadata)
        for row, row_index in iter_metadata_rows(metadata_path, args):
            video_path = resolve_video_path(metadata_path, row.get(args.video_column, row.get("video")), args.video_root)
            if args.skip_missing and (video_path is None or not video_path.is_file()):
                continue
            prompt = row.get(args.prompt_column, row.get("prompt", row.get("caption", "")))
            frame_count = parse_optional_int(row.get("frame"))
            prefix, number = numeric_parts(video_path)
            entries.append(
                ClipEntry(
                    path=video_path,
                    prompt=str(prompt) if prompt is not None else "",
                    frame_count=frame_count,
                    metadata_path=metadata_path,
                    row_index=row_index,
                    prefix=prefix,
                    number=number,
                )
            )
    return entries


def parse_optional_int(value):
    if value is None or value == "":
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


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
    ensure_video_imports()
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


def format_source_paths(source_paths, limit=0):
    display_paths = source_paths
    suffix = ""
    if limit and len(source_paths) > limit:
        display_paths = source_paths[:limit]
        suffix = f" ... (+{len(source_paths) - limit} more)"
    return ", ".join(display_paths) + suffix


def sample_label(group_index, window_index):
    return f"group={group_index:06d} window={window_index:03d}"


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
            frame_iter = iter_video_frames(clip.path, args)
            for frame in frame_iter:
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


def latent_range_to_raw_window(latent_start, latent_end, ratio):
    if latent_end <= latent_start:
        raise ValueError(f"latent_end must be > latent_start, got {latent_start}, {latent_end}")
    if latent_start == 0:
        return 0, 1 + ratio * (latent_end - 1), 0
    return ratio * (latent_start - 1), 1 + ratio * (latent_end - 1), 1


def encode_video_latent(vae, video_cpu, args):
    ensure_torch()
    device = torch.device(args.device)
    vae_dtype = torch_dtype(args.vae_dtype)
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
            raw_start, raw_end, pseudo_prefix = latent_range_to_raw_window(
                halo_start,
                keep_end,
                args.temporal_compression_ratio,
            )
            drop_latents = pseudo_prefix + (keep_start - halo_start)
            video = prepare_video_chunk(video_cpu[:, raw_start:raw_end].contiguous())
            encoded = vae.encode([video])[0].cpu()
            keep = encoded[:, drop_latents : drop_latents + (keep_end - keep_start)].contiguous()
            if keep.shape[1] != keep_end - keep_start:
                raise RuntimeError(f"VAE chunk produced {keep.shape[1]} latent frames, expected {keep_end - keep_start} for keep range [{keep_start}, {keep_end}).")
            latent_parts.append(keep)
            del video, encoded, keep
            if device.type == "cuda":
                torch.cuda.empty_cache()
        return torch.cat(latent_parts, dim=1)


def encode_prompt_embed(text_encoder, prompt, args):
    ensure_torch()
    device = torch.device(args.text_device or args.device)
    with torch.inference_mode():
        contexts = text_encoder([prompt], device)
        context = contexts[0][: args.max_sequence_length].to(dtype=torch_dtype(args.save_dtype)).cpu()
        if context.shape[0] < args.max_sequence_length:
            pad = context.new_zeros(args.max_sequence_length - context.shape[0], context.shape[1])
            context = torch.cat([context, pad], dim=0)
        return context


def validate_args(args):
    expected_raw = 1 + args.temporal_compression_ratio * (args.latent_frames - 1)
    if args.raw_frames != expected_raw:
        raise ValueError(f"--raw-frames must be 1 + ratio * (latent_frames - 1) = {expected_raw}, got {args.raw_frames}.")
    if args.height % 16 != 0 or args.width % 16 != 0:
        raise ValueError(f"height/width must be divisible by 16, got {args.height}x{args.width}.")
    if args.rank < 0 or args.rank >= args.world_size:
        raise ValueError(f"rank must be in [0, world_size), got rank={args.rank}, world_size={args.world_size}.")


def count_group_frames(group, args):
    if all(clip.frame_count is not None for clip in group):
        return sum(clip.frame_count for clip in group)

    ensure_video_imports()
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


def load_models(args):
    ensure_torch()

    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)

    from lightx2v_train.model_zoo.native.wan.modules.t5 import T5EncoderModel
    from lightx2v_train.model_zoo.native.wan.modules.vae2_2 import Wan2_2_VAE

    model_dir = Path(args.model_dir)
    vae_checkpoint = args.vae_checkpoint or str(model_dir / "Wan2.2_VAE.pth")
    t5_checkpoint = args.t5_checkpoint or str(model_dir / "models_t5_umt5-xxl-enc-bf16.pth")
    tokenizer_path = args.tokenizer_path or str(model_dir / "google" / "umt5-xxl")

    print(f"[rank {args.rank}] loading VAE: {vae_checkpoint}", flush=True)
    vae = Wan2_2_VAE(
        vae_pth=vae_checkpoint,
        dtype=torch_dtype(args.vae_dtype),
        device=device,
    )

    text_device = torch.device(args.text_device or args.device)
    print(f"[rank {args.rank}] loading T5: {t5_checkpoint}", flush=True)
    text_encoder = T5EncoderModel(
        text_len=args.max_sequence_length,
        dtype=torch_dtype(args.t5_dtype),
        device=text_device,
        checkpoint_path=t5_checkpoint,
        tokenizer_path=tokenizer_path,
    )
    return vae, text_encoder


def save_cache(path, payload, overwrite):
    ensure_torch()
    if path.exists() and not overwrite:
        return False
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp_path)
    tmp_path.replace(path)
    return True


def main():
    global args
    args = normalize_runtime_args(parse_args())
    validate_args(args)

    entries = load_entries(args)
    groups = group_consecutive(entries)
    if args.max_groups is not None:
        groups = groups[: args.max_groups]
    local_groups = [(index, group) for index, group in enumerate(groups) if index % args.world_size == args.rank]

    print(
        f"[rank {args.rank}] clips={len(entries)} groups={len(groups)} local_groups={len(local_groups)} "
        f"raw_frames={args.raw_frames} latent_frames={args.latent_frames} "
        f"local_rank={args.local_rank} device={args.device} text_device={args.text_device or args.device} "
        f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '')}",
        flush=True,
    )

    if args.dry_run:
        total_windows = 0
        for group_index, group in local_groups:
            frame_count = count_group_frames(group, args)
            windows = frame_count // args.raw_frames
            total_windows += windows
            print(
                f"[rank {args.rank}] group={group_index:06d} clips={len(group)} frames={frame_count} windows={windows} first={group[0].path} last={group[-1].path}",
                flush=True,
            )
        print(f"[rank {args.rank}] dry_run_windows={total_windows}", flush=True)
        return

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / f"manifest_rank{args.rank:02d}.jsonl"

    vae, text_encoder = load_models(args)
    save_dtype = torch_dtype(args.save_dtype)
    saved = 0

    with manifest_path.open("a", encoding="utf-8") as manifest:
        for local_index, (group_index, group) in enumerate(local_groups):
            if args.log_every > 0 and local_index % args.log_every == 0:
                print(
                    f"[rank {args.rank}] processing group {local_index + 1}/{len(local_groups)} (global={group_index}, clips={len(group)})",
                    flush=True,
                )
            for window_index, video, prompt, source_paths in iter_group_windows(group, args):
                out_path = output_dir / f"cache_g{group_index:06d}_w{window_index:03d}.pt"
                label = sample_label(group_index, window_index)
                sources = format_source_paths(source_paths, args.log_source_limit)
                if out_path.exists() and not args.overwrite:
                    print(
                        f"[rank {args.rank}] skip existing {label} output={out_path} sources=[{sources}]",
                        flush=True,
                    )
                    del video
                    continue

                print(
                    f"[rank {args.rank}] start {label} raw_shape={tuple(video.shape)} output={out_path} sources=[{sources}]",
                    flush=True,
                )
                latent = encode_video_latent(vae, video, args)
                if latent.shape[1] != args.latent_frames:
                    raise RuntimeError(f"Expected {args.latent_frames} latent frames, got shape={tuple(latent.shape)}")
                prompt_embed = encode_prompt_embed(text_encoder, prompt, args)

                payload = {
                    "latent": latent.to(dtype=save_dtype).contiguous(),
                    "prompt_embed": prompt_embed.to(dtype=save_dtype).contiguous(),
                    "prompt": prompt,
                    "video_path": "|".join(source_paths),
                    "video_paths": source_paths,
                    "raw_num_frames": args.raw_frames,
                    "latent_num_frames": int(latent.shape[1]),
                    "height": args.height,
                    "width": args.width,
                }
                if save_cache(out_path, payload, args.overwrite):
                    saved += 1
                    print(
                        f"[rank {args.rank}] saved {label} output={out_path} "
                        f"latent_shape={tuple(payload['latent'].shape)} "
                        f"prompt_embed_shape={tuple(payload['prompt_embed'].shape)} "
                        f"prompt_chars={len(prompt)}",
                        flush=True,
                    )
                    manifest.write(
                        json.dumps(
                            {
                                "cache_path": str(out_path),
                                "group_index": group_index,
                                "window_index": window_index,
                                "video_paths": source_paths,
                                "prompt": prompt,
                                "latent_shape": list(payload["latent"].shape),
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    manifest.flush()

                del video, latent, prompt_embed, payload
                if args.max_samples is not None and saved >= args.max_samples:
                    print(f"[rank {args.rank}] reached max_samples={args.max_samples}", flush=True)
                    return

    print(f"[rank {args.rank}] saved={saved} output_dir={output_dir}", flush=True)


if __name__ == "__main__":
    main()
