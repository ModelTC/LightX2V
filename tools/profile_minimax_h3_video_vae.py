#!/usr/bin/env python3
"""Standalone MiniMax-H3 video VAE decode benchmark."""

import argparse
import json
import os
import time
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default="/llm/models/MiniMax-H3")
    parser.add_argument(
        "--config-json",
        default="configs/platforms/intel_xpu/minimax_h3_fl2v_turbo_sla_4step.json",
    )
    parser.add_argument("--frames", type=int, help="Output frame count; defaults to target_video_length")
    parser.add_argument("--height", type=int, help="Output height; defaults to target_height")
    parser.add_argument("--width", type=int, help="Output width; defaults to target_width")
    parser.add_argument("--tile-height", type=int, help="Override VAE decode tile height in pixels")
    parser.add_argument("--tile-width", type=int, help="Override VAE decode tile width in pixels")
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu-offload", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--attn-type", choices=("torch_sdpa", "minimax_h3_vae_cute"))
    parser.add_argument("--fused-qkv", action=argparse.BooleanOptionalAction, default=None)
    return parser.parse_args()


def gib(value):
    return value / 1024**3


def main():
    args = parse_args()
    os.environ.setdefault("PLATFORM", "intel_xpu")
    os.environ.setdefault("DTYPE", "BF16")
    os.environ.setdefault("SENSITIVE_LAYER_DTYPE", "BF16")

    # Imports must follow PLATFORM selection.
    import torch

    from lightx2v.models.networks.minimax_h3.packing import align_num_frames, video_latent_num_frames
    from lightx2v.models.video_encoders.hf.minimax_h3 import MiniMaxH3VideoVAE
    from lightx2v.utils.envs import DTYPE_MAP
    from lightx2v_platform.base.global_var import AI_DEVICE

    config_path = Path(args.config_json)
    with config_path.open(encoding="utf-8") as handle:
        config = json.load(handle)

    frames = align_num_frames(args.frames or int(config["target_video_length"]))
    height = args.height or int(config["target_height"])
    width = args.width or int(config["target_width"])
    scale = int(config.get("vae_spatial_scale_factor", 16))
    if height % scale or width % scale:
        raise ValueError(f"height and width must be divisible by {scale}: got {height}x{width}")
    if (args.tile_height is None) != (args.tile_width is None):
        raise ValueError("--tile-height and --tile-width must be specified together")
    if args.iterations < 1 or args.warmup < 0:
        raise ValueError("--iterations must be positive and --warmup must be non-negative")

    latent_shape = (1, int(config.get("in_channels", 24)), video_latent_num_frames(frames), height // scale, width // scale)
    cpu_offload = config.get("vae_cpu_offload", config.get("cpu_offload", False)) if args.cpu_offload is None else args.cpu_offload
    sensitive_dtype = DTYPE_MAP[config.get("vae_sensitive_layer_dtype", "fp32")]
    device = getattr(torch, AI_DEVICE)

    print(f"Config: {config_path}")
    print(f"Device: {AI_DEVICE}; output: {frames}x{height}x{width}; latent: {latent_shape}")
    print(f"cpu_offload={cpu_offload}; sensitive_dtype={sensitive_dtype}; warmup={args.warmup}; iterations={args.iterations}")
    attn_type = args.attn_type or config.get("vae_attn_type", "torch_sdpa")
    print(f"attn_type={attn_type}; fused_qkv={args.fused_qkv}")

    load_start = time.perf_counter()
    vae = MiniMaxH3VideoVAE.from_pretrained(
        args.model_path,
        device=AI_DEVICE,
        cpu_offload=cpu_offload,
        sensitive_layer_dtype=sensitive_dtype,
        use_compile=config.get("vae_use_compile", False),
        attn_type=attn_type,
        pack_qkv=args.fused_qkv,
    )
    if args.tile_height is not None:
        vae.set_decode_tile_shape(args.tile_height, args.tile_width)
    else:
        tile_shapes = config.get("vae_decode_tile_shape", {})
        tile_shape = tile_shapes.get(f"{height}x{width}")
        if tile_shape:
            vae.set_decode_tile_shape(*tile_shape)
    device.synchronize()
    print(f"VAE load: {time.perf_counter() - load_start:.3f} s")
    print(f"Decode tile: {vae.decode_tile_sample_min_height}x{vae.decode_tile_sample_min_width}")

    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    latents = torch.randn(latent_shape, generator=generator, dtype=torch.float32)

    def decode_once():
        output = vae.decode(latents, return_cpu=True)
        device.synchronize()
        return output

    for index in range(args.warmup):
        output = decode_once()
        print(f"Warmup {index + 1}/{args.warmup}: output={tuple(output.shape)}")
        del output

    reset_peak = getattr(device, "reset_peak_memory_stats", None)
    if reset_peak is not None:
        reset_peak()
    times = []
    for index in range(args.iterations):
        start = time.perf_counter()
        output = decode_once()
        elapsed = time.perf_counter() - start
        times.append(elapsed)
        print(f"Decode {index + 1}/{args.iterations}: {elapsed:.3f} s; output={tuple(output.shape)}")
        del output

    print(f"Decode summary: min={min(times):.3f} s; mean={sum(times) / len(times):.3f} s; max={max(times):.3f} s")
    max_allocated = getattr(device, "max_memory_allocated", None)
    max_reserved = getattr(device, "max_memory_reserved", None)
    if max_allocated is not None:
        message = f"XPU peak allocated: {gib(max_allocated()):.3f} GiB"
        if max_reserved is not None:
            message += f"; peak reserved: {gib(max_reserved()):.3f} GiB"
        print(message)


if __name__ == "__main__":
    main()
