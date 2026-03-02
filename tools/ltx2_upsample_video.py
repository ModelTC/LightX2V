import argparse
import json
import re
from pathlib import Path

import av
import torch
from einops import rearrange
from safetensors import safe_open

from lightx2v.models.input_encoders.hf.ltx2.model import LTX2TextEncoder
from lightx2v.models.networks.ltx2.model import LTX2Model
from lightx2v.models.schedulers.ltx2.scheduler import LTX2Scheduler
from lightx2v.models.video_encoders.hf.ltx2.model import LTX2Upsampler, LTX2VideoVAE
from lightx2v.utils.ltx2_media_io import encode_video
from lightx2v.utils.ltx2_utils import SafetensorsModelStateDictLoader


DTYPE_MAP = {
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
    "fp32": torch.float32,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Upscale a low-resolution video using LTX2 latent upsampler (x2 spatial)."
    )
    parser.add_argument("--input", required=True, help="Input low-resolution video path.")
    parser.add_argument("--output", required=True, help="Output upscaled video path.")
    parser.add_argument("--ltx2_vae_ckpt", required=True, help="LTX2 checkpoint path (contains video VAE).")
    parser.add_argument("--ltx2_upsampler_ckpt", required=True, help="LTX2 spatial upsampler checkpoint path.")
    parser.add_argument("--device", default="cuda", help="Torch device, e.g. cuda or cpu.")
    parser.add_argument(
        "--dtype",
        default="bf16",
        choices=list(DTYPE_MAP.keys()),
        help="Model dtype.",
    )
    parser.add_argument(
        "--cpu_offload",
        action="store_true",
        help="Enable CPU offload when loading LTX2 modules.",
    )
    parser.add_argument(
        "--disable_refine",
        action="store_true",
        help="Disable LTX2 high-resolution denoising stage after latent upsampling.",
    )
    parser.add_argument("--prompt", default="", help="Prompt used by LTX2 refine stage.")
    parser.add_argument("--negative_prompt", default="", help="Negative prompt used by LTX2 refine stage.")
    parser.add_argument("--seed", type=int, default=42, help="Seed for LTX2 refine stage.")
    parser.add_argument(
        "--ltx2_transformer_ckpt",
        default=None,
        help="LTX2 transformer checkpoint used for high-resolution denoising. Defaults to --ltx2_vae_ckpt.",
    )
    parser.add_argument(
        "--ltx2_model_path",
        default=None,
        help="LTX2 model root for Gemma/tokenizer lookup. Defaults to parent directory of transformer ckpt.",
    )
    parser.add_argument(
        "--upsample_sigmas",
        default="0.909375,0.725,0.421875,0.0",
        help="Comma-separated sigma schedule for LTX2 refine stage.",
    )
    parser.add_argument("--sample_shift_max", type=float, default=2.05, help="LTX2 scheduler max shift.")
    parser.add_argument("--sample_shift_base", type=float, default=0.95, help="LTX2 scheduler base shift.")
    parser.add_argument("--sample_guide_scale", type=float, default=1.0, help="CFG guidance scale for refine stage.")
    parser.add_argument("--enable_cfg", action="store_true", help="Enable CFG in LTX2 refine stage.")
    parser.add_argument(
        "--audio_sampling_rate",
        type=int,
        default=24000,
        help="Audio sampling rate used to build LTX2 joint AV latent shape.",
    )
    parser.add_argument("--audio_hop_length", type=int, default=160, help="Audio hop length for LTX2 scheduler.")
    parser.add_argument("--audio_scale_factor", type=int, default=4, help="Audio latent downsample factor for LTX2 scheduler.")
    parser.add_argument("--audio_mel_bins", type=int, default=16, help="Audio mel bins for LTX2 scheduler.")
    parser.add_argument("--fps_override", type=int, default=None, help="Override FPS for LTX2 refine stage.")
    parser.add_argument("--double_precision_rope", dest="double_precision_rope", action="store_true", help="Enable double precision RoPE in LTX2 refine stage.")
    parser.add_argument("--no_double_precision_rope", dest="double_precision_rope", action="store_false", help="Disable double precision RoPE in LTX2 refine stage.")
    parser.set_defaults(double_precision_rope=True)
    return parser.parse_args()


def get_video_fps(video_path: str) -> int:
    container = av.open(video_path)
    try:
        video_stream = next(s for s in container.streams if s.type == "video")
        if video_stream.average_rate is not None:
            return int(round(float(video_stream.average_rate)))
        if video_stream.base_rate is not None:
            return int(round(float(video_stream.base_rate)))
        return 24
    finally:
        container.close()


def load_video_frames(video_path: str) -> torch.Tensor:
    container = av.open(video_path)
    frames = []
    try:
        video_stream = next(s for s in container.streams if s.type == "video")
        for frame in container.decode(video_stream):
            frame_rgb = frame.to_rgb().to_ndarray()
            frames.append(torch.from_numpy(frame_rgb))
    finally:
        container.close()

    if not frames:
        raise ValueError(f"No video frames found in: {video_path}")

    return torch.stack(frames, dim=0)


def validate_video_shape(video: torch.Tensor) -> None:
    frame_count, height, width, _ = video.shape
    if (frame_count - 1) % 8 != 0:
        raise ValueError(
            f"Invalid frame count {frame_count}. LTX2 VAE requires 1 + 8*k frames."
        )
    if height % 32 != 0 or width % 32 != 0:
        raise ValueError(
            f"Invalid resolution {height}x{width}. LTX2 VAE requires height/width to be multiples of 32."
        )


def parse_sigma_list(sigmas: str) -> list[float]:
    values = [float(item.strip()) for item in sigmas.split(",") if item.strip()]
    if len(values) < 2:
        raise ValueError("--upsample_sigmas must include at least 2 values.")
    return values


def load_json_config_from_candidates(model_path: str, transformer_ckpt: str) -> dict:
    cfg = {}
    ckpt_path = Path(transformer_ckpt)
    candidates = [
        Path(model_path) / "config.json",
        Path(model_path) / "transformer" / "config.json",
    ]
    if ckpt_path.is_file():
        candidates.extend(
            [
                ckpt_path.parent / "config.json",
                ckpt_path.parent.parent / "config.json",
            ]
        )
    elif ckpt_path.is_dir():
        candidates.extend(
            [
                ckpt_path / "config.json",
                ckpt_path / "transformer" / "config.json",
            ]
        )

    for path in candidates:
        if path.exists():
            with path.open("r", encoding="utf-8") as f:
                cfg.update(json.load(f))
    return cfg


def infer_num_layers_from_ckpt(transformer_ckpt: str) -> int | None:
    ckpt_path = Path(transformer_ckpt)
    if not ckpt_path.is_file():
        return None
    pattern = re.compile(r"model\.diffusion_model\.transformer_blocks\.(\d+)\.")
    indices = set()
    with safe_open(str(ckpt_path), framework="pt") as f:
        for key in f.keys():
            match = pattern.search(key)
            if match:
                indices.add(int(match.group(1)))
    if not indices:
        return None
    return max(indices) + 1


def build_refine_config(
    transformer_ckpt: str,
    model_path: str,
    fps: int,
    frame_count: int,
    height: int,
    width: int,
    args: argparse.Namespace,
) -> dict:
    config = load_json_config_from_candidates(model_path=model_path, transformer_ckpt=transformer_ckpt)
    metadata_loader = SafetensorsModelStateDictLoader()
    try:
        config.update(metadata_loader.metadata(transformer_ckpt))
    except Exception:
        # Some checkpoints may not carry full/valid metadata config; keep json config fallback.
        pass
    config.update(
        {
            "model_cls": "ltx2",
            "task": "t2av",
            "model_path": model_path,
            "dit_original_ckpt": transformer_ckpt,
            "cpu_offload": args.cpu_offload,
            "enable_cfg": args.enable_cfg,
            "cfg_parallel": False,
            "seq_parallel": False,
            "tensor_parallel": False,
            "parallel": False,
            "sample_guide_scale": args.sample_guide_scale,
            "sample_shift": [args.sample_shift_max, args.sample_shift_base],
            "double_precision_rope": args.double_precision_rope,
            "fps": fps,
            "target_video_length": frame_count,
            "target_height": height,
            "target_width": width,
            "audio_sampling_rate": args.audio_sampling_rate,
            "audio_hop_length": args.audio_hop_length,
            "audio_scale_factor": args.audio_scale_factor,
            "audio_mel_bins": args.audio_mel_bins,
        }
    )
    if "num_layers" not in config:
        inferred = infer_num_layers_from_ckpt(transformer_ckpt)
        if inferred is not None:
            config["num_layers"] = inferred
        else:
            raise KeyError("num_layers is missing in refine config; please provide a valid LTX2 config/model path.")
    # Keep a safe default to avoid attention module key errors on partial configs.
    config.setdefault("attn_type", "sage_attn2")
    return config


def infer_refined_latent(
    upscaled_latent: torch.Tensor,
    transformer_ckpt: str,
    model_path: str,
    prompt: str,
    negative_prompt: str,
    fps: int,
    args: argparse.Namespace,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    c, f, h, w = upscaled_latent.shape
    sigma_values = parse_sigma_list(args.upsample_sigmas)
    config = build_refine_config(
        transformer_ckpt=transformer_ckpt,
        model_path=model_path,
        fps=fps,
        frame_count=f,
        height=h * 32,
        width=w * 32,
        args=args,
    )
    # BaseScheduler requires infer_steps during __init__.
    config["infer_steps"] = len(sigma_values) - 1

    scheduler = LTX2Scheduler(config)
    model = LTX2Model(model_path=transformer_ckpt, config=config, device=device)
    model.set_scheduler(scheduler)

    text_encoder = LTX2TextEncoder(
        checkpoint_path=transformer_ckpt,
        gemma_root=model_path,
        device=device,
        dtype=dtype,
        cpu_offload=False,
    )
    v_context_p, a_context_p, v_context_n, a_context_n = text_encoder.infer(
        prompt=prompt,
        negative_prompt=negative_prompt,
    )
    inputs = {
        "text_encoder_output": {
            "v_context_p": v_context_p,
            "a_context_p": a_context_p,
            "v_context_n": v_context_n,
            "a_context_n": a_context_n,
        },
        "image_encoder_output": None,
    }

    sigma_tensor = torch.tensor(sigma_values, dtype=torch.float32, device=device)
    scheduler.reset_sigmas(sigma_tensor)
    duration = float(f) / float(fps)
    latents_per_second = float(config["audio_sampling_rate"]) / float(config["audio_hop_length"]) / float(config["audio_scale_factor"])
    audio_frames = round(duration * latents_per_second)
    audio_latent_shape = (8, audio_frames, config["audio_mel_bins"])
    scheduler.prepare(
        seed=args.seed,
        video_latent_shape=(c, f, h, w),
        audio_latent_shape=audio_latent_shape,
        initial_video_latent=upscaled_latent,
        noise_scale=sigma_values[0],
    )

    for step_index in range(scheduler.infer_steps):
        scheduler.step_pre(step_index=step_index)
        model.infer(inputs)
        scheduler.step_post()

    refined_latent = scheduler.video_latent_state.latent
    scheduler.clear()
    return refined_latent


def main() -> None:
    args = parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        raise FileNotFoundError(f"Input video not found: {input_path}")

    device = torch.device(args.device)
    dtype = DTYPE_MAP[args.dtype]
    if device.type == "cpu" and dtype != torch.float32:
        print("Warning: forcing --dtype fp32 on CPU for compatibility.")
        dtype = torch.float32

    fps = args.fps_override if args.fps_override is not None else get_video_fps(str(input_path))
    video = load_video_frames(str(input_path))
    validate_video_shape(video)

    # [F, H, W, C] uint8 -> [1, C, F, H, W] normalized to [-1, 1]
    video = rearrange(video, "f h w c -> 1 c f h w").to(torch.float32)
    video = (video / 127.5 - 1.0).to(device=device, dtype=dtype)

    video_vae = LTX2VideoVAE(
        checkpoint_path=args.ltx2_vae_ckpt,
        device=device,
        dtype=dtype,
        load_encoder=True,
        cpu_offload=args.cpu_offload,
    )
    upsampler = LTX2Upsampler(
        checkpoint_path=args.ltx2_upsampler_ckpt,
        device=device,
        dtype=dtype,
        cpu_offload=args.cpu_offload,
    )

    with torch.no_grad():
        latent = video_vae.encode(video)
        if video_vae.encoder is None:
            raise RuntimeError("LTX2 video encoder is not loaded, cannot run latent upsampling.")
        # Align with LTX2 runner: upsample accepts latent and uses encoder stats internally.
        upscaled_latent = upsampler.upsample(latent, video_vae.encoder).squeeze(0)
        if args.disable_refine:
            final_latent = upscaled_latent
        else:
            transformer_ckpt = args.ltx2_transformer_ckpt or args.ltx2_vae_ckpt
            ltx2_model_path = args.ltx2_model_path or str(Path(transformer_ckpt).parent)
            final_latent = infer_refined_latent(
                upscaled_latent=upscaled_latent,
                transformer_ckpt=transformer_ckpt,
                model_path=ltx2_model_path,
                prompt=args.prompt,
                negative_prompt=args.negative_prompt,
                fps=fps,
                args=args,
                device=device,
                dtype=dtype,
            )

        decoded_video = video_vae.decode(final_latent.unsqueeze(0))
        encode_video(
            video=decoded_video,
            fps=fps,
            audio=None,
            audio_sample_rate=None,
            output_path=str(output_path),
            video_chunks_number=1,
        )


if __name__ == "__main__":
    main()
