"""Generate T2AV from a Dense2MoE DCP snapshot or its frozen dense teacher."""

import argparse
import gc
import json
import time
from pathlib import Path

import torch
import torch.distributed.checkpoint as dcp
from diffusers import AutoencoderKLMiniMaxH3, AutoencoderKLMiniMaxH3Audio, MiniMaxH3Scheduler
from diffusers.models.attention_dispatch import attention_backend
from diffusers.modular_pipelines.minimax_h3.packing import unpack_audio_tokens, unpatchify_video_tokens
from diffusers.utils.export_utils import encode_video

from lightx2v_train.model_zoo.minimax_h3.condition_encoder import MiniMaxH3ConditionEncoder
from lightx2v_train.model_zoo.native.minimax_h3 import audio_latent_num_frames, video_latent_num_frames
from lightx2v_train.model_zoo.native.minimax_h3.packing import patchify_video_latents

from .runtime import AVSample, load_model


def load_checkpoint(model, checkpoint):
    reader = dcp.FileSystemReader(checkpoint)
    keys = {name.removeprefix("model.") for name in reader.read_metadata().state_dict_metadata if name.startswith("model.")}
    parameters = dict(model.named_parameters())
    state = {"model": {name: parameters[name].detach() for name in sorted(keys)}}
    dcp.load(state, storage_reader=reader, no_dist=True)
    print(json.dumps({"event": "checkpoint_loaded", "parameters": len(keys), "checkpoint": str(checkpoint)}), flush=True)


@torch.no_grad()
def decode(config, latents, geometry, output):
    device = latents["video"].device
    vae = AutoencoderKLMiniMaxH3.from_pretrained(config["model_path"], subfolder="vae", torch_dtype=torch.float32, local_files_only=True).eval().to(device)
    video = unpatchify_video_tokens(latents["video"][0], video_latent_num_frames(geometry["num_frames"]), geometry["height"] // 16, geometry["width"] // 16, 24, (1, 2, 2))
    mean = video.new_tensor(vae.config.latents_mean).view(1, -1, 1, 1, 1)
    std = video.new_tensor(vae.config.latents_std).view(1, -1, 1, 1, 1)
    with attention_backend("native"), torch.autocast("cuda", dtype=torch.float16):
        video = vae.decode(video * std + mean, return_dict=False)[0]
    pixels_mean = video.new_tensor((0.485, 0.456, 0.406), dtype=torch.float32).view(1, 3, 1, 1, 1)
    pixels_std = video.new_tensor((0.229, 0.224, 0.225), dtype=torch.float32).view(1, 3, 1, 1, 1)
    if not torch.isfinite(video).all():
        raise FloatingPointError("Nonfinite decoded video")
    # Export integer frames directly: the numpy path allocates full-sized
    # float zeros/ones arrays to detect the range, which is costly for video.
    frames = (video.float() * pixels_std + pixels_mean).clamp_(0, 1)[0].permute(1, 2, 3, 0).mul(255).round().to(torch.uint8).contiguous().cpu()
    del vae, video
    gc.collect()
    torch.cuda.empty_cache()
    vae = AutoencoderKLMiniMaxH3Audio.from_pretrained(config["model_path"], subfolder="audio_vae", torch_dtype=torch.float32, local_files_only=True).eval().to(device)
    audio = unpack_audio_tokens(latents["audio"][0], audio_latent_num_frames(geometry["num_frames"]))
    mean = audio.new_tensor(vae.config.latents_mean).view(1, -1, 1)
    std = audio.new_tensor(vae.config.latents_std).view(1, -1, 1)
    with attention_backend("native"):
        audio = vae.decode(audio * std + mean, return_dict=False)[0][:, 0].float().cpu()
    if not torch.isfinite(audio).all():
        raise FloatingPointError("Nonfinite decoded audio")
    encode_video(frames, fps=24, output_path=str(output), audio=audio, audio_sample_rate=32000)
    return {"frames": len(frames), "audio_samples": audio.shape[-1], "audio_rms": audio.square().mean().sqrt().item()}


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--partition", type=Path, required=True)
    parser.add_argument("--prompt-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    variants = parser.add_mutually_exclusive_group()
    variants.add_argument("--dense", action="store_true")
    variants.add_argument("--untrained", action="store_true", help="Use partitioned FFN initialization without loading trained weights")
    parser.add_argument("--decode-only", action="store_true", help="Resume decoding the saved .latents.pt without repeating denoising")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--height", type=int, default=768)
    parser.add_argument("--width", type=int, default=1344)
    parser.add_argument("--num-frames", type=int, default=124)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    metadata = json.loads((args.checkpoint / "progress.json").read_text())
    config = metadata["config"]
    variant = "dense" if args.dense else metadata["stage"] + "_init" if args.untrained else metadata["stage"]
    training_step = 0 if args.dense or args.untrained else metadata["step"]
    geometry = {"height": args.height, "width": args.width, "num_frames": args.num_frames}
    started = time.perf_counter()
    device = torch.device("cuda:0")
    if args.decode_only:
        latents = torch.load(args.output.with_suffix(".latents.pt"), map_location=device, weights_only=True)
        media = decode(config, latents, geometry, args.output)
        result = {
            "output": str(args.output),
            "checkpoint": str(args.checkpoint) if not (args.dense or args.untrained) else None,
            "recipe_checkpoint": str(args.checkpoint),
            "variant": variant,
            "training_step": training_step,
            "seed": args.seed,
            "schedule_points": args.steps,
            "flow_shift": config["flow_shift"],
            **geometry,
            **media,
            "decode_seconds": time.perf_counter() - started,
            "resumed_decode": True,
        }
        args.output.with_suffix(".json").write_text(json.dumps(result, indent=2))
        print(json.dumps(result), flush=True)
        return
    print(json.dumps({"event": "encode_prompt", "variant": variant, "step": training_step}), flush=True)
    encoder = MiniMaxH3ConditionEncoder(config["condition_model_path"], device=device, dtype=torch.bfloat16, cpu_offload=False)
    sample = AVSample(encoder.encode(args.prompt_file.read_text()), geometry, device)
    del encoder
    gc.collect()
    torch.cuda.empty_cache()
    print(json.dumps({"event": "load_dit"}), flush=True)
    permutations = torch.load(args.partition, map_location="cpu", weights_only=True)
    shared = torch.load(Path(config["output_dir"]) / "shared.pt", map_location="cpu", weights_only=True) if metadata["stage"] == "moe" and not (args.dense or args.untrained) else None
    model = load_model(config, "dense" if args.dense else metadata["stage"], permutations, shared_state=shared, sharded=False).eval().requires_grad_(False)
    del shared
    if not (args.dense or args.untrained):
        load_checkpoint(model, args.checkpoint)
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    video = torch.randn((1, 24, video_latent_num_frames(args.num_frames), args.height // 16, args.width // 16), generator=generator).to(device)
    latents = {"video": patchify_video_latents(video).unsqueeze(0), "audio": torch.randn(sample.shapes["audio"], generator=generator).to(device)}
    schedulers = {name: MiniMaxH3Scheduler(shift=shift) for name, shift in config["flow_shift"].items()}
    for scheduler in schedulers.values():
        scheduler.set_timesteps(args.steps)
    torch.cuda.synchronize()
    denoise_start = time.perf_counter()
    for step in range(len(schedulers["video"].timesteps)):
        sigmas = {name: scheduler.sigmas[step] for name, scheduler in schedulers.items()}
        prediction = sample.forward(model, latents, sigmas)
        latents = {name: scheduler.step(prediction[name].float(), scheduler.timesteps[step], latents[name], return_dict=False)[0] for name, scheduler in schedulers.items()}
        torch.cuda.synchronize()
        print(json.dumps({"event": "denoise", "step": step + 1, "total": len(schedulers["video"].timesteps)}), flush=True)
    denoise_seconds = time.perf_counter() - denoise_start
    torch.save({name: value.cpu() for name, value in latents.items()}, args.output.with_suffix(".latents.pt"))
    del model, prediction, sample
    gc.collect()
    torch.cuda.empty_cache()
    print(json.dumps({"event": "decode", "denoise_seconds": denoise_seconds}), flush=True)
    media = decode(config, latents, geometry, args.output)
    result = {
        "output": str(args.output),
        "checkpoint": str(args.checkpoint) if not (args.dense or args.untrained) else None,
        "recipe_checkpoint": str(args.checkpoint),
        "variant": variant,
        "training_step": training_step,
        "seed": args.seed,
        "schedule_points": args.steps,
        "model_evaluations": len(schedulers["video"].timesteps),
        "flow_shift": config["flow_shift"],
        **geometry,
        **media,
        "denoise_seconds": denoise_seconds,
        "total_seconds": time.perf_counter() - started,
    }
    args.output.with_suffix(".json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
