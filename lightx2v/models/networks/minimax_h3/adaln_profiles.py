"""Model-specific timesteps and arithmetic for persistent H3 AdaLN caches."""

import json
from dataclasses import dataclass

import torch

from lightx2v.models.networks.minimax_h3.packing import CONDITION_AUDIO_TIMESTEP, KEYFRAME_NOISE_AUG
from lightx2v.models.schedulers.minimax_h3.scheduler import _make_schedule


@dataclass(frozen=True)
class AdaLNCachePlan:
    cache_name: str
    profiles: list[str]
    timesteps: dict[str, torch.Tensor]
    quantized_projection: bool = False
    checkpoint_layout: str = "native_h3"


def _scheduled_timesteps(config, profiles):
    infer_steps = int(config["infer_steps"])
    _, video = _make_schedule(infer_steps, float(config.get("video_flow_shift", 12.0)), "cpu")
    _, audio = _make_schedule(infer_steps, float(config.get("audio_flow_shift", 3.0)), "cpu")
    batches = {}
    for profile in profiles:
        for step, (video_t, audio_t) in enumerate(zip(video.tolist(), audio.tolist())):
            values = [video_t, audio_t]
            if profile in {"conditioned", "ref2av_video", "ref2av_video_audio"}:
                values.append(max(video_t, KEYFRAME_NOISE_AUG))
            if profile in {"ref2av_video_audio", "taomate_base10_continue"}:
                values.append(CONDITION_AUDIO_TIMESTEP)
            batches[f"{profile}_step_{step:03d}"] = torch.unique(torch.tensor(values, dtype=torch.float32), sorted=True)
    return batches


def cache_plan(config) -> AdaLNCachePlan:
    """Select the model's timestep batches without loading model weights."""
    model = config["model_cls"]
    if model == "minimax_h3_image":
        from lightx2v.models.schedulers.minimax_h3_image.scheduler import REFERENCE_TIMESTEP

        with open(config["pdd_config_path"], encoding="utf-8") as handle:
            pdd_config = json.load(handle)
        if int(config["infer_steps"]) != 6 or pdd_config["num_inference_steps"] != 6:
            raise ValueError("MiniMax-H3-Image PDD requires exactly six inference steps")
        sigmas = pdd_config["sigmas"][:-1]
        # Match the scheduler's Python subtraction before its FP32 tensor cast.
        timesteps = torch.tensor([1.0 - sigma for sigma in sigmas], dtype=torch.float32)
        profiles = ["image_t2i", "image_i2i"]
        batches = {}
        for profile in profiles:
            for step, timestep in enumerate(timesteps.tolist()):
                values = [timestep, REFERENCE_TIMESTEP] if profile == "image_i2i" else [timestep]
                batches[f"{profile}_step_{step:03d}"] = torch.unique(torch.tensor(values, dtype=torch.float32), sorted=True)
        # Round only the directory label; manifest entries retain exact timesteps.
        label = "_".join(f"{sigma:#.4g}" for sigma in sigmas)
        return AdaLNCachePlan(cache_name=f"image_06steps_{label}", profiles=profiles, timesteps=batches, checkpoint_layout="minimax_h3_image")

    suffix = f"{int(config['infer_steps']):02d}steps_shift_{float(config.get('video_flow_shift', 12.0))}_{float(config.get('audio_flow_shift', 3.0))}"
    if model == "minimax_h3_causal":
        from lightx2v.models.schedulers.minimax_h3_causal.clean_scheduler import build_renoise_schedule

        # Reuse the online CUDA schedule, including its exact FP32 rounding.
        video_sigmas, audio_sigmas = build_renoise_schedule(config)
        video = (1 - video_sigmas).cpu().tolist()
        audio = (1 - audio_sigmas).cpu().tolist()
        task = config["task"]
        generate_audio = task == "refa2av"
        batches = {
            f"{task}_clean": torch.tensor([1.0], dtype=torch.float32),
            f"{task}_reference": torch.tensor([KEYFRAME_NOISE_AUG, 1.0] if generate_audio else [KEYFRAME_NOISE_AUG], dtype=torch.float32),
        }
        for step in range(config["infer_steps"]):
            values = [video[step], audio[step] if generate_audio else 1.0]
            batches[f"{task}_media_step_{step:03d}"] = torch.unique(torch.tensor(values, dtype=torch.float32), sorted=True)
        return AdaLNCachePlan(cache_name=f"{task}_{suffix}", profiles=[task], timesteps=batches)

    if model == "taomate_h3":
        profile = config.get("adaln_cache_profile")
        if profile == "taomate_base10":
            if int(config["infer_steps"]) != 9:
                raise ValueError("TaoMate Base10 AdaLN cache requires nine denoising steps")
            profiles = ["taomate_base10_first", "taomate_base10_continue"]
            batches = _scheduled_timesteps(config, profiles)
        elif profile is None:
            from lightx2v.models.schedulers.taomate_h3.scheduler import distilled_sigmas

            profiles = ["taomate_stage3"]
            video = 1.0 - distilled_sigmas(float(config.get("video_flow_shift", 12.0)))[:-1]
            audio = 1.0 - distilled_sigmas(float(config.get("audio_flow_shift", 3.0)))[:-1]
            batches = {f"taomate_stage3_step_{step:03d}": torch.unique(torch.stack((v, a)), sorted=True) for step, (v, a) in enumerate(zip(video, audio))}
            batches["taomate_stage3_clean"] = torch.tensor([1.0], dtype=torch.float32)
        else:
            raise ValueError(f"Unknown TaoMate AdaLN profile: {profile!r}")
        return AdaLNCachePlan(
            cache_name=f"{profile or config['model_variant']}_{suffix}",
            profiles=profiles,
            timesteps=batches,
            quantized_projection=bool(config.get("dit_quantized", False)),
        )

    if model == "streamrender_h3":
        from lightx2v.models.schedulers.streamrender_h3.scheduler import _shifted_timesteps

        if int(config["infer_steps"]) != 2:
            raise ValueError("StreamRender AdaLN cache requires two denoising steps")
        # Match the online FP32 subtraction, including the clean source timestep.
        video = 1.0 - _shifted_timesteps(6)
        audio = 1.0 - _shifted_timesteps(1.5)
        clean = 1.0 - torch.tensor(0.001, dtype=torch.float32)
        batches = {f"streamrender_step_{step:03d}": torch.unique(torch.stack((v, a, clean)), sorted=True) for step, (v, a) in enumerate(zip(video, audio))}
        quantized = bool(config.get("dit_quantized", False))
        return AdaLNCachePlan(
            cache_name=f"{config['task']}_{suffix}",
            profiles=["streamrender"],
            timesteps=batches,
            quantized_projection=quantized,
            checkpoint_layout="native_h3" if quantized else "streamrender_release",
        )

    variant = config["model_variant"]
    if variant == "ref2av":
        # The reference transformer must stay separate from base-model tasks.
        profiles = ["ref2av_video", "ref2av_video_audio"]
    elif variant == "fl2av":
        profiles = ["t2av", "conditioned"]
        if not config.get("text_encoder_disk_streaming", False):
            profiles.append("ref2av_video_audio")
    else:
        raise ValueError(f"No persistent AdaLN cache profile is available for model_variant: {variant!r}")
    # MiniMax caches use original-weight modulation even with FP8 DiT weights.
    return AdaLNCachePlan(cache_name=f"{config.get('adaln_cache_profile') or variant}_{suffix}", profiles=profiles, timesteps=_scheduled_timesteps(config, profiles))
