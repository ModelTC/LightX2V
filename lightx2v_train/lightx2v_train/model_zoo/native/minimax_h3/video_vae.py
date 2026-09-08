"""Loader and pixel conventions for the released MiniMax-H3 video VAE."""

from pathlib import Path

import torch


PIXEL_MEAN = (0.485, 0.456, 0.406)
PIXEL_STD = (0.229, 0.224, 0.225)


def resolve_video_vae_dir(model_path: str | Path) -> Path:
    root = Path(model_path).expanduser().resolve()
    if (root / "vae" / "config.json").is_file():
        return root / "vae"
    if (root / "video_vae" / "config.json").is_file():
        return root / "video_vae"
    if (root / "config.json").is_file():
        return root
    raise FileNotFoundError(f"MiniMax-H3 video VAE config not found below {root}.")


def load_minimax_h3_video_vae(
    model_path: str | Path,
    *,
    torch_dtype: torch.dtype = torch.float32,
    local_files_only: bool = True,
):
    try:
        from diffusers import AutoencoderKLMiniMaxH3
    except (ImportError, AttributeError) as error:
        raise ImportError(
            "MiniMax-H3 VAE distillation requires a Diffusers build containing AutoencoderKLMiniMaxH3."
        ) from error

    vae_dir = resolve_video_vae_dir(model_path)
    return AutoencoderKLMiniMaxH3.from_pretrained(
        str(vae_dir),
        torch_dtype=torch_dtype,
        local_files_only=local_files_only,
        low_cpu_mem_usage=True,
    )


def imagenet_preprocess(video: torch.Tensor) -> torch.Tensor:
    mean = video.new_tensor(PIXEL_MEAN).view(1, 3, 1, 1, 1)
    std = video.new_tensor(PIXEL_STD).view(1, 3, 1, 1, 1)
    return (video - mean) / std


def imagenet_postprocess(video: torch.Tensor) -> torch.Tensor:
    mean = video.new_tensor(PIXEL_MEAN).view(1, 3, 1, 1, 1)
    std = video.new_tensor(PIXEL_STD).view(1, 3, 1, 1, 1)
    return (video * std + mean).clamp(0.0, 1.0)


def normalize_video_latents(vae, latents: torch.Tensor) -> torch.Tensor:
    mean = latents.new_tensor(vae.config.latents_mean).view(1, -1, 1, 1, 1)
    std = latents.new_tensor(vae.config.latents_std).view(1, -1, 1, 1, 1)
    return (latents - mean) / std
