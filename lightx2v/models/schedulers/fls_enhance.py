"""Fovea Latent Sharpening (FLS) post-step enhancement for diffusion schedulers."""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Literal

import torch
import torch.nn.functional as F
from loguru import logger

LayoutType = Literal["seq", "spatial", "video", "video_4d"]


@dataclass
class FLSState:
    previous_pred: torch.Tensor | None = None
    momentum_mask: torch.Tensor | None = None


@dataclass
class FLSConfig:
    enable: bool = False
    fovea_strength: float = 10.0
    sharpness: float = 3.0
    mask_inertia: float = 0.99
    warmup_ratio: float = 0.10
    noise_clamp: float = 0.3
    mask_threshold: float = 0.2


@dataclass
class SpatialMeta:
    layout: LayoutType
    original_shape: torch.Size
    height: int
    width: int
    batch_size: int
    num_frames: int = 1


def fls_config_from_dict(config: dict | None) -> FLSConfig:
    if not config:
        return FLSConfig()
    valid_fields = {field.name for field in fields(FLSConfig)}
    return FLSConfig(**{key: value for key, value in config.items() if key in valid_fields})


def _scheduler_config_dict(scheduler) -> dict:
    config = getattr(scheduler, "config", None)
    return config if isinstance(config, dict) else {}


def get_scheduler_fls_config(scheduler) -> FLSConfig:
    fls_config = getattr(scheduler, "fls_config", None)
    if fls_config is None:
        fls_config = fls_config_from_dict(_scheduler_config_dict(scheduler).get("fls"))
        scheduler.fls_config = fls_config
    return fls_config


def reset_scheduler_fls_state(scheduler) -> None:
    scheduler.fls_state = FLSState()


def init_scheduler_fls(scheduler) -> None:
    scheduler.fls_config = fls_config_from_dict(_scheduler_config_dict(scheduler).get("fls"))
    reset_scheduler_fls_state(scheduler)


def resolve_hw_from_ids(latent_image_ids: torch.Tensor | None, seq_len: int) -> tuple[int | None, int | None]:
    if latent_image_ids is None:
        return None, None
    height = int(latent_image_ids[0, :, 1].max().item()) + 1
    width = int(latent_image_ids[0, :, 2].max().item()) + 1
    if height * width != seq_len:
        logger.warning(f"FLS latent grid mismatch: H*W={height * width} != seq_len={seq_len}, skipping fovea enhancement")
        return None, None
    return height, width


def _normalize_layout(layout: str, tensor: torch.Tensor) -> tuple[LayoutType, torch.Tensor]:
    if layout == "video_4d":
        if tensor.ndim != 4:
            raise ValueError(f"video_4d layout expects 4D tensor (C, F, H, W), got shape {tuple(tensor.shape)}")
        return "video", tensor.unsqueeze(0)
    if layout not in {"seq", "spatial", "video"}:
        raise ValueError(f"Unsupported FLS layout: {layout}")
    return layout, tensor


def to_spatial_nchw(
    tensor: torch.Tensor,
    layout: str,
    height: int | None = None,
    width: int | None = None,
) -> tuple[torch.Tensor, SpatialMeta]:
    original_shape = tensor.shape
    original_layout = layout
    layout, tensor = _normalize_layout(layout, tensor)

    if layout == "spatial":
        if tensor.ndim != 4:
            raise ValueError(f"spatial layout expects (B, C, H, W), got shape {tuple(tensor.shape)}")
        batch_size, _, height, width = tensor.shape
        return tensor.contiguous(), SpatialMeta(
            layout="spatial",
            original_shape=tensor.shape,
            height=height,
            width=width,
            batch_size=batch_size,
        )

    if layout == "seq":
        if tensor.ndim != 3:
            raise ValueError(f"seq layout expects (B, seq_len, C), got shape {tuple(tensor.shape)}")
        if height is None or width is None:
            raise ValueError("seq layout requires height and width")
        batch_size, seq_len, channels = tensor.shape
        if height * width != seq_len:
            raise ValueError(f"seq layout mismatch: H*W={height * width} != seq_len={seq_len}")
        spatial = tensor.view(batch_size, height, width, channels).permute(0, 3, 1, 2).contiguous()
        return spatial, SpatialMeta(
            layout="seq",
            original_shape=tensor.shape,
            height=height,
            width=width,
            batch_size=batch_size,
        )

    if layout == "video":
        if tensor.ndim != 5:
            raise ValueError(f"video layout expects (B, C, F, H, W), got shape {tuple(tensor.shape)}")
        batch_size, channels, num_frames, height, width = tensor.shape
        spatial = tensor.permute(0, 2, 1, 3, 4).reshape(batch_size * num_frames, channels, height, width).contiguous()
        restore_layout: LayoutType = "video_4d" if original_layout == "video_4d" else "video"
        return spatial, SpatialMeta(
            layout=restore_layout,
            original_shape=original_shape,
            height=height,
            width=width,
            batch_size=batch_size,
            num_frames=num_frames,
        )

    raise ValueError(f"Unsupported FLS layout: {layout}")


def from_spatial_nchw(spatial_tensor: torch.Tensor, meta: SpatialMeta) -> torch.Tensor:
    if meta.layout == "spatial":
        return spatial_tensor

    if meta.layout == "seq":
        batch_size, channels, height, width = spatial_tensor.shape
        return spatial_tensor.permute(0, 2, 3, 1).contiguous().view(batch_size, height * width, channels)

    if meta.layout in {"video", "video_4d"}:
        batch_size, channels, height, width = spatial_tensor.shape
        num_frames = meta.num_frames
        if batch_size != meta.batch_size * num_frames:
            raise ValueError(f"video restore mismatch: expected N={meta.batch_size * num_frames}, got {batch_size}")
        video = spatial_tensor.view(meta.batch_size, num_frames, channels, height, width).permute(0, 2, 1, 3, 4).contiguous()
        if len(meta.original_shape) == 4:
            return video.squeeze(0)
        return video

    raise ValueError(f"Unsupported FLS layout: {meta.layout}")


def _build_fovea_mask(
    current_pred_spatial: torch.Tensor,
    previous_pred_spatial: torch.Tensor,
    state: FLSState,
    config: FLSConfig,
) -> torch.Tensor:
    delta = torch.abs(current_pred_spatial - previous_pred_spatial)
    delta_map = torch.mean(delta, dim=1, keepdim=True)
    delta_smooth = F.avg_pool2d(delta_map, kernel_size=5, stride=1, padding=2)

    mean_val = delta_smooth.mean()
    std_val = delta_smooth.std().clamp_min(1e-6)
    threshold = mean_val + (std_val * 0.5)

    current_mask = torch.sigmoid((delta_smooth - threshold) / std_val * 2.0)
    current_mask = torch.where(current_mask < config.mask_threshold, torch.zeros_like(current_mask), current_mask)

    if state.momentum_mask is None:
        state.momentum_mask = current_mask
    else:
        state.momentum_mask = (state.momentum_mask * config.mask_inertia) + (current_mask * (1.0 - config.mask_inertia))
    return state.momentum_mask


def _apply_spatial_enhancement(
    latents_spatial: torch.Tensor,
    active_mask: torch.Tensor,
    config: FLSConfig,
    step: int,
    total_steps: int,
    generator: torch.Generator | None,
) -> torch.Tensor:
    progress = step / max(total_steps - 1, 1)
    decay = 1.0 - progress

    if config.sharpness > 0:
        blurred_latents = F.avg_pool2d(latents_spatial, kernel_size=3, stride=1, padding=1)
        high_freq = latents_spatial - blurred_latents
        contrast_boost = high_freq * active_mask * (config.sharpness * 0.1 * decay)
        latents_spatial = latents_spatial + contrast_boost

    if config.fovea_strength > 0:
        injection_noise = torch.randn(
            latents_spatial.shape,
            generator=generator,
            device=latents_spatial.device,
            dtype=latents_spatial.dtype,
        )
        noise_scale = config.fovea_strength * 0.02 * decay
        perturbation = injection_noise * active_mask * noise_scale
        perturbation = torch.clamp(perturbation, -config.noise_clamp, config.noise_clamp)
        latents_spatial = latents_spatial + perturbation

    return latents_spatial


def apply_fls_enhancement(
    latents: torch.Tensor,
    noise_pred: torch.Tensor,
    step: int,
    total_steps: int,
    state: FLSState,
    config: FLSConfig,
    *,
    layout: str,
    height: int | None = None,
    width: int | None = None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    if not config.enable:
        return latents

    current_pred = noise_pred.clone().detach()

    if step < total_steps * config.warmup_ratio:
        state.previous_pred = current_pred
        return latents

    if state.previous_pred is None:
        state.previous_pred = current_pred
        return latents

    try:
        latents_spatial, latents_meta = to_spatial_nchw(latents, layout, height=height, width=width)
        current_pred_spatial, _ = to_spatial_nchw(current_pred, layout, height=height, width=width)
        previous_pred_spatial, _ = to_spatial_nchw(state.previous_pred, layout, height=height, width=width)
    except ValueError as exc:
        logger.warning(f"FLS skipped due to shape/layout error: {exc}")
        state.previous_pred = current_pred
        return latents

    active_mask = _build_fovea_mask(current_pred_spatial, previous_pred_spatial, state, config)
    latents_spatial = _apply_spatial_enhancement(
        latents_spatial,
        active_mask,
        config,
        step,
        total_steps,
        generator,
    )

    state.previous_pred = current_pred
    return from_spatial_nchw(latents_spatial, latents_meta)


def apply_scheduler_fls_enhancement(
    scheduler,
    latents: torch.Tensor,
    noise_pred: torch.Tensor,
    *,
    layout: str,
    height: int | None = None,
    width: int | None = None,
    latent_image_ids: torch.Tensor | None = None,
    total_steps: int | None = None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    config = get_scheduler_fls_config(scheduler)
    if not config.enable:
        return latents

    if layout == "seq" and (height is None or width is None):
        height, width = resolve_hw_from_ids(latent_image_ids, latents.shape[1])

    state = getattr(scheduler, "fls_state", None)
    if state is None:
        state = FLSState()
        scheduler.fls_state = state

    if total_steps is None:
        timesteps = getattr(scheduler, "timesteps", None)
        total_steps = len(timesteps) if timesteps is not None else getattr(scheduler, "infer_steps", 1)

    if generator is None:
        generator = getattr(scheduler, "generator", None)

    return apply_fls_enhancement(
        latents,
        noise_pred,
        step=scheduler.step_index,
        total_steps=total_steps,
        state=state,
        config=config,
        layout=layout,
        height=height,
        width=width,
        generator=generator,
    )
