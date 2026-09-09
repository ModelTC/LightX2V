# Adapted from Physical Intelligence OpenPI (Apache-2.0), commit 15a9616.
"""OpenPI image preprocessing, including the released Augmax 0.4.1 recipe."""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence
from dataclasses import dataclass

import torch
import torch.nn.functional as F  # noqa: N812

from . import image_tools
from .observation import Observation

logger = logging.getLogger("openpi")

IMAGE_KEYS = (
    "base_0_rgb",
    "left_wrist_0_rgb",
    "right_wrist_0_rgb",
)

IMAGE_RESOLUTION = (224, 224)


@dataclass(frozen=True)
class _GeometricParameters:
    crop_offsets_yx: torch.Tensor
    angles_radians: torch.Tensor


@dataclass(frozen=True)
class _ColorJitterParameters:
    brightness: torch.Tensor
    contrast: torch.Tensor
    saturation: torch.Tensor
    hue: torch.Tensor
    apply: torch.Tensor


def _sample_geometric_parameters(
    batch_size: int,
    device: torch.device,
    image_resolution: tuple[int, int],
) -> _GeometricParameters:
    height, width = image_resolution
    crop_height = int(height * 0.95)
    crop_width = int(width * 0.95)
    limits = torch.tensor(
        ((height - crop_height) / 2.0, (width - crop_width) / 2.0),
        dtype=torch.float32,
        device=device,
    )
    offsets = (torch.rand((batch_size, 2), device=device) * 2.0 - 1.0) * limits
    angles = (torch.rand(batch_size, device=device) * 10.0 - 5.0) * (math.pi / 180.0)
    return _GeometricParameters(offsets, angles)


def _sample_color_jitter_parameters(batch_size: int, device: torch.device) -> _ColorJitterParameters:
    def sample(strength: float) -> torch.Tensor:
        return (torch.rand(batch_size, device=device) * 2.0 - 1.0) * strength

    return _ColorJitterParameters(
        brightness=sample(0.3),
        contrast=sample(0.4),
        hue=sample(0.1),
        saturation=sample(0.5),
        apply=torch.rand(batch_size, device=device) < 0.5,
    )


def _apply_fused_geometric_transform(
    images: torch.Tensor,
    parameters: _GeometricParameters,
    image_resolution: tuple[int, int],
) -> torch.Tensor:
    """Apply RandomCrop(95%), Resize, and Rotate as one Augmax transform."""
    batch_size, height, width, _ = images.shape

    crop_height = int(height * 0.95)
    crop_width = int(width * 0.95)
    output_y = torch.arange(height, dtype=torch.float32, device=images.device) - (height / 2.0 - 0.5)
    output_x = torch.arange(width, dtype=torch.float32, device=images.device) - (width / 2.0 - 0.5)
    output_y, output_x = torch.meshgrid(output_y, output_x, indexing="ij")

    cosine = torch.cos(parameters.angles_radians)[:, None, None]
    sine = torch.sin(parameters.angles_radians)[:, None, None]
    input_y = (cosine * output_y + sine * output_x) * (crop_height / height)
    input_x = (-sine * output_y + cosine * output_x) * (crop_width / width)
    input_y += parameters.crop_offsets_yx[:, 0, None, None] + (height / 2.0 - 0.5)
    input_x += parameters.crop_offsets_yx[:, 1, None, None] + (width / 2.0 - 0.5)

    grid = torch.stack(
        (
            input_x * (2.0 / (width - 1)) - 1.0,
            input_y * (2.0 / (height - 1)) - 1.0,
        ),
        dim=-1,
    )
    transformed = F.grid_sample(
        images.permute(0, 3, 1, 2),
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )
    return transformed.permute(0, 2, 3, 1)


def _rgb_to_hsv(images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    value, argmax = images.max(dim=-1)
    minimum = images.min(dim=-1).values
    value_range = value - minimum
    safe_range = torch.where(value_range == 0.0, torch.ones_like(value_range), value_range)
    second = (argmax + 1) % 3
    third = (argmax + 2) % 3
    second_value = torch.gather(images, -1, second.unsqueeze(-1)).squeeze(-1)
    third_value = torch.gather(images, -1, third.unsqueeze(-1)).squeeze(-1)
    hue = (2.0 * argmax.to(images.dtype) + (second_value - third_value) / safe_range) / 6.0
    hue = torch.where(value_range == 0.0, torch.zeros_like(hue), hue)
    safe_value = torch.where(value == 0.0, torch.ones_like(value), value)
    saturation = torch.where(value == 0.0, torch.zeros_like(value), value_range / safe_value)
    return hue, saturation, value


def _hsv_to_rgb(hue: torch.Tensor, saturation: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
    n = torch.tensor((5.0, 3.0, 1.0), dtype=value.dtype, device=value.device)
    k = torch.remainder(n + hue.unsqueeze(-1) * 6.0, 6.0)
    scale = torch.maximum(torch.zeros_like(k), torch.minimum(torch.minimum(k, 4.0 - k), torch.ones_like(k)))
    return value.unsqueeze(-1) - value.unsqueeze(-1) * saturation.unsqueeze(-1) * scale


def _adjust_brightness(value: torch.Tensor, amount: torch.Tensor) -> torch.Tensor:
    return torch.where(amount < 0.0, value * (1.0 + amount), value * (1.0 - amount) + amount)


def _adjust_contrast(value: torch.Tensor, amount: torch.Tensor) -> torch.Tensor:
    slant = torch.tan((amount + 1.0) * (math.pi / 4.0))
    slant_squared = slant.square()
    first_break = (slant - slant_squared) / (2.0 * (1.0 - slant_squared))
    second_break = 1.0 - first_break
    return torch.where(
        value < first_break,
        value / slant,
        torch.where(value > second_break, value / slant + 1.0 - 1.0 / slant, slant * (value - 0.5) + 0.5),
    )


def _apply_color_jitter(images: torch.Tensor, parameters: _ColorJitterParameters) -> torch.Tensor:
    hue, saturation, value = _rgb_to_hsv(images)
    value = _adjust_brightness(value, parameters.brightness[:, None, None])
    value = _adjust_contrast(value, parameters.contrast[:, None, None])
    hue += parameters.hue[:, None, None]

    # Augmax 0.4.1 samples saturation without applying it.
    transformed = _hsv_to_rgb(hue, saturation, value)
    return torch.where(parameters.apply[:, None, None, None], transformed, images)


def _augment_images(
    images: torch.Tensor,
    *,
    geometric: _GeometricParameters | None,
    color: _ColorJitterParameters,
    image_resolution: tuple[int, int],
) -> torch.Tensor:
    images = images / 2.0 + 0.5
    if geometric is not None:
        images = _apply_fused_geometric_transform(images, geometric, image_resolution)
    return _apply_color_jitter(images, color) * 2.0 - 1.0


def preprocess_observation_pytorch(
    observation,
    *,
    train: bool = False,
    image_keys: Sequence[str] = IMAGE_KEYS,
    image_resolution: tuple[int, int] = IMAGE_RESOLUTION,
):
    if not set(image_keys).issubset(observation.images):
        raise ValueError(f"images dict missing keys: expected {image_keys}, got {list(observation.images)}")

    batch_shape = observation.state.shape[:-1]
    batch_size = observation.state.shape[0]
    device = observation.state.device
    geometric_parameters = None
    base_color_parameters = None
    wrist_color_parameters = None

    out_images = {}
    for key in image_keys:
        image = observation.images[key]
        if image.ndim != 4 or (image.shape[1] != 3 and image.shape[-1] != 3):
            raise ValueError(f"{key} must be BCHW or BHWC RGB, got {tuple(image.shape)}")
        if image.shape[0] != batch_size:
            raise ValueError(f"{key} batch size {image.shape[0]} does not match state batch size {batch_size}")
        channels_first = image.shape[1] == 3
        if channels_first:
            image = image.permute(0, 2, 3, 1)

        if image.shape[1:3] != image_resolution:
            if train:
                raise ValueError(f"Training image {key} has shape {tuple(image.shape[1:3])}; resize the uint8 image with PIL before converting it to [-1, 1]")
            logger.info("Resizing image %s from %s to %s", key, image.shape[1:3], image_resolution)
            image = image_tools.resize_with_pad_torch(image, *image_resolution)

        if train:
            if image.dtype != torch.float32:
                raise ValueError(f"Training image {key} must be float32 in [-1, 1], got {image.dtype}")
            if "wrist" in key:
                if wrist_color_parameters is None:
                    wrist_color_parameters = _sample_color_jitter_parameters(batch_size, device)
                image = _augment_images(
                    image,
                    geometric=None,
                    color=wrist_color_parameters,
                    image_resolution=image_resolution,
                )
            else:
                if geometric_parameters is None:
                    geometric_parameters = _sample_geometric_parameters(batch_size, device, image_resolution)
                if base_color_parameters is None:
                    base_color_parameters = _sample_color_jitter_parameters(batch_size, device)
                image = _augment_images(
                    image,
                    geometric=geometric_parameters,
                    color=base_color_parameters,
                    image_resolution=image_resolution,
                )

        if channels_first:
            image = image.permute(0, 3, 1, 2)
        out_images[key] = image

    out_masks = {}
    for key in out_images:
        if key not in observation.image_masks:
            out_masks[key] = torch.ones(batch_shape, dtype=torch.bool, device=device)
        else:
            out_masks[key] = observation.image_masks[key]

    return Observation(
        images=out_images,
        image_masks=out_masks,
        state=observation.state,
        tokenized_prompt=observation.tokenized_prompt,
        tokenized_prompt_mask=observation.tokenized_prompt_mask,
        token_ar_mask=observation.token_ar_mask,
        token_loss_mask=observation.token_loss_mask,
    )
