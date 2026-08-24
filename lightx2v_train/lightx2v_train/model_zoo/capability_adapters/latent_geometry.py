"""Model-agnostic descriptions of latent tensor geometry."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol


def _resolve_attribute(root, path: str):
    value = root
    for name in path.split("."):
        try:
            value = getattr(value, name)
        except AttributeError as exc:
            raise AttributeError(f"Cannot resolve {path!r}: {type(value).__name__} has no attribute {name!r}.") from exc
    return value


class LatentGeometry(Protocol):
    """Build a model's latent tensor shape from an output spatial size."""

    def shape(self, model, height: int, width: int) -> tuple[int, ...]:
        """Return a singleton latent shape for ``height`` x ``width`` output."""


@dataclass(frozen=True)
class SpatialLatentGeometry:
    """Declarative geometry for image-like 4D or single-frame 5D latents.

    ``spatial_downsample_multiplier`` describes additional downsampling on top
    of the model's VAE scale. ``input_size_multiplier`` captures packing
    requirements such as an even VAE-latent grid.
    """

    channels_path: str
    spatial_scale_path: str = "vae_scale_factor"
    spatial_downsample_multiplier: int = 1
    input_size_multiplier: int = 2
    temporal_size: int | None = None

    def __post_init__(self) -> None:
        if self.spatial_downsample_multiplier <= 0:
            raise ValueError("spatial_downsample_multiplier must be positive.")
        if self.input_size_multiplier <= 0:
            raise ValueError("input_size_multiplier must be positive.")
        if self.temporal_size is not None and self.temporal_size <= 0:
            raise ValueError("temporal_size must be positive when provided.")

    def shape(self, model, height: int, width: int) -> tuple[int, ...]:
        height, width = int(height), int(width)
        channels = int(_resolve_attribute(model, self.channels_path))
        spatial_scale = int(_resolve_attribute(model, self.spatial_scale_path))
        if channels <= 0 or spatial_scale <= 0:
            raise ValueError(f"Latent channels and spatial scale must be positive, got channels={channels}, scale={spatial_scale}.")

        spatial_downsample = spatial_scale * self.spatial_downsample_multiplier
        required_multiple = math.lcm(
            spatial_downsample,
            spatial_scale * self.input_size_multiplier,
        )
        if height <= 0 or width <= 0 or height % required_multiple or width % required_multiple:
            raise ValueError(f"Output height and width must be positive multiples of {required_multiple}, got {height}x{width}.")

        spatial_shape = (
            height // spatial_downsample,
            width // spatial_downsample,
        )
        if self.temporal_size is None:
            return 1, channels, *spatial_shape
        return 1, channels, self.temporal_size, *spatial_shape
