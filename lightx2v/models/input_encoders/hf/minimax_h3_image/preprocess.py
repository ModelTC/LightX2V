"""Reference-image preparation shared by H3-Image's two image encoders."""

import math
from dataclasses import dataclass
from os import PathLike

import numpy as np
import torch
from PIL import Image, ImageOps


@dataclass(frozen=True)
class ReferenceImage:
    qwen_image: Image.Image
    vae_image: Image.Image


def _bound_short_edge(image, min_short_edge, max_short_edge):
    width, height = image.size
    short_edge = min(width, height)
    if min_short_edge <= short_edge <= max_short_edge:
        return image
    scale = (min_short_edge if short_edge < min_short_edge else max_short_edge) / short_edge
    size = (max(32, round(width * scale / 32) * 32), max(32, round(height * scale / 32) * 32))
    return image.resize(size, Image.Resampling.LANCZOS)


def prepare_reference_images(image_paths):
    """Read references in prompt order, retaining each reference's own geometry."""
    if image_paths is None:
        return []
    if isinstance(image_paths, (str, PathLike, Image.Image)):
        image_paths = [image_paths]
    sources = []
    for source in image_paths:
        if isinstance(source, str):
            sources.extend(path.strip() for path in source.split(",") if path.strip())
        else:
            sources.append(source)
    if len(sources) > 16:
        raise ValueError("MiniMax-H3-Image supports at most 16 reference images")
    references = []
    for source in sources:
        if isinstance(source, Image.Image):
            image = ImageOps.exif_transpose(source).convert("RGB")
        else:
            with Image.open(source) as opened:
                image = ImageOps.exif_transpose(opened).convert("RGB")
        if min(image.size) < 32:
            raise ValueError(f"MiniMax-H3-Image references need at least 32 pixels on each side, got {image.size}")
        references.append(
            ReferenceImage(
                qwen_image=_bound_short_edge(image, 1024, 1024),
                vae_image=_bound_short_edge(image, 1024, 2048),
            )
        )
    return references


def get_image_canvas_size(height, width, min_generation_area=2048**2):
    """Resolve the generation canvas; decoding returns to the requested size."""
    if height <= 0 or width <= 0:
        raise ValueError(f"Image height and width must be positive, got {height}x{width}")
    if not 1024 <= min_generation_area <= 4096**2:
        raise ValueError(f"min_generation_area must be within [1024, {4096**2}], got {min_generation_area}")
    scale = max(1.0, math.sqrt(min_generation_area / (height * width)))
    return max(32, round(height * scale / 32) * 32), max(32, round(width * scale / 32) * 32)


def image_to_tensor(image):
    """Return CPU uint8 RGB [1,3,H,W]; the VAE scales pixels on its device."""
    pixels = torch.from_numpy(np.array(image, copy=True))
    return pixels.permute(2, 0, 1).unsqueeze(0)
