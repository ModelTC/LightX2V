# SPDX-License-Identifier: Apache-2.0
"""Prepare the static picture shared by StreamRender's Qwen and H3 paths."""

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps

from .tae_encoder import StreamRenderH3TAEEncoder


@dataclass
class StreamRenderH3Picture:
    image: Image.Image
    latents: torch.Tensor


def _seed(*components: object) -> int:
    digest = hashlib.sha256("_".join(str(value) for value in components).encode()).hexdigest()
    return int(digest, 16) % (2**63)


@torch.no_grad()
def prepare_picture(
    path: str | Path,
    *,
    height: int,
    width: int,
    seed: int,
    encoder: StreamRenderH3TAEEncoder,
    device: torch.device | str,
) -> StreamRenderH3Picture:
    """Match the release's EXIF, Lanczos, TAE, and 0.999 anchor policy."""
    source_path = Path(path).expanduser().resolve()
    with Image.open(source_path) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
    if image.size != (width, height):
        image = image.resize((width, height), Image.Resampling.LANCZOS)
    pixels = torch.from_numpy(np.array(image, copy=True)).to(device).permute(2, 0, 1).unsqueeze(0).float().div_(255)
    clean = encoder.encode_picture(pixels).detach().cpu().float().to(device)
    generator = torch.Generator().manual_seed(_seed(seed, "streaming_picture_noise", source_path))
    noise = torch.randn(clean.shape, generator=generator).to(clean)
    latents = 0.999 * clean + 0.001 * noise
    return StreamRenderH3Picture(image=image, latents=latents)


__all__ = ["StreamRenderH3Picture", "prepare_picture"]
