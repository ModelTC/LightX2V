"""Single-image preprocessing for DA3MONO-LARGE.

Follows the Apache-2.0 Depth Anything 3 InputProcessor's upper-bound resize,
patch alignment, and ImageNet normalization for one RGB image.
"""

from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image


def preprocess_image(path: str | Path, *, process_res: int = 504) -> tuple[torch.Tensor, tuple[int, int]]:
    """Return normalized (1, 3, H, W) input and original (width, height)."""
    if process_res < 14:
        raise ValueError("process_res must be at least one DA3 patch (14 pixels)")
    with Image.open(path) as image:
        rgb = np.asarray(image.convert("RGB"))
    height, width = rgb.shape[:2]
    original_size = (width, height)
    scale = process_res / max(height, width)
    target_width = max(1, round(width * scale))
    target_height = max(1, round(height * scale))
    if (target_width, target_height) != (width, height):
        interpolation = cv2.INTER_CUBIC if scale > 1 else cv2.INTER_AREA
        rgb = cv2.resize(rgb, (target_width, target_height), interpolation=interpolation)

    def align_to_patch(value: int) -> int:
        lower = (value // 14) * 14
        upper = lower + 14
        return max(14, upper if upper - value <= value - lower else lower)

    aligned_width = align_to_patch(target_width)
    aligned_height = align_to_patch(target_height)
    if (aligned_width, aligned_height) != (target_width, target_height):
        interpolation = cv2.INTER_CUBIC if aligned_width > target_width or aligned_height > target_height else cv2.INTER_AREA
        rgb = cv2.resize(rgb, (aligned_width, aligned_height), interpolation=interpolation)
    tensor = torch.from_numpy(rgb.copy()).permute(2, 0, 1).unsqueeze(0).float().div_(255.0)
    mean = tensor.new_tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1)
    std = tensor.new_tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1)
    return (tensor - mean) / std, original_size
