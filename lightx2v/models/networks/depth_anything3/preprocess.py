"""RGB frame preprocessing for DA3MONO-LARGE.

Follows the Apache-2.0 Depth Anything 3 InputProcessor's resolution methods,
patch alignment, and ImageNet normalization for one RGB image.
"""

from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

PROCESS_RES_METHODS = frozenset({"upper_bound_resize", "lower_bound_resize", "upper_bound_crop", "lower_bound_crop"})


def preprocess_image(path: str | Path, *, process_res: int = 504, process_res_method: str = "upper_bound_resize") -> tuple[torch.Tensor, tuple[int, int]]:
    """Return normalized (1, 3, H, W) input and original (width, height)."""
    with Image.open(path) as image:
        rgb = np.asarray(image.convert("RGB"))
    return preprocess_rgb(rgb, process_res=process_res, process_res_method=process_res_method)


def preprocess_rgb(rgb: np.ndarray, *, process_res: int = 504, process_res_method: str = "upper_bound_resize") -> tuple[torch.Tensor, tuple[int, int]]:
    """Preprocess one uint8 RGB frame without writing it to disk."""
    if process_res < 14:
        raise ValueError("process_res must be at least one DA3 patch (14 pixels)")
    if not isinstance(process_res_method, str) or process_res_method not in PROCESS_RES_METHODS:
        raise ValueError(f"Unknown DA3 process_res_method: {process_res_method!r}; expected one of {sorted(PROCESS_RES_METHODS)}")
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8 or not rgb.shape[0] or not rgb.shape[1]:
        raise ValueError("DA3 input frame must be nonempty uint8 RGB [height,width,3]")
    height, width = rgb.shape[:2]
    original_size = (width, height)
    reference_side = max(height, width) if process_res_method.startswith("upper_bound") else min(height, width)
    scale = process_res / reference_side
    target_width = max(1, round(width * scale))
    target_height = max(1, round(height * scale))
    if (target_width, target_height) != (width, height):
        interpolation = cv2.INTER_CUBIC if scale > 1 else cv2.INTER_AREA
        rgb = cv2.resize(rgb, (target_width, target_height), interpolation=interpolation)

    def align_to_patch(value: int) -> int:
        lower = (value // 14) * 14
        upper = lower + 14
        return max(14, upper if upper - value <= value - lower else lower)

    if process_res_method.endswith("resize"):
        aligned_width = align_to_patch(target_width)
        aligned_height = align_to_patch(target_height)
        if (aligned_width, aligned_height) != (target_width, target_height):
            interpolation = cv2.INTER_CUBIC if aligned_width > target_width or aligned_height > target_height else cv2.INTER_AREA
            rgb = cv2.resize(rgb, (aligned_width, aligned_height), interpolation=interpolation)
    else:
        aligned_width = (target_width // 14) * 14
        aligned_height = (target_height // 14) * 14
        if not aligned_width or not aligned_height:
            raise ValueError("DA3 crop alignment would produce an empty image; use a lower_bound method or a larger process_res")
        left = (target_width - aligned_width) // 2
        top = (target_height - aligned_height) // 2
        rgb = rgb[top : top + aligned_height, left : left + aligned_width]
    tensor = torch.from_numpy(rgb.copy()).permute(2, 0, 1).unsqueeze(0).float().div_(255.0)
    mean = tensor.new_tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1)
    std = tensor.new_tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1)
    return (tensor - mean) / std, original_size
