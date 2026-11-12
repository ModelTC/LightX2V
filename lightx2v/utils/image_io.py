"""Write prepared PIL images or OpenCV arrays to files."""

from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def validate_save_image_method(method: str) -> None:
    if method not in ("opencv", "pil"):
        raise ValueError(f"Unsupported save_image_method: {method!r}; expected 'opencv' or 'pil'.")


def save_image(image: Image.Image | np.ndarray, output_path: str | Path, method: str = "opencv") -> None:
    """Write a PIL image for ``pil`` or a uint8 BGR/BGRA/gray array for ``opencv``.

    The caller owns image preparation and profiling from Tensor to file.
    """
    validate_save_image_method(method)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if method == "pil":
        image.save(path)
    elif not cv2.imwrite(str(path), image):
        raise RuntimeError(f"Failed to save image to: {path}")
