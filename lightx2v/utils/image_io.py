"""Write prepared PIL images or OpenCV arrays to files."""

from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def validate_save_image_method(method: str) -> None:
    if method not in ("opencv", "pil"):
        raise ValueError(f"Unsupported save_image_method: {method!r}; expected 'opencv' or 'pil'.")


def validate_png_compress_level(level: int | None) -> None:
    """Validate the optional lossless PNG compression level (0-9)."""
    if level is not None and (isinstance(level, bool) or not isinstance(level, int) or not 0 <= level <= 9):
        raise ValueError("png_compress_level must be an integer from 0 to 9, or None")


def save_image(image: Image.Image | np.ndarray, output_path: str | Path, method: str = "opencv", *, png_compress_level: int | None = None) -> None:
    """Write a PIL image for ``pil`` or a uint8 BGR/BGRA/gray array for ``opencv``.

    The caller owns image preparation and profiling from Tensor to file.
    """
    validate_save_image_method(method)
    validate_png_compress_level(png_compress_level)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    use_png_level = path.suffix.lower() == ".png" and png_compress_level is not None
    if method == "pil":
        options = {"compress_level": png_compress_level} if use_png_level else {}
        image.save(path, **options)
    elif not cv2.imwrite(str(path), image, [cv2.IMWRITE_PNG_COMPRESSION, png_compress_level] if use_png_level else []):
        raise RuntimeError(f"Failed to save image to: {path}")
