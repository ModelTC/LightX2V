"""Write prepared PIL images or OpenCV arrays to files."""

from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def validate_save_image_method(method: str) -> None:
    if method not in ("opencv", "pil"):
        raise ValueError(f"Unsupported save_image_method: {method!r}; expected 'opencv' or 'pil'.")


def save_image(image: Image.Image | np.ndarray, output_path: str | Path, method: str = "opencv", *, output_compression: int | None = None) -> None:
    """Write a PIL image for ``pil`` or a uint8 BGR/BGRA/gray array for ``opencv``.

    The caller owns image preparation and profiling from Tensor to file.
    """
    validate_save_image_method(method)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if output_compression is not None and (type(output_compression) is not int or not 0 <= output_compression <= 100):
        raise ValueError("output_compression must be an integer between 0 and 100")
    suffix = path.suffix.lower()
    lossy = suffix in (".jpg", ".jpeg", ".webp")
    if method == "pil":
        options = {"quality": output_compression} if lossy and output_compression is not None else {}
        image.save(path, **options)
    else:
        parameters = []
        if lossy and output_compression is not None:
            key = cv2.IMWRITE_WEBP_QUALITY if suffix == ".webp" else cv2.IMWRITE_JPEG_QUALITY
            parameters = [key, output_compression]
        if not cv2.imwrite(str(path), image, parameters):
            raise RuntimeError(f"Failed to save image to: {path}")
