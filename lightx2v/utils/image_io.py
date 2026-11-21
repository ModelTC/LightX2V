"""Read reference images and write prepared PIL images or OpenCV arrays."""

from contextlib import ExitStack
from io import BytesIO
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps
from pillow_heif import register_heif_opener

# Pillow's AVIF decoder and HEIF both accept the generic mif1/msf1 brands.
Image.init()
register_heif_opener()


def load_rgb_image(source) -> Image.Image:
    """Read an oriented RGB image, preserving the position of caller-owned streams."""
    with ExitStack() as stack:
        if hasattr(source, "read"):
            try:
                position = source.tell()
                source.seek(0)
            except (AttributeError, OSError):
                source = BytesIO(source.read())
            else:
                stack.callback(source.seek, position)
        if not isinstance(source, Image.Image):
            source = stack.enter_context(Image.open(source))
        image = ImageOps.exif_transpose(source)
        return image if image.mode == "RGB" else image.convert("RGB")


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
