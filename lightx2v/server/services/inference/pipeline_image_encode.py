"""Normalize image runner outputs to encoded image bytes (in-memory, no disk)."""

from __future__ import annotations

import base64
import os
import time
from io import BytesIO
from typing import Any, Optional

import cv2
import numpy as np
import torch
from PIL import Image
from loguru import logger


def _get_png_compression_level() -> int:
    raw = os.getenv("LIGHTX2V_SYNC_PNG_COMPRESSION", "6")
    try:
        level = int(raw)
    except ValueError:
        logger.warning(f"Invalid LIGHTX2V_SYNC_PNG_COMPRESSION={raw}, fallback to 6")
        return 6
    if level < 0 or level > 9:
        logger.warning(f"LIGHTX2V_SYNC_PNG_COMPRESSION={level} out of range [0,9], clamped")
        level = max(0, min(9, level))
    return level


PNG_COMPRESSION_LEVEL = _get_png_compression_level()


def _pil_to_png_bytes(pil_image: Image.Image) -> bytes:
    buf = BytesIO()
    img = pil_image
    if img.mode not in ("RGB", "RGBA"):
        img = img.convert("RGB")
    img.save(buf, format="PNG", compress_level=PNG_COMPRESSION_LEVEL)
    return buf.getvalue()


def _opencv_to_png_bytes(image: np.ndarray) -> bytes:
    success, encoded = cv2.imencode(".png", image)
    if not success:
        raise RuntimeError(f"Failed to encode OpenCV image to PNG: shape={image.shape}, dtype={image.dtype}")
    return encoded.tobytes()


def _first_image(images: Any) -> Any:
    image = images
    while isinstance(image, (list, tuple)):
        if not image:
            return None
        image = image[0]
    return image


def _image_to_png_bytes(image: Any) -> Optional[bytes]:
    if image is None:
        return None
    if isinstance(image, torch.Tensor):
        return _tensor_to_png_bytes(image)
    if isinstance(image, np.ndarray):
        return _opencv_to_png_bytes(image)
    if isinstance(image, Image.Image):
        return _pil_to_png_bytes(image)
    if isinstance(image, str):
        raw = base64.b64decode(image)
        img = Image.open(BytesIO(raw)).convert("RGB")
        return _pil_to_png_bytes(img)
    raise TypeError(f"Unexpected image element type: {type(image)}")


def _tensor_to_uint8_hwc(tensor: torch.Tensor) -> np.ndarray:
    """Normalize a detached CPU tensor to RGB/RGBA (or 2D grayscale) pixels."""
    if tensor.ndim == 4:
        tensor = tensor[0]
    if tensor.ndim != 3:
        raise TypeError(f"Unsupported tensor shape: {tuple(tensor.shape)}")
    if tensor.shape[0] in (1, 3, 4):
        tensor_chw = tensor
    elif tensor.shape[-1] in (1, 3, 4):
        tensor_chw = tensor.permute(2, 0, 1)
    else:
        raise TypeError(f"Unsupported tensor channel layout: {tuple(tensor.shape)}")
    if tensor_chw.dtype.is_floating_point:
        if float(tensor_chw.max()) <= 1.0:
            tensor_chw = (tensor_chw.clamp(0.0, 1.0) * 255.0).round()
        else:
            tensor_chw = tensor_chw.clamp(0.0, 255.0).round()
    array = tensor_chw.to(torch.uint8).permute(1, 2, 0).numpy()
    return array[:, :, 0] if array.shape[-1] == 1 else array


def _numpy_to_pil(image: np.ndarray) -> Image.Image:
    """Match OpenCV PNG -> Pillow RGB conversion without encoding a PNG."""
    if image.ndim == 3 and image.shape[-1] == 1:
        image = image[:, :, 0]
    if image.ndim not in (2, 3) or image.ndim == 3 and image.shape[-1] not in (3, 4):
        raise TypeError(f"Unsupported NumPy image shape: {image.shape}")
    if image.dtype == np.uint16:
        # Pillow decodes 16-bit RGB PNGs to their high 8 bits. Grayscale stays
        # I;16 and convert("RGB") clamps values at 255 instead.
        if image.ndim == 3:
            image = (image >> 8).astype(np.uint8)
    elif image.dtype != np.uint8:
        # OpenCV's PNG fallback saturates and rounds to uint8; it does not
        # interpret float NumPy arrays as normalized [0, 1] pixels.
        image = np.clip(np.rint(image), 0, 255).astype(np.uint8)
    if image.ndim == 3:
        code = cv2.COLOR_BGRA2RGBA if image.shape[-1] == 4 else cv2.COLOR_BGR2RGB
        image = cv2.cvtColor(image, code)
    return Image.fromarray(image).convert("RGB")


def _image_to_pil(image: Any) -> Image.Image:
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    if isinstance(image, torch.Tensor):
        pixels = _tensor_to_uint8_hwc(image.detach().cpu())
        return Image.fromarray(pixels).convert("RGB")
    if isinstance(image, np.ndarray):
        return _numpy_to_pil(image)
    if isinstance(image, str):
        with Image.open(BytesIO(base64.b64decode(image))) as source:
            return source.convert("RGB")
    raise TypeError(f"Unexpected image element type: {type(image)}")


def _tensor_to_png_bytes(image_tensor: torch.Tensor) -> bytes:
    total_start = time.perf_counter()
    task_tag = f"shape={tuple(image_tensor.shape)},dtype={image_tensor.dtype},device={image_tensor.device}"

    cpu_start = time.perf_counter()
    tensor = image_tensor.detach().cpu()
    cpu_ms = (time.perf_counter() - cpu_start) * 1000

    prep_start = time.perf_counter()
    arr = _tensor_to_uint8_hwc(tensor)
    prep_ms = (time.perf_counter() - prep_start) * 1000

    encode_start = time.perf_counter()
    if arr.ndim == 3:
        code = cv2.COLOR_RGBA2BGRA if arr.shape[-1] == 4 else cv2.COLOR_RGB2BGR
        arr = cv2.cvtColor(arr, code)
    png_bytes = _opencv_to_png_bytes(arr)
    encode_ms = (time.perf_counter() - encode_start) * 1000
    total_ms = (time.perf_counter() - total_start) * 1000
    logger.info(f"Tensor->PNG(cv2) cost total={total_ms:.2f}ms cpu_copy={cpu_ms:.2f}ms preprocess={prep_ms:.2f}ms encode={encode_ms:.2f}ms level={PNG_COMPRESSION_LEVEL} [{task_tag}]")
    return png_bytes


def encode_pipeline_return_to_png_bytes(pipeline_return: Any) -> Optional[bytes]:
    """Convert run_pipeline return value to a single PNG byte string, or None if not applicable."""
    if pipeline_return is None:
        return None
    try:
        if isinstance(pipeline_return, tuple) and len(pipeline_return) > 0:
            # e.g. BagelRunner returns (images, audio_or_none)
            pipeline_return = pipeline_return[0]
        if isinstance(pipeline_return, dict):
            pipeline_return = pipeline_return.get("images")
        return _image_to_png_bytes(_first_image(pipeline_return))
    except Exception as e:
        logger.exception(f"Failed to encode pipeline output to PNG: {e}")
        return None
    return None


def encode_pipeline_return_to_image_bytes(pipeline_return, save_result_path=None, output_compression=95):
    """Convert pixels directly to PIL for JPEG/WebP; preserve the PNG encoder."""
    from lightx2v.utils.image_output import image_format_for_path

    output_format = image_format_for_path(save_result_path)
    if output_format == "png":
        return encode_pipeline_return_to_png_bytes(pipeline_return)
    if isinstance(pipeline_return, tuple) and pipeline_return:
        pipeline_return = pipeline_return[0]
    if isinstance(pipeline_return, dict):
        pipeline_return = pipeline_return.get("images")
    value = _first_image(pipeline_return)
    if value is None:
        return None
    image = _image_to_pil(value)
    buffer = BytesIO()
    image.save(buffer, format=output_format.upper(), quality=output_compression)
    return buffer.getvalue()
