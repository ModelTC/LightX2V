"""Normalize image runner outputs to PNG bytes (in-memory, no disk)."""

from __future__ import annotations

import base64
import os
import time
from io import BytesIO
from typing import Any, Optional

import torch
from PIL import Image
from loguru import logger

try:
    from torchvision.io import encode_png as tv_encode_png
except Exception:
    tv_encode_png = None


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


def _pil_images_structure_to_png(images: Any) -> bytes:
    first = images[0]
    if isinstance(first, list):
        pil_image = first[0]
    else:
        pil_image = first
    if not hasattr(pil_image, "save"):
        raise TypeError(f"Unexpected image element type: {type(pil_image)}")
    return _pil_to_png_bytes(pil_image)


def _tensor_to_png_bytes(image_tensor: torch.Tensor) -> bytes:
    total_start = time.perf_counter()
    task_tag = f"shape={tuple(image_tensor.shape)},dtype={image_tensor.dtype},device={image_tensor.device}"

    cpu_start = time.perf_counter()
    tensor = image_tensor.detach().cpu()
    cpu_ms = (time.perf_counter() - cpu_start) * 1000

    if tensor.ndim == 4:
        tensor = tensor[0]
    if tensor.ndim != 3:
        raise TypeError(f"Unsupported tensor shape: {tuple(tensor.shape)}")

    prep_start = time.perf_counter()
    # Normalize layout once: keep CHW for fast PNG encoding path.
    if tensor.shape[0] in (1, 3, 4):
        tensor_chw = tensor
    elif tensor.shape[-1] in (1, 3, 4):
        tensor_chw = tensor.permute(2, 0, 1)
    else:
        raise TypeError(f"Unsupported tensor channel layout: {tuple(tensor.shape)}")

    if tensor_chw.dtype.is_floating_point:
        # Most runners output floats in [0, 1].
        if float(tensor_chw.max()) <= 1.0:
            tensor_chw = (tensor_chw.clamp(0.0, 1.0) * 255.0).round()
        else:
            tensor_chw = tensor_chw.clamp(0.0, 255.0).round()

    tensor_chw = tensor_chw.to(torch.uint8)
    prep_ms = (time.perf_counter() - prep_start) * 1000

    # Fast path: encode PNG directly from CHW uint8 tensor.
    if tv_encode_png is not None:
        encode_start = time.perf_counter()
        png_bytes = tv_encode_png(tensor_chw, compression_level=PNG_COMPRESSION_LEVEL).numpy().tobytes()
        encode_ms = (time.perf_counter() - encode_start) * 1000
        total_ms = (time.perf_counter() - total_start) * 1000
        logger.info(f"Tensor->PNG(tv) cost total={total_ms:.2f}ms cpu_copy={cpu_ms:.2f}ms preprocess={prep_ms:.2f}ms encode={encode_ms:.2f}ms level={PNG_COMPRESSION_LEVEL} [{task_tag}]")
        return png_bytes

    encode_start = time.perf_counter()
    arr = tensor_chw.permute(1, 2, 0).numpy()
    if arr.shape[-1] == 1:
        arr = arr[:, :, 0]
    png_bytes = _pil_to_png_bytes(Image.fromarray(arr))
    encode_ms = (time.perf_counter() - encode_start) * 1000
    total_ms = (time.perf_counter() - total_start) * 1000
    logger.info(f"Tensor->PNG(pil) cost total={total_ms:.2f}ms cpu_copy={cpu_ms:.2f}ms preprocess={prep_ms:.2f}ms encode={encode_ms:.2f}ms level={PNG_COMPRESSION_LEVEL} [{task_tag}]")
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
            images = pipeline_return.get("images")
            if images is None:
                return None
            if isinstance(images, torch.Tensor):
                return _tensor_to_png_bytes(images)
            return _pil_images_structure_to_png(images)
        if isinstance(pipeline_return, list) and len(pipeline_return) > 0:
            if isinstance(pipeline_return[0], torch.Tensor):
                return _tensor_to_png_bytes(pipeline_return[0])
            return _pil_images_structure_to_png(pipeline_return)
        if isinstance(pipeline_return, torch.Tensor):
            return _tensor_to_png_bytes(pipeline_return)
        if isinstance(pipeline_return, Image.Image):
            return _pil_to_png_bytes(pipeline_return)
        if isinstance(pipeline_return, str):
            raw = base64.b64decode(pipeline_return)
            img = Image.open(BytesIO(raw)).convert("RGB")
            return _pil_to_png_bytes(img)
    except Exception as e:
        logger.exception(f"Failed to encode pipeline output to PNG: {e}")
        return None
    return None
