"""Human-viewable relative-depth previews for Depth Anything 3."""

from pathlib import Path

import numpy as np
from PIL import Image


def postprocess_mono_sky(depth: np.ndarray, sky: np.ndarray) -> np.ndarray:
    """Set sky depth to the 99th percentile of non-sky depth, as in DA3 mono.

    Use all non-sky pixels for a deterministic percentile instead of the
    official implementation's random 100k-pixel sample for large images.
    """
    depth = np.asarray(depth, dtype=np.float32)
    sky = np.asarray(sky, dtype=np.float32)
    if depth.ndim != 2 or sky.shape != depth.shape or not np.isfinite(depth).all() or not np.isfinite(sky).all():
        raise ValueError("Depth and sky predictions must be finite two-dimensional arrays of the same shape")
    non_sky = sky < 0.3
    if np.count_nonzero(non_sky) <= 10 or np.count_nonzero(~non_sky) <= 10:
        return depth
    result = depth.copy()
    result[~non_sky] = np.float32(np.quantile(depth[non_sky], 0.99))
    return result


def save_float_depth(depth: np.ndarray, path: str | Path) -> Path:
    """Save non-normalized, sky-corrected model-resolution depth as float32 NPY."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as file:
        np.save(file, np.asarray(depth, dtype=np.float32))
    return output


def save_sky_mask(sky: np.ndarray, path: str | Path) -> Path:
    """Save the official Prediction.sky threshold (>= 0.5) as a binary PNG."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    mask = (np.asarray(sky, dtype=np.float32) >= 0.5).astype(np.uint8) * 255
    Image.fromarray(mask).save(output, format="PNG")
    return output


def save_relative_depth_preview(depth: np.ndarray, sky: np.ndarray | None, path: str | Path, *, size: tuple[int, int]) -> Path:
    """Save an 8-bit grayscale PNG; near is bright and sky is black.

    ``size`` is the original input image size in (width, height) order.
    Percentiles exclude sky so a large sky region does not flatten the scene.
    """
    preview = Image.fromarray(render_relative_depth_preview(depth, sky, size=size))
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    preview.save(output, format="PNG")
    return output


def render_relative_depth_preview(depth: np.ndarray, sky: np.ndarray | None, *, size: tuple[int, int]) -> np.ndarray:
    """Render one relative-depth preview as uint8 grayscale [height,width]."""
    depth = np.asarray(depth, dtype=np.float32)
    if depth.ndim != 2 or not np.isfinite(depth).all():
        raise ValueError("Depth prediction must be a finite two-dimensional array")
    if sky is not None:
        sky = np.asarray(sky, dtype=np.float32)
        if sky.shape != depth.shape:
            raise ValueError("Sky prediction must have the same shape as depth")
        non_sky = sky < 0.3
    else:
        non_sky = np.ones_like(depth, dtype=bool)
    valid = depth[non_sky] if np.any(non_sky) else depth.ravel()
    lo, hi = np.percentile(valid, (1, 99))
    normalized = 1.0 - np.clip((depth - lo) / max(float(hi - lo), 1e-6), 0.0, 1.0)
    normalized[~non_sky] = 0.0
    preview = Image.fromarray(np.rint(normalized * 255).astype(np.uint8))
    if preview.size != size:
        preview = preview.resize(size, Image.Resampling.BILINEAR)
    return np.asarray(preview)
