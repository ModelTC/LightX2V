"""Human-viewable relative-depth previews for Depth Anything 3."""

from pathlib import Path

import numpy as np
from PIL import Image


def save_relative_depth_preview(depth: np.ndarray, sky: np.ndarray | None, path: str | Path, *, size: tuple[int, int]) -> Path:
    """Save an 8-bit grayscale PNG; near is bright and sky is black.

    ``size`` is the original input image size in (width, height) order.
    Percentiles exclude sky so a large sky region does not flatten the scene.
    """
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
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    preview.save(output, format="PNG")
    return output
