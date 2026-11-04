"""Model-specific postprocessing for Depth Anything 3 mono predictions."""

import numpy as np


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
