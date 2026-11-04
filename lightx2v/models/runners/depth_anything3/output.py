"""Depth Anything 3 preview rendering and optional file outputs."""

import os
import tempfile
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image


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


class StreamingNpzWriter:
    """Write per-frame NPY members into one ZIP64 archive without holding frames."""

    def __init__(self, path: str | Path):
        self.output = Path(path)
        self.temporary = None
        self.archive = None

    def __enter__(self):
        self.output.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(prefix=f".{self.output.name}.", suffix=".part", dir=self.output.parent)
        os.close(fd)
        self.temporary = Path(temporary)
        try:
            self.archive = zipfile.ZipFile(self.temporary, mode="w", compression=zipfile.ZIP_DEFLATED, compresslevel=1, allowZip64=True)
        except BaseException:
            self.temporary.unlink(missing_ok=True)
            raise
        return self

    def write_frame(self, key: str, array: np.ndarray) -> None:
        """Write one array as a named NPY member."""
        with self.archive.open(f"{key}.npy", mode="w", force_zip64=True) as member:
            np.lib.format.write_array(member, np.asarray(array), allow_pickle=False)

    def __exit__(self, exc_type, exc, traceback):
        try:
            if exc_type is None:
                self.archive.close()
                os.replace(self.temporary, self.output)
            else:
                try:
                    self.archive.close()
                except Exception:
                    pass  # Preserve the inference error; the partial archive is removed.
        finally:
            self.temporary.unlink(missing_ok=True)
        return False


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
