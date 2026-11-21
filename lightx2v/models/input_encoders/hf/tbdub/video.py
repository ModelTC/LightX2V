"""TBDub's cropped-video input and student continuation layout (25 FPS)."""

import cv2
import numpy as np
import torch
from PIL import Image


def read_video(path, size, fps=25, frame_limit=None):
    capture = cv2.VideoCapture(str(path))
    frames = []
    try:
        source_fps = capture.get(cv2.CAP_PROP_FPS)
        if not capture.isOpened() or not np.isfinite(source_fps) or abs(source_fps - fps) > 0.01:
            raise ValueError(f"TBDub requires a readable {fps}-FPS cropped video; got {source_fps:g} FPS: {path}")
        while frame_limit is None or len(frames) < frame_limit:
            ok, frame = capture.read()
            if not ok:
                break
            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            frames.append(np.asarray(image.resize((size[1], size[0]), Image.Resampling.BILINEAR)))
    finally:
        capture.release()
    if not frames:
        raise ValueError(f"No frames decoded from {path}")
    return np.stack(frames)


def continuation_frames(frames, count, motion_frames=5):
    """Ping-pong a short reference and prepend the student's first-frame history."""
    if len(frames) == 1:
        indices = np.zeros(count, dtype=np.int64)
    else:
        period = 2 * (len(frames) - 1)
        indices = np.arange(count) % period
        indices = np.minimum(indices, period - indices)
    reference = frames[indices]
    return np.concatenate((np.repeat(reference[:1], motion_frames, axis=0), reference)), reference


def video_tensor(frames, device, dtype):
    # The source casts pixels before normalization; normalizing in FP32 first
    # produces different BF16 VAE inputs.
    pixels = torch.from_numpy(np.ascontiguousarray(frames)).to(device=device, dtype=dtype)
    return (pixels * (2.0 / 255.0) - 1).permute(3, 0, 1, 2).contiguous()


def match_reference_colors(frames, reference):
    """Per-frame Lab matching used by the released cropped-video inference."""
    from skimage.color import lab2rgb, rgb2lab

    output = []
    for frame, source in zip(frames, reference, strict=True):
        generated_lab = rgb2lab(frame / 255.0)
        source_lab = rgb2lab(source / 255.0)
        corrected = generated_lab.copy()
        # Preserve source reduction order. Multi-axis reductions accumulate
        # differently on constant frames and can change the zero-std branch.
        for channel in range(3):
            mean = generated_lab[:, :, channel].mean()
            std = generated_lab[:, :, channel].std()
            reference_mean = source_lab[:, :, channel].mean()
            reference_std = source_lab[:, :, channel].std()
            if std == 0:
                corrected[:, :, channel] = reference_mean
            else:
                corrected[:, :, channel] = (corrected[:, :, channel] - mean) * (reference_std / std) + reference_mean
        output.append((lab2rgb(corrected).clip(0, 1) * 255).astype(np.uint8))
    return np.stack(output)
