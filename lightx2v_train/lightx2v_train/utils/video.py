import os
from typing import Iterable, Sequence

import imageio
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


def _to_even_frame(frame: np.ndarray) -> np.ndarray:
    height, width = frame.shape[:2]
    if height % 2 == 0 and width % 2 == 0:
        return frame
    return np.pad(frame, ((0, height % 2), (0, width % 2), (0, 0)), mode="edge")


def save_mp4(frames: Iterable[Image.Image], path: str, fps: int = 8):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    writer = imageio.get_writer(
        path,
        fps=max(fps, 1),
        codec="libx264",
        format="FFMPEG",
        pixelformat="yuv420p",
    )
    try:
        for frame in frames:
            writer.append_data(_to_even_frame(np.array(frame.convert("RGB"))))
    finally:
        writer.close()


def pil_frames_to_video_tensor(frames: Sequence[Image.Image]) -> torch.Tensor:
    if len(frames) == 0:
        raise ValueError("`frames` must be non-empty.")

    frame_tensors = []
    for frame in frames:
        arr = np.array(frame.convert("RGB"), dtype=np.float32) / 255.0
        x = torch.from_numpy(arr).permute(2, 0, 1).contiguous()  # [3, H, W]
        frame_tensors.append(x)
    return torch.stack(frame_tensors, dim=1)  # [3, T, H, W]


def _gaussian_kernel_2d(kernel_size: int, sigma: float, channels: int, device: torch.device, dtype: torch.dtype):
    coords = torch.arange(kernel_size, device=device, dtype=dtype) - (kernel_size - 1) / 2.0
    g = torch.exp(-(coords**2) / (2.0 * sigma * sigma))
    g = g / g.sum()
    kernel_2d = torch.outer(g, g)
    kernel_2d = kernel_2d / kernel_2d.sum()
    kernel_2d = kernel_2d.view(1, 1, kernel_size, kernel_size)
    return kernel_2d.repeat(channels, 1, 1, 1)


def video_psnr(pred: torch.Tensor, target: torch.Tensor, data_range: float = 1.0, eps: float = 1e-8) -> float:
    """
    Compute average PSNR over all frames.
    Expects `pred` and `target` in shape [3, T, H, W] with values in [0, 1].
    """
    if pred.shape != target.shape:
        raise ValueError(f"Shape mismatch: pred={tuple(pred.shape)} target={tuple(target.shape)}")

    pred = pred.float()
    target = target.float()
    mse = (pred - target).pow(2).mean(dim=(0, 2, 3))  # [T]
    psnr = 10.0 * torch.log10((data_range * data_range) / (mse + eps))
    return float(psnr.mean().item())


def video_ssim(
    pred: torch.Tensor,
    target: torch.Tensor,
    data_range: float = 1.0,
    kernel_size: int = 11,
    sigma: float = 1.5,
    k1: float = 0.01,
    k2: float = 0.03,
) -> float:
    """
    Compute average SSIM over all frames.
    Expects `pred` and `target` in shape [3, T, H, W] with values in [0, 1].
    """
    if pred.shape != target.shape:
        raise ValueError(f"Shape mismatch: pred={tuple(pred.shape)} target={tuple(target.shape)}")
    if pred.ndim != 4 or pred.shape[0] != 3:
        raise ValueError(f"Expected [3, T, H, W], got {tuple(pred.shape)}")
    if kernel_size % 2 == 0:
        raise ValueError("`kernel_size` must be odd.")

    pred = pred.float().permute(1, 0, 2, 3).contiguous()  # [T, 3, H, W]
    target = target.float().permute(1, 0, 2, 3).contiguous()  # [T, 3, H, W]
    channels = pred.shape[1]
    kernel = _gaussian_kernel_2d(
        kernel_size=kernel_size,
        sigma=sigma,
        channels=channels,
        device=pred.device,
        dtype=pred.dtype,
    )

    pad = kernel_size // 2
    mu_x = F.conv2d(pred, kernel, padding=pad, groups=channels)
    mu_y = F.conv2d(target, kernel, padding=pad, groups=channels)

    mu_x2 = mu_x * mu_x
    mu_y2 = mu_y * mu_y
    mu_xy = mu_x * mu_y

    sigma_x2 = F.conv2d(pred * pred, kernel, padding=pad, groups=channels) - mu_x2
    sigma_y2 = F.conv2d(target * target, kernel, padding=pad, groups=channels) - mu_y2
    sigma_xy = F.conv2d(pred * target, kernel, padding=pad, groups=channels) - mu_xy

    c1 = (k1 * data_range) ** 2
    c2 = (k2 * data_range) ** 2

    numerator = (2.0 * mu_xy + c1) * (2.0 * sigma_xy + c2)
    denominator = (mu_x2 + mu_y2 + c1) * (sigma_x2 + sigma_y2 + c2)
    ssim_map = numerator / (denominator + 1e-12)
    return float(ssim_map.mean().item())
