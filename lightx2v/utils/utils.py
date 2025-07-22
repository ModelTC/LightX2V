import os
import random
import subprocess
from typing import Optional

from einops import rearrange
import imageio
import imageio_ffmpeg as ffmpeg
from loguru import logger
import numpy as np
import torch
import torchvision


def seed_all(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def save_videos_grid(videos: torch.Tensor, path: str, rescale=False, n_rows=1, fps=24):
    """save videos by video tensor
       copy from https://github.com/guoyww/AnimateDiff/blob/e92bd5671ba62c0d774a32951453e328018b7c5b/animatediff/utils/util.py#L61

    Args:
        videos (torch.Tensor): video tensor predicted by the model
        path (str): path to save video
        rescale (bool, optional): rescale the video tensor from [-1, 1] to  . Defaults to False.
        n_rows (int, optional): Defaults to 1.
        fps (int, optional): video save fps. Defaults to 8.
    """
    videos = rearrange(videos, "b c t h w -> t b c h w")
    outputs = []
    for x in videos:
        x = torchvision.utils.make_grid(x, nrow=n_rows)
        x = x.transpose(0, 1).transpose(1, 2).squeeze(-1)
        if rescale:
            x = (x + 1.0) / 2.0  # -1,1 -> 0,1
        x = torch.clamp(x, 0, 1)
        x = (x * 255).numpy().astype(np.uint8)
        outputs.append(x)

    os.makedirs(os.path.dirname(path), exist_ok=True)
    imageio.mimsave(path, outputs, fps=fps)


def cache_video(
    tensor,
    save_file: str,
    fps=30,
    suffix=".mp4",
    nrow=8,
    normalize=True,
    value_range=(-1, 1),
    retry=5,
):
    save_dir = os.path.dirname(save_file)
    try:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)
    except Exception as e:
        logger.error(f"Failed to create directory: {save_dir}, error: {e}")
        return None

    cache_file = save_file

    # save to cache
    error = None
    for _ in range(retry):
        try:
            # preprocess
            tensor = tensor.clamp(min(value_range), max(value_range))  # type: ignore
            tensor = torch.stack(
                [torchvision.utils.make_grid(u, nrow=nrow, normalize=normalize, value_range=value_range) for u in tensor.unbind(2)],
                dim=1,
            ).permute(1, 2, 3, 0)
            tensor = (tensor * 255).type(torch.uint8).cpu()

            # write video
            writer = imageio.get_writer(cache_file, fps=fps, codec="libx264", quality=8)
            for frame in tensor.numpy():
                writer.append_data(frame)
            writer.close()
            del tensor
            torch.cuda.empty_cache()
            return cache_file
        except Exception as e:
            error = e
            continue
    else:
        logger.info(f"cache_video failed, error: {error}", flush=True)
        return None


def vae_to_comfyui_image(vae_output: torch.Tensor) -> torch.Tensor:
    """
    Convert VAE decoder output to ComfyUI Image format

    Args:
        vae_output: VAE decoder output tensor, typically in range [-1, 1]
                    Shape: [B, C, T, H, W] or [B, C, H, W]

    Returns:
        ComfyUI Image tensor in range [0, 1]
        Shape: [B, H, W, C] for single frame or [B*T, H, W, C] for video
    """
    # Handle video tensor (5D) vs image tensor (4D)
    if vae_output.dim() == 5:
        # Video tensor: [B, C, T, H, W]
        B, C, T, H, W = vae_output.shape
        # Reshape to [B*T, C, H, W] for processing
        vae_output = vae_output.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)

    # Normalize from [-1, 1] to [0, 1]
    images = (vae_output + 1) / 2

    # Clamp values to [0, 1]
    images = torch.clamp(images, 0, 1)

    # Convert from [B, C, H, W] to [B, H, W, C]
    images = images.permute(0, 2, 3, 1).cpu()

    return images


def save_to_video(
    images: torch.Tensor,
    output_path: str,
    fps: float = 24.0,
    method: str = "imageio",
    lossless: bool = False,
    output_pix_fmt: Optional[str] = "yuv420p",
) -> None:
    """
    Save ComfyUI Image tensor to video file

    Args:
        images: ComfyUI Image tensor [N, H, W, C] in range [0, 1]
        output_path: Path to save the video
        fps: Frames per second
        method: Save method - "imageio" or "ffmpeg"
        lossless: Whether to use lossless encoding (ffmpeg method only)
        output_pix_fmt: Pixel format for output (ffmpeg method only)
    """
    assert images.dim() == 4 and images.shape[-1] == 3, "Input must be [N, H, W, C] with C=3"

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    if method == "imageio":
        # Convert to uint8
        frames = (images * 255).cpu().numpy().astype(np.uint8)
        imageio.mimsave(output_path, frames, fps=fps)  # type: ignore

    elif method == "ffmpeg":
        # Convert to numpy and scale to [0, 255]
        frames = (images * 255).cpu().numpy().clip(0, 255).astype(np.uint8)

        # Convert RGB to BGR for OpenCV/FFmpeg
        frames = frames[..., ::-1].copy()

        N, height, width, _ = frames.shape

        # Ensure even dimensions for x264
        width += width % 2
        height += height % 2

        # Get ffmpeg executable from imageio_ffmpeg
        ffmpeg_exe = ffmpeg.get_ffmpeg_exe()

        if lossless:
            command = [
                ffmpeg_exe,
                "-y",  # Overwrite output file if it exists
                "-f",
                "rawvideo",
                "-s",
                f"{int(width)}x{int(height)}",
                "-pix_fmt",
                "bgr24",
                "-r",
                f"{fps}",
                "-loglevel",
                "error",
                "-threads",
                "4",
                "-i",
                "-",  # Input from pipe
                "-vcodec",
                "libx264rgb",
                "-crf",
                "0",
                "-an",  # No audio
                output_path,
            ]
        else:
            command = [
                ffmpeg_exe,
                "-y",  # Overwrite output file if it exists
                "-f",
                "rawvideo",
                "-s",
                f"{int(width)}x{int(height)}",
                "-pix_fmt",
                "bgr24",
                "-r",
                f"{fps}",
                "-loglevel",
                "error",
                "-threads",
                "4",
                "-i",
                "-",  # Input from pipe
                "-vcodec",
                "libx264",
                "-pix_fmt",
                output_pix_fmt,
                "-an",  # No audio
                output_path,
            ]

        # Run FFmpeg
        process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        if process.stdin is None:
            raise BrokenPipeError("No stdin buffer received.")

        # Write frames to FFmpeg
        for frame in frames:
            # Pad frame if needed
            if frame.shape[0] < height or frame.shape[1] < width:
                padded = np.zeros((height, width, 3), dtype=np.uint8)
                padded[: frame.shape[0], : frame.shape[1]] = frame
                frame = padded
            process.stdin.write(frame.tobytes())

        process.stdin.close()
        process.wait()

        if process.returncode != 0:
            error_output = process.stderr.read().decode() if process.stderr else "Unknown error"
            raise RuntimeError(f"FFmpeg failed with error: {error_output}")

    else:
        raise ValueError(f"Unknown save method: {method}")
