import os
import torch
import numpy as np
import subprocess
import imageio
import torchvision
from torch.nn import functional as F
from einops import rearrange
from typing import Tuple, Optional, List
import imageio_ffmpeg as ffmpeg


class VAEDecoderConvertor:
    """Convert VAE decoder output to ComfyUI Image format"""

    @staticmethod
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


class RIFEWrapper:
    """Wrapper for RIFE model to work with ComfyUI Image tensors"""

    def __init__(self, model_dir: str = "train_log", device: Optional[torch.device] = None):
        """
        Initialize RIFE wrapper

        Args:
            model_dir: Directory containing trained model files
            device: Torch device (cuda/cpu). If None, will auto-detect
        """
        self.model_dir = model_dir
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Setup torch for optimal performance
        torch.set_grad_enabled(False)
        if torch.cuda.is_available():
            torch.backends.cudnn.enabled = True
            torch.backends.cudnn.benchmark = True

        # Load model
        from train_log.RIFE_HDv3 import Model

        self.model = Model()
        self.model.load_model(model_dir, -1)
        self.model.eval()
        self.model.device()

    def interpolate_frames(
        self,
        images: torch.Tensor,
        source_fps: float,
        target_fps: float,
        scale: float = 1.0,
    ) -> torch.Tensor:
        """
        Interpolate frames from source FPS to target FPS

        Args:
            images: ComfyUI Image tensor [N, H, W, C] in range [0, 1]
            source_fps: Source frame rate
            target_fps: Target frame rate
            scale: Scale factor for processing

        Returns:
            Interpolated ComfyUI Image tensor [M, H, W, C] in range [0, 1]
        """
        # Validate input
        assert images.dim() == 4 and images.shape[-1] == 3, "Input must be [N, H, W, C] with C=3"

        total_source_frames = images.shape[0]
        height, width = images.shape[1:3]

        # Calculate padding for model
        tmp = max(128, int(128 / scale))
        ph = ((height - 1) // tmp + 1) * tmp
        pw = ((width - 1) // tmp + 1) * tmp
        padding = (0, pw - width, 0, ph - height)

        # Calculate target frame positions
        frame_positions = self._calculate_target_frame_positions(source_fps, target_fps, total_source_frames)

        # Prepare output tensor
        output_frames = []

        for source_idx1, source_idx2, interp_factor in frame_positions:
            if interp_factor == 0.0 or source_idx1 == source_idx2:
                # No interpolation needed, use the source frame directly
                output_frames.append(images[source_idx1])
            else:
                # Get frames to interpolate
                frame1 = images[source_idx1]
                frame2 = images[source_idx2]

                # Convert ComfyUI format [H, W, C] to RIFE format [1, C, H, W]
                # Also convert from [0, 1] to [0, 1] (already in correct range)
                I0 = frame1.permute(2, 0, 1).unsqueeze(0).to(self.device)
                I1 = frame2.permute(2, 0, 1).unsqueeze(0).to(self.device)

                # Pad images
                I0 = F.pad(I0, padding)
                I1 = F.pad(I1, padding)

                # Perform interpolation
                with torch.no_grad():
                    interpolated = self.model.inference(I0, I1, timestep=interp_factor, scale=scale)

                # Convert back to ComfyUI format [H, W, C]
                # Crop to original size and permute dimensions
                interpolated_frame = interpolated[0, :, :height, :width].permute(1, 2, 0).cpu()
                output_frames.append(interpolated_frame)

        # Stack all frames
        return torch.stack(output_frames, dim=0)

    def _calculate_target_frame_positions(self, source_fps: float, target_fps: float, total_source_frames: int) -> List[Tuple[int, int, float]]:
        """
        Calculate which frames need to be generated for the target frame rate.

        Returns:
            List of (source_frame_index1, source_frame_index2, interpolation_factor) tuples
        """
        frame_positions = []

        # Calculate the time duration of the video
        duration = (total_source_frames - 1) / source_fps

        # Calculate number of target frames
        total_target_frames = int(duration * target_fps) + 1

        for target_idx in range(total_target_frames):
            # Calculate the time position of this target frame
            target_time = target_idx / target_fps

            # Calculate the corresponding position in source frames
            source_position = target_time * source_fps

            # Find the two source frames to interpolate between
            source_idx1 = int(source_position)
            source_idx2 = min(source_idx1 + 1, total_source_frames - 1)

            # Calculate interpolation factor (0 means use frame1, 1 means use frame2)
            if source_idx1 == source_idx2:
                interpolation_factor = 0.0
            else:
                interpolation_factor = source_position - source_idx1

            frame_positions.append((source_idx1, source_idx2, interpolation_factor))

        return frame_positions


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
        frames = (images * 255).numpy().astype(np.uint8)
        imageio.mimsave(output_path, frames, fps=fps)

    elif method == "ffmpeg":
        # Convert to numpy and scale to [0, 255]
        frames = (images * 255).numpy().astype(np.uint8)

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


# Alternative save function using torchvision's grid format
def save_videos_grid(
    videos: torch.Tensor,
    path: str,
    rescale: bool = False,
    n_rows: int = 1,
    fps: int = 24,
) -> None:
    """
    Save videos using torchvision grid format

    Args:
        videos: Video tensor [B, C, T, H, W]
        path: Output path
        rescale: Whether to rescale from [-1, 1] to [0, 1]
        n_rows: Number of rows in grid
        fps: Frames per second
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
