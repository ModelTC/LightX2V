import os
from typing import List, Optional, Tuple

import torch
from loguru import logger
from torch.nn import functional as F

from lightx2v.utils.profiler import ProfilingContext


class RIFEWrapper:
    """Wrapper for RIFE model to work with ComfyUI Image tensors"""

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    def __init__(self, model_path, device: Optional[torch.device] = None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Setup torch for optimal performance
        torch.set_grad_enabled(False)
        if torch.cuda.is_available():
            torch.backends.cudnn.enabled = True
            torch.backends.cudnn.benchmark = True

        # Load model
        from .train_log.RIFE_HDv3 import Model

        self.model = Model()
        with ProfilingContext("Load RIFE model"):
            self.model.load_model(model_path, -1)
            self.model.eval()
            self.model.device()

    @ProfilingContext("Interpolate frames")
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

        if source_fps == target_fps:
            return images

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
