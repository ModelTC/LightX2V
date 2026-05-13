# -----------------------------------------------------------------------------
# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# -----------------------------------------------------------------------------

import argparse
from typing import Optional, Tuple

import torch

from lightx2v.models.networks.lyra2.camera_utils import (  # type: ignore
    create_back_trajectory,
    create_dolly_zoom_trajectory,
    create_horizontal_trajectory,
    create_horizontal_with_noise_trajectory,
    create_horizontal_zoom_with_bend_trajectory,
    create_horizontal_zoom_with_noise_and_bend_trajectory,
    create_orbit_trajectory,
    create_rotate_spot_trajectory,
    create_rotate_spot_with_noise_trajectory,
    create_rotate_then_zoom_trajectory,
    create_spiral_horizontal_trajectory,
    create_spiral_trajectory,
)

# Canonical list of supported camera trajectories shared across inference scripts.
CAMERA_TRAJECTORY_CHOICES: Tuple[str, ...] = (
    "original",
    "spiral",
    "spiral_center",
    "spiral_outwards",
    "horizontal",
    "horizontal_noise",
    "horizontal_lift",
    "horizontal_lift_noise",
    "horizontal_zoom",
    "horizontal_zoom_noise",
    "horizontal_zoom_bend",
    "horizontal_zoom_noise_bend",
    "horizontal_zoom_still",
    "horizontal_still",
    "horizontal_simple",
    "vertical_simple",
    "horizontal_outward",
    "back",
    "back_simple",
    "dolly_zoom",
    "horizontal_spiral",
    "orbit_horizontal",
    "orbit_vertical",
    "rotate_zoom_in",
    "rotate_zoom_out",
    "rotate_spot",
    "rotate_spot_noise",
)


def add_camera_traj_args(
    parser: argparse.ArgumentParser,
    *,
    with_video_len: bool = True,
    video_len_flag: str = "video_len",
    video_len_default: int = 93,
    video_len_help: Optional[str] = None,
    with_fps: bool = True,
    fps_default: int = 16,
    trajectory_default: str = "original",
    strength_default: float = 0.2,
) -> None:
    """Attach shared camera trajectory CLI arguments to an argparse parser."""
    if with_video_len:
        help_text = video_len_help if video_len_help is not None else "Video length (number of frames) for camera trajectory."
        parser.add_argument(
            f"--{video_len_flag}",
            type=int,
            default=video_len_default,
            help=help_text,
        )
    if with_fps:
        parser.add_argument(
            "--fps",
            type=int,
            default=fps_default,
            help="Output video frame rate.",
        )
    parser.add_argument(
        "--trajectory",
        type=str,
        default=trajectory_default,
        choices=list(CAMERA_TRAJECTORY_CHOICES),
    )
    parser.add_argument(
        "--direction",
        type=str,
        default="right",
        choices=["left", "right", "up", "down"],
    )
    parser.add_argument(
        "--strength",
        type=float,
        default=strength_default,
    )


def build_camera_trajectory(
    initial_w2c_44: torch.Tensor,
    K_33: torch.Tensor,
    center_depth: float,
    video_len: int,
    trajectory: str,
    direction: str,
    strength: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Shared camera-trajectory builder.

    Returns:
        w2cs: [T,4,4] world-to-camera matrices
        Ks:  [T,3,3] intrinsics per frame
    """
    device = initial_w2c_44.device
    if trajectory == "original":
        w2cs = initial_w2c_44.unsqueeze(0).repeat(video_len, 1, 1)
        Ks = K_33.unsqueeze(0).repeat(video_len, 1, 1)
        return w2cs, Ks

    if trajectory == "spiral":
        radius_x = 0.15 * strength
        radius_y = 0.10 * strength
        new_w2cs = create_spiral_trajectory(
            initial_w2c_44,
            center_depth,
            n_steps=video_len,
            radius_x=radius_x,
            radius_y=radius_y,
            right=(direction == "right"),
            num_circles=2,
        )
        Ks = K_33.unsqueeze(0).repeat(new_w2cs.shape[0], 1, 1)
        return new_w2cs, Ks

    if trajectory == "spiral_center":
        new_w2cs = create_spiral_trajectory(
            initial_w2c_44,
            center_depth,
            n_steps=video_len,
            radius_x=0.03,
            radius_y=0.02,
            right=(direction == "right"),
            start_from_zero=False,
        )
        Ks = K_33.unsqueeze(0).repeat(new_w2cs.shape[0], 1, 1)
        return new_w2cs, Ks

    if trajectory == "spiral_outwards":
        radius_x = 0.3 * strength
        radius_y = 0.2 * strength
        new_w2cs = create_spiral_trajectory(
            initial_w2c_44,
            center_depth,
            n_steps=video_len,
            inwards=False,
            radius_x=radius_x,
            radius_y=radius_y,
            right=(direction == "right"),
        )
        Ks = K_33.unsqueeze(0).repeat(new_w2cs.shape[0], 1, 1)
        return new_w2cs, Ks

    if trajectory == "horizontal":
        new_w2cs = create_horizontal_trajectory(
            initial_w2c_44,
            center_depth,
            n_steps=video_len,
            right=(direction == "right"),
            axis="x",
            distance=strength,
        )
        Ks = K_33.unsqueeze(0).repeat(new_w2cs.shape[0], 1, 1)
        return new_w2cs, Ks

    if trajectory == "horizontal_noise":
        new_w2cs = create_horizontal_with_noise_trajectory(
            initial_w2c_44,
            center_depth,
            n_steps=video_len,
            right=(direction == "right"),
            axis="x",
            distance=strength,
        )
        Ks = K_33.unsqueeze(0).repeat(new_w2cs.shape[0], 1, 1)
        return new_w2cs, Ks

    if trajectory == "horizontal_lift":
        new_w2cs = create_horizontal_trajectory(
            initial_w2c_44,
            center_depth,
            n_steps=video_len,
            right=(direction == "right"),
            axis="y",
            distance=strength,
        )
        Ks = K_33.unsqueeze(0).repeat(new_w2cs.shape[0], 1, 1)
        return new_w2cs, Ks

    if trajectory == "horizontal_lift_noise":
        new_w2cs = create_horizontal_with_noise_trajectory(
            initial_w2c_44,
            center_depth,
            n_steps=video_len,
            right=(direction == "right"),
            axis="y",
            distance=strength,
        )
        Ks = K_33.unsqueeze(0).repeat(new_w2cs.shape[0], 1, 1)
        return new_w2cs, Ks

    if trajectory == "horizontal_zoom":
        new_w2cs = create_horizontal_trajectory(
            initial_w2c_44,
            center_depth,
            n_steps=video_len,
            right=(direction == "right"),
            axis="z",
            distance=strength,
        )
        Ks = K_33.unsqueeze(0).repeat(new_w2cs.shape[0], 1, 1)
        return new_w2cs, Ks

    if trajectory == "horizontal_zoom_noise":
        new_w2cs = create_horizontal_with_noise_trajectory(
            initial_w2c_44,
            center_depth,
            n_steps=video_len,
            right=(direction == "right"),
            axis="z",
            distance=strength,
        )
        Ks = K_33.unsqueeze(0).repeat(new_w2cs.shape[0], 1, 1)
        return new_w2cs, Ks

    if trajectory == "horizontal_zoom_bend":
        new_w2cs = create_horizontal_zoom_with_bend_trajectory(
            initial_w2c_44,
            center_depth,
            n_steps=video_len,
            right=(direction == "right"),
            axis="z",
            distance=strength,
        )
        Ks = K_33.unsqueeze(0).repeat(new_w2cs.shape[0], 1, 1)
        return new_w2cs, Ks

    if trajectory == "horizontal_zoom_noise_bend":
        new_w2cs = create_horizontal_zoom_with_noise_and_bend_trajectory(
            initial_w2c_44,
            center_depth,
            n_steps=video_len,
            right=(direction == "right"),
            axis="z",
            distance=strength,
        )
        Ks = K_33.unsqueeze(0).repeat(new_w2cs.shape[0], 1, 1)
        return new_w2cs, Ks

    if trajectory == "horizontal_spiral":
        radius_x = 0.15 * strength * 0.25 * 0.5
        radius_y = 0.10 * strength * 0.25 * 0.5
        num_circles = 2
        new_w2cs = create_spiral_horizontal_trajectory(
            initial_w2c_44,
            center_depth,
            n_steps=video_len,
            right=(direction == "right"),
            axis="z",
            distance=strength,
            radius_x=radius_x,
            radius_y=radius_y,
            num_circles=num_circles,
        )
        Ks = K_33.unsqueeze(0).repeat(new_w2cs.shape[0], 1, 1)
        return new_w2cs, Ks

    if trajectory == "horizontal_zoom_still":
        half = video_len // 2
        seq1 = create_horizontal_trajectory(
            initial_w2c_44,
            center_depth,
            n_steps=half,
            right=(direction == "right"),
            axis="z",
            distance=strength,
        )
        seq2 = seq1[-1:].repeat(video_len - seq1.shape[0], 1, 1)
        new_w2cs = torch.cat([seq1, seq2], dim=0)
        Ks = K_33.unsqueeze(0).repeat(new_w2cs.shape[0], 1, 1)
        return new_w2cs, Ks

    if trajectory == "horizontal_still":
        half = video_len // 2
        seq1 = create_horizontal_trajectory(
            initial_w2c_44,
            center_depth,
            n_steps=half,
            right=(direction == "right"),
            axis="x",
            distance=strength,
        )
        seq2 = seq1[-1:].repeat(video_len - seq1.shape[0], 1, 1)
        new_w2cs = torch.cat([seq1, seq2], dim=0)
        Ks = K_33.unsqueeze(0).repeat(new_w2cs.shape[0], 1, 1)
        return new_w2cs, Ks

    if trajectory == "horizontal_simple":
        new_w2cs = initial_w2c_44.unsqueeze(0).repeat(video_len, 1, 1)
        shift = torch.linspace(0.0, strength, video_len, device=device) * center_depth
        if direction == "right":
            new_w2cs[:, 0, 3] -= shift
        else:
            new_w2cs[:, 0, 3] += shift
        Ks = K_33.unsqueeze(0).repeat(new_w2cs.shape[0], 1, 1)
        return new_w2cs, Ks

    if trajectory == "vertical_simple":
        new_w2cs = initial_w2c_44.unsqueeze(0).repeat(video_len, 1, 1)
        shift = torch.linspace(0.0, strength, video_len, device=device) * center_depth
        if direction == "up":
            new_w2cs[:, 1, 3] += shift
        else:
            new_w2cs[:, 1, 3] -= shift
        Ks = K_33.unsqueeze(0).repeat(new_w2cs.shape[0], 1, 1)
        return new_w2cs, Ks

    if trajectory == "horizontal_outward":
        new_w2cs = create_horizontal_trajectory(
            initial_w2c_44,
            center_depth,
            n_steps=video_len,
            right=(direction == "right"),
            axis="x",
            distance=strength,
            outwards=True,
        )
        Ks = K_33.unsqueeze(0).repeat(new_w2cs.shape[0], 1, 1)
        return new_w2cs, Ks

    if trajectory == "back":
        seq = create_back_trajectory(
            initial_w2c_44.unsqueeze(0).repeat(video_len, 1, 1),
            center_depth,
            right=(direction == "right"),
        )
        Ks = K_33.unsqueeze(0).repeat(seq.shape[0], 1, 1)
        return seq, Ks

    if trajectory == "back_simple":
        seq = create_back_trajectory(
            initial_w2c_44.unsqueeze(0).repeat(video_len, 1, 1),
            center_depth,
            right=(direction == "right"),
            invert_pos=True,
            radius_x=0.15,
            radius_y=0.1,
        )
        Ks = K_33.unsqueeze(0).repeat(seq.shape[0], 1, 1)
        return seq, Ks

    if trajectory == "dolly_zoom":
        seq, Ks = create_dolly_zoom_trajectory(initial_w2c_44, K_33, center_depth, n_steps=video_len)
        return seq, Ks

    if trajectory == "orbit_horizontal":
        new_w2cs = create_orbit_trajectory(
            initial_w2c_44,
            center_depth,
            n_steps=video_len,
            angle=strength,
            axis="y",
            direction=direction,
        )
        Ks = K_33.unsqueeze(0).repeat(new_w2cs.shape[0], 1, 1)
        return new_w2cs, Ks

    if trajectory == "orbit_vertical":
        new_w2cs = create_orbit_trajectory(
            initial_w2c_44,
            center_depth,
            n_steps=video_len,
            angle=strength,
            axis="x",
            direction=direction,
        )
        Ks = K_33.unsqueeze(0).repeat(new_w2cs.shape[0], 1, 1)
        return new_w2cs, Ks

    if trajectory == "rotate_zoom_in":
        new_w2cs = create_rotate_then_zoom_trajectory(
            initial_w2c_44,
            center_depth,
            n_steps=video_len,
            rotate_direction=direction,
            zoom_direction="right",
            rotation_angle=20.0,
            zoom_distance=strength,
        )
        Ks = K_33.unsqueeze(0).repeat(new_w2cs.shape[0], 1, 1)
        return new_w2cs, Ks

    if trajectory == "rotate_zoom_out":
        new_w2cs = create_rotate_then_zoom_trajectory(
            initial_w2c_44,
            center_depth,
            n_steps=video_len,
            rotate_direction=direction,
            zoom_direction="left",
            rotation_angle=20.0,
            zoom_distance=strength,
        )
        Ks = K_33.unsqueeze(0).repeat(new_w2cs.shape[0], 1, 1)
        return new_w2cs, Ks

    if trajectory == "rotate_spot":
        new_w2cs = create_rotate_spot_trajectory(
            initial_w2c_44,
            center_depth,
            n_steps=video_len,
            rotate_direction=direction,
            rotation_angle=1.0 * strength,
        )
        Ks = K_33.unsqueeze(0).repeat(new_w2cs.shape[0], 1, 1)
        return new_w2cs, Ks

    if trajectory == "rotate_spot_noise":
        new_w2cs = create_rotate_spot_with_noise_trajectory(
            initial_w2c_44,
            center_depth,
            n_steps=video_len,
            rotate_direction=direction,
            rotation_angle=1.0 * strength,
        )
        Ks = K_33.unsqueeze(0).repeat(new_w2cs.shape[0], 1, 1)
        return new_w2cs, Ks

    raise NotImplementedError(f"Unsupported trajectory: {trajectory}")
