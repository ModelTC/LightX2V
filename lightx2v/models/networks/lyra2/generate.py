# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Lyra-2 ZoomGS AR generation helpers.

Extracted from lyra_2/_src/inference/lyra2_zoomgs_inference.py:
  - _camera_centers_from_w2c
  - _correct_trajectory_ground_parallel
  - _generate_one_direction
"""

from __future__ import annotations

import torch
from loguru import logger

from lightx2v.models.networks.lyra2.camera_traj_utils import build_camera_trajectory
from lightx2v.models.networks.lyra2.lyra2_ar_inference import run_lyra2_sample, safe_to


def _camera_centers_from_w2c(w2c: torch.Tensor) -> torch.Tensor:
    R = w2c[:, :3, :3]
    t = w2c[:, :3, 3]
    return -(R.transpose(1, 2) @ t.unsqueeze(-1)).squeeze(-1)


def _correct_trajectory_ground_parallel(
    w2cs_T_44: torch.Tensor,
    ground_normal_cam: torch.Tensor,
) -> torch.Tensor:
    """Re-project w2c translations so camera moves parallel to the ground plane.

    The original trajectory's translation direction (typically camera z-axis) is
    projected onto the ground plane, preserving the total displacement magnitude.
    Camera orientation (rotation) is kept unchanged.
    """
    T = w2cs_T_44.shape[0]
    n = ground_normal_cam.to(w2cs_T_44.device, dtype=w2cs_T_44.dtype)

    t0 = w2cs_T_44[0, :3, 3]
    displacements = w2cs_T_44[:, :3, 3] - t0.unsqueeze(0)  # (T, 3)

    # Project each displacement onto the ground plane
    n_dot_d = (displacements * n.unsqueeze(0)).sum(dim=-1, keepdim=True)  # (T, 1)
    projected = displacements - n_dot_d * n.unsqueeze(0)  # (T, 3)

    # Preserve original displacement magnitudes
    orig_norms = displacements.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    proj_norms = projected.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    projected = projected * (orig_norms / proj_norms)
    # First frame stays at origin (no displacement)
    projected[0] = 0.0

    corrected = w2cs_T_44.clone()
    corrected[:, :3, 3] = t0.unsqueeze(0) + projected
    return corrected


def _generate_one_direction(
    *,
    model,
    args,
    img_bchw: torch.Tensor,
    depth_hw: torch.Tensor,
    mask_hw: torch.Tensor,
    K_33: torch.Tensor,
    t5_embeddings: torch.Tensor,
    neg_t5_embeddings: torch.Tensor,
    trajectory: str,
    direction: str,
    strength: float,
    N: int,
    da3_model=None,
    process_group=None,
    log_prefix: str = "",
    ground_normal_cam: torch.Tensor | None = None,
    upward_shift: float = 0.0,
    zoom_out_upward_ratio: float = 0.0,
) -> dict | None:
    """Run AR spatial inference for a single camera trajectory direction."""
    device = model.tensor_kwargs.get("device", None)
    H, W = img_bchw.shape[-2:]

    initial_w2c = torch.eye(4, dtype=torch.float32, device=device)
    center_depth = torch.quantile(depth_hw[mask_hw > 0.5], 0.25)

    w2cs_T_44, Ks_T_33 = build_camera_trajectory(
        initial_w2c,
        K_33.to(initial_w2c),
        center_depth,
        N,
        trajectory,
        direction,
        strength,
    )

    if zoom_out_upward_ratio > 0.0:
        cam_centers = _camera_centers_from_w2c(w2cs_T_44)
        z_disp = cam_centers[:, 2] - cam_centers[0, 2]
        backward_amount = (-z_disp).clamp(min=0)
        upward_amount = backward_amount * zoom_out_upward_ratio
        cam_centers_shifted = cam_centers.clone()
        cam_centers_shifted[:, 1] -= upward_amount
        R = w2cs_T_44[:, :3, :3]
        new_t = -(R @ cam_centers_shifted.unsqueeze(-1)).squeeze(-1)
        w2cs_T_44 = w2cs_T_44.clone()
        w2cs_T_44[:, :3, 3] = new_t
        logger.info(f"{log_prefix} [upward_tilt] Added upward ratio={zoom_out_upward_ratio:.3f}, max_upward={upward_amount.max().item():.4f}")

    if ground_normal_cam is not None:
        w2cs_T_44 = _correct_trajectory_ground_parallel(w2cs_T_44, ground_normal_cam)

        if upward_shift > 0.0:
            n = ground_normal_cam.to(w2cs_T_44.device, dtype=w2cs_T_44.dtype)
            T = w2cs_T_44.shape[0]
            ramp = torch.linspace(0, upward_shift, T, device=w2cs_T_44.device, dtype=w2cs_T_44.dtype)
            w2cs_T_44 = w2cs_T_44.clone()
            w2cs_T_44[:, :3, 3] -= ramp.unsqueeze(-1) * n.unsqueeze(0)

    w2cs_b_t_44 = w2cs_T_44.unsqueeze(0).to(dtype=torch.float32)
    Ks_b_t_33 = Ks_T_33.unsqueeze(0).to(dtype=torch.float32)

    depth_b_thw = depth_hw.unsqueeze(0).unsqueeze(0).repeat(1, N, 1, 1).to(device=device)

    data_batch = {
        "video": img_bchw.unsqueeze(2),
        "t5_text_embeddings": t5_embeddings,
        "neg_t5_text_embeddings": neg_t5_embeddings,
        "fps": torch.tensor([args.fps], dtype=torch.int32, device=device),
        "padding_mask": torch.zeros((1, 1, H, W), dtype=model.tensor_kwargs["dtype"], device=device),
        "is_preprocessed": torch.tensor([True], dtype=torch.bool, device=device),
        "camera_w2c": w2cs_b_t_44,
        "intrinsics": Ks_b_t_33,
        "depth": depth_b_thw,
    }

    skip_keys = {"camera_w2c", "intrinsics", "depth"}
    data_batch = safe_to(
        data_batch,
        device=model.tensor_kwargs.get("device", None),
        dtype=model.tensor_kwargs.get("dtype", None),
        skip_keys=skip_keys,
    )

    saved_num_frames = args.num_frames
    args.num_frames = N
    try:
        result = run_lyra2_sample(
            model,
            data_batch,
            args,
            process_group=process_group,
            da3_model=da3_model,
            show_progress=True,
            log_prefix=log_prefix,
        )
    finally:
        args.num_frames = saved_num_frames

    return result
