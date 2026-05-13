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

"""Single-image depth encoding utilities for Lyra-2 ZoomGS.

Extracted from lyra_2/_src/inference/lyra2_zoomgs_inference.py:
  - _da3_infer_depth_intrinsics_single
  - _fit_ground_normal_from_depth
"""

from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from loguru import logger


def _da3_infer_depth_intrinsics_single(
    da3_model,
    img_rgb_uint8: torch.Tensor,
    target_hw: Tuple[int, int],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """DA3 single-image depth: RGB uint8 HWC -> (image_chw01, depth_hw, K_33, mask_hw)."""
    Ht, Wt = target_hw
    img_np = img_rgb_uint8.detach().cpu().numpy()
    img_resized = cv2.resize(img_np, (Wt, Ht), interpolation=cv2.INTER_LINEAR)

    image_chw01 = torch.from_numpy(img_resized.astype(np.float32) / 255.0)
    image_chw01 = image_chw01.permute(2, 0, 1).unsqueeze(0).contiguous()

    images = [img_resized.astype(np.uint8)]
    prediction = da3_model.inference(
        image=images,
        extrinsics=None,
        intrinsics=None,
        align_to_input_ext_scale=True,
        infer_gs=False,
        process_res=int(max(Ht, Wt)),
        process_res_method="upper_bound_resize",
        export_dir=None,
        export_format="mini_npz",
    )

    depths_np = getattr(prediction, "depth", None)
    if depths_np is None:
        raise RuntimeError("DA3 prediction has no 'depth' field.")
    if isinstance(depths_np, torch.Tensor):
        depth_np = depths_np[0].detach().cpu().numpy()
    else:
        depth_np = np.asarray(depths_np)[0]
    Hd, Wd = depth_np.shape[-2:]

    depth_t = torch.from_numpy(depth_np.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    if (Hd, Wd) != (Ht, Wt):
        depth_t = F.interpolate(depth_t, size=(Ht, Wt), mode="bilinear", align_corners=False)
    depth_hw = depth_t[0, 0]
    depth_hw = torch.nan_to_num(depth_hw, nan=1e4).clamp(min=0, max=1e4)
    mask_hw = (depth_hw < 999.9).to(dtype=torch.float32)

    try:
        ixts_np = getattr(prediction, "intrinsics", None)
        if ixts_np is None:
            raise AttributeError
        if isinstance(ixts_np, torch.Tensor):
            K_np = ixts_np[0].detach().cpu().numpy()
        else:
            K_np = np.asarray(ixts_np)[0]
        K_33 = torch.from_numpy(K_np.astype(np.float32))
        scale_x = float(Wt) / float(Wd)
        scale_y = float(Ht) / float(Hd)
        K_33 = K_33.clone()
        K_33[0, 0] *= scale_x
        K_33[1, 1] *= scale_y
        K_33[0, 2] *= scale_x
        K_33[1, 2] *= scale_y
    except Exception:
        fx = fy = max(Ht, Wt) * 1.5
        cx, cy = Wt / 2.0, Ht / 2.0
        K_33 = torch.tensor([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=torch.float32)

    return image_chw01, depth_hw, K_33, mask_hw


def _fit_ground_normal_from_depth(
    depth_hw: torch.Tensor,
    K_33: torch.Tensor,
    mask_hw: torch.Tensor,
    bottom_frac: float = 0.4,
    ransac_iters: int = 200,
    ransac_thresh: float = 0.05,
) -> torch.Tensor | None:
    """Fit a ground plane from the bottom portion of the depth map.

    Returns the plane normal in camera space (pointing 'up' away from ground),
    or None if fitting fails.
    """
    H, W = depth_hw.shape
    y_start = int(H * (1.0 - bottom_frac))

    valid = (mask_hw[y_start:] > 0.5) & (depth_hw[y_start:] > 0.01) & (depth_hw[y_start:] < 500.0)
    if valid.sum() < 50:
        return None

    ys, xs = torch.where(valid)
    ys = ys + y_start
    depths = depth_hw[ys, xs]

    fx, fy = K_33[0, 0], K_33[1, 1]
    cx, cy = K_33[0, 2], K_33[1, 2]
    X = (xs.float() - cx) / fx * depths
    Y = (ys.float() - cy) / fy * depths
    Z = depths
    pts = torch.stack([X, Y, Z], dim=-1)  # (N, 3)

    N_pts = pts.shape[0]
    best_normal = None
    best_inliers = 0

    for _ in range(ransac_iters):
        idx = torch.randint(0, N_pts, (3,))
        p0, p1, p2 = pts[idx[0]], pts[idx[1]], pts[idx[2]]
        v1 = p1 - p0
        v2 = p2 - p0
        n = torch.cross(v1, v2, dim=0)
        norm = n.norm()
        if norm < 1e-8:
            continue
        n = n / norm
        d = -torch.dot(n, p0)
        dists = (pts @ n + d).abs()
        inlier_count = (dists < ransac_thresh * Z.abs().clamp(min=0.1)).sum().item()
        if inlier_count > best_inliers:
            best_inliers = inlier_count
            best_normal = n

    if best_normal is None:
        return None

    # Ensure normal points "up" in camera space (negative y direction = up in image coords)
    if best_normal[1] > 0:
        best_normal = -best_normal

    logger.info(f"[ground_plane] Fitted normal: [{best_normal[0]:.4f}, {best_normal[1]:.4f}, {best_normal[2]:.4f}], inliers: {best_inliers}/{N_pts}")
    return best_normal
