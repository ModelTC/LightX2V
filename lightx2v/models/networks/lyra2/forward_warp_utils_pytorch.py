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

from typing import List, Optional, Tuple, Union

import torch
from einops import rearrange

def get_max_exponent_for_dtype(dtype):
    # Set the maximum exponent based on dtype
    if dtype == torch.bfloat16:
        return 80.0  # Safe maximum exponent for bfloat16
    elif dtype == torch.float16:
        return 10.0  # Safe maximum exponent for float16
    elif dtype == torch.float32:
        return 80.0  # Safe maximum exponent for float32
    elif dtype == torch.float64:
        return 700.0  # Safe maximum exponent for float64
    else:
        return 80.0  # Default safe value

def inverse_with_conversion(mtx):
    return torch.linalg.inv(mtx.to(torch.float32)).to(mtx.dtype)


def reliable_depth_mask_range_batch(depth, window_size=5, ratio_thresh=0.05, eps=1e-6):
    assert window_size % 2 == 1, "Window size must be odd."
    if depth.dim() == 3:   # Input shape: (B, H, W)
        depth_unsq = depth.unsqueeze(1)
    elif depth.dim() == 4:  # Already has shape (B, 1, H, W)
        depth_unsq = depth
    else:
        raise ValueError("depth tensor must be of shape (B, H, W) or (B, 1, H, W)")

    local_max = torch.nn.functional.max_pool2d(depth_unsq, kernel_size=window_size, stride=1, padding=window_size // 2)
    local_min = -torch.nn.functional.max_pool2d(-depth_unsq, kernel_size=window_size, stride=1, padding=window_size // 2)
    local_mean = torch.nn.functional.avg_pool2d(depth_unsq, kernel_size=window_size, stride=1, padding=window_size // 2)
    ratio = (local_max - local_min) / (local_mean + eps)
    reliable_mask = (ratio < ratio_thresh) & (depth_unsq > 0)
    reliable_mask = reliable_mask

    return reliable_mask


def forward_warp_multiframes(
    frame1: torch.Tensor,
    mask1: Optional[torch.Tensor],
    depth1: Optional[torch.Tensor],
    transformation1: Optional[torch.Tensor],
    transformation2: torch.Tensor,
    intrinsic1: Optional[torch.Tensor],
    intrinsic2: Optional[torch.Tensor],
    is_image=True,
    render_depth=False,
    world_points1=None,
    clean_points: bool = False,
    clean_points_continuity: bool = False,  # clean points based on depth continuity
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    :param frame1: (b, v, 3, h, w). If frame1 is not in the range [-1, 1], either set is_image=False when calling
                    bilinear_splatting on frame within this function, or modify clipping in bilinear_splatting()
                    method accordingly.
    :param mask1: (b, v, 1, h, w) - 1 for known, 0 for unknown. Optional
    :param depth1: (b, v, 1, h, w). Optional if world_points1 is provided.
    :param transformation1: (b, v, 4, 4) source view w2c. Required if depth1 is provided and world_points1 is None.
    :param transformation2: (b, 4, 4) extrinsic transformation matrix of target view: [R, t; 0, 1]
    :param intrinsic1: (b, v, 3, 3) source intrinsics. Required if depth1 is provided and world_points1 is None.
    :param intrinsic2: (b, 3, 3) camera intrinsic matrix for target view. Optional
    :param world_points1: (b, v, h, w, 3) optional precomputed world points.
    :param clean_points: bool, enable point cleaning.
    :param clean_points_continuity: bool, use depth continuity for cleaning.
    """
    device = frame1.device
    b, v, c, h, w = frame1.shape
    if mask1 is None:
        mask1 = torch.ones(size=(b, v, 1, h, w), device=device, dtype=frame1.dtype)

    # If world_points1 isn't provided, build it from RGBD + per-view camera parameters.
    if world_points1 is None:
        assert depth1 is not None, "depth1 must be provided when world_points1 is None"
        assert transformation1 is not None, "transformation1 (w2c) must be provided when world_points1 is None"
        assert intrinsic1 is not None, "intrinsic1 must be provided when world_points1 is None"
        assert depth1.shape[:2] == (b, v)
        assert transformation1.shape[:2] == (b, v)
        assert intrinsic1.shape[:2] == (b, v)

        depth1 = torch.nan_to_num(depth1, nan=1e4)
        depth1 = torch.clamp(depth1, min=0, max=1e4)

        # Valid mask: depth>0 plus optional continuity cleaning.
        mask_valid = (depth1 > 0).to(dtype=mask1.dtype, device=device)
        if clean_points and clean_points_continuity:
            depth_flat = rearrange(depth1, "b v c h w -> (b v) c h w")
            cont_mask_flat = reliable_depth_mask_range_batch(depth_flat).to(dtype=mask1.dtype, device=device)
            cont_mask = rearrange(cont_mask_flat, "(b v) c h w -> b v c h w", b=b, v=v)
            mask_valid = mask_valid * cont_mask
        mask1 = mask1 * mask_valid

        depth_flat = rearrange(depth1, "b v c h w -> (b v) c h w")
        w2c_flat = rearrange(transformation1, "b v c d -> (b v) c d")
        K_flat = rearrange(intrinsic1, "b v c d -> (b v) c d")
        mask_flat = rearrange(mask1 > 0.5, "b v c h w -> (b v) c h w")
        world_pts_flat = unproject_points(
            depth=depth_flat,
            w2c=w2c_flat,
            intrinsic=K_flat,
            is_depth=True,
            is_ftheta=False,
            mask=mask_flat,
            return_sparse=False,
        )  # [(b*v), h, w, 3]
        world_points1 = rearrange(world_pts_flat, "(b v) h w c -> b v h w c", b=b, v=v)

    assert world_points1 is not None and world_points1.shape == (b, v, h, w, 3)

    frame1 = frame1.reshape(b * v, c, h, w)
    transformation2 = transformation2.unsqueeze(1).repeat(1, v, 1, 1).view(-1, 4, 4)
    intrinsic2 = intrinsic2.unsqueeze(1).repeat(1, v, 1, 1).view(-1, 3, 3)
    world_points1 = rearrange(world_points1, "b v h w c-> (b v) h w c")
    mask1 = rearrange(mask1, "b v c h w-> (b v) c h w")
    # Avoid in-place ops on potentially expanded/broadcasted mask tensors
    mask1 = mask1.clone()

    trans_points1 = project_points(world_points1, transformation2, intrinsic2)
    mask1 = mask1 * (trans_points1[:, :, :, 2, 0].unsqueeze(1) > 0)
    trans_coordinates = trans_points1[:, :, :, :2, 0] / (trans_points1[:, :, :, 2:3, 0] + 1e-7)
    trans_coordinates = trans_coordinates.permute(0, 3, 1, 2)  # b, 2, h, w
    trans_depth1 = trans_points1[:, :, :, 2, 0].unsqueeze(1)

    grid = create_grid(b * v, h, w, device=device)  # .to(trans_coordinates)
    flow12 = trans_coordinates - grid
    warped_frame2, mask2 = bilinear_splatting(frame1, mask1, trans_depth1, flow12, None, is_image=is_image, n_views=v)
    if render_depth:
        warped_depth2 = bilinear_splatting(trans_depth1, mask1, trans_depth1, flow12, None, is_image=False, n_views=v)[
            0
        ][:, 0]
        return warped_frame2, mask2, warped_depth2, flow12
    return warped_frame2, mask2, None, flow12


def unproject_points(depth: torch.Tensor,
                     w2c: torch.Tensor,
                     intrinsic: torch.Tensor,
                     is_depth: bool = True,
                     is_ftheta: bool = False,
                     mask: Optional[torch.Tensor] = None,
                     return_sparse: bool = False) -> Union[torch.Tensor, List[torch.Tensor]]:
    """
    Unprojects depth values into 3D world points.

    If is_ftheta is False the pinhole model is used; if True then the intrinsic
    is interpreted as [cx, cy, width, height, poly_coeffs..., is_bw_poly] and the
    fθ model is used.

    If return_sparse is True, returns a list of B tensors, where each tensor is of
    shape (N, 3) and contains the unprojected points for that batch.
    """
    b, _, h, w = depth.shape
    device = depth.device
    dtype = depth.dtype
    if mask is None:
        mask = depth > 0
    if mask.dim() == depth.dim() and mask.shape[1] == 1:
        mask = mask[:, 0]

    idx = torch.nonzero(mask)
    if idx.numel() == 0:
        # No valid points: return an empty list (sparse) or a zero tensor (dense)
        if return_sparse:
            return [torch.empty((0, 3), device=device, dtype=dtype) for _ in range(b)]
        else:
            return torch.zeros((b, h, w, 3), device=device, dtype=dtype)

    b_idx, y_idx, x_idx = idx[:, 0], idx[:, 1], idx[:, 2]

    if not is_ftheta:
        # ---- Pinhole model (sparse computation) ----
        intrinsic_inv = inverse_with_conversion(intrinsic)  # (b, 3, 3)

        x_valid = x_idx.to(dtype)
        y_valid = y_idx.to(dtype)
        ones = torch.ones_like(x_valid)
        pos = torch.stack([x_valid, y_valid, ones], dim=1).unsqueeze(-1)  # (N, 3, 1)

        intrinsic_inv_valid = intrinsic_inv[b_idx]  # (N, 3, 3)
        unnormalized_pos = torch.matmul(intrinsic_inv_valid, pos)  # (N, 3, 1)

        depth_valid = depth[b_idx, 0, y_idx, x_idx].view(-1, 1, 1)
        if is_depth:
            world_points_cam = depth_valid * unnormalized_pos
        else:
            norm_val = torch.norm(unnormalized_pos, dim=1, keepdim=True)
            direction = unnormalized_pos / (norm_val + 1e-8)
            world_points_cam = depth_valid * direction

        ones_h = torch.ones((world_points_cam.shape[0], 1, 1),
                            device=device, dtype=dtype)
        world_points_homo = torch.cat([world_points_cam, ones_h], dim=1)  # (N, 4, 1)

        trans = inverse_with_conversion(w2c)  # (b, 4, 4)
        trans_valid = trans[b_idx]  # (N, 4, 4)
        world_points_transformed = torch.matmul(trans_valid, world_points_homo)  # (N, 4, 1)
        sparse_points = world_points_transformed[:, :3, 0]  # (N, 3)
    else:
        # ---- fθ model (sparse computation) ----
        x_valid = x_idx.to(dtype)
        y_valid = y_idx.to(dtype)

        cx_valid = intrinsic[b_idx, 0].view(-1)
        cy_valid = intrinsic[b_idx, 1].view(-1)
        xd = x_valid - cx_valid
        yd = y_valid - cy_valid
        norm_xy = torch.sqrt(xd**2 + yd**2 + 1e-8)

        poly_coeffs_valid = intrinsic[b_idx, 4:-1]  # (N, d)
        d_coeff = poly_coeffs_valid.shape[1]

        powers = torch.arange(d_coeff, device=device, dtype=dtype).view(1, d_coeff)
        norm_powers = norm_xy.view(-1, 1).pow(powers)  # (N, d)
        alpha = (poly_coeffs_valid * norm_powers).sum(dim=1)
        sin_alpha = torch.sin(alpha)
        cos_alpha = torch.cos(alpha)
        scale = sin_alpha / (norm_xy + 1e-8)
        ray_x = scale * xd
        ray_y = scale * yd
        ray_z = cos_alpha

        near_zero = norm_xy < 1e-6
        ray_x[near_zero] = 0.0
        ray_y[near_zero] = 0.0
        ray_z[near_zero] = 1.0
        rays = torch.stack([ray_x, ray_y, ray_z], dim=1)  # (N, 3)
        rays = rays / ray_z.unsqueeze(-1)

        depth_valid = depth[b_idx, 0, y_idx, x_idx].view(-1, 1)
        if is_depth:
            world_points_cam = depth_valid * rays
        else:
            ray_norm = torch.norm(rays, dim=1, keepdim=True)
            world_points_cam = depth_valid * (rays / (ray_norm + 1e-8))

        ones_h = torch.ones((world_points_cam.shape[0], 1),
                            device=device, dtype=dtype)
        world_points_homo = torch.cat([world_points_cam, ones_h], dim=1)  # (N, 4)
        world_points_homo = world_points_homo.unsqueeze(-1)  # (N, 4, 1)
        trans = inverse_with_conversion(w2c)  # (b, 4, 4)
        trans_valid = trans[b_idx]  # (N, 4, 4)
        world_points_transformed = torch.matmul(trans_valid, world_points_homo)  # (N, 4, 1)
        sparse_points = world_points_transformed[:, :3, 0]  # (N, 3)

    if return_sparse:
        counts = torch.bincount(b_idx, minlength=b).tolist()
        sparse_list = []
        offset = 0
        for count in counts:
            if count > 0:
                sparse_list.append(sparse_points[offset:offset+count])
            else:
                sparse_list.append(torch.empty((0, 3), device=device, dtype=dtype))
            offset += count
        return sparse_list
    else:
        out_points = torch.zeros((b, h, w, 3), device=device, dtype=dtype)
        out_points[b_idx, y_idx, x_idx, :] = sparse_points
        return out_points


def project_points(world_points: torch.Tensor, w2c: torch.Tensor, intrinsic: torch.Tensor):
    """
    Projects 3D world points back into 2D pixel space.
    """
    world_points = world_points.unsqueeze(-1)  # (b, h, w, 3) -> # (b, h, w, 3, 1)
    b, h, w, _, _ = world_points.shape

    ones_4d = torch.ones((b, h, w, 1, 1), device=world_points.device, dtype=world_points.dtype)  # (b, h, w, 1, 1)
    world_points_homo = torch.cat([world_points, ones_4d], dim=3)  # (b, h, w, 4, 1)

    # Apply transformation2 to convert world points to camera space
    trans_4d = w2c[:, None, None]  # (b, 1, 1, 4, 4)
    camera_points_homo = torch.matmul(trans_4d, world_points_homo)  # (b, h, w, 4, 1)

    # Remove homogeneous coordinate and project to image plane
    camera_points = camera_points_homo[:, :, :, :3]  # (b, h, w, 3, 1)
    intrinsic_4d = intrinsic[:, None, None]  # (b, 1, 1, 3, 3)
    projected_points = torch.matmul(intrinsic_4d, camera_points)  # (b, h, w, 3, 1)

    return projected_points


def bilinear_splatting(
    frame1: torch.Tensor,
    mask1: Optional[torch.Tensor],
    depth1: torch.Tensor,
    flow12: torch.Tensor,
    flow12_mask: Optional[torch.Tensor],
    is_image: bool = False,
    n_views=1,
    depth_weight_scale=50,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Bilinear splatting
    :param frame1: (b,c,h,w)
    :param mask1: (b,1,h,w): 1 for known, 0 for unknown. Optional
    :param depth1: (b,1,h,w)
    :param flow12: (b,2,h,w)
    :param flow12_mask: (b,1,h,w): 1 for valid flow, 0 for invalid flow. Optional
    :param is_image: if true, output will be clipped to (-1,1) range
    :return: warped_frame2: (b,c,h,w)
             mask2: (b,1,h,w): 1 for known and 0 for unknown
    """
    b, c, h, w = frame1.shape
    device = frame1.device
    dtype = frame1.dtype
    if mask1 is None:
        mask1 = torch.ones(size=(b, 1, h, w), device=device, dtype=dtype)  # .to(frame1)
    if flow12_mask is None:
        flow12_mask = torch.ones(size=(b, 1, h, w), device=device, dtype=dtype)  # .to(flow12)
    grid = create_grid(b, h, w, device=device, dtype=dtype).to(dtype)  # .to(frame1)
    trans_pos = flow12 + grid

    trans_pos_offset = trans_pos + 1
    trans_pos_floor = torch.floor(trans_pos_offset).long()
    trans_pos_ceil = torch.ceil(trans_pos_offset).long()
    trans_pos_offset = torch.stack(
        [torch.clamp(trans_pos_offset[:, 0], min=0, max=w + 1), torch.clamp(trans_pos_offset[:, 1], min=0, max=h + 1)],
        dim=1,
    )
    trans_pos_floor = torch.stack(
        [torch.clamp(trans_pos_floor[:, 0], min=0, max=w + 1), torch.clamp(trans_pos_floor[:, 1], min=0, max=h + 1)],
        dim=1,
    )
    trans_pos_ceil = torch.stack(
        [torch.clamp(trans_pos_ceil[:, 0], min=0, max=w + 1), torch.clamp(trans_pos_ceil[:, 1], min=0, max=h + 1)],
        dim=1,
    )

    prox_weight_nw = (1 - (trans_pos_offset[:, 1:2] - trans_pos_floor[:, 1:2])) * (
        1 - (trans_pos_offset[:, 0:1] - trans_pos_floor[:, 0:1])
    )
    prox_weight_sw = (1 - (trans_pos_ceil[:, 1:2] - trans_pos_offset[:, 1:2])) * (
        1 - (trans_pos_offset[:, 0:1] - trans_pos_floor[:, 0:1])
    )
    prox_weight_ne = (1 - (trans_pos_offset[:, 1:2] - trans_pos_floor[:, 1:2])) * (
        1 - (trans_pos_ceil[:, 0:1] - trans_pos_offset[:, 0:1])
    )
    prox_weight_se = (1 - (trans_pos_ceil[:, 1:2] - trans_pos_offset[:, 1:2])) * (
        1 - (trans_pos_ceil[:, 0:1] - trans_pos_offset[:, 0:1])
    )

    # Calculate depth weights, preventing overflow and removing saturation
    # Clamp depth to be non-negative before log1p
    clamped_depth1 = torch.clamp(depth1, min=0)
    log_depth1 = torch.log1p(clamped_depth1) # Use log1p for better precision near 0
    # Normalize and scale log depth
    exponent = log_depth1 / (log_depth1.max() + 1e-7) * depth_weight_scale
    # Clamp exponent before exp to prevent overflow
    max_exponent = get_max_exponent_for_dtype(depth1.dtype)
    clamped_exponent = torch.clamp(exponent, max=max_exponent)
    # Compute depth weights with added epsilon for stability when dividing later
    depth_weights = torch.exp(clamped_exponent) + 1e-7


    weight_nw = torch.moveaxis(prox_weight_nw * mask1 * flow12_mask / depth_weights, [0, 1, 2, 3], [0, 3, 1, 2])
    weight_sw = torch.moveaxis(prox_weight_sw * mask1 * flow12_mask / depth_weights, [0, 1, 2, 3], [0, 3, 1, 2])
    weight_ne = torch.moveaxis(prox_weight_ne * mask1 * flow12_mask / depth_weights, [0, 1, 2, 3], [0, 3, 1, 2])
    weight_se = torch.moveaxis(prox_weight_se * mask1 * flow12_mask / depth_weights, [0, 1, 2, 3], [0, 3, 1, 2])

    warped_frame = torch.zeros(size=(b, h + 2, w + 2, c), dtype=dtype, device=device)  # .to(frame1)
    warped_weights = torch.zeros(size=(b, h + 2, w + 2, 1), dtype=dtype, device=device)  # .to(frame1)

    frame1_cl = torch.moveaxis(frame1, [0, 1, 2, 3], [0, 3, 1, 2])
    batch_indices = torch.arange(b, device=device, dtype=torch.long)[:, None, None]  # .to(frame1.device)
    warped_frame.index_put_(
        (batch_indices, trans_pos_floor[:, 1], trans_pos_floor[:, 0]), frame1_cl * weight_nw, accumulate=True
    )
    warped_frame.index_put_(
        (batch_indices, trans_pos_ceil[:, 1], trans_pos_floor[:, 0]), frame1_cl * weight_sw, accumulate=True
    )
    warped_frame.index_put_(
        (batch_indices, trans_pos_floor[:, 1], trans_pos_ceil[:, 0]), frame1_cl * weight_ne, accumulate=True
    )
    warped_frame.index_put_(
        (batch_indices, trans_pos_ceil[:, 1], trans_pos_ceil[:, 0]), frame1_cl * weight_se, accumulate=True
    )

    warped_weights.index_put_((batch_indices, trans_pos_floor[:, 1], trans_pos_floor[:, 0]), weight_nw, accumulate=True)
    warped_weights.index_put_((batch_indices, trans_pos_ceil[:, 1], trans_pos_floor[:, 0]), weight_sw, accumulate=True)
    warped_weights.index_put_((batch_indices, trans_pos_floor[:, 1], trans_pos_ceil[:, 0]), weight_ne, accumulate=True)
    warped_weights.index_put_((batch_indices, trans_pos_ceil[:, 1], trans_pos_ceil[:, 0]), weight_se, accumulate=True)
    if n_views > 1:
        warped_frame = warped_frame.reshape(b // n_views, n_views, h + 2, w + 2, c).sum(1)
        warped_weights = warped_weights.reshape(b // n_views, n_views, h + 2, w + 2, 1).sum(1)

    warped_frame_cf = torch.moveaxis(warped_frame, [0, 1, 2, 3], [0, 2, 3, 1])
    warped_weights_cf = torch.moveaxis(warped_weights, [0, 1, 2, 3], [0, 2, 3, 1])
    cropped_warped_frame = warped_frame_cf[:, :, 1:-1, 1:-1]
    cropped_weights = warped_weights_cf[:, :, 1:-1, 1:-1]
    cropped_weights = torch.nan_to_num(cropped_weights, nan=1000.0)

    mask = cropped_weights > 0
    zero_value = -1 if is_image else 0
    zero_tensor = torch.tensor(zero_value, dtype=frame1.dtype, device=frame1.device)
    warped_frame2 = torch.where(mask, cropped_warped_frame / cropped_weights, zero_tensor)
    mask2 = mask.to(frame1)
    if is_image:
        warped_frame2 = torch.clamp(warped_frame2, min=-1, max=1)
    return warped_frame2, mask2


def create_grid(b: int, h: int, w: int, device="cpu", dtype=torch.float) -> torch.Tensor:
    """
    Create a dense grid of (x,y) coordinates of shape (b, 2, h, w).
    """
    x = torch.arange(0, w, device=device, dtype=dtype).view(1, 1, 1, w).expand(b, 1, h, w)
    y = torch.arange(0, h, device=device, dtype=dtype).view(1, 1, h, 1).expand(b, 1, h, w)
    return torch.cat([x, y], dim=1)
