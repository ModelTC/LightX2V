import torch


def depth_to_camera_coords(depthmap, camera_intrinsics):
    """
    Convert depth map to 3D camera coordinates.

    Args:
        depthmap (BxHxW tensor): Batch of depth maps
        camera_intrinsics (Bx3x3 tensor): Camera intrinsics matrix for each camera

    Returns:
        X_cam (BxHxWx3 tensor): 3D points in camera coordinates
        valid_mask (BxHxW tensor): Mask indicating valid depth pixels
    """
    B, H, W = depthmap.shape
    device = depthmap.device
    dtype = depthmap.dtype

    # Ensure intrinsics are float
    camera_intrinsics = camera_intrinsics.float()

    # Extract focal lengths and principal points
    fx = camera_intrinsics[:, 0, 0]  # (B,)
    fy = camera_intrinsics[:, 1, 1]  # (B,)
    cx = camera_intrinsics[:, 0, 2]  # (B,)
    cy = camera_intrinsics[:, 1, 2]  # (B,)

    # Generate pixel grid
    v_grid, u_grid = torch.meshgrid(torch.arange(H, dtype=dtype, device=device), torch.arange(W, dtype=dtype, device=device), indexing="ij")

    # Reshape for broadcasting: (1, H, W)
    u_grid = u_grid.unsqueeze(0)
    v_grid = v_grid.unsqueeze(0)

    # Compute 3D camera coordinates
    # X = (u - cx) * Z / fx
    # Y = (v - cy) * Z / fy
    # Z = depth
    z_cam = depthmap  # (B, H, W)
    x_cam = (u_grid - cx.view(B, 1, 1)) * z_cam / fx.view(B, 1, 1)
    y_cam = (v_grid - cy.view(B, 1, 1)) * z_cam / fy.view(B, 1, 1)

    # Stack to form (B, H, W, 3)
    X_cam = torch.stack([x_cam, y_cam, z_cam], dim=-1)

    # Valid depth mask
    valid_mask = depthmap > 0.0

    return X_cam, valid_mask


def depth_to_world_coords_points(depth_map: torch.Tensor, extrinsic: torch.Tensor, intrinsic: torch.Tensor, eps=1e-8) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Convert a batch of depth maps to world coordinates.

    Args:
        depth_map (torch.Tensor): (B, H, W) Depth map
        extrinsic (torch.Tensor): (B, 4, 4) Camera extrinsic matrix (camera-to-world transformation)
        intrinsic (torch.Tensor): (B, 3, 3) Camera intrinsic matrix

    Returns:
        world_coords_points (torch.Tensor): (B, H, W, 3) World coordinates
        camera_points (torch.Tensor): (B, H, W, 3) Camera coordinates
        point_mask (torch.Tensor): (B, H, W) Valid depth mask
    """
    if depth_map is None:
        return None, None, None

    # Valid depth mask (B, H, W)
    point_mask = depth_map > eps

    # Convert depth map to camera coordinates (B, H, W, 3)
    camera_points, _ = depth_to_camera_coords(depth_map, intrinsic)

    # Apply extrinsic matrix (camera -> world)
    R_cam_to_world = extrinsic[:, :3, :3]  # (B, 3, 3)
    t_cam_to_world = extrinsic[:, :3, 3]  # (B, 3)

    # Transform (B, H, W, 3) x (B, 3, 3)^T + (B, 3) -> (B, H, W, 3)
    world_coords_points = torch.einsum("bhwi,bji->bhwj", camera_points, R_cam_to_world) + t_cam_to_world[:, None, None, :]

    return world_coords_points, camera_points, point_mask


def closed_form_inverse_se3(se3: torch.Tensor) -> torch.Tensor:
    """
    Efficiently invert batched SE(3) matrices of shape (B, 4, 4).

    Args:
        se3 (torch.Tensor): (B, 4, 4) Transformation matrices

    Returns:
        out (torch.Tensor): (B, 4, 4) Inverse transformation matrices
    """
    assert se3.ndim == 3 and se3.shape[1:] == (4, 4), f"se3 must be (B, 4, 4), got {se3.shape}"
    R = se3[:, :3, :3]  # (B, 3, 3)
    t = se3[:, :3, 3]  # (B, 3)
    Rt = R.transpose(1, 2)  # (B, 3, 3)
    t_inv = -torch.bmm(Rt, t.unsqueeze(-1)).squeeze(-1)  # (B, 3)
    out = se3.new_zeros(se3.shape)
    out[:, :3, :3] = Rt
    out[:, :3, 3] = t_inv
    out[:, 3, 3] = 1.0
    return out
