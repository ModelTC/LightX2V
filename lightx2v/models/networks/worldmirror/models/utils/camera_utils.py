import torch

from .rotation import quat_to_rotmat, rotmat_to_quat


def camera_params_to_vector(ext, intr, image_hw=None):
    """Convert camera matrices to a compact vector."""
    # ext: (..., 3, 4): Camera-to-world extrinsic [R|t]
    # intr: (..., 3, 3): Intrinsics
    # image_hw: (h, w)
    R = ext[..., :3, :3]  # Rotation part
    t = ext[..., :3, 3]  # Translation part
    q = rotmat_to_quat(R)  # Quaternion (wxyz)
    h, w = image_hw
    fov_v = 2.0 * torch.atan(h * 0.5 / intr[..., 1, 1])  # Vertical FOV
    fov_u = 2.0 * torch.atan(w * 0.5 / intr[..., 0, 0])  # Horizontal FOV
    vec = torch.stack([t[..., 0], t[..., 1], t[..., 2], q[..., 0], q[..., 1], q[..., 2], q[..., 3], fov_v, fov_u], dim=-1).float()
    return vec


def extrinsics_to_vector(ext):
    """Convert extrinsics to [t, q] vector."""
    # ext: (..., 3, 4)
    R = ext[..., :3, :3]
    t = ext[..., :3, 3]
    q = rotmat_to_quat(R)
    vec = torch.stack([t[..., 0], t[..., 1], t[..., 2], q[..., 0], q[..., 1], q[..., 2], q[..., 3]], dim=-1).float()
    return vec


def vector_to_extrinsics(cam_vec):
    """Convert [t, q] vector to extrinsic matrix."""
    # cam_vec: (..., 7)
    q = cam_vec[..., 3:7]
    t = cam_vec[..., :3]
    R = quat_to_rotmat(q)
    ext = torch.cat([R, t.unsqueeze(-1)], dim=-1)
    return ext


def vector_to_camera_matrices(cam_vec, image_hw=None, build_intr=True):
    """Reconstruct extrinsic and intrinsic matrix from vector."""
    # cam_vec: (..., 9)
    intr = None
    # Decompose vector
    t = cam_vec[..., 0:3]
    q = cam_vec[..., 3:7]
    fov_v = cam_vec[..., 7]
    fov_u = cam_vec[..., 8]

    # Build extrinsic: [R|t]
    R = quat_to_rotmat(q)
    ext = torch.cat([R, t.unsqueeze(-1)], dim=-1)

    # Build intrinsic if needed
    if build_intr:
        h, w = image_hw
        fy = h * 0.5 / torch.tan(fov_v * 0.5)
        fx = w * 0.5 / torch.tan(fov_u * 0.5)
        shape = cam_vec.shape[:-1] + (3, 3)
        intr = torch.zeros(shape, device=cam_vec.device, dtype=cam_vec.dtype)
        intr[..., 0, 0] = fx
        intr[..., 1, 1] = fy
        intr[..., 0, 2] = w * 0.5
        intr[..., 1, 2] = h * 0.5
        intr[..., 2, 2] = 1.0

    return ext, intr
