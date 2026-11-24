import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor


def rotation_6d_to_matrix(d6: Tensor) -> Tensor:
    """
    Converts 6D rotation representation by Zhou et al. [1] to rotation matrix
    using Gram--Schmidt orthogonalization per Section B of [1].
    Args:
        d6: 6D rotation representation, of size (*, 6)

    Returns:
        batch of rotation matrices of size (*, 3, 3)

    [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
    On the Continuity of Rotation Representations in Neural Networks.
    IEEE Conference on Computer Vision and Pattern Recognition, 2019.
    Retrieved from http://arxiv.org/abs/1812.07035
    """

    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = F.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-2)


def matrix_to_rotation_6d(matrix: Tensor) -> Tensor:
    """
    Converts rotation matrices to 6D rotation representation by Zhou et al. [1]
    by dropping the last row. Note that 6D representation is not unique.
    Args:
        matrix: batch of rotation matrices of size (*, 3, 3)

    Returns:
        6D rotation representation, of size (*, 6)

    [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
    On the Continuity of Rotation Representations in Neural Networks.
    IEEE Conference on Computer Vision and Pattern Recognition, 2019.
    Retrieved from http://arxiv.org/abs/1812.07035
    """
    batch_dim = matrix.size()[:-2]
    return matrix[..., :2, :].clone().reshape(batch_dim + (6,))


def standardize_quaternion(quaternions: Tensor) -> Tensor:
    """
    Convert a unit quaternion to a standard form: one in which the real
    part is non negative.

    Args:
        quaternions: Quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Standardized quaternions as tensor of shape (..., 4).
    """
    return torch.where(quaternions[..., 0:1] < 0, -quaternions, quaternions)


def _sqrt_positive_part(x: Tensor) -> Tensor:
    """Returns torch.sqrt(torch.max(0, x)) but with a zero subgradient where x is 0."""
    ret = torch.zeros_like(x)
    positive_mask = x > 0
    if torch.is_grad_enabled():
        ret[positive_mask] = torch.sqrt(x[positive_mask])
    else:
        ret = torch.where(positive_mask, torch.sqrt(x), ret)
    return ret


def matrix_to_quaternion(matrix: Tensor) -> Tensor:
    """Convert rotations given as rotation matrices to quaternions.

    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).

    Returns:
        quaternions with real part first, as tensor of shape (..., 4).
    """
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")

    batch_dim = matrix.shape[:-2]
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(matrix.reshape(batch_dim + (9,)), dim=-1)

    q_abs = _sqrt_positive_part(
        torch.stack(
            [
                1.0 + m00 + m11 + m22,
                1.0 + m00 - m11 - m22,
                1.0 - m00 + m11 - m22,
                1.0 - m00 - m11 + m22,
            ],
            dim=-1,
        )
    )

    # we produce the desired quaternion multiplied by each of r, i, j, k
    quat_by_rijk = torch.stack(
        [
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], dim=-1),
        ],
        dim=-2,
    )

    # We floor here at 0.1 but the exact level is not important; if q_abs is small,
    # the candidate won't be picked.
    flr = torch.tensor(0.1).to(dtype=q_abs.dtype, device=q_abs.device)
    quat_candidates = quat_by_rijk / (2.0 * q_abs[..., None].max(flr))

    # if not for numerical problems, quat_candidates[i] should be same (up to a sign),
    # forall i; we pick the best-conditioned one (with the largest denominator)
    out = quat_candidates[F.one_hot(q_abs.argmax(dim=-1), num_classes=4) > 0.5, :].reshape(batch_dim + (4,))
    return standardize_quaternion(out)


def quaternion_to_axis_angle(quaternions: Tensor) -> Tensor:
    """Convert rotations given as quaternions to axis/angle.

    Args:
        quaternions: quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Rotations given as a vector in axis angle form, as a tensor
            of shape (..., 3), where the magnitude is the angle
            turned anticlockwise in radians around the vector's
            direction.
    """
    norms = torch.norm(quaternions[..., 1:], p=2, dim=-1, keepdim=True)
    half_angles = torch.atan2(norms, quaternions[..., :1])
    angles = 2 * half_angles
    eps = 1e-6
    small_angles = angles.abs() < eps
    sin_half_angles_over_angles = torch.empty_like(angles)
    sin_half_angles_over_angles[~small_angles] = torch.sin(half_angles[~small_angles]) / angles[~small_angles]
    # for x small, sin(x/2) is about x/2 - (x/2)^3/6
    # so sin(x/2)/x is about 1/2 - (x*x)/48
    sin_half_angles_over_angles[small_angles] = 0.5 - (angles[small_angles] * angles[small_angles]) / 48
    return quaternions[..., 1:] / sin_half_angles_over_angles


def matrix_to_axis_angle(matrix: Tensor) -> Tensor:
    """Convert rotations given as rotation matrices to axis/angle.

    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).

    Returns:
        Rotations given as a vector in axis angle form, as a tensor
            of shape (..., 3), where the magnitude is the angle
            turned anticlockwise in radians around the vector's
            direction.
    """
    return quaternion_to_axis_angle(matrix_to_quaternion(matrix))


def quaternion_to_matrix(quaternions: Tensor) -> Tensor:
    """Convert rotations given as quaternions to rotation matrices.

    Args:
        quaternions: quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    r, i, j, k = torch.unbind(quaternions, -1)
    # pyre-fixme[58]: `/` is not supported for operand types `float` and `Tensor`.
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))


def axis_angle_to_quaternion(axis_angle: Tensor) -> Tensor:
    """Convert rotations given as axis/angle to quaternions.

    Args:
        axis_angle: Rotations given as a vector in axis angle form,
            as a tensor of shape (..., 3), where the magnitude is
            the angle turned anticlockwise in radians around the
            vector's direction.

    Returns:
        quaternions with real part first, as tensor of shape (..., 4).
    """
    angles = torch.norm(axis_angle, p=2, dim=-1, keepdim=True)
    half_angles = angles * 0.5
    eps = 1e-6
    small_angles = angles.abs() < eps
    sin_half_angles_over_angles = torch.empty_like(angles)
    sin_half_angles_over_angles[~small_angles] = torch.sin(half_angles[~small_angles]) / angles[~small_angles]
    # for x small, sin(x/2) is about x/2 - (x/2)^3/6
    # so sin(x/2)/x is about 1/2 - (x*x)/48
    sin_half_angles_over_angles[small_angles] = 0.5 - (angles[small_angles] * angles[small_angles]) / 48
    quaternions = torch.cat([torch.cos(half_angles), axis_angle * sin_half_angles_over_angles], dim=-1)
    return quaternions


def axis_angle_to_matrix(axis_angle: Tensor) -> Tensor:
    """Convert rotations given as axis/angle to rotation matrices.

    Args:
        axis_angle: Rotations given as a vector in axis angle form,
            as a tensor of shape (..., 3), where the magnitude is
            the angle turned anticlockwise in radians around the
            vector's direction.

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    return quaternion_to_matrix(axis_angle_to_quaternion(axis_angle))


def get_T_w2c_from_wcparams(global_orient_w: Tensor, transl_w: Tensor, global_orient_c: Tensor, transl_c: Tensor, offset: Tensor) -> Tensor:
    """
    Args:
        global_orient_w: Tensor, (F, 3)
        transl_w: Tensor, (F, 3)
        global_orient_c: Tensor, (F, 3)
        transl_c: Tensor, (F, 3)
        offset: Tensor, (*, 3)
    Returns:
        T_w2c: Tensor, (F, 4, 4)
    """
    assert global_orient_w.shape == transl_w.shape and len(global_orient_w.shape) == 2
    assert global_orient_c.shape == transl_c.shape and len(global_orient_c.shape) == 2

    R_w = axis_angle_to_matrix(global_orient_w)  # (F, 3, 3)
    t_w = transl_w  # (F, 3)
    R_c = axis_angle_to_matrix(global_orient_c)  # (F, 3, 3)
    t_c = transl_c  # (F, 3)

    R_w2c = R_c @ R_w.transpose(-1, -2)  # (F, 3, 3)
    t_w2c = t_c + offset - torch.einsum("fij,fj->fi", R_w2c, t_w + offset)  # (F, 3)
    T_w2c = torch.eye(4, device=global_orient_w.device).repeat(R_w.size(0), 1, 1)  # (F, 4, 4)
    T_w2c[..., :3, :3] = R_w2c  # (F, 3, 3)
    T_w2c[..., :3, 3] = t_w2c  # (F, 3)
    return T_w2c


def get_R_c2gv(R_w2c, axis_gravity_in_w=None):
    """
    Args:
        R_w2c: (*, 3, 3)
    Returns:
        R_c2gv: (*, 3, 3)
    """
    if axis_gravity_in_w is None:
        axis_gravity_in_w = [0, 0, -1]
    if isinstance(axis_gravity_in_w, list):
        axis_gravity_in_w = torch.tensor(axis_gravity_in_w).float()  # gravity direction in world coord
    axis_z_in_c = torch.tensor([0, 0, 1]).float()

    # get gv-coord axes in in c-coord
    axis_y_of_gv = R_w2c @ axis_gravity_in_w  # (*, 3)
    axis_x_of_gv = axis_y_of_gv.cross(axis_z_in_c.expand_as(axis_y_of_gv), dim=-1)
    # normalize
    axis_x_of_gv_norm = axis_x_of_gv.norm(dim=-1, keepdim=True)
    axis_x_of_gv = axis_x_of_gv / (axis_x_of_gv_norm + 1e-5)
    axis_x_of_gv[axis_x_of_gv_norm.squeeze(-1) < 1e-5] = torch.tensor([1.0, 0.0, 0.0])  # use cam x-axis as axis_x_of_gv
    axis_z_of_gv = axis_x_of_gv.cross(axis_y_of_gv, dim=-1)

    R_gv2c = torch.stack([axis_x_of_gv, axis_y_of_gv, axis_z_of_gv], dim=-1)  # (*, 3, 3)
    R_c2gv = R_gv2c.transpose(-1, -2)  # (*, 3, 3)
    return R_c2gv


def get_c_rootparam(global_orient: Tensor, transl: Tensor, T_w2c: Tensor, offset: Tensor) -> tuple[Tensor, Tensor]:
    """
    Args:
        global_orient: Tensor, (F, 3)
        transl: Tensor, (F, 3)
        T_w2c: Tensor, (*, 4, 4)
        offset: Tensor, (3,)
    Returns:
        R_c: Tensor, (F, 3)
        t_c: Tensor, (F, 3)
    """
    assert global_orient.shape == transl.shape and len(global_orient.shape) == 2
    R_w = axis_angle_to_matrix(global_orient)  # (F, 3, 3)
    t_w = transl  # (F, 3)

    R_w2c = T_w2c[..., :3, :3]  # (*, 3, 3)
    t_w2c = T_w2c[..., :3, 3]  # (*, 3)
    if len(R_w2c.shape) == 2:
        R_w2c = R_w2c[None].expand(R_w.size(0), -1, -1)  # (F, 3, 3)
        t_w2c = t_w2c[None].expand(t_w.size(0), -1)

    R_c = matrix_to_axis_angle(R_w2c @ R_w)  # (F, 3)
    t_c = torch.einsum("fij,fj->fi", R_w2c, t_w + offset) + t_w2c - offset  # (F, 3)
    return R_c, t_c


def compute_cam_angvel(R_w2c, padding_last=True):
    """
    R_w2c : (F, 3, 3)
    """
    # R @ R0 = R1, so R = R1 @ R0^T
    cam_angvel = matrix_to_rotation_6d(R_w2c[1:] @ R_w2c[:-1].transpose(-1, -2))  # (F-1, 6)
    # cam_angvel = (cam_angvel - torch.tensor([[1, 0, 0, 0, 1, 0]])) * FPS
    assert padding_last
    cam_angvel = torch.cat([cam_angvel, cam_angvel[-1:]], dim=0)  # (F, 6)
    return cam_angvel.float()


def rot6d_to_rotation_matrix(rot6d):
    """Convert 6D rotation representation to 3x3 rotation matrix.

    Based on Zhou et al., "On the Continuity of Rotation Representations in Neural Networks", CVPR 2019
    Args:
        rot6d: torch tensor of shape (batch_size, 6) of 6d rotation representations.
    Returns:
        rotation_matrix: torch tensor of shape (batch_size, 3, 3) of corresponding rotation matrices.
    """
    # x = rot6d.view(-1, 3, 2)
    x = rot6d.view(*rot6d.shape[:-1], 3, 2)
    a1 = x[..., 0]
    a2 = x[..., 1]
    b1 = F.normalize(a1, dim=-1)
    b2 = F.normalize(a2 - torch.einsum("...i,...i->...", b1, a2).unsqueeze(-1) * b1, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-1)


def rotation_matrix_to_rot6d(rotation_matrix):
    """Convert 3x3 rotation matrix to 6D rotation representation.

    Args:
        rotation_matrix: torch tensor of shape (batch_size, 3, 3) of corresponding rotation matrices.
    Returns:
        rot6d: torch tensor of shape (batch_size, 6) of 6d rotation representations.
    """
    v1 = rotation_matrix[..., 0:1]
    v2 = rotation_matrix[..., 1:2]
    rot6d = torch.cat([v1, v2], dim=-1).reshape(*v1.shape[:-2], 6)
    return rot6d


def quaternion_to_rotation_matrix(quaternion):
    """Convert quaternion coefficients to rotation matrix.

    Args:
        quaternion: torch tensor of shape (batch_size, 4) in (w, x, y, z) representation.
    Returns:
        rotation matrix corresponding to the quaternion, torch tensor of shape (batch_size, 3, 3)
    """

    norm_quaternion = quaternion
    norm_quaternion = norm_quaternion / norm_quaternion.norm(p=2, dim=-1, keepdim=True)
    w, x, y, z = norm_quaternion[..., 0], norm_quaternion[..., 1], norm_quaternion[..., 2], norm_quaternion[..., 3]

    w2, x2, y2, z2 = w.pow(2), x.pow(2), y.pow(2), z.pow(2)
    wx, wy, wz = w * x, w * y, w * z
    xy, xz, yz = x * y, x * z, y * z

    rotation_matrix = torch.stack(
        [
            w2 + x2 - y2 - z2,
            2 * xy - 2 * wz,
            2 * wy + 2 * xz,
            2 * wz + 2 * xy,
            w2 - x2 + y2 - z2,
            2 * yz - 2 * wx,
            2 * xz - 2 * wy,
            2 * wx + 2 * yz,
            w2 - x2 - y2 + z2,
        ],
        dim=-1,
    )
    rotation_matrix = rotation_matrix.view(*quaternion.shape[:-1], 3, 3)
    return rotation_matrix


def quaternion_to_angle_axis(quaternion: Tensor) -> Tensor:
    """
    This function is borrowed from https://github.com/kornia/kornia

    Convert quaternion vector to angle axis of rotation.

    Adapted from ceres C++ library: ceres-solver/include/ceres/rotation.h

    Args:
        quaternion (Tensor): tensor with quaternions.

    Return:
        Tensor: tensor with angle axis of rotation.

    Shape:
        - Input: :math:`(*, 4)` where `*` means, any number of dimensions
        - Output: :math:`(*, 3)`

    Example:
        >>> quaternion = torch.rand(2, 4)  # Nx4
        >>> angle_axis = tgm.quaternion_to_angle_axis(quaternion)  # Nx3
    """
    if not torch.is_tensor(quaternion):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(quaternion)}")

    if not quaternion.shape[-1] == 4:
        raise ValueError(f"Input must be a tensor of shape Nx4 or 4. Got {quaternion.shape}")
    # unpack input and compute conversion
    q1: torch.Tensor = quaternion[..., 1]
    q2: torch.Tensor = quaternion[..., 2]
    q3: torch.Tensor = quaternion[..., 3]
    sin_squared_theta: torch.Tensor = q1 * q1 + q2 * q2 + q3 * q3

    sin_theta: torch.Tensor = torch.sqrt(sin_squared_theta)
    cos_theta: torch.Tensor = quaternion[..., 0]
    two_theta: torch.Tensor = 2.0 * torch.where(cos_theta < 0.0, torch.atan2(-sin_theta, -cos_theta), torch.atan2(sin_theta, cos_theta))

    k_pos: torch.Tensor = two_theta / sin_theta
    k_neg: torch.Tensor = 2.0 * torch.ones_like(sin_theta)
    k: torch.Tensor = torch.where(sin_squared_theta > 0.0, k_pos, k_neg)

    angle_axis: torch.Tensor = torch.zeros_like(quaternion)[..., :3]
    angle_axis[..., 0] += q1 * k
    angle_axis[..., 1] += q2 * k
    angle_axis[..., 2] += q3 * k
    return angle_axis


def rotation_matrix_to_quaternion(rotation_matrix, eps=1e-6):
    """
    This function is borrowed from https://github.com/kornia/kornia

    Convert 3x4 rotation matrix to 4d quaternion vector

    This algorithm is based on algorithm described in
    https://github.com/KieranWynn/pyquaternion/blob/master/pyquaternion/quaternion.py#L201

    Args:
        rotation_matrix (Tensor): the rotation matrix to convert.

    Return:
        Tensor: the rotation in quaternion

    Shape:
        - Input: :math:`(N, 3, 4)`
        - Output: :math:`(N, 4)`

    Example:
        >>> input = torch.rand(4, 3, 4)  # Nx3x4
        >>> output = tgm.rotation_matrix_to_quaternion(input)  # Nx4
    """
    if not torch.is_tensor(rotation_matrix):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(rotation_matrix)}")

    if len(rotation_matrix.shape) > 3:
        raise ValueError(f"Input size must be a three dimensional tensor. Got {rotation_matrix.shape}")
    if rotation_matrix.shape[-2:] != (3, 4):
        hom = torch.tensor([0, 0, 1], dtype=rotation_matrix.dtype, device=rotation_matrix.device).reshape(1, 3, 1).expand(rotation_matrix.shape[0], -1, -1)
        rotation_matrix = torch.cat([rotation_matrix, hom], dim=-1)

    rmat_t = torch.transpose(rotation_matrix, 1, 2)

    mask_d2 = rmat_t[:, 2, 2] < eps

    mask_d0_d1 = rmat_t[:, 0, 0] > rmat_t[:, 1, 1]
    mask_d0_nd1 = rmat_t[:, 0, 0] < -rmat_t[:, 1, 1]

    t0 = 1 + rmat_t[:, 0, 0] - rmat_t[:, 1, 1] - rmat_t[:, 2, 2]
    q0 = torch.stack(
        [rmat_t[:, 1, 2] - rmat_t[:, 2, 1], t0, rmat_t[:, 0, 1] + rmat_t[:, 1, 0], rmat_t[:, 2, 0] + rmat_t[:, 0, 2]],
        -1,
    )
    t0_rep = t0.repeat(4, 1).t()

    t1 = 1 - rmat_t[:, 0, 0] + rmat_t[:, 1, 1] - rmat_t[:, 2, 2]
    q1 = torch.stack(
        [rmat_t[:, 2, 0] - rmat_t[:, 0, 2], rmat_t[:, 0, 1] + rmat_t[:, 1, 0], t1, rmat_t[:, 1, 2] + rmat_t[:, 2, 1]],
        -1,
    )
    t1_rep = t1.repeat(4, 1).t()

    t2 = 1 - rmat_t[:, 0, 0] - rmat_t[:, 1, 1] + rmat_t[:, 2, 2]
    q2 = torch.stack(
        [rmat_t[:, 0, 1] - rmat_t[:, 1, 0], rmat_t[:, 2, 0] + rmat_t[:, 0, 2], rmat_t[:, 1, 2] + rmat_t[:, 2, 1], t2],
        -1,
    )
    t2_rep = t2.repeat(4, 1).t()

    t3 = 1 + rmat_t[:, 0, 0] + rmat_t[:, 1, 1] + rmat_t[:, 2, 2]
    q3 = torch.stack(
        [t3, rmat_t[:, 1, 2] - rmat_t[:, 2, 1], rmat_t[:, 2, 0] - rmat_t[:, 0, 2], rmat_t[:, 0, 1] - rmat_t[:, 1, 0]],
        -1,
    )
    t3_rep = t3.repeat(4, 1).t()

    mask_c0 = mask_d2 * mask_d0_d1
    mask_c1 = mask_d2 * ~mask_d0_d1
    mask_c2 = ~mask_d2 * mask_d0_nd1
    mask_c3 = ~mask_d2 * ~mask_d0_nd1
    mask_c0 = mask_c0.view(-1, 1).type_as(q0)
    mask_c1 = mask_c1.view(-1, 1).type_as(q1)
    mask_c2 = mask_c2.view(-1, 1).type_as(q2)
    mask_c3 = mask_c3.view(-1, 1).type_as(q3)

    q = q0 * mask_c0 + q1 * mask_c1 + q2 * mask_c2 + q3 * mask_c3
    q /= torch.sqrt(t0_rep * mask_c0 + t1_rep * mask_c1 + t2_rep * mask_c2 + t3_rep * mask_c3)
    q *= 0.5
    return q


def rotation_matrix_to_angle_axis(rotation_matrix):
    """
    This function is borrowed from https://github.com/kornia/kornia

    Convert 3x4 rotation matrix to Rodrigues vector

    Args:
        rotation_matrix (Tensor): rotation matrix.

    Returns:
        Tensor: Rodrigues vector transformation.

    Shape:
        - Input: :math:`(N, 3, 4)`
        - Output: :math:`(N, 3)`

    Example:
        >>> input = torch.rand(2, 3, 4)  # Nx4x4
        >>> output = tgm.rotation_matrix_to_angle_axis(input)  # Nx3
    """
    origin_shape = rotation_matrix.shape[:-2]
    flat_rot = rotation_matrix.reshape(-1, *rotation_matrix.shape[-2:])
    if flat_rot.shape[1:] == (3, 3):
        rot_mat = flat_rot
        hom = torch.tensor([0, 0, 1], dtype=rotation_matrix.dtype, device=rotation_matrix.device).reshape(1, 3, 1).expand(rot_mat.shape[0], -1, -1)
        flat_rot = torch.cat([rot_mat, hom], dim=-1)

    quaternion = rotation_matrix_to_quaternion(flat_rot)
    aa = quaternion_to_angle_axis(quaternion)
    aa[torch.isnan(aa)] = 0.0
    aa = aa.reshape(*origin_shape, 3)
    return aa


def quat_to_rotmat(quat):
    """Convert quaternion coefficients to rotation matrix.

    Args:
        quat: size = [B, 4] 4 <===>(w, x, y, z)
    Returns:
        Rotation matrix corresponding to the quaternion -- size = [B, 3, 3]
    """
    norm_quat = quat
    norm_quat = norm_quat / norm_quat.norm(p=2, dim=1, keepdim=True)
    w, x, y, z = norm_quat[:, 0], norm_quat[:, 1], norm_quat[:, 2], norm_quat[:, 3]

    B = quat.size(0)

    w2, x2, y2, z2 = w.pow(2), x.pow(2), y.pow(2), z.pow(2)
    wx, wy, wz = w * x, w * y, w * z
    xy, xz, yz = x * y, x * z, y * z

    rotMat = torch.stack(
        [
            w2 + x2 - y2 - z2,
            2 * xy - 2 * wz,
            2 * wy + 2 * xz,
            2 * wz + 2 * xy,
            w2 - x2 + y2 - z2,
            2 * yz - 2 * wx,
            2 * xz - 2 * wy,
            2 * wx + 2 * yz,
            w2 - x2 - y2 + z2,
        ],
        dim=1,
    ).view(B, 3, 3)
    return rotMat


def angle_axis_to_rotation_matrix(theta):
    """Convert axis-angle representation to rotation matrix.

    Args:
        theta: size = [B, 3]
    Returns:
        Rotation matrix corresponding to the quaternion -- size = [B, 3, 3]
    """
    origin_shape = theta.shape[:-1]
    flat_theta = theta.reshape(-1, 3)
    l1norm = torch.norm(flat_theta + 1e-8, p=2, dim=1)
    angle = torch.unsqueeze(l1norm, -1)
    normalized = torch.div(flat_theta, angle)
    angle = angle * 0.5
    v_cos = torch.cos(angle)
    v_sin = torch.sin(angle)
    quat = torch.cat([v_cos, v_sin * normalized], dim=1)
    rot_mat = quat_to_rotmat(quat)
    return rot_mat.reshape(*origin_shape, 3, 3)


def rotation_matrix_to_euler_angles(rotation_matrix):
    """Convert 3x3 rotation matrix to Euler angles."""
    is_torch = False
    if isinstance(rotation_matrix, Tensor):
        is_torch = True
        device = rotation_matrix.device
        rotation_matrix = rotation_matrix.cpu().numpy()
    from scipy.spatial.transform import Rotation

    rot_flat = rotation_matrix.reshape(-1, 3, 3)
    euler_angles = Rotation.from_matrix(rot_flat).as_euler("xyz", degrees=True)
    if is_torch:
        return torch.from_numpy(euler_angles).to(device)
    return euler_angles


def euler_angles_to_rotation_matrix(euler_angles, degrees=True):
    """Convert Euler angles to 3x3 rotation matrix.

    Args:
        euler_angles: Euler angles in xyz order, shape = [B, 3] or any shape with last dimension 3
        degrees: Whether the angles are in degrees (True) or radians (False)

    Returns:
        Rotation matrix corresponding to the Euler angles, shape = [..., 3, 3]
    """
    from scipy.spatial.transform import Rotation

    orig_shape = euler_angles.shape[:-1]
    euler_flat = euler_angles.reshape(-1, 3)
    rot_flat = Rotation.from_euler("xyz", euler_flat, degrees=degrees).as_matrix()
    return rot_flat.reshape(*orig_shape, 3, 3)


def get_local_transl_vel(transl, global_orient_R, fps=30):
    """
    transl velocity is in local coordinate (or, SMPL-coord)
    Args:
        transl: (*, L, 3)
        global_orient: (*, L, 3, 3)
    Returns:
        transl_vel: (*, L, 3)
    """
    transl_vel = transl[..., 1:, :] - transl[..., :-1, :]  # (B, L-1, 3)
    transl_vel = torch.cat([torch.zeros_like(transl_vel[:1]), transl_vel], dim=-2)  # (B, L, 3)  last-padding
    transl_vel = transl_vel * fps

    # v_local = R^T @ v_global
    local_transl_vel = torch.einsum("...lij,...li->...lj", global_orient_R, transl_vel)
    return local_transl_vel


def compute_transl_full_cam(pred_cam, bbx_xys, K_fullimg):
    s, tx, ty = pred_cam[..., 0], pred_cam[..., 1], pred_cam[..., 2]
    focal_length = K_fullimg[..., 0, 0]

    icx = K_fullimg[..., 0, 2]
    icy = K_fullimg[..., 1, 2]
    sb = s * bbx_xys[..., 2]
    cx = 2 * (bbx_xys[..., 0] - icx) / (sb + 1e-9)
    cy = 2 * (bbx_xys[..., 1] - icy) / (sb + 1e-9)
    tz = 2 * focal_length / (sb + 1e-9)

    cam_t = torch.stack([tx + cx, ty + cy, tz], dim=-1)
    return cam_t


def quaternion_fix_continuity(q: Tensor) -> Tensor:
    """Force quaternion continuity across the time dimension by selecting the representation (q or -q) with minimal
    distance (or, equivalently, maximal dot product) between two consecutive frames."""
    assert q.ndim in (
        2,
        3,
    ), f"Expected 3D tensor (L, J, 4), or 2D tensor (L, 4), but got shape {q.shape}"
    assert q.shape[-1] == 4, f"Last dimension should be 4 for quaternions, got {q.shape[-1]}"
    if q.shape[0] <= 1:
        return q.clone()  # single frame or empty sequence, no need to process

    result = q.clone()
    # compute the dot product between consecutive frames (L-1, J) or (L-1)
    dot_products = torch.sum(q[1:] * q[:-1], dim=-1)
    # find the negative dot product (indicates need to flip sign)
    flip_mask = dot_products < 0
    # accumulate the flip mask, ensure consistency
    # if a frame needs to be flipped, all subsequent frames need to be flipped the same number of times
    flip_mask = (torch.cumsum(flip_mask.int(), dim=0) % 2).bool()
    # flip the sign of the frames that need to be flipped
    result[1:][flip_mask] *= -1
    return result


def rot_mat2trans_mat(rot_mat: np.ndarray) -> np.ndarray:
    # assert rot_mat.shape == (3, 3)
    trans_mat = np.identity(4)
    if len(rot_mat.shape) == 2:
        pass
    elif len(rot_mat.shape) == 3:
        trans_mat = np.tile(trans_mat, [rot_mat.shape[0], 1, 1])
    elif len(rot_mat.shape) == 4:
        trans_mat = np.tile(trans_mat, [rot_mat.shape[0], rot_mat.shape[1], 1, 1])
    else:
        raise NotImplementedError
    trans_mat[..., :3, :3] = rot_mat
    return trans_mat


def trans2trans_mat(trans: np.ndarray) -> np.ndarray:
    assert trans.shape[-1] == 3
    assert (len(trans.shape) == 1) or (len(trans.shape) == 2) or (len(trans.shape) == 3), trans.shape
    if len(trans.shape) == 1:
        trans_mat = np.identity(4)
        trans_mat[:3, 3] = trans
    elif len(trans.shape) == 2:
        trans_mat = np.tile(np.identity(4), [trans.shape[0], 1, 1])
        trans_mat[:, :3, 3] = trans
    elif len(trans.shape) == 3:
        trans_mat = np.tile(np.identity(4), [trans.shape[0], trans.shape[1], 1, 1])
        trans_mat[:, :, :3, 3] = trans
    else:
        raise NotImplementedError
    return trans_mat


def gaussian_kernel1d(sigma: float, order: int, radius: int) -> np.ndarray:
    """Computes a 1D Gaussian convolution kernel.

    (from scipy)
    """
    if order < 0:
        raise ValueError("order must be non-negative")
    exponent_range = np.arange(order + 1)
    sigma2 = sigma * sigma
    x = np.arange(-radius, radius + 1)
    phi_x = np.exp(-0.5 / sigma2 * x**2)
    phi_x = phi_x / phi_x.sum()

    if order == 0:
        return phi_x
    else:
        # f(x) = q(x) * phi(x) = q(x) * exp(p(x))
        # f'(x) = (q'(x) + q(x) * p'(x)) * phi(x)
        # p'(x) = -1 / sigma ** 2
        # Implement q'(x) + q(x) * p'(x) as a matrix operator and apply to the
        # coefficients of q(x)
        q = np.zeros(order + 1)
        q[0] = 1
        D = np.diag(exponent_range[1:], 1)  # D @ q(x) = q'(x)
        P = np.diag(np.ones(order) / -sigma2, -1)  # P @ q(x) = q(x) * p'(x)
        Q_deriv = D + P
        for _ in range(order):
            q = Q_deriv.dot(q)
        q = (x[:, None] ** exponent_range).dot(q)
        return q * phi_x


def slice_seq_with_padding(whole_seq: np.ndarray, middle_idx: int, length: int) -> np.ndarray:
    whole_seq_padded = whole_seq.copy()
    if middle_idx - length // 2 < 0:
        # need padding
        l_pad_len = length // 2 - middle_idx
        whole_seq_padded = np.concatenate([np.stack([whole_seq_padded[0]] * l_pad_len), whole_seq_padded], axis=0)
    else:
        l_pad_len = 0
    if middle_idx + length - length // 2 > len(whole_seq):
        r_pad_len = middle_idx + length - length // 2 - len(whole_seq)
        whole_seq_padded = np.concatenate([whole_seq_padded, np.stack([whole_seq_padded[-1]] * r_pad_len)], axis=0)
    else:
        r_pad_len = 0
    assert len(whole_seq_padded) == len(whole_seq) + l_pad_len + r_pad_len
    middle_idx_padded = middle_idx + l_pad_len
    assert middle_idx_padded - length // 2 >= 0
    assert middle_idx_padded + length - length // 2 <= len(whole_seq_padded)
    return whole_seq_padded[middle_idx_padded - length // 2 : middle_idx_padded - length // 2 + length]


def wavg_quaternion_markley(Q: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """
    Averaging Quaternions.
    This is a python implementation of Tolga Birdal's algorithm by https://stackoverflow.com/a/49690919

    Arguments:
        Q(ndarray): an Mx4 ndarray of quaternions.
        weights(list): an M elements list, a weight for each quaternion.

    refer to Tolga Birdal's matlab implementation on
    https://ww2.mathworks.cn/matlabcentral/fileexchange/40098-tolgabirdal-averaging_quaternions?s_tid=prof_contriblnk&s_tid=mwa_osa_a
    by Tolga Birdal
    Q is an Mx4 matrix of quaternions. weights is an Mx1 vector, a weight for
    each quaternion.
    Qavg is the weighted average quaternion
    This function is especially useful for example when clustering poses
    after a matching process. In such cases a form of weighting per rotation
    is available (e.g. number of votes), which can guide the trust towards a
    specific pose. weights might then be interpreted as the vector of votes
    per pose.
    Markley, F. Landis, Yang Cheng, John Lucas Crassidis, and Yaakov Oshman.
    "Averaging quaternions." Journal of Guidance, Control, and Dynamics 30,
    no. 4 (2007): 1193-1197.
    """

    # Form the symmetric accumulator matrix
    # pdb.set_trace()
    A = np.zeros((4, 4))
    M = Q.shape[0]
    wSum = 0

    for i in range(M):
        q = Q[i, :]
        w_i = weights[i]
        if q[0] < 0:
            # handle the antipodal configuration
            q = -q
        A += w_i * (np.outer(q, q))  # rank 1 update
        wSum += w_i

    # scale
    A /= wSum

    # Get the eigenvector corresponding to largest eigen value
    return np.linalg.eigh(A)[1][:, -1]
