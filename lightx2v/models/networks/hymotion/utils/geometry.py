import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor


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
