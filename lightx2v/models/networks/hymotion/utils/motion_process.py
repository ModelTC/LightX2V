import numpy as np
import torch


def smooth_quats(quats: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    from .geometry import gaussian_kernel1d, quaternion_fix_continuity, slice_seq_with_padding, wavg_quaternion_markley

    if len(quats) == 0 or sigma <= 0:
        return quats.copy()

    q_all = quaternion_fix_continuity(torch.from_numpy(quats)).numpy()

    results = q_all.copy()
    truncate = 4.0
    order = 0
    lw = int(truncate * float(sigma) + 0.5)
    weights = gaussian_kernel1d(sigma=sigma, order=order, radius=lw)[::-1]
    kernel_len = len(weights)

    for fr in range(len(q_all)):
        cur_quats = slice_seq_with_padding(q_all, fr, kernel_len)  # (K,4)
        ref = cur_quats[kernel_len // 2 : kernel_len // 2 + 1]  # (1,4)
        dots = (cur_quats * ref).sum(axis=-1, keepdims=True)  # (K,1)
        cur_quats = np.where(dots < 0.0, -cur_quats, cur_quats)

        results[fr, :] = wavg_quaternion_markley(cur_quats, weights)

    return results.copy()


def smooth_rotation(
    quats: np.ndarray,
    # joint_names: List[str],
    # smooth_joints: List[str],
    sigma: float = 1.0,
) -> np.ndarray:
    from .geometry import quaternion_fix_continuity

    if quats.ndim == 4:
        is_batch = True
    else:
        is_batch = False
        quats = quats[None, ...]
    for b in range(quats.shape[0]):
        for j_idx in range(quats.shape[2]):
            cur_quats = quats[b, :, j_idx].copy()
            cur_quats_t = quaternion_fix_continuity(torch.from_numpy(cur_quats)).numpy()
            quats[b, :, j_idx] = smooth_quats(cur_quats_t, sigma=sigma)
    if not is_batch:
        quats = quats.squeeze(0)
    return quats
