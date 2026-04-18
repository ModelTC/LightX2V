"""
Render interpolated video from Gaussian Splatting predictions.

Interpolates smooth camera trajectories using SLERP quaternions,
renders each frame via gsplat, and saves MP4 videos.
"""

import os
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from ..models.models.rasterization import GaussianSplatRenderer


def rotation_matrix_to_quaternion(R):
    """Convert rotation matrix to quaternion (scalar-first: [w, x, y, z]).

    Note: This uses the Hamilton convention [w, x, y, z], which differs from
    models/utils/rotation.py that uses PyTorch3D convention [x, y, z, w].
    """
    trace = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]

    q = torch.zeros(R.shape[:-2] + (4,), device=R.device, dtype=R.dtype)

    mask1 = trace > 0
    s = torch.sqrt(trace[mask1] + 1.0) * 2
    q[mask1, 0] = 0.25 * s
    q[mask1, 1] = (R[mask1, 2, 1] - R[mask1, 1, 2]) / s
    q[mask1, 2] = (R[mask1, 0, 2] - R[mask1, 2, 0]) / s
    q[mask1, 3] = (R[mask1, 1, 0] - R[mask1, 0, 1]) / s

    mask2 = (~mask1) & (R[..., 0, 0] > R[..., 1, 1]) & (R[..., 0, 0] > R[..., 2, 2])
    s = torch.sqrt(1.0 + R[mask2, 0, 0] - R[mask2, 1, 1] - R[mask2, 2, 2]) * 2
    q[mask2, 0] = (R[mask2, 2, 1] - R[mask2, 1, 2]) / s
    q[mask2, 1] = 0.25 * s
    q[mask2, 2] = (R[mask2, 0, 1] + R[mask2, 1, 0]) / s
    q[mask2, 3] = (R[mask2, 0, 2] + R[mask2, 2, 0]) / s

    mask3 = (~mask1) & (~mask2) & (R[..., 1, 1] > R[..., 2, 2])
    s = torch.sqrt(1.0 + R[mask3, 1, 1] - R[mask3, 0, 0] - R[mask3, 2, 2]) * 2
    q[mask3, 0] = (R[mask3, 0, 2] - R[mask3, 2, 0]) / s
    q[mask3, 1] = (R[mask3, 0, 1] + R[mask3, 1, 0]) / s
    q[mask3, 2] = 0.25 * s
    q[mask3, 3] = (R[mask3, 1, 2] + R[mask3, 2, 1]) / s

    mask4 = (~mask1) & (~mask2) & (~mask3)
    s = torch.sqrt(1.0 + R[mask4, 2, 2] - R[mask4, 0, 0] - R[mask4, 1, 1]) * 2
    q[mask4, 0] = (R[mask4, 1, 0] - R[mask4, 0, 1]) / s
    q[mask4, 1] = (R[mask4, 0, 2] + R[mask4, 2, 0]) / s
    q[mask4, 2] = (R[mask4, 1, 2] + R[mask4, 2, 1]) / s
    q[mask4, 3] = 0.25 * s

    return q


def quaternion_to_rotation_matrix(q):
    """Convert quaternion (scalar-first: [w, x, y, z]) to rotation matrix."""
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]

    norm = torch.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = w / norm, x / norm, y / norm, z / norm

    R = torch.zeros(q.shape[:-1] + (3, 3), device=q.device, dtype=q.dtype)

    R[..., 0, 0] = 1 - 2 * (y * y + z * z)
    R[..., 0, 1] = 2 * (x * y - w * z)
    R[..., 0, 2] = 2 * (x * z + w * y)
    R[..., 1, 0] = 2 * (x * y + w * z)
    R[..., 1, 1] = 1 - 2 * (x * x + z * z)
    R[..., 1, 2] = 2 * (y * z - w * x)
    R[..., 2, 0] = 2 * (x * z - w * y)
    R[..., 2, 1] = 2 * (y * z + w * x)
    R[..., 2, 2] = 1 - 2 * (x * x + y * y)

    return R


def slerp_quaternions(q1, q2, t):
    """Spherical linear interpolation between quaternions."""
    dot = (q1 * q2).sum(dim=-1, keepdim=True)

    mask = dot < 0
    q2 = torch.where(mask, -q2, q2)
    dot = torch.where(mask, -dot, dot)

    DOT_THRESHOLD = 0.9995
    mask_linear = dot > DOT_THRESHOLD

    result = torch.zeros_like(q1)

    if mask_linear.any():
        result_linear = q1 + t * (q2 - q1)
        norm = torch.norm(result_linear, dim=-1, keepdim=True)
        result_linear = result_linear / norm
        result = torch.where(mask_linear, result_linear, result)

    mask_slerp = ~mask_linear
    if mask_slerp.any():
        theta_0 = torch.acos(torch.abs(dot))
        sin_theta_0 = torch.sin(theta_0)

        theta = theta_0 * t
        sin_theta = torch.sin(theta)

        s0 = torch.cos(theta) - dot * sin_theta / sin_theta_0
        s1 = sin_theta / sin_theta_0

        result_slerp = (s0 * q1) + (s1 * q2)
        result = torch.where(mask_slerp, result_slerp, result)

    return result


def render_interpolated_video(
    gs_renderer: GaussianSplatRenderer,
    splats: dict,
    camtoworlds: torch.Tensor,
    intrinsics: torch.Tensor,
    hw: tuple,
    out_path: Path,
    interp_per_pair: int = 20,
    loop_reverse: bool = True,
    save_mode: str = "split",
    frame_times: list = None,
    render_depth: bool = False,
) -> None:
    """Render an interpolated fly-through video from Gaussian splat predictions.

    Args:
        gs_renderer: GaussianSplatRenderer instance (from the model).
        splats: Dict with keys 'means', 'scales', 'quats', 'opacities', 'sh'/'colors'.
        camtoworlds: Camera-to-world matrices [B, S, 4, 4].
        intrinsics: Camera intrinsic matrices [B, S, 3, 3].
        hw: Tuple of (height, width) for rendering.
        out_path: Output path (without extension).
        interp_per_pair: Number of interpolated frames per camera pair.
        loop_reverse: Append reversed video for smooth looping.
        save_mode: 'split' (separate rgb/depth) or 'both' (combined).
        frame_times: Optional list of timestamps for adaptive interpolation.
        render_depth: Whether to also render depth video.
    """
    import moviepy.editor as mpy

    b, s, _, _ = camtoworlds.shape
    h, w = hw

    def build_interpolated_traj(index, base_interp_per_pair: int):
        exts, ints = [], []
        tmp_camtoworlds = camtoworlds[:, index]
        tmp_intrinsics = intrinsics[:, index]

        use_time_based = frame_times is not None and len(frame_times) == len(index)
        if use_time_based and len(index) > 1:
            times = np.array([frame_times[i] for i in index], dtype=np.float32)
            gaps = np.diff(times)
            gaps[gaps < 0] = 0.0
            total_gap = float(gaps.sum())
            target_total_interp = max(1, (len(index) - 1) * base_interp_per_pair)
            gap_scale = target_total_interp / total_gap if total_gap > 1e-6 else 0.0
        else:
            gaps = None
            gap_scale = None

        for i in range(len(index) - 1):
            exts.append(tmp_camtoworlds[:, i : i + 1])
            ints.append(tmp_intrinsics[:, i : i + 1])
            R0, t0 = tmp_camtoworlds[:, i, :3, :3], tmp_camtoworlds[:, i, :3, 3]
            R1, t1 = tmp_camtoworlds[:, i + 1, :3, :3], tmp_camtoworlds[:, i + 1, :3, 3]

            q0 = rotation_matrix_to_quaternion(R0)
            q1 = rotation_matrix_to_quaternion(R1)

            if use_time_based:
                gap = float(gaps[i]) if gaps is not None else 0.0
                num_interp = max(0, int(round(gap * gap_scale)))
            else:
                num_interp = base_interp_per_pair

            for j in range(1, num_interp + 1):
                alpha = j / (num_interp + 1)
                t_interp = (1 - alpha) * t0 + alpha * t1
                q_interp = slerp_quaternions(q0, q1, alpha)
                R_interp = quaternion_to_rotation_matrix(q_interp)

                ext = torch.eye(4, device=R_interp.device, dtype=R_interp.dtype)[None].repeat(b, 1, 1)
                ext[:, :3, :3] = R_interp
                ext[:, :3, 3] = t_interp

                K0 = tmp_intrinsics[:, i]
                K1 = tmp_intrinsics[:, i + 1]
                K = (1 - alpha) * K0 + alpha * K1

                exts.append(ext[:, None])
                ints.append(K[:, None])

        exts = torch.cat(exts, dim=1)[:1]
        ints = torch.cat(ints, dim=1)[:1]
        return exts, ints

    def build_wobble_traj(nums, delta, anchor_idx=0, yaw_rad=0.0, pivot=None, forward_ratio=0.0):
        anchor_c2w = camtoworlds[:, anchor_idx : anchor_idx + 1]
        anchor_K = intrinsics[:, anchor_idx : anchor_idx + 1]
        need_rebuild = (yaw_rad != 0.0 or forward_ratio != 0.0) and pivot is not None
        if need_rebuild:
            device = camtoworlds.device
            dtype = torch.float32
            anc = anchor_c2w[0, 0].to(dtype)
            P = anc[:3, 3]
            R = anc[:3, :3]
            if yaw_rad != 0.0:
                U = -R[:, 1]
                U = U / (U.norm() + 1e-8)
                theta = torch.tensor(yaw_rad, device=device, dtype=dtype)
                cos_t, sin_t = torch.cos(theta), torch.sin(theta)
                K_mat = torch.zeros(3, 3, device=device, dtype=dtype)
                K_mat[0, 1], K_mat[0, 2] = -U[2], U[1]
                K_mat[1, 0], K_mat[1, 2] = U[2], -U[0]
                K_mat[2, 0], K_mat[2, 1] = -U[1], U[0]
                I3 = torch.eye(3, device=device, dtype=dtype)
                R_yaw = I3 + sin_t * K_mat + (1 - cos_t) * (K_mat @ K_mat)
                pivot_f = pivot.to(dtype)
                P = R_yaw @ (P - pivot_f) + pivot_f
                R = R_yaw @ R
            if forward_ratio != 0.0:
                pivot_f = pivot.to(dtype)
                P = P + (pivot_f - P) * float(forward_ratio)
            anc_new = torch.eye(4, device=device, dtype=dtype)
            anc_new[:3, :3] = R
            anc_new[:3, 3] = P
            anchor_c2w = anc_new[None, None].expand(camtoworlds.shape[0], 1, 4, 4).contiguous()
        t = torch.linspace(0, 1, nums, dtype=torch.float32, device=camtoworlds.device)
        t = (torch.cos(torch.pi * (t + 1)) + 1) / 2
        tf = torch.eye(4, dtype=torch.float32, device=camtoworlds.device)
        radius = delta * 0.15
        tf = tf.broadcast_to((*radius.shape, t.shape[0], 4, 4)).clone()
        radius = radius[..., None]
        radius = radius * t
        tf[..., 0, 3] = torch.sin(2 * torch.pi * t) * radius
        tf[..., 1, 3] = -torch.cos(2 * torch.pi * t) * radius
        exts = anchor_c2w @ tf
        ints = anchor_K.repeat(1, exts.shape[1], 1, 1)
        return exts, ints

    render_traj = os.environ.get("RENDER_TRAJ", "interp").lower()
    yaw_deg = float(os.environ.get("RENDER_TRAJ_YAW_DEG", "0"))
    forward_ratio = float(os.environ.get("RENDER_TRAJ_FORWARD", "0"))
    yaw_rad = yaw_deg * torch.pi / 180.0

    if render_traj == "wobble":
        scene_center = splats["means"][0].median(dim=0).values
        all_ext, all_int = build_wobble_traj(
            interp_per_pair * 12,
            scene_center.norm(dim=-1)[None],
            anchor_idx=s // 2,
            yaw_rad=yaw_rad,
            pivot=scene_center,
            forward_ratio=forward_ratio,
        )
        loop_reverse = True
    elif s > 1:
        all_ext, all_int = build_interpolated_traj([i for i in range(s)], interp_per_pair)
        if yaw_deg != 0.0 or forward_ratio != 0.0:
            device = all_ext.device
            dtype = torch.float32
            scene_center = splats["means"][0].median(dim=0).values.to(dtype)
            anc = camtoworlds[:, s // 2][0].to(dtype)
            U = -anc[:3, 1]
            U = U / (U.norm() + 1e-8)
            theta = torch.tensor(yaw_rad, device=device, dtype=dtype)
            cos_t, sin_t = torch.cos(theta), torch.sin(theta)
            K_mat = torch.zeros(3, 3, device=device, dtype=dtype)
            K_mat[0, 1], K_mat[0, 2] = -U[2], U[1]
            K_mat[1, 0], K_mat[1, 2] = U[2], -U[0]
            K_mat[2, 0], K_mat[2, 1] = -U[1], U[0]
            I3 = torch.eye(3, device=device, dtype=dtype)
            R_yaw = I3 + sin_t * K_mat + (1 - cos_t) * (K_mat @ K_mat)
            all_ext_f = all_ext.to(dtype).clone()
            P = all_ext_f[..., :3, 3]
            R = all_ext_f[..., :3, :3]
            if yaw_deg != 0.0:
                P_shift = P - scene_center
                P_rot = (R_yaw[None, None] @ P_shift[..., None]).squeeze(-1)
                P = P_rot + scene_center
                R = R_yaw[None, None] @ R
            if forward_ratio != 0.0:
                P = P + (scene_center - P) * forward_ratio
            all_ext_f[..., :3, :3] = R
            all_ext_f[..., :3, 3] = P
            all_ext = all_ext_f
    else:
        all_ext, all_int = build_wobble_traj(interp_per_pair * 12, splats["means"][0].median(dim=0).values.norm(dim=-1)[None])

    rendered_rgbs, rendered_depths = [], []
    chunk = 40

    # Always prune splats to remove scale outliers
    try:
        pruned_splats = gs_renderer.prune_gs(splats, gs_renderer.voxel_size)
    except (AttributeError, RuntimeError):
        pruned_splats = splats

    for st in tqdm(range(0, all_ext.shape[1], chunk)):
        ed = min(st + chunk, all_ext.shape[1])
        colors, depths, _ = gs_renderer.rasterizer.rasterize_batches(
            pruned_splats["means"][:1],
            pruned_splats["quats"][:1],
            pruned_splats["scales"][:1],
            pruned_splats["opacities"][:1],
            pruned_splats["sh"][:1] if "sh" in pruned_splats else pruned_splats["colors"][:1],
            all_ext[:, st:ed].to(torch.float32),
            all_int[:, st:ed].to(torch.float32),
            width=w,
            height=h,
            sh_degree=gs_renderer.sh_degree if "sh" in pruned_splats else None,
        )
        rendered_rgbs.append(colors)
        if render_depth:
            rendered_depths.append(depths)

    rgbs = torch.cat(rendered_rgbs, dim=1)[0]  # [N, H, W, 3]
    if render_depth:
        depths_all = torch.cat(rendered_depths, dim=1)[0, ..., 0]  # [N, H, W]
    del rendered_rgbs, rendered_depths

    def _depth_vis(d: torch.Tensor) -> torch.Tensor:
        """Simple turbo colormap depth visualization."""
        import matplotlib.pyplot as plt

        valid = d > 0
        if valid.any():
            near = d[valid].float().quantile(0.01).log()
        else:
            near = torch.tensor(0.0, device=d.device)
        far = d.flatten().float().quantile(0.99).log()
        x = d.float().clamp(min=1e-9).log()
        x = 1.0 - (x - near) / (far - near + 1e-9)
        x_np = x.cpu().numpy()
        colored = torch.from_numpy(plt.cm.turbo(x_np)[..., :3]).permute(2, 0, 1).float()
        return colored

    rgb_frames = []
    depth_frames = []

    if render_depth:
        for rgb, dep in zip(rgbs, depths_all):
            rgb_frames.append(rgb.permute(2, 0, 1))
            depth_frames.append(_depth_vis(dep))
    else:
        for rgb in rgbs:
            rgb_frames.append(rgb.permute(2, 0, 1))

    def _make_video(frames, path):
        video = torch.stack([f.cpu() for f in frames]).clamp(0, 1)
        video = video.permute(0, 2, 3, 1)
        video = (video * 255).to(torch.uint8).numpy()
        if loop_reverse and video.shape[0] > 1:
            video = np.concatenate([video, video[::-1][1:-1]], axis=0)
        clip = mpy.ImageSequenceClip(list(video), fps=30)
        clip.write_videofile(str(path), logger=None)

    out_path = Path(out_path)
    out_path.mkdir(parents=True, exist_ok=True)
    if save_mode == "split":
        _make_video(rgb_frames, out_path / "rendered_rgb.mp4")
        if render_depth:
            _make_video(depth_frames, out_path / "rendered_depth.mp4")
    elif save_mode == "both" and render_depth:
        combined = [torch.cat([r, d], dim=1) for r, d in zip(rgb_frames, depth_frames)]
        _make_video(combined, out_path / "rendered.mp4")

    print(f"Video saved to {out_path} (mode: {save_mode})")
    torch.cuda.empty_cache()
