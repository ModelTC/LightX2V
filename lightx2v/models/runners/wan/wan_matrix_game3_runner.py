import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.distributed as dist
import torchvision.transforms.functional as TF
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.schedulers.scheduling_utils import KarrasDiffusionSchedulers, SchedulerMixin, SchedulerOutput
from diffusers.utils import deprecate
from PIL import Image
from einops import rearrange
from loguru import logger
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation, Slerp

from lightx2v.models.runners.wan.wan_runner import Wan22DenseRunner, build_wan_model_with_lora
from lightx2v.models.schedulers.scheduler import BaseScheduler
from lightx2v.server.metrics import monitor_cli
from lightx2v.utils.envs import GET_DTYPE
from lightx2v.utils.profiler import GET_RECORDER_MODE, ProfilingContext4DebugL1, ProfilingContext4DebugL2
from lightx2v.utils.registry_factory import RUNNER_REGISTER
from lightx2v_platform.base.global_var import AI_DEVICE

torch_device_module = getattr(torch, AI_DEVICE)


_PROJECT_ROOT = Path(__file__).resolve().parents[4]
_MATRIX_GAME3_CONFIG_ROOT_RELATIVE = Path("Matrix-Game-3.0")
_MATRIX_GAME3_DEFAULT_NEGATIVE_PROMPT = (
    "Vibrant colors, overexposure, static, blurred details, subtitles, style, artwork, "
    "painting, still image, overall grayness, worst quality, low quality, JPEG compression "
    "residue, ugly, mutilated, extra fingers, poorly drawn hands, poorly drawn faces, "
    "deformed, disfigured, malformed limbs, fused fingers, still image, cluttered background, "
    "three legs, crowded background, walking backwards"
)
_MATRIX_GAME3_WSAD_OFFSET = 12.35
_MATRIX_GAME3_DIAGONAL_OFFSET = 8.73
_MATRIX_GAME3_MOUSE_PITCH_SENSITIVITY = 15.0
_MATRIX_GAME3_MOUSE_YAW_SENSITIVITY = 15.0
_MATRIX_GAME3_MOUSE_THRESHOLD = 0.02


@dataclass
class MatrixGame3SegmentState:
    """Precomputed inputs and bookkeeping for one Matrix-Game-3 segment.

    The runner generates video in overlapping chunks. For each chunk we cache:
    - the absolute frame window covered by this segment;
    - the latent tensor shape the scheduler should sample;
    - how many latent frames are fixed by conditioning instead of sampled;
    - the condition tensors that will be forwarded through `dit_cond_dict`;
    - how many decoded RGB frames should be trimmed before concatenation.
    """

    segment_idx: int
    first_clip: bool
    current_start_frame_idx: int
    current_end_frame_idx: int
    frame_count: int
    fixed_latent_frames: int
    latent_shape: list[int]
    decode_trim_frames: int
    append_latent_start: int
    keyboard_cond: torch.Tensor
    mouse_cond: torch.Tensor
    vae_encoder_out: torch.Tensor
    dit_cond_dict: dict[str, Any]


def _expand_path_candidates(path_value: Any) -> list[Path]:
    """Resolve a user-provided path against cwd and the project root when needed."""
    raw_path = Path(str(path_value)).expanduser()
    if raw_path.is_absolute():
        return [raw_path]
    candidates = [Path.cwd() / raw_path]
    project_relative = _PROJECT_ROOT / raw_path
    if project_relative != candidates[0]:
        candidates.append(project_relative)
    return candidates


def _matrix_game3_combine_data(data, num_frames=57, keyboard_dim=4, mouse=True):
    assert num_frames % 4 == 1
    keyboard_condition = torch.zeros((num_frames, keyboard_dim))
    if mouse:
        mouse_condition = torch.zeros((num_frames, 2))

    current_frame = 0
    selections = [12]

    while current_frame < num_frames:
        rd_frame = selections[random.randint(0, len(selections) - 1)]
        rd = random.randint(0, len(data) - 1)
        keyboard_sample = data[rd]["keyboard_condition"]
        if mouse:
            mouse_sample = data[rd]["mouse_condition"]

        if current_frame == 0:
            keyboard_condition[:1] = keyboard_sample[:1]
            if mouse:
                mouse_condition[:1] = mouse_sample[:1]
            current_frame = 1
        else:
            rd_frame = min(rd_frame, num_frames - current_frame)
            repeat_time = rd_frame // 4
            keyboard_condition[current_frame : current_frame + rd_frame] = keyboard_sample.repeat(repeat_time, 1)
            if mouse:
                mouse_condition[current_frame : current_frame + rd_frame] = mouse_sample.repeat(repeat_time, 1)
            current_frame += rd_frame

    if mouse:
        return {
            "keyboard_condition": keyboard_condition,
            "mouse_condition": mouse_condition,
        }
    return {"keyboard_condition": keyboard_condition}


def _matrix_game3_basic_tensor_probe(tensor: torch.Tensor, head_values: int = 8) -> dict[str, Any]:
    tensor = tensor.detach()
    tensor_fp32 = tensor.to(dtype=torch.float32)
    flattened = tensor_fp32.reshape(-1)
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "device": str(tensor.device),
        "min": float(tensor_fp32.min().item()),
        "max": float(tensor_fp32.max().item()),
        "mean": float(tensor_fp32.mean().item()),
        "std": float(tensor_fp32.std(unbiased=False).item()),
        "head": flattened[:head_values].cpu().tolist(),
    }


def _matrix_game3_tensor_probe(tensor: torch.Tensor, mask: Optional[torch.Tensor] = None, head_values: int = 8) -> dict[str, Any]:
    probe = {"full_latent": _matrix_game3_basic_tensor_probe(tensor, head_values=head_values)}
    if mask is None or mask.shape != tensor.shape:
        return probe

    mask_fp32 = mask.detach().to(dtype=torch.float32)
    time_activity = mask_fp32.amax(dim=(0, 2, 3))
    sampled_indices = torch.nonzero(time_activity > 0.0, as_tuple=False).flatten()
    fixed_latent_frames = int(sampled_indices[0].item()) if sampled_indices.numel() > 0 else tensor.shape[1]
    probe["fixed_latent_frames"] = fixed_latent_frames

    if fixed_latent_frames < tensor.shape[1]:
        sampled_region = tensor[:, fixed_latent_frames:]
        probe["sampled_region"] = _matrix_game3_basic_tensor_probe(sampled_region, head_values=head_values)
    else:
        probe["sampled_region"] = "empty (all latent frames are fixed)"

    return probe


def _matrix_game3_log_debug_payload(name: str, payload: dict[str, Any]) -> None:
    logger.info("[matrix-game-3][debug] {}:\n{}", name, json.dumps(payload, ensure_ascii=False, indent=4, default=str))


def _matrix_game3_resolve_debug_first_frame_path(save_result_path: Optional[str]) -> Path:
    if save_result_path:
        output_path = Path(save_result_path)
        suffix = output_path.suffix if output_path.suffix else ".mp4"
        return output_path.with_name(output_path.name[: -len(suffix)] + ".debug_first_frame.png")
    return Path.cwd() / "matrix_game3.debug_first_frame.png"


def _matrix_game3_resolve_debug_frame_path(save_result_path: Optional[str], frame_index: int) -> Path:
    if save_result_path:
        output_path = Path(save_result_path)
        suffix = output_path.suffix if output_path.suffix else ".mp4"
        return output_path.with_name(output_path.name[: -len(suffix)] + f".debug_frame_{frame_index}.png")
    return Path.cwd() / f"matrix_game3.debug_frame_{frame_index}.png"


def _matrix_game3_resolve_debug_steps_dir(save_result_path: Optional[str]) -> Path:
    if save_result_path:
        output_path = Path(save_result_path)
        suffix = output_path.suffix if output_path.suffix else ".mp4"
        return output_path.with_name(output_path.name[: -len(suffix)] + ".debug_steps")
    return Path.cwd() / "matrix_game3.debug_steps"


def _matrix_game3_bench_actions_universal(num_frames, num_samples_per_action=4):
    actions_single_action = [
        "forward",
        "left",
        "right",
    ]
    actions_double_action = [
        "forward_left",
        "forward_right",
    ]

    actions_single_camera = [
        "camera_l",
        "camera_r",
    ]
    actions_to_test = actions_double_action * 5 + actions_single_camera * 5 + actions_single_action * 5
    for action in (actions_single_action + actions_double_action):
        for camera in actions_single_camera:
            actions_to_test.append(f"{action}_{camera}")

    base_action = actions_single_action + actions_single_camera
    keyboard_idx = {
        "forward": 0,
        "back": 1,
        "left": 2,
        "right": 3,
    }
    cam_value = 0.1
    camera_value_map = {
        "camera_up": [cam_value, 0],
        "camera_down": [-cam_value, 0],
        "camera_l": [0, -cam_value],
        "camera_r": [0, cam_value],
        "camera_ur": [cam_value, cam_value],
        "camera_ul": [cam_value, -cam_value],
        "camera_dr": [-cam_value, cam_value],
        "camera_dl": [-cam_value, -cam_value],
    }

    data = []
    for action_name in actions_to_test:
        keyboard_condition = [[0, 0, 0, 0, 0, 0] for _ in range(num_samples_per_action)]
        mouse_condition = [[0, 0] for _ in range(num_samples_per_action)]

        for sub_action in base_action:
            if sub_action not in action_name:
                continue
            if sub_action in camera_value_map:
                mouse_condition = [camera_value_map[sub_action] for _ in range(num_samples_per_action)]
            elif sub_action in keyboard_idx:
                col = keyboard_idx[sub_action]
                for row in keyboard_condition:
                    row[col] = 1

        data.append(
            {
                "keyboard_condition": torch.tensor(keyboard_condition),
                "mouse_condition": torch.tensor(mouse_condition),
            }
        )

    return _matrix_game3_combine_data(data, num_frames, keyboard_dim=6, mouse=True)


def _matrix_game3_compute_next_pose_from_action(current_pose, keyboard_action, mouse_action):
    x, y, z, pitch, yaw = current_pose
    w, s, a, d = keyboard_action[:4]
    mouse_x, mouse_y = mouse_action[:2]

    delta_pitch = _MATRIX_GAME3_MOUSE_PITCH_SENSITIVITY * mouse_x if abs(mouse_x) >= _MATRIX_GAME3_MOUSE_THRESHOLD else 0.0
    delta_yaw = _MATRIX_GAME3_MOUSE_YAW_SENSITIVITY * mouse_y if abs(mouse_y) >= _MATRIX_GAME3_MOUSE_THRESHOLD else 0.0

    new_pitch = pitch + delta_pitch
    new_yaw = yaw + delta_yaw

    while new_yaw > 180:
        new_yaw -= 360
    while new_yaw < -180:
        new_yaw += 360

    local_forward = 0.0
    if w > 0.5 and s < 0.5:
        local_forward = _MATRIX_GAME3_WSAD_OFFSET
    elif s > 0.5 and w < 0.5:
        local_forward = -_MATRIX_GAME3_WSAD_OFFSET

    local_right = 0.0
    if d > 0.5 and a < 0.5:
        local_right = _MATRIX_GAME3_WSAD_OFFSET
    elif a > 0.5 and d < 0.5:
        local_right = -_MATRIX_GAME3_WSAD_OFFSET

    if abs(local_forward) > 0.1 and abs(local_right) > 0.1:
        local_forward = np.sign(local_forward) * _MATRIX_GAME3_DIAGONAL_OFFSET
        local_right = np.sign(local_right) * _MATRIX_GAME3_DIAGONAL_OFFSET

    avg_yaw = float((yaw + new_yaw) / 2.0)
    yaw_rad = float(np.deg2rad(avg_yaw))
    cos_yaw = np.cos(yaw_rad)
    sin_yaw = np.sin(yaw_rad)

    delta_x = cos_yaw * local_forward - sin_yaw * local_right
    delta_y = sin_yaw * local_forward + cos_yaw * local_right
    return np.array([x + delta_x, y + delta_y, z, new_pitch, new_yaw], dtype=np.float32)


def _matrix_game3_compute_all_poses_from_actions(keyboard_conditions, mouse_conditions, first_pose=None, return_last_pose=False):
    total_frames = len(keyboard_conditions)
    all_poses = np.zeros((total_frames, 5), dtype=np.float32)
    if first_pose is not None:
        all_poses[0] = first_pose

    for idx in range(total_frames - 1):
        all_poses[idx + 1] = _matrix_game3_compute_next_pose_from_action(
            all_poses[idx],
            keyboard_conditions[idx],
            mouse_conditions[idx],
        )

    if return_last_pose:
        last_pose = _matrix_game3_compute_next_pose_from_action(
            all_poses[-1],
            keyboard_conditions[-1],
            mouse_conditions[-1],
        )
        return all_poses, last_pose
    return all_poses


def _matrix_game3_interpolate_camera_poses(src_indices, src_rot_mat, src_trans_vec, tgt_indices):
    interp_func_trans = interp1d(
        src_indices,
        src_trans_vec,
        axis=0,
        kind="linear",
        bounds_error=False,
        fill_value="extrapolate",
    )
    interpolated_trans_vec = interp_func_trans(tgt_indices)

    src_quat_vec = Rotation.from_matrix(src_rot_mat)
    quats = src_quat_vec.as_quat().copy()
    for idx in range(1, len(quats)):
        if np.dot(quats[idx], quats[idx - 1]) < 0:
            quats[idx] = -quats[idx]
    src_quat_vec = Rotation.from_quat(quats)
    slerp_func_rot = Slerp(src_indices, src_quat_vec)
    interpolated_rot_quat = slerp_func_rot(tgt_indices)
    interpolated_rot_mat = interpolated_rot_quat.as_matrix()

    poses = np.zeros((len(tgt_indices), 4, 4), dtype=np.float32)
    poses[:, :3, :3] = interpolated_rot_mat
    poses[:, :3, 3] = interpolated_trans_vec
    poses[:, 3, 3] = 1.0
    return torch.from_numpy(poses).float()


def _matrix_game3_se3_inverse(transform):
    rotation = transform[:, :3, :3]
    translation = transform[:, :3, 3:]
    rotation_inv = rotation.transpose(-1, -2)
    translation_inv = -torch.bmm(rotation_inv, translation)
    inverse = torch.eye(4, device=transform.device, dtype=transform.dtype)[None, :, :].repeat(transform.shape[0], 1, 1)
    inverse[:, :3, :3] = rotation_inv
    inverse[:, :3, 3:] = translation_inv
    return inverse


def _matrix_game3_compute_relative_poses(c2ws_mat, framewise=False, normalize_trans=True):
    ref_w2cs = _matrix_game3_se3_inverse(c2ws_mat[0:1])
    relative_poses = torch.matmul(ref_w2cs, c2ws_mat)
    relative_poses[0] = torch.eye(4, device=c2ws_mat.device, dtype=c2ws_mat.dtype)
    if framewise:
        relative_poses_framewise = torch.bmm(_matrix_game3_se3_inverse(relative_poses[:-1]), relative_poses[1:])
        relative_poses[1:] = relative_poses_framewise
    if normalize_trans:
        translations = relative_poses[:, :3, 3]
        max_norm = torch.norm(translations, dim=-1).max()
        if max_norm > 0:
            relative_poses[:, :3, 3] = translations / max_norm
    return relative_poses


@torch.no_grad()
def _matrix_game3_create_meshgrid(n_frames, height, width, bias=0.5, device="cuda", dtype=torch.float32):
    x_range = torch.arange(width, device=device, dtype=dtype)
    y_range = torch.arange(height, device=device, dtype=dtype)
    grid_y, grid_x = torch.meshgrid(y_range, x_range, indexing="ij")
    grid_xy = torch.stack([grid_x, grid_y], dim=-1).view([-1, 2]) + bias
    return grid_xy[None, ...].repeat(n_frames, 1, 1)


def _matrix_game3_get_plucker_embeddings(c2ws_mat, intrinsics, height, width):
    n_frames = c2ws_mat.shape[0]
    grid_xy = _matrix_game3_create_meshgrid(n_frames, height, width, device=c2ws_mat.device, dtype=c2ws_mat.dtype)
    fx, fy, cx, cy = intrinsics.chunk(4, dim=-1)
    i = grid_xy[..., 0]
    j = grid_xy[..., 1]
    zs = torch.ones_like(i)
    xs = (i - cx) / fx * zs
    ys = (j - cy) / fy * zs

    directions = torch.stack([xs, ys, zs], dim=-1)
    directions = directions / directions.norm(dim=-1, keepdim=True)
    rays_d = directions @ c2ws_mat[:, :3, :3].transpose(-1, -2)
    rays_o = c2ws_mat[:, :3, 3]
    rays_o = rays_o[:, None, :].expand_as(rays_d)
    return torch.cat([rays_o, rays_d], dim=-1).view([n_frames, height, width, 6])


def _matrix_game3_select_memory_idx_fov(extrinsics_all, current_start_frame_idx, selected_index_base, return_confidence=False, use_gpu=False):
    if not use_gpu:
        use_gpu = True

    device = extrinsics_all.device if isinstance(extrinsics_all, torch.Tensor) else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if isinstance(extrinsics_all, np.ndarray):
        extrinsics_tensor = torch.from_numpy(extrinsics_all).to(device).float()
    else:
        extrinsics_tensor = extrinsics_all.to(device).float()

    video_w, video_h = 1280, 720
    fov_rad = np.deg2rad(90)
    fx = video_w / (2 * np.tan(fov_rad / 2))
    fy = video_h / (2 * np.tan(fov_rad / 2))

    if current_start_frame_idx <= 1:
        empty_index = [0] * len(selected_index_base)
        empty_conf = [0.0] * len(selected_index_base)
        return (empty_index, empty_conf) if return_confidence else empty_index

    candidate_indices = torch.arange(1, current_start_frame_idx, device=device)
    rotation = extrinsics_tensor[candidate_indices, :3, :3]
    translation = extrinsics_tensor[candidate_indices, :3, 3:4]
    rotation_inv = rotation.transpose(1, 2)
    translation_inv = -torch.bmm(rotation_inv, translation)

    selected_index = []
    selected_confidence = []
    near, far = 0.1, 30.0
    num_side = 10
    z_samples = torch.linspace(near, far, num_side, device=device)
    x_samples = torch.linspace(-1, 1, num_side, device=device)
    y_samples = torch.linspace(-1, 1, num_side, device=device)
    grid_x, grid_y, grid_z = torch.meshgrid(x_samples, y_samples, z_samples, indexing="ij")
    points_cam_base = torch.stack(
        [
            grid_x.reshape(-1) * grid_z.reshape(-1) * (video_w / (2 * fx)),
            grid_y.reshape(-1) * grid_z.reshape(-1) * (video_h / (2 * fy)),
            grid_z.reshape(-1),
        ],
        dim=0,
    )

    for base_idx in selected_index_base:
        extrinsics = extrinsics_tensor[base_idx]
        points_world = extrinsics[:3, :3] @ points_cam_base + extrinsics[:3, 3:4]
        points_world_batched = points_world.unsqueeze(0)
        points_in_candidates = torch.bmm(rotation_inv, points_world_batched.expand(len(candidate_indices), -1, -1)) + translation_inv

        x = points_in_candidates[:, 0, :]
        y = points_in_candidates[:, 1, :]
        z = points_in_candidates[:, 2, :]
        u = (x * fx / torch.clamp(z, min=1e-6)) + video_w / 2
        v = (y * fy / torch.clamp(z, min=1e-6)) + video_h / 2

        in_view = (z > near) & (z < far) & (u >= 0) & (u <= video_w) & (v >= 0) & (v <= video_h)
        ratios = in_view.float().mean(dim=1)
        best_idx = torch.argmax(ratios)
        selected_index.append(candidate_indices[best_idx].item())
        selected_confidence.append(ratios[best_idx].item())

    return (selected_index, selected_confidence) if return_confidence else selected_index


def _matrix_game3_get_extrinsics(video_rotation, video_position):
    num_frames = len(video_rotation)
    extrinsics_vid = []
    for idx in range(num_frames):
        roll, pitch, yaw = video_rotation[idx]
        roll, pitch, yaw = np.radians([roll, pitch, yaw])

        rotation_z = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
        rotation_y = np.array([[np.cos(pitch), 0, np.sin(pitch)], [0, 1, 0], [-np.sin(pitch), 0, np.cos(pitch)]])
        rotation_x = np.array([[1, 0, 0], [0, np.cos(roll), -np.sin(roll)], [0, np.sin(roll), np.cos(roll)]])
        rotation = rotation_z @ rotation_y @ rotation_x

        extrinsics = np.eye(4, dtype=np.float32)
        extrinsics[:3, :3] = rotation
        extrinsics[:3, 3] = video_position[idx]
        extrinsics_vid.append(extrinsics)

    rotation_init = np.array(
        [
            [0, 0, 1],
            [1, 0, 0],
            [0, -1, 0],
        ],
        dtype=np.float32,
    )
    extrinsics = torch.from_numpy(np.array(extrinsics_vid, dtype=np.float32))
    extrinsics[:, :3, :3] = extrinsics[:, :3, :3] @ rotation_init
    extrinsics[:, :3, 3] = extrinsics[:, :3, 3] * 0.01
    return extrinsics


def _matrix_game3_get_intrinsics(height, width):
    fov_deg = 90
    fov_rad = np.deg2rad(fov_deg)
    fx = width / (2 * np.tan(fov_rad / 2))
    fy = height / (2 * np.tan(fov_rad / 2))
    cx = width / 2
    cy = height / 2
    return torch.tensor([fx, fy, cx, cy])


def _matrix_game3_interpolate_camera_poses_handedness(src_indices, src_rot_mat, src_trans_vec, tgt_indices):
    dets = np.linalg.det(src_rot_mat)
    flip_handedness = dets.size > 0 and np.median(dets) < 0.0
    if flip_handedness:
        flip_mat = np.diag([1.0, 1.0, -1.0]).astype(src_rot_mat.dtype)
        src_rot_mat = src_rot_mat @ flip_mat
    c2ws = _matrix_game3_interpolate_camera_poses(
        src_indices=src_indices,
        src_rot_mat=src_rot_mat,
        src_trans_vec=src_trans_vec,
        tgt_indices=tgt_indices,
    )
    if flip_handedness:
        flip_mat_t = torch.from_numpy(flip_mat).to(c2ws.device, dtype=c2ws.dtype)
        c2ws[:, :3, :3] = c2ws[:, :3, :3] @ flip_mat_t
    return c2ws


class _MatrixGame3ConditionsShim:
    Bench_actions_universal = staticmethod(_matrix_game3_bench_actions_universal)


class _MatrixGame3UtilsShim:
    compute_all_poses_from_actions = staticmethod(_matrix_game3_compute_all_poses_from_actions)


class _MatrixGame3CamUtilsShim:
    _interpolate_camera_poses_handedness = staticmethod(_matrix_game3_interpolate_camera_poses_handedness)
    compute_relative_poses = staticmethod(_matrix_game3_compute_relative_poses)
    get_plucker_embeddings = staticmethod(_matrix_game3_get_plucker_embeddings)
    select_memory_idx_fov = staticmethod(_matrix_game3_select_memory_idx_fov)
    get_extrinsics = staticmethod(_matrix_game3_get_extrinsics)
    get_intrinsics = staticmethod(_matrix_game3_get_intrinsics)


class FlowUniPCMultistepScheduler(SchedulerMixin, ConfigMixin):
    """Inlined Matrix-Game-3 FlowUniPC scheduler from the official implementation."""

    _compatibles = [scheduler.name for scheduler in KarrasDiffusionSchedulers]
    order = 1

    @register_to_config
    def __init__(
        self,
        num_train_timesteps: int = 1000,
        solver_order: int = 2,
        prediction_type: str = "flow_prediction",
        shift: Optional[float] = 1.0,
        use_dynamic_shifting: bool = False,
        thresholding: bool = False,
        dynamic_thresholding_ratio: float = 0.995,
        sample_max_value: float = 1.0,
        predict_x0: bool = True,
        solver_type: str = "bh2",
        lower_order_final: bool = True,
        disable_corrector: List[int] = [],
        solver_p: SchedulerMixin = None,
        timestep_spacing: str = "linspace",
        steps_offset: int = 0,
        final_sigmas_type: Optional[str] = "zero",
    ):
        if solver_type not in ["bh1", "bh2"]:
            if solver_type in ["midpoint", "heun", "logrho"]:
                self.register_to_config(solver_type="bh2")
            else:
                raise NotImplementedError(f"{solver_type} is not implemented for {self.__class__}")

        self.predict_x0 = predict_x0
        self.num_inference_steps = None
        alphas = np.linspace(1, 1 / num_train_timesteps, num_train_timesteps)[::-1].copy()
        sigmas = 1.0 - alphas
        sigmas = torch.from_numpy(sigmas).to(dtype=torch.float32)

        if not use_dynamic_shifting:
            sigmas = shift * sigmas / (1 + (shift - 1) * sigmas)

        self.sigmas = sigmas
        self.timesteps = sigmas * num_train_timesteps
        self.model_outputs = [None] * solver_order
        self.timestep_list = [None] * solver_order
        self.lower_order_nums = 0
        self.disable_corrector = disable_corrector
        self.solver_p = solver_p
        self.last_sample = None
        self._step_index = None
        self._begin_index = None
        self.sigmas = self.sigmas.to("cpu")
        self.sigma_min = self.sigmas[-1].item()
        self.sigma_max = self.sigmas[0].item()

    @property
    def step_index(self):
        return self._step_index

    @property
    def begin_index(self):
        return self._begin_index

    def set_begin_index(self, begin_index: int = 0):
        self._begin_index = begin_index

    def set_timesteps(
        self,
        num_inference_steps: Union[int, None] = None,
        device: Union[str, torch.device] = None,
        sigmas: Optional[List[float]] = None,
        mu: Optional[Union[float, None]] = None,
        shift: Optional[Union[float, None]] = None,
    ):
        if self.config.use_dynamic_shifting and mu is None:
            raise ValueError("you have to pass a value for `mu` when `use_dynamic_shifting` is set to be `True`")

        if sigmas is None:
            sigmas = np.linspace(self.sigma_max, self.sigma_min, num_inference_steps + 1).copy()[:-1]

        if self.config.use_dynamic_shifting:
            sigmas = self.time_shift(mu, 1.0, sigmas)
        else:
            if shift is None:
                shift = self.config.shift
            sigmas = shift * sigmas / (1 + (shift - 1) * sigmas)

        if self.config.final_sigmas_type == "sigma_min":
            sigma_last = ((1 - self.alphas_cumprod[0]) / self.alphas_cumprod[0]) ** 0.5
        elif self.config.final_sigmas_type == "zero":
            sigma_last = 0
        else:
            raise ValueError(f"`final_sigmas_type` must be one of 'zero', or 'sigma_min', but got {self.config.final_sigmas_type}")

        timesteps = sigmas * self.config.num_train_timesteps
        sigmas = np.concatenate([sigmas, [sigma_last]]).astype(np.float32)
        self.sigmas = torch.from_numpy(sigmas)
        self.timesteps = torch.from_numpy(timesteps).to(device=device, dtype=torch.int64)
        self.num_inference_steps = len(timesteps)
        self.model_outputs = [None] * self.config.solver_order
        self.lower_order_nums = 0
        self.last_sample = None
        if self.solver_p:
            self.solver_p.set_timesteps(self.num_inference_steps, device=device)
        self._step_index = None
        self._begin_index = None
        self.sigmas = self.sigmas.to("cpu")

    def _threshold_sample(self, sample: torch.Tensor) -> torch.Tensor:
        dtype = sample.dtype
        batch_size, channels, *remaining_dims = sample.shape
        if dtype not in (torch.float32, torch.float64):
            sample = sample.float()
        sample = sample.reshape(batch_size, channels * np.prod(remaining_dims))
        abs_sample = sample.abs()
        s = torch.quantile(abs_sample, self.config.dynamic_thresholding_ratio, dim=1)
        s = torch.clamp(s, min=1, max=self.config.sample_max_value)
        s = s.unsqueeze(1)
        sample = torch.clamp(sample, -s, s) / s
        sample = sample.reshape(batch_size, channels, *remaining_dims)
        return sample.to(dtype)

    def _sigma_to_t(self, sigma):
        return sigma * self.config.num_train_timesteps

    def _sigma_to_alpha_sigma_t(self, sigma):
        return 1 - sigma, sigma

    def time_shift(self, mu: float, sigma: float, t: torch.Tensor):
        return math.exp(mu) / (math.exp(mu) + (1 / t - 1) ** sigma)

    def convert_model_output(self, model_output: torch.Tensor, *args, sample: torch.Tensor = None, **kwargs) -> torch.Tensor:
        timestep = args[0] if len(args) > 0 else kwargs.pop("timestep", None)
        if sample is None:
            if len(args) > 1:
                sample = args[1]
            else:
                raise ValueError("missing `sample` as a required keyward argument")
        if timestep is not None:
            deprecate(
                "timesteps",
                "1.0.0",
                "Passing `timesteps` is deprecated and has no effect as model output conversion is now handled via an internal counter `self.step_index`",
            )

        sigma = self.sigmas[self.step_index]
        _, sigma_t = self._sigma_to_alpha_sigma_t(sigma)

        if self.predict_x0:
            if self.config.prediction_type == "flow_prediction":
                sigma_t = self.sigmas[self.step_index]
                x0_pred = sample - sigma_t * model_output
            else:
                raise ValueError(
                    f"prediction_type given as {self.config.prediction_type} must be one of `epsilon`, `sample`, `v_prediction` or `flow_prediction` for the UniPCMultistepScheduler."
                )
            if self.config.thresholding:
                x0_pred = self._threshold_sample(x0_pred)
            return x0_pred

        if self.config.prediction_type == "flow_prediction":
            sigma_t = self.sigmas[self.step_index]
            epsilon = sample - (1 - sigma_t) * model_output
        else:
            raise ValueError(
                f"prediction_type given as {self.config.prediction_type} must be one of `epsilon`, `sample`, `v_prediction` or `flow_prediction` for the UniPCMultistepScheduler."
            )
        if self.config.thresholding:
            sigma_t = self.sigmas[self.step_index]
            x0_pred = sample - sigma_t * model_output
            x0_pred = self._threshold_sample(x0_pred)
            epsilon = model_output + x0_pred
        return epsilon

    def multistep_uni_p_bh_update(self, model_output: torch.Tensor, *args, sample: torch.Tensor = None, order: int = None, **kwargs) -> torch.Tensor:
        prev_timestep = args[0] if len(args) > 0 else kwargs.pop("prev_timestep", None)
        if sample is None:
            if len(args) > 1:
                sample = args[1]
            else:
                raise ValueError("missing `sample` as a required keyward argument")
        if order is None:
            if len(args) > 2:
                order = args[2]
            else:
                raise ValueError("missing `order` as a required keyward argument")
        if prev_timestep is not None:
            deprecate(
                "prev_timestep",
                "1.0.0",
                "Passing `prev_timestep` is deprecated and has no effect as model output conversion is now handled via an internal counter `self.step_index`",
            )
        model_output_list = self.model_outputs
        s0 = self.timestep_list[-1]
        m0 = model_output_list[-1]
        x = sample

        if self.solver_p:
            x_t = self.solver_p.step(model_output, s0, x).prev_sample
            return x_t

        sigma_t, sigma_s0 = self.sigmas[self.step_index + 1], self.sigmas[self.step_index]
        alpha_t, sigma_t = self._sigma_to_alpha_sigma_t(sigma_t)
        alpha_s0, sigma_s0 = self._sigma_to_alpha_sigma_t(sigma_s0)
        lambda_t = torch.log(alpha_t) - torch.log(sigma_t)
        lambda_s0 = torch.log(alpha_s0) - torch.log(sigma_s0)
        h = lambda_t - lambda_s0
        device = sample.device

        rks = []
        d1s = []
        for idx in range(1, order):
            si = self.step_index - idx
            mi = model_output_list[-(idx + 1)]
            alpha_si, sigma_si = self._sigma_to_alpha_sigma_t(self.sigmas[si])
            lambda_si = torch.log(alpha_si) - torch.log(sigma_si)
            rk = (lambda_si - lambda_s0) / h
            rks.append(rk)
            d1s.append((mi - m0) / rk)

        rks.append(1.0)
        rks = torch.tensor(rks, device=device)
        matrix_r = []
        vector_b = []
        hh = -h if self.predict_x0 else h
        h_phi_1 = torch.expm1(hh)
        h_phi_k = h_phi_1 / hh - 1
        factorial_i = 1

        if self.config.solver_type == "bh1":
            b_h = hh
        elif self.config.solver_type == "bh2":
            b_h = torch.expm1(hh)
        else:
            raise NotImplementedError()

        for idx in range(1, order + 1):
            matrix_r.append(torch.pow(rks, idx - 1))
            vector_b.append(h_phi_k * factorial_i / b_h)
            factorial_i *= idx + 1
            h_phi_k = h_phi_k / hh - 1 / factorial_i

        matrix_r = torch.stack(matrix_r)
        vector_b = torch.tensor(vector_b, device=device)

        if len(d1s) > 0:
            d1s = torch.stack(d1s, dim=1)
            if order == 2:
                rhos_p = torch.tensor([0.5], dtype=x.dtype, device=device)
            else:
                rhos_p = torch.linalg.solve(matrix_r[:-1, :-1], vector_b[:-1]).to(device).to(x.dtype)
        else:
            d1s = None

        if self.predict_x0:
            x_t_ = sigma_t / sigma_s0 * x - alpha_t * h_phi_1 * m0
            pred_res = torch.einsum("k,bkc...->bc...", rhos_p, d1s) if d1s is not None else 0
            x_t = x_t_ - alpha_t * b_h * pred_res
        else:
            x_t_ = alpha_t / alpha_s0 * x - sigma_t * h_phi_1 * m0
            pred_res = torch.einsum("k,bkc...->bc...", rhos_p, d1s) if d1s is not None else 0
            x_t = x_t_ - sigma_t * b_h * pred_res
        return x_t.to(x.dtype)

    def multistep_uni_c_bh_update(
        self,
        this_model_output: torch.Tensor,
        *args,
        last_sample: torch.Tensor = None,
        this_sample: torch.Tensor = None,
        order: int = None,
        **kwargs,
    ) -> torch.Tensor:
        this_timestep = args[0] if len(args) > 0 else kwargs.pop("this_timestep", None)
        if last_sample is None:
            if len(args) > 1:
                last_sample = args[1]
            else:
                raise ValueError("missing `last_sample` as a required keyward argument")
        if this_sample is None:
            if len(args) > 2:
                this_sample = args[2]
            else:
                raise ValueError("missing `this_sample` as a required keyward argument")
        if order is None:
            if len(args) > 3:
                order = args[3]
            else:
                raise ValueError("missing `order` as a required keyward argument")
        if this_timestep is not None:
            deprecate(
                "this_timestep",
                "1.0.0",
                "Passing `this_timestep` is deprecated and has no effect as model output conversion is now handled via an internal counter `self.step_index`",
            )

        model_output_list = self.model_outputs
        m0 = model_output_list[-1]
        x = last_sample
        x_t = this_sample
        model_t = this_model_output

        sigma_t, sigma_s0 = self.sigmas[self.step_index], self.sigmas[self.step_index - 1]
        alpha_t, sigma_t = self._sigma_to_alpha_sigma_t(sigma_t)
        alpha_s0, sigma_s0 = self._sigma_to_alpha_sigma_t(sigma_s0)
        lambda_t = torch.log(alpha_t) - torch.log(sigma_t)
        lambda_s0 = torch.log(alpha_s0) - torch.log(sigma_s0)
        h = lambda_t - lambda_s0
        device = this_sample.device

        rks = []
        d1s = []
        for idx in range(1, order):
            si = self.step_index - (idx + 1)
            mi = model_output_list[-(idx + 1)]
            alpha_si, sigma_si = self._sigma_to_alpha_sigma_t(self.sigmas[si])
            lambda_si = torch.log(alpha_si) - torch.log(sigma_si)
            rk = (lambda_si - lambda_s0) / h
            rks.append(rk)
            d1s.append((mi - m0) / rk)

        rks.append(1.0)
        rks = torch.tensor(rks, device=device)
        matrix_r = []
        vector_b = []
        hh = -h if self.predict_x0 else h
        h_phi_1 = torch.expm1(hh)
        h_phi_k = h_phi_1 / hh - 1
        factorial_i = 1

        if self.config.solver_type == "bh1":
            b_h = hh
        elif self.config.solver_type == "bh2":
            b_h = torch.expm1(hh)
        else:
            raise NotImplementedError()

        for idx in range(1, order + 1):
            matrix_r.append(torch.pow(rks, idx - 1))
            vector_b.append(h_phi_k * factorial_i / b_h)
            factorial_i *= idx + 1
            h_phi_k = h_phi_k / hh - 1 / factorial_i

        matrix_r = torch.stack(matrix_r)
        vector_b = torch.tensor(vector_b, device=device)
        d1s = torch.stack(d1s, dim=1) if len(d1s) > 0 else None

        if order == 1:
            rhos_c = torch.tensor([0.5], dtype=x.dtype, device=device)
        else:
            rhos_c = torch.linalg.solve(matrix_r, vector_b).to(device).to(x.dtype)

        if self.predict_x0:
            x_t_ = sigma_t / sigma_s0 * x - alpha_t * h_phi_1 * m0
            corr_res = torch.einsum("k,bkc...->bc...", rhos_c[:-1], d1s) if d1s is not None else 0
            d1_t = model_t - m0
            x_t = x_t_ - alpha_t * b_h * (corr_res + rhos_c[-1] * d1_t)
        else:
            x_t_ = alpha_t / alpha_s0 * x - sigma_t * h_phi_1 * m0
            corr_res = torch.einsum("k,bkc...->bc...", rhos_c[:-1], d1s) if d1s is not None else 0
            d1_t = model_t - m0
            x_t = x_t_ - sigma_t * b_h * (corr_res + rhos_c[-1] * d1_t)
        return x_t.to(x.dtype)

    def index_for_timestep(self, timestep, schedule_timesteps=None):
        if schedule_timesteps is None:
            schedule_timesteps = self.timesteps
        indices = (schedule_timesteps == timestep).nonzero()
        pos = 1 if len(indices) > 1 else 0
        return indices[pos].item()

    def _init_step_index(self, timestep):
        if self.begin_index is None:
            if isinstance(timestep, torch.Tensor):
                timestep = timestep.to(self.timesteps.device)
            self._step_index = self.index_for_timestep(timestep)
        else:
            self._step_index = self._begin_index

    def step(
        self,
        model_output: torch.Tensor,
        timestep: Union[int, torch.Tensor],
        sample: torch.Tensor,
        return_dict: bool = True,
        generator=None,
    ) -> Union[SchedulerOutput, Tuple]:
        if self.num_inference_steps is None:
            raise ValueError("Number of inference steps is 'None', you need to run 'set_timesteps' after creating the scheduler")

        if self.step_index is None:
            self._init_step_index(timestep)

        use_corrector = self.step_index > 0 and self.step_index - 1 not in self.disable_corrector and self.last_sample is not None
        model_output_convert = self.convert_model_output(model_output, sample=sample)
        if use_corrector:
            sample = self.multistep_uni_c_bh_update(
                this_model_output=model_output_convert,
                last_sample=self.last_sample,
                this_sample=sample,
                order=self.this_order,
            )

        for idx in range(self.config.solver_order - 1):
            self.model_outputs[idx] = self.model_outputs[idx + 1]
            self.timestep_list[idx] = self.timestep_list[idx + 1]

        self.model_outputs[-1] = model_output_convert
        self.timestep_list[-1] = timestep

        if self.config.lower_order_final:
            this_order = min(self.config.solver_order, len(self.timesteps) - self.step_index)
        else:
            this_order = self.config.solver_order

        self.this_order = min(this_order, self.lower_order_nums + 1)
        assert self.this_order > 0

        self.last_sample = sample
        prev_sample = self.multistep_uni_p_bh_update(
            model_output=model_output,
            sample=sample,
            order=self.this_order,
        )

        if self.lower_order_nums < self.config.solver_order:
            self.lower_order_nums += 1
        self._step_index += 1

        if not return_dict:
            return (prev_sample,)
        return SchedulerOutput(prev_sample=prev_sample)

    def scale_model_input(self, sample: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        return sample

    def add_noise(self, original_samples: torch.Tensor, noise: torch.Tensor, timesteps: torch.IntTensor) -> torch.Tensor:
        sigmas = self.sigmas.to(device=original_samples.device, dtype=original_samples.dtype)
        if original_samples.device.type == "mps" and torch.is_floating_point(timesteps):
            schedule_timesteps = self.timesteps.to(original_samples.device, dtype=torch.float32)
            timesteps = timesteps.to(original_samples.device, dtype=torch.float32)
        else:
            schedule_timesteps = self.timesteps.to(original_samples.device)
            timesteps = timesteps.to(original_samples.device)

        if self.begin_index is None:
            step_indices = [self.index_for_timestep(timestep, schedule_timesteps) for timestep in timesteps]
        elif self.step_index is not None:
            step_indices = [self.step_index] * timesteps.shape[0]
        else:
            step_indices = [self.begin_index] * timesteps.shape[0]

        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < len(original_samples.shape):
            sigma = sigma.unsqueeze(-1)

        alpha_t, sigma_t = self._sigma_to_alpha_sigma_t(sigma)
        return alpha_t * original_samples + sigma_t * noise

    def __len__(self):
        return self.config.num_train_timesteps


class MatrixGame3OfficialSchedulerAdapter(BaseScheduler):
    """Adapt the official MG3 FlowUniPC scheduler to LightX2V's scheduler interface.

    The distilled path is fairly tolerant of LightX2V's generic Wan scheduler, but the
    base model is much more sensitive to scheduler semantics under 50-step CFG. This
    adapter keeps the rest of the LightX2V lifecycle untouched while delegating the
    actual UniPC stepping logic to the official Matrix-Game-3 implementation.
    """

    def __init__(self, config, scheduler_cls):
        super().__init__(config)
        self.scheduler_cls = scheduler_cls
        self.sample_shift = self.config["sample_shift"]
        self.sample_guide_scale = self.config["sample_guide_scale"]
        self.noise_pred = None
        self.noise_pred_cond = None
        self.noise_pred_uncond = None
        self.noise_pred_guided = None
        self.forward_kwargs_cond = None
        self.forward_kwargs_uncond = None
        self.mask = None
        self.vae_encoder_out = None
        self.timestep_input = None
        self._solver = None
        self._generator = None
        self.debug_stop_requested = False
        self._debug_step_probe_enabled = bool(self.config.get("debug_stop_after_scheduler_probe", False))
        self._debug_timesteps_logged = False
        self._debug_probe_step = int(self.config.get("debug_scheduler_probe_step", max(self.infer_steps - 1, 0)))

    def _reset_solver(self):
        self._solver = self.scheduler_cls()
        self._solver.set_timesteps(self.infer_steps, device=AI_DEVICE, shift=self.sample_shift)
        if self._debug_step_probe_enabled and not self._debug_timesteps_logged:
            _matrix_game3_log_debug_payload(
                "scheduler_timesteps_head",
                {
                    "timesteps_head": self._solver.timesteps[:3].detach().cpu().tolist(),
                },
            )
            self._debug_timesteps_logged = True

    def prepare(self, seed, latent_shape, image_encoder_output=None):
        self.debug_stop_requested = False
        self._debug_timesteps_logged = False
        self._generator = torch.Generator(device=AI_DEVICE).manual_seed(seed)
        self.latents = torch.randn(tuple(latent_shape), dtype=GET_DTYPE(), device=AI_DEVICE, generator=self._generator)
        self.vae_encoder_out = image_encoder_output.get("vae_encoder_out") if image_encoder_output is not None else None
        if self.vae_encoder_out is not None:
            self.vae_encoder_out = self.vae_encoder_out.to(device=AI_DEVICE, dtype=GET_DTYPE())
        self.noise_pred = None
        self.noise_pred_cond = None
        self.noise_pred_uncond = None
        self.noise_pred_guided = None
        self.forward_kwargs_cond = None
        self.forward_kwargs_uncond = None
        self.mask = torch.ones_like(self.latents)
        self._reset_solver()

    def reset(self, seed, latent_shape, step_index=None):
        self.debug_stop_requested = False
        self._debug_timesteps_logged = False
        self._generator = torch.Generator(device=AI_DEVICE).manual_seed(seed)
        self.latents = torch.randn(tuple(latent_shape), dtype=GET_DTYPE(), device=AI_DEVICE, generator=self._generator)
        if self.vae_encoder_out is not None:
            self.vae_encoder_out = self.vae_encoder_out.to(device=AI_DEVICE, dtype=GET_DTYPE())
        self.noise_pred = None
        self.noise_pred_cond = None
        self.noise_pred_uncond = None
        self.noise_pred_guided = None
        self.forward_kwargs_cond = None
        self.forward_kwargs_uncond = None
        if self.mask is not None:
            self.mask = self.mask.to(device=AI_DEVICE, dtype=GET_DTYPE())
        self._reset_solver()
        if step_index is not None:
            self.step_index = step_index

    def step_pre(self, step_index):
        super().step_pre(step_index)
        self.noise_pred = None
        self.noise_pred_cond = None
        self.noise_pred_uncond = None
        self.noise_pred_guided = None
        self.forward_kwargs_cond = None
        self.forward_kwargs_uncond = None
        self.timestep_input = torch.stack([self._solver.timesteps[self.step_index].to(device=AI_DEVICE)])

    def step_post(self):
        timestep = self._solver.timesteps[self.step_index].to(device=self.latents.device)
        prev_sample = self._solver.step(
            # Keep the model output in its original precision. The official MG3
            # pipeline feeds float32 noise predictions into UniPC even when the
            # latent state is bf16, and downcasting here introduces a tiny
            # per-step drift that the 50-step base model visibly amplifies.
            self.noise_pred,
            timestep,
            self.latents,
            return_dict=False,
        )[0]
        if self.mask is not None and self.vae_encoder_out is not None:
            prev_sample = (1.0 - self.mask) * self.vae_encoder_out + self.mask * prev_sample
        self.latents = prev_sample.to(dtype=GET_DTYPE())
        if self._debug_step_probe_enabled and self.step_index == self._debug_probe_step:
            _matrix_game3_log_debug_payload(f"step_{self.step_index}_output_latent", _matrix_game3_tensor_probe(self.latents, mask=self.mask))
            self.debug_stop_requested = True

    def clear(self):
        self._solver = None
        self.debug_stop_requested = False
        self._debug_timesteps_logged = False
        self.noise_pred = None
        self.noise_pred_cond = None
        self.noise_pred_uncond = None
        self.noise_pred_guided = None
        self.forward_kwargs_cond = None
        self.forward_kwargs_uncond = None


@RUNNER_REGISTER("wan2.2_matrix_game3")
class WanMatrixGame3Runner(Wan22DenseRunner):
    """Runner-only Matrix-Game-3 adapter on top of the existing Wan2.2 lifecycle.

    Official provenance:
    - CLI / mode semantics: Matrix-Game-3/generate.py
    - Segment lengths / condition assembly: pipeline/inference_pipeline.py
    - Interactive action refresh: pipeline/inference_interactive_pipeline.py
    - Keyboard / mouse dimensions: utils/conditions.py
    - Pose / plucker helpers: utils/cam_utils.py and utils/utils.py
    - Structural config truth: Matrix-Game-3.0/*/config.json

    Execution model:
    - Reuse Wan2.2 text encoder / scheduler / VAE lifecycle from `Wan22DenseRunner`.
    - Replace the normal i2v input path with a first-frame-only conditioning scheme.
    - Convert keyboard, mouse, and camera trajectories into per-segment DiT conditions.
    - Roll latent history across overlapping segments, then trim duplicated decoded frames.
    """

    def __init__(self, config):
        with config.temporarily_unlocked():
            # The public pipeline still instantiates us as "wan2.2_matrix_game3", but
            # the shared Wan2.2 runner expects `model_cls == "wan2.2"` for common setup.
            original_model_cls = str(config.get("model_cls", "wan2.2_matrix_game3"))
            config["runner_model_cls"] = original_model_cls
            config["model_cls"] = "wan2.2"
            config["mode"] = "matrix_game3"
            config["use_image_encoder"] = False
            config["use_base_model"] = bool(config.get("use_base_model", False))
            config["vae_type"] = str(config.get("vae_type", "mg_lightvae_v2"))
            if "lightvae_pruning_rate" not in config:
                if config["vae_type"] == "mg_lightvae":
                    config["lightvae_pruning_rate"] = 0.5
                elif config["vae_type"] == "mg_lightvae_v2":
                    config["lightvae_pruning_rate"] = 0.75
            if "sub_model_folder" not in config:
                config["sub_model_folder"] = "base_model" if config["use_base_model"] else "base_distilled_model"
            config["num_channels_latents"] = int(config.get("num_channels_latents", 48))
            config["vae_stride"] = tuple(config.get("vae_stride", (4, 16, 16)))
            config["patch_size"] = tuple(config.get("patch_size", (1, 2, 2)))
            # Load the official MG3 sub-model config before the parent runner
            # constructs the scheduler. The shared Wan scheduler expects fields
            # like `dim` and `num_heads` to already exist in `self.config`.
            self.config = config
            self.matrix_game3_model_cls = original_model_cls
            self._load_matrix_game3_model_config()
        super().__init__(config)

        self.matrix_game3_model_cls = original_model_cls
        # Official MG3 timeline convention:
        # - first segment predicts 57 frames from the input image;
        # - later segments operate on a 56-frame window;
        # - every new segment contributes 40 new frames and reuses 16 historical frames.
        action_config = self.config.get("action_config", {})
        self.first_clip_frame = int(self.config.get("first_clip_frame", 57))
        self.clip_frame = int(self.config.get("clip_frame", 56))
        self.incremental_segment_frames = int(self.config.get("incremental_segment_frames", 40))
        self.past_frame = int(self.config.get("past_frame", 16))
        self.conditioning_latent_frames = int(self.config.get("conditioning_latent_frames", 4))
        self.mouse_dim_in = int(self.config.get("mouse_dim_in", action_config.get("mouse_dim_in", 2)))
        self.keyboard_dim_in = int(self.config.get("keyboard_dim_in", action_config.get("keyboard_dim_in", 6)))

        # Session-scoped caches filled by `_prepare_matrix_game3_session()` and then
        # consumed incrementally as each segment is initialized and decoded.
        self._segment_states: dict[int, MatrixGame3SegmentState] = {}
        self._official_modules: Optional[dict[str, Any]] = None
        self._mg3_lat_h: Optional[int] = None
        self._mg3_lat_w: Optional[int] = None
        self._mg3_target_h: Optional[int] = None
        self._mg3_target_w: Optional[int] = None
        self._mg3_base_intrinsics: Optional[torch.Tensor] = None
        self._mg3_intrinsics_all: Optional[torch.Tensor] = None
        self._mg3_keyboard_all: Optional[torch.Tensor] = None
        self._mg3_mouse_all: Optional[torch.Tensor] = None
        self._mg3_extrinsics_all: Optional[torch.Tensor] = None
        self._mg3_num_iterations: int = 1
        self._mg3_expected_total_frames: int = self.first_clip_frame
        self._mg3_interactive = bool(self.config.get("interactive", False))
        self._mg3_last_pose = np.zeros(5, dtype=np.float32)
        self._mg3_current_segment_state: Optional[MatrixGame3SegmentState] = None
        self._mg3_current_segment_full_latents: Optional[torch.Tensor] = None
        self._mg3_generated_latent_history: list[torch.Tensor] = []
        self._mg3_tail_latents: Optional[torch.Tensor] = None
        self._mg3_noise_generator: Optional[torch.Generator] = None

    def set_inputs(self, inputs):
        super().set_inputs(inputs)
        # Some callers still use `pose`, others use `action_path`. Mirror both so the
        # runner remains compatible with older LightX2V entry points.
        if "action_path" in self.input_info.__dataclass_fields__:
            self.input_info.action_path = inputs.get("action_path", inputs.get("pose", ""))
        if "pose" in self.input_info.__dataclass_fields__:
            self.input_info.pose = inputs.get("pose", inputs.get("action_path", ""))

    def run_text_encoder(self, input_info):
        # Official Matrix-Game-3 base inference uses a non-empty default negative
        # prompt for CFG. If the caller leaves `--negative_prompt` empty, reuse the
        # official default so the unconditional branch matches the reference path.
        if self.config.get("enable_cfg", False) and not getattr(input_info, "negative_prompt", ""):
            input_info.negative_prompt = self.config.get("sample_neg_prompt", _MATRIX_GAME3_DEFAULT_NEGATIVE_PROMPT)
            logger.info("[matrix-game-3] negative_prompt not provided; falling back to the official sample_neg_prompt for CFG.")
        return super().run_text_encoder(input_info)

    def load_transformer(self):
        from lightx2v.models.networks.wan.matrix_game3_model import (
            WanMtxg3Model,
            WanMtxg3OfficialBaseModel,
        )

        # The backbone is still a Wan2.2 DiT, but Matrix-Game-3 swaps in a dedicated
        # network wrapper that understands keyboard / mouse / camera conditions.
        model_kwargs = {
            "model_path": self.config["model_path"],
            "config": self.config,
            "device": self.init_device,
        }
        lora_configs = self.config.get("lora_configs")
        if not lora_configs:
            if self.config.get("use_base_model", False):
                try:
                    logger.info("[matrix-game-3] base-model path will use the official WanModel forward for denoising.")
                    return WanMtxg3OfficialBaseModel(**model_kwargs)
                except Exception as exc:
                    logger.warning(
                        "[matrix-game-3] failed to initialize official base-model forward ({}); falling back to the custom LightX2V MG3 model.",
                        exc,
                    )
            return WanMtxg3Model(**model_kwargs)
        return build_wan_model_with_lora(WanMtxg3Model, self.config, model_kwargs, lora_configs, model_type="wan2.2")

    def init_scheduler(self):
        # MG3 relies on a fixed latent prefix that must be re-injected after every
        # diffusion step. The inlined FlowUniPC adapter already mirrors the official
        # per-step semantics while keeping the denoiser itself on the native
        # LightX2V path, so prefer it for both base and distilled checkpoints.
        try:
            self.scheduler = MatrixGame3OfficialSchedulerAdapter(self.config, FlowUniPCMultistepScheduler)
            logger.info("[matrix-game-3] using inlined FlowUniPCMultistepScheduler for MG3 sampling.")
            return
        except Exception as exc:
            logger.warning(
                "[matrix-game-3] failed to initialize inlined MG3 scheduler ({}); falling back to LightX2V WanScheduler.",
                exc,
            )
        super().init_scheduler()

    def _get_sub_model_folder(self) -> str:
        """Resolve which MG3 sub-model folder should be used for config lookup."""
        return str(self.config.get("sub_model_folder", "base_model" if self.config.get("use_base_model", False) else "base_distilled_model"))

    def resolve_model_config_path(self) -> Path:
        """Resolve the MG3 base/distilled config.json with explicit override support."""
        configured_path = self.config.get("matrix_game3_config_path")
        if configured_path:
            for candidate in _expand_path_candidates(configured_path):
                if candidate.is_file():
                    return candidate
            raise FileNotFoundError(
                "Matrix-Game-3 config.json is missing for "
                f"matrix_game3_config_path={configured_path!r}. "
                "The runner needs this file to align latent channels, patch size, and action_config with the checkpoint. "
                "Please set config['matrix_game3_config_path'] to a valid config.json path."
            )

        sub_model_folder = self._get_sub_model_folder()
        candidates: list[Path] = []
        model_path = self.config.get("model_path")
        if model_path:
            for candidate_root in _expand_path_candidates(model_path):
                candidate = candidate_root / sub_model_folder / "config.json"
                if candidate not in candidates:
                    candidates.append(candidate)
        candidates.append(_PROJECT_ROOT / _MATRIX_GAME3_CONFIG_ROOT_RELATIVE / sub_model_folder / "config.json")

        for candidate in candidates:
            if candidate.is_file():
                return candidate

        checked_locations = ", ".join(str(candidate) for candidate in candidates)
        raise FileNotFoundError(
            "Matrix-Game-3 sub-model config.json could not be resolved. "
            f"Checked: {checked_locations}. "
            "The runner needs this file to determine the official base/distilled structure. "
            "Please set config['matrix_game3_config_path'], or provide a valid config['model_path'] and "
            "config['sub_model_folder'] (defaulted from config['use_base_model'])."
        )

    def _load_matrix_game3_model_config(self):
        """Merge the official MG3 config so latent/channel sizes match the checkpoint."""
        config_path = self.resolve_model_config_path()
        with config_path.open("r") as f:
            model_config = json.load(f)

        with self.config.temporarily_unlocked():
            self.config.update(model_config)
            self.config["num_channels_latents"] = int(model_config.get("in_dim", self.config.get("num_channels_latents", 48)))
            self.config["vae_stride"] = tuple(self.config.get("vae_stride", (4, 16, 16)))
            self.config["patch_size"] = tuple(model_config.get("patch_size", self.config.get("patch_size", (1, 2, 2))))
            self.config["sample_neg_prompt"] = self.config.get("sample_neg_prompt", _MATRIX_GAME3_DEFAULT_NEGATIVE_PROMPT)

        action_config = self.config.get("action_config", {})
        self.keyboard_dim_in = int(self.config.get("keyboard_dim_in", action_config.get("keyboard_dim_in", 6)))
        self.mouse_dim_in = int(self.config.get("mouse_dim_in", action_config.get("mouse_dim_in", 2)))

    def _get_official_modules(self) -> dict[str, Any]:
        """Expose inlined Matrix-Game-3 helpers through a module-like interface."""
        if self._official_modules is not None:
            return self._official_modules

        modules = {
            "conditions": _MatrixGame3ConditionsShim,
            "cam_utils": _MatrixGame3CamUtilsShim,
            "utils": _MatrixGame3UtilsShim,
        }
        self._official_modules = modules
        return modules

    def _get_expected_total_frames(self, raw_total_frames: Optional[int] = None) -> tuple[int, int]:
        """Resolve how many segments to run.

        Matrix-Game-3 only supports lengths of `57 + 40 * k`. If a control sequence
        does not align exactly, the tail is ignored so the segment schedule stays valid.
        """
        num_iterations = self.config.get("num_iterations", None)
        if num_iterations is not None:
            num_iterations = max(int(num_iterations), 1)
            return num_iterations, self.first_clip_frame + (num_iterations - 1) * self.incremental_segment_frames

        if raw_total_frames is None:
            return 1, self.first_clip_frame

        if raw_total_frames <= self.first_clip_frame:
            return 1, self.first_clip_frame

        additional_frames = raw_total_frames - self.first_clip_frame
        num_iterations = 1 + max(additional_frames // self.incremental_segment_frames, 0)
        expected_total_frames = self.first_clip_frame + (num_iterations - 1) * self.incremental_segment_frames
        if additional_frames % self.incremental_segment_frames != 0:
            logger.warning(
                "[matrix-game-3] raw control sequence has {} frames; truncating tail to {} frames so it matches 57 + 40*k.",
                raw_total_frames,
                expected_total_frames,
            )
        return num_iterations, expected_total_frames

    def _segment_latent_shape(self, lat_h: int, lat_w: int, frame_count: int) -> list[int]:
        """Compute `[C, T, H, W]` latent shape for one segment window."""
        return [
            self.config.get("num_channels_latents", 48),
            (frame_count - 1) // self.config["vae_stride"][0] + 1,
            lat_h,
            lat_w,
        ]

    @ProfilingContext4DebugL1(
        "Run VAE Encoder",
        recorder_mode=GET_RECORDER_MODE(),
        metrics_func=monitor_cli.lightx2v_run_vae_encoder_image_duration,
        metrics_labels=["WanMatrixGame3Runner"],
    )
    def run_vae_encoder(self, img):
        # Unlike the generic Wan2.2 i2v path, MG3 only encodes the first frame. The
        # remaining temporal slots are left zeroed and later mixed with scheduler noise.
        target_h = int(self.config["target_height"])
        target_w = int(self.config["target_width"])
        target_ratio = target_h / target_w
        input_h, input_w = img.height, img.width
        if input_h / input_w > target_ratio:
            crop_h = int(input_w * target_ratio)
            crop_w = input_w
        else:
            crop_h = input_h
            crop_w = int(input_h / target_ratio)

        crop_x = int(round((input_w - crop_w) / 2.0))
        crop_y = int(round((input_h - crop_h) / 2.0))
        image_uint8 = torch.from_numpy(np.array(img)).unsqueeze(0).permute(0, 3, 1, 2)
        image_uint8 = image_uint8[:, :, crop_y : crop_y + crop_h, crop_x : crop_x + crop_w]
        image_tensor = image_uint8.float().div_(255.0)
        image_tensor = torch.nn.functional.interpolate(
            image_tensor,
            size=(target_h, target_w),
            mode="bicubic",
            align_corners=True,
            antialias=True,
        )
        image_tensor = image_tensor.mul_(2.0).sub_(1.0).transpose(0, 1).to(AI_DEVICE)
        first_frame_latent = self.get_vae_encoder_output(image_tensor)
        lat_h = target_h // self.config["vae_stride"][1]
        lat_w = target_w // self.config["vae_stride"][2]
        latent_shape = self._segment_latent_shape(lat_h, lat_w, self.first_clip_frame)
        vae_encoder_out = torch.zeros(latent_shape, device=first_frame_latent.device, dtype=first_frame_latent.dtype)
        vae_encoder_out[:, : first_frame_latent.shape[1]] = first_frame_latent
        return vae_encoder_out, latent_shape

    @ProfilingContext4DebugL2("Run Encoders")
    def _run_input_encoder_local_i2v(self):
        # MG3 does not use the CLIP image encoder branch. The conditioning payload is:
        # - text encoder output from the normal Wan pipeline;
        # - a first-frame VAE latent;
        # - segment metadata prepared for later `init_run_segment()` calls.
        _, img_ori = self.read_image_input(self.input_info.image_path)
        vae_encoder_out, latent_shape = self.run_vae_encoder(img_ori)
        self.input_info.latent_shape = latent_shape
        text_encoder_output = self.run_text_encoder(self.input_info)
        self._prepare_matrix_game3_session(img_ori, latent_shape, vae_encoder_out)
        torch_device_module.empty_cache()
        return self.get_encoder_output_i2v(None, vae_encoder_out, text_encoder_output)

    def get_encoder_output_i2v(self, clip_encoder_out, vae_encoder_out, text_encoder_output, img=None):
        # Keep the standard LightX2V output contract so downstream scheduler / model
        # code can stay unchanged. Segment-specific conditions are injected later.
        image_encoder_output = {
            "clip_encoder_out": clip_encoder_out,
            "vae_encoder_out": vae_encoder_out,
            "dit_cond_dict": {},
        }
        return {
            "text_encoder_output": text_encoder_output,
            "image_encoder_output": image_encoder_output,
        }

    def _prepare_matrix_game3_session(self, pil_image: Image.Image, latent_shape: list[int], vae_encoder_out: torch.Tensor):
        # Official source:
        # - Non-interactive path mirrors pipeline/inference_pipeline.py
        # - Interactive segment refreshing mirrors pipeline/inference_interactive_pipeline.py
        # - Camera/action fallback semantics follow the user's requested runner contract
        #
        # This method performs all once-per-request setup:
        # - resolve spatial sizes used by camera/plucker helpers;
        # - reset cached segment state and latent history;
        # - pre-load the entire control sequence for offline mode; or
        # - defer control acquisition to segment time for interactive mode.
        self._get_official_modules()
        self._segment_states.clear()
        self._mg3_generated_latent_history = []
        self._mg3_tail_latents = None
        self._mg3_current_segment_state = None
        self._mg3_current_segment_full_latents = None
        self._mg3_interactive = bool(self.config.get("interactive", False))
        self._mg3_last_pose = np.zeros(5, dtype=np.float32)
        self._mg3_lat_h = int(latent_shape[-2])
        self._mg3_lat_w = int(latent_shape[-1])
        self._mg3_target_h = self._mg3_lat_h * self.config["vae_stride"][1]
        self._mg3_target_w = self._mg3_lat_w * self.config["vae_stride"][2]
        self._mg3_base_intrinsics = self._default_intrinsics().to(dtype=torch.float32)

        if self._mg3_interactive:
            num_iterations = self.config.get("num_iterations", 1)
            self._mg3_num_iterations = max(int(num_iterations), 1)
            self._mg3_expected_total_frames = self.first_clip_frame + (self._mg3_num_iterations - 1) * self.incremental_segment_frames
            self._mg3_keyboard_all = None
            self._mg3_mouse_all = None
            self._mg3_extrinsics_all = None
            self._mg3_intrinsics_all = None
            return

        action_path = self.input_info.action_path or self.input_info.pose or ""
        raw_controls = self._load_control_payload(action_path)
        raw_total_frames = self._infer_raw_total_frames(raw_controls)
        self._mg3_num_iterations, self._mg3_expected_total_frames = self._get_expected_total_frames(raw_total_frames)

        # Match the official Matrix-Game-3 demo pipeline: when the user does not
        # provide an external action file, fall back to the benchmark universal
        # action sequence instead of a fully static zero-control clip.
        if not raw_controls:
            modules = self._get_official_modules()
            logger.warning(
                "[matrix-game-3] action_path missing or empty; falling back to official Bench_actions_universal({}).",
                self._mg3_expected_total_frames,
            )
            raw_controls = self._normalize_payload_keys(modules["conditions"].Bench_actions_universal(self._mg3_expected_total_frames))

        self._mg3_keyboard_all, self._mg3_mouse_all, self._mg3_extrinsics_all, self._mg3_intrinsics_all = self._build_noninteractive_controls(raw_controls)

    def _infer_raw_total_frames(self, payload: dict[str, Any]) -> Optional[int]:
        """Infer sequence length from whichever control tensor is present."""
        lengths = []
        for value in payload.values():
            if value is None:
                continue
            if isinstance(value, np.ndarray):
                if value.ndim >= 1:
                    lengths.append(int(value.shape[0]))
            elif isinstance(value, torch.Tensor):
                if value.ndim >= 1:
                    lengths.append(int(value.shape[0]))
            elif isinstance(value, list):
                lengths.append(len(value))
        return max(lengths) if lengths else None

    def _load_control_payload(self, action_path: str) -> dict[str, Any]:
        """Load keyboard/mouse/pose/intrinsics controls from a file or a directory."""
        if not action_path:
            logger.warning("[matrix-game-3] action_path missing, fallback to zero keyboard/mouse and identity poses.")
            return {}

        path = Path(action_path)
        if not path.exists():
            logger.warning("[matrix-game-3] action_path not found: {}. Fallback to zero keyboard/mouse and identity poses.", action_path)
            return {}

        if path.is_dir():
            return self._load_control_payload_from_dir(path)
        return self._load_control_payload_from_file(path)

    def _load_control_payload_from_dir(self, path: Path) -> dict[str, Any]:
        """Best-effort directory loader that accepts several common file names."""
        payload: dict[str, Any] = {}
        name_groups = {
            "keyboard_cond": ["keyboard_cond.npy", "keyboard_condition.npy", "keyboard_cond.pt", "keyboard_condition.pt", "keyboard_cond.json", "keyboard_condition.json"],
            "mouse_cond": ["mouse_cond.npy", "mouse_condition.npy", "mouse_cond.pt", "mouse_condition.pt", "mouse_cond.json", "mouse_condition.json"],
            "poses": ["poses.npy", "pose.npy", "poses.pt", "pose.pt", "poses.json", "pose.json", "c2ws.npy", "c2w.npy"],
            "intrinsics": ["intrinsics.npy", "intrinsics.pt", "intrinsics.json", "Ks.npy", "K.npy"],
        }
        for key, names in name_groups.items():
            for file_name in names:
                candidate = path / file_name
                if not candidate.exists():
                    continue
                payload[key] = self._load_control_payload_from_file(candidate).get(key)
                break
        return payload

    def _load_control_payload_from_file(self, path: Path) -> dict[str, Any]:
        """Parse one control file and map it onto the normalized payload schema."""
        suffix = path.suffix.lower()
        stem = path.stem.lower()
        if suffix == ".npz":
            data = dict(np.load(path, allow_pickle=True))
            return self._normalize_payload_keys(data)
        if suffix == ".json":
            with path.open("r") as f:
                data = json.load(f)
            return self._normalize_payload_keys(data)
        if suffix == ".npy":
            array = np.load(path, allow_pickle=True)
        elif suffix in {".pt", ".pth"}:
            array = torch.load(path, map_location="cpu")
            if isinstance(array, dict):
                return self._normalize_payload_keys(array)
        else:
            raise ValueError(f"unsupported action_path format: {path}")

        if "keyboard" in stem:
            return {"keyboard_cond": array}
        if "mouse" in stem:
            return {"mouse_cond": array}
        if "intrinsic" in stem or stem in {"k", "ks"}:
            return {"intrinsics": array}
        if "pose" in stem or "c2w" in stem:
            return {"poses": array}
        raise ValueError(f"unsupported action_path file name: {path}")

    def _normalize_payload_keys(self, data: dict[str, Any]) -> dict[str, Any]:
        """Collapse different naming conventions into the runner's canonical keys."""
        payload: dict[str, Any] = {}
        key_aliases = {
            "keyboard_cond": {"keyboard_cond", "keyboard_condition"},
            "mouse_cond": {"mouse_cond", "mouse_condition"},
            "poses": {"poses", "pose", "c2ws", "c2w", "extrinsics"},
            "intrinsics": {"intrinsics", "k", "ks"},
        }
        for target_key, aliases in key_aliases.items():
            for key, value in data.items():
                if str(key).lower() in aliases:
                    payload[target_key] = value
                    break
        return payload

    def _default_intrinsics(self) -> torch.Tensor:
        """Generate the default camera intrinsics for the current output resolution."""
        modules = self._get_official_modules()
        assert self._mg3_target_h is not None and self._mg3_target_w is not None
        return modules["cam_utils"].get_intrinsics(self._mg3_target_h, self._mg3_target_w)

    def _to_tensor(self, value: Any, dtype=torch.float32) -> Optional[torch.Tensor]:
        """Convert numpy/list/scalar inputs into CPU tensors for normalization."""
        if value is None:
            return None
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().to(dtype=dtype)
        if isinstance(value, np.ndarray):
            return torch.from_numpy(value).to(dtype=dtype)
        if isinstance(value, list):
            return torch.tensor(value, dtype=dtype)
        return torch.tensor(value, dtype=dtype)

    def _resize_time_axis(self, tensor: torch.Tensor, total_frames: int) -> torch.Tensor:
        # MG3 expects exact per-frame control lengths. To make the runner tolerant of
        # slightly malformed inputs, short sequences are padded by repeating the last
        # value and long sequences are truncated.
        if tensor.shape[0] == total_frames:
            return tensor
        if tensor.shape[0] == 1:
            return tensor.repeat(total_frames, *([1] * (tensor.ndim - 1)))
        if tensor.shape[0] < total_frames:
            pad = tensor[-1:].repeat(total_frames - tensor.shape[0], *([1] * (tensor.ndim - 1)))
            logger.warning(
                "[matrix-game-3] control length {} shorter than expected {}, padding with the last value.",
                tensor.shape[0],
                total_frames,
            )
            return torch.cat([tensor, pad], dim=0)
        logger.warning(
            "[matrix-game-3] control length {} longer than expected {}, truncating the tail.",
            tensor.shape[0],
            total_frames,
        )
        return tensor[:total_frames]

    def _normalize_keyboard_cond(self, value: Any, total_frames: int) -> torch.Tensor:
        """Normalize keyboard controls into `[1, T, keyboard_dim_in]`."""
        if value is None:
            return torch.zeros((1, total_frames, self.keyboard_dim_in), dtype=torch.float32)
        tensor = self._to_tensor(value)
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)
        if tensor.ndim == 3 and tensor.shape[0] == 1:
            tensor = tensor.squeeze(0)
        if tensor.ndim != 2 or tensor.shape[-1] != self.keyboard_dim_in:
            raise ValueError(f"keyboard_cond shape mismatch, expected [T,{self.keyboard_dim_in}], got {tuple(tensor.shape)}")
        tensor = self._resize_time_axis(tensor, total_frames)
        return tensor.unsqueeze(0)

    def _normalize_mouse_cond(self, value: Any, total_frames: int) -> torch.Tensor:
        """Normalize mouse controls into `[1, T, mouse_dim_in]`."""
        if value is None:
            return torch.zeros((1, total_frames, self.mouse_dim_in), dtype=torch.float32)
        tensor = self._to_tensor(value)
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)
        if tensor.ndim == 3 and tensor.shape[0] == 1:
            tensor = tensor.squeeze(0)
        if tensor.ndim != 2 or tensor.shape[-1] != self.mouse_dim_in:
            raise ValueError(f"mouse_cond shape mismatch, expected [T,{self.mouse_dim_in}], got {tuple(tensor.shape)}")
        tensor = self._resize_time_axis(tensor, total_frames)
        return tensor.unsqueeze(0)

    def _normalize_intrinsics(self, value: Any, total_frames: int) -> Optional[torch.Tensor]:
        """Accept either flattened `[fx, fy, cx, cy]` or 3x3 intrinsics matrices."""
        if value is None:
            return None
        tensor = self._to_tensor(value)
        if tensor.ndim == 1:
            if tensor.shape[0] == 4:
                tensor = tensor.unsqueeze(0)
            elif tensor.shape[0] == 9:
                tensor = tensor.view(3, 3).unsqueeze(0)
        if tensor.ndim == 3 and tensor.shape[-2:] == (3, 3):
            tensor = torch.stack([tensor[..., 0, 0], tensor[..., 1, 1], tensor[..., 0, 2], tensor[..., 1, 2]], dim=-1)
        if tensor.ndim != 2 or tensor.shape[-1] != 4:
            raise ValueError(f"intrinsics shape mismatch, expected [T,4] or [T,3,3], got {tuple(tensor.shape)}")
        return self._resize_time_axis(tensor, total_frames)

    def _normalize_poses(self, value: Any, total_frames: int) -> Optional[torch.Tensor]:
        """Normalize poses into `[T, 4, 4]` camera-to-world extrinsics."""
        if value is None:
            return None
        tensor = self._to_tensor(value)
        if tensor.ndim == 2 and tensor.shape[-1] == 5:
            # The official action pipeline also uses a compact 5D pose
            # `[x, y, z, pitch, yaw]`. Convert it here to full extrinsics.
            modules = self._get_official_modules()
            rotations = np.concatenate([np.zeros((tensor.shape[0], 1), dtype=np.float32), tensor[:, 3:5].numpy()], axis=1).tolist()
            positions = tensor[:, :3].numpy().tolist()
            tensor = modules["cam_utils"].get_extrinsics(rotations, positions).to(dtype=torch.float32)
        if tensor.ndim == 3 and tensor.shape[-2:] == (4, 4):
            tensor = self._resize_time_axis(tensor, total_frames)
            return tensor
        raise ValueError(f"poses shape mismatch, expected [T,4,4] or [T,5], got {tuple(tensor.shape)}")

    def _build_noninteractive_controls(self, payload: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        # Official source:
        # - utils/conditions.py defines keyboard_dim_in=6 and mouse_dim_in=2 semantics
        # - utils/utils.py computes poses from actions when explicit poses are absent
        #
        # Offline mode materializes the whole control trajectory up front so later
        # segments only need cheap slicing instead of re-reading user inputs.
        total_frames = self._mg3_expected_total_frames
        keyboard_cond = self._normalize_keyboard_cond(payload.get("keyboard_cond"), total_frames)
        mouse_cond = self._normalize_mouse_cond(payload.get("mouse_cond"), total_frames)
        intrinsics_all = self._normalize_intrinsics(payload.get("intrinsics"), total_frames)

        poses = self._normalize_poses(payload.get("poses"), total_frames)
        if poses is None:
            modules = self._get_official_modules()
            if not payload:
                # No action file at all: keep the camera fixed at identity.
                identity_pose = torch.eye(4, dtype=torch.float32).unsqueeze(0).repeat(total_frames, 1, 1)
                poses = identity_pose
            else:
                # Action file exists but explicit poses do not: reconstruct camera motion
                # with the official action-to-pose integrator.
                first_pose = np.zeros(5, dtype=np.float32)
                all_poses = modules["utils"].compute_all_poses_from_actions(
                    keyboard_cond.squeeze(0).cpu(),
                    mouse_cond.squeeze(0).cpu(),
                    first_pose=first_pose,
                )
                positions = all_poses[:, :3].tolist()
                rotations = np.concatenate([np.zeros((all_poses.shape[0], 1), dtype=np.float32), all_poses[:, 3:5]], axis=1).tolist()
                poses = modules["cam_utils"].get_extrinsics(rotations, positions).to(dtype=torch.float32)
        return keyboard_cond, mouse_cond, poses, intrinsics_all

    def get_video_segment_num(self):
        self.video_segment_num = self._mg3_num_iterations

    def init_run(self):
        # This mostly mirrors `DefaultRunner.init_run()`, but we immediately override
        # the scheduler state with the first segment's custom latent/mask setup.
        self.gen_video_final = None
        self.get_video_segment_num()
        self._mg3_noise_generator = torch.Generator(device=AI_DEVICE).manual_seed(self.input_info.seed)
        self._mg3_generated_latent_history = []
        self._mg3_tail_latents = None
        self._mg3_current_segment_full_latents = None
        self._mg3_current_segment_state = None

        if self.config.get("lazy_load", False) or self.config.get("unload_modules", False):
            self.model = self.load_transformer()
            self.model.set_scheduler(self.scheduler)

        self.model.scheduler.prepare(seed=self.input_info.seed, latent_shape=self.input_info.latent_shape, image_encoder_output=self.inputs["image_encoder_output"])
        self._apply_segment_scheduler_state(self._build_or_get_segment_state(0))
        self.inputs["image_encoder_output"]["vae_encoder_out"] = None

    def _append_interactive_segment_controls(self, segment_idx: int):
        """Collect one segment worth of controls from stdin in interactive mode."""
        modules = self._get_official_modules()
        first_clip = segment_idx == 0
        action_frames = self.first_clip_frame if first_clip else self.incremental_segment_frames

        if not dist.is_initialized() or dist.get_rank() == 0:
            actions = self._prompt_current_action()
            # The prompt returns one action token; MG3 applies it uniformly across the
            # newly generated frame span for that segment.
            keyboard_curr = actions["keyboard"].repeat(action_frames, 1)
            mouse_curr = actions["mouse"].repeat(action_frames, 1)
            if first_clip:
                first_pose = np.zeros(5, dtype=np.float32)
            else:
                first_pose = self._mg3_last_pose
            all_poses, last_pose = modules["utils"].compute_all_poses_from_actions(
                keyboard_curr.cpu(),
                mouse_curr.cpu(),
                first_pose=first_pose,
                return_last_pose=True,
            )
            positions = all_poses[:, :3].tolist()
            rotations = np.concatenate([np.zeros((all_poses.shape[0], 1), dtype=np.float32), all_poses[:, 3:5]], axis=1).tolist()
            extrinsics_curr = modules["cam_utils"].get_extrinsics(rotations, positions).to(dtype=torch.float32)
            payload = [
                keyboard_curr.numpy(),
                mouse_curr.numpy(),
                extrinsics_curr.numpy(),
                last_pose.astype(np.float32),
            ]
        else:
            payload = [None, None, None, None]

        if dist.is_initialized():
            dist.broadcast_object_list(payload, src=0)

        keyboard_curr = torch.from_numpy(payload[0]).to(dtype=torch.float32).unsqueeze(0)
        mouse_curr = torch.from_numpy(payload[1]).to(dtype=torch.float32).unsqueeze(0)
        extrinsics_curr = torch.from_numpy(payload[2]).to(dtype=torch.float32)
        self._mg3_last_pose = np.array(payload[3], dtype=np.float32)

        if self._mg3_keyboard_all is None:
            self._mg3_keyboard_all = keyboard_curr
            self._mg3_mouse_all = mouse_curr
            self._mg3_extrinsics_all = extrinsics_curr
        else:
            # Interactive mode grows the global control timeline as segments progress.
            self._mg3_keyboard_all = torch.cat([self._mg3_keyboard_all, keyboard_curr], dim=1)
            self._mg3_mouse_all = torch.cat([self._mg3_mouse_all, mouse_curr], dim=1)
            self._mg3_extrinsics_all = torch.cat([self._mg3_extrinsics_all, extrinsics_curr], dim=0)

    def _prompt_current_action(self) -> dict[str, torch.Tensor]:
        """Minimal CLI UX for interactive MG3 generation."""
        cam_value = 0.1
        print()
        print("-" * 30)
        print("PRESS [I, K, J, L, U] FOR CAMERA TRANSFORM")
        print("(I: up, K: down, J: left, L: right, U: no move)")
        print("PRESS [W, S, A, D, Q] FOR MOVEMENT")
        print("(W: forward, S: back, A: left, D: right, Q: no move)")
        print("-" * 30)

        camera_value_map = {
            "i": [cam_value, 0.0],
            "k": [-cam_value, 0.0],
            "j": [0.0, -cam_value],
            "l": [0.0, cam_value],
            "u": [0.0, 0.0],
        }
        keyboard_idx = {
            "w": [1, 0, 0, 0, 0, 0],
            "s": [0, 1, 0, 0, 0, 0],
            "a": [0, 0, 1, 0, 0, 0],
            "d": [0, 0, 0, 1, 0, 0],
            "q": [0, 0, 0, 0, 0, 0],
        }
        while True:
            idx_mouse = input("Please input the mouse action (e.g. `U`):\n").strip().lower()
            idx_keyboard = input("Please input the keyboard action (e.g. `W`):\n").strip().lower()
            if idx_mouse in camera_value_map and idx_keyboard in keyboard_idx:
                return {
                    "mouse": torch.tensor(camera_value_map[idx_mouse], dtype=torch.float32),
                    "keyboard": torch.tensor(keyboard_idx[idx_keyboard], dtype=torch.float32),
                }

    def _interpolate_intrinsics(self, intrinsics_seq: Optional[torch.Tensor], src_indices: np.ndarray, tgt_indices: np.ndarray) -> torch.Tensor:
        """Interpolate intrinsics onto the latent timeline used by the DiT."""
        assert self._mg3_base_intrinsics is not None
        if intrinsics_seq is None:
            return self._mg3_base_intrinsics.to(dtype=torch.float32).repeat(len(tgt_indices), 1)

        intrinsics_seq = intrinsics_seq.to(dtype=torch.float32)
        if intrinsics_seq.shape[0] == 1:
            return intrinsics_seq.repeat(len(tgt_indices), 1)

        src_indices = np.asarray(src_indices, dtype=np.float32)
        tgt_indices = np.asarray(tgt_indices, dtype=np.float32)
        src_indices = np.clip(np.round(src_indices).astype(np.int64), 0, intrinsics_seq.shape[0] - 1)
        src_intrinsics = intrinsics_seq[src_indices]
        out = []
        for column_idx in range(src_intrinsics.shape[-1]):
            column = np.interp(tgt_indices, src_indices.astype(np.float32), src_intrinsics[:, column_idx].cpu().numpy())
            out.append(torch.from_numpy(column).to(dtype=torch.float32))
        return torch.stack(out, dim=-1)

    def _build_plucker_from_c2ws(
        self,
        c2ws_seq: torch.Tensor,
        src_indices: np.ndarray,
        tgt_indices: np.ndarray,
        framewise: bool,
        intrinsics_seq: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Official source:
        # - utils/cam_utils.py: interpolate poses, compute relative poses, build plucker rays
        # - utils/utils.py: build_plucker_from_c2ws reshaping convention
        #
        # The model consumes camera control as plucker ray embeddings aligned to latent
        # time and latent spatial resolution, not as raw pose matrices.
        modules = self._get_official_modules()
        assert self._mg3_target_h is not None and self._mg3_target_w is not None
        assert self._mg3_lat_h is not None and self._mg3_lat_w is not None
        c2ws_np = c2ws_seq.cpu().numpy()
        c2ws_infer = (
            modules["cam_utils"]
            ._interpolate_camera_poses_handedness(
                src_indices=src_indices,
                src_rot_mat=c2ws_np[:, :3, :3],
                src_trans_vec=c2ws_np[:, :3, 3],
                tgt_indices=tgt_indices,
            )
            .to(device=c2ws_seq.device)
        )
        # `framewise=True` means each timestep is represented relative to its own local
        # frame history, which matches the official per-segment conditioning path.
        c2ws_infer = modules["cam_utils"].compute_relative_poses(c2ws_infer, framewise=framewise)
        Ks = self._interpolate_intrinsics(intrinsics_seq, src_indices, tgt_indices).to(device=c2ws_infer.device, dtype=c2ws_infer.dtype)
        plucker = modules["cam_utils"].get_plucker_embeddings(c2ws_infer, Ks, self._mg3_target_h, self._mg3_target_w)
        c1 = self._mg3_target_h // self._mg3_lat_h
        c2 = self._mg3_target_w // self._mg3_lat_w
        plucker = rearrange(
            plucker,
            "f (h c1) (w c2) c -> (f h w) (c c1 c2)",
            c1=c1,
            c2=c2,
        )
        plucker = plucker[None, ...]
        return rearrange(
            plucker,
            "b (f h w) c -> b c f h w",
            f=len(tgt_indices),
            h=self._mg3_lat_h,
            w=self._mg3_lat_w,
        )

    def _build_plucker_from_pose(self, c2ws_pose: torch.Tensor, intrinsics_seq: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Build plucker embeddings when poses are already on the target timeline."""
        modules = self._get_official_modules()
        assert self._mg3_target_h is not None and self._mg3_target_w is not None
        assert self._mg3_lat_h is not None and self._mg3_lat_w is not None
        if intrinsics_seq is None:
            Ks = self._mg3_base_intrinsics.to(device=c2ws_pose.device, dtype=c2ws_pose.dtype).repeat(c2ws_pose.shape[0], 1)
        else:
            Ks = intrinsics_seq.to(device=c2ws_pose.device, dtype=c2ws_pose.dtype)
        plucker = modules["cam_utils"].get_plucker_embeddings(c2ws_pose, Ks, self._mg3_target_h, self._mg3_target_w)
        c1 = self._mg3_target_h // self._mg3_lat_h
        c2 = self._mg3_target_w // self._mg3_lat_w
        plucker = rearrange(
            plucker,
            "f (h c1) (w c2) c -> (f h w) (c c1 c2)",
            c1=c1,
            c2=c2,
        )
        plucker = plucker[None, ...]
        return rearrange(
            plucker,
            "b (f h w) c -> b c f h w",
            f=c2ws_pose.shape[0],
            h=self._mg3_lat_h,
            w=self._mg3_lat_w,
        )

    def _build_memory_metadata(self, segment_idx: int, current_start_frame_idx: int, current_end_frame_idx: int) -> dict[str, Any]:
        # Official source: pipeline/inference_pipeline.py and utils/cam_utils.py.
        # Current downstream model code only requires c2ws_plucker_emb / keyboard_cond / mouse_cond,
        # but we still stage the memory-facing metadata here so the runner owns segment bookkeeping.
        #
        # Later segments can attend to a sparse set of previously generated latent
        # frames. This method selects those frames, prepares their latent indices, and
        # builds the matching plucker embeddings for the memory branch.
        if segment_idx == 0 or not self._mg3_generated_latent_history:
            return {
                "x_memory": None,
                "timestep_memory": None,
                "keyboard_cond_memory": None,
                "mouse_cond_memory": None,
                "memory_latent_idx": None,
                "plucker_emb_with_memory": None,
            }

        modules = self._get_official_modules()
        assert self._mg3_extrinsics_all is not None
        assert self._mg3_base_intrinsics is not None

        def align_frame_to_block(frame_idx: int) -> int:
            return (frame_idx - 1) // 4 * 4 + 1 if frame_idx > 0 else 1

        def get_latent_idx(frame_idx: int) -> int:
            return (frame_idx - 1) // 4 + 1 if frame_idx > 0 else 0

        selected_index_base = [current_end_frame_idx - offset for offset in range(1, 34, 8)]
        selected_index = modules["cam_utils"].select_memory_idx_fov(
            self._mg3_extrinsics_all,
            current_start_frame_idx,
            selected_index_base,
            use_gpu=torch.cuda.is_available(),
        )
        if selected_index:
            # The official code hard-pins the oldest memory anchor to frame 4.
            selected_index[-1] = 4

        memory_pluckers = []
        latent_idx = []
        for mem_idx, reference_idx in zip(selected_index, selected_index_base):
            latent_idx.append(get_latent_idx(mem_idx))
            mem_idx_aligned = align_frame_to_block(mem_idx)
            mem_block = self._mg3_extrinsics_all[mem_idx_aligned : mem_idx_aligned + 4]
            mem_src = np.linspace(mem_idx_aligned, mem_idx_aligned + 3, mem_block.shape[0])
            mem_tgt = np.array([mem_idx_aligned + 3], dtype=np.float32)
            mem_pose = modules["cam_utils"]._interpolate_camera_poses_handedness(
                src_indices=mem_src,
                src_rot_mat=mem_block[:, :3, :3].cpu().numpy(),
                src_trans_vec=mem_block[:, :3, 3].cpu().numpy(),
                tgt_indices=mem_tgt,
            )
            reference_pose = self._mg3_extrinsics_all[reference_idx : reference_idx + 1]
            rel_pair = torch.cat([reference_pose, mem_pose], dim=0)
            rel_pose = modules["cam_utils"].compute_relative_poses(rel_pair, framewise=False)[1:2]
            memory_pluckers.append(self._build_plucker_from_pose(rel_pose.to(device=AI_DEVICE)))

        current_plucker = self._build_or_get_segment_camera_only(segment_idx)
        plucker_with_memory = torch.cat(memory_pluckers + [current_plucker], dim=2) if memory_pluckers else current_plucker
        src = torch.cat(self._mg3_generated_latent_history, dim=1)
        valid_latent_idx = [idx for idx in latent_idx if 0 <= idx < src.shape[1]]
        if valid_latent_idx != latent_idx:
            logger.warning(
                "[matrix-game-3] memory latent index truncated from {} to {} because generated latent history is shorter.",
                latent_idx,
                valid_latent_idx,
            )
        x_memory = src[:, valid_latent_idx].unsqueeze(0).to(device=AI_DEVICE, dtype=GET_DTYPE()) if valid_latent_idx else None
        if x_memory is None:
            timestep_memory = None
            keyboard_cond_memory = None
            mouse_cond_memory = None
        else:
            timestep_memory = x_memory.new_zeros((1, x_memory.shape[2] * x_memory.shape[3] * x_memory.shape[4] // 4))
            keyboard_cond_memory = -torch.ones((1, len(valid_latent_idx), self.keyboard_dim_in), device=x_memory.device, dtype=x_memory.dtype)
            mouse_cond_memory = torch.ones((1, len(valid_latent_idx), self.mouse_dim_in), device=x_memory.device, dtype=x_memory.dtype)

        return {
            "x_memory": x_memory,
            "timestep_memory": timestep_memory,
            "keyboard_cond_memory": keyboard_cond_memory,
            "mouse_cond_memory": mouse_cond_memory,
            "memory_latent_idx": valid_latent_idx,
            "plucker_emb_with_memory": plucker_with_memory,
        }

    def _build_or_get_segment_camera_only(self, segment_idx: int) -> torch.Tensor:
        """Access just the camera plucker embedding without rebuilding other state."""
        state = self._segment_states.get(segment_idx)
        if state is not None and "c2ws_plucker_emb" in state.dit_cond_dict:
            return state.dit_cond_dict["c2ws_plucker_emb"]
        state = self._build_or_get_segment_state(segment_idx)
        return state.dit_cond_dict["c2ws_plucker_emb"]

    def _build_or_get_segment_state(self, segment_idx: int) -> MatrixGame3SegmentState:
        """Materialize one segment's complete conditioning package.

        This is the core of the adapter. It decides:
        - which absolute frames this segment covers;
        - which latent timesteps are fixed from prior context;
        - which camera/action conditions should be sliced for this window;
        - which overlap should be trimmed after decoding.
        """
        if segment_idx in self._segment_states:
            return self._segment_states[segment_idx]

        if self._mg3_interactive and (self._mg3_keyboard_all is None or self._mg3_keyboard_all.shape[1] < self.first_clip_frame + segment_idx * self.incremental_segment_frames):
            self._append_interactive_segment_controls(segment_idx)

        assert self._mg3_keyboard_all is not None
        assert self._mg3_mouse_all is not None
        assert self._mg3_extrinsics_all is not None
        first_clip = segment_idx == 0

        def get_latent_idx(frame_idx: int) -> int:
            return (frame_idx - 1) // 4 + 1 if frame_idx > 0 else 0

        current_end_frame_idx = self.first_clip_frame if first_clip else self.first_clip_frame + segment_idx * self.incremental_segment_frames
        current_start_frame_idx = 0 if first_clip else current_end_frame_idx - self.clip_frame
        frame_count = self.first_clip_frame if first_clip else self.clip_frame
        latent_start_idx = get_latent_idx(current_start_frame_idx)
        latent_end_idx = get_latent_idx(current_end_frame_idx)
        fixed_latent_frames = 1 if first_clip else self.conditioning_latent_frames
        # After decoding, the first RGB frames of every later segment correspond to
        # history that was already emitted by the previous segment, so they are dropped.
        decode_trim_frames = 0 if first_clip else 1 + self.config["vae_stride"][0] * (fixed_latent_frames - 1)
        append_latent_start = 0 if first_clip else fixed_latent_frames

        c2ws_chunk = self._mg3_extrinsics_all[current_start_frame_idx:current_end_frame_idx].to(device=AI_DEVICE)
        src_indices = np.linspace(current_start_frame_idx, current_end_frame_idx - 1, frame_count)

        intrinsics_chunk = None
        if self._mg3_intrinsics_all is not None:
            intrinsics_chunk = self._mg3_intrinsics_all[current_start_frame_idx:current_end_frame_idx]

        latent_shape = self._segment_latent_shape(self._mg3_lat_h, self._mg3_lat_w, frame_count)
        # The latent timeline is coarser than RGB time because Wan2.2 uses a temporal
        # VAE stride of 4. Later segments start interpolation at `start + 3` so the
        # first 4 latent slots line up with the carried-over conditioning tail.
        tgt_indices = np.linspace(0 if first_clip else current_start_frame_idx + 3, current_end_frame_idx - 1, latent_shape[1])

        camera_only = self._build_plucker_from_c2ws(
            c2ws_chunk,
            src_indices=src_indices,
            tgt_indices=tgt_indices,
            framewise=True,
            intrinsics_seq=intrinsics_chunk,
        ).to(device=AI_DEVICE, dtype=GET_DTYPE())

        keyboard_cond = self._mg3_keyboard_all[:, current_start_frame_idx:current_end_frame_idx].to(device=AI_DEVICE, dtype=GET_DTYPE())
        mouse_cond = self._mg3_mouse_all[:, current_start_frame_idx:current_end_frame_idx].to(device=AI_DEVICE, dtype=GET_DTYPE())

        vae_encoder_out = torch.zeros(latent_shape, device=AI_DEVICE, dtype=GET_DTYPE())
        if first_clip:
            # Segment 0 is anchored by the input image latent in the first temporal slot.
            vae_encoder_out[:, :1] = self.inputs["image_encoder_output"]["vae_encoder_out"][:, :1]
        else:
            if self._mg3_tail_latents is None:
                raise RuntimeError("matrix-game-3 segment requested without previous tail latents")
            # Later segments are conditioned on the last 4 latent frames produced by the
            # previous segment, which creates temporal continuity across chunk boundaries.
            vae_encoder_out[:, : self.conditioning_latent_frames] = self._mg3_tail_latents.to(device=AI_DEVICE, dtype=GET_DTYPE())

        # Fields below intentionally stay in the standard LightX2V image_encoder_output["dit_cond_dict"]
        # container so downstream model / infer / weights code can consume them without a new top-level protocol.
        dit_cond_dict: dict[str, Any] = {
            "keyboard_cond": keyboard_cond,
            "mouse_cond": mouse_cond,
            "c2ws_plucker_emb": camera_only,
            "predict_latent_idx": (latent_start_idx, latent_end_idx),
            "segment_frame_range": (current_start_frame_idx, current_end_frame_idx),
            "segment_idx": segment_idx,
            "first_clip": first_clip,
        }
        dit_cond_dict.update(self._build_memory_metadata(segment_idx, current_start_frame_idx, current_end_frame_idx))

        state = MatrixGame3SegmentState(
            segment_idx=segment_idx,
            first_clip=first_clip,
            current_start_frame_idx=current_start_frame_idx,
            current_end_frame_idx=current_end_frame_idx,
            frame_count=frame_count,
            fixed_latent_frames=fixed_latent_frames,
            latent_shape=latent_shape,
            decode_trim_frames=decode_trim_frames,
            append_latent_start=append_latent_start,
            keyboard_cond=keyboard_cond,
            mouse_cond=mouse_cond,
            vae_encoder_out=vae_encoder_out,
            dit_cond_dict=dit_cond_dict,
        )
        self._segment_states[segment_idx] = state
        return state

    def _apply_segment_scheduler_state(self, segment_state: MatrixGame3SegmentState):
        """Seed the scheduler latents and mask for the current segment window."""
        scheduler = self.model.scheduler
        latents = torch.randn(
            tuple(segment_state.latent_shape),
            device=AI_DEVICE,
            dtype=GET_DTYPE(),
            generator=self._mg3_noise_generator,
        )
        scheduler.vae_encoder_out = segment_state.vae_encoder_out.to(device=AI_DEVICE, dtype=GET_DTYPE())
        scheduler.mask = torch.ones_like(latents)
        # Mask value 0 means "keep the provided latent conditioning", while 1 means
        # "sample this slot from noise through the diffusion process".
        scheduler.mask[:, : segment_state.fixed_latent_frames] = 0
        scheduler.latents = (1.0 - scheduler.mask) * scheduler.vae_encoder_out + scheduler.mask * latents

    @ProfilingContext4DebugL1(
        "Init run segment",
        recorder_mode=GET_RECORDER_MODE(),
        metrics_func=monitor_cli.lightx2v_run_init_run_segment_duration,
        metrics_labels=["WanMatrixGame3Runner"],
    )
    def init_run_segment(self, segment_idx):
        # Official source: pipeline/inference_pipeline.py and inference_interactive_pipeline.py
        # refresh per-segment action / camera / latent-conditioning state here so the outer lifecycle
        # remains the standard LightX2V segment loop.
        #
        # The base runner calls this before every segment. We use that hook to swap in
        # the next segment's control tensors and, for later segments, reset the scheduler
        # so it samples against the new latent shape and conditioning mask.
        self.segment_idx = segment_idx
        segment_state = self._build_or_get_segment_state(segment_idx)
        self._mg3_current_segment_state = segment_state
        self.input_info.latent_shape = segment_state.latent_shape
        self.inputs["image_encoder_output"]["dit_cond_dict"] = segment_state.dit_cond_dict
        self.inputs["image_encoder_output"]["vae_encoder_out"] = segment_state.vae_encoder_out
        if segment_idx > 0:
            self.model.scheduler.reset(self.input_info.seed, segment_state.latent_shape)
            self._apply_segment_scheduler_state(segment_state)

    def run_segment(self, segment_idx=0):
        infer_steps = self.model.scheduler.infer_steps
        dump_all_scheduler_steps = bool(self.config.get("debug_dump_all_scheduler_steps", False))
        debug_steps_dir = _matrix_game3_resolve_debug_steps_dir(getattr(self.input_info, "save_result_path", None)) if dump_all_scheduler_steps else None
        if debug_steps_dir is not None:
            debug_steps_dir.mkdir(parents=True, exist_ok=True)
            summary_path = debug_steps_dir / "steps_summary.jsonl"
            if summary_path.exists():
                summary_path.unlink()
            mask_path = debug_steps_dir / "mask.pt"
            torch.save(self.model.scheduler.mask.detach().cpu(), mask_path)
            logger.info("[matrix-game-3][debug] saving per-step scheduler tensors to {}", debug_steps_dir)

        for step_index in range(infer_steps):
            with ProfilingContext4DebugL1(
                "Run Dit every step",
                recorder_mode=GET_RECORDER_MODE(),
                metrics_func=monitor_cli.lightx2v_run_per_step_dit_duration,
                metrics_labels=[step_index + 1, infer_steps],
            ):
                if self.video_segment_num == 1:
                    self.check_stop()
                logger.info(f"==> step_index: {step_index + 1} / {infer_steps}")
                latent_before = self.model.scheduler.latents.detach().clone()

                with ProfilingContext4DebugL1("step_pre"):
                    self.model.scheduler.step_pre(step_index=step_index)

                with ProfilingContext4DebugL1("🚀 infer_main"):
                    self.model.infer(self.inputs)
                noise_pred = self.model.scheduler.noise_pred.detach().clone() if self.model.scheduler.noise_pred is not None else None
                noise_pred_cond = (
                    self.model.scheduler.noise_pred_cond.detach().clone() if getattr(self.model.scheduler, "noise_pred_cond", None) is not None else None
                )
                noise_pred_uncond = (
                    self.model.scheduler.noise_pred_uncond.detach().clone() if getattr(self.model.scheduler, "noise_pred_uncond", None) is not None else None
                )
                noise_pred_guided = (
                    self.model.scheduler.noise_pred_guided.detach().clone() if getattr(self.model.scheduler, "noise_pred_guided", None) is not None else None
                )
                forward_kwargs_cond = getattr(self.model.scheduler, "forward_kwargs_cond", None)
                forward_kwargs_uncond = getattr(self.model.scheduler, "forward_kwargs_uncond", None)

                with ProfilingContext4DebugL1("step_post"):
                    self.model.scheduler.step_post()
                latent_after = self.model.scheduler.latents.detach().clone()

                if dump_all_scheduler_steps:
                    step_payload = {
                        "step_index": step_index,
                        "timestep": float(self.model.scheduler._solver.timesteps[step_index].item())
                        if getattr(self.model.scheduler, "_solver", None) is not None
                        else None,
                        "fixed_latent_frames": int((self.model.scheduler.mask.detach().to(torch.float32).amax(dim=(0, 2, 3)) > 0.0).nonzero(as_tuple=False)[0].item())
                        if self.model.scheduler.mask is not None and self.model.scheduler.mask.ndim == 4
                        and (self.model.scheduler.mask.detach().to(torch.float32).amax(dim=(0, 2, 3)) > 0.0).any()
                        else None,
                        "mask": _matrix_game3_basic_tensor_probe(self.model.scheduler.mask.amax(dim=(0, 2, 3)))
                        if self.model.scheduler.mask is not None and self.model.scheduler.mask.ndim == 4
                        else _matrix_game3_basic_tensor_probe(self.model.scheduler.mask)
                        if self.model.scheduler.mask is not None
                        else None,
                        "latent_before": _matrix_game3_basic_tensor_probe(latent_before),
                        "noise_pred_cond": _matrix_game3_basic_tensor_probe(noise_pred_cond) if noise_pred_cond is not None else None,
                        "noise_pred_uncond": _matrix_game3_basic_tensor_probe(noise_pred_uncond) if noise_pred_uncond is not None else None,
                        "noise_pred_guided": _matrix_game3_basic_tensor_probe(noise_pred_guided) if noise_pred_guided is not None else None,
                        "forward_kwargs_cond": forward_kwargs_cond,
                        "forward_kwargs_uncond": forward_kwargs_uncond,
                        "noise_pred": _matrix_game3_basic_tensor_probe(noise_pred) if noise_pred is not None else None,
                        "latent_after": _matrix_game3_basic_tensor_probe(latent_after),
                    }
                    torch.save(latent_before.cpu(), debug_steps_dir / f"step_{step_index:03d}_latent_before.pt")
                    if noise_pred_cond is not None:
                        torch.save(noise_pred_cond.cpu(), debug_steps_dir / f"step_{step_index:03d}_noise_pred_cond.pt")
                    if noise_pred_uncond is not None:
                        torch.save(noise_pred_uncond.cpu(), debug_steps_dir / f"step_{step_index:03d}_noise_pred_uncond.pt")
                    if noise_pred_guided is not None:
                        torch.save(noise_pred_guided.cpu(), debug_steps_dir / f"step_{step_index:03d}_noise_pred_guided.pt")
                    if noise_pred is not None:
                        torch.save(noise_pred.cpu(), debug_steps_dir / f"step_{step_index:03d}_noise_pred.pt")
                    torch.save(latent_after.cpu(), debug_steps_dir / f"step_{step_index:03d}_latent_after.pt")
                    with (debug_steps_dir / "steps_summary.jsonl").open("a", encoding="utf-8") as f:
                        f.write(json.dumps(step_payload, ensure_ascii=False) + "\n")
                    _matrix_game3_log_debug_payload(f"step_{step_index}_scheduler_debug", step_payload)

                if getattr(self.model.scheduler, "debug_stop_requested", False):
                    logger.warning("[matrix-game-3][debug] stopping after scheduler latent probe.")
                    break

                if self.progress_callback:
                    current_step = segment_idx * infer_steps + step_index + 1
                    total_all_steps = self.video_segment_num * infer_steps
                    self.progress_callback((current_step / total_all_steps) * 100, 100)

        if segment_idx is not None and segment_idx == self.video_segment_num - 1:
            del self.inputs
            torch_device_module.empty_cache()

        latents = self.model.scheduler.latents
        self._mg3_current_segment_full_latents = latents.detach().clone()
        return latents

    def _log_vae_decode_probe_and_save_frame(self, decoded_images: torch.Tensor) -> None:
        _matrix_game3_log_debug_payload("vae_decode_output", _matrix_game3_basic_tensor_probe(decoded_images))

        if decoded_images.ndim != 5:
            logger.warning("[matrix-game-3][debug] unexpected decoded image ndim: {}. Skipping first-frame save.", decoded_images.ndim)
            return

        save_result_path = getattr(self.input_info, "save_result_path", None)
        frame_indices = [1, 32, 56]
        total_frames = decoded_images.shape[2]
        for frame_index in frame_indices:
            if frame_index >= total_frames:
                logger.warning(
                    "[matrix-game-3][debug] requested decoded frame {} but total decoded frames are only {}. Skipping.",
                    frame_index,
                    total_frames,
                )
                continue
            frame = decoded_images[0, :, frame_index].detach().float().add_(1.0).mul_(0.5).clamp_(0.0, 1.0).cpu()
            frame_path = _matrix_game3_resolve_debug_frame_path(save_result_path, frame_index)
            frame_path.parent.mkdir(parents=True, exist_ok=True)
            TF.to_pil_image(frame).save(frame_path)
            logger.info("[matrix-game-3][debug] saved decoded frame {} to {}", frame_index, frame_path)

    def run_main(self):
        self.init_run()
        if self.config.get("compile", False) and hasattr(self.model, "comple"):
            self.model.select_graph_for_compile(self.input_info)
        for segment_idx in range(self.video_segment_num):
            logger.info(f"🔄 start segment {segment_idx + 1}/{self.video_segment_num}")
            with ProfilingContext4DebugL1(
                f"segment end2end {segment_idx + 1}/{self.video_segment_num}",
                recorder_mode=GET_RECORDER_MODE(),
                metrics_func=monitor_cli.lightx2v_run_segments_end2end_duration,
                metrics_labels=["WanMatrixGame3Runner"],
            ):
                self.check_stop()
                self.init_run_segment(segment_idx)
                latents = self.run_segment(segment_idx)
                if self.config.get("debug_stop_before_vae_decode", False):
                    logger.warning("[matrix-game-3][debug] scheduler step dump complete. Stopping before VAE decode.")
                    self.end_run()
                    return None
                if getattr(self.model.scheduler, "debug_stop_requested", False):
                    logger.warning("[matrix-game-3][debug] scheduler probe complete. Stopping before VAE decode.")
                    self.end_run()
                    return None
                if self.config.get("use_stream_vae", False):
                    frames = []
                    for frame_segment in self.run_vae_decoder_stream(latents):
                        frames.append(frame_segment)
                        logger.info(f"frame sagment: {len(frames)} done")
                    self.gen_video = torch.cat(frames, dim=2)
                else:
                    self.gen_video = self.run_vae_decoder(latents)
                if self.config.get("debug_stop_after_vae_decode", False):
                    self._log_vae_decode_probe_and_save_frame(self.gen_video)
                    logger.warning("[matrix-game-3][debug] stopping after VAE decode probe.")
                    self.end_run()
                    return None
                self.end_run_segment(segment_idx)
        gen_video_final = self.process_images_after_vae_decoder()
        self.end_run()
        return gen_video_final

    def end_run_segment(self, segment_idx=None):
        """Carry segment outputs forward and remove overlap from decoded frames."""
        if self._mg3_current_segment_state is None or self._mg3_current_segment_full_latents is None:
            raise RuntimeError("matrix-game-3 end_run_segment called before the current segment state was prepared")

        full_latents = self._mg3_current_segment_full_latents
        # full_latents follows Wan2.2 runner convention: [C, T, H, W].
        # Keep only the tail that should condition the next segment.
        self._mg3_tail_latents = full_latents[:, -self.conditioning_latent_frames :].detach().clone()
        # Only append genuinely new latent timesteps to history; the carried-over prefix
        # belongs to the previous segment and would otherwise duplicate memory entries.
        new_latents = full_latents[:, self._mg3_current_segment_state.append_latent_start :].detach().clone()
        self._mg3_generated_latent_history.append(new_latents)

        segment_video = self.gen_video
        if self._mg3_current_segment_state.decode_trim_frames > 0:
            # Remove RGB frames that correspond to the reused latent prefix.
            segment_video = segment_video[:, :, self._mg3_current_segment_state.decode_trim_frames :]
        self.gen_video = segment_video
        self.gen_video_final = segment_video if self.gen_video_final is None else torch.cat([self.gen_video_final, segment_video], dim=2)
        self._mg3_current_segment_state = None
        self._mg3_current_segment_full_latents = None

    def process_images_after_vae_decoder(self):
        # `DefaultRunner.process_images_after_vae_decoder()` expects `gen_video_final`
        # to already contain the full stitched clip.
        if self.gen_video_final is None:
            self.gen_video_final = self.gen_video
        return super().process_images_after_vae_decoder()
