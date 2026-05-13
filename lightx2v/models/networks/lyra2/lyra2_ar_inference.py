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

import gc
import os
from typing import List, Optional

import numpy as np
import torch
import tqdm
from einops import rearrange
from loguru import logger
from megatron.core import parallel_state

from lightx2v.models.networks.lyra2.forward_warp_utils_pytorch import (
    reliable_depth_mask_range_batch,
)
from lightx2v.models.networks.lyra2.lyra2_utils import to as misc_to
from lightx2v.models.networks.lyra2.sparse3d_cache import Sparse3DCache
from lightx2v.models.video_encoders.lyra2.video import save_img_or_video

torch.enable_grad(False)


def _get_vae_handles(model):
    vae_iface = model.tokenizer
    vae_wrap = vae_iface.model  # WanVAE wrapper
    vae_core = vae_wrap.model  # WanVAE_ core
    return vae_iface, vae_wrap, vae_core


def _prime_encoder_cache_with_history(init_video, vae_wrap, vae_core, model=None, enable_offload=False):
    """Advance encoder cache through the history pixels and return history_latents plus the live cache."""
    # Offload diffusion model to CPU before VAE operations
    _offload_diffusion_to_cpu(model, enable_offload)

    # Fresh encode using model helpers, then clone caches and normalize
    vae_core.clear_cache()
    with vae_wrap.context:
        video_cast = init_video.to(vae_wrap.dtype) if not vae_wrap.is_amp else init_video
        feats = model._vae_encode_range_stream(video_cast, 0, video_cast.shape[2], skip_first_frame=False)
        enc_feat_cache = model._clone_vae_cache(vae_core._enc_feat_map)
        history_latents = model._encoder_feats_to_normalized_latents(feats).contiguous().to(init_video.dtype)

    # Restore diffusion model to GPU after VAE operations
    _restore_diffusion_to_gpu(model, enable_offload)

    return history_latents, enc_feat_cache


def _decode_new_latent_chunk(vae_wrap, vae_core, dec_feat_cache, latent_chunk, latent_offset, model=None, enable_offload=False):
    """Stream-decode new latent chunk given current decoder cache; return pixel frames for this chunk."""
    # Offload diffusion model to CPU before VAE operations
    _offload_diffusion_to_cpu(model, enable_offload)

    # Unnormalize per-frame to mu using offset, then apply channel unscale, conv2, and stream through decoder
    B, C, T_new, H, W = latent_chunk.shape
    # Build per-frame stats slice for offset positions
    if T_new == 1 and latent_offset == 0:
        mu = latent_chunk * vae_wrap.img_std.type_as(latent_chunk) + vae_wrap.img_mean.type_as(latent_chunk)
    else:
        mu = latent_chunk * vae_wrap.video_std[:, :, :1].type_as(latent_chunk) + vae_wrap.video_mean[:, :, :1].type_as(latent_chunk)
    # Channel unscale
    mean_c, inv_std_c = vae_wrap.scale[0], vae_wrap.scale[1]
    if torch.is_tensor(mean_c):
        z = mu / inv_std_c.view(1, vae_core.z_dim, 1, 1, 1).type_as(mu) + mean_c.view(1, vae_core.z_dim, 1, 1, 1).type_as(mu)
    else:
        z = mu / inv_std_c + mean_c
    with vae_wrap.context:
        if not vae_wrap.is_amp:
            z = z.to(vae_wrap.dtype)
        x = vae_core.conv2(z)
        # Decode one temporal slice at a time to mirror encoder streaming and keep memory low
        outs = []
        for t in range(T_new):
            feat_idx = [0]
            out_t = vae_core.decoder(x[:, :, t : t + 1, :, :], feat_cache=dec_feat_cache, feat_idx=feat_idx)
            outs.append(out_t)
        video_chunk = torch.cat(outs, dim=2)

    # Restore diffusion model to GPU after VAE operations
    _restore_diffusion_to_gpu(model, enable_offload)

    return video_chunk


def _add_tiny_offsets_to_extrinsics_for_pose_alignment(
    extrinsics_np: np.ndarray,
    variance_threshold: float = 1e-10,
    offset_scale: float = 1e-6,
    offset_freq_x: float = 0.00001,
    offset_freq_y: float = 0.000013,
    offset_freq_z: float = 0.000007,
) -> np.ndarray:
    """
    Add tiny offsets to extrinsics to avoid degenerate covariance in pose alignment.

    This function detects when camera positions are nearly identical (which causes
    degenerate covariance errors in Umeyama alignment) and adds minimal offsets
    only when needed.

    Args:
        extrinsics_np: [N, 4, 4] numpy array of extrinsics (world-to-camera matrices)
        variance_threshold: Threshold for position variance to detect degenerate case (default 1e-10)
        offset_scale: Scale of offsets to add (default 1e-6)
        offset_freq_x: Frequency for X-axis offset pattern (default 0.00001)
        offset_freq_y: Frequency for Y-axis offset pattern (default 0.000013)
        offset_freq_z: Frequency for Z-axis offset pattern (default 0.000007)

    Returns:
        Modified extrinsics_np with tiny offsets added if needed
    """
    # Extract camera positions from extrinsics (world-to-camera: translation is -R^T @ camera_pos)
    camera_positions = []
    for ext in extrinsics_np:
        R = ext[:3, :3]
        t = ext[:3, 3]
        # Recover camera position: camera_pos = -R^T @ t
        camera_pos = -R.T @ t
        camera_positions.append(camera_pos)
    camera_positions = np.stack(camera_positions, axis=0)

    # Check if positions are nearly identical (small variance)
    pos_variance = np.var(camera_positions, axis=0).sum()
    if pos_variance < variance_threshold:
        # Add tiny offsets to provide minimal variation for pose alignment
        # Using slow sinusoidal patterns with different frequencies per axis:
        # - Different frequencies ensure independent variation on each axis
        # - Mixing sin/cos ensures smooth, continuous variation
        # - Very slow frequencies keep offsets imperceptible
        # - Based on absolute frame index 'i' for consistency across autoregressive chunks
        extrinsics_modified = extrinsics_np.copy()
        for i, ext in enumerate(extrinsics_modified):
            offset = np.array(
                [
                    offset_scale * np.sin(2 * np.pi * i * offset_freq_x),  # X: sin with configurable frequency
                    offset_scale * np.cos(2 * np.pi * i * offset_freq_y),  # Y: cos with configurable frequency
                    offset_scale * np.sin(2 * np.pi * i * offset_freq_z),  # Z: sin with configurable frequency
                ]
            )
            R = ext[:3, :3]
            # Modify translation: new_t = old_t - R @ offset
            extrinsics_modified[i, :3, 3] = ext[:3, 3] - R @ offset
        return extrinsics_modified

    return extrinsics_np


def _predict_da3_depth_window(
    *,
    da3_model,
    history_frames: torch.Tensor,
    start_index: int,
    abs_last_idx: int,
    cam_w2c: torch.Tensor,
    intrinsics: torch.Tensor,
    frame_interval: int,
    max_history_frames: int,
    process_res: int = 504,
    process_res_method: str = "upper_bound_resize",
    add_pose_alignment_offsets: bool = False,
    include_ar_chunk_last_frames: bool = False,
    ar_chunk_size_frames: int | None = None,
    return_raw_predicted_pose: bool = False,
):
    """Depth Anything 3 inference for a temporal window ending at abs_last_idx.

    - Select frames: last, last-N, last-2N, ... going backwards, up to max_history_frames and >= 0.
    - Optionally also include the last frame of each AR chunk: 0, N, 2N, ... (de-duplicated).
      When enabled, these extra frames are added on top of the base selection (no total-count budget enforcement).
    - Uses images from history_frames (in [-1,1]) converted to uint8, and corresponding cameras.
    - Returns numpy prediction plus the selected absolute frame indices.
    """
    assert history_frames.dim() == 5, "history_frames must be [B,C,T,H,W]"
    B, C, T_total, H, W = history_frames.shape
    if B != 1:
        raise ValueError("DA3 backend currently supports batch size B=1 only.")

    if abs_last_idx < 0:
        return {"frame_indices": [], "prediction": None}

    # Build absolute frame indices along the global video timeline
    selected_frames: List[int] = []
    step = max(int(frame_interval), 1)
    for k in range(int(max_history_frames)):
        f = abs_last_idx - k * step
        if f < 0:
            break
        selected_frames.append(f)
    if include_ar_chunk_last_frames:
        if ar_chunk_size_frames is None:
            raise ValueError("ar_chunk_size_frames must be provided when include_ar_chunk_last_frames=True")
        chunk = max(int(ar_chunk_size_frames), 1)

        selected_frames.extend(list(range(0, int(abs_last_idx) + 1, chunk)))

    selected_frames = sorted(set(int(x) for x in selected_frames))
    if not selected_frames:
        return {"frame_indices": [], "prediction": None}

    images: List[np.ndarray] = []
    exts: List[np.ndarray] = []
    ixts: List[np.ndarray] = []

    hist_cpu = history_frames[0].detach().cpu()  # [C,T,H,W]
    cam_cpu = cam_w2c.detach().cpu()
    intr_cpu = intrinsics.detach().cpu()

    for f in selected_frames:
        t = start_index + f
        if t < 0 or t >= T_total:
            continue
        frame_chw = hist_cpu[:, t]  # [C,H,W] in [-1,1]
        frame_0_1 = (frame_chw * 0.5 + 0.5).clamp(0.0, 1.0)
        frame_hwc = frame_0_1.permute(1, 2, 0).float().numpy()
        images.append(np.clip(frame_hwc * 255.0 + 0.5, 0, 255).astype(np.uint8))
        exts.append(cam_cpu[0, f].numpy().astype(np.float32))
        ixts.append(intr_cpu[0, f].numpy().astype(np.float32))

    if not images:
        return {"frame_indices": [], "prediction": None}

    extrinsics_np = np.stack(exts, axis=0).astype(np.float32)
    intrinsics_np = np.stack(ixts, axis=0).astype(np.float32)

    # Use the input image long side as DA3 process_res; with upper_bound_resize this avoids resizing.
    process_res = int(max(images[0].shape[0], images[0].shape[1]))
    process_res_method = "upper_bound_resize"

    # Optionally add tiny offsets to extrinsics to avoid degenerate covariance in pose alignment
    # This only affects DA3 inference, not the actual camera trajectory
    if add_pose_alignment_offsets:
        extrinsics_np = _add_tiny_offsets_to_extrinsics_for_pose_alignment(
            extrinsics_np,
        )

    prediction = da3_model.inference(
        image=images,
        extrinsics=extrinsics_np,
        intrinsics=intrinsics_np,
        align_to_input_extrinsics=not return_raw_predicted_pose,
        align_to_input_ext_scale=not return_raw_predicted_pose,
        infer_gs=False,
        process_res=process_res,
        process_res_method=process_res_method,
        reorder_cam_token_by_reference=True,
        export_dir=None,
        export_format="mini_npz",
    )

    return {
        "frame_indices": selected_frames,
        "prediction": prediction,
    }


def _camera_centers_from_w2c(w2c: torch.Tensor) -> torch.Tensor:
    """Compute camera centers from world-to-camera matrices."""
    R = w2c[:, :3, :3]
    t = w2c[:, :3, 3]
    return -(R.transpose(1, 2) @ t.unsqueeze(-1)).squeeze(-1)


def _intrinsics_vec_to_k33(intrinsics_vec: torch.Tensor) -> torch.Tensor:
    """Convert VIPE intrinsics (fx,fy,cx,cy,...) to 3x3 matrices."""
    if intrinsics_vec.ndim != 2 or intrinsics_vec.shape[1] < 4:
        raise ValueError(f"Expected intrinsics shape (T,>=4), got {tuple(intrinsics_vec.shape)}")
    fx, fy, cx, cy = (intrinsics_vec[:, 0], intrinsics_vec[:, 1], intrinsics_vec[:, 2], intrinsics_vec[:, 3])
    T = intrinsics_vec.shape[0]
    K = torch.zeros((T, 3, 3), dtype=intrinsics_vec.dtype, device=intrinsics_vec.device)
    K[:, 0, 0] = fx
    K[:, 1, 1] = fy
    K[:, 0, 2] = cx
    K[:, 1, 2] = cy
    K[:, 2, 2] = 1.0
    return K


def _offload_diffusion_to_cpu(model, enable_offload: bool):
    """Move diffusion model to CPU if offload is enabled."""
    if enable_offload and hasattr(model, "net"):
        model.net.cpu()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _restore_diffusion_to_gpu(model, enable_offload: bool):
    """Move diffusion model back to GPU if offload is enabled."""
    if enable_offload and hasattr(model, "net"):
        model.net.to(model.tensor_kwargs.get("device", "cuda"))


def safe_to(obj, device=None, dtype=None, skip_keys: set | None = None):
    """Recursively move tensors to device/dtype while skipping dtype conversion for specific keys.

    - skip_keys: keys in dict for which we only move to device (keep original dtype)
    """
    if skip_keys is None:
        skip_keys = set()

    def _move_tensor(t: torch.Tensor, force_dtype: bool) -> torch.Tensor:
        if device is None and (dtype is None or not force_dtype):
            return t
        if device is not None and dtype is not None and force_dtype:
            return t.to(device=device, dtype=dtype, non_blocking=True)
        if device is not None:
            return t.to(device=device, non_blocking=True)
        if dtype is not None and force_dtype:
            return t.to(dtype=dtype)
        return t

    if torch.is_tensor(obj):
        return _move_tensor(obj, force_dtype=True)
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            if torch.is_tensor(v):
                out[k] = _move_tensor(v, force_dtype=(k not in skip_keys))
            else:
                out[k] = safe_to(v, device=device, dtype=dtype, skip_keys=skip_keys)
        return out
    if isinstance(obj, (list, tuple)):
        seq = [safe_to(v, device=device, dtype=dtype, skip_keys=skip_keys) for v in obj]
        return type(obj)(seq)
    return obj


def save_output(to_show, vid_save_path):
    legancy_to_show = (1.0 + torch.stack(to_show, dim=0).clamp(-1, 1)) / 2.0  # [n, b, c, t, h, w]

    video_array = (rearrange(legancy_to_show, "n b c t h w -> t (n h) (b w) c") * 255).to(torch.uint8).cpu().numpy()
    logger.info(f"video_array.shape: {video_array.shape} value: {video_array.max()}, {video_array.min()}, save to {vid_save_path}")
    base_stem, _ = os.path.splitext(vid_save_path)
    save_img_or_video(
        rearrange(legancy_to_show, "n b c t h w -> c t (n h) (b w)"),
        base_stem,
        fps=16,
    )
    # Also save a subsampled preview that keeps every 8th frame.
    stride = 8
    subsampled = legancy_to_show[:, :, :, ::stride]
    subsampled_stem = f"{base_stem}_stride{stride}"
    save_img_or_video(
        rearrange(subsampled, "n b c t h w -> c t (n h) (b w)"),
        subsampled_stem,
        fps=16,
    )
    logger.info(f"save video to {vid_save_path}")


class Lyra2InferencePipeline:
    """Stateful pipeline for Lyra2 autoregressive inference."""

    def __init__(
        self,
        *,
        model,
        args,
        first_frame,
        first_depth,
        first_cam_w2c,
        first_intrinsics,
        da3_model=None,
        cp_group=None,
        base_t5_text_embeddings=None,
        base_neg_t5_text_embeddings=None,
        padding_mask=None,
        fps=None,
        vipe_input_dump_dir: Optional[str] = None,
        vipe_input_dump_prefix: Optional[str] = None,
        multiview_data: Optional[dict] = None,
    ):
        self.model = model
        self.args = args
        self.cp_group = cp_group
        self.frames_per_latent = model.framepack_num_frames_per_latent
        self.tokens_per_step = model.framepack_num_new_latent_frames
        self.T_hist = model.framepack_total_max_num_latent_frames - self.tokens_per_step
        self.start_index = (self.T_hist - 1) * self.frames_per_latent
        self.repeat_pixels = (self.T_hist - 1) * self.frames_per_latent + 1

        init_video = first_frame.repeat(1, 1, self.repeat_pixels, 1, 1)
        self.history_frames = init_video
        _, self.vae_wrap, self.vae_core = _get_vae_handles(model)
        self.history_latents, self.enc_feat_cache = _prime_encoder_cache_with_history(init_video, self.vae_wrap, self.vae_core, model, args.offload)
        # Align latent dtype/device to model tensor kwargs for downstream layers.
        self.history_latents = misc_to(self.history_latents, **self.model.tensor_kwargs)
        self.vae_core.clear_cache()
        self.dec_feat_cache = [None] * self.vae_core._conv_num
        _ = _decode_new_latent_chunk(
            self.vae_wrap,
            self.vae_core,
            self.dec_feat_cache,
            self.history_latents,
            latent_offset=0,
            model=model,
            enable_offload=args.offload,
        )
        self.first_latent = self.history_latents[:, :, :1]
        if args.offload:
            self.history_latents = self.history_latents.cpu()
            self.history_frames = self.history_frames.cpu()

        self.last_hist_frame = first_frame[:, :, 0]

        cfg = model.config
        # Lyra2 is collapsed to the pose-conditioned target branch.
        self.use_pose = True
        self.use_plucker = True
        self.use_plucker_relative = False
        self.use_plucker_no_intrinsics = False
        self.use_image_spatial = bool(getattr(cfg, "spatial_memory_use_image", False))
        self.merge_history_buffers = False
        self.num_retrieval_views: int = int(getattr(args, "num_retrieval_views", 1))
        self.warp_video_collect: List[torch.Tensor] = []
        self.vipe_input_dump_dir = vipe_input_dump_dir
        self.vipe_input_dump_prefix = vipe_input_dump_prefix

        cam_w2c_first = first_cam_w2c
        if cam_w2c_first.dim() == 3:
            cam_w2c_first = cam_w2c_first.unsqueeze(1)
        intrinsics_first = first_intrinsics
        if intrinsics_first.dim() == 3:
            intrinsics_first = intrinsics_first.unsqueeze(1)
        self.cam_w2c = cam_w2c_first.to(torch.float32)
        self.intrinsics = intrinsics_first.to(torch.float32)

        # Pose-conditioning state:
        # - We keep a single Sparse3DCache (downsample=4, store_values=True) used for:
        #   1) spatial overlap retrieval
        #   2) depth lookup by frame_id for accumulated-PD warping
        self.retrieval_cache: Optional[Sparse3DCache] = None
        # Most recent "buffer" depth for warping (B,1,H,W). We store it explicitly to avoid requiring
        # the latest history frame to be present in the cache.
        self.buffer_depth_latest: Optional[torch.Tensor] = None
        # Optional per-pixel validity mask (B,1,H,W) for UI/visualization (e.g. non-sky).
        self.buffer_mask_latest: Optional[torch.Tensor] = None
        self.buffer_depth_latest_frame_idx: Optional[int] = None

        self.depth_backend = getattr(args, "depth_backend", "da3")
        self.local_da3_model = da3_model if self.depth_backend == "da3" else None
        self.vipe = None
        # MoGe backend is intentionally not supported in this inference script.

        if self.use_pose:
            assert first_depth is not None, "first_depth is required when pose conditioning is enabled"
            B_cam, T_cam = self.cam_w2c.shape[0], self.cam_w2c.shape[1]
            assert T_cam >= 1, "need at least first camera for pose mode"
            first_img_bchw = first_frame[:, :, 0]
            first_depth_b1hw = first_depth
            if first_depth_b1hw.dim() == 3:
                first_depth_b1hw = first_depth_b1hw.unsqueeze(1)
            first_w2c = self.cam_w2c[:, 0]
            first_K = self.intrinsics[:, 0]
            if self.depth_backend == "da3":
                if self.local_da3_model is None:
                    from lightx2v.models.input_encoders.hf.lyra2.depth_utils import load_da3_model

                    da3_device = model.tensor_kwargs.get("device", "cuda" if torch.cuda.is_available() else "cpu")
                    self.local_da3_model = load_da3_model(
                        da3_model_name=args.da3_model_name,
                        da3_model_path_custom=args.da3_model_path_custom,
                        device=da3_device,
                    )
                    self.local_da3_model.eval()
            else:
                raise ValueError(f"Unsupported depth_backend='{self.depth_backend}' for this inference script.")

            store_device = "cpu" if args.offload else str(first_img_bchw.device.type)
            # For inference we optionally store original depth values in the same cache used for retrieval.
            # This is required when use_accumulated_pcd=True (warping needs per-frame depth lookup by frame_id).
            self.retrieval_cache = Sparse3DCache(
                downsample=4,
                store_device=store_device,
                store_values=True,
            )

            mv_ids = getattr(args, "multiview_ids", None)
            if mv_ids and multiview_data is not None:
                # Multiview input: seed cache with specified frames using negative IDs.
                mv_video = multiview_data["video"]  # [B, C, T, H, W]
                mv_depth = multiview_data["depth"]  # [B, T, ...] or [B, T, 1, H, W]
                mv_w2c = multiview_data["camera_w2c"]  # [B, T, 4, 4]
                mv_K = multiview_data["intrinsics"]  # [B, T, 3, 3]
                for i, src_idx in enumerate(mv_ids):
                    neg_id = -(i + 1)
                    d = mv_depth[:, src_idx].to(torch.float32)
                    if d.dim() == 3:
                        d = d.unsqueeze(1)
                    w = mv_w2c[:, src_idx].to(torch.float32)
                    k = mv_K[:, src_idx].to(torch.float32)
                    self.retrieval_cache.add(d, w, k, latent_index=neg_id, frame_id=neg_id)
                    rgb = mv_video[:, :, src_idx].to(torch.float32)  # [B, C, H, W]
                    self.retrieval_cache.store_rgb(neg_id, rgb)
                    logger.info(f"Multiview cache: added frame src_idx={src_idx} as cache id={neg_id}")
                # Use the first multiview frame's depth as the initial buffer depth.
                first_mv_depth = mv_depth[:, mv_ids[0]].to(torch.float32)
                if first_mv_depth.dim() == 3:
                    first_mv_depth = first_mv_depth.unsqueeze(1)
                self.buffer_depth_latest = first_mv_depth
                self.buffer_depth_latest_frame_idx = 0
            else:
                # Default: seed cache with the first frame (frame_id=0).
                self.retrieval_cache.add(
                    first_depth_b1hw.to(torch.float32),
                    first_w2c.to(torch.float32),
                    first_K.to(torch.float32),
                    latent_index=0,
                    frame_id=0,
                )
                self.buffer_depth_latest = first_depth_b1hw.to(torch.float32)
                self.buffer_depth_latest_frame_idx = 0
            # Seed mask (best-effort): valid where depth > 0. DA3 sky mask is applied later during updates.
            self.buffer_mask_latest = (self.buffer_depth_latest > 0).to(torch.float32)
            if args.offload:
                self.buffer_depth_latest = self.buffer_depth_latest.cpu()
                self.buffer_mask_latest = self.buffer_mask_latest.cpu()
        else:
            self.local_da3_model = None

        self.tokens_generated = 0
        self.ar_idx = 0

        # Predicted-pose update state (populated by _update_depth_cache when da3_use_predicted_pose=True).
        self._predicted_pose_last_w2c: Optional[torch.Tensor] = None
        self._predicted_pose_updated_seed_depth: Optional[torch.Tensor] = None
        self._predicted_pose_updated_seed_mask: Optional[torch.Tensor] = None
        self._predicted_pose_is_first_segment: bool = False

        self.base_t5_text_embeddings = base_t5_text_embeddings
        self.base_neg_t5_text_embeddings = base_neg_t5_text_embeddings
        self.padding_mask = padding_mask
        self.fps = fps

        # Snapshot for one-step undo (populated by save_snapshot).
        self._snapshot: dict | None = None

    # ------------------------------------------------------------------ #
    # Snapshot / revert helpers (one-level undo)
    # ------------------------------------------------------------------ #

    def save_snapshot(self) -> None:
        """Save the current pipeline state so that the next generation can be reverted."""

        def _clone_cache_list(cache_list):
            if cache_list is None:
                return None
            return [x.clone() if isinstance(x, torch.Tensor) else x for x in cache_list]

        snap: dict = {}

        # Tensors that get appended – store their current temporal size so we can crop.
        snap["history_frames_T"] = int(self.history_frames.shape[2])
        snap["history_latents_T"] = int(self.history_latents.shape[2])
        snap["cam_w2c_T"] = int(self.cam_w2c.shape[1])
        snap["intrinsics_T"] = int(self.intrinsics.shape[1])

        # VAE caches – must deep-clone (list of tensors or Nones).
        snap["enc_feat_cache"] = _clone_cache_list(self.enc_feat_cache)
        snap["dec_feat_cache"] = _clone_cache_list(self.dec_feat_cache)

        # Scalar / small-tensor state.
        snap["ar_idx"] = self.ar_idx
        snap["tokens_generated"] = self.tokens_generated
        snap["last_hist_frame"] = self.last_hist_frame.clone()

        # Depth state.
        snap["buffer_depth_latest"] = self.buffer_depth_latest.clone() if self.buffer_depth_latest is not None else None
        snap["buffer_mask_latest"] = self.buffer_mask_latest.clone() if self.buffer_mask_latest is not None else None
        snap["buffer_depth_latest_frame_idx"] = self.buffer_depth_latest_frame_idx

        # Sparse3DCache – record current list lengths for trimming.
        if self.retrieval_cache is not None:
            snap["cache_len"] = len(self.retrieval_cache._world_points)
            snap["cache_rgb_keys"] = set(self.retrieval_cache._rgbs.keys())
        else:
            snap["cache_len"] = 0
            snap["cache_rgb_keys"] = set()

        # Warp video collect.
        snap["warp_video_collect_len"] = len(self.warp_video_collect)

        # Predicted-pose state.
        snap["_predicted_pose_last_w2c"] = self._predicted_pose_last_w2c.clone() if isinstance(self._predicted_pose_last_w2c, torch.Tensor) else None
        snap["_predicted_pose_updated_seed_depth"] = self._predicted_pose_updated_seed_depth.clone() if isinstance(self._predicted_pose_updated_seed_depth, torch.Tensor) else None
        snap["_predicted_pose_updated_seed_mask"] = self._predicted_pose_updated_seed_mask.clone() if isinstance(self._predicted_pose_updated_seed_mask, torch.Tensor) else None
        snap["_predicted_pose_is_first_segment"] = self._predicted_pose_is_first_segment

        self._snapshot = snap

    def revert_to_snapshot(self) -> bool:
        """Revert pipeline state to the last saved snapshot. Returns True on success."""
        snap = self._snapshot
        if snap is None:
            return False

        # Crop appendable tensors.
        self.history_frames = self.history_frames[:, :, : snap["history_frames_T"]].contiguous()
        self.history_latents = self.history_latents[:, :, : snap["history_latents_T"]].contiguous()
        self.cam_w2c = self.cam_w2c[:, : snap["cam_w2c_T"]].contiguous()
        self.intrinsics = self.intrinsics[:, : snap["intrinsics_T"]].contiguous()

        # Restore VAE caches.
        self.enc_feat_cache = snap["enc_feat_cache"]
        self.dec_feat_cache = snap["dec_feat_cache"]

        # Scalars.
        self.ar_idx = snap["ar_idx"]
        self.tokens_generated = snap["tokens_generated"]
        self.last_hist_frame = snap["last_hist_frame"]

        # Depth state.
        self.buffer_depth_latest = snap["buffer_depth_latest"]
        self.buffer_mask_latest = snap["buffer_mask_latest"]
        self.buffer_depth_latest_frame_idx = snap["buffer_depth_latest_frame_idx"]

        # Trim Sparse3DCache back to its snapshot length.
        if self.retrieval_cache is not None:
            old_len = snap["cache_len"]
            self.retrieval_cache._world_points = self.retrieval_cache._world_points[:old_len]
            self.retrieval_cache._latent_indices = self.retrieval_cache._latent_indices[:old_len]
            self.retrieval_cache._frame_ids = self.retrieval_cache._frame_ids[:old_len]
            if self.retrieval_cache._store_values:
                self.retrieval_cache._depths = self.retrieval_cache._depths[:old_len]
                self.retrieval_cache._w2cs = self.retrieval_cache._w2cs[:old_len]
                self.retrieval_cache._Ks = self.retrieval_cache._Ks[:old_len]
            kept_rgb_keys = snap["cache_rgb_keys"]
            self.retrieval_cache._rgbs = {k: v for k, v in self.retrieval_cache._rgbs.items() if k in kept_rgb_keys}

        # Warp video collect.
        self.warp_video_collect = self.warp_video_collect[: snap["warp_video_collect_len"]]

        # Predicted-pose state.
        self._predicted_pose_last_w2c = snap["_predicted_pose_last_w2c"]
        self._predicted_pose_updated_seed_depth = snap["_predicted_pose_updated_seed_depth"]
        self._predicted_pose_updated_seed_mask = snap["_predicted_pose_updated_seed_mask"]
        self._predicted_pose_is_first_segment = snap["_predicted_pose_is_first_segment"]

        # Invalidate snapshot after revert (single-level undo).
        self._snapshot = None

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return True

    def _append_cameras(self, cam_w2c_chunk: torch.Tensor, intrinsics_chunk: torch.Tensor):
        cam_chunk = cam_w2c_chunk
        intr_chunk = intrinsics_chunk
        if cam_chunk.dim() == 3:
            cam_chunk = cam_chunk.unsqueeze(1)
        if intr_chunk.dim() == 3:
            intr_chunk = intr_chunk.unsqueeze(1)
        cam_chunk = cam_chunk.to(torch.float32)
        intr_chunk = intr_chunk.to(torch.float32)
        self.cam_w2c = torch.cat([self.cam_w2c, cam_chunk], dim=1)
        self.intrinsics = torch.cat([self.intrinsics, intr_chunk], dim=1)

    def _prepare_text_embeddings(self, t5_text_embeddings, neg_t5_text_embeddings):
        pos = t5_text_embeddings if t5_text_embeddings is not None else self.base_t5_text_embeddings
        neg = neg_t5_text_embeddings if neg_t5_text_embeddings is not None else self.base_neg_t5_text_embeddings
        return pos, neg

    def autoregressive_step(
        self,
        *,
        cam_w2c_chunk,
        intrinsics_chunk,
        t5_text_embeddings=None,
        neg_t5_text_embeddings=None,
        is_last_step=False,
    ):
        self._append_cameras(cam_w2c_chunk, intrinsics_chunk)
        start_px_idx = 1 + self.ar_idx * self.model.framepack_num_new_video_frames
        end_px_idx = start_px_idx + self.model.framepack_num_new_video_frames

        total_latents_now = int(self.history_latents.shape[2])
        # Use cached counts computed in model._init_lyra2_metadata.
        num_temporal_hist = int(self.model.framepack_num_temporal_hist)
        num_spatial_hist = int(self.model.framepack_num_spatial_hist)

        temporal_selected: List[int] = self.model._select_temporal_history_indices(total_latents_now, num_temporal_hist)

        cfg = self.model.config
        use_image_spatial = bool(cfg.spatial_memory_use_image)

        # Move history latents to the model device/dtype for selection/inference.
        history_full = misc_to(self.history_latents, **self.model.tensor_kwargs)
        # Unified input preparation: reuse Lyra2Model._prepare_lyra2_inputs.
        # For now we only support pose-conditioned inference here.
        assert self.retrieval_cache is not None, "retrieval_cache must be initialized for pose mode."
        assert self.buffer_depth_latest is not None, "buffer_depth_latest must be initialized for pose mode."

        device = history_full.device
        video_hist_abs = self.history_frames[:, :, self.start_index :]
        video_all = misc_to(video_hist_abs, **self.model.tensor_kwargs)
        # Build a virtual video_indices timeline: [0 repeated prefix] + [1..end_px_idx-1]
        video_indices_t = torch.tensor(
            [0] * int(self.repeat_pixels) + list(range(1, int(end_px_idx))),
            device=device,
            dtype=torch.long,
        )

        # Dummy generation tail; `_prepare_lyra2_inputs()` overwrites it with pose conditioning.
        B, C_lat, _T_hist, H_lat, W_lat = history_full.shape
        T_new_lat = int(self.model.framepack_num_new_latent_frames)
        gen_cond_dummy = torch.zeros((B, C_lat, T_new_lat, H_lat, W_lat), device=device, dtype=history_full.dtype)

        # Buffer depth (most recent history pixel frame) for warping.
        buffer_depth = self.buffer_depth_latest.to(device=device, dtype=torch.float32)
        if buffer_depth.dim() == 3:
            buffer_depth = buffer_depth.unsqueeze(1)

        # Keep original skip behavior from the previous implementation.
        spatial_cache_skip_last_n = 0

        # Collect warped pixels for visualization if pose conditioning is enabled.
        prev_collect = bool(getattr(self.model, "_collect_return_condition_state", False))
        try:
            self.model._collect_return_condition_state = True
            latents_full, cond_latent, _mask, buffer_cond_latents = self.model._prepare_lyra2_inputs(
                history_full=history_full,
                gen_cond=gen_cond_dummy,
                spatial_cache=self.retrieval_cache,
                video=video_all,
                buffer_depth_B_1_H_W=buffer_depth,
                camera_w2c=self.cam_w2c,
                intrinsics=self.intrinsics,
                video_indices=video_indices_t,
                is_training=False,
                spatial_cache_skip_last_n=int(spatial_cache_skip_last_n),
                num_retrieval_views=self.num_retrieval_views,
            )
        finally:
            self.model._collect_return_condition_state = prev_collect
        gc.collect()
        torch.cuda.empty_cache()
        warp_pixels = getattr(self.model, "_latest_condition_state_pixels", None)
        if self.use_pose and warp_pixels is not None:
            if isinstance(warp_pixels, torch.Tensor) and warp_pixels.dim() == 5:
                if int(warp_pixels.shape[1]) > 3:
                    warp_pixels = warp_pixels[:, :3]
            self.warp_video_collect.append(warp_pixels.detach().float().cpu())
        history_window = latents_full[:, :, :-T_new_lat]

        self._restore_model_to_gpu()
        net = getattr(self.model, "net", None)
        if net is not None and hasattr(net, "reset_dit_profile_step"):
            # Each sampler step runs cond + uncond denoise, so DiT forward count is 2x.
            net.reset_dit_profile_step(total_steps=int(self.args.num_sampling_step) * 2)
        pos_text, neg_text = self._prepare_text_embeddings(t5_text_embeddings, neg_t5_text_embeddings)
        last_hist_frame_cast = misc_to(self.last_hist_frame, **self.model.tensor_kwargs)
        padding_mask_cast = misc_to(self.padding_mask, **self.model.tensor_kwargs) if self.padding_mask is not None else None
        if not self.args.use_dmd_scheduler:
            gen_chunk = self.model.inference(
                history_latents=history_window,
                cond_latent=cond_latent,
                cond_latent_mask=_mask,
                cond_latent_buffer=buffer_cond_latents,
                guidance=self.args.guidance,
                seed=int(self.args.seed + self.ar_idx),
                num_steps=self.args.num_sampling_step,
                shift=self.args.shift,
                t5_text_embeddings=pos_text,
                neg_t5_text_embeddings=neg_text,
                last_hist_frame=last_hist_frame_cast,
                fps=self.fps,
                padding_mask=padding_mask_cast,
            )
        else:
            gen_chunk = self.model.inference_dmd(
                history_latents=history_window,
                cond_latent=cond_latent,
                cond_latent_mask=_mask,
                cond_latent_buffer=buffer_cond_latents,
                guidance=self.args.guidance,
                seed=int(self.args.seed + self.ar_idx),
                num_steps=self.args.num_sampling_step,
                shift=self.args.shift,
                t5_text_embeddings=pos_text,
                neg_t5_text_embeddings=neg_text,
                last_hist_frame=last_hist_frame_cast,
                fps=self.fps,
                padding_mask=padding_mask_cast,
            )
        gen_chunk = gen_chunk[:, :, : self.model.framepack_num_new_latent_frames]
        new_generated_frames = _decode_new_latent_chunk(
            self.vae_wrap,
            self.vae_core,
            self.dec_feat_cache,
            gen_chunk,
            latent_offset=self.history_latents.shape[2],
            model=self.model,
            enable_offload=self.args.offload,
        )
        if self.args.offload:
            self.history_frames = torch.cat(
                [self.history_frames, new_generated_frames.to(self.history_frames.dtype).cpu()],
                dim=2,
            )
        else:
            self.history_frames = torch.cat(
                [self.history_frames, new_generated_frames.to(self.history_frames.dtype)],
                dim=2,
            )

        with self.vae_wrap.context:
            px_cast = new_generated_frames.to(self.vae_wrap.dtype) if not self.vae_wrap.is_amp else new_generated_frames
            feats_gen, enc_cache_out = self.model.vae_encode_with_cache(
                enc_cache=self.enc_feat_cache,
                video=px_cast,
                start_t=0,
                end_t=px_cast.shape[2],
                return_cache=True,
            )
            self.enc_feat_cache[:] = enc_cache_out
            gen_chunk_reencoded = self.model._encoder_feats_to_normalized_latents(feats_gen).to(self.history_latents.dtype)
        if self.args.offload:
            self.history_latents = torch.cat(
                [self.history_latents, gen_chunk_reencoded.to(self.history_latents.dtype).cpu()],
                dim=2,
            )
        else:
            self.history_latents = torch.cat(
                [self.history_latents, gen_chunk_reencoded.to(self.history_latents.dtype)],
                dim=2,
            )
        self.last_hist_frame = new_generated_frames[:, :, -1]
        self.tokens_generated += self.model.framepack_num_new_latent_frames

        if self.use_pose and not is_last_step:
            self._update_depth_cache(end_px_idx)
        del gen_chunk, new_generated_frames, gen_chunk_reencoded
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self.ar_idx += 1
        return {"abort": False}

    def _restore_model_to_gpu(self):
        _restore_diffusion_to_gpu(self.model, self.args.offload)

    def _update_depth_cache(self, end_px_idx):
        if self.depth_backend == "da3":
            assert self.local_da3_model is not None, "DA3 model must be initialized for pose mode."
            assert self.retrieval_cache is not None, "retrieval_cache must exist for pose mode."
            assert self.buffer_depth_latest is not None, "buffer_depth_latest must exist for pose mode."

            offload_da3 = bool(getattr(self.args, "offload_da3_diffusion", False))
            if offload_da3:
                _offload_diffusion_to_cpu(self.model, True)
            try:
                da3_out = _predict_da3_depth_window(
                    da3_model=self.local_da3_model,
                    history_frames=self.history_frames,
                    start_index=self.start_index,
                    abs_last_idx=end_px_idx - 1,
                    cam_w2c=self.cam_w2c,
                    intrinsics=self.intrinsics,
                    frame_interval=int(self.args.da3_frame_interval),
                    max_history_frames=int(self.args.da3_max_history_frames),
                    add_pose_alignment_offsets=True,
                    include_ar_chunk_last_frames=bool(getattr(self.args, "da3_include_ar_chunk_last_frames", False)),
                    ar_chunk_size_frames=int(self.model.framepack_num_new_video_frames),
                    return_raw_predicted_pose=(bool(getattr(self.args, "da3_use_predicted_pose", False)) and (self.ar_idx == 0 or bool(getattr(self.args, "da3_predicted_pose_continuation", False)))),
                )
            finally:
                if offload_da3:
                    _restore_diffusion_to_gpu(self.model, True)
            da3_frames: List[int] = da3_out["frame_indices"]
            da3_pred = da3_out["prediction"]
            assert da3_pred is not None and len(da3_frames) > 0, "DA3 depth window prediction required for cache update"

            depths_np = da3_pred.depth
            sky_np = getattr(da3_pred, "sky", None)
            H0 = int(self.history_frames.shape[-2])
            W0 = int(self.history_frames.shape[-1])

            # --- Predicted-pose alignment (if enabled) ---
            use_predicted_pose = bool(getattr(self.args, "da3_use_predicted_pose", False))
            aligned_w2c_per_local: Optional[List[torch.Tensor]] = None
            depth_scale_factor: float = 1.0
            is_first_segment = False

            if use_predicted_pose:
                pred_ext = getattr(da3_pred, "extrinsics", None)
                if pred_ext is not None:
                    pred_ext_t = torch.as_tensor(np.asarray(pred_ext), dtype=torch.float32).to(self.cam_w2c.device)
                    if pred_ext_t.dim() == 3:
                        pred_ext_t = pred_ext_t.unsqueeze(0)
                    pred_w2c_all = pred_ext_t[0]  # [N, 3or4, 4]
                    if pred_w2c_all.shape[-2] == 3:
                        pad = torch.zeros((pred_w2c_all.shape[0], 4, 4), dtype=pred_w2c_all.dtype, device=pred_w2c_all.device)
                        pad[:, :3, :4] = pred_w2c_all
                        pad[:, 3, 3] = 1.0
                        pred_w2c_all = pad

                    step_size = int(self.model.framepack_num_new_video_frames)
                    new_frame_start = end_px_idx - step_size
                    hist_local = [i for i, f in enumerate(da3_frames) if f < new_frame_start]
                    new_local = [i for i, f in enumerate(da3_frames) if f >= new_frame_start]

                    is_first_segment = len(hist_local) == 1 and da3_frames[hist_local[0]] == 0

                    if is_first_segment:
                        # Case a): history = frame 0 only.
                        # Normalise by inv(pred[0]) so frame 0 becomes identity,
                        # then scale the w2c translations by a trajectory-length
                        # ratio (so that relative camera distances match the
                        # pipeline), and finally transform into pipeline world space.
                        all_local = list(range(len(da3_frames)))

                        inv_p0 = torch.linalg.inv(pred_w2c_all[hist_local[0]])
                        aligned_all = pred_w2c_all @ inv_p0.unsqueeze(0)  # frame 0 → identity

                        # Compute depth/pose scale via trajectory-length ratio.
                        assert len(all_local) >= 2, "Expected at least two frames for trajectory-length scale calculation."
                        ref_c2w_pos = torch.stack([torch.linalg.inv(self.cam_w2c[0, da3_frames[i]].to(torch.float32))[:3, 3] for i in all_local])
                        pred_c2w_pos = torch.stack([torch.linalg.inv(pred_w2c_all[i])[:3, 3] for i in all_local])
                        ref_traj_len = (ref_c2w_pos[1:] - ref_c2w_pos[:-1]).norm(dim=-1).sum().item()
                        pred_traj_len = (pred_c2w_pos[1:] - pred_c2w_pos[:-1]).norm(dim=-1).sum().item()
                        depth_scale_factor = ref_traj_len / pred_traj_len if pred_traj_len > 1e-8 else 1.0

                        # Scale w2c translations so relative distances match pipeline.
                        # (frame 0 has t=0 after normalisation, so it stays exactly at origin.)
                        aligned_all[:, :3, 3] = aligned_all[:, :3, 3] * depth_scale_factor

                        # Transform into pipeline world space via pipeline's frame-0 w2c.
                        pipeline_w2c_0 = self.cam_w2c[0, da3_frames[hist_local[0]]].to(torch.float32)
                        aligned_all = aligned_all @ pipeline_w2c_0.unsqueeze(0)

                        aligned_w2c_per_local = [aligned_all[i].unsqueeze(0) for i in range(len(da3_frames))]
                        logger.info(f"[da3_use_predicted_pose] Case A (first segment): frame0-normalise + traj-length scale={depth_scale_factor:.6f} on {len(all_local)} frames")
                        for li in range(len(da3_frames)):
                            f_abs_i = da3_frames[li]
                            pipeline_pos = torch.linalg.inv(self.cam_w2c[0, f_abs_i].to(torch.float32))[:3, 3]
                            aligned_pos = torch.linalg.inv(aligned_all[li])[:3, 3]
                            residual = (pipeline_pos - aligned_pos).norm().item()
                            is_hist = "hist" if li in hist_local else "new "
                            logger.info(f"  [{is_hist}] frame {f_abs_i}: pipeline_pos={pipeline_pos.cpu().numpy()}, aligned_pos={aligned_pos.cpu().numpy()}, residual={residual:.6f}")
                    elif bool(getattr(self.args, "da3_predicted_pose_continuation", False)):
                        assert len(hist_local) >= 2, "Expected at least one history frame for continuation segment."
                        # Case b): continuation segment. Traj-length scale,
                        # normalise at history-last, then anchor to pipeline pose.
                        all_local = list(range(len(da3_frames)))
                        hist_last_local = hist_local[-1]

                        # Trajectory-length scale on history frames only (already
                        # DA3-aligned in prior chunks, so reliable for scale).

                        ref_c2w_pos = torch.stack([torch.linalg.inv(self.cam_w2c[0, da3_frames[i]].to(torch.float32))[:3, 3] for i in hist_local])
                        pred_c2w_pos = torch.stack([torch.linalg.inv(pred_w2c_all[i])[:3, 3] for i in hist_local])
                        ref_traj_len = (ref_c2w_pos[1:] - ref_c2w_pos[:-1]).norm(dim=-1).sum().item()
                        pred_traj_len = (pred_c2w_pos[1:] - pred_c2w_pos[:-1]).norm(dim=-1).sum().item()
                        depth_scale_factor = ref_traj_len / pred_traj_len if pred_traj_len > 1e-8 else 1.0

                        # Scale predicted w2c translations to match pipeline magnitude.
                        scaled_pred = pred_w2c_all.clone()
                        scaled_pred[:, :3, 3] = scaled_pred[:, :3, 3] * depth_scale_factor

                        # Align: map scaled pred[hist_last] → pipeline[hist_last].
                        pipeline_w2c_anchor = self.cam_w2c[0, da3_frames[hist_last_local]].to(torch.float32)
                        T_align = torch.linalg.inv(scaled_pred[hist_last_local]) @ pipeline_w2c_anchor
                        aligned_all = scaled_pred @ T_align.unsqueeze(0)

                        aligned_w2c_per_local = [aligned_all[i].unsqueeze(0) for i in range(len(da3_frames))]
                        logger.info(
                            f"[da3_use_predicted_pose] Case B (continuation): "
                            f"traj-length scale={depth_scale_factor:.6f}, "
                            f"anchor=hist_last (local={hist_last_local}, abs={da3_frames[hist_last_local]}), "
                            f"aligned {len(da3_frames)} frames using {len(hist_local)} history frames. "
                            f"hist_frame_ids={[da3_frames[i] for i in hist_local]}, "
                            f"new_frame_ids={[da3_frames[i] for i in new_local]}"
                        )
                        for li in range(len(da3_frames)):
                            f_abs_i = da3_frames[li]
                            pipeline_pos = torch.linalg.inv(self.cam_w2c[0, f_abs_i].to(torch.float32))[:3, 3]
                            aligned_pos = torch.linalg.inv(aligned_all[li])[:3, 3]
                            residual = (pipeline_pos - aligned_pos).norm().item()
                            is_hist = "hist" if li in hist_local else "new "
                            logger.info(f"  [{is_hist}] frame {f_abs_i}: pipeline_pos={pipeline_pos.cpu().numpy()}, aligned_pos={aligned_pos.cpu().numpy()}, residual={residual:.6f}")

                    # Update self.cam_w2c for new frames with aligned predicted poses.
                    if aligned_w2c_per_local is not None:
                        updated_abs_ids = []
                        for li in new_local:
                            f_abs_li = da3_frames[li]
                            self.cam_w2c[:, f_abs_li] = aligned_w2c_per_local[li].to(self.cam_w2c.dtype)
                            updated_abs_ids.append(f_abs_li)
                        logger.info(f"[da3_use_predicted_pose] Updated cam_w2c for frames: {updated_abs_ids}")
                else:
                    logger.warning("[da3_use_predicted_pose] DA3 prediction has no extrinsics; falling back to pipeline poses.")

            existing_ids = set(getattr(self.retrieval_cache, "_latent_indices", []))
            stride = max(int(self.model.config.spatial_memory_stride), 1)

            for local_i, f_abs in enumerate(da3_frames):
                depth_np = depths_np[local_i]
                depth_t = torch.from_numpy(depth_np).to(self.cam_w2c.device, dtype=torch.float32)
                if depth_t.dim() == 2:
                    depth_t = depth_t.unsqueeze(0).unsqueeze(0)
                elif depth_t.dim() == 3:
                    depth_t = depth_t.unsqueeze(0)
                depth_t = torch.nn.functional.interpolate(
                    depth_t,
                    size=(H0, W0),
                    mode="bilinear",
                    align_corners=False,
                )

                if use_predicted_pose and depth_scale_factor != 1.0:
                    depth_t = depth_t * depth_scale_factor

                # Optional sky mask for UI/visualization only; do not modify depth_t.
                valid_mask_t = None
                if sky_np is not None:
                    sky_arr = np.asarray(sky_np)[local_i].astype(np.uint8)  # [H,W], 1 for sky
                    sky_t = torch.from_numpy(sky_arr).to(self.cam_w2c.device, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
                    if int(sky_t.shape[-2]) != H0 or int(sky_t.shape[-1]) != W0:
                        sky_t = torch.nn.functional.interpolate(sky_t, size=(H0, W0), mode="nearest")
                    sky_hw = sky_t > 0.5
                    valid_mask_t = (~sky_hw).to(torch.float32)

                # Further filter with depth reliability mask (same as Gen3C forward-warp cleanup).
                depth_rel = reliable_depth_mask_range_batch(depth_t, ratio_thresh=0.15)
                # Some variants may return a tuple/list; first element is the mask.
                if isinstance(depth_rel, (tuple, list)):
                    depth_rel = depth_rel[0]
                depth_rel = depth_rel.to(dtype=torch.float32, device=depth_t.device)
                valid_mask_t = depth_rel if valid_mask_t is None else (valid_mask_t * depth_rel)

                if aligned_w2c_per_local is not None:
                    w2c_t = aligned_w2c_per_local[local_i].to(torch.float32)
                else:
                    w2c_t = self.cam_w2c[:, f_abs].to(torch.float32)
                K_t = self.intrinsics[:, f_abs].to(torch.float32)
                # Track latest buffer depth (the most recent history pixel frame).
                if int(f_abs) == int(end_px_idx - 1):
                    self.buffer_depth_latest = depth_t.detach()
                    self.buffer_depth_latest_frame_idx = int(f_abs)
                    if valid_mask_t is not None:
                        self.buffer_mask_latest = valid_mask_t.detach()
                    else:
                        self.buffer_mask_latest = (depth_t > 0).to(torch.float32).detach()

                # Add to overlap-selection cache depending on mode.
                if not getattr(self.args, "disable_cache_update", False):
                    cache_id = int(f_abs) if self.use_image_spatial else int((int(f_abs) + int(self.start_index)) // int(self.frames_per_latent))

                    # Case a): also update frame 0's depth/pose in the cache.
                    if use_predicted_pose and is_first_segment and int(f_abs) == 0 and cache_id in existing_ids:
                        self.retrieval_cache.update_by_frame_id(
                            frame_id=0,
                            depth_B_1_H_W=depth_t,
                            w2c_B_4_4=w2c_t,
                            K_B_3_3=K_t,
                        )
                        continue

                    if cache_id in existing_ids:
                        continue
                    if self.use_image_spatial and int(f_abs) != 0 and (int(f_abs) % int(stride) != 0):
                        continue
                    existing_ids.add(cache_id)
                    self.retrieval_cache.add(
                        depth_t,
                        w2c_t,
                        K_t,
                        latent_index=int(cache_id),
                        frame_id=int(f_abs),
                    )

            # Store predicted-pose update info for the persistent wrapper to return to client.
            if use_predicted_pose and aligned_w2c_per_local is not None:
                last_f_abs = int(end_px_idx - 1)
                last_local = None
                for li, f in enumerate(da3_frames):
                    if int(f) == last_f_abs:
                        last_local = li
                        break
                if last_local is not None:
                    self._predicted_pose_last_w2c = aligned_w2c_per_local[last_local].detach().cpu()
                if is_first_segment:
                    frame0_local = None
                    for li, f in enumerate(da3_frames):
                        if int(f) == 0:
                            frame0_local = li
                            break
                    if frame0_local is not None:
                        d0_np = depths_np[frame0_local]
                        d0_t = torch.from_numpy(d0_np).to(torch.float32)
                        if d0_t.dim() == 2:
                            d0_t = d0_t.unsqueeze(0).unsqueeze(0)
                        elif d0_t.dim() == 3:
                            d0_t = d0_t.unsqueeze(0)
                        d0_t = torch.nn.functional.interpolate(d0_t, size=(H0, W0), mode="bilinear", align_corners=False)
                        if depth_scale_factor != 1.0:
                            d0_t = d0_t * depth_scale_factor
                        self._predicted_pose_updated_seed_depth = d0_t.detach().cpu()
                        # Mask for seed depth
                        seed_mask = (d0_t > 0).to(torch.float32)
                        if sky_np is not None:
                            sky0 = np.asarray(sky_np)[frame0_local].astype(np.uint8)
                            sky0_t = torch.from_numpy(sky0).unsqueeze(0).unsqueeze(0).to(torch.float32)
                            if sky0_t.shape[-2:] != (H0, W0):
                                sky0_t = torch.nn.functional.interpolate(sky0_t, size=(H0, W0), mode="nearest")
                            seed_mask = seed_mask * (~(sky0_t > 0.5)).to(torch.float32)
                        self._predicted_pose_updated_seed_mask = seed_mask.detach().cpu()
                self._predicted_pose_is_first_segment = is_first_segment
            if self.args.offload:
                self.buffer_depth_latest = self.buffer_depth_latest.cpu()
                if self.buffer_mask_latest is not None:
                    self.buffer_mask_latest = self.buffer_mask_latest.cpu()
            return

        raise ValueError(f"Only depth_backend='da3' is supported in this script (VIPE backend removed).")

    def build_outputs(self, da3_gs_export_stem, log_prefix):
        video = self.history_frames[:, :, self.start_index :]
        warp_video = None
        if self.use_pose and len(self.warp_video_collect) > 0:
            warp_video = torch.cat(self.warp_video_collect, dim=2)
            first_frame = video[:, :, :1]
            warp_video = torch.cat([first_frame.cpu(), warp_video], dim=2)

        video_out = video.float().cpu()
        warp_out = warp_video.float().cpu() if warp_video is not None else None
        warp_out_merged = None

        del self.history_frames, self.history_latents, self.enc_feat_cache, self.dec_feat_cache
        del self.warp_video_collect
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return {
            "video": video_out,
            "warp_video": warp_out,
            "warp_video_merged": warp_out_merged,
            "use_pose": self.use_pose,
            "use_plucker": self.use_plucker,
        }


def run_lyra2_sample(
    model,
    data_batch,
    args,
    *,
    process_group=None,
    da3_model=None,
    show_progress=False,
    log_prefix="Start AR spatial inference",
    da3_gs_export_stem=None,
    vipe_input_dump_dir=None,
):
    """Shared Lyra2 autoregressive generation logic for a single prepared sample."""
    model._normalize_video_databatch_inplace(data_batch)

    cp_group = None
    if args.context_parallel_size > 1:
        cp_group = process_group if process_group is not None else parallel_state.get_context_parallel_group()

    init_frame = data_batch["video"][:, :, :1]
    first_depth = data_batch["depth"][:, 0]
    first_cam_w2c = data_batch["camera_w2c"][:, 0]
    first_intrinsics = data_batch["intrinsics"][:, 0]

    # Multiview input: pass full data tensors so the pipeline can seed the cache.
    multiview_data = None
    if getattr(args, "multiview_ids", None):
        multiview_data = {
            "video": data_batch["video"],
            "depth": data_batch["depth"],
            "camera_w2c": data_batch["camera_w2c"],
            "intrinsics": data_batch["intrinsics"],
        }

    pipeline = Lyra2InferencePipeline(
        model=model,
        args=args,
        first_frame=init_frame,
        first_depth=first_depth,
        first_cam_w2c=first_cam_w2c,
        first_intrinsics=first_intrinsics,
        da3_model=da3_model,
        cp_group=cp_group,
        base_t5_text_embeddings=data_batch.get("t5_text_embeddings", None),
        base_neg_t5_text_embeddings=data_batch.get("neg_t5_text_embeddings", None),
        padding_mask=data_batch.get("padding_mask", None),
        fps=data_batch.get("fps", None),
        vipe_input_dump_dir=vipe_input_dump_dir,
        vipe_input_dump_prefix=log_prefix,
        multiview_data=multiview_data,
    )

    num_frames = int(args.num_frames)
    assert (num_frames - 1) % (pipeline.tokens_per_step * pipeline.frames_per_latent) == 0, (
        f"N-1 must be divisible by tokens_per_step*frames_per_latent, but got {num_frames - 1} and {pipeline.tokens_per_step * pipeline.frames_per_latent}"
    )

    tokens_needed = (num_frames - 1 + pipeline.frames_per_latent - 1) // pipeline.frames_per_latent
    num_iters = (tokens_needed + pipeline.tokens_per_step - 1) // pipeline.tokens_per_step

    with torch.no_grad():
        logger.info(log_prefix)
        for ar_idx in tqdm.tqdm(range(num_iters)):
            start_px_idx = 1 + ar_idx * model.framepack_num_new_video_frames
            end_px_idx = start_px_idx + model.framepack_num_new_video_frames
            cam_chunk = data_batch["camera_w2c"][:, start_px_idx:end_px_idx]
            intr_chunk = data_batch["intrinsics"][:, start_px_idx:end_px_idx]

            if "t5_chunk_keys" in data_batch:
                t5_chunk_embeddings = data_batch["t5_chunk_embeddings"]
                t5_chunk_mask = data_batch["t5_chunk_mask"]
                B = int(data_batch["t5_chunk_keys"].shape[0])
                if args.ablate_same_t5:
                    pos_t5 = t5_chunk_embeddings[:, 0]
                    data_batch["t5_text_embeddings"] = pos_t5
                    data_batch["t5_text_mask"] = t5_chunk_mask[:, 0]
                else:
                    last_hist_px_abs = ar_idx * model.framepack_num_new_video_frames + 1
                    sample_frame_indices = data_batch["sample_frame_indices"]
                    t5_chunk_keys = data_batch["t5_chunk_keys"]
                    F_total = int(sample_frame_indices.shape[1])
                    idx_clamped = min(max(0, last_hist_px_abs), F_total - 1)
                    first_abs_idx_B = sample_frame_indices[:, idx_clamped].to(dtype=torch.long)
                    selected_emb_list = []
                    selected_mask_list = []
                    for b in range(B):
                        keys_b = t5_chunk_keys[b]
                        Kb = int(keys_b.numel())
                        val = int(first_abs_idx_B[b].item())
                        pos = torch.searchsorted(keys_b, torch.tensor([val], device=keys_b.device, dtype=keys_b.dtype), right=True).item()
                        sel_idx = max(0, min(int(pos) - 1, Kb - 1))
                        emb_b = t5_chunk_embeddings[b, sel_idx]
                        msk_b = t5_chunk_mask[b, sel_idx]
                        selected_emb_list.append(emb_b)
                        selected_mask_list.append(msk_b)
                    pos_t5 = torch.stack(selected_emb_list, dim=0)
                    data_batch["t5_text_embeddings"] = pos_t5
                    data_batch["t5_text_mask"] = torch.stack(selected_mask_list, dim=0)
            else:
                pos_t5 = data_batch["t5_text_embeddings"]
            neg_t5 = data_batch["neg_t5_text_embeddings"]

            step_out = pipeline.autoregressive_step(
                cam_w2c_chunk=cam_chunk,
                intrinsics_chunk=intr_chunk,
                t5_text_embeddings=pos_t5,
                neg_t5_text_embeddings=neg_t5,
                is_last_step=ar_idx == num_iters - 1,
            )

            if step_out["abort"]:
                break

    return pipeline.build_outputs(da3_gs_export_stem, log_prefix)
