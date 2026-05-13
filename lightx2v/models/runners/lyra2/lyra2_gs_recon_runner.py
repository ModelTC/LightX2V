"""Runner for Lyra-2 Gaussian Splatting Reconstruction (video → GS scene + rendered video).

Pipeline:
  init_modules()
    └── load_model()     # Load DA3 depth/reconstruction model

  run_pipeline(input_info)
    ├── run_input_encoder()   # Probe video, collect frames for VIPE and DA3
    └── run_main()
          ├── _run_vipe()           # Camera pose estimation (VIPE)
          ├── _run_da3_recon()      # DA3 depth + Gaussian reconstruction
          └── _run_gs_render()      # GS rendering + save video

Config JSON mapping:
  model_path    → path to DA3 reconstruction checkpoint file
                  (e.g. /checkpoints/recon/model.pt)
  lyra_repo     → Lyra-2 source-code repository root; used for sys.path so that
                  VIPE (lyra_2/_src/inference/vipe/) can be imported
  no_vipe       → skip VIPE; use DA3's own pose predictions for rendering
  da3_max_frames → max frames subsampled from video for DA3 (default 128)

InputInfo (uses Lyra2GSReconInputInfo via --task lyra2_gs_recon):
  video_path        → source .mp4 video file
  save_result_path  → output directory (reconstructed_scene.ply, gs_trajectory.mp4, …)
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path
from typing import List

import cv2
import numpy as np
import torch
from loguru import logger

from lightx2v.models.runners.base_runner import BaseRunner
from lightx2v.server.metrics import monitor_cli
from lightx2v.utils.profiler import GET_RECORDER_MODE, ProfilingContext4DebugL1
from lightx2v.utils.registry_factory import RUNNER_REGISTER


def _ensure_lyra2_on_path(lyra_repo: str) -> None:
    if not os.path.isdir(lyra_repo):
        raise FileNotFoundError(f"[Lyra2GSRecon] Lyra-2 repo not found: '{lyra_repo}'. Set 'lyra_repo' in the config JSON.")
    if lyra_repo not in sys.path:
        sys.path.insert(0, lyra_repo)
        logger.info(f"[Lyra2GSRecon] Added to sys.path: {lyra_repo}")


# ---------------------------------------------------------------------------
# Video utilities (adapted from vipe_da3_gs_recon.py)
# ---------------------------------------------------------------------------


def _probe_video(video_path: str):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Failed to open video: {video_path}")
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    cap.release()
    if fps <= 1e-6:
        fps = 30.0
    return frame_count, fps


def _sample_indices(num_frames: int, stride: int, max_views: int = 0) -> List[int]:
    stride = max(int(stride), 1)
    indices = list(range(0, int(num_frames), stride))
    if max_views > 0:
        indices = indices[: int(max_views)]
    return indices


def _uniform_subsample_indices(num_frames: int, max_frames: int) -> List[int]:
    if num_frames <= 0:
        return []
    if max_frames <= 0 or num_frames <= max_frames:
        return list(range(num_frames))
    return np.floor(np.linspace(0, num_frames - 1, num=max_frames)).astype(np.int64).tolist()


def _read_video_frames_rgb(video_path: str, indices: List[int]) -> List[np.ndarray]:
    if not indices:
        return []
    wanted = set(int(i) for i in indices)
    last_needed = int(max(wanted))
    frames: List[np.ndarray] = []
    read_ids: List[int] = []

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Failed to open video: {video_path}")

    frame_idx = 0
    try:
        while True:
            ok, frame_bgr = cap.read()
            if not ok or frame_bgr is None:
                break
            if frame_idx in wanted:
                frames.append(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
                read_ids.append(frame_idx)
                if len(frames) == len(wanted):
                    break
            if frame_idx >= last_needed:
                break
            frame_idx += 1
    finally:
        cap.release()

    return frames


def _collect_video_frames(video_path: str, max_frames: int) -> tuple:
    """Read all frames (up to max_frames) and return (frames, fps)."""
    frame_count, fps = _probe_video(video_path)

    if frame_count > 0:
        total = frame_count if max_frames <= 0 else min(frame_count, max_frames)
        indices = list(range(total))
        images = _read_video_frames_rgb(video_path, indices)
        return images, indices, fps

    cap = cv2.VideoCapture(video_path)
    frames_tmp: List[np.ndarray] = []
    try:
        while True:
            ok, frame_bgr = cap.read()
            if not ok or frame_bgr is None:
                break
            frames_tmp.append(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
            if max_frames > 0 and len(frames_tmp) >= max_frames:
                break
    finally:
        cap.release()

    return frames_tmp, list(range(len(frames_tmp))), fps


def _pad_to_44(mat: np.ndarray) -> np.ndarray:
    if mat.shape[-2:] == (4, 4):
        return mat
    padded = np.zeros((*mat.shape[:-2], 4, 4), dtype=mat.dtype)
    padded[..., :3, :4] = mat[..., :3, :4]
    padded[..., 3, 3] = 1.0
    return padded


def _intrinsics_vec_to_k33(intrinsics_vec: torch.Tensor) -> torch.Tensor:
    fx, fy, cx, cy = intrinsics_vec[:, 0], intrinsics_vec[:, 1], intrinsics_vec[:, 2], intrinsics_vec[:, 3]
    t = int(intrinsics_vec.shape[0])
    k = torch.zeros((t, 3, 3), dtype=intrinsics_vec.dtype, device=intrinsics_vec.device)
    k[:, 0, 0] = fx
    k[:, 1, 1] = fy
    k[:, 0, 2] = cx
    k[:, 1, 2] = cy
    k[:, 2, 2] = 1.0
    return k


def _interpolate_w2c(w2c_keyframes: np.ndarray, key_indices: List[int], n_total: int) -> np.ndarray:
    w2c_keyframes = _pad_to_44(w2c_keyframes)
    if len(key_indices) == 1:
        return np.repeat(w2c_keyframes[:1], n_total, axis=0).astype(np.float32)

    from scipy.spatial.transform import Rotation, Slerp

    c2w = np.linalg.inv(w2c_keyframes)
    times_key = np.array(key_indices, dtype=np.float64)
    rotations = Rotation.from_matrix(c2w[:, :3, :3])
    translations = c2w[:, :3, 3].astype(np.float64)
    slerp = Slerp(times_key, rotations)
    times_all = np.arange(n_total, dtype=np.float64)
    times_clamped = np.clip(times_all, times_key[0], times_key[-1])
    rot_interp = slerp(times_clamped)
    trans_interp = np.column_stack([np.interp(times_clamped, times_key, translations[:, dim]) for dim in range(3)])
    c2w_dense = np.zeros((n_total, 4, 4), dtype=np.float64)
    c2w_dense[:, :3, :3] = rot_interp.as_matrix()
    c2w_dense[:, :3, 3] = trans_interp
    c2w_dense[:, 3, 3] = 1.0
    return np.linalg.inv(c2w_dense).astype(np.float32)


def _compute_aligned_pred_w2c(pred_extr_np: np.ndarray, input_w2c_np: np.ndarray) -> np.ndarray:
    """Align DA3-predicted extrinsics to VIPE-estimated extrinsics via Umeyama."""
    from depth_anything_3.utils.geometry import affine_inverse_np  # type: ignore
    from depth_anything_3.utils.pose_align import align_poses_umeyama  # type: ignore

    pred_44 = _pad_to_44(pred_extr_np.copy())
    inp_44 = _pad_to_44(input_w2c_np.copy())
    r, t, s = align_poses_umeyama(pred_44, inp_44)
    r_inv = r.T
    pred_c2w = affine_inverse_np(pred_44)
    aligned_c2w = np.zeros_like(pred_c2w)
    aligned_c2w[:, :3, :3] = np.einsum("ij,njk->nik", r_inv, pred_c2w[:, :3, :3])
    trans_shifted = pred_c2w[:, :3, 3] - t[None, :]
    aligned_c2w[:, :3, 3] = np.einsum("ij,nj->ni", r_inv, trans_shifted) / s
    aligned_c2w[:, 3, 3] = 1.0
    return affine_inverse_np(aligned_c2w).astype(np.float32)


def _save_video_mp4(video_path: str, frames_thwc: np.ndarray, fps: float) -> None:
    os.makedirs(os.path.dirname(video_path), exist_ok=True)
    from moviepy.video.io.ImageSequenceClip import ImageSequenceClip

    frames_uint8 = frames_thwc
    if frames_uint8.dtype != np.uint8:
        frames_uint8 = np.clip(frames_uint8, 0, 255).astype(np.uint8)
    clip = ImageSequenceClip([frame for frame in frames_uint8], fps=float(max(fps, 1.0)))
    try:
        clip.write_videofile(
            video_path,
            codec="libx264",
            audio=False,
            fps=float(max(fps, 1.0)),
            ffmpeg_params=["-crf", "18", "-preset", "slow", "-pix_fmt", "yuv420p"],
        )
    finally:
        clip.close()


def _load_gaussian_ply_to_gaussians(ply_path: str, device: torch.device):
    from depth_anything_3.specs import Gaussians  # type: ignore
    from plyfile import PlyData

    ply = PlyData.read(ply_path)
    vertices = ply["vertex"].data
    names = list(vertices.dtype.names or [])

    def _stack(prefix, count):
        return np.stack(
            [vertices[f"{prefix}{i}"].astype(np.float32, copy=False) for i in range(count)],
            axis=1,
        )

    means = np.stack([vertices[ax].astype(np.float32) for ax in "xyz"], axis=1)
    scales = np.exp(_stack("scale_", 3))
    rotations = _stack("rot_", 4)
    opacities = 1.0 / (1.0 + np.exp(-vertices["opacity"].astype(np.float32)))
    f_dc = _stack("f_dc_", 3)
    f_rest_keys = sorted([k for k in names if k.startswith("f_rest_")], key=lambda k: int(k.split("_")[-1]))
    if f_rest_keys:
        f_rest = np.stack([vertices[k].astype(np.float32) for k in f_rest_keys], axis=1)
        f_rest = f_rest.reshape(f_rest.shape[0], 3, f_rest.shape[1] // 3)
        harmonics = np.concatenate([f_dc[:, :, None], f_rest], axis=2)
    else:
        harmonics = f_dc[:, :, None]

    return Gaussians(
        means=torch.from_numpy(means).to(device).unsqueeze(0),
        scales=torch.from_numpy(scales).to(device).unsqueeze(0),
        rotations=torch.from_numpy(rotations).to(device).unsqueeze(0),
        harmonics=torch.from_numpy(harmonics).to(device).unsqueeze(0),
        opacities=torch.from_numpy(opacities).to(device).unsqueeze(0),
    )


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


@RUNNER_REGISTER("lyra2_gs_recon")
class Lyra2GSReconRunner(BaseRunner):
    """LightX2V runner for Lyra-2 GS Reconstruction (video → Gaussians + rendered video)."""

    def __init__(self, config):
        super().__init__(config)
        _ensure_lyra2_on_path(config["lyra_repo"])
        self.da3_model = None
        self._prev_cwd = None
        self._device = None

    # ------------------------------------------------------------------
    # Init / Model Loading
    # ------------------------------------------------------------------

    @ProfilingContext4DebugL1("Init modules")
    def init_modules(self):
        lyra_repo = self.config["lyra_repo"]
        model_path = self.config["model_path"]

        _ensure_lyra2_on_path(lyra_repo)

        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"[Lyra2GSRecon] DA3 checkpoint not found: '{model_path}'. Set 'model_path' to the absolute path of the DA3 .pt file.")
        logger.info(f"[Lyra2GSRecon] DA3 checkpoint: {model_path}")

        self._prev_cwd = os.getcwd()
        os.chdir(lyra_repo)
        logger.info(f"[Lyra2GSRecon] Working directory set to: {lyra_repo}")

        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        with ProfilingContext4DebugL1("Load DA3 model"):
            self.da3_model = self._load_da3_model()

        self.config.lock()

    def _load_da3_model(self):
        from lightx2v.models.input_encoders.hf.lyra2.depth_utils import load_da3_model

        da3_model = load_da3_model(
            da3_model_name=self.config.get("da3_model_name", "depth-anything/DA3NESTED-GIANT-LARGE-1.1"),
            da3_model_path_custom=self.config["model_path"],
            device=str(self._device),
        )
        da3_model.eval()
        logger.info("[Lyra2GSRecon] DA3 model loaded.")
        return da3_model

    # ------------------------------------------------------------------
    # Input Encoding
    # ------------------------------------------------------------------

    @ProfilingContext4DebugL1("Run input encoder")
    def run_input_encoder(self):
        """Probe video and read all frames."""
        cfg = self.config
        video_path = self.input_info.video_path
        if not video_path or not os.path.isfile(video_path):
            raise FileNotFoundError(f"[Lyra2GSRecon] Video not found: '{video_path}'. Pass --video_path to the launch script.")

        with ProfilingContext4DebugL1("Read video frames"):
            max_frames = cfg.get("max_frames", 0)
            images_all, indices_all, fps = _collect_video_frames(video_path, max_frames)
            if not images_all:
                raise RuntimeError(f"[Lyra2GSRecon] No frames read from video: {video_path}")
            logger.info(f"[Lyra2GSRecon] Video: {len(images_all)} frames, fps={fps:.2f}  → {video_path}")

        da3_max_frames = cfg.get("da3_max_frames", 128)
        indices_da3_rel = _uniform_subsample_indices(len(images_all), da3_max_frames)
        images_da3 = [images_all[i] for i in indices_da3_rel]
        indices_da3 = [indices_all[i] for i in indices_da3_rel]

        logger.info(f"[Lyra2GSRecon] DA3 frames: {len(images_da3)} / {len(images_all)} total")

        return {
            "images_all": images_all,
            "indices_all": indices_all,
            "images_da3": images_da3,
            "indices_da3": indices_da3,
            "indices_da3_rel": indices_da3_rel,
            "fps": fps,
        }

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def _run_vipe(self, images_all, indices_da3_rel, fps, vipe_output_path):
        """Run VIPE on all frames, return (w2c_vipe, K_vipe, w2c_da3, K_da3)."""
        frames_np = np.stack(images_all, axis=0).astype(np.float32) / 255.0
        frames_thwc = torch.from_numpy(frames_np).contiguous()

        # Lazy import – VIPE lives inside lyra_repo which is on sys.path
        from lyra_2._src.inference.vipe_da3_gs_recon import (  # type: ignore
            _import_vipe_class,
            _intrinsics_vec_to_k33,
            _vipe_default_overrides,
        )

        VIPE = _import_vipe_class()
        vipe_overrides = _vipe_default_overrides(Path(vipe_output_path))
        vipe_kwargs = {"fast_mode": not bool(self.config.get("vipe_full_mode", False))}
        logger.info("[Lyra2GSRecon] Loading VIPE …")
        vipe = VIPE(vipe_overrides, **vipe_kwargs)

        logger.info(f"[Lyra2GSRecon] Running VIPE on {len(images_all)} frames …")
        vipe_out = vipe.infer_frames(frames_thwc, fps=fps, name=Path(self.input_info.video_path).stem)

        c2w = vipe_out.extrinsics_c2w.to(dtype=torch.float32)
        w2c = torch.linalg.inv(c2w)
        k_vipe = _intrinsics_vec_to_k33(vipe_out.intrinsics.to(dtype=torch.float32))

        w2c_np_vipe = w2c.cpu().numpy().astype(np.float32)
        k_np_vipe = k_vipe.cpu().numpy().astype(np.float32)
        w2c_np_da3 = w2c_np_vipe[indices_da3_rel]
        k_np_da3 = k_np_vipe[indices_da3_rel]

        return w2c_np_vipe, k_np_vipe, w2c_np_da3, k_np_da3

    def _run_da3_recon(self, images_da3, w2c_np_da3, k_np_da3, da3_process_res, da3_process_method, output_dir):
        """Run DA3 depth + GS reconstruction. Returns (pred, da3_pred_w2c, da3_pred_k)."""
        cfg = self.config
        skip_vipe = cfg.get("no_vipe", False)
        da3_pred_w2c = None
        da3_pred_k = None

        if skip_vipe:
            # Pass 1: get DA3-predicted poses
            logger.info(f"[Lyra2GSRecon] DA3 pass 1 (pose): views={len(images_da3)}, infer_gs=False")
            pred_pose = self.da3_model.inference(
                image=images_da3,
                extrinsics=None,
                intrinsics=None,
                align_to_input_extrinsics=False,
                align_to_input_ext_scale=False,
                infer_gs=False,
                process_res=da3_process_res,
                process_res_method=da3_process_method,
                export_dir=None,
                export_format="mini_npz",
            )
            if pred_pose.extrinsics is None or pred_pose.intrinsics is None:
                raise RuntimeError("[Lyra2GSRecon] DA3 pass 1 did not return poses.")
            w2c_np_da3 = _pad_to_44(np.asarray(pred_pose.extrinsics, dtype=np.float32))
            k_np_da3 = np.asarray(pred_pose.intrinsics, dtype=np.float32)
            da3_pred_w2c = w2c_np_da3.copy()
            da3_pred_k = k_np_da3.copy()
            del pred_pose
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        logger.info(f"[Lyra2GSRecon] DA3 GS recon: views={len(images_da3)}, infer_gs=True")
        pred = self.da3_model.inference(
            image=images_da3,
            extrinsics=w2c_np_da3,
            intrinsics=k_np_da3,
            align_to_input_extrinsics=False,
            align_to_input_ext_scale=False,
            infer_gs=True,
            process_res=da3_process_res,
            process_res_method=da3_process_method,
            export_dir=None,
            export_format="mini_npz",
            use_aligned_pred_cam=True,
            gs_down_ratio=cfg.get("gs_down_ratio", 2),
            gs_scale_extra_multiplier=cfg.get("gs_scale_extra_multiplier", 1.0),
            gs_ds_feature_mode=cfg.get("gs_ds_feature_mode", True),
        )
        return pred, da3_pred_w2c, da3_pred_k

    @ProfilingContext4DebugL1("Run main")
    def run_main(self):
        """VIPE + DA3 reconstruction + GS rendering + save outputs."""
        cfg = self.config
        enc = self.inputs
        video_path = self.input_info.video_path
        video_stem = Path(video_path).stem

        output_dir = Path(getattr(self.input_info, "save_result_path", None) or cfg.get("output_path", "outputs/lyra2_gs_recon")) / video_stem
        output_dir.mkdir(parents=True, exist_ok=True)

        done_marker = output_dir / ".done"
        if done_marker.is_file() and not cfg.get("force", False):
            logger.info(f"[Lyra2GSRecon] Skipping {video_stem}: {done_marker} exists. Set 'force': true in config to re-run.")
            return {"output_dir": str(output_dir)}

        images_all = enc["images_all"]
        indices_all = enc["indices_all"]
        images_da3 = enc["images_da3"]
        indices_da3 = enc["indices_da3"]
        indices_da3_rel = enc["indices_da3_rel"]
        fps = enc["fps"]
        skip_vipe = cfg.get("no_vipe", False)

        # DA3 process resolution
        da3_process_res = cfg.get("da3_process_res", None)
        da3_process_method = cfg.get("da3_process_method", "upper_bound_resize")
        if da3_process_res is None:
            max_resolution = cfg.get("max_resolution", 0)
            if max_resolution > 0:
                da3_process_res = int(max_resolution)
                da3_process_method = "lower_bound_resize"
            else:
                h0, w0 = images_da3[0].shape[:2]
                da3_process_res = int(max(h0, w0))
                da3_process_method = "upper_bound_resize"

        # Ensure DA3 sys.path entries are set for depth_anything_3 helpers
        from lyra_2._src.inference.vipe_da3_gs_recon import _ensure_da3_on_syspath  # type: ignore

        _ensure_da3_on_syspath()

        from depth_anything_3.utils.gsply_helpers import save_gaussian_ply  # type: ignore

        with tempfile.TemporaryDirectory(prefix="lyra2_gs_recon_") as tmpdir:
            # ---- VIPE or DA3-only pose estimation ----
            w2c_np_da3 = None
            k_np_da3 = None
            w2c_np_vipe = None
            k_np_vipe = None
            da3_pred_w2c = None
            da3_pred_k = None

            if not skip_vipe:
                with ProfilingContext4DebugL1("Run VIPE"):
                    w2c_np_vipe, k_np_vipe, w2c_np_da3, k_np_da3 = self._run_vipe(images_all, indices_da3_rel, fps, Path(tmpdir) / "vipe_out")
                    np.savez(
                        output_dir / "vipe_predictions.npz",
                        w2c_vipe=w2c_np_vipe,
                        intrinsics_vipe=k_np_vipe,
                        w2c_da3=w2c_np_da3,
                        intrinsics_da3=k_np_da3,
                        indices_vipe=np.asarray(indices_all, dtype=np.int64),
                        indices_da3=np.asarray(indices_da3, dtype=np.int64),
                        fps=np.asarray([fps], dtype=np.float32),
                    )

            # ---- DA3 depth + GS reconstruction ----
            with ProfilingContext4DebugL1("Run DA3 GS recon"):
                pred, da3_pred_w2c, da3_pred_k = self._run_da3_recon(
                    images_da3,
                    w2c_np_da3,
                    k_np_da3,
                    da3_process_res,
                    da3_process_method,
                    output_dir,
                )

            # ---- Save PLY ----
            with ProfilingContext4DebugL1("Save PLY"):
                final_ply_path = output_dir / "reconstructed_scene.ply"
                depth_t = torch.from_numpy(np.asarray(pred.depth, dtype=np.float32)).float()
                prune_perc = cfg.get("gs_ply_prune_opacity_percentile", None)
                save_gaussian_ply(
                    pred.gaussians,
                    str(final_ply_path),
                    ctx_depth=depth_t.unsqueeze(-1),
                    prune_by_opacity_percentile=prune_perc,
                    prune_border_gs=not (prune_perc is not None and prune_perc > 0),
                )
                logger.info(f"[Lyra2GSRecon] Saved PLY: {final_ply_path}")

            # ---- Determine render poses ----
            if skip_vipe:
                w2c_render = _interpolate_w2c(da3_pred_w2c, indices_da3_rel, len(images_all))
                k_render = np.repeat(da3_pred_k[:1], len(images_all), axis=0).astype(np.float32)
            elif cfg.get("use_da3_render_pose", True) and pred.extrinsics is not None:
                aligned_w2c_da3 = _compute_aligned_pred_w2c(np.asarray(pred.extrinsics, dtype=np.float32), w2c_np_da3)
                w2c_render = _interpolate_w2c(aligned_w2c_da3, indices_da3_rel, len(images_all))
                k_render = k_np_vipe
            else:
                w2c_render = w2c_np_vipe
                k_render = k_np_vipe

            # Save camera data
            cameras_data = {
                "w2c_render": w2c_render,
                "indices_da3": np.asarray(indices_da3, dtype=np.int64),
                "fps": np.asarray([fps], dtype=np.float32),
                "no_vipe": np.asarray([int(skip_vipe)], dtype=np.int32),
            }
            np.savez(output_dir / "cameras.npz", **cameras_data)

            del pred
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # ---- GS rendering ----
            with ProfilingContext4DebugL1("GS rendering"):
                from depth_anything_3.model.utils.gs_renderer import run_renderer_in_chunk_w_trj_mode  # type: ignore

                gs_device = self._device
                if hasattr(self.da3_model, "model"):
                    try:
                        gs_device = next(self.da3_model.model.parameters()).device
                    except StopIteration:
                        pass

                gaussians = _load_gaussian_ply_to_gaussians(str(final_ply_path), device=gs_device)
                render_extr = torch.from_numpy(w2c_render).to(device=gs_device, dtype=gaussians.means.dtype)[None]
                render_intr = torch.from_numpy(k_render).to(device=gs_device, dtype=gaussians.means.dtype)[None]
                if render_extr.shape[-2:] == (3, 4):
                    pad = torch.tensor([0, 0, 0, 1], device=gs_device, dtype=gaussians.means.dtype).view(1, 1, 1, 4)
                    render_extr = torch.cat([render_extr, pad.expand(render_extr.shape[0], render_extr.shape[1], -1, -1)], dim=-2)

                render_h, render_w = images_all[0].shape[:2]
                render_fps_cfg = cfg.get("render_fps", None)
                render_fps = float(render_fps_cfg) if render_fps_cfg is not None else float(max(1, round(fps)))
                logger.info(f"[Lyra2GSRecon] Rendering {render_extr.shape[1]} frames at {render_h}x{render_w} fps={render_fps:.2f}")
                color, depth = run_renderer_in_chunk_w_trj_mode(
                    gaussians=gaussians,
                    extrinsics=render_extr,
                    intrinsics=render_intr,
                    image_shape=(render_h, render_w),
                    chunk_size=int(cfg.get("render_chunk_size", 1)),
                    trj_mode="original",
                    use_sh=True,
                    color_mode="RGB+ED",
                    enable_tqdm=True,
                )
                frames_render = color[0].clamp(0.0, 1.0).mul(255.0).byte().permute(0, 2, 3, 1).cpu().numpy()

            with ProfilingContext4DebugL1("Save results"):
                video_out = output_dir / "gs_trajectory.mp4"
                _save_video_mp4(str(video_out), frames_render, fps=render_fps)
                logger.info(f"[Lyra2GSRecon] Saved GS video: {video_out}")

                del gaussians, render_extr, render_intr, color, depth, frames_render
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        done_marker.write_text("done\n")
        logger.info(f"[Lyra2GSRecon] Done. Output dir: {output_dir}")
        return {"output_dir": str(output_dir)}

    # ------------------------------------------------------------------
    # Pipeline orchestration
    # ------------------------------------------------------------------

    @ProfilingContext4DebugL1(
        "RUN pipeline",
        recorder_mode=GET_RECORDER_MODE(),
        metrics_func=monitor_cli.lightx2v_worker_request_duration,
        metrics_labels=["Lyra2GSReconRunner"],
    )
    def run_pipeline(self, input_info):
        """Run the full GS reconstruction pipeline for one video."""
        if GET_RECORDER_MODE():
            monitor_cli.lightx2v_worker_request_count.inc()
        self.input_info = input_info

        logger.info(f"[Lyra2GSRecon] Starting  video={input_info.video_path}")
        logger.info(f"[Lyra2GSRecon] Output → {getattr(input_info, 'save_result_path', None) or self.config.get('output_path')}")

        self.inputs = self.run_input_encoder()
        result = self.run_main()
        self.end_run()

        if GET_RECORDER_MODE():
            monitor_cli.lightx2v_worker_request_success.inc()
        logger.info(f"[Lyra2GSRecon] Done. Results at: {result['output_dir']}")
        return result

    def end_run(self):
        """Restore working directory."""
        self.inputs = None
        self.input_info = None
        if self._prev_cwd and os.path.isdir(self._prev_cwd):
            os.chdir(self._prev_cwd)
            logger.info(f"[Lyra2GSRecon] Restored working directory to: {self._prev_cwd}")
