"""
Inference utilities for WorldMirror pipeline.

Includes: image preprocessing, input preparation, prior loading, mask computation,
result saving, and timing utilities.
"""

import glob
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from ..models.utils.camera_utils import vector_to_camera_matrices
from ..models.utils.geometry import depth_to_world_coords_points
from .geometry import depth_edge, normals_edge
from .save_utils import (
    save_camera_params,
    save_depth_npy,
    save_depth_png,
    save_gs_ply,
    save_normal_png,
    save_points_ply,
)
from .video_utils import video_to_image_frames, video_to_image_frames_new
from .visual_util import download_file_from_url, segment_sky

_IO_WORKERS = 8

# ============================================================
# Image Preprocessing
# ============================================================


def _handle_alpha_channel(img_data):
    """Process RGBA images by blending with white background."""
    if img_data.mode == "RGBA":
        white_bg = Image.new("RGBA", img_data.size, (255, 255, 255, 255))
        img_data = Image.alpha_composite(white_bg, img_data)
    return img_data.convert("RGB")


def _calculate_resize_dims(orig_w, orig_h, max_dim, resize_strategy, patch_size=14):
    """Calculate new dimensions based on resize strategy."""
    if orig_w >= orig_h:
        new_w = max_dim
        new_h = round(orig_h * (new_w / orig_w) / patch_size) * patch_size
    else:
        new_h = max_dim
        new_w = round(orig_w * (new_h / orig_h) / patch_size) * patch_size
    return new_w, new_h


def _apply_padding(tensor_img, target_dim):
    """Apply padding to make tensor square."""
    h_pad = target_dim - tensor_img.shape[1]
    w_pad = target_dim - tensor_img.shape[2]
    if h_pad > 0 or w_pad > 0:
        pad_top, pad_bottom = h_pad // 2, h_pad - h_pad // 2
        pad_left, pad_right = w_pad // 2, w_pad - w_pad // 2
        return torch.nn.functional.pad(
            tensor_img,
            (pad_left, pad_right, pad_top, pad_bottom),
            mode="constant",
            value=1.0,
        )
    return tensor_img


def prepare_images_to_tensor(file_paths, resize_strategy="crop", target_size=518):
    """Process image files into uniform tensor batch [1, N, 3, H, W]."""
    if not file_paths:
        raise ValueError("At least 1 image is required")
    if resize_strategy not in ["crop", "pad"]:
        raise ValueError("Strategy must be 'crop' or 'pad'")

    tensor_list = []
    converter = transforms.ToTensor()

    for file_path in file_paths:
        img_data = Image.open(file_path)
        img_data = _handle_alpha_channel(img_data)
        orig_w, orig_h = img_data.size
        new_w, new_h = _calculate_resize_dims(orig_w, orig_h, target_size, resize_strategy)

        img_data = img_data.resize((new_w, new_h), Image.Resampling.BICUBIC)
        tensor_img = converter(img_data)

        if resize_strategy == "crop":
            if new_h > target_size:
                crop_start = (new_h - target_size) // 2
                tensor_img = tensor_img[:, crop_start : crop_start + target_size, :]
            if new_w > target_size:
                crop_start = (new_w - target_size) // 2
                tensor_img = tensor_img[:, :, crop_start : crop_start + target_size]
        elif resize_strategy == "pad":
            tensor_img = _apply_padding(tensor_img, target_size)

        tensor_list.append(tensor_img)

    shapes = set((t.shape[1], t.shape[2]) for t in tensor_list)
    if len(shapes) > 1:
        raise ValueError(f"Inconsistent resolutions after preprocessing: {shapes}. All input images must have the same aspect ratio.")

    batch_tensor = torch.stack(tensor_list)
    if batch_tensor.dim() == 3:
        batch_tensor = batch_tensor.unsqueeze(0)
    return batch_tensor.unsqueeze(0)


# ============================================================
# Input Preparation
# ============================================================


def prepare_input(input_path, target_size=518, fps=1, video_strategy="new", min_frames=1, max_frames=64, temp_dir=None):
    """Read images or extract video frames. Returns (img_paths, subdir_name)."""
    input_path = Path(input_path)
    video_exts = [".mp4", ".avi", ".mov", ".webm", ".gif"]

    if input_path.is_file() and input_path.suffix.lower() in video_exts:
        subdir_name = input_path.stem
        frames_dir = Path(temp_dir or "/tmp") / f"frames_{subdir_name}"
        frames_dir.mkdir(parents=True, exist_ok=True)
        min_f = max(1, min_frames)
        max_f = min(64, max_frames)
        if video_strategy == "new":
            img_paths = video_to_image_frames_new(
                str(input_path),
                str(frames_dir),
                min_frames=min_f,
                max_frames=max_f,
                fallback_fps=fps,
            )
        else:
            img_paths = video_to_image_frames(str(input_path), str(frames_dir), fps=fps)
            if len(img_paths) > max_f:
                indices = np.linspace(0, len(img_paths) - 1, max_f, dtype=int)
                img_paths = [img_paths[i] for i in indices]
        if not img_paths:
            raise RuntimeError(f"Failed to extract frames from {input_path}")
        img_paths = sorted(img_paths)
        print(f"[Input] Extracted {len(img_paths)} frames from video: {input_path}")
    elif input_path.is_dir():
        subdir_name = input_path.name
        img_paths = []
        for ext in ["*.jpeg", "*.jpg", "*.png", "*.webp"]:
            img_paths.extend(glob.glob(os.path.join(str(input_path), ext)))
        if not img_paths:
            raise FileNotFoundError(f"No images found in {input_path}")
        img_paths = sorted(img_paths)
        print(f"[Input] Loaded {len(img_paths)} images from: {input_path}")
    elif input_path.is_file() and input_path.suffix.lower() in [".jpg", ".jpeg", ".png", ".webp"]:
        subdir_name = input_path.stem
        img_paths = [str(input_path)]
        print(f"[Input] Single image input: {input_path}")
    else:
        raise ValueError(f"Invalid input path: {input_path}")

    return img_paths, subdir_name


def compute_adaptive_target_size(img_paths, max_target_size=518, patch_size=14):
    """Compute inference resolution = min(image_longest_edge, max_target_size).

    Rounds down to nearest multiple of patch_size. Avoids upsampling small images.
    """
    first_img = Image.open(img_paths[0])
    orig_w, orig_h = first_img.size
    longest_edge = max(orig_w, orig_h)
    effective = min(longest_edge, max_target_size)
    effective = (effective // patch_size) * patch_size
    return max(effective, patch_size * 2)


# ============================================================
# Prior Loading
# ============================================================


def compute_preprocessing_transform(img_paths, target_size, patch_size=14):
    """Compute the resize + center-crop transform applied by prepare_images_to_tensor.

    Returns dict with orig/new/final sizes and scale/crop parameters.
    """
    first_img = Image.open(img_paths[0])
    orig_w, orig_h = first_img.size
    new_w, new_h = _calculate_resize_dims(orig_w, orig_h, target_size, "crop", patch_size)

    crop_y = (new_h - target_size) // 2 if new_h > target_size else 0
    crop_x = (new_w - target_size) // 2 if new_w > target_size else 0

    return {
        "orig_w": orig_w,
        "orig_h": orig_h,
        "new_w": new_w,
        "new_h": new_h,
        "crop_x": crop_x,
        "crop_y": crop_y,
        "final_w": min(new_w, target_size),
        "final_h": min(new_h, target_size),
        "scale_x": new_w / orig_w,
        "scale_y": new_h / orig_h,
    }


def load_prior_camera(prior_cam_path, img_paths, preprocess_transform=None):
    """Load camera priors from JSON. Returns (extrinsics [1,N,4,4], intrinsics [1,N,3,3])."""
    with open(prior_cam_path, "r") as f:
        cam_data = json.load(f)

    stem_to_idx = {Path(p).stem: i for i, p in enumerate(img_paths)}
    N = len(img_paths)

    extrinsics = None
    extr_list = cam_data.get("extrinsics", [])
    if extr_list:
        extr_array = np.zeros((N, 4, 4), dtype=np.float32)
        matched = 0
        for entry in extr_list:
            cam_id = str(entry["camera_id"])
            idx = stem_to_idx.get(cam_id)
            if idx is None and cam_id.isdigit() and int(cam_id) < N:
                idx = int(cam_id)
            if idx is not None:
                extr_array[idx] = np.array(entry["matrix"], dtype=np.float32)
                matched += 1
        if matched == N:
            extrinsics = torch.from_numpy(extr_array).unsqueeze(0)
            print(f"[Prior] Loaded extrinsics for {matched}/{N} cameras")
        else:
            print(f"[Prior] Warning: extrinsics matched {matched}/{N}, disabling")

    intrinsics = None
    intr_list = cam_data.get("intrinsics", [])
    if intr_list:
        intr_array = np.zeros((N, 3, 3), dtype=np.float32)
        matched = 0
        for entry in intr_list:
            cam_id = str(entry["camera_id"])
            idx = stem_to_idx.get(cam_id)
            if idx is None and cam_id.isdigit() and int(cam_id) < N:
                idx = int(cam_id)
            if idx is not None:
                intr_array[idx] = np.array(entry["matrix"], dtype=np.float32)
                matched += 1
        if matched == N:
            intrinsics = torch.from_numpy(intr_array).unsqueeze(0)
            print(f"[Prior] Loaded intrinsics for {matched}/{N} cameras")
        else:
            print(f"[Prior] Warning: intrinsics matched {matched}/{N}, disabling")

    if intrinsics is not None and preprocess_transform is not None:
        sx, sy = preprocess_transform["scale_x"], preprocess_transform["scale_y"]
        cx_off, cy_off = preprocess_transform["crop_x"], preprocess_transform["crop_y"]
        intrinsics = intrinsics.clone()
        intrinsics[:, :, 0, :] *= sx
        intrinsics[:, :, 1, :] *= sy
        intrinsics[:, :, 0, 2] -= cx_off
        intrinsics[:, :, 1, 2] -= cy_off

    return extrinsics, intrinsics


def _read_depth_file(depth_path):
    """Read a single depth file (.npy, .exr, .png). Returns float32 [H, W]."""
    ext = Path(depth_path).suffix.lower()
    if ext == ".npy":
        depthmap = np.load(depth_path).astype(np.float32)
        if depthmap.ndim == 3:
            depthmap = depthmap[:, :, 0]
    elif ext == ".exr":
        depthmap = cv2.imread(depth_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH).astype(np.float32)
        if depthmap.ndim == 3:
            depthmap = depthmap[:, :, 0]
    elif ext == ".png":
        depthmap = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        if depthmap is None:
            raise FileNotFoundError(f"Cannot read depth PNG: {depth_path}")
        depthmap = depthmap.astype(np.float32)
        if depthmap.ndim == 3:
            depthmap = depthmap[:, :, 0]
        if depthmap.max() > 255:
            depthmap = depthmap / 1000.0
    else:
        raise ValueError(f"Unsupported depth format: {ext}")
    return np.nan_to_num(depthmap, nan=0, posinf=0, neginf=0)


def load_prior_depth(prior_depth_path, img_paths, target_h, target_w, preprocess_transform=None):
    """Load depth priors from a folder. Returns [1, N, H, W] or None."""
    depth_dir = Path(prior_depth_path)
    if not depth_dir.is_dir():
        return None

    depth_files = {}
    for f in sorted(depth_dir.iterdir()):
        if f.suffix.lower() in (".npy", ".exr", ".png"):
            if f.stem not in depth_files or f.suffix.lower() == ".npy":
                depth_files[f.stem] = str(f)

    N = len(img_paths)
    depth_maps = []
    for img_p in img_paths:
        img_stem = Path(img_p).stem
        dpath = depth_files.get(img_stem)
        if dpath is None:
            img_nums = "".join(filter(str.isdigit, img_stem))
            for dstem, dc in depth_files.items():
                if img_nums and img_nums == "".join(filter(str.isdigit, dstem)):
                    dpath = dc
                    break
        if dpath is None:
            return None

        depthmap = _read_depth_file(dpath)
        if preprocess_transform is not None:
            nw, nh = preprocess_transform["new_w"], preprocess_transform["new_h"]
            cx, cy = preprocess_transform["crop_x"], preprocess_transform["crop_y"]
            fw, fh = preprocess_transform["final_w"], preprocess_transform["final_h"]
            if depthmap.shape[:2] != (nh, nw):
                depthmap = cv2.resize(depthmap, (nw, nh), interpolation=cv2.INTER_LINEAR)
            depthmap = depthmap[cy : cy + fh, cx : cx + fw]
        else:
            if depthmap.shape[:2] != (target_h, target_w):
                depthmap = cv2.resize(depthmap, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        depth_maps.append(depthmap)

    depth_tensor = torch.from_numpy(np.stack(depth_maps, axis=0)).unsqueeze(0)
    print(f"[Prior] Loaded {N} depth maps from {prior_depth_path}")
    return depth_tensor


# ============================================================
# Mask Computation
# ============================================================


def create_filter_mask(
    pts3d_conf,
    depth_preds,
    normal_preds,
    sky_mask,
    confidence_percentile=10.0,
    edge_normal_threshold=5.0,
    edge_depth_threshold=0.03,
    apply_confidence_mask=True,
    apply_edge_mask=True,
    apply_sky_mask=False,
    gs_depth_preds=None,
):
    """Create filter mask based on confidence, edges, and sky segmentation.

    Returns pts_mask [S,H,W] or (pts_mask, gs_mask) tuple if gs_depth_preds given.
    """
    S, H, W = pts3d_conf.shape[:3]
    final_mask_list = []
    gs_mask_list = [] if gs_depth_preds is not None else None

    for i in range(S):
        final_mask = None
        if apply_confidence_mask:
            threshold = np.quantile(pts3d_conf[i], confidence_percentile / 100.0)
            conf_mask = pts3d_conf[i] >= threshold
            final_mask = conf_mask if final_mask is None else final_mask & conf_mask

        pre_edge_mask = final_mask

        if apply_edge_mask:
            n_edges = normals_edge(normal_preds[i], tol=edge_normal_threshold, mask=pre_edge_mask)
            d_edges = depth_edge(depth_preds[i, :, :, 0], rtol=edge_depth_threshold, mask=pre_edge_mask)
            edge_mask = ~(d_edges & n_edges)
            final_mask = edge_mask if final_mask is None else final_mask & edge_mask

            if gs_depth_preds is not None:
                gs_d_edges = depth_edge(gs_depth_preds[i, :, :, 0], rtol=edge_depth_threshold, mask=pre_edge_mask)
                gs_edge_mask = ~(gs_d_edges & n_edges)
                gs_frame_mask = gs_edge_mask if pre_edge_mask is None else pre_edge_mask & gs_edge_mask

        if apply_sky_mask:
            final_mask = sky_mask[i] if final_mask is None else final_mask & sky_mask[i]
            if gs_depth_preds is not None and apply_edge_mask:
                gs_frame_mask = gs_frame_mask & sky_mask[i]

        final_mask_list.append(final_mask)
        if gs_mask_list is not None:
            gs_mask_list.append(gs_frame_mask if apply_edge_mask else final_mask)

    def _stack(ml):
        return np.stack(ml, axis=0) if ml[0] is not None else np.ones((S, H, W), dtype=bool)

    pts_mask = _stack(final_mask_list)
    if gs_mask_list is not None:
        return pts_mask, _stack(gs_mask_list)
    return pts_mask


def _compute_sky_mask_from_model(predictions, H, W, S, threshold=0.5):
    """Build sky mask from model predictions. Returns [S,H,W] bool or None."""
    for key in ("gs_depth_mask_logits", "gs_depth_mask", "depth_mask_logits", "depth_mask"):
        if key in predictions:
            prob = predictions[key].sigmoid() if "logits" in key else predictions[key]
            dm = prob[0].detach().cpu()
            if dm.dim() == 4 and dm.shape[-1] == 1:
                dm = dm.squeeze(-1)
            if dm.dim() != 3 or dm.shape[0] != S:
                return None
            mask = (dm > threshold).numpy().astype(bool)
            if mask.shape[1] != H or mask.shape[2] != W:
                mask = np.stack([cv2.resize(mask[i].astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST) > 0 for i in range(S)], axis=0)
            return mask
    return None


def compute_sky_mask(img_paths, H, W, S, predictions=None, source="auto", model_threshold=0.5, processed_aspect_ratio=None):
    """Compute sky segmentation mask [S,H,W] (True=non-sky, False=sky)."""
    if source == "model":
        mask = _compute_sky_mask_from_model(predictions, H, W, S, model_threshold) if predictions else None
        return mask if mask is not None else np.ones((S, H, W), dtype=bool)

    skyseg_path = "skyseg.onnx"
    if not os.path.exists(skyseg_path):
        download_file_from_url(
            "https://huggingface.co/JianyuanWang/skyseg/resolve/main/skyseg.onnx",
            skyseg_path,
        )
    import onnxruntime

    session = onnxruntime.InferenceSession(skyseg_path)
    sky_list = []
    for i in range(S):
        if processed_aspect_ratio is not None:
            pil_img = Image.open(img_paths[i]).convert("RGB")
            sw, sh = pil_img.size
            if sw / sh > processed_aspect_ratio:
                cw = int(round(sh * processed_aspect_ratio))
                ch = sh
            else:
                cw = sw
                ch = int(round(sw / processed_aspect_ratio))
            left, top = (sw - cw) // 2, (sh - ch) // 2
            pil_img = pil_img.crop((left, top, left + cw, top + ch))
            frame = segment_sky(cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR), session)
        else:
            frame = segment_sky(img_paths[i], session)
        if frame.shape[:2] != (H, W):
            frame = cv2.resize(frame, (W, H))
        sky_list.append(frame)

    sky_mask = np.stack(sky_list, axis=0) > 0
    if source == "auto" and predictions is not None:
        model_mask = _compute_sky_mask_from_model(predictions, H, W, S, model_threshold)
        if model_mask is not None:
            sky_mask = sky_mask & model_mask
    return sky_mask


def compute_filter_mask(
    predictions,
    imgs,
    img_paths,
    H,
    W,
    S,
    apply_confidence_mask=False,
    apply_edge_mask=False,
    apply_sky_mask=False,
    confidence_percentile=10.0,
    edge_normal_threshold=5.0,
    edge_depth_threshold=0.03,
    sky_mask=None,
    use_gs_depth=False,
):
    """Compute unified filter mask. Returns (filter_mask, gs_filter_mask) tuple."""
    if not (apply_confidence_mask or apply_edge_mask or apply_sky_mask):
        return np.ones((S, H, W), dtype=bool), None

    if apply_sky_mask and sky_mask is None:
        sky_mask = compute_sky_mask(img_paths, H, W, S, processed_aspect_ratio=W / H)
    elif sky_mask is None:
        sky_mask = np.ones((S, H, W), dtype=bool)

    if "pts3d_conf" in predictions:
        conf_np = predictions["pts3d_conf"][0].detach().cpu().float().numpy()
    elif "depth_conf" in predictions:
        conf_np = predictions["depth_conf"][0].detach().cpu().float().numpy()
    else:
        conf_np = np.ones((S, H, W), dtype=np.float32)
    depth_np = predictions["depth"][0].detach().cpu().float().numpy()
    normal_np = predictions["normals"][0].detach().cpu().float().numpy()

    gs_depth_np = None
    if use_gs_depth and "gs_depth" in predictions:
        raw = predictions["gs_depth"][0].detach().cpu().float().numpy()
        gs_depth_np = raw if raw.ndim == 4 else raw[..., np.newaxis]

    result = create_filter_mask(
        conf_np,
        depth_np,
        normal_np,
        sky_mask,
        confidence_percentile=confidence_percentile,
        edge_normal_threshold=edge_normal_threshold,
        edge_depth_threshold=edge_depth_threshold,
        apply_confidence_mask=apply_confidence_mask,
        apply_edge_mask=apply_edge_mask,
        apply_sky_mask=apply_sky_mask,
        gs_depth_preds=gs_depth_np,
    )

    if gs_depth_np is not None:
        pts_mask, gs_mask = result
        total = pts_mask.size
        print(f"[Mask] Filter: pts kept {pts_mask.sum()}/{total}, gs kept {gs_mask.sum()}/{total}")
        return pts_mask, gs_mask

    print(f"[Mask] Filter: kept {result.sum()}/{result.size} points")
    return result, None


# ============================================================
# Save Utilities
# ============================================================


def _timed_call(func, *args, **kwargs):
    t0 = time.perf_counter()
    result = func(*args, **kwargs)
    return result, time.perf_counter() - t0


def _save_depth_parallel(depth_cpu, depth_dir, S):
    def _save_one(i):
        save_depth_png(depth_dir / f"depth_{i:04d}.png", depth_cpu[i, :, :, 0])
        save_depth_npy(depth_dir / f"depth_{i:04d}.npy", depth_cpu[i, :, :, 0])

    with ThreadPoolExecutor(max_workers=_IO_WORKERS) as pool:
        list(pool.map(_save_one, range(S)))


def _save_conf_parallel(depth_conf_cpu, conf_dir, S):
    def _save_one(i):
        conf = depth_conf_cpu[i]
        c_min, c_max = conf.min(), conf.max()
        norm = (conf - c_min) / (c_max - c_min) if c_max - c_min > 1e-8 else torch.ones_like(conf)
        Image.fromarray((norm.clamp(0, 1) * 255).to(torch.uint8).numpy(), mode="L").save(str(conf_dir / f"conf_{i + 1:04d}.png"))

    with ThreadPoolExecutor(max_workers=_IO_WORKERS) as pool:
        list(pool.map(_save_one, range(S)))


def _save_normal_parallel(normals_cpu, normal_dir, S):
    def _save_one(i):
        save_normal_png(normal_dir / f"normal_{i:04d}.png", normals_cpu[i])

    with ThreadPoolExecutor(max_workers=_IO_WORKERS) as pool:
        list(pool.map(_save_one, range(S)))


def _save_sky_mask_parallel(sky_mask, sky_mask_dir, S):
    def _save_one(i):
        Image.fromarray((~sky_mask[i]).astype(np.uint8) * 255, mode="L").save(str(sky_mask_dir / f"sky_mask_{i:04d}.png"))

    with ThreadPoolExecutor(max_workers=_IO_WORKERS) as pool:
        list(pool.map(_save_one, range(S)))


def _voxel_prune_gaussians(means, scales, quats, colors, opacities, weights, voxel_size=0.002):
    """Voxel-based merging of Gaussian splats via weighted average."""
    N = means.shape[0]
    if N == 0:
        return means, scales, quats, colors, opacities

    voxel_idx = (means / voxel_size).floor().long()
    voxel_idx = voxel_idx - voxel_idx.min(dim=0)[0]
    vmax = voxel_idx.max(dim=0)[0] + 1
    flat = voxel_idx[:, 0] * vmax[1] * vmax[2] + voxel_idx[:, 1] * vmax[2] + voxel_idx[:, 2]

    unique, inv = torch.unique(flat, return_inverse=True)
    K = len(unique)
    if K == N:
        return means, scales, quats, colors, opacities

    w = weights
    wsum = torch.zeros(K, dtype=w.dtype).scatter_add_(0, inv, w).clamp(min=1e-8)

    def _wavg(vals):
        out = torch.zeros(K, *vals.shape[1:], dtype=vals.dtype)
        for d in range(vals.shape[1]):
            out[:, d].scatter_add_(0, inv, vals[:, d] * w)
        return out / wsum.unsqueeze(-1)

    m_opa = torch.zeros(K, dtype=opacities.dtype).scatter_add_(0, inv, w * w) / wsum
    m_quats = torch.zeros(K, 4, dtype=quats.dtype)
    for d in range(4):
        m_quats[:, d].scatter_add_(0, inv, quats[:, d] * w)
    m_quats = m_quats / m_quats.norm(dim=1, keepdim=True).clamp(min=1e-8)

    print(f"[Save] Voxel prune: {N} -> {K} gaussians")
    return _wavg(means), _wavg(scales), m_quats, _wavg(colors), m_opa


def _compress_points_voxel_then_sample(pts_np, cols_np, max_points=2_000_000, voxel_size=0.005):
    """Compress point cloud: voxel merge then uniform random sampling."""
    n_in = int(pts_np.shape[0])
    if n_in == 0:
        return pts_np, cols_np

    if voxel_size > 0:
        voxel = np.floor(pts_np / voxel_size).astype(np.int64)
        voxel -= voxel.min(axis=0, keepdims=True)
        _, inv = np.unique(voxel, axis=0, return_inverse=True)
        k = int(inv.max()) + 1
        if k < n_in:
            counts = np.maximum(np.bincount(inv, minlength=k).astype(np.float32), 1.0)
            pts_np = np.stack([np.bincount(inv, weights=pts_np[:, d], minlength=k) for d in range(3)], axis=1).astype(np.float32) / counts[:, None]
            cols_np = np.clip(np.round(np.stack([np.bincount(inv, weights=cols_np[:, d].astype(np.float32), minlength=k) for d in range(3)], axis=1) / counts[:, None]), 0, 255).astype(np.uint8)

    if max_points > 0 and pts_np.shape[0] > max_points:
        idx = np.random.default_rng(42).choice(pts_np.shape[0], size=max_points, replace=False)
        pts_np, cols_np = pts_np[idx], cols_np[idx]
    return pts_np, cols_np


def _compute_points_from_depth(depth_pred, imgs, extrinsics, intrinsics, S, H, W, filter_mask=None):
    """Derive 3D point cloud from depth + camera outputs."""
    depth_pred, extrinsics, intrinsics = depth_pred.float(), extrinsics.float(), intrinsics.float()
    points_list, colors_list = [], []
    for i in range(S):
        d = depth_pred[0, i, :, :, 0]
        w2c = torch.cat([extrinsics[i][:3, :4], torch.tensor([[0, 0, 0, 1]], device=extrinsics.device)], dim=0)
        c2w = torch.linalg.inv(w2c)[:3, :4]
        pts_i, _, mask = depth_to_world_coords_points(d[None], c2w[None], intrinsics[i][None])
        img_colors = (imgs[0, i].permute(1, 2, 0) * 255).to(torch.uint8)
        valid = mask[0]
        if filter_mask is not None:
            valid = valid & torch.from_numpy(filter_mask[i]).to(valid.device)
        if valid.sum().item() > 0:
            points_list.append(pts_i[0][valid])
            colors_list.append(img_colors[valid])

    if not points_list:
        return np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.uint8)
    return (torch.cat(points_list).detach().cpu().float().numpy(), torch.cat(colors_list).detach().cpu().to(torch.uint8).numpy())


def _save_colmap_lightweight(extrinsics, intrinsics, outdir, final_w, final_h, S, image_names):
    """Save lightweight COLMAP reconstruction (cameras + images only)."""
    import pycolmap

    sparse_dir = outdir / "sparse" / "0"
    sparse_dir.mkdir(parents=True, exist_ok=True)
    scene = pycolmap.Reconstruction()
    for i in range(S):
        focal_avg = (intrinsics[i][0, 0] + intrinsics[i][1, 1]) / 2
        camera = pycolmap.Camera(
            model="SIMPLE_PINHOLE",
            width=final_w,
            height=final_h,
            params=np.array([focal_avg, intrinsics[i][0, 2], intrinsics[i][1, 2]]),
            camera_id=i + 1,
        )
        scene.add_camera(camera)
        cam_from_world = pycolmap.Rigid3d(pycolmap.Rotation3d(extrinsics[i][:3, :3]), extrinsics[i][:3, 3])
        img = pycolmap.Image(id=i + 1, name=image_names[i], camera_id=i + 1, cam_from_world=cam_from_world)
        img.registered = True
        scene.add_image(img)
    scene.write(str(sparse_dir))
    print(f"[Save] COLMAP sparse -> {sparse_dir}")


def save_results(
    predictions,
    imgs,
    img_paths,
    outdir,
    save_depth=True,
    save_normal=True,
    save_gs=True,
    save_camera=True,
    save_colmap=False,
    save_points=True,
    save_sky_mask=False,
    save_conf=False,
    log_time=False,
    max_resolution=1920,
    filter_mask=None,
    gs_filter_mask=None,
    sky_mask=None,
    compress_pts=True,
    compress_pts_max_points=2_000_000,
    compress_pts_voxel_size=0.002,
    compress_gs_max_points=5_000_000,
):
    """Save all results with parallel I/O. Returns timing dict."""
    timings = {}
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    B, S, C, H, W = imgs.shape

    ar = W / H
    max_w = max(Image.open(p).size[0] for p in img_paths)
    new_w, new_h = max_w, int(round(max_w / ar))
    longest = max(new_w, new_h)
    if longest > max_resolution:
        sf = max_resolution / longest
        new_w, new_h = int(new_w * sf), int(new_h * sf)
    new_w -= new_w % 2
    new_h -= new_h % 2
    image_names = [f"image_{i + 1:04d}.jpg" for i in range(S)]

    depth_cpu = predictions["depth"][0].detach().cpu() if "depth" in predictions else None
    conf_cpu = predictions.get("depth_conf", [None])[0]
    if conf_cpu is not None:
        conf_cpu = conf_cpu.detach().cpu()
    normals_cpu = predictions["normals"][0].detach().cpu() if "normals" in predictions else None

    futures = {}
    executor = ThreadPoolExecutor(max_workers=_IO_WORKERS)

    if save_depth and depth_cpu is not None:
        d_dir = outdir / "depth"
        d_dir.mkdir(exist_ok=True)
        futures["save_depth"] = executor.submit(_timed_call, _save_depth_parallel, depth_cpu, d_dir, S)

    if save_conf and conf_cpu is not None:
        c_dir = outdir / "depth_conf"
        c_dir.mkdir(exist_ok=True)
        futures["save_conf"] = executor.submit(_timed_call, _save_conf_parallel, conf_cpu, c_dir, S)

    if save_normal and normals_cpu is not None:
        n_dir = outdir / "normal"
        n_dir.mkdir(exist_ok=True)
        futures["save_normal"] = executor.submit(_timed_call, _save_normal_parallel, normals_cpu, n_dir, S)

    if save_sky_mask and sky_mask is not None:
        sm_dir = outdir / "sky_mask"
        sm_dir.mkdir(exist_ok=True)
        futures["save_sky_mask"] = executor.submit(_timed_call, _save_sky_mask_parallel, sky_mask, sm_dir, S)

    if save_gs and "splats" in predictions:
        sp = predictions["splats"]
        means = sp["means"][0].reshape(-1, 3).detach().cpu()
        scales = sp["scales"][0].reshape(-1, 3).detach().cpu()
        quats = sp["quats"][0].reshape(-1, 4).detach().cpu()
        colors = (sp["sh"][0] if "sh" in sp else sp["colors"][0]).reshape(-1, 3).detach().cpu()
        opacities = sp["opacities"][0].reshape(-1).detach().cpu()
        weights = sp["weights"][0].reshape(-1).detach().cpu() if "weights" in sp else torch.ones_like(opacities)

        keep = None
        if gs_filter_mask is not None:
            keep = torch.from_numpy(gs_filter_mask.reshape(-1)).bool()
        elif filter_mask is not None:
            keep = torch.from_numpy(filter_mask.reshape(-1)).bool()
        if keep is not None:
            means, scales, quats = means[keep], scales[keep], quats[keep]
            colors, opacities, weights = colors[keep], opacities[keep], weights[keep]

        means, scales, quats, colors, opacities = _voxel_prune_gaussians(means, scales, quats, colors, opacities, weights)
        if compress_gs_max_points > 0 and means.shape[0] > compress_gs_max_points:
            idx = torch.from_numpy(np.random.default_rng(42).choice(means.shape[0], size=compress_gs_max_points, replace=False)).long()
            means, scales, quats, colors, opacities = means[idx], scales[idx], quats[idx], colors[idx], opacities[idx]

        futures["save_gs_ply"] = executor.submit(_timed_call, save_gs_ply, outdir / "gaussians.ply", means, scales, quats, colors, opacities)

    if save_camera and "camera_poses" in predictions and "camera_intrs" in predictions:
        cam_p = predictions["camera_poses"][0].detach().cpu().float().numpy()
        cam_i = predictions["camera_intrs"][0].detach().cpu().float().numpy()
        futures["save_camera"] = executor.submit(_timed_call, save_camera_params, cam_p, cam_i, str(outdir))

    if save_points and "depth" in predictions and "camera_params" in predictions:
        e3x4, intr = vector_to_camera_matrices(predictions["camera_params"], image_hw=(H, W))
        pts_np, cols_np = _compute_points_from_depth(predictions["depth"], imgs, e3x4[0], intr[0], S, H, W, filter_mask=filter_mask)
        futures["save_points"] = executor.submit(_timed_call, _save_points_artifacts, outdir / "points.ply", pts_np, cols_np, compress_pts, compress_pts_max_points, compress_pts_voxel_size)

    if save_colmap and "camera_params" in predictions:
        e3x4, intr = vector_to_camera_matrices(predictions["camera_params"], image_hw=(new_h, new_w))
        futures["save_colmap"] = executor.submit(
            _timed_call, _save_colmap_lightweight, e3x4[0].detach().cpu().float().numpy(), intr[0].detach().cpu().float().numpy(), outdir, new_w, new_h, S, image_names
        )

    for key, future in futures.items():
        result, elapsed = future.result()
        if log_time:
            timings[key] = elapsed
            if isinstance(result, dict):
                timings.update(result)

    executor.shutdown(wait=False)
    return timings


def _save_points_artifacts(path, pts_np, cols_np, compress=False, max_points=2_000_000, voxel_size=0.005):
    timings = {}
    if compress:
        t0 = time.perf_counter()
        pts_np, cols_np = _compress_points_voxel_then_sample(pts_np, cols_np, max_points, voxel_size)
        timings["compress_points"] = time.perf_counter() - t0
    save_points_ply(path, pts_np, cols_np)
    return timings


# ============================================================
# Timing Report
# ============================================================


def print_and_save_timings(timings, outdir):
    """Print formatted timing table and save to JSON."""

    def _p(label, value, indent=0):
        print(f"{'  ' * (indent + 1)}{label:<38s} {value:>10.3f}s")

    print(f"\n{'=' * 72}\n  TIMING REPORT\n{'=' * 72}")

    print("  [Serial Stages]")
    for key, label in [("data_loading", "Data loading"), ("inference_preprocess", "Inference preprocess"), ("inference", "Model inference"), ("compute_mask", "Compute filter mask")]:
        if key in timings:
            _p(label, timings[key], 1)

    save_wall = timings.get("save_total_wall")
    save_keys = [
        ("save_depth", "Depth"),
        ("save_conf", "Depth conf"),
        ("save_normal", "Normal"),
        ("save_sky_mask", "Sky mask"),
        ("save_gs_ply", "Gaussians"),
        ("save_camera", "Camera"),
        ("save_points", "Points"),
        ("save_colmap", "COLMAP"),
    ]
    present = [(k, n) for k, n in save_keys if k in timings]
    if save_wall is not None or present:
        print("  [Save Stage | Parallel]")
        if save_wall is not None:
            _p("Save wall-clock", save_wall, 1)
        for k, name in present:
            _p(f"- {name}", timings[k], 2)

    if "case_total" in timings:
        print("  [Total]")
        _p("Case total", timings["case_total"], 1)

    if "gpu_mem_peak_per_rank_gb" in timings:
        print("  [GPU Memory]")
        for i, p in enumerate(timings["gpu_mem_peak_per_rank_gb"]):
            print(f"    Rank {i}: {p:.2f} GB")
        print(f"    Average: {timings['gpu_mem_peak_avg_gb']:.2f} GB")
    elif "gpu_mem_peak_gb" in timings:
        print("  [GPU Memory]")
        print(f"    Peak: {timings['gpu_mem_peak_gb']:.2f} GB")

    print(f"{'=' * 72}\n")

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    with open(outdir / "pipeline_timing.json", "w") as f:
        json.dump(timings, f, indent=2)
