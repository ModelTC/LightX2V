"""
Utilities for saving images, depths, normals, point clouds, and Gaussian splat data.
tencent
"""

import json
import os
from io import BytesIO
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from plyfile import PlyData, PlyElement


def save_camera_params(extrinsics, intrinsics, target_dir):
    """
    Save camera parameters (extrinsics and intrinsics) in JSON format

    Args:
        extrinsics: numpy array, shape [N, 4, 4] - extrinsic matrices for N cameras
        intrinsics: numpy array, shape [N, 3, 3] - intrinsic matrices for N cameras
        target_dir: str - directory to save the parameters

    Returns:
        str: path to the saved file
    """
    camera_data = {"num_cameras": int(extrinsics.shape[0]), "extrinsics": [], "intrinsics": []}

    # Convert each camera's parameters to list format
    for i in range(extrinsics.shape[0]):
        camera_data["extrinsics"].append(
            {
                "camera_id": i,
                "matrix": extrinsics[i].tolist(),  # [4, 4] -> list
            }
        )
        camera_data["intrinsics"].append(
            {
                "camera_id": i,
                "matrix": intrinsics[i].tolist(),  # [3, 3] -> list
            }
        )

    # Save as JSON file
    camera_params_path = os.path.join(target_dir, "camera_params.json")
    with open(camera_params_path, "w") as f:
        json.dump(camera_data, f, indent=2)

    return camera_params_path


def save_image_png(path: Path, image_tensor: torch.Tensor) -> None:
    # image_tensor: [H, W, 3]
    img = (image_tensor.detach().cpu() * 255.0).to(torch.uint8).numpy()
    Image.fromarray(img).save(str(path))


def save_depth_png(path: Path, depth_tensor: torch.Tensor) -> None:
    # depth_tensor: [H, W]
    d = depth_tensor.detach()
    d = d - d.min()
    d = d / (d.max() + 1e-9)
    img = (d.clamp(0, 1) * 255.0).to(torch.uint8).cpu().numpy()
    Image.fromarray(img, mode="L").save(str(path))


def save_depth_npy(path: Path, depth_tensor: torch.Tensor) -> None:
    # depth_tensor: [H, W]
    # Save actual depth values in numpy format
    d = depth_tensor.detach().cpu().numpy()
    np.save(str(path), d)


def save_normal_png(path: Path, normal_hwc: torch.Tensor) -> None:
    # normal_hwc: [H, W, 3], in [-1, 1]
    n = (normal_hwc.detach().cpu() + 1.0) * 0.5
    img = (n.clamp(0, 1) * 255.0).to(torch.uint8).numpy()
    Image.fromarray(img).save(str(path))


def _build_vertex_ply_element(pts: np.ndarray, colors: np.ndarray) -> PlyElement:
    """Build a PLY vertex element from points and colors arrays.

    Args:
        pts: Point coordinates, shape [N, 3], dtype float32
        colors: RGB colors, shape [N, 3], dtype uint8

    Returns:
        PlyElement describing the vertices
    """
    vertex_dtype = [("x", "f4"), ("y", "f4"), ("z", "f4"), ("red", "u1"), ("green", "u1"), ("blue", "u1")]
    vertex_elements = np.empty(len(pts), dtype=vertex_dtype)
    vertex_elements["x"] = pts[:, 0]
    vertex_elements["y"] = pts[:, 1]
    vertex_elements["z"] = pts[:, 2]
    vertex_elements["red"] = colors[:, 0]
    vertex_elements["green"] = colors[:, 1]
    vertex_elements["blue"] = colors[:, 2]
    return PlyElement.describe(vertex_elements, "vertex")


def save_scene_ply(path: Path, points_xyz: torch.Tensor, point_colors: torch.Tensor, valid_mask: torch.Tensor = None) -> None:
    """Save point cloud to PLY format"""
    pts = points_xyz.detach().cpu().to(torch.float32).numpy().reshape(-1, 3)
    colors = point_colors.detach().cpu().to(torch.uint8).numpy().reshape(-1, 3)

    # Filter out invalid points (NaN, Inf)
    if valid_mask is None:
        valid_mask = np.isfinite(pts).all(axis=1)
    else:
        valid_mask = valid_mask.detach().cpu().numpy().reshape(-1)
    pts = pts[valid_mask]
    colors = colors[valid_mask]

    # Handle empty point cloud
    if len(pts) == 0:
        pts = np.array([[0, 0, 0]], dtype=np.float32)
        colors = np.array([[255, 255, 255]], dtype=np.uint8)

    PlyData([_build_vertex_ply_element(pts, colors)]).write(str(path))


def save_points_ply(path: Path, pts_np: np.ndarray, cols_np: np.ndarray) -> None:
    """Save point cloud to PLY format from numpy arrays"""
    PlyData([_build_vertex_ply_element(pts_np, cols_np)]).write(str(path))


def _build_gs_ply_data(means, scales, rotations, rgbs, opacities, quantile_threshold):
    """Build Gaussian splat PLY data with scale-based filtering.

    Args:
        means: Gaussian centers [N, 3]
        scales: Gaussian scales [N, 3]
        rotations: Gaussian rotations as quaternions [N, 4]
        rgbs: RGB colors [N, 3]
        opacities: Opacity values [N]
        quantile_threshold: Percentile threshold for scale filtering (e.g. 0.98 or 0.90)

    Returns:
        PlyData object ready to be written or returned
    """
    scale_threshold = torch.quantile(scales.max(dim=-1)[0], quantile_threshold, dim=0)
    filter_mask = scales.max(dim=-1)[0] <= scale_threshold

    means = means[filter_mask].reshape(-1, 3)
    scales = scales[filter_mask].reshape(-1, 3)
    rotations = rotations[filter_mask].reshape(-1, 4)
    rgbs = rgbs[filter_mask].reshape(-1, 3)
    opacities = opacities[filter_mask].reshape(-1)

    attributes = ["x", "y", "z", "nx", "ny", "nz"]
    for i in range(3):
        attributes.append(f"f_dc_{i}")
    attributes.append("opacity")
    for i in range(3):
        attributes.append(f"scale_{i}")
    for i in range(4):
        attributes.append(f"rot_{i}")

    dtype_full = [(attribute, "f4") for attribute in attributes]
    elements = np.empty(means.shape[0], dtype=dtype_full)

    attributes_data = (
        means.float().detach().cpu().numpy(),
        torch.zeros_like(means).float().detach().cpu().numpy(),
        rgbs.detach().cpu().contiguous().numpy(),
        opacities[..., None].detach().cpu().numpy(),
        scales.log().detach().cpu().numpy(),
        rotations.detach().cpu().numpy(),
    )
    attributes_data = np.concatenate(attributes_data, axis=1)
    elements[:] = list(map(tuple, attributes_data))

    return PlyData([PlyElement.describe(elements, "vertex")])


def save_gs_ply(path: Path, means: torch.Tensor, scales: torch.Tensor, rotations: torch.Tensor, rgbs: torch.Tensor, opacities: torch.Tensor) -> None:
    """
    Export Gaussian splat data to PLY format.

    Args:
        path: Output PLY file path
        means: Gaussian centers [N, 3]
        scales: Gaussian scales [N, 3]
        rotations: Gaussian rotations as quaternions [N, 4]
        rgbs: RGB colors [N, 3]
        opacities: Opacity values [N]
    """
    # Ensure float32 for quantile and numpy conversion (bf16 not supported)
    means, scales, rotations, rgbs, opacities = (t.float() for t in (means, scales, rotations, rgbs, opacities))
    plydata = _build_gs_ply_data(means, scales, rotations, rgbs, opacities, quantile_threshold=0.98)
    plydata.write(str(path))


def convert_gs_to_ply(means, scales, rotations, rgbs, opacities):
    """
    Export Gaussian splat data to PLY format.

    Args:
        means: Gaussian centers [N, 3]
        scales: Gaussian scales [N, 3]
        rotations: Gaussian rotations as quaternions [N, 4]
        rgbs: RGB colors [N, 3]
        opacities: Opacity values [N]
    """
    return _build_gs_ply_data(means, scales, rotations, rgbs, opacities, quantile_threshold=0.90)


def process_ply_to_splat(plydata, output_path):
    vert = plydata["vertex"]
    sorted_indices = np.argsort(-np.exp(vert["scale_0"] + vert["scale_1"] + vert["scale_2"]) / (1 + np.exp(-vert["opacity"])))
    buffer = BytesIO()
    for idx in sorted_indices:
        v = plydata["vertex"][idx]
        position = np.array([v["x"], v["y"], v["z"]], dtype=np.float32)
        scales = np.exp(
            np.array(
                [v["scale_0"], v["scale_1"], v["scale_2"]],
                dtype=np.float32,
            )
        )
        rot = np.array(
            [v["rot_0"], v["rot_1"], v["rot_2"], v["rot_3"]],
            dtype=np.float32,
        )
        SH_C0 = 0.28209479177387814
        color = np.array(
            [
                0.5 + SH_C0 * v["f_dc_0"],
                0.5 + SH_C0 * v["f_dc_1"],
                0.5 + SH_C0 * v["f_dc_2"],
                1 / (1 + np.exp(-v["opacity"])),
            ]
        )
        buffer.write(position.tobytes())
        buffer.write(scales.tobytes())
        buffer.write((color * 255).clip(0, 255).astype(np.uint8).tobytes())
        buffer.write(((rot / np.linalg.norm(rot)) * 128 + 128).clip(0, 255).astype(np.uint8).tobytes())
    value = buffer.getvalue()
    with open(output_path, "wb") as f:
        f.write(value)

    return output_path
