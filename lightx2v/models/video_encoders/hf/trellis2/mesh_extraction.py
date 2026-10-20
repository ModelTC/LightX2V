# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License; see lightx2v/models/networks/trellis2/LICENSE.

"""CPU inference mesh extraction for TRELLIS.2.

This mirrors the official O-Voxel dual-grid equations while replacing only the
CUDA hashmap with a CPU lookup. Neural decoder outputs are untouched.
"""

from __future__ import annotations

from typing import Iterable

import torch

_EDGE_NEIGHBOUR_OFFSETS = torch.tensor([[[0, 0, 0], [0, 0, 1], [0, 1, 1], [0, 1, 0]], [[0, 0, 0], [1, 0, 0], [1, 0, 1], [0, 0, 1]], [[0, 0, 0], [0, 1, 0], [1, 1, 0], [1, 0, 0]]], dtype=torch.int64)
_QUAD_SPLIT_1 = torch.tensor([0, 1, 2, 0, 2, 3], dtype=torch.long)
_QUAD_SPLIT_2 = torch.tensor([0, 1, 3, 3, 1, 2], dtype=torch.long)


def _triple(value: int | Iterable[int]) -> torch.Tensor:
    if isinstance(value, int):
        value = (value, value, value)
    return torch.tensor(tuple(value), dtype=torch.int64)


@torch.no_grad()
def flexible_dual_grid_to_mesh(coords, dual_vertices, intersected_flag, split_weight, aabb, voxel_size=None, grid_size=None, train=False):
    if train:
        raise NotImplementedError("Inference mesh extraction does not implement training extraction")
    coords_cpu = coords.detach().to("cpu", dtype=torch.int64)
    dual_cpu = dual_vertices.detach().to("cpu", dtype=torch.float32)
    flags_cpu = intersected_flag.detach().to("cpu", dtype=torch.bool)
    split_cpu = None if split_weight is None else split_weight.detach().to("cpu", dtype=torch.float32)
    aabb_cpu = torch.as_tensor(aabb, dtype=torch.float32)
    if voxel_size is None:
        if grid_size is None:
            raise ValueError("voxel_size or grid_size is required")
        grid_cpu = _triple(grid_size)
        voxel_cpu = (aabb_cpu[1] - aabb_cpu[0]) / grid_cpu
    else:
        voxel_cpu = torch.as_tensor(voxel_size, dtype=torch.float32).flatten()
        if voxel_cpu.numel() == 1:
            voxel_cpu = voxel_cpu.repeat(3)
        grid_cpu = ((aabb_cpu[1] - aabb_cpu[0]) / voxel_cpu).round().to(torch.int64)
    n_vertices = coords_cpu.shape[0]
    if n_vertices == 0:
        raise ValueError("decoder produced no active dual-grid vertices")
    strides = torch.tensor([int(grid_cpu[1] * grid_cpu[2]), int(grid_cpu[2]), 1], dtype=torch.int64)
    active_keys = (coords_cpu * strides).sum(dim=1)
    (sorted_keys, order) = torch.sort(active_keys)
    edge_neighbours = coords_cpu.reshape(n_vertices, 1, 1, 3) + _EDGE_NEIGHBOUR_OFFSETS
    flags_cpu = flags_cpu.reshape(n_vertices, -1)
    if flags_cpu.shape[1] != 3:
        raise ValueError(f"unexpected intersected flag shape: {tuple(flags_cpu.shape)}")
    connected = edge_neighbours[flags_cpu]
    if connected.numel() == 0:
        raise ValueError("decoder produced no intersected dual-grid edges")
    query_keys = (connected.reshape(-1, 3) * strides).sum(dim=1).reshape(-1, 4)
    positions = torch.searchsorted(sorted_keys, query_keys)
    safe_positions = positions.clamp_max(n_vertices - 1)
    valid = (positions < n_vertices) & (sorted_keys[safe_positions] == query_keys)
    valid &= (coords_cpu[order[safe_positions]] == connected).all(dim=-1)
    valid = valid.all(dim=1)
    quad_indices = order[safe_positions][valid]
    if quad_indices.shape[0] == 0:
        raise ValueError("decoder produced no connected dual-grid quads")
    mesh_vertices = (coords_cpu.float() + dual_cpu) * voxel_cpu + aabb_cpu[0]
    attempt_0 = quad_indices[:, _QUAD_SPLIT_1]
    attempt_1 = quad_indices[:, _QUAD_SPLIT_2]
    if split_cpu is None:
        normals_00 = torch.cross(mesh_vertices[attempt_0[:, 1]] - mesh_vertices[attempt_0[:, 0]], mesh_vertices[attempt_0[:, 2]] - mesh_vertices[attempt_0[:, 0]], dim=1)
        normals_01 = torch.cross(mesh_vertices[attempt_0[:, 2]] - mesh_vertices[attempt_0[:, 1]], mesh_vertices[attempt_0[:, 3]] - mesh_vertices[attempt_0[:, 1]], dim=1)
        normals_10 = torch.cross(mesh_vertices[attempt_1[:, 1]] - mesh_vertices[attempt_1[:, 0]], mesh_vertices[attempt_1[:, 2]] - mesh_vertices[attempt_1[:, 0]], dim=1)
        normals_11 = torch.cross(mesh_vertices[attempt_1[:, 2]] - mesh_vertices[attempt_1[:, 1]], mesh_vertices[attempt_1[:, 3]] - mesh_vertices[attempt_1[:, 1]], dim=1)
        align_0 = (normals_00 * normals_01).sum(dim=1, keepdim=True).abs()
        align_1 = (normals_10 * normals_11).sum(dim=1, keepdim=True).abs()
        triangles = torch.where(align_0 > align_1, attempt_0, attempt_1)
    else:
        weights = split_cpu[quad_indices].reshape(-1, 4)
        choose_0 = weights[:, 0] * weights[:, 2] > weights[:, 1] * weights[:, 3]
        triangles = torch.where(choose_0[:, None], attempt_0, attempt_1)
    return (mesh_vertices, triangles.reshape(-1, 3).to(torch.int32))
