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


def _fill_mesh_holes_torch(vertices, faces, max_hole_perimeter):
    """CPU/Torch equivalent of CuMesh's center-fan boundary-loop fill."""
    device = vertices.device
    verts = vertices.detach().to(device="cpu", dtype=torch.float32)
    tris = faces.detach().to(device="cpu", dtype=torch.int64)
    if tris.numel() == 0:
        return vertices, faces

    edge_rows = torch.cat((tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]]), dim=0)
    edge_rows = torch.sort(edge_rows, dim=1).values
    # Encode each sorted undirected edge as an exact int64 key. For nonnegative
    # vertex indices this preserves lexicographic ordering while avoiding the
    # expensive two-dimensional ``unique(dim=0)`` path on large CPU meshes.
    vertex_count = verts.shape[0]
    if vertex_count > 3037000499:
        raise ValueError("Mesh vertex count exceeds int64 edge-key capacity")
    edge_keys, counts = torch.unique(
        edge_rows[:, 0] * vertex_count + edge_rows[:, 1], return_counts=True
    )
    edges = torch.stack((edge_keys // vertex_count, edge_keys % vertex_count), dim=1)
    boundary_edges = edges[counts == 1]
    if boundary_edges.numel() == 0:
        return vertices, faces

    # CuMesh only fills closed manifold boundary loops. A boundary vertex with
    # anything other than two boundary neighbors is not part of such a loop.
    adjacency = [[] for _ in range(verts.shape[0])]
    for edge_id, (v0, v1) in enumerate(boundary_edges.tolist()):
        adjacency[v0].append(edge_id)
        adjacency[v1].append(edge_id)

    eligible = {i for i, edge in enumerate(boundary_edges.tolist())
                if len(adjacency[edge[0]]) == 2 and len(adjacency[edge[1]]) == 2}
    loops = []
    while eligible:
        first_edge = min(eligible)
        start, current = boundary_edges[first_edge].tolist()
        loop_edges = [first_edge]
        loop_vertices = [start, current]
        previous_edge = first_edge
        while current != start:
            choices = [edge_id for edge_id in adjacency[current]
                       if edge_id != previous_edge and edge_id in eligible]
            if len(choices) != 1:
                break
            previous_edge = choices[0]
            a, b = boundary_edges[previous_edge].tolist()
            current = b if a == current else a
            if current != start:
                loop_vertices.append(current)
            loop_edges.append(previous_edge)
        if current == start and len(loop_edges) == len(loop_vertices):
            perimeter = torch.linalg.vector_norm(
                verts[boundary_edges[loop_edges, 0]] - verts[boundary_edges[loop_edges, 1]],
                dim=1,
            ).sum().item()
            if perimeter < max_hole_perimeter:
                loops.append(loop_edges)
        eligible.difference_update(loop_edges)

    if not loops:
        return vertices, faces

    new_vertices = []
    new_faces = []
    vertex_count = verts.shape[0]
    for loop_id, loop_edges in enumerate(loops):
        loop = boundary_edges[loop_edges]
        midpoint_positions = (verts[loop[:, 0]] + verts[loop[:, 1]]) * 0.5
        new_vertices.append(midpoint_positions.mean(dim=0))
        center_id = vertex_count + loop_id
        new_faces.append(torch.cat((loop, torch.full((len(loop), 1), center_id, dtype=torch.int64)), dim=1))

    filled_vertices = torch.cat((verts, torch.stack(new_vertices)), dim=0).to(device=device)
    filled_faces = torch.cat((tris, *new_faces), dim=0).to(device=device, dtype=faces.dtype)
    return filled_vertices, filled_faces


def fill_mesh_holes(vertices, faces, max_hole_perimeter=3e-2, backend="cumesh"):
    """Fill small boundary loops with CuMesh, the MLU kernel, or MetaX-safe Torch."""
    if backend == "torch":
        return _fill_mesh_holes_torch(vertices, faces, max_hole_perimeter)
    if backend != "cumesh":
        raise ValueError(f"Unsupported mesh hole filling backend: {backend}")

    from lightx2v_platform.base.global_var import AI_DEVICE
    if torch.device(AI_DEVICE).type == "mlu":
        from lightx2v_platform.ops.mesh.cambricon_mlu import fill_mesh_holes as mlu_fill_mesh_holes
        device = vertices.device
        result = mlu_fill_mesh_holes(vertices.to(AI_DEVICE), faces.to(AI_DEVICE), max_hole_perimeter)
        return tuple(value.to(device) for value in result)

    import cumesh
    device = vertices.device
    mesh = cumesh.CuMesh()
    mesh.init(vertices.float().cuda(), faces.int().cuda())
    mesh.get_edges()
    mesh.get_boundary_info()
    if mesh.num_boundaries == 0:
        return vertices, faces
    mesh.get_vertex_edge_adjacency()
    mesh.get_vertex_boundary_adjacency()
    mesh.get_manifold_boundary_adjacency()
    mesh.read_manifold_boundary_adjacency()
    mesh.get_boundary_connected_components()
    mesh.get_boundary_loops()
    if mesh.num_boundary_loops == 0:
        return vertices, faces
    mesh.fill_holes(max_hole_perimeter=max_hole_perimeter)
    vertices, faces = mesh.read()
    return vertices.to(device), faces.to(device)


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
