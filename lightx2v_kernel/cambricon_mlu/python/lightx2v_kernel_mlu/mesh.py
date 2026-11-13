"""CuMesh-compatible boundary topology and center-fan hole filling on MLU.

CUDA source: JeffreyXiang/CuMesh@12289e1062f0603f2f0d0771b02e1395d247f26f.
Custom kernel names/algorithms are preserved; torch MLU handles CUB primitives.
"""

import math

import torch
import torch_mlu  # Registers the MLU device and loads its extension libraries.

from . import _C


def _run(op, *buffers, n, arg=0, threshold=0.0):
    _C._run(op, list(buffers), n, arg, threshold)


def _offsets(counts):
    return torch.cat((counts.new_zeros(1), counts.cumsum(0, dtype=torch.int32)))


def _indices(mask):
    return mask.nonzero().flatten().to(torch.int32)


def _adjacency(edges, vertex_count, boundaries=None):
    counts = torch.zeros(vertex_count, dtype=torch.int32, device=edges.device)
    n = len(edges) if boundaries is None else len(boundaries)
    if boundaries is None:
        _run(1, edges, counts, n=n)
    else:
        _run(4, edges, boundaries, counts, n=n)
    offsets = _offsets(counts)
    adjacency = counts.new_empty(2 * n)
    counts.zero_()
    if boundaries is None:
        _run(2, edges, adjacency, offsets, counts, n=n)
    else:
        _run(5, edges, boundaries, adjacency, offsets, counts, n=n)
    return adjacency, offsets, counts


def _components(adjacency, count):
    ids = torch.empty(count, device=adjacency.device, dtype=torch.int32)
    _run(14, ids, n=count, arg=1)
    if len(adjacency):
        converged = ids.new_ones(1)
        # Each successful hook strictly lowers a root. This bound detects a
        # broken implementation instead of silently returning incomplete labels.
        for _ in range(count + 1):
            converged.fill_(1)
            _run(8, adjacency, ids, converged, n=len(adjacency))
            _run(9, ids, n=count)
            if converged.item():
                break
        else:
            raise RuntimeError("MLU boundary connected components did not converge")
    _, inverse = torch.unique(ids, sorted=True, return_inverse=True)
    return inverse.to(torch.int32)


@torch.no_grad()
def fill_mesh_holes(vertices, faces, max_hole_perimeter=3e-2, *, return_debug=False):
    """Fill closed manifold boundary components with perimeter < threshold.

    Inputs: contiguous-convertible float vertices [V,3], integer faces [F,3],
    on the same MLU. Outputs preserve all original vertices/faces and append
    a mean-of-edge-midpoints vertex and one triangle per edge of each small loop.
    """
    if vertices.device.type != "mlu" or faces.device != vertices.device:
        raise ValueError("vertices and faces must be on the same MLU")
    if vertices.ndim != 2 or vertices.shape[1] != 3 or faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError("Expected vertices [V,3] and faces [F,3]")
    if faces.dtype not in (torch.int32, torch.int64):
        raise ValueError("Face indices must be int32 or int64")
    if not math.isfinite(max_hole_perimeter) or max_hole_perimeter < 0:
        raise ValueError("max_hole_perimeter must be finite and nonnegative")
    if len(vertices) >= 2**31 or len(faces) >= (2**31 // 3):
        raise ValueError("Mesh exceeds the int32 indexing range")
    if vertices.numel() and not torch.isfinite(vertices).all().item():
        raise ValueError("Non-finite mesh vertices")
    if faces.numel() and (faces.min().item() < 0 or faces.max().item() >= len(vertices)):
        raise ValueError("Face index out of range")
    if faces.numel() and ((faces[:, 0] == faces[:, 1]) | (faces[:, 1] == faces[:, 2]) | (faces[:, 2] == faces[:, 0])).any().item():
        raise ValueError("Faces with repeated vertex indices are unsupported")
    vertices, faces = vertices.float().contiguous(), faces.int().contiguous()
    debug = {}

    def finish(v=vertices, f=faces):
        return (v, f, debug) if return_debug else (v, f)

    if not len(faces):
        return finish()
    expanded = torch.empty(3 * len(faces), dtype=torch.int64, device=faces.device)
    _run(0, faces, expanded, n=len(faces))
    edges, incidence = torch.unique(expanded, sorted=True, return_counts=True)
    incidence = incidence.int()
    boundaries = _indices(incidence == 1)
    debug.update(edges=edges, edge2face_cnt=incidence, boundaries=boundaries)
    if not len(boundaries):
        return finish()
    vertex_boundary = torch.zeros(len(vertices), device=faces.device, dtype=torch.uint8)
    _run(3, edges, boundaries, incidence, vertex_boundary, n=len(boundaries))
    vedges, vedge_offsets, _ = _adjacency(edges, len(vertices))
    vbound, vbound_offsets, _ = _adjacency(edges, len(vertices), boundaries)
    manifold = torch.empty_like(vertex_boundary)
    _run(6, vedges, vedge_offsets, incidence, manifold, n=len(vertices))
    manifold_vertices = _indices((manifold != 0) & (vertex_boundary != 0))
    pairs = boundaries.new_empty((len(manifold_vertices), 2))
    _run(7, manifold_vertices, vbound, vbound_offsets, pairs, n=len(pairs))
    components = _components(pairs, len(boundaries))
    component_count = int(components.max().item()) + 1
    is_loop = components.new_empty(component_count)
    _run(15, is_loop, n=component_count, arg=1)
    _run(10, edges, boundaries, components, vbound, vbound_offsets, is_loop, n=len(boundaries))
    order = torch.argsort(components, stable=True)
    ordered_components = components[order]
    keep = is_loop[ordered_components.long()] != 0
    loop_edges = boundaries[order][keep].contiguous()
    loop_components = ordered_components[keep].contiguous()
    debug.update(vert_is_boundary=vertex_boundary, vert_is_manifold=manifold,
                 manifold_bound_adj=pairs, bound_conn_comp_ids=components, is_loop=is_loop)
    if not len(loop_edges):
        return finish()
    _, counts = torch.unique_consecutive(loop_components, return_counts=True)
    offsets = _offsets(counts.int())
    lengths = vertices.new_empty(len(loop_edges))
    _run(11, vertices, edges, loop_edges, lengths, n=len(loop_edges))
    perimeters = vertices.new_empty(len(counts))
    _run(24, lengths, offsets, perimeters, n=len(counts), arg=1)
    mask = torch.empty(len(counts), device=faces.device, dtype=torch.uint8)
    _run(20, perimeters, mask, n=len(counts), threshold=max_hole_perimeter)
    debug.update(loop_boundaries=loop_edges, loop_boundaries_offset=offsets,
                 loop_boundary_lengths=lengths, loop_perimeters=perimeters, loop_mask=mask)
    if not mask.any().item():
        return finish()
    loop_ids = boundaries.new_zeros(len(loop_edges))
    _run(19, offsets[1:-1].contiguous(), loop_ids, n=max(0, len(counts) - 1))
    loop_ids = loop_ids.cumsum(0, dtype=torch.int32)
    chosen = mask[loop_ids.long()] != 0
    selected_edges = loop_edges[chosen].contiguous()
    selected_counts = counts[mask.bool()].int().contiguous()
    selected_offsets = _offsets(selected_counts)
    selected_ids = boundaries.new_zeros(len(selected_edges))
    _run(19, selected_offsets[1:-1].contiguous(), selected_ids, n=max(0, len(selected_counts) - 1))
    selected_ids = selected_ids.cumsum(0, dtype=torch.int32)
    midpoints = vertices.new_empty((len(selected_edges), 3))
    _run(12, vertices, edges, selected_edges, midpoints, n=len(selected_edges))
    centers = vertices.new_empty((len(selected_counts), 3))
    _run(24, midpoints, selected_offsets, centers, n=len(selected_counts), arg=3)
    _run(21, centers, selected_counts, n=len(selected_counts))
    new_vertices = vertices.new_empty(centers.shape)
    _run(23, centers, new_vertices, n=len(centers))
    new_faces = faces.new_empty((len(selected_edges), 3))
    _run(13, edges, selected_edges, selected_ids, new_faces, n=len(selected_edges), arg=len(vertices))
    debug.update(selected_edges=selected_edges, selected_offsets=selected_offsets,
                 selected_loop_ids=selected_ids, midpoints=midpoints, centers=centers,
                 new_vertices=new_vertices, new_faces=new_faces)
    return finish(torch.cat((vertices, new_vertices)), torch.cat((faces, new_faces)))
