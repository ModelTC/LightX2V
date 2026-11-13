"""Validate every ported kernel and mesh filling; export identical CUDA inputs."""

import argparse
import json
from pathlib import Path

import numpy as np

from source_reference import ROOT, build_reference, check_arrays, run_cpu, verify_cuda_exports

OUTPUTS = [[1], [1], [1, 3], [3], [2], [2, 4], [3], [3], [1, 2], [0],
           [5], [3], [3], [3], [0], [0], [2], [2], [1], [1], [1], [0], [1], [1], [2]]


def reference_fill(vertices, faces, threshold):
    """Independent NumPy topology oracle following CuMesh's manifold rules."""
    if not len(faces):
        return vertices.copy(), faces.copy(), np.empty(0, dtype=np.int32)
    pairs = np.sort(faces[:, [[0, 1], [1, 2], [2, 0]]].reshape(-1, 2), axis=1)
    encoded = (pairs[:, 0].astype(np.int64) << 32) | pairs[:, 1]
    edges, incidence = np.unique(encoded, return_counts=True)
    endpoints = np.stack((edges >> 32, edges & 0xFFFFFFFF), axis=1).astype(np.int32)
    boundaries = np.flatnonzero(incidence == 1)
    vedges, vbound = [[] for _ in vertices], [[] for _ in vertices]
    for e, (a, b) in enumerate(endpoints):
        vedges[a].append(e)
        vedges[b].append(e)
    for i, e in enumerate(boundaries):
        for vertex in endpoints[e]:
            vbound[vertex].append(i)
    parent = list(range(len(boundaries)))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for vertex in range(len(vertices)):
        if len(vbound[vertex]) == 2 and all(incidence[e] <= 2 for e in vedges[vertex]):
            a, b = (root(i) for i in vbound[vertex])
            parent[max(a, b)] = min(a, b)
    labels = np.asarray([root(i) for i in range(len(boundaries))], dtype=np.int32)
    _, labels = np.unique(labels, return_inverse=True)
    centers, triangles = [], []
    for component in np.unique(labels):
        ids = np.flatnonzero(labels == component)
        closed = all(any(j != i and labels[j] == component for j in vbound[v])
                     for i in ids for v in endpoints[boundaries[i]])
        if not closed:
            continue
        selected = endpoints[boundaries[ids]]
        difference = vertices[selected[:, 1]] - vertices[selected[:, 0]]
        lengths = np.sqrt(np.sum(difference * difference, axis=1))
        if np.sum(lengths, dtype=np.float32) >= np.float32(threshold):
            continue
        center = ((vertices[selected[:, 0]] + vertices[selected[:, 1]]) * np.float32(.5)).mean(axis=0)
        new_id = len(vertices) + len(centers)
        centers.append(center)
        triangles.extend((int(b), int(a), new_id) for a, b in selected)
    if not centers:
        return vertices.copy(), faces.copy(), labels.astype(np.int32)
    return (np.concatenate((vertices, np.asarray(centers, dtype=np.float32))),
            np.concatenate((faces, np.asarray(triangles, dtype=np.int32))), labels.astype(np.int32))


def fixtures():
    triangle = np.array([[0, 0, 0], [.001, 0, 0], [0, .001, 0]], dtype=np.float32)
    face = np.array([[0, 1, 2]], dtype=np.int32)
    yield "small_triangle", triangle, face, .03
    # Real TRELLIS meshes exceed 65535 vertices. Preserve the entire uint64
    # edge key rather than accidentally passing only small-index fixtures.
    high_vertices = np.zeros((70003, 3), dtype=np.float32)
    high_vertices[-3:] = triangle
    yield "vertex_indices_above_65535", high_vertices, face + 70000, .03
    yield "large_triangle", triangle * 100, face, .03
    yield "empty", triangle[:0], face[:0], .03
    tetra = np.concatenate((triangle, np.array([[0, 0, .001]], dtype=np.float32)))
    tetra_faces = np.array([[0, 2, 1], [0, 1, 3], [1, 2, 3], [2, 0, 3]], dtype=np.int32)
    yield "closed_tetrahedron", tetra, tetra_faces, .03
    yield "tetrahedron_missing_face", tetra, tetra_faces[:-1], .03
    yield "two_sizes", np.concatenate((triangle, triangle * 100 + .2)), np.concatenate((face, face + 3)), .03
    nonmanifold = np.array([[0, 0, 0], [.001, 0, 0], [0, .001, 0], [0, 0, .001], [0, -.001, 0]], dtype=np.float32)
    yield "nonmanifold_edge", nonmanifold, np.array([[0, 1, 2], [1, 0, 3], [0, 1, 4]], dtype=np.int32), .03
    square = np.array([[0, 0, 0], [.0075, 0, 0], [.0075, .0075, 0], [0, .0075, 0]], dtype=np.float32)
    sf = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
    for threshold in [np.nextafter(np.float32(.03), np.float32(0)), np.float32(.03), np.nextafter(np.float32(.03), np.float32(1))]:
        yield f"strict_threshold_{float(threshold)}", square, sf, float(threshold)
    # Shared endpoints stress atomics; many loops exercise > taskDim work items.
    for count in [33, 257]:
        vertices = np.concatenate([triangle + np.array([i * .01, 0, 0], dtype=np.float32) for i in range(count)])
        faces = np.concatenate([face + i * 3 for i in range(count)])
        yield f"many_loops_{count}", vertices, faces, .03
    n = 20
    x, y = np.meshgrid(np.arange(n), np.arange(n))
    grid = np.stack((x.ravel(), y.ravel(), np.zeros(n*n)), axis=1).astype(np.float32) * .0001
    gf = np.array([[a, a+1, a+n] for a in range(n*n) if a % n < n-1 and a // n < n-1] +
                  [[a+1, a+n+1, a+n] for a in range(n*n) if a % n < n-1 and a // n < n-1], dtype=np.int32)
    yield "shared_edge_grid", grid, gf, .03


def validate(args):
    import torch
    import torch_mlu
    import lightx2v_kernel_mlu.mesh as mesh

    assert torch.mlu.is_available() and torch.mlu.device_count() == 1
    reference = build_reference(args.cumesh_source, args.output / "reference-build")
    original = mesh._run
    mapping = json.loads((ROOT / "kernel-map.json").read_text())
    names = {r["op"]: r["cuda_kernel"] for r in mapping["kernels"]}
    names[24] = "CUB_segmented_sum_replacement"
    cases, errors, tested = [], {}, set()
    export = args.output / "export"
    export.mkdir(parents=True)

    def checked(op, *buffers, n, arg=0, threshold=0.0):
        before = [x.detach().cpu().numpy().copy() for x in buffers]
        expected = [x.copy() for x in before]
        if op == 24:
            for i in range(n):
                start, end = expected[1][i:i+2]
                expected[2][i] = expected[0][start:end].sum(axis=0, dtype=np.float32)
        else:
            run_cpu(reference, op, expected, n, arg, threshold)
        original(op, *buffers, n=n, arg=arg, threshold=threshold)
        actual = [x.detach().cpu().numpy().copy() for x in buffers]
        for index in OUTPUTS[op]:
            check_arrays(op, index, actual[index], expected[index], before)
            if np.issubdtype(actual[index].dtype, np.floating) and actual[index].size:
                errors[op] = max(errors.get(op, 0), float(np.abs(actual[index] - expected[index]).max()))
        # Export one identical-input case for each kernel and each nontrivial n.
        if n and (op, n) not in tested:
            file = f"{len(cases):04d}-{names[op]}.npz"
            np.savez(export / file, **{f"before_{i}": x for i, x in enumerate(before)},
                     **{f"after_{i}": x for i, x in enumerate(actual)})
            cases.append(dict(op=op, kernel=names[op], n=n, arg=arg, threshold=threshold,
                              buffer_count=len(buffers), output_indices=OUTPUTS[op], file=file))
            tested.add((op, n))

    mesh._run = checked
    results = []
    try:
        for name, v, f, threshold in fixtures():
            ev, ef, labels = reference_fill(v, f, threshold)
            av, af, debug = mesh.fill_mesh_holes(torch.from_numpy(v).mlu(), torch.from_numpy(f).mlu(), threshold, return_debug=True)
            np.testing.assert_allclose(av.cpu().numpy(), ev, rtol=1e-5, atol=2e-7)
            np.testing.assert_array_equal(af.cpu().numpy(), ef)
            if "bound_conn_comp_ids" in debug:
                np.testing.assert_array_equal(debug["bound_conn_comp_ids"].cpu().numpy(), labels)
            results.append({"fixture": name, "status": "passed", "input_vertices": len(v), "output_vertices": len(av)})
        # Utility kernels used by upstream CUB orchestration but replaced by
        # equivalent torch operations in mesh.py still have dedicated MLU tests.
        indices = torch.tensor([2, 0, 3, 1], dtype=torch.int32, device="mlu")
        values = torch.tensor([11, 22, 33, 44], dtype=torch.int32, device="mlu")
        checked(16, indices, values, torch.empty_like(values), n=4)
        checked(17, values, indices, torch.empty_like(values), n=4)
        checked(18, values, torch.empty_like(values[:3]), n=3)
        checked(22, values, torch.empty_like(values), n=4)
        invalid_cases = [
            (torch.tensor([[float('nan'), 0, 0], [1, 0, 0], [0, 1, 0]], device="mlu"),
             torch.tensor([[0, 1, 2]], device="mlu", dtype=torch.int32)),
            (torch.zeros((3, 3), device="mlu"), torch.tensor([[0, 0, 2]], device="mlu", dtype=torch.int32)),
            (torch.zeros((3, 3), device="mlu"), torch.tensor([[2**32, 1, 2]], device="mlu", dtype=torch.int64)),
        ]
        for v, f in invalid_cases:
            try:
                mesh.fill_mesh_holes(v, f)
            except ValueError:
                pass
            else:
                raise AssertionError("Invalid mesh input was accepted")
    finally:
        mesh._run = original
        (export / "cases.json").write_text(json.dumps(cases, indent=2) + "\n")
    covered = {op for op, _ in tested}
    assert covered == set(range(25)), f"Untested kernels: {set(range(25)) - covered}"
    report = {"reference": "upstream_cuda_source_serial_cpu_semantics", "actual_cuda_comparison": "not_run",
              "upstream_commit": mapping["upstream_commit"], "device": torch.mlu.get_device_name(),
              "kernel_count": 24, "added_cub_replacements": 1, "exported_cases": len(cases),
              "float_max_absolute_errors": {names[k]: v for k, v in errors.items()},
              "integer_rule": "exact except CSR row permutations, adjacency pair order and transient hook parent trees",
              "final_component_labels": "exact", "final_faces": "exact", "fixtures": results}
    (args.output / "comparison.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cumesh-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--verify-cuda-export", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.verify_cuda_export:
        raise SystemExit(verify_cuda_exports(args.cumesh_source, args.verify_cuda_export, args.output))
    else:
        validate(args)
