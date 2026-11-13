# Cambricon MLU mesh kernels

CuMesh-compatible small-hole filling for TRELLIS.2. Sources are adapted from
[CuMesh](https://github.com/JeffreyXiang/CuMesh) commit
`12289e1062f0603f2f0d0771b02e1395d247f26f` under its MIT license (see LICENSE).

## Build

Use a Cambricon development environment with matching PyTorch, torch_mlu,
Neuware/CNCC and a C++ compiler. The validated development stack is Python 3.12,
PyTorch 2.9.1, torch_mlu 1.31.1 and CNToolkit 4.6.2 on MLU590-M9D.
Build from this directory with the runtime's Python:

```bash
MAX_JOBS=2 python -m pip install --no-build-isolation --no-deps .
```

The repository MLU Dockerfile also builds this package. Compiled objects/wheels
are specific to that vendor ABI; rebuild after changing the vendor stack.

## API and algorithm

```python
from lightx2v_kernel_mlu import fill_mesh_holes
vertices, faces = fill_mesh_holes(vertices, faces, max_hole_perimeter=0.03)
```

Inputs are vertices `[V,3]` and integer faces `[F,3]` on the same MLU. The result
uses FP32 vertices and int32 faces. Original vertices/faces are retained. A
closed boundary component is filled only when its perimeter is strictly smaller
than the threshold. Its new vertex is the mean of edge midpoints; one triangle
`[edge_high, edge_low, new_vertex]` is appended for every boundary edge, exactly
following the upstream center-fan algorithm. Invalid indices, nonfinite vertices
and faces with repeated vertex IDs are rejected.

The platform wrapper is `lightx2v_platform/ops/mesh/cambricon_mlu`; TRELLIS moves
its CPU decoder mesh to the selected MLU for this operation and returns the mesh
to its original device. Normal inference and developer evaluation use this same
model code. Importing another platform does not require this extension.

## CUDA kernel correspondence

`kernel-map.json` fixes each upstream function hash. The MLU bodies preserve the
CUDA work-item algorithm, replacing CUDA thread IDs with MLU task-strided loops
and integer atomic operations with BANG atomics. The packed MLU Vec3f uses three
floats; CUDA's unused fourth padding word is not exposed in logical tensor data.
The uint64 edge keys are accessed as two uint32 words with the same byte layout.
This avoids scalar uint64 expression truncation observed with this CNCC build
for real meshes whose vertex IDs exceed 65535. Logical key values are unchanged.

| CUDA source | CUDA kernel | MLU kernel |
| --- | --- | --- |
| `src/connectivity.cu` | `expand_edges_kernel` | `expand_edges_kernel` |
| `src/connectivity.cu` | `get_vertex_edge_cnt_kernel` | `get_vertex_edge_cnt_kernel` |
| `src/connectivity.cu` | `get_vertex_edge_adjacency_kernel` | `get_vertex_edge_adjacency_kernel` |
| `src/connectivity.cu` | `set_boundary_vertex_kernel` | `set_boundary_vertex_kernel` |
| `src/connectivity.cu` | `get_vertex_boundary_cnt_kernel` | `get_vertex_boundary_cnt_kernel` |
| `src/connectivity.cu` | `get_vertex_boundary_adjacency_kernel` | `get_vertex_boundary_adjacency_kernel` |
| `src/connectivity.cu` | `get_vertex_is_manifold_kernel` | `get_vertex_is_manifold_kernel` |
| `src/connectivity.cu` | `set_manifold_bound_adj_kernel` | `set_manifold_bound_adj_kernel` |
| `src/shared.cu` | `hook_edges_kernel` | `hook_edges_kernel` |
| `src/shared.cu` | `compress_components_kernel` | `compress_components_kernel` |
| `src/connectivity.cu` | `is_bound_conn_comp_loop_kernel` | `is_bound_conn_comp_loop_kernel` |
| `src/clean_up.cu` | `compute_loop_boundary_lengths` | `compute_loop_boundary_lengths` |
| `src/clean_up.cu` | `compute_loop_boundary_midpoints` | `compute_loop_boundary_midpoints` |
| `src/clean_up.cu` | `connect_new_vertices_kernel` | `connect_new_vertices_kernel` |
| `src/shared.h` | `arange_kernel` | `arange_kernel` |
| `src/shared.h` | `fill_kernel` | `fill_kernel` |
| `src/shared.h` | `scatter_kernel` | `scatter_kernel` |
| `src/shared.h` | `index_kernel` | `index_kernel` |
| `src/shared.h` | `diff_kernel` | `diff_kernel` |
| `src/shared.h` | `set_flag_kernel` | `set_flag_kernel` |
| `src/shared.h` | `compare_kernel` | `compare_kernel` |
| `src/shared.h` | `inplace_div_kernel` | `inplace_div_kernel` |
| `src/shared.h` | `get_diff_kernel` | `get_diff_kernel` |
| `src/clean_up.cu` | `copy_vec3f_to_float3_kernel` | `copy_vec3f_to_float3_kernel` |

CUB sort, selection, run-length encoding and scans are implemented with torch
MLU sorting, unique/unique_consecutive, indexing and cumsum. The additional
`segment_sum_kernel` replaces CUB segmented sums. It sums FP32 values in order
within each segment; CUB's parallel reduction tree can round differently.

CSR adjacency append order and the order of a boundary-adjacency pair are
unspecified in CUDA because of atomics, so validation compares their contents.
Intermediate union-find parent trees can differ under concurrent hooking; final
compressed component labels and mesh faces must match exactly. Float checks use
`atol=2e-7, rtol=1e-5`. Threshold decisions must also agree; do not hide a changed
fill decision behind a float tolerance.

## Validation and actual CUDA comparison

Clone the pinned CuMesh source separately; it is required only for validation.
The test compiles the original CUDA kernel bodies as a serial CPU semantic
reference, then compares identical inputs to the MLU kernels, checks complete
meshes with an independent NumPy topology oracle, and exports the test inputs
and MLU outputs for NVIDIA execution:

```bash
python tests/compare_kernels.py --cumesh-source /path/to/CuMesh \
  --output /path/to/mlu-comparison
```

Use one exclusively allocated MLU. Fixtures cover closed meshes, empty meshes,
small and large holes, multiple loops, non-manifold edges, shared-edge grids and
both sides of the strict perimeter threshold. All 24 upstream custom kernels,
including utility kernels, and the new segmented sum are covered.

On an NVIDIA host with PyTorch CUDA and NVCC, copy the exported directory and run:

```bash
python tests/compare_kernels.py --cumesh-source /path/to/CuMesh \
  --verify-cuda-export /path/to/mlu-comparison/export \
  --output /path/to/cuda-comparison
```

This second command executes the original CUDA kernel bodies on the exact
exported inputs and compares against the saved MLU outputs. A CPU semantic
reference pass is reported separately and is **not** a CUDA hardware pass.
The added segmented-sum kernel is covered by CPU-reference tests; this export
command covers upstream custom kernels and does not claim a CUB hardware test.
