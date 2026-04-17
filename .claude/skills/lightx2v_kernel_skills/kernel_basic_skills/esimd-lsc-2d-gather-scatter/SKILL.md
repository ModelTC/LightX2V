---
name: esimd-lsc-2d-gather-scatter
description: >
  LSC 2D block load/store, 1D block load/store, and gather/scatter operations
  in Intel ESIMD. Use this skill when working with lsc_load_2d, lsc_store_2d,
  lsc_prefetch_2d, config_2d_mem_access, block_load, block_store, gather, or
  scatter in ESIMD kernels. Covers 2D surface descriptors, transposed VNNI loads,
  tile size constraints, cache hints, and common pitfalls like the rvalue
  bit_cast_view bug and half transpose limitation.
---

# ESIMD LSC 2D Block Operations, Gather, and Scatter

Comprehensive guide to all LSC (Load/Store/Cache) memory access patterns in Intel ESIMD: 2D block operations for tiled matrix access, 1D block operations for contiguous access, and gather/scatter for non-contiguous access.

**Version**: 1.0.0
**Last Updated**: 2026-03-12

---

## Table of Contents

1. [2D Block Operations](#2d-block-operations)
2. [Surface Descriptor](#surface-descriptor)
3. [Transposed 2D Load (VNNI)](#transposed-2d-load-vnni)
4. [2D Prefetch](#2d-prefetch)
5. [1D Block Load / Store](#1d-block-load--store)
6. [Gather / Scatter](#gather--scatter)
7. [Cache Hints](#cache-hints)
8. [Tile Size Constraints](#tile-size-constraints)
9. [Common Pitfalls](#common-pitfalls)
10. [Choosing the Right Access Pattern](#choosing-the-right-access-pattern)

---

## 2D Block Operations

`lsc_load_2d` and `lsc_store_2d` access rectangular tiles from global memory using a 2D surface descriptor. They are the primary mechanism for loading matrix tiles for DPAS/XMX operations.

### lsc_load_2d

```cpp
#include <sycl/ext/intel/esimd.hpp>
using namespace sycl::ext::intel::esimd;

// Load a W x H tile of type T, NBlk blocks
auto tile = lsc_load_2d<T, W, H, NBlk,
                        Transposed,    // bool: transpose on load
                        Transformed,   // bool: VNNI transform
                        L1Hint, L3Hint>(payload);
```

**Returns**: `simd<T, W * H * NBlk>` — a flat vector of all loaded elements.

### lsc_store_2d

```cpp
lsc_store_2d<T, W, H, NBlk,
             L1Hint, L3Hint>(payload, data);
```

**Parameters**: `data` is `simd<T, W * H * NBlk>`.

---

## Surface Descriptor

The `config_2d_mem_access<T, W, H, NBlk>` descriptor defines the 2D surface and the tile within it.

```cpp
config_2d_mem_access<T, W, H, NBlk> payload(
    base_ptr,       // const T* — base address of the 2D surface
    surface_width,  // uint32_t — surface width in BYTES minus 1
    surface_height, // uint32_t — surface height in ROWS minus 1
    surface_pitch,  // uint32_t — surface pitch in BYTES minus 1
    x_offset,       // int — tile x offset in ELEMENTS (not bytes!)
    y_offset        // int — tile y offset in ROWS
);
```

**Critical**: `surface_width`, `surface_height`, and `surface_pitch` are all specified as *value minus 1*. The x offset is in **elements**, while width and pitch are in **bytes**.

```cpp
// Example: load 16x8 tile of half from a [rows, cols] matrix
const int cols = 256;
const int rows = 1024;
config_2d_mem_access<half, 16, 8, 1> payload(
    ptr,
    (uint32_t)(cols * sizeof(half)) - 1u,   // width in bytes - 1
    (uint32_t)(rows) - 1u,                   // height in rows - 1
    (uint32_t)(cols * sizeof(half)) - 1u,   // pitch in bytes - 1
    col_offset,                              // x in elements
    row_offset                               // y in rows
);
auto tile = lsc_load_2d<half, 16, 8, 1,
                        false, false,
                        cache_hint::cached, cache_hint::cached>(payload);
```

### Updating Offsets

```cpp
payload.set_x(new_col_offset);  // in elements
payload.set_y(new_row_offset);  // in rows
```

---

## Transposed 2D Load (VNNI)

Setting `Transposed=true` transposes the tile on load: `[H rows x W cols]` in memory becomes `[W rows x H cols]` in registers. This is essential for producing DPAS B operands in VNNI layout.

### Constraint: Only works with uint32_t, NOT half

```cpp
// BAD: compile error — Transposed=true not supported for half
lsc_load_2d<half, 8, 16, 1, /*Transposed=*/true, false, ...>(payload);

// GOOD: use uint32_t, then bit_cast_view to half
config_2d_mem_access<uint32_t, K/2, N, 1> k_payload(
    reinterpret_cast<const uint32_t*>(k_ptr),
    (uint32_t)(HEAD_DIM * sizeof(half)) - 1u,
    (uint32_t)(num_blocks) - 1u,
    (uint32_t)(HEAD_DIM * sizeof(half)) - 1u,
    0, 0);

auto k_u32  = lsc_load_2d<uint32_t, K/2, N, 1,
                           /*Transposed=*/true, /*Transformed=*/false,
                           cache_hint::cached, cache_hint::cached>(k_payload);
// MUST store to named variable first (see pitfalls)
auto k_tile = k_u32.template bit_cast_view<half>().read();
```

The `uint32_t` type packs two FP16 values per element, and the transpose produces the VNNI-packed layout that DPAS src1 expects: pairs of FP16 values interleaved across the systolic depth.

### Transposed Load Width Constraints

For transposed loads, the **maximum width** depends on the element type:
- `uint32_t` transpose: max width = 8 (so max K/2 = 8, meaning K = 16 FP16 elements)
- `half` transpose: max width = 4 (but half transpose is NOT supported on Xe2)

---

## 2D Prefetch

```cpp
lsc_prefetch_2d<T, W, H, NBlk,
                L1Hint, L3Hint>(payload);
```

Use to hide memory latency by prefetching the next tile while computing the current one. Typical pattern:

```cpp
// Prefetch next K tile during VS phase (cross-phase prefetch)
k_payload.set_y(next_k_offset);
lsc_prefetch_2d<uint32_t, K/2, N, 1,
                cache_hint::cached, cache_hint::cached>(k_payload);
```

---

## 1D Block Load / Store

For contiguous 1D access, `block_load` and `block_store` are simpler and often faster than 2D operations.

```cpp
// Load N contiguous elements of type T
simd<half, 128> data = block_load<half, 128>(ptr + offset);

// Store N contiguous elements
block_store<half, 128>(ptr + offset, data);

// With cache hints
simd<half, 128> data = block_load<half, 128>(ptr + offset,
    overaligned_tag<16>{},       // alignment hint
    properties{cache_hint_L1<cache_hint::streaming>,
               cache_hint_L3<cache_hint::cached>});
```

**When to prefer `block_load` over `lsc_load_2d`**:
- Accessing full rows of a matrix (contiguous in memory)
- Loading into SLM cooperatively (each thread loads V_ROWS full rows)
- Simpler code, fewer descriptor setup costs
- Measured ~4% faster than `lsc_load_2d` for cooperative K+V loads to SLM

---

## Gather / Scatter

For non-contiguous access where each SIMD lane reads/writes a different address.

```cpp
// Gather: load one element per lane from different offsets
simd<uint32_t, 16> offsets = ...;  // byte offsets from base
simd<half, 16> data = gather<half, 16>(base_ptr, offsets);

// Scatter: store one element per lane to different offsets
scatter<half, 16>(base_ptr, offsets, data);

// Predicated gather/scatter with mask
simd_mask<16> pred = ...;
simd<half, 16> data = gather<half, 16>(base_ptr, offsets, pred);
scatter<half, 16>(base_ptr, offsets, data, pred);
```

**When to use gather/scatter**:
- Strided access patterns (e.g., reading a column from a row-major matrix)
- Index-based access (e.g., embedding lookup)
- Scatter output to non-contiguous positions

**Performance**: Gather/scatter are significantly slower than block operations. Prefer `block_load`/`block_store` or `lsc_load_2d`/`lsc_store_2d` whenever possible.

---

## Cache Hints

All LSC operations accept L1 and L3 cache hints:

| Hint | Meaning | Use Case |
|------|---------|----------|
| `cache_hint::cached` | Cache in L1/L3 | Data reused soon (default for most loads) |
| `cache_hint::uncached` | Bypass cache | Streaming writes, one-time reads |
| `cache_hint::streaming` | Streaming hint | Sequential access, mild temporal locality |
| `cache_hint::write_back` | Write-back | Store with cache retention |

**Important**: For `block_store`, **default hints (no hints)** are often fastest. Explicit `(streaming, write_back)` measured 33% slower; `(streaming, uncached)` measured 7x slower.

```cpp
// Best for block_store — use default, no explicit hints
block_store<half, N>(ptr + offset, data);

// For 2D loads that will be reused — cached/cached
lsc_load_2d<..., cache_hint::cached, cache_hint::cached>(payload);
```

---

## Tile Size Constraints

### 2D Block Load/Store Limits

| Parameter | Constraint |
|-----------|-----------|
| Tile width (in bytes) | `W * sizeof(T) <= 64` bytes |
| Tile height | `H <= 32` rows |
| Surface pitch | Must be `>= surface_width` (both in bytes) |
| Alignment | Base pointer must be 64-byte aligned (surface base) |
| X offset | Must be element-aligned for the type |

### Practical Maximum Tile Sizes

| Type | Max W (elements) | Max H | Max Elements |
|------|------------------|-------|-------------|
| `half` (2B) | 32 | 32 | 1024 |
| `float` (4B) | 16 | 32 | 512 |
| `uint32_t` (4B) | 16 | 32 | 512 |
| `uint8_t` (1B) | 64 | 32 | 2048 |

### 2D Store Height Limit

`lsc_store_2d` with `half`: max height = 8 per store. For larger tiles, split into multiple stores.

---

## Common Pitfalls

### 1. `lsc_load_2d` Returns Rvalue — Cannot Chain `.bit_cast_view()`

```cpp
// BAD: rvalue, compiler error or undefined behavior
auto tile = lsc_load_2d<uint32_t, K/2, N, 1, true, false, ...>(payload)
                .template bit_cast_view<half>().read();

// GOOD: store to named variable first
auto k_u32  = lsc_load_2d<uint32_t, K/2, N, 1, true, false, ...>(payload);
auto k_tile = k_u32.template bit_cast_view<half>().read();
```

This is a fundamental ESIMD API constraint. `lsc_load_2d` returns an rvalue `simd` object; calling `.bit_cast_view()` on an rvalue produces a dangling reference.

### 2. X Offset is in Elements, Not Bytes

```cpp
// BAD: byte offset for x — loads wrong columns
config_2d_mem_access<half, 16, 8, 1> payload(ptr, wd, ht, pt,
    (int)(col * sizeof(half)), row);  // WRONG: x in bytes

// GOOD: element offset for x
config_2d_mem_access<half, 16, 8, 1> payload(ptr, wd, ht, pt,
    (int)col, row);  // CORRECT: x in elements
```

### 3. Width/Height/Pitch Are Value Minus 1

```cpp
// BAD: raw values
config_2d_mem_access<half, 16, 8, 1> payload(ptr,
    cols * sizeof(half),      // WRONG: not minus 1
    rows,                     // WRONG: not minus 1
    cols * sizeof(half),      // WRONG: not minus 1
    0, 0);

// GOOD: subtract 1
config_2d_mem_access<half, 16, 8, 1> payload(ptr,
    (uint32_t)(cols * sizeof(half)) - 1u,
    (uint32_t)(rows) - 1u,
    (uint32_t)(cols * sizeof(half)) - 1u,
    0, 0);
```

### 4. Transposed Load Does NOT Work with `half`

On Xe2 hardware, `Transposed=true` is only supported for `uint32_t` (and `uint64_t`). Use `uint32_t` with `bit_cast_view<half>` for VNNI transposed FP16 loads.

### 5. `lsc_store_2d` Height Limit for FP16

Max store height for `half` is 8. For a 16-row tile, split into two 8-row stores:

```cpp
lsc_store_2d<half, W, 8, 1, ...>(payload, data_upper);
payload.set_y(payload.get_y() + 8);
lsc_store_2d<half, W, 8, 1, ...>(payload, data_lower);
```

---

## Choosing the Right Access Pattern

| Access Pattern | Operation | When to Use |
|---------------|-----------|-------------|
| `lsc_load_2d` / `lsc_store_2d` | 2D tiled access | Matrix tiles for DPAS, strided 2D regions |
| `block_load` / `block_store` | 1D contiguous | Full rows, SLM cooperative loads, simple patterns |
| `gather` / `scatter` | Non-contiguous | Indexed access, strided columns, irregular patterns |
| `lsc_prefetch_2d` | 2D prefetch | Hide latency for next iteration's tiles |

**Decision flow**:
1. Is the data contiguous? Use `block_load` / `block_store` (simplest, often fastest).
2. Is the data a rectangular tile? Use `lsc_load_2d` / `lsc_store_2d`.
3. Is the data non-contiguous? Use `gather` / `scatter` (slowest, use as last resort).

---

## Related Skills

| Skill | Relevance |
|-------|-----------|
| `intel-esimd-base` | Foundational ESIMD programming, includes memory access overview |
| `esimd-lsc-slm` | SLM-specific LSC operations (slm_block_load, lsc_slm_gather/scatter) |
| `xe2-dpas-patterns` | DPAS tiling and VNNI layout that 2D loads produce |
| `intel-esimd-attn-gemm` | Attention GEMM kernels using lsc_load_2d for K tiles |
| `xe2-sdp-hd256` | HD=256 SDP kernel using lsc_slm_scatter for S transpose |
| `intel-gpu-kernel-opt` | General optimization methodology including memory access tuning |
