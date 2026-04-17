---
name: intel-esimd-base
description: >
  Foundational Intel ESIMD GPU programming skill. Use this skill proactively
  whenever the user is writing, optimizing, or debugging any SYCL/ESIMD kernel
  for Intel GPUs — including Intel Arc, Iris Xe, or Data Center GPU Max. Covers
  kernel design, memory access patterns (block_load, gather, SLM), data types,
  vectorization, workgroup patterns, hardware characteristics, performance
  analysis, and troubleshooting. Trigger this even when the user does not
  explicitly say "ESIMD" — invoke it for any Intel GPU kernel development,
  performance bottleneck questions, or SYCL optimization tasks targeting Intel
  hardware.
---

# Intel ESIMD Base Skill

Foundational guidance for writing, optimizing, and debugging Intel GPU ESIMD kernels — applicable to any kernel regardless of algorithm. Covers kernel design, all memory access patterns, data types, vectorization, workgroup patterns, hardware characteristics, performance analysis, and troubleshooting.

---

## Table of Contents

1. [What Is ESIMD](#what-is-esimd)
2. [Hardware Characteristics](#hardware-characteristics)
3. [Optimization Priority Checklist](#optimization-priority-checklist)
4. [Kernel Design](#kernel-design)
5. [Memory Access Patterns](#memory-access-patterns)
6. [Data Types](#data-types)
7. [Vectorization Techniques](#vectorization-techniques)
8. [Workgroup and SLM Patterns](#workgroup-and-slm-patterns)
9. [Register File vs SLM](#register-file-vs-slm)
10. [Loop Optimization](#loop-optimization)
11. [Hybrid Scalar/Vector Strategy](#hybrid-scalarvector-strategy)
12. [Testing New Kernels](#testing-new-kernels)
13. [Common Performance Traps](#common-performance-traps)
14. [Quick Reference](#quick-reference)

---

## What Is ESIMD

ESIMD (Explicit SIMD) is a C++ programming model for Intel GPU Execution Units (EUs). Each ESIMD work-item maps to one EU thread and operates on explicit `simd<T, N>` register vectors rather than the implicit SIMD of standard SYCL. This gives direct control over:
- Register allocation and vector widths
- Memory access patterns (`block_load`, `gather`, SLM)
- Hardware intrinsics (`hmax`, `hmin`, `pack_mask`, `fbl`, `xmx::dpas`)

**Required headers and annotation**:
```cpp
#include <sycl/sycl.hpp>
#include <sycl/ext/intel/esimd.hpp>

// Kernel lambda must be annotated:
[=](sycl::id<1> idx) SYCL_ESIMD_KERNEL { ... }
// or equivalently:
[=](sycl::nd_item<1> item) [[intel::sycl_explicit_simd]] { ... }
```

**Namespace**: bring in ESIMD intrinsics with:
```cpp
using namespace sycl::ext::intel::esimd;
```

---

## Hardware Characteristics

### Intel Arc GPU / iGPU (Xe architecture)

| Resource | Spec |
|----------|------|
| EUs (Execution Units) | 128–512 (model dependent) |
| Threads per EU | 8 |
| Total threads | EUs × 8 |
| GRF registers per thread (Xe1) | 128 × 32B = **4 KB** |
| GRF registers per thread (Xe2+) | 256 × 32B = **8 KB** |
| SIMD width per thread (Xe1) | 256-bit → **16 FP16 elements** per instruction |
| SIMD width per thread (Xe2+) | 512-bit → **32 FP16 elements** per instruction |
| SLM per sub-slice | 128 KB |
| Recommended SLM | < 64 KB |

**Register file budget**:

| Architecture | GRF per thread | FP16 capacity | FP32 capacity |
|---|---|---|---|
| **Xe1** (Arc Alchemist, iGPU Gen12) | 128 × 32B = **4 KB** | up to ~2048 elements | up to ~1024 elements |
| **Xe2+** (Arc Battlemage, Lunar Lake+) | 256 × 32B = **8 KB** | up to ~4096 elements | up to ~2048 elements |

- 512 FP16 values = 1 KB — 25% of Xe1 budget (safe), 12.5% of Xe2 budget (very safe)
- Register spilling to L1 cache incurs significant latency
- If a kernel exceeds the register budget, enable **large GRF mode** (see Compilation)

**SIMD instruction width**:

| Architecture | SIMD width | FP16 elements/instruction |
|---|---|---|
| **Xe1** (Arc Alchemist, iGPU Gen12) | 256-bit | **16 FP16** per instruction |
| **Xe2+** (Arc Battlemage, Lunar Lake+) | 512-bit | **32 FP16** per instruction |

- For `simd<half, N>` with N > the native width, the compiler **automatically splits** the operation into multiple native-width instructions — no manual tiling needed.
- Optimal `simd` widths: multiples of 16 (Xe1) or 32 (Xe2+). E.g., `simd<half, 128>` = 8 instructions on Xe1, 4 on Xe2+.
- This means Xe2+ issues **2× fewer instructions** for the same vector width, contributing to its higher throughput alongside the larger GRF.

**Query thread count at runtime** (instead of hard-coding):
```cpp
const int ts = (int)q.get_device()
    .get_info<sycl::info::device::max_compute_units>() * 8;
```

---

## Optimization Priority Checklist

Apply in this order — each step can dwarf the next:

| Priority | Technique | Typical Impact |
|----------|-----------|---------------|
| 0 | **Kernel design** (range dim, work/thread, template params) | **10–35×** |
| 1 | Memory access (block_load, alignment, coalescing) | 2–5× |
| 2 | Data types (FP16 > FP32, uint4/int8 for weights) | 2–8× |
| 3 | Vectorization (hmax/hmin, SIMD ops, avoid scalar loops) | 1.5–2× |
| 4 | Parallelization (workgroup size, SLM reduction) | 10–30% |
| 5 | Register file vs SLM (~1 cycle vs ~30 cycles) | up to 2× |
| 6 | Algorithm complexity (O(N) vs O(K×N), unroll, hoist) | varies |

---

## Kernel Design

This is the single most impactful decision. A bad design can be **10–35× slower** before any micro-optimization.

### Rule 1: Match Kernel Range Dimensionality to Data Shape

Use multi-dimensional `range<D>` so indices map directly to data dimensions. Never decode a 1D linear index into multi-dimensional coordinates at runtime.

```cpp
// BAD: 1D range + manual 4D decode → 1% bandwidth
q.parallel_for(range<1>(bsz * heads * seq * blocks),
    [=](id<1> idx) SYCL_ESIMD_KERNEL {
        int d4 =  idx[0] % blocks;
        int d3 = (idx[0] / blocks) % seq;
        int d2 = (idx[0] / (blocks * seq)) % heads;
        int d1 =  idx[0] / (blocks * seq * heads);
        output[idx[0]] = process(input[idx[0]]);  // 1 element/thread
    });

// GOOD: 3D range → ~73% bandwidth (21.7× faster before any other change)
q.parallel_for(range<3>(bsz, heads, seq),
    [=](id<3> idx) SYCL_ESIMD_KERNEL {
        int b = idx[0], h = idx[1], s = idx[2];
        // each thread processes many output elements for this (b, h, s)
    });
```

### Rule 2: Each Thread Must Do Meaningful Work

Processing 1 element per thread creates massive scheduling overhead. Target **32–128+ elements per thread** (or one complete row/column/tile).

```cpp
// GOOD: thread processes 128 elements
constexpr int ELEMS = 128;
const int base = idx[0] * ELEMS;
simd<sycl::half, ELEMS> data = block_load<sycl::half, ELEMS>(input + base);
// ... process data ...
block_store<sycl::half, ELEMS>(output + base, data);
```

### Rule 3: Use Template Parameters for All Compile-Time Constants

Runtime variables prevent loop unrolling and constant propagation. Make all array sizes and loop bounds `constexpr` or template parameters.

```cpp
// BAD: runtime N prevents unrolling
void kernel(int N) { for (int i = 0; i < N; i++) ... }

// GOOD: compile-time N enables full unroll and SIMD vectorization
template<int N>
void kernel() {
    #pragma unroll
    for (int i = 0; i < N; i++) ...
}
```

### Rule 4: Separate Load and Compute Loops

When computing multiple dot products or reductions, issue all `block_load`s before any compute. This lets the hardware pipeline memory fetches:

```cpp
// BAD: each load stalls its own compute
for (int i = 0; i < K; i++) {
    auto v = block_load<half, D>(ptr + i * D);
    results[i] = detail::sum<half, half, D>(v * w);
}

// GOOD: all loads in-flight before compute begins (+29% measured)
simd<half, D> vecs[K];
for (int i = 0; i < K; i++) vecs[i] = block_load<half, D>(ptr + i * D);
for (int i = 0; i < K; i++) results[i] = detail::sum<half, half, D>(vecs[i] * w);
```

---

## Memory Access Patterns

### Block Load / Store (Contiguous Access)

The primary memory primitive. Maps to one or a few hardware memory transactions.

```cpp
// Load N elements starting at ptr (ptr must be N*sizeof(T) aligned by default)
simd<T, N> data = block_load<T, N>(ptr);
simd<T, N> data = block_load<T, N>(ptr + offset);  // offset in elements

// Store
block_store<T, N>(ptr + offset, data);
```

**Efficiency rules**:
- Prefer loads of **128B or 256B** (e.g., `block_load<half, 128>` = 256B, `block_load<float, 64>` = 256B)
- All threads should access consecutive memory addresses (coalescing)
- Larger individual loads are more efficient than many small ones

**CRITICAL — address alignment**: `block_load` and `block_store` require the **byte address to be a multiple of 4** by default. For 4-byte types (`float`, `int`) this is automatically satisfied. For 2-byte types (`half`, `bfloat16`) an odd element index produces a non-4-byte-aligned byte address and gives wrong results. **When you cannot guarantee the address is 4-byte aligned, you must pass an `alignment<>` property.**

```cpp
// ── half / bfloat16 (2 bytes each) ───────────────────────────────────────
// Element index 3 → byte address +6 → NOT a multiple of 4 → wrong results!
simd<half, 8> data = block_load<half, 8>(ptr + 3);                          // WRONG
simd<half, 8> data = block_load<half, 8>(ptr + 3, properties{alignment<2>}); // OK

// Safe rule: always use alignment<2> for half* / bfloat16* block_load/store
// unless you can statically prove the element offset is even (byte offset % 4 == 0).
block_store<half, 8>(ptr + 3, data, properties{alignment<2>});               // OK

// ── float / int (4 bytes each) ────────────────────────────────────────────
// 4-byte elements: any element index gives a 4-byte-aligned byte address → safe.
simd<float, 8> data = block_load<float, 8>(ptr + 3);   // OK (byte addr = +12)

// ── uint8_t (1 byte each) ────────────────────────────────────────────────
// 1-byte elements: any byte address → always need alignment<1>.
simd<uint8_t, 8> data = block_load<uint8_t, 8>(ptr + 3, properties{alignment<1>});
```

**Summary by type**:
| Type | Size | Default required alignment | When to annotate |
|------|------|---------------------------|-----------------|
| `float`, `int`, `uint32_t` | 4 B | 4 B | never needed (4B × any index = multiple of 4) |
| `half`, `bfloat16` | 2 B | 4 B | whenever element offset may be odd |
| `uint8_t`, `int8_t` | 1 B | 4 B | always (`alignment<1>`) |

**Symptom of missing annotation**: incorrect values loaded/stored, sporadic failures depending on offset parity, no compile error.

### Gather / Scatter (Non-Contiguous Access)

Use when elements are at irregular or strided offsets — e.g., sliding windows, strided output.

```cpp
// Gather: load N elements from per-lane byte offsets
simd<T, N> data = gather<T, N>(base_ptr, simd<uint32_t, N>(byte_offsets));

// Scatter: store N elements to per-lane byte offsets
scatter<T, N>(base_ptr, simd<uint32_t, N>(byte_offsets), data);
```

**Key rules**:
- Offsets are in **bytes**, not element indices. Convert: `byte_offsets = elem_offsets * sizeof(T)`
- Offset type must be `simd<uint32_t, N>` — unsigned 32-bit
- Negative element indices cast to `uint32_t` wrap to huge values → clamp before cast
- Optimal batch size: **N=32** for FP16 (matches hardware gather width)
- Slightly slower than `block_load` for contiguous data, much faster for strided patterns

**Batch gather pattern** (process 32 outputs in parallel):
```cpp
constexpr int BS = 32;
simd<int, BS> rel = simd<int, BS>(0, 1) * STRIDE;  // [0, S, 2S, ..., 31S]

for (int out = 0; out < num_outputs; out += BS) {
    // Clamp offsets to valid range (branchless — safe for idempotent ops like max)
    simd<int, BS> offs = max(out * STRIDE + rel + OFFSET, 0);
    offs = min(offs, max_elem);

    simd<T, BS> vals = gather<T, BS>(
        input + base, simd<uint32_t, BS>(offs) * sizeof(T));

    // ... process vals ...

    int n = std::min(BS, num_outputs - out);
    if (n == BS) block_store<T, BS>(output + out_base + out, result);
    else         for (int i = 0; i < n; i++) output[out_base + out + i] = result[i];
}
```

**Performance reference**: gather achieved 72.8% bandwidth on a strided sliding-window pooling kernel (82.9× total speedup over naive scalar 1D design).

### SLM — Shared Local Memory

Use for inter-thread communication within a workgroup. Requires `nd_range` and `barrier()`.

```cpp
// Initialize SLM (compile-time size, inside kernel)
constexpr int SLM_BYTES = GROUP_SIZE * sizeof(float);
slm_init<SLM_BYTES>();

// Per-thread write (byte offset)
slm_block_store<float, 1>(local_id * sizeof(float), simd<float, 1>(partial));

// Synchronize all threads in workgroup
barrier();

// Read all GROUP_SIZE values (thread 0 only, or all threads for broadcast)
simd<float, GROUP_SIZE> parts = slm_block_load<float, GROUP_SIZE>(0);
```

**SLM limits and cost**:
- Max 128 KB per sub-slice; keep usage < 64 KB
- Access latency ~30 cycles vs ~1 cycle for register file
- `slm_init` must be called inside the kernel, unconditionally
- `barrier()` must be reached by **all** threads in the workgroup — never inside a branch that some threads skip

---

## Data Types

### FP16 vs FP32

**Use FP16** for: weights, activations, intermediate results in neural network ops, any case where error < 0.1 is acceptable.

**Use FP32** for: accumulators summing >1000 terms (use `simd<float, N> acc` even when inputs are FP16), numerical stability-critical paths, softmax exponents.

```cpp
// Pattern: FP16 inputs → FP32 accumulation → FP16 output
simd<sycl::half, 128> a = block_load<sycl::half, 128>(ptr_a);
simd<sycl::half, 128> b = block_load<sycl::half, 128>(ptr_b);
simd<float, 128> acc(0.f);
acc += convert<float>(a) * convert<float>(b);
float result = sycl::ext::intel::esimd::detail::sum<float, float, 128>(acc);
block_store<sycl::half, 1>(out, simd<sycl::half, 1>(sycl::half(result)));
```

---

## Vectorization Techniques

### Arithmetic

```cpp
// Element-wise ops (all broadcast scalars automatically)
simd<T, N> c = a + b;
simd<T, N> c = a * b + scalar;   // FMA when T=float
simd_mask<N> m = a > b;
simd<T, N> r = merge(a, b, m);   // r[i] = m[i] ? a[i] : b[i]
```

### Scalar × Vector (No Explicit Broadcast Needed)

Intel GPU hardware natively supports multiplying a scalar by a `simd` vector — the scalar is broadcast implicitly by the FMA unit. **No `replicate_w` or `simd<T,N>(scalar)` constructor needed.**

```cpp
// GOOD: extract scalar from simd, multiply directly — one instruction
half k_val = k_tile[tt * SUB_HD + ii];   // simd_view → half (implicit conversion)
acc[ii] += k_val * v_row;                // scalar * simd<half, N> → simd<half, N>

// BAD: explicit broadcast via replicate_w — wastes a GRF shuffle instruction
simd<half, N> k_bcast = k_tile.template replicate_w<N, 1>(tt * SUB_HD + ii);
acc[ii] += k_bcast * v_row;
```

**Why**: `replicate_w<N, 1>(i)` emits a hardware GRF shuffle (region select) to fill an N-wide register. When used only as a multiplier, the GPU's scalar-broadcast path in the EU instruction set handles this for free inside the FMA — eliminating the shuffle saves one instruction per (tt, ii) pair.

**Extraction note**: `simd<T,N>::operator[]` returns a `simd_view`, which implicitly converts to `T` in arithmetic contexts. The result of `half k_val = simd_vec[i]` is a true scalar `half` register.

### Reductions

```cpp
// Horizontal sum — USE detail::sum, NOT reduce<T>(vec, std::plus<T>())
// WHY: with `using namespace sycl::ext::intel::esimd`, the name `reduce` is
// ambiguous between esimd::reduce and C++17 std::reduce (iterator overload).
// The compiler silently picks std::reduce which returns 0. This is a silent bug
// with no compile error that produces wrong results.
float s = sycl::ext::intel::esimd::detail::sum<float, float, N>(vec);
sycl::half s = sycl::ext::intel::esimd::detail::sum<sycl::half, sycl::half, N>(vec);

// Horizontal max / min (direct hardware instruction, works on FP16)
sycl::half mx = hmax<sycl::half>(vec);
sycl::half mn = hmin<sycl::half>(vec);
float      mx = hmax<float>(vec);

// For FP16 reductions that need FP32 precision:
float mx = hmax<float>(convert<float>(vec));  // upcast first
```

### Sub-Range Access (`select`)

```cpp
// Read a contiguous sub-range: elements [offset, offset+COUNT)
simd<T, COUNT> sub = vec.template select<COUNT, 1>(offset);

// Read with stride: elements [offset, offset+STRIDE, offset+2*STRIDE, ...]
simd<T, COUNT> sub = vec.template select<COUNT, STRIDE>(offset);

// Write a sub-range
vec.template select<COUNT, 1>(offset) = other_vec;
```

`select` returns a `simd_view` (not `simd`) — assign to a named `simd<T,N>` when you need to pass it to a function.

### Type Conversion

```cpp
simd<float, N>      fp32 = convert<float>(fp16_vec);
simd<sycl::half, N> fp16 = convert<sycl::half>(fp32_vec);
simd<float, N>      fp32 = convert<float>(uint8_vec);

// Bit-cast reinterpret (zero cost, requires lvalue)
auto u32_vec = some_expression();  // store to named variable first
auto i32_vec = u32_vec.template bit_cast_view<int32_t>();  // reinterpret in place
```

### `simd` Constructor Patterns

```cpp
// Broadcast scalar
simd<float, N> v(0.f);            // all zeros
simd<sycl::half, N> v(-65504.f); // FP16 -inf

// Arithmetic sequence: [start, start+step, start+2*step, ...]
simd<int, 32> iota(0, 1);         // [0, 1, 2, ..., 31]
simd<int, 32> strided(0, 4);      // [0, 4, 8, ..., 124]
```

### pack_mask and fbl

Used to find the first lane matching a condition, without scalar loops:

```cpp
// pack_mask: simd_mask<N> → uint32_t bitmask (N = 8, 16, or 32 only)
uint32_t m = pack_mask(heap == min_val);   // bit i = 1 if heap[i] == min_val

// fbl (find bit lowest): index of least significant set bit; -1 if all zero
uint32_t pos = fbl(m);                     // scalar
simd<uint32_t, K> pos_vec = fbl(mask_vec); // vectorized per-element

// For N > 32: split into 32-element chunks, vectorized fbl, reconstruct global index
constexpr int MASK_N = N / 32;
simd<uint32_t, MASK_N> masks;
#pragma unroll
for (int mi = 0; mi < MASK_N; mi++) {
    simd<sycl::half, 32> chunk = vec.template select<32, 1>(mi * 32);
    masks[mi] = pack_mask(chunk == target);
}
simd<uint32_t, MASK_N> fbl_r   = fbl(masks);
simd<int32_t,  MASK_N> bit_idx = fbl_r.template bit_cast_view<int32_t>();
simd<int32_t,  MASK_N> glob    = bit_idx + simd<int32_t, MASK_N>(0, 32);
glob = merge(bit_idx, glob, bit_idx == -1);  // keep -1 for chunks with no match
int pos = (int)hmax<int32_t>(glob);          // first match; -1 if none
```

---

## Workgroup and SLM Patterns

### Pattern 1: Independent Threads (No SLM)

Each thread works on a private slice of data. Use `range<D>` (not `nd_range`).

```cpp
q.submit([&](sycl::handler& cgh) {
    cgh.parallel_for<class MyKernel>(
        sycl::range<1>(num_rows),
        [=](sycl::id<1> idx) SYCL_ESIMD_KERNEL {
            using namespace sycl::ext::intel::esimd;
            const int row = (int)idx[0];
            simd<sycl::half, ROW_DIM> data =
                block_load<sycl::half, ROW_DIM>(input + row * ROW_DIM);
            // ... process ...
            block_store<sycl::half, ROW_DIM>(output + row * ROW_DIM, result);
        });
});
```

### Pattern 2: Workgroup Reduction via SLM

Each workgroup computes one output element; GROUP_SIZE threads split the K dimension.

```cpp
q.submit([&](sycl::handler& cgh) {
    cgh.parallel_for<class ReduceKernel>(
        sycl::nd_range<1>(num_outputs * GROUP_SIZE, GROUP_SIZE),
        [=](sycl::nd_item<1> item) SYCL_ESIMD_KERNEL {
            using namespace sycl::ext::intel::esimd;

            const int group_id = (int)item.get_group(0);
            const int local_id = (int)item.get_local_id(0);

            slm_init<GROUP_SIZE * sizeof(float)>();

            // Each thread computes partial result over its K slice
            float partial = compute_partial(group_id, local_id);

            // Store partial to SLM
            slm_block_store<float, 1>(
                local_id * sizeof(float), simd<float, 1>(partial));
            barrier();

            // Thread 0 reduces and writes output
            if (local_id == 0) {
                simd<float, GROUP_SIZE> parts =
                    slm_block_load<float, GROUP_SIZE>(0);
                float result = sycl::ext::intel::esimd::detail::sum<
                    float, float, GROUP_SIZE>(parts);
                output[group_id] = sycl::half(result);
            }
        });
});
```

**Guidelines**:
- `GROUP_SIZE` 4–8 is optimal; larger sizes increase SLM pressure and barrier overhead
- `slm_init` must be called unconditionally inside the kernel body
- `barrier()` must be reached by all GROUP_SIZE threads — never inside a conditional

---

## Register File vs SLM

When working data fits in registers (≤ ~2 KB), prefer registers over SLM.

| | Register file | SLM |
|--|--------------|-----|
| Latency | ~1 cycle | ~30 cycles |
| Requires `nd_range` | No | Yes |
| Requires `barrier()` | No | Yes |
| Per-thread budget | 4 KB | 128 KB / sub-slice (shared) |
| GROUP_SIZE overhead | None | Yes |

```cpp
// BAD: SLM for per-thread data (30× slower than needed)
slm_init<1024>();
slm_block_store<sycl::half, 32>(offset, chunk);
simd<sycl::half, 32> v = slm_block_load<sycl::half, 32>(offset);

// GOOD: register file (1 cycle, simpler kernel)
simd<sycl::half, 512> data;                         // 1 KB, 25% of 4 KB budget
data.template select<32, 1>(chunk_idx * 32) = chunk;
simd<sycl::half, 32> v = data.template select<32, 1>(chunk_idx * 32);
```

**SLM is still the right choice** for shared data across threads in a workgroup (e.g., K-dimension reduction where GROUP_SIZE > 1). The rule is: SLM for **inter-thread** sharing, registers for **intra-thread** working data.

---

## Loop Optimization

### #pragma unroll

Add to every loop with a compile-time iteration count:

```cpp
#pragma unroll
for (int i = 0; i < STATIC_N; i++) { ... }  // fully unrolled

#pragma unroll 4
for (int i = 0; i < large_static_n; i++) { ... }  // unrolled 4 at a time
```

Without `#pragma unroll`, the compiler may generate a loop counter and branch, preventing vectorization and adding overhead.

### Hoist Loop-Invariant Work

```cpp
// BAD: recompute per iteration
for (int i = 0; i < N; i++) {
    float scale = 1.f / std::sqrt((float)HEAD_DIM);  // recomputed each time
    results[i] = dot(q[i], k) * scale;
}

// GOOD: hoist
const float scale = 1.f / std::sqrt((float)HEAD_DIM);
for (int i = 0; i < N; i++) results[i] = dot(q[i], k) * scale;
```

---

## Hybrid Scalar/Vector Strategy

For kernels with boundary cases (first/last element, partial tiles), use vectorized code for the hot path and simple scalar code for edges:

```cpp
for (int i = 0; i < num_outputs; i++) {
    int start = i * STRIDE - 1;
    int end   = start + WINDOW;

    if (start < 0 || end > num_elems) {
        // Boundary: correct scalar path
        start = std::max(0, start);
        end   = std::min(num_elems, end);
        float mx = -INFINITY;
        for (int j = start; j < end; j++) mx = std::max(mx, (float)input[j]);
        output[i] = sycl::half(mx);
    } else {
        // Hot path: vectorized (handles majority of iterations)
        simd<sycl::half, WINDOW> v =
            block_load<sycl::half, WINDOW>(input + start,
                                            properties{alignment<2>});
        output[i] = sycl::half(hmax<float>(convert<float>(v)));
    }
}
```

**Benefit**: avoids complex vectorized boundary logic, keeps the hot path clean, easy to debug.
**Measured**: 1.31 ms (scalar only) → 0.808 ms (hybrid) = 1.62× speedup.

For kernels where offsets can be clamped safely (e.g., max pooling where reading duplicate elements is harmless), prefer **branchless offset clamping** over hybrid:

```cpp
// Fully branchless — no hot/cold path split needed
simd<int, BS> offs = min(max(computed_offs, 0), max_elem);
simd<T, BS> vals   = gather<T, BS>(ptr, simd<uint32_t, BS>(offs) * sizeof(T));
// max(vals, running_max) — duplicate reads are harmless for max
```

---

## Testing New Kernels

Every new kernel must have both an **accuracy test** and a **performance test**. Follow these rules:

### Rules

| Rule | Detail |
|------|--------|
| **Always add both tests** | Accuracy test + performance test for every new kernel |
| **Random input** | Use `rand()` / `srand(42)` for both tests — never all-zeros or trivially structured data |
| **Accuracy: small input** | Use a small input size to keep the CPU reference fast |
| **Accuracy: run once** | Run kernel once, run reference once, compare all outputs |
| **Accuracy: absolute + relative error** | Compute both `abs_err = |out - ref|` and `rel_err = abs_err / (|ref| + eps)`. Report both. PASS requires both below threshold (e.g. abs < 1e-2 **or** rel < 1e-3). Pure absolute error gives false FAILs for large-magnitude outputs and false PASSes for small-magnitude ones. |
| **Accuracy: print worst elements** | For the element with `max_abs_err` and the element with `max_rel_err`, print the index, `out` value, and `ref` value. This immediately shows whether errors are isolated outliers or systematic, and reveals near-zero reference values that inflate relative error. |
| **Perf: large input** | Use a larger input size to saturate memory/compute |
| **Perf: 5 runs, best result** | Run kernel 5 times, report the **minimum** (best) time — no warmup needed |
| **Perf: report FLOPS + BW** | Compute and print both TFLOPS and GB/s where applicable |

### Accuracy Test Pattern

```cpp
// Small input — fast CPU reference
constexpr int ACC_LEN = 1024;  // smaller than perf test

half* in_host  = sycl::malloc_host<half>(ACC_LEN, q);
half* out_host = sycl::malloc_host<half>(ACC_LEN, q);
half* in_dev   = sycl::malloc_device<half>(ACC_LEN, q);
half* out_dev  = sycl::malloc_device<half>(ACC_LEN, q);

// Random input
srand(42);
for (int i = 0; i < ACC_LEN; i++)
    in_host[i] = half(((float)rand() / RAND_MAX) * 2.f - 1.f);

q.memcpy(in_dev, in_host, ACC_LEN * sizeof(half)).wait();

// Run kernel once
my_kernel(q, in_dev, out_dev, ACC_LEN).wait();
q.memcpy(out_host, out_dev, ACC_LEN * sizeof(half)).wait();

// CPU reference once — check both absolute and relative error, record worst elements
double max_abs_err = 0.0, max_rel_err = 0.0;
int    max_abs_idx = 0,   max_rel_idx = 0;
double max_abs_ref = 0.0, max_rel_ref = 0.0;
double max_abs_out = 0.0, max_rel_out = 0.0;
for (int i = 0; i < ACC_LEN; i++) {
    float ref = cpu_reference(in_host, i);
    double abs_err = std::abs((double)out_host[i] - (double)ref);
    double rel_err = abs_err / (std::abs((double)ref) + 1e-6);
    if (abs_err > max_abs_err) {
        max_abs_err = abs_err; max_abs_idx = i;
        max_abs_ref = ref;     max_abs_out = (double)out_host[i];
    }
    if (rel_err > max_rel_err) {
        max_rel_err = rel_err; max_rel_idx = i;
        max_rel_ref = ref;     max_rel_out = (double)out_host[i];
    }
}
// PASS if either abs OR rel is within threshold (handles large/small magnitude outputs)
bool pass = max_abs_err < 1e-2 || max_rel_err < 1e-3;
printf("  accuracy: max_abs_err=%.6f (idx=%d out=%.6f ref=%.6f)"
       "  max_rel_err=%.6f (idx=%d out=%.6f ref=%.6f)  %s\n",
       max_abs_err, max_abs_idx, max_abs_out, max_abs_ref,
       max_rel_err, max_rel_idx, max_rel_out, max_rel_ref,
       pass ? "PASS" : "FAIL");
```

### Performance Test Pattern

```cpp
// Large input — saturate memory/compute
constexpr int PERF_LEN = 1 << 20;  // larger than accuracy test
constexpr int RUNS = 5;

// Random input (same srand(42) for reproducibility)
srand(42);
for (int i = 0; i < PERF_LEN; i++)
    in_host[i] = half(((float)rand() / RAND_MAX) * 2.f - 1.f);
q.memcpy(in_dev, in_host, PERF_LEN * sizeof(half)).wait();

// Run 5 times, keep best (use profiling events for kernel-only time)
double best_ns = std::numeric_limits<double>::max();
for (int r = 0; r < RUNS; r++) {
    sycl::event ev = my_kernel(q, in_dev, out_dev, PERF_LEN);
    ev.wait();
    double ns = (double)(
        ev.get_profiling_info<sycl::info::event_profiling::command_end>() -
        ev.get_profiling_info<sycl::info::event_profiling::command_start>());
    if (ns < best_ns) best_ns = ns;
}

double ms    = best_ns * 1e-6;
double bytes = (double)(PERF_LEN) * sizeof(half) * 2;  // read + write
double bw_gb = (bytes / (best_ns * 1e-9)) / 1e9;
double flops = 2.0 * PERF_LEN;  // adjust per kernel
double tflops= (flops / (best_ns * 1e-9)) / 1e12;

printf("  perf: %.3f ms   %.2f TFLOPS   %.1f GB/s\n", ms, tflops, bw_gb);
```

### Notes

- Use `sycl::property::queue::enable_profiling{}` on the queue for event-based timing.
- Event profiling measures **kernel-only** time, excluding host↔device memcpy — always prefer this over wall-clock timing.
- For kernels with no FLOP count (e.g., pure memory ops like topk), report only GB/s.
- The "best of 5" policy avoids GPU scheduler noise without requiring a separate warmup pass.
- Always check `max_err < threshold` appropriate to FP16 precision (typically `< 1e-2` for accumulating kernels).

---

## Common Performance Traps

### 1D Range with Manual Index Decode (10–35× slowdown)

Already covered in Kernel Design above. Most impactful mistake.

### Element-Wise Memory Access

```cpp
// BAD: 128 separate scalar loads
for (int i = 0; i < 128; i++) val[i] = ptr[offset + i];

// GOOD: one 256B block load
simd<sycl::half, 128> val = block_load<sycl::half, 128>(ptr + offset);
```

### Non-Aligned Offset

```cpp
// BAD: offset 7 is not aligned
simd<float, 32> v = block_load<float, 32>(ptr, 7 * sizeof(float));

// GOOD: offset must be a multiple of load width (or use alignment<> property)
simd<float, 32> v = block_load<float, 32>(ptr, 8 * sizeof(float));
```

### Excessive Synchronization

```cpp
// BAD: barrier per iteration
for (int i = 0; i < N; i++) { compute(i); barrier(); }

// GOOD: barrier once after all writes
for (int i = 0; i < N; i++) slm_block_store<float, 1>(i*4, partial[i]);
barrier();
// ... reduce ...
```

### Register Spilling

```cpp
// BAD: many large vectors live simultaneously → register spill to L1
simd<float, 256> a, b, c, d, e, f, g, h;  // 8 KB — exceeds 4 KB budget

// GOOD: reuse temporaries
simd<float, 128> tmp;
for (...) { tmp = load(); acc += process(tmp); }
```

### Branch Divergence

```cpp
// BAD: threads in same EU take different paths → serialized execution
if (local_id < 2) { path_a(); } else { path_b(); }

// GOOD: mask-based select (both paths execute, result selected per lane)
simd_mask<GROUP_SIZE> m = (simd<int,GROUP_SIZE>(0,1) < 2);
simd<T, N> result = merge(path_a_result, path_b_result, m);
```

### Uninitialized `simd<T, N>` Gives Garbage

`simd<T, N> v;` without initialization before element reads is undefined behavior. Always initialize before use — either broadcast-construct or seed with real data:

```cpp
simd<half, K> heap(-65504.f);           // broadcast init
simd<half, K> heap = block_load<...>(); // or initialize from load
```

### `alignment<2>{}` Parse Error in Template Context

```cpp
block_load<half, N>(ptr, properties{alignment<2>});   // GOOD
block_load<half, N>(ptr, properties{alignment<2>{}});  // BAD — parse error
```

### Inner Lambda Must Use `[[intel::sycl_explicit_simd]]`

Inner lambdas calling ESIMD intrinsics inside a `SYCL_ESIMD_KERNEL` must be marked `[[intel::sycl_explicit_simd]]`. Without it, the compiler errors: "SYCL kernel without 'sycl_explicit_simd' attribute can't call a function with this attribute". A `#define` macro is a valid fallback if the compiler doesn't support the attribute on lambdas.

### `sycl::min` / `sycl::max` Illegal in ESIMD Kernels

```cpp
// BAD — compile error in ESIMD
int n = sycl::min(a, b);

// GOOD
int n = std::min(a, b);
int n = (a < b) ? a : b;
```

### `reduce<T>(vec, std::plus<T>())` — Silent Zero Return

With `using namespace sycl::ext::intel::esimd` in scope, `reduce<float>(vec, std::plus<float>())` resolves to C++17 `std::reduce` (iterator overload) instead of the ESIMD intrinsic. It compiles without error and silently returns 0.

```cpp
// BAD — silently returns 0 (std::reduce, not esimd::reduce)
float s = reduce<float>(vec, std::plus<float>());

// GOOD — fully qualified, unambiguous
float s = sycl::ext::intel::esimd::detail::sum<float, float, N>(vec);
sycl::half s = sycl::ext::intel::esimd::detail::sum<sycl::half, sycl::half, N>(vec);
```

---

## Quick Reference

### Headers

```cpp
#include <sycl/sycl.hpp>
#include <sycl/ext/intel/esimd.hpp>
// For xmx::dpas:
#include <sycl/ext/intel/esimd/xmx/dpas.hpp>
// For lsc_load_2d:
#include <sycl/ext/intel/experimental/esimd/memory.hpp>

using namespace sycl::ext::intel::esimd;
using namespace sycl::ext::intel::experimental::esimd;
namespace xmx = sycl::ext::intel::esimd::xmx;
```

### Memory

```cpp
// Block load/store — address must be a multiple of 4 bytes by default
simd<T, N> v = block_load<T, N>(ptr + offset);                              // float/int: always OK
simd<T, N> v = block_load<T, N>(ptr + offset, properties{alignment<2>});    // half/bf16: odd offset
simd<T, N> v = block_load<T, N>(ptr + offset, properties{alignment<1>});    // uint8_t: any offset
block_store<T, N>(ptr + offset, v);
block_store<T, N>(ptr + offset, v, properties{alignment<2>});               // half/bf16: odd offset

// Gather/scatter (non-contiguous, byte offsets)
simd<T, N> v = gather<T, N>(base, simd<uint32_t, N>(byte_offs));
scatter<T, N>(base, simd<uint32_t, N>(byte_offs), v);

// SLM
slm_init<BYTES>();                                    // inside kernel, unconditional
simd<T, N> v = slm_block_load<T, N>(byte_offset);
slm_block_store<T, N>(byte_offset, v);
barrier();
```

### Vector Ops

```cpp
// Arithmetic (scalar broadcast automatic)
simd<T,N> c = a + b;   simd<T,N> c = a * scalar;   simd<T,N> fma = a * b + c;

// Comparison and merge
simd_mask<N> m = a > b;
simd<T, N> r = merge(a, b, m);   // r[i] = m[i] ? a[i] : b[i]

// Sub-range
simd<T, C> sub = vec.template select<C, S>(start);   // read
vec.template select<C, 1>(off) = other;               // write

// Convert
simd<T2, N> v2 = convert<T2>(v);
auto r = lvalue.template bit_cast_view<T2>().read();  // reinterpret (lvalue only!)

// Arithmetic sequence
simd<int, N> iota(0, 1);    // [0,1,...,N-1]
simd<int, N> stride(0, S);  // [0,S,2S,...]

// Reductions (CORRECT)
float s = sycl::ext::intel::esimd::detail::sum<float, float, N>(vec);
T mx = hmax<T>(vec);   T mn = hmin<T>(vec);   // horizontal max/min

// pack_mask + fbl (find first matching lane, N ≤ 32)
uint32_t pos = fbl(pack_mask(vec == val));
```

### Kernel Skeletons

```cpp
// Independent threads (no SLM)
q.submit([&](sycl::handler& cgh) {
    cgh.parallel_for<class K>(sycl::range<D>(...),
        [=](sycl::id<D> idx) SYCL_ESIMD_KERNEL {
            using namespace sycl::ext::intel::esimd;
            // ...
        });
});

// Workgroup reduction (SLM)
q.submit([&](sycl::handler& cgh) {
    cgh.parallel_for<class K>(sycl::nd_range<1>(N * G, G),
        [=](sycl::nd_item<1> item) SYCL_ESIMD_KERNEL {
            using namespace sycl::ext::intel::esimd;
            const int gid = item.get_group(0);
            const int lid = item.get_local_id(0);
            slm_init<G * sizeof(float)>();
            float partial = compute(gid, lid);
            slm_block_store<float,1>(lid*4, simd<float,1>(partial));
            barrier();
            if (lid == 0) {
                auto parts = slm_block_load<float, G>(0);
                output[gid] = sycl::half(
                    sycl::ext::intel::esimd::detail::sum<float,float,G>(parts));
            }
        });
});
```

### Compilation

```bash
# Release
icpx -fsycl -O3 kernel.cpp -o kernel

# Debug
icpx -fsycl -g -O0 kernel.cpp -o kernel_debug

# List available SYCL devices
sycl-ls
```

**Register file and large GRF mode**:

Intel GPU threads have a fixed register file per thread:
- **Xe1** (Arc Alchemist, Gen12 iGPU): 128 GRF × 32B = **4 KB** per thread
- **Xe2+** (Arc Battlemage, Lunar Lake+): 256 GRF × 32B = **8 KB** per thread

If a kernel uses too many `simd<T, N>` variables and exceeds this budget, the compiler spills registers to L1 cache, causing severe performance degradation. To double the register file size per thread, enable **large GRF mode**:

```bash
# Enable large GRF mode (doubles register file: 4 KB → 8 KB on Xe1, 8 KB → 16 KB on Xe2+)
# Note: doubling GRF halves the number of threads that can run simultaneously per EU,
# so use only when register pressure is the bottleneck.
icpx -fsycl -O3 kernel.cpp -Xs "-doubleGRF" -o kernel
```

**When to use `-doubleGRF`**:
- Kernel has many large `simd<T, N>` live simultaneously (e.g., `q_tiles[MS][KS]` arrays in XMX kernels)
- Compiler warns about register spilling, or profiler shows high L1 traffic
- Performance improves after enabling it (verify — not always beneficial due to reduced thread occupancy)

**Rule of thumb**: on Xe1, if total live FP16 elements across all `simd` variables in the hot loop exceeds ~1500, consider `-doubleGRF`.

---

