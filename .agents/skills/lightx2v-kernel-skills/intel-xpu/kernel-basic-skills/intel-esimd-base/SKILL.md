---
name: intel-esimd-base
description: >-
  Write or debug Intel ESIMD kernels using explicit simd vectors, memory operations, reductions and workgroup synchronization. Use for ESIMD implementation questions; use intel-gpu-hw-info for SKU facts and intel-gpu-kernel-opt for a profiling-led optimization task.
---

# Intel ESIMD Base Skill

Implement and debug explicit-SIMD kernels using the target SDK and the selected algorithm's memory and numerical contracts. Read the detailed patterns only when that part of the implementation is involved.

---

## Table of Contents

1. [What Is ESIMD](#what-is-esimd)
2. [Hardware Characteristics](#hardware-characteristics)
3. [Optimization Priority Checklist](#optimization-priority-checklist)
4. [Implementation Guide](#implementation-guide)
5. [Testing New Kernels](#testing-new-kernels)
6. [Common Performance Traps](#common-performance-traps)
7. [Quick Reference](#quick-reference)

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

### Device and register mode

Record the device SKU, compiler version and GRF mode before choosing vector widths or workgroup sizes. Intel's [architecture table](https://www.intel.com/content/www/us/en/docs/oneapi/optimization-guide-gpu/2025-2/intel-xe-gpu-architecture.html) gives these examples:

| Device / mode | GRF per hardware thread | Raw capacity |
|---|---|---|
| Arc A770 / Alchemist | 128 × 32 B | 4 KiB |
| Arc B580 / Xe2 regular | 128 × 64 B | 8 KiB |
| Arc B580 / Xe2 large GRF | 256 × 64 B | 16 KiB |
| Lunar Lake / Xe2 regular or large GRF | 128 or 256 × 64 B | 8 or 16 KiB |

Raw capacity includes addresses, accumulators and compiler temporaries; it is not all available for one simd array. A 512-element FP16 vector uses 1 KiB, but that alone does not establish that the kernel will avoid spilling. Check compiler spill reports and benchmark the occupancy tradeoff before choosing large GRF mode.

GRF width is not a universal instruction-throughput rule. Inspect generated code for the chosen vector length and data type. Query device limits and use the [hardware reference](../intel-gpu-hw-info/SKILL.md) for the relevant SKU; do not infer a portable resident-thread count by multiplying max_compute_units by a fixed factor.

---

## Optimization Priority Checklist

Use this as a starting order, then follow measurements. The impact ranges below are historical workload observations, not guaranteed gains on a new kernel:

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

## Implementation Guide

Read the patterns for the part of the kernel being implemented:

| Topic and reference | Key constraints |
| --- | --- |
| [Kernel design](references/programming-patterns.md#kernel-design) | Choose range dimensionality, work per thread and compile-time parameters from the actual tensor layout. Measure layout changes before generalizing an old speedup. |
| [Memory access](references/programming-patterns.md#memory-access-patterns) | Use block access for contiguous storage and gather/scatter for irregular access. Verify byte offsets, bounds, alignment and SLM synchronization. |
| [Data types](references/programming-patterns.md#data-types) | Preserve the numerical contract when choosing storage and accumulation types; lower precision is a measured tradeoff. |
| [Vectorization](references/programming-patterns.md#vectorization-techniques) | Keep arithmetic in simd vectors, initialize every read lane, and verify reduction operation types against the installed SDK. |
| [Workgroups and SLM](references/programming-patterns.md#workgroup-and-slm-patterns) | Use independent work-items when sharing is unnecessary. For shared reductions, every participating work-item must obey the same synchronization protocol. |
| [Registers and SLM](references/programming-patterns.md#register-file-vs-slm) | Budget all live registers and compiler temporaries. Use SLM for actual sharing or measured register-pressure relief, with its synchronization cost included. |
| [Loop optimization](references/programming-patterns.md#loop-optimization) | Hoist invariant work and choose unrolling from instruction count, register use and timing; full unrolling is not automatically best. |
| [Hybrid scalar/vector strategy](references/programming-patterns.md#hybrid-scalarvector-strategy) | Use vector operations for bulk work and a scalar tail or control phase where the algorithm benefits; verify selection and tie semantics. |

## Testing New Kernels

Check a CPU reference on small inputs before interpreting performance. Reject non-finite results, apply per-element absolute/relative tolerances, test zero/near-zero references and propagate failures to the process exit status. Performance runs are required for performance claims, not documentation review.

Use the [validation templates](references/validation.md#testing-new-kernels) when implementing correctness checks and benchmarks.

## Common Performance Traps

Use compiler diagnostics and a minimal reproducer for SDK-sensitive intrinsics. Preserve the distinction between a historical failure and a general API rule.

Consult [known compiler and performance pitfalls](references/troubleshooting.md#common-performance-traps) when diagnosing these failures.

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
// USM block access, no cache hints; check the final address (base + offset).
simd<T, N> v = block_load<T, N>(ptr + offset);  // default: 4B for sizeof(T) <= 4; 8B for sizeof(T) == 8
simd<half, N> h = block_load<half, N>(half_ptr + offset, properties{alignment<2>});
simd<uint8_t, N> b = block_load<uint8_t, N>(byte_ptr + offset, properties{alignment<1>});
block_store<T, N>(ptr + offset, v);            // unpredicated default: 16B alignment
block_store<half, N>(half_ptr + offset, h, properties{alignment<2>});

// Gather/scatter (non-contiguous, byte offsets)
simd<T, N> v = gather<T, N>(base, simd<uint32_t, N>(byte_offs));
scatter<T, N>(base, simd<uint32_t, N>(byte_offs), v);

// SLM
slm_init<BYTES>();                                    // inside kernel, unconditional
simd<T, N> v = slm_block_load<T, N>(byte_offset);
slm_block_store<T, N>(byte_offset, v);
barrier();
```

`alignment<>` declares guaranteed address alignment; it does not realign memory. See [Block Load / Store](references/programming-patterns.md#block-load--store-contiguous-access) for examples and overload restrictions.

### Vector Ops

```cpp
// Arithmetic (scalar broadcast automatic)
simd<T,N> c = a + b;   simd<T,N> c = a * scalar;   simd<T,N> fma = a * b + c;

// Comparison and merge
simd_mask<N> m = a > b;
simd<T, N> r = b;
r.merge(a, m);                  // r[i] = m[i] ? a[i] : b[i]

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

On B580 and Lunar Lake, regular/large GRF modes provide 8/16 KiB per hardware thread; see [Hardware Characteristics](#hardware-characteristics). Extra live vectors can spill to scratch memory. A larger register allocation can reduce occupancy, so compare both modes using the same kernel and input.

```bash
# BMG AOT example; choose the target and supported flags for the installed SDK.
icpx -fsycl -fsycl-targets=spir64_gen -O3 kernel.cpp \
    -Xs "-device bmg -options -doubleGRF" -o kernel
```

**When to use `-doubleGRF`**:
- Kernel has many large `simd<T, N>` live simultaneously (e.g., `q_tiles[MS][KS]` arrays in XMX kernels)
- Compiler warns about register spilling, or profiler shows high L1 traffic
- Performance improves after enabling it (verify — not always beneficial due to reduced thread occupancy)

Do not assume every Xe generation supports the same GRF modes or compiler flags.

---
