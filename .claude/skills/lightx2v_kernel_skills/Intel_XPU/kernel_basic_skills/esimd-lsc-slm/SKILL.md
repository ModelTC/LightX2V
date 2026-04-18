---
name: esimd-lsc-slm
description: >
  LSC Shared Local Memory (SLM) operations in Intel ESIMD. Use this skill when
  working with slm_init, slm_block_load, slm_block_store, lsc_slm_gather,
  lsc_slm_scatter, SLM layout design, barrier synchronization, named barriers,
  cooperative SLM loading, or any kernel that uses workgroup shared memory on
  Intel GPUs. Covers SLM size limits, bank conflicts, the lsc_slm_scatter
  transpose trick, and common pitfalls like forgetting slm_init or conditional
  barriers causing GPU hangs.
---

# ESIMD LSC SLM (Shared Local Memory) Operations

Complete guide to Shared Local Memory operations in Intel ESIMD: initialization, block load/store, gather/scatter, barriers, layout design, and advanced patterns like SLM-based transpose and named barrier pipelining.

**Version**: 1.0.0
**Last Updated**: 2026-03-12

---

## Table of Contents

1. [SLM Initialization](#slm-initialization)
2. [SLM Block Load / Store](#slm-block-load--store)
3. [LSC SLM Gather / Scatter](#lsc-slm-gather--scatter)
4. [SLM Size Limits](#slm-size-limits)
5. [Barrier Synchronization](#barrier-synchronization)
6. [Named Barriers](#named-barriers)
7. [SLM Layout Design](#slm-layout-design)
8. [Common Patterns](#common-patterns)
9. [The lsc_slm_scatter Transpose Trick](#the-lsc_slm_scatter-transpose-trick)
10. [Common Pitfalls](#common-pitfalls)

---

## SLM Initialization

`slm_init<BYTES>()` must be the **first statement** in any ESIMD kernel that uses SLM. It declares the SLM allocation size for the workgroup.

```cpp
cgh.parallel_for<KernelName>(nd_rng,
    [=](sycl::nd_item<3> item) SYCL_ESIMD_KERNEL {
        using namespace sycl::ext::intel::esimd;

        slm_init<65536>();  // MUST be first — allocate 64 KB SLM

        // ... rest of kernel ...
    });
```

**Rules**:
- Must be called **unconditionally** at the top of the kernel
- The template parameter is in **bytes**
- Cannot be called conditionally or inside a branch
- All threads in the workgroup share the same SLM allocation

---

## SLM Block Load / Store

Contiguous access to SLM using byte offsets.

```cpp
// Block load: read N elements of type T from SLM at byte_offset
simd<half, 128> data = slm_block_load<half, 128>(byte_offset);

// Block store: write N elements of type T to SLM at byte_offset
slm_block_store<half, 128>(byte_offset, data);
```

**Parameters**:
- `T` — element type (half, float, uint32_t, etc.)
- `N` — number of elements (must be power of 2, typically 8..512)
- `byte_offset` — offset into SLM in **bytes** (uint32_t)

**Example: Cooperative Q load into SLM**

```cpp
constexpr int Q_SLM_SIZE = 128 * 256 * sizeof(half);  // 128 Q rows x 256 D cols
constexpr int ROWS_PER_THREAD = 128 / NUM_THREADS;

slm_init<Q_SLM_SIZE>();

// Each thread loads its share of Q rows
int my_row = lid * ROWS_PER_THREAD;
#pragma unroll
for (int r = 0; r < ROWS_PER_THREAD; r++) {
    simd<half, 256> q_row = block_load<half, 256>(Q_ptr + (my_row + r) * 256);
    slm_block_store<half, 256>(
        (my_row + r) * 256 * (int)sizeof(half), q_row);
}
barrier();  // Wait for all threads to finish loading
```

---

## LSC SLM Gather / Scatter

Non-contiguous SLM access where each SIMD lane accesses a different SLM location.

### lsc_slm_gather

```cpp
// Gather NElts elements per address, VL addresses
// Template: <T, NElts, DS, VL>
simd<uint32_t, 16> offsets = ...;  // byte offsets into SLM
auto data = lsc_slm_gather<uint32_t, 1, lsc_data_size::u32, 16>(offsets);
```

### lsc_slm_scatter

```cpp
// Scatter NElts elements per address, VL addresses
simd<uint32_t, 16> offsets = ...;  // byte offsets into SLM
simd<uint32_t, 32> data = ...;     // NElts * VL elements (element-major layout)
lsc_slm_scatter<uint32_t, 2, lsc_data_size::u32, 16>(offsets, data);
```

### NElts > 1: Element-Major Data Layout

When `NElts > 1`, the data vector must be in **element-major** order, NOT address-major:

```cpp
// For NElts=2, VL=16:
// data[0..15]  = element 0 for all 16 addresses
// data[16..31] = element 1 for all 16 addresses

// CORRECT: element-major
simd<uint32_t, 32> data;
data.select<16, 1>(0)  = packed_elem0;  // elem 0 for all addrs
data.select<16, 1>(16) = packed_elem1;  // elem 1 for all addrs

// WRONG: address-major interleave (produces systematic errors)
data.select<16, 2>(0) = packed_elem0;   // DO NOT USE
data.select<16, 2>(1) = packed_elem1;   // DO NOT USE
```

---

## SLM Size Limits

| Platform | SLM per XE Core | Notes |
|----------|-----------------|-------|
| Xe2 (BMG / Battlemage) | 64 KB | Shared across all WGs on the same XE core |
| Xe3 (PTL / Panther Lake) | 64 KB | iGPU, shared across all WGs on the same XE core |
| Xe HPC (PVC) | 128 KB | Data center GPU |

**SLM is shared across all workgroups** executing on the same XE core. Larger SLM allocations reduce occupancy (fewer WGs can run concurrently per XE core).

**SLM budget example (HD=256 SDP kernel, 128 KB total)**:
```
0x00000 - 0x0FFFF:  Q_SLM    (64 KB)  — Q tile [128 Q x 256 D] fp16
0x10000 - 0x17FFF:  S_SLM    (32 KB)  — Scores [128 Q x 128 KV] fp16
0x18000 - 0x18FFF:  MAX_SLM  (4 KB)   — Cross-SG max reduction
0x19000 - 0x19FFF:  SUM_SLM  (4 KB)   — Cross-SG sum reduction
```

---

## Barrier Synchronization

### Full Barrier

```cpp
barrier();  // All threads in the workgroup must reach this point
```

**Rules**:
- **Every thread in the workgroup MUST hit the barrier** — no conditional skipping
- Typically used in pairs: one after SLM writes, one before next iteration's SLM writes
- Each barrier has overhead; minimize barrier count per iteration

**Standard two-barrier pattern**:
```cpp
for (int t = 0; t < kv_len; t += VS) {
    // Phase 1: Cooperative load into SLM
    slm_block_store<half, HD>(..., data);
    barrier();  // Barrier 1: Wait for SLM fill

    // Phase 2: Compute using SLM data
    auto tile = slm_block_load<half, HD>(...);
    // ... compute ...

    barrier();  // Barrier 2: Guard SLM before next iteration
}
```

---

## Named Barriers

Named barriers (`nbarrier`) enable fine-grained synchronization where different thread subsets synchronize independently, and support split arrive/wait for latency hiding.

```cpp
// Declare named barrier
nbarrier nb;
nb.init_nbarrier(barrier_id, nbarrier_role::producer_consumer, num_threads);

// Split arrive/wait pattern
nb.arrive();          // Signal: "I'm done writing"
// ... do independent work (V loads, prefetch, compensation) ...
nb.wait();            // Block: "Wait for all others to signal"
```

**Use case: Split barrier B in SDP kernel**

The arrive/wait window spans V loads + K prefetch + deferred compensation = 100+ cycles of useful work between arrive and wait:

```cpp
// After S SLM writes complete:
nb_B.arrive();                    // Signal S writes done

// Independent work while waiting for all S writes:
auto v_tile = lsc_load_2d<...>(); // V loads from global
lsc_prefetch_2d<...>();           // Next K prefetch
a_tile *= delta;                  // Deferred compensation

nb_B.wait();                      // Now safe to read S from SLM
auto s_tile = slm_block_load<...>();
```

See `xe2-nbarrier-pipelining` skill for full named barrier patterns.

---

## SLM Layout Design

### Contiguous Layout

Each row stored contiguously. Best for `slm_block_load` / `slm_block_store`.

```
Row 0: [elem0, elem1, ..., elemN-1]  @ offset 0
Row 1: [elem0, elem1, ..., elemN-1]  @ offset N*sizeof(T)
...
```

### Strided Layout

Elements interleaved by stride. Can cause bank conflicts but enables different access patterns.

### Avoiding Bank Conflicts

SLM is organized in banks (typically 32 banks on Intel GPUs). Bank conflicts occur when multiple threads in the same subgroup access the same bank but different addresses.

**Tips**:
- Ensure consecutive threads access consecutive banks (contiguous layout naturally avoids conflicts)
- If multiple threads read the same address, it is a broadcast (no conflict)
- Padding rows by 1-2 elements can break stride conflicts in certain patterns

---

## Common Patterns

### 1. Cooperative Load into SLM

All threads in a workgroup cooperatively load a shared tile from global memory into SLM, then each thread reads its portion.

```cpp
constexpr int TILE_ROWS = 128;
constexpr int TILE_COLS = 128;
constexpr int ROWS_PER_THREAD = TILE_ROWS / NUM_THREADS;

for (int t = 0; t < seq_len; t += TILE_ROWS) {
    // Each thread loads ROWS_PER_THREAD rows
    #pragma unroll
    for (int r = 0; r < ROWS_PER_THREAD; r++) {
        int row = lid * ROWS_PER_THREAD + r;
        simd<half, TILE_COLS> row_data = block_load<half, TILE_COLS>(
            global_ptr + (t + row) * stride);
        slm_block_store<half, TILE_COLS>(
            row * TILE_COLS * (int)sizeof(half), row_data);
    }
    barrier();

    // All threads can now read any part of the SLM tile
    simd<half, TILE_COLS> my_data = slm_block_load<half, TILE_COLS>(
        my_row * TILE_COLS * (int)sizeof(half));
    // ... compute ...
    barrier();
}
```

### 2. K-Split Reduction via SLM

Multiple threads compute partial sums, then reduce through SLM.

```cpp
// Each thread writes its partial result to SLM
slm_block_store<float, N>(lid * N * sizeof(float), partial_sum);
barrier();

// Thread 0 reduces all partial sums
if (lid == 0) {
    simd<float, N> total(0.f);
    for (int i = 0; i < NUM_THREADS; i++) {
        total += slm_block_load<float, N>(i * N * sizeof(float));
    }
    block_store<float, N>(output_ptr, total);
}
```

### 3. Transpose via SLM

Write data in one layout, read in another. Useful for transposing small matrices.

```cpp
// Write row-major into SLM
for (int r = 0; r < ROWS; r++)
    slm_block_store<half, COLS>(r * COLS * sizeof(half), row_data[r]);
barrier();

// Read column-major from SLM (transposed)
for (int c = 0; c < COLS; c++) {
    simd<uint32_t, ROWS> col_offsets;
    for (int r = 0; r < ROWS; r++)
        col_offsets[r] = (r * COLS + c) * sizeof(half);
    auto col = lsc_slm_gather<half, 1, lsc_data_size::u16, ROWS>(col_offsets);
}
```

---

## The lsc_slm_scatter Transpose Trick

From the HD=256 SDP kernel: replace ~270 register transpose `mov` instructions with `lsc_slm_scatter`, gaining +1.7 TFLOPS (84 to 88 TFLOPS).

### Problem

QK GEMM produces scores in `[KV=16][Q=16]` layout per thread. VS GEMM needs `[Q][KV]` layout. Register transpose generates hundreds of `mov(4|M0)` instructions in ISA.

### Solution: Pack + Scatter

1. **Pack** adjacent KV rows into `uint32_t` via shift+OR:
   ```cpp
   simd<uint32_t, 16> packed = low_kv_u16 | (high_kv_u16 << 16);
   ```

2. **Scatter** to transposed SLM positions:
   ```cpp
   // Compute Q-major addresses
   simd<uint32_t, 16> q_addrs = ...;  // byte offsets for 16 Q positions

   // Element-major data: elem0 for all addrs, then elem1 for all addrs
   simd<uint32_t, 32> data;
   data.select<16, 1>(0)  = packed0;  // first u32 per Q addr
   data.select<16, 1>(16) = packed1;  // second u32 per Q addr

   lsc_slm_scatter<uint32_t, 2, lsc_data_size::u32, 16>(q_addrs, data);
   ```

3. **Read** from SLM with `slm_block_load` — layout is now Q-major KV-fast.

### Key Details

- 8 scatter calls per thread (2 q_pairs x 4 kv_groups)
- NElts=2 writes 2 uint32 per address = 4 FP16 KV values per call
- Data **must** be element-major (see NElts layout section above)
- Address-major interleave produces ~0.12 relative RMS error (systematic bug)

### What Did NOT Work

| Approach | TFLOPS | Why it failed |
|----------|--------|---------------|
| `lsc_slm_gather` (block_store + gather read) | 73 (-12.9) | SLM gathers too expensive |
| SLM scatter NElts=1 (64 D16 messages) | 69 | Per-element scatter BW too low |
| VNNI interleave (stride-2 uint32 pairs) | 82.7 (neutral) | Compiler decomposed stride-8 gather into more movs |

---

## Common Pitfalls

### 1. Forgetting `slm_init`

```cpp
// BAD: kernel uses SLM but no slm_init — undefined behavior or crash
[=](sycl::nd_item<3> item) SYCL_ESIMD_KERNEL {
    slm_block_store<half, 128>(0, data);  // UB: SLM not initialized
};

// GOOD: always init first
[=](sycl::nd_item<3> item) SYCL_ESIMD_KERNEL {
    slm_init<32768>();  // First statement
    slm_block_store<half, 128>(0, data);
};
```

### 2. Conditional Barriers Causing GPU Hang

```cpp
// BAD: some threads skip barrier — GPU deadlock
if (lid < 8) {
    // ... do work ...
    barrier();  // ONLY threads 0-7 hit this — HANG
}

// GOOD: all threads hit barrier unconditionally
if (lid < 8) {
    // ... do work ...
}
barrier();  // ALL threads hit this
```

A conditional barrier is the most common cause of GPU hangs in ESIMD kernels. The GPU will wait indefinitely for threads that never arrive.

### 3. Byte Offset vs Element Offset Confusion

```cpp
// BAD: element offset where byte offset is expected
slm_block_store<half, 128>(row * 128, data);  // WRONG: row*128 elements

// GOOD: byte offset
slm_block_store<half, 128>(row * 128 * (int)sizeof(half), data);  // CORRECT: bytes
```

All SLM operations use **byte offsets**. Multiply element indices by `sizeof(T)`.

### 4. SLM Size Exceeding Hardware Limit

```cpp
// BAD: 128 KB on BMG with other WGs needing SLM — reduced occupancy
slm_init<131072>();  // 128 KB — only 1 WG per XE core on BMG

// BETTER: minimize SLM footprint for higher occupancy
slm_init<65536>();   // 64 KB — allows 2 WGs per XE core (if SLM is the bottleneck)
```

### 5. Reading SLM Before Barrier After Write

```cpp
// BAD: race condition — may read stale data
slm_block_store<half, 128>(0, my_data);
auto other_data = slm_block_load<half, 128>(other_offset);  // RACE!

// GOOD: barrier between write and cross-thread read
slm_block_store<half, 128>(0, my_data);
barrier();
auto other_data = slm_block_load<half, 128>(other_offset);  // SAFE
```

### 6. lsc_slm_scatter Element-Major Layout Violation

See the [NElts > 1 section](#nelts--1-element-major-data-layout) above. Using address-major interleave with NElts > 1 produces silent data corruption (systematic numerical errors, no crash).

---

## Related Skills

| Skill | Relevance |
|-------|-----------|
| `intel-esimd-base` | Foundational ESIMD programming including SLM overview |
| `esimd-lsc-2d-gather-scatter` | Global memory LSC operations (2D block load, gather/scatter) |
| `xe2-nbarrier-pipelining` | Named barrier arrive/wait patterns for latency hiding |
| `xe2-sdp-hd256` | HD=256 SDP kernel using lsc_slm_scatter for S transpose |
| `intel-esimd-attn-gemm` | Attention GEMM kernels with cooperative SLM K+V caching |
| `intel-gpu-kernel-opt` | General optimization including SLM bank conflict analysis |
