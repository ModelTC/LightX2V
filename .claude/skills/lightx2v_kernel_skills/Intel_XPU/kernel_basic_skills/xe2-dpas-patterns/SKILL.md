---
name: xe2-dpas-patterns
description: Use this skill when writing, loading operands for, or storing results from XMX DPAS instructions on Intel Xe2 (Lunar Lake/LNL, Battlemage/BMG) GPU using SYCL ESIMD. Xe2 is the GPU architecture; LNL and BMG are product names. Covers all four DPAS operand load patterns (lsc_load_2d, lsc_gather, VNNI packing), scatter/store write-back, Usage 1 vs Usage 2 orientation, and the SOA property of lsc_gather. Applicable to any kernel using DPAS: GEMM, attention, convolution, etc.
---

# Xe2 DPAS Patterns Skill

Reference for loading and storing DPAS operands on Intel Xe2 (Lunar Lake/LNL, Battlemage/BMG) via SYCL ESIMD.
All patterns are validated in `assets/fp16_dpas_ult.cpp` (4 test cases, all PASS).

---

## DPAS Register Contract (FP16, XE2)

```cpp
// xmx::dpas<RC, SD, Tacc, Tc, Tb, Ta>(acc, b_tile, a_tile)
// XE2: RC=8 fixed, SD=8 fixed (32-bit systolic depth)
// FP16: SD=8 systolic steps × 2 fp16/step = 16 K-elements per call
xmx::dpas<8, 8, sycl::half, sycl::half, sycl::half, sycl::half>(acc, b_tile, a_tile)
```

| Register | Type | Size | Layout | Role |
|----------|------|------|--------|------|
| `a_tile` | `simd<half, 8*16>` | 128 half | `[m*16+k]` row-major | 8 M-rows × 16 K-cols |
| `b_tile` | `simd<half, 16*16>` | 256 half | `uint32[k_pair*16+n]` VNNI | 16 K-elems × 16 N-cols |
| `acc`    | `simd<half, 8*16>` | 128 half | `[m*16+n]` row-major | 8 M-rows × 16 N-cols |

Result: `acc[m*16+n] += sum_k  a_tile[m*16+k] * b_tile_fp16(n,k)`

> **Arg order**: `dpas(acc, b_tile, a_tile)` — b_tile is 2nd, a_tile is 3rd. Easy to mix up.

---

## Two Usage Orientations

### Usage 1 — Standard (M=8, N=16, K=16)

```
A[M×K] → a_tile     B[N×K] or B_T[K×N] → b_tile
dpas(acc, b_tile, a_tile) → acc[m*16+n] = C[m,n]   (direct store)
```

### Usage 2 — Swapped (M=16, N=8, K=16)

Swap roles: A→b_tile (gather VNNI), B→a_tile (load_2d or gather).
Useful when N < 16 at the DPAS call level.

```
A[M×K] → b_tile (gather VNNI, 16 M-lanes)
B[N×K] → a_tile (load_2d or gather)
dpas(acc, b_tile=A_vnni, a_tile=B) → acc[n*16+m] = C[m,n]  (TRANSPOSED — needs scatter write-back)
```

---

## VNNI Layout (b_tile)

```
b_tile uint32[k_pair * 16 + n_col] = packed { fp16[n_col, k_pair*2], fp16[n_col, k_pair*2+1] }
  k_pair = 0..7  (outer, stride 16)
  n_col  = 0..15 (inner, stride 1)
```

---

## lsc_gather SOA Property (key insight)

`lsc_gather<T, NElts, DS, L1H, L2H, N>(ptr, offsets)` returns `simd<T, N*NElts>` in **SOA** layout:

```
result[elem * N + lane]  =  T at  ptr[ byte_offset[lane] + elem * sizeof(T) ]
```

This SOA property is what makes gather produce DPAS-ready layouts without register repack:
- For **b_tile** (B[N,K]): use N rows as lanes → `result[k_pair*16+n]` = VNNI
- For **a_tile** (B_T[K,N]): use K rows as lanes with fp16/u16 type → `result[e*16+k]` = N-outer a_tile

---

## Four Load Patterns

### Pattern 1: a_tile from A[M×K] — lsc_load_2d, no transform

```cpp
xesimd::config_2d_mem_access<sycl::half, 16/*BW=K*/, 8/*BH=M*/, 1> payA(
    A, K*2u-1u, M-1u, K*2u-1u, 0u, 0u);
simd<sycl::half, 8*16> a_tile = xesimd::lsc_load_2d<
    sycl::half, 16, 8, 1, false/*T*/, false/*VNNI*/, cached, cached>(payA);
// a_tile[m*16+k] = A[m, k]  ✓
```

### Pattern 2: b_tile from B_T[K×N] — lsc_load_2d VNNI=true

```cpp
xesimd::config_2d_mem_access<sycl::half, 16/*BW=N*/, 16/*BH=K*/, 1> payB(
    B_T, N*2u-1u, K-1u, N*2u-1u, 0u, 0u);
simd<sycl::half, 16*16> b_tile = xesimd::lsc_load_2d<
    sycl::half, 16, 16, 1, false, true/*VNNI*/, cached, cached>(payB);
// b_tile uint32[k_pair*16+n] = {B_T[k*2,n], B_T[k*2+1,n]}  VNNI ✓
```

### Pattern 3: b_tile from B[N×K] — lsc_gather u32/NElts=8, N-lanes

```cpp
// N=16 lanes (one per output N-column), NElts=8 k-pairs per lane (= 16 fp16 = K)
const uint32_t* B_u32 = reinterpret_cast<const uint32_t*>(B);
simd<uint32_t, 16> b_off;
for (int n = 0; n < 16; n++)
    b_off[n] = (uint32_t)(n_base + n) * K * 2u;  // byte offset to B[n, k_start]
simd<sycl::half, 16*16> b_tile;
b_tile.template bit_cast_view<uint32_t>() =
    xesimd::lsc_gather<uint32_t, 8,
        xesimd::lsc_data_size::u32,
        xesimd::cache_hint::cached, xesimd::cache_hint::cached,
        16, uint32_t>(B_u32, b_off);
// SOA: result_u32[k_pair*16+n] = {B[n,k+k_pair*2], B[n,k+k_pair*2+1]}  VNNI ✓
```

### Pattern 4: a_tile from B_T[K×N] — lsc_gather fp16/u16, K-lanes

`lsc_load_2d transpose=true` is **rejected for fp16** at compile time. Use fp16 gather instead:

```cpp
// K=16 lanes (one per K-row of B_T), NElts=8 fp16 per lane (= one full row = N elements)
// SOA: result[e*16+k] = B_T[k][e] = B[e,k]  → matches a_tile[n*16+k]=B[n,k] directly ✓
// WARNING: do NOT use lsc_gather<uint32_t,4> — uint32 packs {B[n_u32*2,k], B[n_u32*2+1,k]}
//   (different N rows, same k) which does NOT match a_tile uint32 layout (same N, adjacent k).
simd<uint32_t, 16> a_off;
for (int k = 0; k < K; k++)
    a_off[k] = (uint32_t)k * (uint32_t)N * 2u;  // byte offset to B_T[k, 0]
simd<sycl::half, 8*16> a_tile =
    xesimd::lsc_gather<sycl::half, 8,
        xesimd::lsc_data_size::u16,
        xesimd::cache_hint::cached, xesimd::cache_hint::cached,
        16, uint32_t>(B_T, a_off);
// a_tile[n*16+k] = B[n, k]  ✓  (no bit_cast_view, no register repack)
```

---

## Store / Write-back Patterns

### Usage 1 write-back — lsc_store_2d (acc is direct C[m,n])

```cpp
xesimd::lsc_store_2d<sycl::half, 16, 8,
    xesimd::cache_hint::write_back, xesimd::cache_hint::write_back>(
    C, N*2u-1u, M-1u, N*2u-1u, (uint32_t)n_start, (uint32_t)m_start, acc);
```

### Usage 2 write-back — lsc_scatter fp16/u16 (acc is transposed C[n,m])

`acc[ni*16+mj] = C[mj, ni]`. Scatter SOA: `data[e*16+lane]` → `ptr[offset[lane] + e*sizeof(half)]`.
Setting `offset[mj] = mj * N * sizeof(half)` and `data = acc` works directly — no repack needed.

```cpp
// SOA: data[e*16+mj] → C[mj, e]
// data[e*16+mj] = C[mj,e] = acc[e*16+mj]  → data = acc directly ✓
simd<uint32_t, 16> sc_off;
for (int mj = 0; mj < 16; mj++)
    sc_off[mj] = (uint32_t)(m_base + mj) * N_total * 2u + (uint32_t)n_base * 2u;
xesimd::lsc_scatter<sycl::half, 8,
    xesimd::lsc_data_size::u16,
    xesimd::cache_hint::write_back, xesimd::cache_hint::write_back,
    16, uint32_t>(C, sc_off, acc);
// acc passed as data directly — no register repack ✓
```

---

## Summary: No-Shuffle Rules

| Operation | Method | Shuffle needed? |
|-----------|--------|-----------------|
| a_tile ← A[M,K] | `lsc_load_2d` no-transform | none |
| b_tile ← B_T[K,N] | `lsc_load_2d` VNNI=true | none (hardware) |
| b_tile ← B[N,K] | `lsc_gather<u32,8,u32,N=16>` N-lanes | none (SOA=VNNI) |
| a_tile ← B_T[K,N] | `lsc_gather<half,8,u16,N=16>` K-lanes | none (SOA=N-outer) |
| store C (Usage 1) | `lsc_store_2d` | none |
| store C (Usage 2) | `lsc_scatter<half,8,u16,N=16>` M-lanes | none (SOA matches acc) |

**All six operations are shuffle-free when using the correct gather/scatter type and lane axis.**

---

## Asset

| File | Purpose |
|------|---------|
| `assets/fp16_dpas_ult.cpp` | Unit test for all 4 load patterns + scatter write-back. All cases PASS. |

Compile and run:
```bash
icpx fp16_dpas_ult.cpp -o fp16_dpas_ult.exe \
  -fsycl -fsycl-targets=spir64_gen \
  -Xs "-device bmg -options -doubleGRF"
powershell.exe -Command "& './fp16_dpas_ult.exe'"
```
