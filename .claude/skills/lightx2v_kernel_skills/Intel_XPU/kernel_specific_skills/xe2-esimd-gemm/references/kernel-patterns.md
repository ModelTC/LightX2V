# SYCL ESIMD Kernel Patterns for BMG GEMM

## Namespace and Includes

```cpp
#include <sycl/sycl.hpp>
#include <sycl/ext/intel/esimd.hpp>
#include <sycl/ext/intel/experimental/esimd/memory.hpp>

namespace sycl_ext   = sycl::ext::intel::esimd;
namespace xesimd     = sycl::ext::intel::experimental::esimd;
namespace xmx        = sycl::ext::intel::experimental::esimd::xmx;
```

## Tile Dimensions (Best Configuration)

```
A tile: 32 rows × 16 cols  (half, loaded as 4 sub-tiles of 8×16)
B tile: 16 rows × 32 cols  (half, VNNI packed, loaded as 4 sub-tiles of 16×8)
C tile: 32 rows × 32 cols  (float accumulators, 4×4 DPAS outputs)

Work-group: M=256, N=256  (one thread per WG)
Thread computes: 4 A-row-panels × 4 B-col-panels = 16 DPAS blocks
K-step: 32 (inner loop increments K by 32 each iteration)
N_SUBS: 2  (double-buffer substeps per K-step iteration)
K_SUB: 16  (= K_STEP / N_SUBS)
```

## DPAS Call — Signature and Register Layout

```cpp
// dpas<RC, SD, Tacc, Tc, Tb, Ta>(acc, b_tile, a_tile)
// On XE2/XE3: RC=8 (fixed), SD=8 (fixed = 32-bit systolic depth)
// For FP16: SD=8 systolic steps × 2 fp16/step = 16 fp16 K-elements per call
xmx::dpas<8, 8, sycl::half, sycl::half, sycl::half, sycl::half>(acc, b_tile, a_tile)
```

**Variable naming convention** (use these names to match dpas argument order):

| Register | Type | Size | Layout | Meaning |
|----------|------|------|--------|---------|
| `a_tile` | `simd<half, 8*16>` | 128 half | `[m*16+k]` row-major | 8 M-rows × 16 K-cols |
| `b_tile` | `simd<half, 16*16>` | 256 half | `uint32[k_pair*16+n_col]` VNNI | 16 K-elems × 16 N-cols |
| `acc`    | `simd<half, 8*16>` | 128 half | `[m*16+n]` row-major | 8 M-rows × 16 N-cols |

Result: `acc[m*16+n] += sum_k  a_tile[m*16+k] * b_tile_fp16(n,k)`

> **Historical note**: older code in this repo used `aa[buf][n]` for b_tile and `bb[buf][m]` for a_tile — names are backwards. Renamed files end in `_v2` and use correct names.

## VNNI Layout (b_tile)

DPAS b_tile must be VNNI-packed. For FP16 the layout in the 256-half register is:

```
uint32[k_pair * 16 + n_col]  =  packed { fp16[n_col, k_pair*2],  fp16[n_col, k_pair*2+1] }
   k_pair = 0..7  (outer, stride 16)
   n_col  = 0..15 (inner, stride 1)
```

Two ways to produce this layout — see next section.

## DPAS Load Patterns: Usage 1 vs Usage 2

### Usage 1 — standard (M=8, N=16, K=16 per dpas call)

```
A [M×K]  →  a_tile   B [N×K] or B_T [K×N]  →  b_tile
dpas(acc, b_tile, a_tile)  →  acc[m*16+n] = C[m, n]
```

**a_tile from A[M×K] — lsc_load_2d, no transform:**
```cpp
xesimd::config_2d_mem_access<sycl::half, 16/*BW=K*/, 8/*BH=M*/, 1> payA(
    A, K*2-1/*surfW bytes*/, M-1/*surfH*/, K*2-1/*surfP*/, 0u, 0u);
simd<sycl::half, 8*16> a_tile = xesimd::lsc_load_2d<
    sycl::half, 16, 8, 1, false/*transpose*/, false/*vnni*/, cached, cached>(payA);
// a_tile[m*16 + k] = A[m_start+m, k_start+k]  row-major ✓
```

**b_tile from B_T[K×N] — lsc_load_2d with VNNI=true:**
```cpp
xesimd::config_2d_mem_access<sycl::half, 16/*BW=N*/, 16/*BH=K*/, 1> payB(
    B_T, N*2-1, K-1, N*2-1, 0u, 0u);
simd<sycl::half, 16*16> b_tile = xesimd::lsc_load_2d<
    sycl::half, 16, 16, 1, false, true/*VNNI*/, cached, cached>(payB);
// b_tile uint32[k_pair*16+n] = {B_T[k*2,n], B_T[k*2+1,n]}  VNNI ✓
```

**b_tile from B[N×K] — lsc_gather SOA (no transpose needed):**
```cpp
// lsc_gather result layout is SOA: result[elem * N + lane]
// → with 16 N-lanes and NElts=8 k-pairs:
//   result[k_pair * 16 + n_col] = uint32 at (offset[n_col] + k_pair*4)
//                               = {B[n_col, k+k_pair*2], B[n_col, k+k_pair*2+1]}
//   = VNNI layout uint32[k_pair*16+n_col]  ✓  (no repack required)
const uint32_t* B_u32 = reinterpret_cast<const uint32_t*>(B);
simd<uint32_t, 16> b_offsets;
for (int n = 0; n < 16; n++)
    b_offsets[n] = (n_base + n) * K * sizeof(half);  // byte offset to B[n, k_start]
simd<sycl::half, 16*16> b_tile;
b_tile.template bit_cast_view<uint32_t>() =
    xesimd::lsc_gather<uint32_t, 8,
        xesimd::lsc_data_size::u32,
        xesimd::cache_hint::cached, xesimd::cache_hint::cached,
        16, uint32_t>(B_u32, b_offsets);
```

**Store C[8×16] — lsc_store_2d:**
```cpp
xesimd::lsc_store_2d<sycl::half, 16, 8, write_back, write_back>(
    C, N*2-1, M-1, N*2-1, n_start, m_start, acc);
```

---

### Usage 2 — swapped (M=16, N=8, K=16 per dpas call)

A is loaded as b_tile (VNNI), B is loaded as a_tile. Avoids transposing B when N < M.

```
A [M×K]  →  b_tile (gather VNNI, 16 M-lanes)
B [N×K]  →  a_tile (2D load, BH=8 N-rows)
dpas(acc, b_tile=A_vnni, a_tile=B)
  →  acc[n_i*16 + m_j] = sum_k B[n_i,k]*A[m_j,k] = C[m_j, n_i]  ← TRANSPOSED
```

**b_tile from A[M×K] — same gather as above, 16 M-lanes:**
```cpp
const uint32_t* A_u32 = reinterpret_cast<const uint32_t*>(A);
simd<uint32_t, 16> b_offsets;
for (int m = 0; m < 16; m++)
    b_offsets[m] = (m_base + m) * K * sizeof(half);  // byte offset to A[m, k_start]
simd<sycl::half, 16*16> b_tile;
b_tile.template bit_cast_view<uint32_t>() =
    xesimd::lsc_gather<uint32_t, 8, u32, cached, cached, 16, uint32_t>(A_u32, b_offsets);
// b_tile uint32[k_pair*16+m_j] = {A[m_j, k*2], A[m_j, k*2+1]}  VNNI ✓
```

**a_tile from B[N×K] — lsc_load_2d, BH=8 N-rows, no transform:**
```cpp
xesimd::config_2d_mem_access<sycl::half, 16/*BW=K*/, 8/*BH=N*/, 1> payB(
    B, K*2-1/*surfW bytes*/, N-1/*surfH*/, K*2-1/*surfP*/, 0u, 0u);
simd<sycl::half, 8*16> a_tile = xesimd::lsc_load_2d<
    sycl::half, 16, 8, 1, false, false, cached, cached>(payB);
// a_tile[n*16+k] = B[n, k]  ✓
```

**a_tile from B_T[K×N] — lsc_gather fp16/u16, K as lanes, SOA → N-outer (no repack):**

`lsc_load_2d transpose=true` is rejected for fp16 at compile time. Use `lsc_gather<sycl::half, 8, u16>` instead: treat the **K rows as the 16 gather lanes**, NElts=8 fp16 per lane = one full row.
The SOA layout maps directly to a_tile's N-outer layout — no `bit_cast_view`, no repack.

```cpp
// K=16 lanes, NElts=8 fp16/lane (one full B_T row per lane)
// offset[k] = k * N * sizeof(half)  (byte offset to start of B_T row k)
// SOA: result[e*16+k] = fp16 at (offset[k] + e*sizeof(half))
//                     = B_T[k][e] = B[e, k]
// a_tile[n*16+k] = B[n, k]  → same index when e=n, direct assign ✓
//
// WARNING: do NOT use lsc_gather<uint32_t, 4> here — uint32 packs {B[n_u32*2,k], B[n_u32*2+1,k]}
// (different N rows, same k) but a_tile uint32 layout is {B[n,k*2], B[n,k*2+1]}
// (same N row, adjacent k). fp16/u16 gather avoids this mismatch entirely.
simd<uint32_t, 16> a_off;
for (int k = 0; k < 16; k++)
    a_off[k] = (uint32_t)k * (uint32_t)N * 2u;  // byte offset to B_T[k, 0]
simd<sycl::half, 8*16> a_tile =
    xesimd::lsc_gather<sycl::half, 8,
        xesimd::lsc_data_size::u16,
        xesimd::cache_hint::cached, xesimd::cache_hint::cached,
        16, uint32_t>(B_T, a_off);
// a_tile[n*16+k] = B[n, k]  ✓
```

> **Key insight**: For b_tile from B[N×K], gather uses N rows as lanes → VNNI layout.
> For a_tile from B_T[K×N], gather uses K rows as lanes → N-outer a_tile. Same SOA trick, different axis.
> Use fp16/u16 element type (not u32) for a_tile so the per-element SOA indexing matches directly.

**Write-back C[16×8] — acc is transposed, two options:**

Option A (lsc_scatter fp16/u16, vectorized, data = acc directly — preferred):
```cpp
// acc[ni*16+mj] = C[mj, ni]
// scatter SOA: data[e*16+lane] → ptr[ offset[lane] + e*sizeof(half) ]
//   data[e*16+mj] → C[mj, e]   (when offset[mj] = mj * N * sizeof(half))
//   data[e*16+mj] = C[mj,e] = acc[e*16+mj]  → data = acc directly ✓
simd<uint32_t, 16> sc_off;
for (int mj = 0; mj < 16; mj++)
    sc_off[mj] = (uint32_t)(m_base + mj) * N_total * 2u + (uint32_t)n_base * 2u;
xesimd::lsc_scatter<sycl::half, 8,
    xesimd::lsc_data_size::u16,
    xesimd::cache_hint::write_back, xesimd::cache_hint::write_back,
    16, uint32_t>(C, sc_off, acc);
// acc passed directly — no register repack needed ✓
```

Option B (scalar, slow — avoid):
```cpp
// acc[n_i*16 + m_j] = C[m_j, n_i]
for (int ni = 0; ni < 8; ni++)
    for (int mj = 0; mj < 16; mj++)
        C[mj * N_total + ni] = acc[ni * 16 + mj];
```

Option C (lsc_scatter u32, vectorized but needs register repack):
```cpp
// 16 M-lanes, each writes 4 uint32 (= 8 fp16) to C[m, n_start..n_start+7]
// scatter SOA: scatter_data[p*16+mj] at (offset[mj] + p*4)
//            = uint32{acc[2p*16+mj], acc[(2p+1)*16+mj]}  ← not contiguous in acc
// → requires register repack; Usage 2 generally not preferred over Usage 1.
simd<uint32_t, 16> sc_off;
for (int mj = 0; mj < 16; mj++)
    sc_off[mj] = (m_base + mj) * N_total * sizeof(half) + n_base * sizeof(half);
// (assemble scatter_data with repack, then:)
xesimd::lsc_scatter<uint32_t, 4, u32, write_back, write_back, 16, uint32_t>(
    reinterpret_cast<uint32_t*>(C), sc_off, scatter_data);
```

> **Recommendation**: prefer Usage 1 (N=16 output) and tile the M dimension. Usage 2 is only useful when N < 16 and is fixed at runtime.

---

## lsc_gather SOA Property (key insight)

`lsc_gather<T, NElts, DS, L1H, L2H, N>` returns `simd<T, N*NElts>` in **SOA** (Structure-of-Arrays) layout:

```
result[elem * N + lane]  =  T value at  ptr[ offset[lane] + elem * sizeof(T) ]
```

NOT AOS (`lane * NElts + elem`). This means NElts=8 with 16 N-lanes naturally produces `[k_pair * 16 + n_col]` = DPAS VNNI format, with zero additional register reorganization.

## `config_2d_mem_access` Payload CSE Pattern

Construct once outside K-loop. Only update coordinates inside:

```cpp
// Surface descriptors (computed once, outside kernel loops)
const uint32_t surfW_A = (uint32_t)K * sizeof(sycl::half) - 1u;
const uint32_t surfH_A = (uint32_t)M - 1u;
const uint32_t surfW_B = (uint32_t)N * sizeof(sycl::half) - 1u;
const uint32_t surfH_B = (uint32_t)K - 1u;

// Payload objects (constructed once)
xesimd::config_2d_mem_access<sycl::half, 16, 8, 1> payA(
    A, surfW_A, surfH_A, surfW_A, /*x=*/0u, /*y=*/(uint32_t)m_start);
xesimd::config_2d_mem_access<sycl::half, 16, 8, 1> payB(
    B, surfW_B, surfH_B, surfW_B, /*x=*/(uint32_t)n_start, /*y=*/0u);

// Inside K-loop:
payA.set_x((uint32_t)(k * sizeof(sycl::half)));   // byte offset in x
for (int m = 0; m < 4; m++) {
    payA.set_y((uint32_t)(m_start + m * 8));
    aa[buf][m] = xesimd::lsc_load_2d<sycl::half, 16, 8, 1,
        false, false,
        xesimd::cache_hint::cached, xesimd::cache_hint::cached>(payA);
}
payB.set_y((uint32_t)k);
for (int n = 0; n < 4; n++) {
    payB.set_x((uint32_t)((n_start + n * 8) * sizeof(sycl::half)));
    bb[buf][n] = xesimd::lsc_load_2d<sycl::half, 16, 8, 1,
        false, true,   // VNNI=true for B
        xesimd::cache_hint::cached, xesimd::cache_hint::cached>(payB);
}
```

**Note**: `set_x()` for A takes byte offset; `set_y()` takes row index (element units). Check header for exact semantics per version.

**Note**: `lsc_prefetch_2d` does NOT accept `config_2d_mem_access` — use the inline form:
```cpp
xesimd::lsc_prefetch_2d<sycl::half, 16, 8, 1,
    xesimd::cache_hint::streaming, xesimd::cache_hint::cached>(
    A, surfW_A, surfH_A, surfW_A, x_byte, y_row);
```

## Double-Buffer Pattern

```cpp
// Tile arrays: [2 buffers][4 sub-tiles]
// a_tile = A register tiles (used as 3rd arg of dpas)
// b_tile = B VNNI register tiles (used as 2nd arg of dpas)
simd<sycl::half, 8*16>  a_tile[2][4];   // A tiles: a_tile[buf][m_blk]
simd<sycl::half, 16*16> b_tile[2][4];   // B tiles: b_tile[buf][n_blk]  (VNNI)
simd<sycl::half, 8*16>  acc[4][4] = {0};

// DPAS call:
acc[m][n] = xmx::dpas<8, 8, sycl::half, sycl::half, sycl::half, sycl::half>(
    acc[m][n], b_tile[buf][n], a_tile[buf][m]);
//            ^^^^^^^^^^^^^^^^^^^  b_tile is 2nd arg
//                                 a_tile is 3rd arg

// Prologue: preload into buf=0
{  /* load a_tile[0], b_tile[0] for k=0, sub=0 */ }

for (int k = 0; k < K; k += K_STEP) {
    // sub 0: use buf=0, load into buf=1
    // sub 1: use buf=1, load into buf=0  (or skip if last k)
    // DPAS uses current buf while loads fill next buf
}
```

Key invariant: with `N_SUBS=2` (even), the last load of each K-step always writes `buf=0`, so the prologue loads into `buf=0` correctly for the next iteration.

## Index Increment Pattern (saves XVE ops)

```cpp
// BAD: full recompute each iter (3+ XVE ops)
payA.set_x((uint32_t)(k + sub * K_SUB) * sizeof(half));

// GOOD: increment (1 XVE op)
uint32_t kx = 0u;
for (...) {
    payA.set_x(kx);
    kx += (uint32_t)(K_SUB * sizeof(half));
}
```

## C Tile Store Pattern

```cpp
// After K-loop, convert float accumulators to half and store
const uint32_t surfW_C = (uint32_t)N * sizeof(sycl::half) - 1u;
xesimd::config_2d_mem_access<sycl::half, 8, 8, 1> payC(
    C, surfW_C, surfH_A, surfW_C, 0u, 0u);

for (int m = 0; m < 4; m++) {
    payC.set_y((uint32_t)(m_start + m * 8));
    for (int n = 0; n < 4; n++) {
        payC.set_x((uint32_t)((n_start + n * 8) * sizeof(sycl::half)));
        auto c_half = sycl_ext::convert<sycl::half>(cc[m][n]);
        xesimd::lsc_store_2d<sycl::half, 8, 8, 1>(payC, c_half);
    }
}
```

## No-If Inner Loop Split

To avoid `if (k_pf < K)` in prefetch:

```cpp
const int pf_limit = K - PF_DIST * K_STEP;

// Phase 1: k in [0, pf_limit) — unconditional prefetch at k + PF_DIST*K_STEP
for (int k = 0; k < pf_limit; k += K_STEP) { /* load + dpas + prefetch */ }

// Phase 2: k in [pf_limit, K) — no prefetch
for (int k = pf_limit; k < K; k += K_STEP) { /* load + dpas only */ }
```

## Z-Order Walk (light-weight Morton decode)

```cpp
// Deinterleave even bits → wg_m, odd bits → wg_n
auto z_even = [](uint32_t z) -> uint32_t {
    z &= 0x55555555u; z = (z|(z>>1)) & 0x33333333u;
    z = (z|(z>>2)) & 0x0f0f0f0fu; return (z|(z>>4)) & 0x00ff00ffu | ((z>>8)&0xff00);
};
auto z_odd = [](uint32_t z) -> uint32_t { return z_even(z >> 1); };

uint32_t wg_id = (uint32_t)(item.get_group(0) * N_WG_N + item.get_group(1));
uint32_t wg_m  = z_even(wg_id);
uint32_t wg_n  = z_odd(wg_id);
```

**Note**: Z-order walk showed no benefit on BMG 4096×4096 GEMM (all 256 WGs run concurrently). Keep LINEAR walk.

## SPIR-V Linker Error Prevention

Move runtime-constant conditions outside the kernel:

```cpp
// BAD: branch inside kernel → possible SPIR-V linker issue
if (some_flag) { do_A(); } else { do_B(); }

// GOOD: template dispatch on host
if (some_flag) q.submit([](auto h){ kernel<true>(h); });
else           q.submit([](auto h){ kernel<false>(h); });
```
