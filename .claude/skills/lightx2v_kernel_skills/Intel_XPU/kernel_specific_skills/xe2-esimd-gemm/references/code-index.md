# Sample Code Index

All source files are in `assets/`. Copy to a working directory, compile, and run.

## Compile command (all files use the same command)

```bash
icpx <file>.cpp -o <file>.exe \
  -fsycl -fsycl-targets=spir64_gen \
  -Xs "-device bmg -options -doubleGRF"
```

Run on Windows:
```bash
powershell.exe -Command "& './<file>.exe'"
```

---

## `assets/fp16_dpas_ult.cpp` — **DPAS Unit-Level Test** 🔬

> Also available as a standalone skill: **`xe2-dpas-patterns`** — covers all DPAS load/store patterns reusable beyond GEMM.

**Purpose**: Validates all four DPAS load/store patterns at minimal tile size (single kernel thread, no outer tiling). Run this first when porting to a new layout or debugging VNNI correctness.

**Four cases tested** (M=8/16, N=16/8, K=16, single nd_item):

| Case | B layout | B→DPAS path | Shuffle? | Result layout |
|------|----------|-------------|----------|---------------|
| C1 | B[N,K]   | `lsc_gather<u32,8,u32,N=16>` N-lanes SOA → b_tile | none | acc[m*16+n]=C[m,n] |
| C2 | B_T[K,N] | `lsc_load_2d<half,16,16,VNNI=true>` → b_tile | none | acc[m*16+n]=C[m,n] |
| C3 | B[N,K]   | `lsc_load_2d<half,16,8>` → a_tile | none | acc[n*16+m]=C[m,n]ᵀ |
| C4 | B_T[K,N] | `lsc_gather<half,8,u16,N=16>` K-lanes SOA → a_tile | none | acc[n*16+m]=C[m,n]ᵀ |

All four cases use `lsc_scatter<half,8,u16,N=16>` for write-back (Cases 1–2 use `lsc_store_2d`).

**Expected output** (all four pass):
```
C1: b_tile←B[N,K] gather            rel RMS = 0.0178%  PASS
C2: b_tile←B_T[K,N] load_2d VNNI   rel RMS = 0.0178%  PASS
C3: a_tile←B[N,K] load_2d           rel RMS = 0.0203%  PASS
C4: a_tile←B_T[K,N] gather fp16     rel RMS = 0.0203%  PASS
ALL PASS
```

---

## `assets/fp16_gemm_nopf_v2.cpp` — **Best kernel (current naming)** ⭐

**Result**: ~117 TFLOPS — same kernel as `fp16_gemm_nopf.cpp` with corrected variable naming.

**Variable naming fix** (vs old files):
- `bb[buf][m]` → `a_tile[buf][m]`  (A register tiles, 3rd dpas arg)
- `aa[buf][n]` → `b_tile[buf][n]`  (B VNNI tiles, 2nd dpas arg)
- DPAS call: `dpas(acc[m][n], b_tile[buf][n], a_tile[buf][m])` ← matches dpas arg order

**B layout**: B_T[K×N] row-major — loaded via `lsc_load_2d<half,16,16,VNNI=true>`.

**Key kernel parameters**:
```
WG tile: M=256, N=256  (one WG per 256×256 output block)
Thread: 4 A-panels × 4 B-panels = 16 DPAS outputs per thread
a_tile: simd<half, 8*16>   lsc_load_2d BW=16 BH=8  VNNI=false
b_tile: simd<half, 16*16>  lsc_load_2d BW=16 BH=16 VNNI=true
acc:    simd<half, 8*16>   [m*16+n] row-major
K_STEP=32, K_SUB=16, double-buffer
```

**Payload CSE pattern**:
```cpp
config_2d_mem_access<half,16,8,1>  payA(A,   surfW_A, surfH_A, surfW_A, 0u, 0u);
config_2d_mem_access<half,16,16,1> payB(B_T, surfW_B, surfH_B, surfW_B, 0u, 0u);
// In loop: only set_x / set_y
payA.set_x((uint32_t)(k + K_SUB));   // byte offset along K
payA.set_y((uint32_t)(m_start + m*8));
payB.set_y((uint32_t)(k + K_SUB));
payB.set_x((uint32_t)(n_start + n*16));
```

**DPAS call** (correct naming):
```cpp
acc[m][n] = xmx::dpas<8,8,half,half,half,half>(acc[m][n], b_tile[buf][n], a_tile[buf][m]);
//                                              ^^^^^^^^^^^^^^^^^^^  b_tile is 2nd arg
//                                                                   a_tile is 3rd arg
```

---

## `assets/fp16_gemm_gather_v2.cpp` — **Gather variant (no B transpose)** ⭐

**Result**: ~114 TFLOPS — avoids B_T[K,N] transpose by using `lsc_gather` for b_tile directly from B[N,K].

**Key difference from nopf_v2**: B is stored as [N×K] row-major (not transposed). Uses `lsc_gather<uint32_t,8,u32,N=16>` SOA to produce VNNI b_tile without any register repack.

**B gather pattern** (core contribution):
```cpp
// Precompute row byte offsets once outside K-loop
const uint32_t Kbytes = K * sizeof(half);
for (int nb = 0; nb < 4; nb++)
    for (int nl = 0; nl < 16; nl++)
        base_B[nb][nl] = (n_start + nb*16 + nl) * Kbytes;  // row start, bytes

// In loop (each substep):
simd<uint32_t,16> byte_off = base_B[nb] + k_ab;  // k_ab incremented by K_SUB*2
b_tile[buf][nb].bit_cast_view<uint32_t>() =
    lsc_gather<uint32_t,8,u32,cached,cached,16>(B_u32, byte_off);
// SOA: result[k_pair*16+n] = {B[n, k+k_pair*2], B[n, k+k_pair*2+1]}  VNNI ✓
```

**K-loop offset tracking**: uses `k_a` (fp16 element index for payA.set_x) and `k_ab` (byte offset for gather), both incremented with `+=` only.

---

## `assets/fp16_gemm_nopf.cpp` — **Original nopf** (old `aa`/`bb` naming)

**Result**: 117.10 TFLOPS, 1.85 GHz, ~4% XVE ALU2 (86.7% of 135T peak)

**What it demonstrates**:
- `config_2d_mem_access` payload CSE pattern (construct once, only `set_x`/`set_y` in loop)
- No prefetch (GEMM is compute-bound — prefetch hurts)
- Double-buffer double-substep pattern (K_STEP=32, N_SUBS=2, buf[0]/buf[1])
- Prologue + main loop + last-K-block structure (no `if` in K-loop body)
- 32 threads per WG (WG_SIZE=32 mandatory with doubleGRF)
- B matrix stored transposed (K×N layout) for coalesced access with VNNI

**Key kernel parameters**:
```
WG tile: M=256, N=256  (one WG per 256×256 output block)
Thread: 4 A-panels × 4 B-panels = 16 DPAS outputs per thread
A tile: 8×16 half   (lsc_load_2d BW=16, BH=8, VNNI=false)
B tile: 16×16 half  (lsc_load_2d BW=16, BH=16, VNNI=true)
C accum: 8×16 half  (xmx::dpas<8,8,...>)
K_STEP=32, K_SUB=16, N_SUBS=2
```

**surfW/surfH formula** (critical — bytes for width, elements for height):
```cpp
surfW_A = K * sizeof(half) - 1   // = K*2 - 1
surfH_A = M - 1
surfW_B = N * sizeof(half) - 1   // = N*2 - 1
surfH_B = K - 1
```

**Payload CSE pattern**:
```cpp
// Construct ONCE outside K-loop:
config_2d_mem_access<half,16,8,1>  payA(A,   surfW_A, surfH_A, surfW_A, 0u, 0u);
config_2d_mem_access<half,16,16,1> payB(B_T, surfW_B, surfH_B, surfW_B, 0u, 0u);
// In loop: only set_x / set_y
payA.set_x((uint32_t)(k + K_SUB));   // byte offset along K
payA.set_y((uint32_t)(m_start + m*8));
```

**DPAS call** (old `aa`/`bb` naming — see `fp16_gemm_nopf_v2.cpp` for corrected names):
```cpp
acc[m][n] = xmx::dpas<8, 8, half, half, half, half>(acc[m][n], aa[buf][n], bb[buf][m]);
// aa[buf][n] = b_tile (B_T VNNI, 2nd arg),  bb[buf][m] = a_tile (A tile, 3rd arg)
```

**Warmup + benchmark**: 5s sleep → 20 warmup → 1000 timed iters, single `q.wait()` at end.

---

## `assets/fp16_gemm_nopf3.cpp` — Best variant (+0.3%) with induction vars

**Result**: 117.44 TFLOPS (highest measured, +0.3% over nopf)

**What it demonstrates**:
- Everything from nopf, plus:
- `uint32_t ks = K_SUB, kn = K_STEP; ks += K_STEP, kn += K_STEP` as loop induction vars
- `set_x(ks)` / `set_y(ks)` → pure MOV (no ADD at call site), saving 4 XVE ops/K_STEP
- Template parameter `LoadMode` for sweeping CACHED vs L1UC variants in one binary
- Demonstrates catastrophic effect of L1UC (43.90T, −62.6%) for reference

**Key pattern**:
```cpp
for (uint32_t ks = K_SUB, kn = K_STEP; kn < (uint32_t)K;
     ks += K_STEP, kn += K_STEP)
{
    payA.set_x(ks);  // pure MOV — no ADD at call site
    ...
    payA.set_x(kn);  // pure MOV
    ...
}
```

---

## `assets/fp16_gemm_nopf_verify.cpp` — Correctness checker

**Result**: PASS — rel RMS 0.060%, 0 real outliers at M=N=K=256

**What it demonstrates**:
- Small-problem correctness test (M=N=K=256, CPU fp32 reference)
- B_T layout: `B_T[k*N + n] = B[n*K + k]`
- Exact copy of FP16_GEMM_NOPF kernel struct — verifying the benchmark binary is correct
- RMS error computation with near-zero denominator guard

**Run this first** before benchmarking to confirm the kernel produces correct results.

Expected output:
```
PASS: FP16 GEMM NOPF result is CORRECT.
      (Max rel err ~0.300% < 5% threshold, no outliers)
```

---

## `assets/fp16_gemm_noif.cpp` — Pre-optimization baseline

**Result**: 109.55 TFLOPS, 1.75 GHz, ~40% XVE ALU2 (81.1% of 135T peak)

**What it demonstrates**:
- The before-optimization kernel: inline `lsc_load_2d` args rebuilt every call
- Per-call descriptor rebuild pattern (the anti-pattern to avoid):
  ```cpp
  // BAD — compiler does NOT CSE K*2-1, M-1 across calls:
  bb[0][m] = lsc_load_2d<half,16,8,1,false,false,cached,cached>(
      A, K*sizeof(half)-1, M-1, K*sizeof(half)-1,
      k + K_SUB, m_start + m*8);
  ```
- PF_DIST=2 prefetch (also shows how prefetch was structured — and why it was removed)
- Demonstrates the **40% XVE ALU2** problem: rebuilding surface descriptors per call
- Uses `using namespace sycl::ext::intel::experimental::esimd` (no alias) — older style

**Why to keep**: Starting point for understanding the 7.4 TFLOPS gain from payload CSE.
The gap `(117.10 - 109.55) / 109.55 = +6.9%` comes purely from eliminating descriptor rebuilds.

---

## Performance Ladder (reproduce in this order)

```
1. Verify correctness:
   icpx fp16_gemm_nopf_verify.cpp -o verify.exe ...
   ./verify.exe
   → Expected: PASS

2. Baseline (before optimization):
   icpx fp16_gemm_noif.cpp -o noif.exe ...
   ./noif.exe
   → Expected: ~109.55 TFLOPS, 1.75 GHz

3. Best kernel (payload CSE, no prefetch):
   icpx fp16_gemm_nopf.cpp -o nopf.exe ...
   ./nopf.exe
   → Expected: ~117.10 TFLOPS, 1.85 GHz

4. Best+induction vars sweep:
   icpx fp16_gemm_nopf3.cpp -o nopf3.exe ...
   ./nopf3.exe
   → Expected: CACHED ~117.44T, L1UC ~43.9T
```

---

## Known Non-Reproducible Optimizations (don't attempt again)

These were tested and definitively failed — do not spend time re-testing:

| Experiment | Result | File |
|------------|--------|------|
| Distributed prefetch (PF_DIST=1..8) | −8% to −24% | fp16_gemm_distpf.cpp |
| Z-order / COL_SWIZZLE walk order | −2% to −25% | fp16_gemm_walk.cpp |
| K_STEP=64/128/256 | −6% (EU scheduler window) | fp16_gemm_kstep_nopf.cpp |
| 2×payA + 2×payB alternating | −2.8% | fp16_gemm_2pay.cpp |
| 4×payA + 4×payB | JIT crash (3840B spill) | fp16_gemm_nopf2.cpp |
| L1UC cache hint on loads | −63% | fp16_gemm_nopf3.cpp LIUC variant |
