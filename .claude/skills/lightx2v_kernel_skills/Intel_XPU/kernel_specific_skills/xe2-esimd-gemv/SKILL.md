---
name: xe2-esimd-gemv
description: >
  Use this skill when writing, optimizing, benchmarking, or debugging W4A16 or W8A16 GEMV
  kernels targeting Intel Xe2 (Lunar Lake/LNL, Battlemage/BMG) GPU using SYCL ESIMD. Xe2 is the GPU architecture; LNL and BMG are product names. Also covers general
  FP16 GEMV patterns. Covers quantized weight dequantization, SIMD vs scalar interleaving,
  K-split SLM reduction, VL/ROWS tuning, workgroup decomposition, uint4 unpacking, FP32
  accumulation, SLM barriers, performance methodology, and all hardware constraints.
---

# Xe2 ESIMD GEMV Kernels — W4A16, W8A16 & FP16

Specialized knowledge for memory-bandwidth-bound quantized and FP16 GEMV (M=1 matrix-vector multiply) on Intel Xe2 (Lunar Lake/LNL, Battlemage/BMG).
Reference files hold detail; this file holds critical rules and workflow.

**Platform**: Xe2 (BMG) — 520 GB/s DRAM, 32 XE cores x 8 EUs x 8 threads = 2048 HW threads

**Achieved results**: W4A16 **571 GB/s** (110% roofline), W8A16 **552 GB/s** (106% roofline)

---

## Quick-Reference Rules (must follow every time)

### Hardware limits
- **2048 hardware threads** total (32 XE × 8 EU × 8 threads). Design WG sizes so `num_groups × local_size` fills this.
- **No `doubleGRF` required** for GEMV — kernels are memory-bound, GRF pressure is low.
- **SLM**: 64 KB per XE core. For GEMV with K-split: `SLM_SIZE = ROWS × K_SPLIT × sizeof(float)`. This is tiny; no pressure.
- **Barriers**: every thread in a WG must execute the same number of `barrier()` calls. Unequal counts cause GPU hang.

### SIMD dequant — the #1 rule (1.98× speedup)
- **NEVER use scalar loops** to interleave lo/hi nibbles into a SIMD register. This is the single most impactful mistake.
- **ALWAYS use `simd::template select<COUNT, STRIDE>(OFFSET)`** for strided writes:
  ```cpp
  // BAD  — scalar loop, 289 GB/s
  for (int i = 0; i < 64; i++) {
      weight_f[base + i * 2]     = lo[i];
      weight_f[base + i * 2 + 1] = hi[i];
  }

  // GOOD — SIMD strided select, 571 GB/s
  weight_f.template select<64, 2>(base + 0) = lo;   // even positions
  weight_f.template select<64, 2>(base + 1) = hi;   // odd positions
  ```
- All nibble extraction must also stay SIMD: use `p & 0x0F` and `(p >> 4) & 0x0F` directly on `simd<uint8_t,64>`.
- Dequant arithmetic (subtract zero-point, multiply scale) must operate on full `simd<float,64>`.

### W4A16 dequantization pattern
- **Weight layout**: `[N, K/2]` uint8 — two uint4 nibbles packed per byte, lo nibble = even k, hi nibble = odd k.
- **Scale layout**: `[N, K/BLOCK_SIZE]` fp16 — one scale per 128-element block (BLOCK_SIZE=128).
- **Formula**: `weight_fp = (uint4_val - 8) × scale`  (symmetric, zero_point=8)
- **Block loop** (unrolled, `NUM_BLOCKS = VL/128`):
  ```cpp
  auto p = weight_packed.template select<64, 1>(blk * 64);  // 64 bytes = 128 nibbles
  simd<float, 64> lo = p & 0x0F;
  simd<float, 64> hi = (p >> 4) & 0x0F;
  lo = (lo - 8.0f) * sc;
  hi = (hi - 8.0f) * sc;
  weight_f.template select<64, 2>(blk * 128 + 0) = lo;
  weight_f.template select<64, 2>(blk * 128 + 1) = hi;
  ```

### W8A16 dequantization pattern
- **Weight layout**: `[N, K]` int8.
- **Scale layout**: `[N]` fp16 — one scale per row (no blocking needed).
- **Formula**: `weight_fp = int8_val × scale`
- Simple: load `simd<int8_t, VL>`, convert to float, multiply scalar scale. No interleaving needed.
- No K-split required — simple row-parallel dispatch achieves 552 GB/s.

### K-split strategy
- **K-split** threads in a WG share one output row: each computes a slice `[k_start, k_start+K/K_SPLIT)`, stores to SLM, then thread-0-of-slice reduces.
- **Thread layout**: `local_id = row_thread_id × K_SPLIT + k_thread_id`
- **Recommended K_SPLIT**: 2 for W4A16 (best balance); 1 (none) for W8A16.
- K_SPLIT > 4 adds SLM overhead with diminishing returns. Never exceed 8.
- `local_size = ROWS × K_SPLIT`. `num_groups = ceil(N / ROWS)`.

### SLM reduction
- **`slm_init(SLM_SIZE)` must be the very first statement** in the kernel — before any other code.
- SLM layout: `[ROWS][K_SPLIT]` floats. Offset = `(row_thread_id × K_SPLIT + k_thread_id) × sizeof(float)`.
- Use vectorized SLM load for reduction — never scalar loop:
  ```cpp
  // K_SPLIT == 2
  simd<float, 2> r = slm_block_load<float, 2>(slm_base);
  final_sum = r[0] + r[1];
  // K_SPLIT == 4 or 8
  simd<float, K_SPLIT> r = slm_block_load<float, K_SPLIT>(slm_base);
  final_sum = reduce<float>(r, std::plus<>());
  ```
- Only `k_thread_id == 0` threads write the final output after reduction.

### VL (vector length) tuning
- **VL = 1024** is optimal for both W4A16 and W8A16 on BMG.
- VL = 256 or 512 work but give ~5–10% lower bandwidth.
- **VL = 2048 causes compiler stack overflow** — do not use.
- VL must be a multiple of 128 (dequant block size) for W4A16.

### ROWS tuning
- ROWS controls how many output rows one WG computes.
- **ROWS = 4 with K_SPLIT = 2** (local_size = 8) is optimal for W4A16.
- Target: `ceil(N/ROWS) × local_size ≈ 2048` to fill all HW threads.
- Larger ROWS (> 16) with large K reduces EU utilization; smaller ROWS wastes dispatch overhead.

### Multiple accumulators (latency hiding)
- Use `simd<float, 8> partial_sums = 0.0f` with rotating index (`acc_idx = (acc_idx+1) & 0x7`) across K iterations.
- Final reduce: `float s = reduce<float>(partial_sums, std::plus<>())`.
- This hides FMA latency on memory-bound kernels and is low cost.

### API namespace
- Use `sycl::ext::intel::esimd` for `block_load`, `block_store`, `slm_init`, `slm_block_load`, `slm_block_store`, `barrier`, `reduce`.
- Include: `<sycl/sycl.hpp>` and `<sycl/ext/intel/esimd.hpp>`.
- `using namespace sycl::ext::intel::esimd;` simplifies calls.

### Performance testing
- **Cache-busting is mandatory** — GEMV weight matrix fits easily in L2/L3; without cache-busting you will measure L2 bandwidth (~1400 GB/s), not DRAM.
- Allocate **32 weight copies** with different seeds; rotate `weight_idx = i % num_copies` each iteration.
- **Warmup**: 5 iterations minimum before timing.
- **Iteration count**: 50–100 timed iterations; take median.
- Use **GPU event profiling** (`command_start`/`command_end`) not wall-clock for per-kernel timing.
- **Bandwidth formula**:
  - W4A16: `bytes = K×2 + N×(K/2) + N×(K/128)×2 + N×2`  (input fp16 + weight uint8 + scale fp16 + output fp16)
  - W8A16: `bytes = K×2 + N×K×1 + N×2 + N×2`  (input fp16 + weight int8 + scale fp16 + output fp16)
- See `references/perf-testing.md` for boilerplate.

### Correctness testing
- Check **NaN count** before computing max_diff.
- Use FP32 CPU reference. Compare with threshold: `abs_diff > 1.0 && rel_error > 2%` = fail.
- FP16 accumulation introduces ~1% relative error — this is expected and acceptable.
- Test at small sizes (N=256, K=512) before benchmarking at N=8192/16384.
- See `references/correctness-testing.md`.

### Compile command
```bash
icpx <src>.cpp -o <out>.exe \
  -fsycl -fsycl-targets=spir64_gen \
  -Xs "-device bmg -options -doubleGRF"
```
- `-O3` is optional but does not significantly change GEMV performance (memory-bound).
- Do **not** use `-fsycl-targets=intel_gpu_bmg_g21`.
- Spill warning (`kernel uses N bytes of scratch space`) indicates GRF pressure — reduce VL if seen.
- Run on Windows: `powershell.exe -Command "& './<out>.exe'"`

---

## Workflow

1. **Choose kernel type**: W4A16 (nibble-packed, needs dequant) or W8A16 (byte-packed, simpler).
2. **Write dequant** following the SIMD select pattern above — never scalar loops.
3. **Set parameters**: start with VL=1024, ROWS=4, K_SPLIT=2 (W4A16) or VL=1024, K_SPLIT=1 (W8A16).
4. **Compile** — check for spill warnings. Any spill > 0 means potential regression.
5. **Correctness test** at small size (N=256, K=512). Check NaN, check max_diff.
6. **Benchmark** with 32 weight copies (cache-bust), 5 warmup + 50 timed, median timing.
7. **Sweep** VL ∈ {256, 512, 1024} and ROWS ∈ {4, 8, 12, 16} if not at target.
8. **Verify** bandwidth formula accounts for all memory traffic (input + weight + scale + output).

---

## Assets (ready to compile)

```bash
icpx <file>.cpp -o <file>.exe -fsycl -fsycl-targets=spir64_gen -Xs "-device bmg -options -doubleGRF"
powershell.exe -Command "& './<file>.exe'"
```

| Asset | BW | Purpose |
|-------|----|---------|
| `assets/w4a16_simd_optimized.cpp` | **571 GB/s** | **Production W4A16** — SIMD select dequant, K-split=2, ROWS=4, VL=1024; sweeps multiple configs |
| `assets/w8a16_nocache.cpp` | **552 GB/s** | **Production W8A16** — simple row-parallel, VL=1024, 32 weight copies; sweeps VL |

Expected on BMG (N=8192–16384, K=4096–8192):
- `w4a16_simd_optimized.exe` → **571 GB/s** (110% of 520 GB/s roofline)
- `w8a16_nocache.exe` → **552 GB/s** (106% of 520 GB/s roofline)

---

## Reference files

| File | Contents |
|------|----------|
| `references/hardware-constraints.md` | BMG thread count, SLM limits, VL limits, memory bandwidth |
| `references/kernel-patterns.md` | SIMD dequant patterns, K-split layout, SLM reduction code, multiple-accumulator pattern, bandwidth formula |
| `references/perf-testing.md` | Cache-bust boilerplate, timing harness, random init, bandwidth formula |
| `references/correctness-testing.md` | CPU reference pattern, thresholds, NaN check, corner cases |
| `references/optimization-history.md` | Full journey: scalar loop (289 GB/s) → SIMD select (571 GB/s), every experiment with results |
| `references/code-index.md` | Per-file annotations, parameter summary, performance ladder |
