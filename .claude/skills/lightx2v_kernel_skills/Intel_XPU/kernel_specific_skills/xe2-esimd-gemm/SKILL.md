---
name: xe2-esimd-gemm
description: This skill should be used when writing, optimizing, benchmarking, or debugging high-performance FP16 GEMM kernels (or similar dense-compute SYCL ESIMD kernels) targeting Intel Xe2 (Lunar Lake/LNL, Battlemage/BMG) GPU. Xe2 is the GPU architecture; LNL and BMG are product names. Covers ESIMD API, XMX DPAS, hardware constraints, performance methodology, optimization patterns, and known pitfalls.
---

# Xe2 (Lunar Lake/LNL, Battlemage/BMG) ESIMD GEMM Skill

Specialized knowledge for authoring and optimizing SYCL ESIMD matrix-multiply kernels on Intel Xe2 (Lunar Lake/LNL, Battlemage/BMG) architecture.
Reference files hold detail; this file holds the critical rules and workflow.

---

## Quick-Reference Rules (must follow every time)

### Hardware limits
- **Max WG threads = 32** when `doubleGRF` is on (256 × 64-byte GRF = 16 KB/thread). Never set `nd_range` work-group size > 32. See `references/hardware-constraints.md`.
- **Always compile with `doubleGRF`** — it is mandatory for large tile kernels. Do not remove it.
- **Barriers**: every thread in a WG must execute the same number of `barrier()` calls. Unequal counts cause GPU hang.

### Inner-loop rules
- **No `if`, no `?:`** inside the K-loop body. Move all runtime conditionals to the host or split the loop into phases.
- **Use `index +=` increments** instead of recomputing full expressions each iteration. Saves XVE ALU ops.
- **More XVE + XMX parallelism lowers frequency**. Minimize XVE ops inside the K-loop; DPAS dominates and XVE pressure throttles the clock.

### API namespace
- Use `sycl::ext::intel::experimental::esimd` (xesimd) for `lsc_load_2d`, `lsc_prefetch_2d`, `config_2d_mem_access`, `xmx::dpas`.
- `lsc_prefetch_2d` does **not** accept a payload object (`config_2d_mem_access`) — use the inline 6-argument form: `lsc_prefetch_2d<T,BW,BH,N,L1H,L2H>(ptr, surfW, surfH, surfPitch, x, y)`.
- `lsc_load_2d` supports the payload object API (`config_2d_mem_access`) with `set_x()` / `set_y()`.

### Payload CSE (critical optimization)
- Construct `config_2d_mem_access` once **outside** the K-loop with static fields (ptr, surfW, surfH, surfPitch). Only call `set_x()` / `set_y()` inside the loop. This eliminates per-call descriptor rebuilds and dramatically reduces XVE activity.

### DPAS / VNNI
- See `references/kernel-patterns.md` for tile sizes, VNNI packing, double-buffer pattern, and dpas call signature.

### Performance testing
- **Use different buffers** (cache-busting) for memory-bound kernels; rotate through an array of input buffer pairs each iteration.
- **Random non-zero init** — avoid all-zero inputs; they hide NaN/denormal bugs and give unrealistically fast cache results.
- **Warmup**: run ≥ 20 iterations before timing.
- **Iteration count**: 100 iters minimum; 1000 iters preferred for stable measurement on compute-bound kernels.
- **Sleep before timing** (5 s) to let GPU reach steady frequency before measurement begins.
- See `references/perf-testing.md` for boilerplate.

### Correctness testing
- Check for **NaN in output** before computing relative error (NaN denominator silently passes threshold checks).
- Use an FP32 CPU reference; compare half-precision GPU output with `rel_rms < 0.5%` as pass criterion.
- Test at M=N=K=256 (fast) before benchmarking at 4096.

### Tile walk order
- Hilbert / Z-order (Morton) walk of work-group indices can improve L2 reuse **only** when not all WGs run concurrently. On BMG with a 4096×4096 problem all 256 WGs run simultaneously — walk order has no effect.
- Z-order walk: `wg_m = z_deinterleave_even(wg_id)`, `wg_n = z_deinterleave_odd(wg_id)`. Keep index compute light (bit ops only).

### Compile command
```bash
icpx <src>.cpp -o <out>.exe \
  -fsycl -fsycl-targets=spir64_gen \
  -Xs "-device bmg -options -doubleGRF"
```
Do **not** use `-fsycl-targets=intel_gpu_bmg_g21` (wrong target string).
Do **not** use `-O3` or `-doubleGRF` as top-level flags — they are ignored or warn.

### SPIR-V linker errors
- Cause: kernel contains a runtime branch (`if`) whose condition can be evaluated on the host. Move the branch outside the SYCL kernel (template or host-side dispatch) to eliminate the SPIR-V link-time issue.

---

## Workflow

1. **Write kernel** following inner-loop rules above.
2. **Compile** with the command above; check for spill warnings (`warning: ... spilled ... bytes`). Spill > 0 usually means GRF budget exceeded — reduce tile size or payload count.
3. **Correctness test** at small size (M=N=K=256). Check NaN, check rel RMS.
4. **Benchmark** at production size (M=N=K=4096) with 20 warmup + 1000 timed iters, 5 s pre-sleep.
5. **Profile** with VTune or `--collect gpu-hotspots` to measure XVE ALU2 %. Target < 5% XVE for DPAS-bound kernel.
6. **Iterate**: reduce XVE ops via payload CSE, index increment, loop splitting.

---

## Sample code (ready to compile)

Copy assets to a working directory, then compile and run:

```bash
icpx <file>.cpp -o <file>.exe -fsycl -fsycl-targets=spir64_gen -Xs "-device bmg -options -doubleGRF"
powershell.exe -Command "& './<file>.exe'"
```

| Asset | TFLOPS | Purpose |
|-------|--------|---------|
| `assets/fp16_gemm_nopf_v2.cpp` | **~117T** | **Best kernel (current)** — B_T[K,N] layout, correct `a_tile`/`b_tile` naming, payload CSE |
| `assets/fp16_gemm_gather_v2.cpp` | **~114T** | **Gather variant (current)** — B[N,K] layout (no transpose), `lsc_gather<u32,8,N=16>` for b_tile |
| `assets/fp16_gemm_nopf.cpp` | 117.10 | Original nopf — old `aa`/`bb` naming (see _v2 for corrected names) |
| `assets/fp16_gemm_nopf3.cpp` | **117.44** | Highest measured — induction-var XVE reduction; also tests L1UC (43.9T) |
| `assets/fp16_gemm_nopf_verify.cpp` | — | Correctness checker (M=N=K=256, CPU ref). Run before benchmarking. |
| `assets/fp16_gemm_noif.cpp` | 109.55 | Pre-optimization baseline showing 40% XVE problem (inline descriptor rebuild) |

To reproduce all results in order, see `references/code-index.md`.

---

## Reference files

| File | Contents |
|------|----------|
| `references/code-index.md` | Per-file annotations, key patterns, performance ladder, failed experiments |
| `references/hardware-constraints.md` | Xe2/BMG GRF, L1, SLM, WG, barrier limits |
| `references/kernel-patterns.md` | DPAS tile layout, VNNI packing, double-buffer pattern, payload CSE code |
| `references/lsc-memory-ops.md` | Full LSC API: `lsc_load_2d`, `lsc_store_2d`, `lsc_prefetch_2d`, `lsc_gather`, `lsc_scatter`, `config_2d_mem_access`, cache hints |
| `references/perf-testing.md` | Cache-bust boilerplate, timing harness, random init, NaN check |
| `references/optimization-history.md` | Exhaustive record of every optimization tried on this GEMM with TFLOPS results |
