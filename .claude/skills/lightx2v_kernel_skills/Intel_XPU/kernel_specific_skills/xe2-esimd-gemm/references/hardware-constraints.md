# Intel Xe2 / BMG Hardware Constraints

## Execution Units

| Resource | Xe2 (BMG) value | Notes |
|----------|-----------------|-------|
| XVE threads per EU | 8 | hardware |
| EU per DSS (sub-slice) | 8 | |
| DSS per tile (BMG) | 32 | |
| Max HW threads per DSS | 64 | but limited by GRF below |

## GRF (General Register File)

| Mode | GRF/thread | Max WG threads |
|------|-----------|----------------|
| normalGRF (default) | 128 × 32 B = 4 KB | 64 |
| **doubleGRF** | **256 × 32 B = 8 KB** (each 64 B wide) | **32** |

**Rule: always enable `doubleGRF` for large-tile GEMM.** It exposes 16 KB addressable register space per thread, enabling holding large A/B tile accumulator arrays without spill.

Compile flag: `-Xs "-device bmg -options -doubleGRF"` (inside the `-Xs` string, not as a top-level flag).

### Spill detection
The compiler emits: `warning: ... kernel uses ... bytes of spill/fill`. Any spill > 0 causes significant performance regression (off-chip scratch reads/writes). Reduce tile size, payload count, or loop unroll factor to eliminate spill.

## L1 / L2 Cache

| Cache | Size (BMG) | Scope |
|-------|-----------|-------|
| L1 | 64 KB per DSS | shared by all threads in DSS |
| L2 | ~32 MB | shared across all DSS on chip |
| SLM (shared local memory) | 64 KB per DSS | software-managed, accessible within WG |

### Cache hints for `lsc_load_2d`
- `cache_hint::cached / cached` — both L1 and L2 cached (default for GEMM A/B loads)
- `cache_hint::uncached / cached` (L1UC) — bypasses L1, hits L2. **Catastrophic for GEMM inner loop** (observed −63% TFLOPS). Do not use L1UC for A/B tile loads.
- Prefetch: `lsc_prefetch_2d` uses `streaming / cached` or `cached / cached` hints.

## SLM

- 64 KB per DSS, shared across all threads in the WG.
- Useful for sharing data between threads in same WG (requires barrier synchronization).
- Not used in the current single-thread-per-tile GEMM design.

## Work Group Constraints

- **WG threads ≤ 32** with doubleGRF (hardware enforced).
- Current GEMM uses 32 threads per WG (4 M-tiles × 8 N-tiles = 32 — wait: actually the thread is the WG worker; the WG tile is M=256, N=256 but computed by a single thread computing 4×8=32 DPAS tiles... verify per-kernel design).
- The GEMM kernel in this project uses `nd_range<2>({M/256, N/256}, {1, 1})` — each WG is exactly 1 thread. WG size constraint is satisfied.

## Barriers

- `sycl::group_barrier(wg)` or `__esimd_barrier()` must be called the **same number of times** by **every thread** in the WG. Divergent barrier counts → GPU hang (silent, no error message, process stalls indefinitely).
- Safe pattern: if only one thread conditionally needs barrier, move the conditional outside SYCL kernel or ensure all threads take the same branch count.

## Frequency / Throttling

- BMG base clock ~1.85 GHz under GEMM load (observed with VTune).
- Adding more parallel XVE or XMX work **increases power** → frequency drops.
- Minimizing XVE ops in the K-loop inner body is essential to maintain clock speed.
- Observed: 4% XVE ALU2 activity at peak; any increase hurts throughput disproportionately.

## Theoretical Peak

| Metric | Value |
|--------|-------|
| FP16 XMX peak (BMG) | ~135 TFLOPS (nominal) |
| Observed best GEMM | ~118 TFLOPS (~87% of peak) |
| Practical ceiling | ~120 TFLOPS (estimated) |
