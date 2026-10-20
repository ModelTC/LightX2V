# Intel Xe2 / BMG Hardware Constraints

## Execution Units

| Resource | Arc B580 / Xe2 value | Notes |
|----------|-----------------|-------|
| XVE threads per EU | 8 | hardware |
| EU per DSS (sub-slice) | 8 | |
| Xe-Cores | 20 | SKU-specific; not a limit for all BMG products |
| Max HW threads per DSS | 64 | but limited by GRF below |

## GRF (General Register File)

| Mode | GRF per hardware thread |
|------|-------------------------|
| regular GRF | 128 × 64 B = 8 KiB |
| **large GRF / doubleGRF** | **256 × 64 B = 16 KiB** |

The bundled large-tile examples use `doubleGRF` to provide 16 KiB per hardware thread. Check compiler spills and occupancy before applying it to a different kernel; extra GRFs do not guarantee better performance.

The SKU values above and below follow [Intel's architecture table](https://www.intel.com/content/www/us/en/docs/oneapi/optimization-guide-gpu/2025-2/intel-xe-gpu-architecture.html). The historical performance section retains its original experiment's figures and must not be interpreted as B580 measurements.

Compile flag: `-Xs "-device bmg -options -doubleGRF"` (inside the `-Xs` string, not as a top-level flag).

### Spill detection
The compiler emits: `warning: ... kernel uses ... bytes of spill/fill`. Any spill > 0 causes significant performance regression (off-chip scratch reads/writes). Reduce tile size, payload count, or loop unroll factor to eliminate spill.

## L1 / L2 Cache

| Cache | Size (Arc B580) | Scope |
|-------|-----------|-------|
| L1 | 256 KB per Xe-Core | hardware-managed |
| L3 | 18 MB | device-wide |
| SLM (shared local memory) | 128 KB per Xe-Core | software-managed; check per-workgroup allocation limits |

### Cache hints for `lsc_load_2d`
- `cache_hint::cached / cached` — both L1 and L2 cached (default for GEMM A/B loads)
- `cache_hint::uncached / cached` (L1UC) bypasses L1. It reduced throughput by 63% in the recorded experiment; compare it on the actual kernel before changing A/B loads.
- Prefetch: `lsc_prefetch_2d` uses `streaming / cached` or `cached / cached` hints.

## SLM

- B580 has 128 KB per Xe-Core; the compiler's permitted allocation and residency also constrain each workgroup.
- Useful for sharing data between threads in same WG (requires barrier synchronization).
- Not used in the current single-thread-per-tile GEMM design.

## Work Group Constraints

- The bundled NOPF, gather and no-if harnesses launch 32 work-items per workgroup with a 1D `nd_range`; inspect each asset's `WG_SIZE` or local range before changing its tile mapping.
- Resident hardware threads, workgroup size and register allocation are different limits. Query the target device/kernel and check compiler resource reports rather than deriving a universal maximum from `doubleGRF` alone.

## Barriers

- `sycl::group_barrier(wg)` or `__esimd_barrier()` must be called the **same number of times** by **every thread** in the WG. Divergent barrier counts → GPU hang (silent, no error message, process stalls indefinitely).
- Safe pattern: if only one thread conditionally needs barrier, move the conditional outside SYCL kernel or ensure all threads take the same branch count.

## Frequency / Throttling

- BMG base clock ~1.85 GHz under GEMM load (observed with VTune).
- Adding more parallel XVE or XMX work **increases power** → frequency drops.
- Minimizing XVE ops in the K-loop inner body is essential to maintain clock speed.
- Observed: 4% XVE ALU2 activity at peak; any increase hurts throughput disproportionately.

## Theoretical Peak

Historical experiment values below use the original BMG configuration and nominal peak; they are not universal Xe2 specifications. Record the actual SKU, clock and compiler when reproducing them.

| Metric | Value |
|--------|-------|
| FP16 XMX peak (BMG) | ~135 TFLOPS (nominal) |
| Observed best GEMM | ~118 TFLOPS (~87% of peak) |
| Practical ceiling | ~120 TFLOPS (estimated) |
