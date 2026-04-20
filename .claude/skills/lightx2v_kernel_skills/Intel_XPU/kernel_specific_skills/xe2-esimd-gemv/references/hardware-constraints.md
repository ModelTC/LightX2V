# BMG Hardware Constraints for GEMV

## Memory bandwidth
- **DRAM bandwidth**: 520 GB/s
- **L2 bandwidth**: ~1400 GB/s (much higher — cache-busting is mandatory to measure DRAM)
- GEMV is fully memory-bound (arithmetic intensity ≈ 0.5 FLOP/byte for W8A16, lower for W4)

## Execution units
- 32 XE cores × 8 EUs × 8 threads = **2048 hardware threads**
- Each EU executes one SIMD-16 instruction per cycle
- For GEMV: design global thread count ≈ 2048 to saturate all EUs

## GRF (General Register File)
- Default: 128 × 32-byte GRF per thread = 4 KB
- With `doubleGRF`: 256 × 32-byte GRF = 8 KB
- GEMV kernels do NOT need doubleGRF — low register pressure
- Compile with doubleGRF anyway (no harm, consistent with other kernels)

## SLM (Shared Local Memory)
- 64 KB per XE core, shared by all WGs on that core
- For K-split GEMV: `SLM_SIZE = ROWS × K_SPLIT × sizeof(float)` — typically < 1 KB, no contention
- `slm_init(bytes)` must be the first kernel statement

## VL (vector length) limits
- Maximum practical VL for GEMV: **1024** elements
- VL=2048 causes compiler stack overflow (JIT register allocation failure)
- Safe values: 128, 256, 512, 1024

## Work-group size
- Max WG threads with doubleGRF: 32
- For GEMV: `local_size = ROWS × K_SPLIT`. Keep ≤ 32.
  - Recommended: 8 (R=4 K=2), 16 (R=8 K=2), 4 (R=4 K=1)
- GEMV does not need large WGs (no DPAS tile sharing)

## Block load alignment
- `block_load<T, N>` requires the pointer to be N×sizeof(T) aligned
- Weight arrays allocated with `malloc_device` are 4KB aligned — safe
- VL × sizeof(fp16) = 1024×2 = 2 KB — well within alignment

## Barrier cost
- Each `barrier()` call stalls all threads in WG until all reach it
- In K-split GEMV: exactly one barrier per WG needed (between compute and reduce)
- Symmetric barrier count is mandatory — GPU hangs if any thread skips the barrier

## Profiling
- Use `queue::enable_profiling()` property for accurate per-kernel timing
- `event::get_profiling_info<command_start/command_end>()` returns nanoseconds
- Wall-clock timing includes launch overhead — GPU event profiling is more accurate
