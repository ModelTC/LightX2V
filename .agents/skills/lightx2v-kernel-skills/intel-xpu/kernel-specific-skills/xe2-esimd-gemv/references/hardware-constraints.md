# BMG Hardware Constraints for GEMV

Bandwidth and 32-core scheduling figures below are historical experiment inputs; the exact SKU is not recorded. B580 capacities use the [hardware reference](../../../kernel-basic-skills/intel-gpu-hw-info/SKILL.md). Query the actual device before reusing a launch configuration.

## Memory bandwidth
- **DRAM bandwidth**: 520 GB/s
- **L2 bandwidth**: ~1400 GB/s (much higher — cache-busting is mandatory to measure DRAM)
- GEMV is fully memory-bound (arithmetic intensity ≈ 0.5 FLOP/byte for W8A16, lower for W4)

## Execution units
- Historical assumption: 32 XE cores × 8 EUs × 8 threads = **2048 hardware threads per device**. B580 has 20 cores and **1280 threads per device** (64 per core).
- Each EU executes one SIMD-16 instruction per cycle
- For GEMV: use the target's thread capacity as a starting point, then measure occupancy and throughput; 2048 is not a universal launch target

## GRF (General Register File)
- B580 regular mode: 128 × 64-byte GRF per thread = 8 KiB
- B580 large register mode (`doubleGRF`): 256 × 64-byte GRF = 16 KiB
- The bundled GEMV kernels have low register pressure; check compiler spill reports
- Measure regular versus large register mode when tuning: extra registers can reduce residency

## SLM (Shared Local Memory)
- B580: 128 KiB per XE core, shared by resident WGs; up to 128 KiB per WG
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
- USM `block_load` defaults to 4-byte alignment for elements up to 4 bytes, or 8-byte alignment for 8-byte elements; it does not require `N × sizeof(T)` alignment
- Check the final address after adding offsets. Without cache hints, declare a smaller guaranteed alignment with `properties{alignment<2>}` for half/bf16 or `alignment<1>` for bytes
- Allocation size and vector length do not prove address alignment. See [Block Load / Store](../../../kernel-basic-skills/intel-esimd-base/references/programming-patterns.md#block-load--store-contiguous-access) for store and overload restrictions

## Barrier cost
- Each `barrier()` call stalls all threads in WG until all reach it
- In K-split GEMV: exactly one barrier per WG needed (between compute and reduce)
- Symmetric barrier count is mandatory — GPU hangs if any thread skips the barrier

## Profiling
- Use `queue::enable_profiling()` property for accurate per-kernel timing
- `event::get_profiling_info<command_start/command_end>()` returns nanoseconds
- Wall-clock timing includes launch overhead — GPU event profiling is more accurate
