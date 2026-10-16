---
name: intel-gpu-hw-info
description: "Check Intel Xe2/Xe3 GPU resource limits and roofline inputs: SKU-specific core/thread counts, GRF, SLM, cache, bandwidth and compute. Distinguish verified specifications from historical kernel experiment assumptions."
---

# Intel GPU Hardware Specifications

Reference notes from Xe2/Xe3 kernel experiments. Treat SKU-dependent core counts, frequencies, bandwidth, cache and measured throughput as historical inputs, not universal device constants. Query the actual device and verify compiler target support before choosing launch limits or building a roofline.

Verified Xe2 resource sizes below use Intel's [GPU architecture table](https://www.intel.com/content/www/us/en/docs/oneapi/optimization-guide-gpu/2025-2/intel-xe-gpu-architecture.html): B580 and the listed 8-core Lunar Lake configuration. The historical 32-core BMG and 12-core PTL experiment inputs are retained separately; they are not specifications for every SKU. KiB denotes 1024 bytes.

**Version**: 1.0.0
**Last Updated**: 2026-10-16

---

## Architecture Overview

- **Xe2** is a GPU architecture used in multiple products:
  - **Battlemage (BMG)**: Discrete GPU (dGPU), Intel Arc B-series
  - **Lunar Lake (LNL)**: Integrated GPU (iGPU), Intel Core Ultra 200V series
- **Xe3** is a GPU architecture used in:
  - **Panther Lake (PTL)**: Integrated GPU (iGPU)
  - **Panther Lake-H (PTLH)**: High-performance integrated GPU

These products support SYCL programming, but sharing a programming model does not imply identical ISA, intrinsic support, or launch limits. Verify the target SKU and compiler before porting ESIMD code.

---

## Xe2 Architecture -- Battlemage (BMG) Discrete GPU

| Parameter | Value |
|---|---|
| GPU Type | Discrete (dGPU), Intel Arc B580 |
| XE Cores | 20 (B580) |
| EUs per XE Core | 8 |
| Threads per EU | 8 |
| Total HW Threads | 1280 = 20 cores x 8 vector engines x 8 threads |
| GRF Mode | 128 / 256 regs x 64 bytes = 8 / 16 KiB per thread (regular / large mode) |
| ESIMD WG configuration | Historical kernels use at most 32 work-items with doubleGRF; verify kernel/toolchain limits |
| SLM per XE Core / max per WG | 128 KiB / 128 KiB (B580) |
| Memory Type | GDDR6 |
| Historical Memory Bandwidth | ~520 GB/s (32-core experiment input, not a B580 specification) |
| Historical FP16 XMX (DPAS) Peak | ~135 TFLOPS (same experiment) |
| Historical FP16 ALU (non-XMX) | ~17 TFLOPS (same experiment) |
| DPAS Systolic Depth | 8 |
| DPAS Repeat Count | 8 |
| L3 Cache | 18 MB (B580) |
| Historical Power Range | ~150-225W (not a B580 specification) |

### BMG Notes
- B580: 8 vector engines x 8 threads = 64 hardware threads per core, 1280 per device. The older 32-core experiment assumed 2048 per device; its exact SKU is not recorded.
- The bundled large-tile GEMM/attention experiments use doubleGRF. Check spills and occupancy before choosing regular or large register mode for a new kernel.
- Historical bandwidth inputs were ~520 GB/s practical and ~550 GB/s theoretical. The old cache estimate was ~16 MB, called "L2" in experiment notes; use the SKU's documented cache levels/capacities for new work.
- Query device `local_mem_size` and device/kernel work-group limits. General SYCL work-group sizes count work-items and are not interchangeable with ESIMD hardware-thread limits.

---

## Xe2 Architecture -- Lunar Lake (LNL) Integrated GPU

| Parameter | Value |
|---|---|
| GPU Type | Integrated (iGPU), Intel Core Ultra 200V |
| XE Cores | 8 |
| EUs per XE Core | 8 |
| Threads per EU | 8 |
| Total HW Threads | 512 |
| SLM per XE Core / max per WG | 128 KiB / 128 KiB (listed 8-core configuration) |
| GRF Mode | 128 / 256 regs x 64 bytes = 8 / 16 KiB per thread |
| Memory Type | LPDDR5x (shared with CPU) |
| Historical Memory Bandwidth | ~90 GB/s; shared-memory bandwidth depends on the system |

### LNL Notes
- Shared system memory means CPU activity impacts GPU bandwidth
- Fewer cores require careful workgroup sizing to avoid underutilization
- Register width and SLM capacity match the listed B580 configuration; verify intrinsic and compiler target support separately

---

## Xe3 Architecture -- Panther Lake (PTL) Integrated GPU

The following are inputs and observations from the historical 12-core experiment, not independently verified Xe3 product limits. In particular, do not infer PTL/PTLH SLM or GRF capacity from Xe2.

| Parameter | Value |
|---|---|
| GPU Type | Integrated (iGPU) |
| XE Cores | 12 |
| EUs per XE Core | 8 |
| Threads per EU | 8 |
| Total HW Threads | 768 |
| GRF Mode | Experiment assumed doubleGRF, 256 x 64 bytes = 16 KiB/thread; verify target support |
| ESIMD WG configuration | 32 work-items with doubleGRF in the experiment; verify kernel/toolchain limits |
| SLM per XE Core | Not verified here; check the target SKU and device-reported local memory limit |
| Memory Type | LPDDR5x (shared with CPU) |
| Memory Bandwidth | ~112 GB/s |
| FP16 XMX (DPAS) Peak | ~55 TFLOPS |
| FP16 ALU (non-XMX) | ~14 TFLOPS |
| Concurrent WGs/Core | 1 (32t WG), 2 (16t WG), 4 (8t WG) |
| Wave count critical | Yes -- only 12 cores, thread config matters |

### PTL Notes
- Only 12 XE cores means wave count is the dominant performance factor
- Concurrent WG capacity per core depends on threads per WG:
  - 32 threads/WG: 1 WG per core, 12 concurrent WGs total
  - 16 threads/WG: 2 WGs per core, 24 concurrent WGs total
  - 8 threads/WG: 4 WGs per core, 48 concurrent WGs total
- For recurrent (latency-bound) kernels, 8 threads/WG is optimal (fewer waves)
- For decode (throughput-bound) kernels at batch >= 4, 32 threads/WG is better
- Shared system RAM: budget approximately 1 GB for device buffers to avoid OOM
- Xe3 is a different generation from BMG (Xe2); verify compiler target and intrinsic support when porting

---

## Xe3 Architecture -- Panther Lake-H (PTLH) Integrated GPU

| Parameter | Value |
|---|---|
| GPU Type | Integrated (iGPU), high-performance variant |
| XE Cores | TBD (expected more than PTL) |
| Memory Type | LPDDR5x or DDR5 (shared with CPU) |
| Memory Bandwidth | TBD (expected higher than PTL 112 GB/s) |

---

## Architecture Comparison

| Parameter | Xe2 B580 (dGPU) | Xe2 LNL (listed 8-core iGPU) | Xe3 PTL (historical assumptions) |
|---|---|---|---|
| XE Cores | 20 | 8 | 12 |
| Memory Type | GDDR6 dedicated | LPDDR5x shared | LPDDR5x shared |
| SLM/Core | 128 KiB | 128 KiB | Verify SKU |
| GRF (regular / large mode) | 8 / 16 KiB per thread | 8 / 16 KiB per thread | Verify SKU/compiler |
| HW Threads per device | 1280 | 512 | 768 assumed |
| EUs/Core | 8 | 8 | 8 |
| Threads/EU | 8 | 8 | 8 |

Historical bandwidth/compute inputs are recorded below; do not use the 32-core experiment's roofline for B580 without measurement.

---

## Roofline Parameters

These are historical experiment inputs, not SKU-wide specifications. Recompute the roofline with the target device's measured bandwidth and relevant compute ceiling.

### BMG (32-core dGPU)
```
Peak FP16 XMX: 135 TFLOPS
Peak BW:       520 GB/s = 0.52 TB/s
Ridge Point:   135 / 0.52 = ~260 FLOPs/byte
```
- Kernels with arithmetic intensity (AI) < 260 are memory-bound
- Kernels with AI > 260 are compute-bound
- Most attention/GEMM kernels with HEAD_DIM >= 128 are compute-bound on BMG
- Practical peak BW is 520 GB/s; theoretical is 550 GB/s
- XVE FP32 (non-XMX): approximately 4 TFLOPS
- Kernel launch overhead: approximately 5-10 microseconds

### PTL (12-core iGPU)
```
Peak FP16 XMX: 55 TFLOPS
Peak BW:       112 GB/s = 0.112 TB/s
Ridge Point:   55 / 0.112 = ~491 FLOPs/byte
```
- Higher ridge point means a wider memory-bound regime
- PTL's lower bandwidth makes more workloads memory-bound compared to BMG
- Shared memory with CPU means effective bandwidth varies with system load

### Roofline Calculation Examples

**GEMM: C[M,N] = A[M,K] x B[K,N]**
```
FLOPs = 2 * M * N * K
Bytes  = (M*K + K*N + M*N) * sizeof(half)
AI     = FLOPs / Bytes
```

**Flash Attention:**
```
FLOPs = 4 * q_len * kv_len * head_dim * num_heads
      + 2 * q_len * kv_len * num_heads  (softmax)
```

### Roofline Classification Thresholds
```
actual_ms > bw_limit_ms * 3.0  --> LATENCY BOUND (serial dependency or launch overhead)
actual_ms > bw_limit_ms * 1.5  --> PARTIALLY BW BOUND (stalls or contention)
actual_ms <= bw_limit_ms * 1.1 --> AT BW ROOFLINE
```

---

## DPAS/XMX Instruction Details (Xe2 and Xe3)

### Instruction Format
```cpp
xmx::dpas<SD, RC, T_dst, T_src0, T_src1, T_src2>(acc, src1, src2)
```
- **SD = 8**: Systolic depth (fixed for Xe2/Xe3)
- **RC = 8**: Repeat count
- **T_dst, T_src0**: Accumulator type (typically float)
- **T_src1, T_src2**: Input types (typically half or bfloat16)

### Compute Per DPAS Call
- FP16: K = SD x 2 = 16 elements per systolic step
- BF16: K = SD x 2 = 16 elements per systolic step
- Output tile: M = RC rows x N = 16 cols per DPAS call

### VNNI Layout Requirement
- B operand requires VNNI layout (interleaved 2-element packing)
- Use Transposed 2D load with uint32_t to get hardware VNNI transform
- VNNI packing: two FP16 values stored in one uint32_t, alternating across the K dimension

### DPAS Pipeline Optimization
- Consecutive DPAS instructions without intervening sends/movs maximize XMX utilization
- Minimize XVE ALU operations inside DPAS loops (XVE + XMX parallelism lowers frequency)
- Use payload CSE: construct `config_2d_mem_access` once outside K-loop, only call `set_x()`/`set_y()` inside

---

## Memory Subsystem

### Global Memory
- **block_load**: Contiguous 1D loads, up to 256 bytes per load for D32 (FP32)
  - `block_load<float, 64>` = 256 bytes (maximum for D32)
  - `block_load<float, 128>` = 512 bytes -- EXCEEDS LSC limit, must split to 2 x 64
  - `block_load<half, 128>` = 256 bytes (within limit)
- **gather**: Non-contiguous loads via scatter/gather
- **lsc_load_2d**: 2D block loads with hardware coalescing (preferred for tiled access)
- **lsc_prefetch_2d**: 2D prefetch (uses inline 6-argument form, not payload object)

### SLM (Shared Local Memory)
- B580 and the listed Lunar Lake configuration: 128 KiB per XE core, shared across resident WGs; up to 128 KiB per WG. Verify other SKUs separately.
- 32 banks, 4 bytes per bank
- Bank conflict avoidance: pad SLM rows, ensure consecutive threads access consecutive banks
- `slm_init(SLM_SIZE)` must be the very first statement in the kernel

### L3/LLC Cache
- B580: 18 MB L3 and 256 KB L1 per XE core in Intel's table
- The historical experiment estimated 8-16 MB shared cache and called it L2; retain that label when reading its results, not as a B580 specification
- Without cache busting, cache reuse can inflate apparent DRAM bandwidth; the historical notes reported 2-3x

### GRF (General Register File)
- B580 and listed Lunar Lake: 16 KiB/thread in large register mode (256 registers x 64 bytes)
- Regular mode: 128 registers x 64 bytes = 8 KiB/thread
- Bundled large-tile GEMM/attention experiments use doubleGRF; measure its register-pressure/occupancy tradeoff for new kernels
- Bundled GEMV kernels have low GRF pressure; memory-bound behavior alone does not establish the register requirement
- Register spill to stack memory kills performance -- always check compiler output

### Cache Hints for Stores
- For `lsc_block_store`: L2 must be `uncached` or `write_back` -- NOT `cached`
- Valid store hints: `(streaming, uncached)`, `(write_through, uncached)`, `(write_back, write_back)`
- For loads: `(streaming, cached)` and `(cached, cached)` are both valid

---

## Key Differences: dGPU vs iGPU

The numerical ranges below are historical tuning inputs, not allocation or performance guarantees for a SKU.

| Aspect | dGPU (BMG) | iGPU (LNL, PTL) |
|---|---|---|
| Memory | Dedicated GDDR6 | Shared system RAM |
| Bandwidth | High (520 GB/s) | Limited (90-112 GB/s) |
| Core Count | High (20-32) | Low (8-12) |
| Power Budget | 150-225W | 15-30W (shared package) |
| Memory Budget | Full VRAM | ~1 GB safe allocation |
| Wave Count | Less critical | Dominant perf factor |
| WG Sizing | Larger WGs fine | Smaller WGs often better |
| CPU Contention | None | CPU impacts GPU BW |

### Portability
- Shared SYCL programming model; verify compilation and correctness on each target
- Check ESIMD intrinsic, DPAS, and SLM support against the target and installed SDK
- Performance tuning differs: iGPU needs smaller WG sizes, wave count optimization matters more
- Compile target differs: `-device bmg` vs `-device ptl`

---

## Thread Configuration Quick Reference

These are historical scheduling calculations. For B580 use 20 x 8 x 8 = 1280 hardware threads per device; actual residency also depends on register mode, SLM, and kernel resources. Hardware-thread counts are not launch-size limits.

### BMG (32 XE Cores)
```
32 cores x 8 EU/core x 8 threads/EU = 2048 HW threads
With doubleGRF: max 32 threads per WG
Design WG sizes so num_groups x local_size fills 2048 threads
```

### PTL (12 XE Cores)
```
12 cores x 8 EU/core x 8 threads/EU = 768 HW threads
With doubleGRF: max 32 threads per WG

Threads/WG | EUs/WG | WGs/Core | Concurrent WGs (12 cores) | Waves for H=36
32         | 8      | 1        | 12                        | 3
16         | 4      | 2        | 24                        | 2
8          | 2      | 4        | 48                        | 1
```

### LNL (8 XE Cores)
```
8 cores x 8 EU/core x 8 threads/EU = 512 HW threads
```

---

## Compilation Targets

```bash
# BMG (Xe2 dGPU)
icpx kernel.cpp -o kernel.exe -fsycl -fsycl-targets=spir64_gen \
  -Xs "-device bmg -options -doubleGRF"

# PTL (Xe3 iGPU)
icpx kernel.cpp -o kernel.exe -fsycl -fsycl-targets=spir64_gen \
  -Xs "-device ptl -options -doubleGRF"
```

- Do NOT use `-fsycl-targets=intel_gpu_bmg_g21` (wrong target string)
- Do NOT use `-O3` or `-doubleGRF` as top-level flags (they are ignored or warn)
- Check for spill warnings in compiler output: any spill > 0 indicates GRF pressure

---

## Related Skills

| Skill | What it covers |
|---|---|
| `intel-gpu-kernel-opt` | General optimization methodology using these HW specs |
| `xe2-esimd-gemm` | GEMM tuning specific to Xe2 |
| `xe2-esimd-gemv` | GEMV tuning specific to Xe2/BMG |
| `xe2-dpas-patterns` | DPAS tile patterns for Xe2 |
| `intel-esimd-base` | Foundational ESIMD programming |
