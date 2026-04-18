---
name: intel-gpu-hw-info
description: >
  Definitive reference for Intel GPU hardware specifications across architectures.
  Covers Xe2 (Lunar Lake/LNL, Battlemage/BMG) and Xe3 (Panther Lake/PTL,
  Panther Lake-H/PTLH) GPU hardware: XE core counts, memory bandwidth, XMX/DPAS
  compute, GRF sizes, SLM limits, thread counts, EU layout, L3 cache, TDP. Use
  whenever the user asks about Intel GPU specs, hardware comparison, architecture
  differences, roofline parameters, or thread/memory limits. Trigger for questions
  like "how many XE cores", "what is BMG bandwidth", "PTL vs BMG", "Xe2 specs",
  "LNL GPU", etc.
---

# Intel GPU Hardware Specifications

Definitive reference for Intel GPU hardware across Xe2 and Xe3 architectures. All values sourced from kernel optimization work, profiling results, and verified benchmarks.

**Version**: 1.0.0
**Last Updated**: 2026-03-12

---

## Architecture Overview

- **Xe2** is a GPU architecture used in multiple products:
  - **Battlemage (BMG)**: Discrete GPU (dGPU), Intel Arc B-series
  - **Lunar Lake (LNL)**: Integrated GPU (iGPU), Intel Core Ultra 200V series
- **Xe3** is a GPU architecture used in:
  - **Panther Lake (PTL)**: Integrated GPU (iGPU)
  - **Panther Lake-H (PTLH)**: High-performance integrated GPU

All Xe2 and Xe3 products share the same ISA and programming model. Code is portable across dGPU and iGPU variants, though performance tuning differs significantly.

---

## Xe2 Architecture -- Battlemage (BMG) Discrete GPU

| Parameter | Value |
|---|---|
| GPU Type | Discrete (dGPU), Intel Arc B580/B570 |
| XE Cores | 20 (B570) / 20 (B580) -- up to 32 in development samples |
| EUs per XE Core | 8 |
| Threads per EU | 8 |
| Total HW Threads | up to 2048 (32-core config) |
| GRF Mode | doubleGRF: 256 regs x 64 bytes = 16 KB/thread |
| Max WG Threads | 32 (with doubleGRF) |
| SLM per XE Core | 64 KB |
| Memory Type | GDDR6 |
| Memory Bandwidth | ~520 GB/s (32-core config) |
| FP16 XMX (DPAS) Peak | ~135 TFLOPS (32-core config) |
| FP16 ALU (non-XMX) | ~17 TFLOPS |
| DPAS Systolic Depth | 8 |
| DPAS Repeat Count | 8 |
| L3/LLC Cache | ~8-16 MB |
| TDP | ~150-225W |

### BMG Notes
- 32 XE cores x 8 EU x 8 threads = 2048 hardware threads total
- doubleGRF is mandatory for large tile GEMM/attention kernels (256 GRF entries per thread)
- Without doubleGRF, max WG threads is 64 but register pressure becomes the bottleneck
- Practical peak BW is ~520 GB/s; theoretical is ~550 GB/s
- L2 capacity is approximately 16 MB shared across all cores

---

## Xe2 Architecture -- Lunar Lake (LNL) Integrated GPU

| Parameter | Value |
|---|---|
| GPU Type | Integrated (iGPU), Intel Core Ultra 200V |
| XE Cores | 8 |
| EUs per XE Core | 8 |
| Threads per EU | 8 |
| Total HW Threads | 512 |
| Memory Type | LPDDR5x (shared with CPU) |
| Memory Bandwidth | ~90 GB/s |
| Same ISA as BMG | Yes -- same Xe2 architecture |

### LNL Notes
- Shared system memory means CPU activity impacts GPU bandwidth
- Fewer cores require careful workgroup sizing to avoid underutilization
- Same doubleGRF, SLM, and DPAS capabilities as BMG at the per-core level

---

## Xe3 Architecture -- Panther Lake (PTL) Integrated GPU

| Parameter | Value |
|---|---|
| GPU Type | Integrated (iGPU) |
| XE Cores | 12 |
| EUs per XE Core | 8 |
| Threads per EU | 8 |
| Total HW Threads | 768 |
| GRF Mode | doubleGRF: 256 regs x 64 bytes = 16 KB/thread |
| Max WG Threads | 32 (with doubleGRF) |
| SLM per XE Core | 64 KB |
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
- Same ISA as BMG (Xe2), but different microarchitecture generation (Xe3)

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

| Parameter | Xe2 BMG (dGPU) | Xe2 LNL (iGPU) | Xe3 PTL (iGPU) |
|---|---|---|---|
| XE Cores | 20-32 | 8 | 12 |
| Memory BW | 520 GB/s | ~90 GB/s | ~112 GB/s |
| FP16 XMX | 135 TFLOPS | TBD | 55 TFLOPS |
| Memory Type | GDDR6 dedicated | LPDDR5x shared | LPDDR5x shared |
| TDP | 150-225W | 15-30W | 15-30W |
| SLM/Core | 64 KB | 64 KB | 64 KB |
| GRF | 16 KB/thread | 16 KB/thread | 16 KB/thread |
| HW Threads | 2048 | 512 | 768 |
| EUs/Core | 8 | 8 | 8 |
| Threads/EU | 8 | 8 | 8 |

---

## Roofline Parameters

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
- 64 KB per XE core, shared across all WGs on the same core
- 32 banks, 4 bytes per bank
- Bank conflict avoidance: pad SLM rows, ensure consecutive threads access consecutive banks
- `slm_init(SLM_SIZE)` must be the very first statement in the kernel

### L3/LLC Cache
- BMG: approximately 8-16 MB, acts as implicit cache for global reads
- L2 capacity: approximately 16 MB shared across all cores
- Without cache busting, L2 can inflate bandwidth measurements by 2-3x

### GRF (General Register File)
- 16 KB/thread in doubleGRF mode (256 registers x 64 bytes)
- Without doubleGRF: 128 registers x 64 bytes = 8 KB/thread
- doubleGRF is mandatory for large tile GEMM/attention kernels
- Not required for memory-bound kernels like GEMV (low GRF pressure)
- Register spill to stack memory kills performance -- always check compiler output

### Cache Hints for Stores
- For `lsc_block_store`: L2 must be `uncached` or `write_back` -- NOT `cached`
- Valid store hints: `(streaming, uncached)`, `(write_through, uncached)`, `(write_back, write_back)`
- For loads: `(streaming, cached)` and `(cached, cached)` are both valid

---

## Key Differences: dGPU vs iGPU

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
- Same ISA and programming model -- code compiles and runs on both
- Same ESIMD intrinsics, DPAS instructions, SLM operations
- Performance tuning differs: iGPU needs smaller WG sizes, wave count optimization matters more
- Compile target differs: `-device bmg` vs `-device ptl`

---

## Thread Configuration Quick Reference

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
| `xe3-esimd-kernels` | Kernel patterns specific to Xe3/PTL |
| `xe2-dpas-patterns` | DPAS tile patterns for Xe2 |
| `intel-esimd-base` | Foundational ESIMD programming |
