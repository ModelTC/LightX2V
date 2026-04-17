---
name: intel-gpu-kernel-opt
description: >
  General Intel GPU kernel optimization methodology. Use this skill when
  profiling or optimizing any ESIMD or SYCL kernel on Intel GPUs, performing
  roofline analysis, diagnosing bottlenecks (register spill, SLM bank conflicts,
  barrier overhead, memory coalescing), comparing Xe2 vs Xe3 hardware, or
  planning an optimization workflow. Covers VTune and GTPin profiling, key
  metrics (TFLOPS, GB/s, peak %), hardware comparison (Xe2: LNL/BMG vs Xe3: PTL/PTLH), and
  optimization patterns (prefetch, load/compute separation, loop unrolling,
  SIMD width selection). Xe2 is the architecture for Lunar Lake (LNL) and Battlemage (BMG);
  Xe3 is the architecture for Panther Lake (PTL) and Panther Lake-H (PTLH).
  Trigger for any Intel GPU performance question.
---

# Intel GPU Kernel Optimization Methodology

Systematic approach to profiling, diagnosing, and optimizing GPU kernels on Intel Xe2 (Lunar Lake/LNL, Battlemage/BMG) and Xe3 (Panther Lake/PTL, Panther Lake-H/PTLH) architectures. Covers roofline analysis, bottleneck identification, profiling tools, hardware comparison, and proven optimization patterns.

**Version**: 1.0.0
**Last Updated**: 2026-03-12

---

## Table of Contents

1. [Optimization Workflow](#optimization-workflow)
2. [Roofline Analysis](#roofline-analysis)
3. [Key Performance Metrics](#key-performance-metrics)
4. [Hardware Comparison: Xe2 vs Xe3](#hardware-comparison-xe2-vs-xe3)
5. [Common Bottlenecks](#common-bottlenecks)
6. [Profiling Tools](#profiling-tools)
7. [Optimization Patterns](#optimization-patterns)
8. [Optimization Priority Checklist](#optimization-priority-checklist)
9. [Case Study: SDP HD=256 Optimization Journey](#case-study-sdp-hd256-optimization-journey)

---

## Optimization Workflow

The fundamental cycle for GPU kernel optimization:

```
Profile --> Identify Bottleneck --> Optimize --> Verify --> Repeat
```

### Step-by-Step

1. **Baseline**: Establish a correct, reproducible benchmark with warmup iterations
2. **Profile**: Capture hardware metrics (VTune GPU Compute Hotspots or GTPin)
3. **Identify bottleneck**: Classify as compute-bound, memory-bound, or latency-bound
4. **Hypothesize**: Form a specific hypothesis about what limits performance
5. **Optimize**: Apply a single targeted optimization
6. **Verify**: Measure both correctness AND performance
7. **Record**: Document the result (positive or negative) for future reference
8. **Repeat**: Return to step 2

**Critical rule**: Apply ONE optimization at a time. Multiple simultaneous changes make it impossible to attribute performance differences.

---

## Roofline Analysis

### Classification

A kernel is either **compute-bound** or **memory-bound**:

```
Arithmetic Intensity (AI) = FLOPs / Bytes_transferred
Ridge Point = Peak_TFLOPS / Peak_BW_TB/s

If AI > Ridge Point: compute-bound (optimize ALU utilization)
If AI < Ridge Point: memory-bound (optimize memory access)
```

### Xe2 (BMG) Roofline

```
Peak FP16 XMX:  135 TFLOPS
Peak BW:        520 GB/s = 0.52 TB/s
Ridge Point:    135 / 0.52 = ~260 FLOPs/byte
```

Most attention/GEMM kernels with HEAD_DIM >= 128 are compute-bound on BMG.

### Xe3 (PTL) Roofline

```
Peak FP16 XMX:  TBD (depends on SKU, ~30-50 TFLOPS estimated)
Peak BW:        112 GB/s = 0.112 TB/s
Ridge Point:    ~300-450 FLOPs/byte (estimated)
```

PTL's lower bandwidth makes more workloads memory-bound compared to BMG.

### Practical Roofline Calculation

```cpp
// GEMM: C[M,N] = A[M,K] x B[K,N]
double flops = 2.0 * M * N * K;
double bytes = (M*K + K*N + M*N) * sizeof(half);  // read A, B; write C
double ai = flops / bytes;

// Flash Attention: 4*q*kv*d*heads + 2*q*kv*heads (softmax)
double attn_flops = 4.0 * q_len * kv_len * head_dim * num_heads
                  + 2.0 * q_len * kv_len * num_heads;
```

---

## Key Performance Metrics

| Metric | Formula | Target |
|--------|---------|--------|
| TFLOPS | FLOPs / (time_ms * 1e9) | Higher is better |
| GB/s | Bytes / (time_ms * 1e6) | Compare to peak BW |
| % of Peak Compute | Measured TFLOPS / Peak TFLOPS * 100 | > 60% is good |
| % of Peak BW | Measured GB/s / Peak GB/s * 100 | > 70% is good |
| XMX Utilization | (from VTune) XMX busy cycles / total cycles | > 50% for GEMM |
| Register Spill | (from compiler output or VTune) | 0 is ideal |
| Occupancy | Active threads / Max threads | Higher generally better |

### Benchmarking Best Practices

```cpp
// Standard benchmark pattern
constexpr int WARMUP = 5;
constexpr int ITERATIONS = 100;

// Warmup (let GPU clock stabilize, populate caches)
for (int i = 0; i < WARMUP; i++) {
    kernel.execute();
    q.wait();
}

// Timed iterations
auto start = std::chrono::high_resolution_clock::now();
for (int i = 0; i < ITERATIONS; i++) {
    kernel.execute();
    q.wait();
}
auto end = std::chrono::high_resolution_clock::now();
double avg_ms = std::chrono::duration<double, std::milli>(end - start).count() / ITERATIONS;
```

---

## Hardware Comparison: Xe2 vs Xe3

| Parameter | Xe2 (LNL / Lunar Lake, BMG / Battlemage) | Xe3 (PTL / Panther Lake, PTLH / Panther Lake-H) |
|-----------|----------------------|------------------------|
| GPU Type | Discrete (dGPU) | Integrated (iGPU) |
| XE Cores | 32 | 12 |
| Memory BW | ~520 GB/s (GDDR6) | ~112 GB/s (LPDDR5x shared) |
| Max Threads | 2048 per XE core | TBD |
| SLM per XE Core | 64 KB | 64 KB |
| GRF Mode | doubleGRF (256 regs/thread) | TBD |
| FP16 XMX Peak | ~135 TFLOPS | TBD (~30-50 TFLOPS est.) |
| DPAS Systolic Depth | 8 | TBD |
| L3 Cache | ~8-16 MB | TBD |
| Power | ~150-225W TDP | ~15-30W (shared package) |

### Architecture Implications

**BMG (Xe2)**:
- Compute-rich: 32 XE cores, high TFLOPS ceiling
- High bandwidth: 520 GB/s dedicated GDDR6
- Optimization focus: maximize XMX utilization, minimize register spill
- doubleGRF essential for large tile sizes (256 GRF entries per thread)

**PTL (Xe3)**:
- Bandwidth-constrained: 112 GB/s shared with CPU
- Fewer cores: 12 XE cores, lower absolute throughput
- Optimization focus: minimize memory traffic, maximize data reuse
- Shared memory: CPU activity impacts GPU bandwidth

---

## Common Bottlenecks

### 1. Register Spill

**Symptom**: Performance far below compute roofline despite high arithmetic intensity.

**Diagnosis**: Build with `-save-temps`, check compiler output for spill/fill counts. In VTune, look for high "Send" cycle counts from register spill loads/stores.

**Solutions**:
- Reduce tile sizes (e.g., M=16 to M=8)
- Use doubleGRF (`-Xs "-options -doubleGRF"`)
- Reduce live register ranges (reorder computations)
- Split large kernels

### 2. SLM Bank Conflicts

**Symptom**: SLM access latency higher than expected, visible in VTune shared memory stalls.

**Diagnosis**: Check SLM access patterns for stride conflicts. 32 banks, 4 bytes/bank on Intel GPUs.

**Solutions**:
- Pad SLM rows to avoid stride conflicts
- Ensure consecutive threads access consecutive banks
- Rearrange data layout in SLM

### 3. Barrier Overhead

**Symptom**: Many barriers per iteration, visible as synchronization stalls in profiler.

**Diagnosis**: Count barriers per kernel iteration. Each barrier has 50-100 cycle overhead.

**Solutions**:
- Minimize barrier count (combine phases)
- Use named barriers for subset synchronization
- Use split arrive/wait to overlap independent work
- Eliminate redundant barriers (e.g., double-buffered SLM may remove a barrier that was never blocking)

### 4. Memory Coalescing Issues

**Symptom**: Bandwidth utilization far below peak.

**Diagnosis**: Check access patterns. Consecutive threads should access consecutive addresses.

**Solutions**:
- Ensure contiguous `block_load` patterns
- Use `lsc_load_2d` for 2D tiled access (hardware handles coalescing)
- Avoid scatter/gather when block operations suffice

### 5. Low DPAS/XMX Utilization

**Symptom**: Compute-bound kernel with low XMX busy percentage.

**Diagnosis**: In ISA, check for consecutive DPAS instructions vs intervening sends/movs. VTune XMX busy metric.

**Solutions**:
- Increase DPAS density (more consecutive DPAS without interruption)
- Pipeline data loads with compute (prefetch next tile, compute current)
- Reduce register transpose overhead between DPAS phases
- Use `{Atomic}` annotations where possible (compiler-dependent)

### 6. Instruction Overhead (mov Instructions)

**Symptom**: Large blocks of `mov(4|M0)` instructions in ISA dump, especially for data shuffling.

**Diagnosis**: ISA disassembly analysis. Common in register transpose operations.

**Solutions**:
- Use SLM for data layout transformations (write one layout, read another)
- `lsc_slm_scatter` for transposing via SLM (proved +1.7 TFLOPS in SDP HD=256)
- Separate type conversion from data movement (FP32->FP16 before transpose)

---

## Profiling Tools

### VTune GPU Compute/Media Hotspots

Full-featured profiler for Intel GPUs. Captures hardware counters, timeline, and per-kernel metrics.

```powershell
# Capture GPU Compute Hotspots
vtune -collect gpu-hotspots -knob gpu-sampling-interval=1 -- ./kernel.exe

# Key metrics to examine:
# - XMX Busy %
# - EU Active %
# - L3 Bandwidth
# - SLM Bandwidth
# - GPU Occupancy
# - Stall reasons (Scoreboard, Send, Inst Fetch, etc.)
```

See `intel-gpu-vtune-profiling` skill for detailed VTune workflow.

### GTPin (ISA-Level Profiling)

Instruction-level profiling showing cycle counts per ISA instruction. Essential for micro-optimization.

```powershell
# GTPin profiling
set GTPIN_KIT=C:\path\to\gtpin
%GTPIN_KIT%\Bin\gtpin.exe --installDir %GTPIN_KIT%\Bin ^
    -t LatencyProfiler -- ./kernel.exe
```

See `xe2-gtpin-profiling` skill for GTPin setup and ISA analysis.

### ISA Disassembly

```powershell
# Build with ISA dump
icpx kernel.cpp -o kernel.exe -fsycl -fsycl-targets=spir64_gen `
    -Xs "-device bmg -options -doubleGRF" -O2 -w -save-temps

# Disassemble
ocloc disasm -file <spirv_file> -device bmg -dump isa_dump/

# Key things to look for:
# - DPAS instruction count and density (consecutive DPAS)
# - mov instruction count (especially mov(4|M0) for register shuffles)
# - send instruction count (memory operations)
# - Spill/fill patterns (stack access)
```

---

## Optimization Patterns

### 1. Load/Compute Separation

Separate all data loads from all compute operations. Allows the GPU to pipeline loads and fill memory latency with compute.

```cpp
// BAD: interleaved load/compute
for (int i = 0; i < N; i++) {
    auto data = block_load<half, 128>(ptr + i * 128);
    result[i] = compute(data);
}

// GOOD: all loads first, then all compute (+29% measured)
simd<half, 128> data[N];
for (int i = 0; i < N; i++) data[i] = block_load<half, 128>(ptr + i * 128);
for (int i = 0; i < N; i++) result[i] = compute(data[i]);
```

### 2. Cross-Phase Prefetch

Prefetch data for the next phase while computing the current phase.

```cpp
// During VS GEMM phase, prefetch K for next QK iteration
for (int k = 0; k < kv_len; k += KV_CHUNK) {
    // QK phase
    qk_compute(k);

    // Prefetch next K during VS phase
    if (k + KV_CHUNK < kv_len)
        lsc_prefetch_2d<...>(k_payload);  // Next K tile

    // VS phase
    vs_compute(k);
}
```

### 3. Pipelined K Loads

Load the next data tile during the current DPAS computation.

```cpp
// Load first K tile
auto k_tile = lsc_load_2d<...>(k_payload);

for (int d = 0; d < D_BLKS; d++) {
    // Start next load
    k_payload.set_x((d + 1) * K);
    auto k_next = (d < D_BLKS - 1) ? lsc_load_2d<...>(k_payload) : k_tile;

    // Compute with current tile (overlaps with next load)
    acc = xmx::dpas<...>(acc, k_tile, q_tile[d]);

    k_tile = k_next;
}
```

### 4. Loop Unrolling

```cpp
// Explicit unroll for critical inner loops
#pragma unroll
for (int i = 0; i < TILE_SIZE; i++) {
    acc += data[i] * weights[i];
}
```

`#pragma unroll` is essential for ESIMD inner loops. Without it, the compiler may generate loop overhead that dominates small tile computations.

### 5. SIMD Width Selection

Choose SIMD width based on the operation:

```cpp
// SIMD16 for reductions (reciprocal, sum)
simd<float, 16> inv = 1.0f / sum;  // Single SIMD16 divide

// SIMD32 for mad fusion
// fp32_sum = fp32_sum * delta + local_sum generates SIMD32 mad with 2x throughput

// Match SIMD width to data tile size for zero-cost select operations
simd<half, 128> full_row;
auto slice = full_row.select<16, 1>(offset);  // Zero-cost view, no data movement
```

### 6. Deferred Compensation (Online Softmax)

In flash attention, defer the `A_tile *= delta` correction to before VS GEMM rather than interleaving inside the VS loop:

```cpp
// BAD: interleaved compensation inside VS inner loop
for (int kv = 0; kv < KV_PER_SG; kv++) {
    a_tile[kv] *= delta;  // Breaks DPAS pipeline
    acc = dpas(acc, s_tile, v_tile);
}

// GOOD: deferred compensation before VS phase
for (int ii = 0; ii < Q_ROWS; ii++) a_tile[ii] *= delta;  // All at once
for (int kv = 0; kv < KV_PER_SG; kv++) {
    acc = dpas(acc, s_tile, v_tile);  // Clean DPAS pipeline
}
```

### 7. Type Conversion Before Data Movement

When both type conversion and data shuffling are needed, do conversion first to reduce the data volume for the shuffle.

```cpp
// BAD: transpose fp32 data (4 bytes per element), then convert
// Moves 2x the data during transpose

// GOOD: convert fp32 -> fp16 first, then transpose fp16 data (+14% measured)
simd<half, N> fp16_data = convert<half>(fp32_data);
// Now transpose fp16_data (half the register traffic)
```

---

## Optimization Priority Checklist

Apply optimizations in this priority order (highest impact first):

1. **Eliminate register spill** — Use doubleGRF, reduce tile sizes, reorder operations
2. **Maximize DPAS density** — Consecutive DPAS instructions, pipeline loads
3. **Minimize data movement** — SLM transpose instead of register transpose, type convert before shuffle
4. **Optimize memory access** — block_load over gather, 2D loads for tiled access, prefetch
5. **Reduce barrier count** — Named barriers, split arrive/wait, eliminate redundant barriers
6. **Fine-tune arithmetic** — SIMD reciprocal, mad fusion, vectorized exp
7. **Tune tile sizes** — Balance occupancy vs work per thread vs register pressure

---

## Case Study: SDP HD=256 Optimization Journey

The HD=256 flash attention kernel on BMG illustrates the methodology:

| Stage | TFLOPS | % Peak | Key Optimization |
|-------|--------|--------|-----------------|
| Baseline (VNNI) | 64 | 47% | Initial working kernel |
| Deferred compensation | 70 | 52% | Clean DPAS pipeline |
| Named barriers | 71 | 53% | Small barrier overhead reduction |
| **FP16 before transpose** | **82** | **61%** | **Type convert before shuffle (-50% register traffic)** |
| SIMD reciprocal + mad fusion | 83 | 61% | Fine arithmetic tuning |
| 32-thread WG + pipelined K | 84 | 62% | More DPAS parallelism |
| **lsc_slm_scatter S transpose** | **88** | **65%** | **SLM scatter eliminates ~270 mov instructions** |

Key takeaways:
- The two biggest wins were **data movement optimizations** (FP16 transpose, SLM scatter)
- Arithmetic micro-optimizations gave small but cumulative gains
- Several approaches that seemed promising (double-buffered SLM, early V loads, VNNI interleave) were neutral or regressive
- Always measure; intuition about GPU performance is often wrong

---

## Related Skills

| Skill | Relevance |
|-------|-----------|
| `intel-gpu-vtune-profiling` | Detailed VTune GPU profiling workflow |
| `xe2-gtpin-profiling` | GTPin ISA-level profiling on Xe2 |
| `intel-esimd-base` | Foundational ESIMD programming |
| `esimd-lsc-2d-gather-scatter` | LSC 2D/1D/gather/scatter operations |
| `esimd-lsc-slm` | SLM operations and patterns |
| `xe2-sdp-hd256` | HD=256 SDP kernel optimization case study |
| `xe2-sdp-kernels` | HD=128 SDP kernels |
| `xe2-dpas-patterns` | DPAS/XMX tiling and VNNI layout |
| `xe2-nbarrier-pipelining` | Named barrier arrive/wait patterns |
| `xe3-onednn-fp16-gemm` | oneDNN FP16 GEMM on Xe3 |
| `xe3-esimd-kernels` | ESIMD kernels on Xe3/PTL |
| `sycl-esimd-build` | Build flags, doubleGRF, spill detection |
