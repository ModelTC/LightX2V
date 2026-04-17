---
name: xe2-sdp-bf16
description: Use this skill when writing, optimizing, or debugging BF16 Flash Attention SDP kernels targeting Intel Xe2 (Lunar Lake/LNL, Battlemage/BMG) GPU using SYCL ESIMD. Xe2 is the GPU architecture; LNL and BMG are product names. Covers the bf16 and bf16io (hybrid) kernel architectures, bf16 ALU limitations on Xe2, type-conversion cost hiding, compensation strategies, DPAS type selection, and all optimization discoveries from the bf16 kernel development journey. Use whenever the user mentions bf16, bfloat16, mixed-precision SDP, or type conversion overhead in attention kernels.
---

# Xe2 (Lunar Lake/LNL, Battlemage/BMG) ESIMD Flash Attention — BF16 Kernels

Specialized knowledge for bf16 and bf16io (hybrid) flash attention on Intel Xe2 (Lunar Lake/LNL, Battlemage/BMG).
Builds on the fp16 SDP skill (`xe2-sdp-kernels`). This file covers bf16-specific rules and discoveries.

---

## Quick-Reference: The Three Best Kernels

| Kernel | I/O type | QK DPAS | SxV DPAS | Accumulator | Compensation | Perf vs fp16 | Accuracy (rms) |
|--------|----------|---------|----------|-------------|--------------|--------------|-----------------|
| **fp16 opt** | fp16 | fp16 | fp16 | fp16 | fp16 (native) | 1.000x baseline | 1.2e-3 |
| **bf16** | bf16 | bf16 | bf16 | bf16 | bf16 (slow!) | ~1.17x slower | 9.8e-3 |
| **bf16io** | bf16 | bf16 | fp16 | fp16 | fp16 (native) | **0.96x faster** | 2.1e-3 |

**Winner: bf16io** — accepts bf16 tensors, runs faster than fp16, better accuracy than bf16.

---

## Critical Rule #1: Xe2 Has No Native BF16 ALU

On Battlemage (Xe2), the XVE (vector engine) has **no native bf16 arithmetic instructions**.
Every bf16 multiply/add goes through: `bf16 → fp32 → multiply → fp32 → bf16` (~3x cost vs native fp16).

**Impact on SDP kernels:**
- `finalOutput *= compensationTemp` where both are `simd<bf16, N>` generates ~3x more XVE instructions than the same operation in fp16
- This is the **primary reason** the pure bf16 kernel is 17% slower than fp16
- The bf16io kernel avoids this by keeping the accumulator and compensation in fp16

**Rule: Never use bf16 for ALU-heavy operations on Xe2.** Use bf16 only where XMX DPAS consumes it directly (QK matmul). For everything else (accumulator, compensation, softmax weights), use fp16 or fp32.

---

## Critical Rule #2: DPAS Type Selection

XMX systolic array always computes internally in fp32, regardless of input type. The throughput is identical for:
```cpp
dpas<8, 8, float, float, bf16, bf16>(...)  // bf16 inputs, fp32 accum
dpas<8, 8, float, float, fp16, fp16>(...)  // fp16 inputs, fp32 accum
dpas<8, 8, bf16,  bf16,  bf16, bf16>(...)  // bf16 inputs, bf16 accum
dpas<8, 8, fp16,  fp16,  fp16, fp16>(...)  // fp16 inputs, fp16 accum
```

**Choose DPAS types based on the surrounding ALU**, not DPAS throughput:
- QK DPAS: `dpas<8,8,float,float,bf16,bf16>` — bf16 inputs avoid Q/K conversion, fp32 accum for softmax precision
- SxV DPAS: `dpas<8,8,fp16,fp16,fp16,fp16>` — fp16 accum enables native fp16 compensation multiply

---

## Critical Rule #3: V Conversion Interleaving (Zero-Cost Type Conversion)

Converting V from bf16→fp16 costs ~512 XVE instructions per loop iteration (32 chunks × 16 elements).
On Xe2, XVE and XMX are **independent execution units** that can run in parallel.

**The discovery:** Interleave one V conversion chunk after each SxV DPAS call. While XMX processes the DPAS (systolic latency ~8 cycles), XVE performs the bf16→fp16 conversion for free.

```cpp
// First half of SxV: 32 DPAS + 32 V conversions interleaved
auto vAsBf16 = fp16VState.template bit_cast_view<bf16>();
#pragma unroll
for (int nn = 0; nn < 2; nn++) {
  #pragma unroll
  for (int l = 0; l < 2; l++) {
    // SLM block loads...
    #pragma unroll
    for (int ll = 0; ll < 8; ll++) {
      // fp16 DPAS call (executes on XMX)
      ccTile = dpas<8, 8, fp16, fp16, fp16, fp16>(...);
      // V conversion: 1 chunk per DPAS (XVE works while XMX busy)
      {
        int32_t ci = nn * 16 + l * 8 + ll;
        simd<float, 16> cvt = vAsBf16.select<16, 1>(16 * ci);
        fp16VState.select<16, 1>(16 * ci) = cvt;
      }
    }
  }
}
// Second half of SxV: 32 DPAS, no conversion (already done)
```

**Constraint:** Only 32 conversion chunks fit in the first half (32 DPAS calls). The second half has 32 more DPAS calls with no conversion work — they execute at full speed. This is exactly enough: 32 chunks × 16 elements = 512 elements = one V tile.

**Performance impact:** Reduces bf16io overhead from 12% to **negative** (bf16io becomes 4% faster than fp16).

---

## Critical Rule #4: Where NOT to Put V Conversion

Three placement strategies were tested. Only one works:

| Strategy | Where V conversion happens | Result |
|----------|--------------------------|--------|
| **Between V-load and softmax** (original) | After V 2D load, before softmax | 1.12x slower (conversion on critical path) |
| **Interleaved in QK DPAS** | During QK matmul | Mixed: 8K improved, 16K regressed (V load must move before QK, adding blocking latency) |
| **Interleaved in SxV DPAS** (winner) | During S×V matmul | **0.96x** (conversion fully hidden) |

**Why SxV interleaving wins:**
1. V conversion result is for the **next** iteration's SLM scatter, not the current SxV — no data dependency
2. V load stays after QK (no extra blocking latency)
3. SxV has 64 DPAS calls — plenty of XMX latency to hide 32 conversion chunks
4. Conversion finishes before SLM scatter at end of SxV section

**Why QK interleaving fails at large sizes:**
- V 2D load must move BEFORE QK DPAS (so conversion can overlap with QK)
- 2D loads are synchronous/blocking — V load latency delays QK start
- At large kv_len, this added latency dominates the conversion savings

---

## Critical Rule #5: Barrier + Interleave Optimization

Three optimizations that apply to ALL kernel variants (fp16, bf16, bf16io):

### 1. Barrier before compensation (not after)
```
barrier();                              // SLM protected, loads can issue immediately
finalOutput *= compensationTemp;        // ALU overlaps with SLM block_load latency
// S×V DPAS begins...
```
Moving barrier earlier allows compensation ALU to run while SLM loads are in flight.

### 2. Early type-convert of compensationTemp
Convert `fp32SoftMaxCompensation → fp16/bf16 compensationTemp` **before** the sum reduction.
Sum reduction is pure ALU — it overlaps with the type-conversion latency.
```cpp
// Convert BEFORE sum reduction (latency hidden by sum ALU)
compensationTemp.select<16, 1>(0) = fp32SoftMaxCompensation;
compensationTemp.select<16, 1>(16) = fp32SoftMaxCompensation;
// Sum reduction follows (overlaps with convert)
```

### 3. Interleave compensation by output-dim half (l-group)
Split finalOutput into two halves (l=0: elements 0..1023, l=1: elements 1024..2047).
Compensate l=0, then do l=0 SxV DPAS. While DPAS pipeline drains, compensate l=1.
```
compensate l=0 [1024 elements]
SxV DPAS l=0   [32 DPAS calls — XMX busy]
compensate l=1 [1024 elements — overlaps with l=0 DPAS pipeline]
SxV DPAS l=1   [32 DPAS calls]
```

**Measured improvement:** ~1.3-1.5% for fp16, ~3-5% for bf16 (bf16 benefits more because compensation is more expensive).

---

## Critical Rule #6: fp32 Accumulator — DO NOT USE

Attempted: `simd<float, 2048> finalOutput` (fp32 accumulator for bf16 kernel).

**Result: 4.5x slowdown due to catastrophic register spilling.**

fp32 accumulator needs 128 GRFs (vs 64 for bf16/fp16). This pushes total GRF usage past the 256 doubleGRF budget. The compiler spills to memory, destroying performance.

**The fp32 accumulator approach is fundamentally incompatible with the SDP tile sizes on Xe2.** The bf16io hybrid approach (fp16 accumulator) is the correct solution.

---

## Kernel Architecture: bf16 (Pure BF16)

File: `flash.attn.b.mha128.bf16.h`
Entry: `flashAttnBMha128Bf16Precomputed()`

- **Q/K/V/Output**: all bf16 in memory
- **QK DPAS**: `dpas<8,8,float,float,bf16,bf16>` — fp32 accumulator for softmax precision
- **SxV DPAS**: `dpas<8,8,bf16,bf16,bf16,bf16>` — bf16 accumulator
- **Compensation**: `finalOutput(bf16) *= compensationTemp(bf16)` — SLOW on Xe2 (no native bf16 ALU)
- **Softmax weights**: packed as bf16 VNNI
- **V handling**: loaded as fp16-typed (same 16-bit hardware), bit_cast_view to bf16 for DPAS
- **Performance**: ~1.17x slower than fp16 (compensation overhead)
- **Accuracy**: rms ~1e-2 (lower than fp16 due to bf16 truncation in accumulator)

### When to use bf16 kernel
- When you need a simple bf16 baseline for comparison
- When the rest of the pipeline is already fully bf16 and you want type consistency
- NOT recommended for production (bf16io is strictly better)

---

## Kernel Architecture: bf16io (Hybrid — RECOMMENDED)

File: `flash.attn.b.mha128.bf16io.h`
Entry: `flashAttnBMha128Bf16IoPrecomputed()`

- **Q/K/V/Output**: all bf16 in memory (same interface as bf16 kernel)
- **QK DPAS**: `dpas<8,8,float,float,bf16,bf16>` — Q/K stay bf16, no conversion needed
- **SxV DPAS**: `dpas<8,8,fp16,fp16,fp16,fp16>` — fp16 inputs, fp16 accumulator
- **Compensation**: `finalOutput(fp16) *= compensationTemp(fp16)` — NATIVE on Xe2
- **Softmax weights**: packed as fp16 VNNI (same as fp16 kernel)
- **V handling**: loaded as fp16-typed (bf16 bits), converted bf16→fp16 interleaved in SxV DPAS
- **Output**: fp16 accumulator → fp32 → bf16 at the very end
- **Performance**: ~0.96x (4% FASTER than fp16)
- **Accuracy**: rms ~2e-3 (5x better than bf16, close to fp16)

### Why bf16io is faster than fp16
1. Q stays bf16 (no fp16→bf16 conversion needed for QK DPAS — saves nothing, but also costs nothing)
2. K stays bf16 (same — bf16 bits loaded as fp16-typed, bit_cast_view to bf16)
3. V conversion is free (hidden behind SxV DPAS XMX latency)
4. fp16 compensation is native (same as fp16 kernel)
5. Net effect: same compute as fp16, but slightly less memory traffic overhead from Q/K being 16-bit already

### Data flow diagram
```
Q (bf16) ──────────────────────┐
K (bf16) ──────────────────────┤
                               ▼
                    QK DPAS (bf16×bf16→fp32)
                               │
                               ▼
                         Softmax (fp32)
                               │
                               ▼
                    Pack weights → fp16 VNNI
                               │
V (bf16) ─── load as fp16 ────┐│
    │                         ││
    └─ convert bf16→fp16 ─────┤│  ← interleaved in SxV DPAS (free)
       (during SxV DPAS)      ▼▼
                    SxV DPAS (fp16×fp16→fp16)
                               │
                               ▼
                    fp16 accumulator
                               │
                               ▼
                    normAlpha × (1/sum) in fp32
                               │
                               ▼
                         Output (bf16)
```

---

## Optimization History: What Was Tried and Why

### Approach 1: Pure bf16 kernel
- All DPAS in bf16, bf16 accumulator, bf16 compensation
- Result: 1.17x slower (bf16 compensation is ~3x cost on Xe2)
- Lesson: bf16 ALU operations are the bottleneck, not DPAS

### Approach 2: fp32 accumulator (bf16 v2)
- Changed `simd<bf16, 2048>` → `simd<float, 2048>` to make compensation native fp32
- Result: 4.5x slower (catastrophic register spilling, 128→64 extra GRFs)
- Lesson: fp32 accumulator doesn't fit in Xe2's 256 doubleGRF budget for SDP tiles

### Approach 3: bf16io all-convert
- Convert ALL of Q, K, V from bf16→fp16 at load time
- Use pure fp16 kernel internally
- Result: 1.9x slower (Q+K conversion = 8192 elements, dominates)
- Lesson: Converting Q and K is too expensive; they appear in the QK inner loop

### Approach 4: bf16io K-only convert
- Q stays bf16 for QK DPAS, K converted bf16→fp16 for fp16 QK DPAS
- Result: still slow — K is 4096 elements per iteration, loaded fresh each time
- Lesson: K conversion can't be amortized (loaded fresh each QK iteration)

### Approach 5: bf16io hybrid (V-only convert)
- Q/K stay bf16 for bf16 QK DPAS
- Only V converted bf16→fp16 (512 elements per iteration)
- SxV uses fp16 DPAS + fp16 accumulator
- Result: 1.12x slower (V conversion on critical path)
- Lesson: even 512 conversions per iteration matter when on the critical path

### Approach 6: bf16io hybrid + V conversion in SxV (WINNER)
- Same as approach 5, but V conversion interleaved inside SxV DPAS calls
- XVE converts while XMX processes DPAS — zero cost
- Result: **0.96x** (faster than fp16!)
- Lesson: XVE/XMX parallelism is the key to hiding type conversion

---

## Correctness Testing

### Thresholds
| Kernel | max_diff threshold | Expected rms |
|--------|-------------------|--------------|
| fp16 opt | 0.05 | ~1.2e-3 |
| bf16 | 0.15 | ~1.0e-2 |
| bf16io | 0.15 | ~2.0e-3 |

bf16 has wider threshold because bf16 truncation (8-bit mantissa vs fp16's 10-bit) accumulates through the SxV multiply-accumulate chain.

### CPU reference must match input type
Always quantize the CPU reference inputs to match the kernel's input type:
```cpp
// For bf16 kernels: reference uses bf16-quantized values
std::vector<bf16> h_Q_bf16(q_size);
for (size_t i = 0; i < q_size; i++) h_Q_bf16[i] = bf16(h_Q_f32[i]);
// Convert back to fp32 for CPU reference
std::vector<float> h_Q_ref(q_size);
for (size_t i = 0; i < q_size; i++) h_Q_ref[i] = (float)h_Q_bf16[i];
cpu_sdp_reference(h_Q_ref.data(), ...);
```
Without this, you're comparing bf16 kernel output against fp32-precision reference — the error will always exceed threshold.

---

## Performance Results (Xe2/BMG, 32 heads, HD=128, non-causal)

| Config | fp16 opt (ms) | Roofline% | bf16 (ratio) | bf16io (ratio) |
|--------|--------------|-----------|--------------|----------------|
| 8Kx8K | 9.93 | 82.4% | x1.169 | **x0.976** |
| 8Kx16K | 19.67 | 83.1% | x1.170 | **x0.966** |
| 16Kx16K | 39.21 | 83.4% | x1.172 | **x0.964** |

bf16io consistently 3-4% faster than fp16 across all sizes.

---

## Assets (ready to compile)

```bash
icpx <file>.cpp -o <file>.exe -fsycl -fsycl-targets=spir64_gen -Xs "-device bmg -options -doubleGRF" -O3
```

| Asset | Purpose |
|-------|---------|
| `assets/flash.attn.b.mha128.bf16.h` | Pure bf16 kernel (baseline) |
| `assets/flash.attn.b.mha128.bf16io.h` | **Hybrid bf16io kernel (recommended)** |
| `assets/flash.attn.b.mha128.fp16.opt.h` | Optimized fp16 kernel (barrier+interleave) |
| `assets/test_sdp_best.cpp` | Unified test: correctness + perf for all 3 kernels |

---

## Reference files

| File | Contents |
|------|----------|
| `references/bf16-xe2-alu-analysis.md` | Why bf16 ALU is slow on Xe2, instruction counts, cycle analysis |
| `references/v-conversion-strategies.md` | All V conversion placements tested, why SxV interleaving wins |
| `references/barrier-interleave-opt.md` | The 3 barrier+interleave optimizations with measured impact |
