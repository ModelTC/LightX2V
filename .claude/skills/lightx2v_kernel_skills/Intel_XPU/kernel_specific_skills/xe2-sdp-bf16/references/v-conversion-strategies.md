# V Conversion Strategies: All Approaches Tested

## Problem Statement

The bf16io hybrid kernel needs V values converted from bf16 to fp16 before SxV DPAS.
Each iteration loads one V tile: 64 rows × 128 cols = 8192 elements.
But only 512 elements per thread (16 threads share the 64×128 tile via SLM).

The V data flow is:
1. V loaded from global memory (bf16 bits in fp16-typed 2D load)
2. V converted bf16 → fp16 (512 elements per thread)
3. V scattered to SLM (ping-pong buffer)
4. SLM barrier
5. V loaded from SLM for SxV DPAS (fp16)

The question: where in the loop to place step 2 (conversion)?

---

## Strategy 1: Between V-Load and SLM Scatter (Original)

```
QK DPAS → V load → V CONVERT → SLM scatter → barrier → softmax → SxV DPAS
```

**Result: 1.112x at 8K, 1.120x at 16K**

V conversion sits on the critical path between V load and SLM scatter.
The 512-element conversion takes ~32 XVE instructions (16 elements/instruction × 2 cycles each).
At ~1.85 GHz, this is ~35 ns — small but adds up over 128+ iterations.

The problem: conversion is **sequential** — nothing else can run in parallel because:
- V load must complete before conversion starts (data dependency)
- SLM scatter needs converted data (data dependency)
- No XMX work available to overlap with

---

## Strategy 2: Interleaved in QK DPAS (Failed)

```
V load → QK DPAS + V CONVERT interleaved → SLM scatter → barrier → softmax → SxV DPAS
```

Moved V 2D load to BEFORE QK DPAS, then interleaved conversion chunks with QK DPAS calls.

**Result: 1.092x at 8K (improved), 1.139x at 16K (WORSE)**

Mixed results because:
- **V 2D load is synchronous/blocking** on Xe2 — it stalls the thread until data arrives
- Moving V load before QK means QK start is delayed by V load latency (~50-100ns)
- At small sizes (8K), this latency is absorbed by QK's longer duration
- At large sizes (16K), the V load latency becomes a bottleneck because QK is already saturating memory bandwidth

**Key lesson: Never move a blocking 2D load earlier in the pipeline unless you can prove the latency will be absorbed.**

The interleaving itself worked (XVE conversion overlapped with XMX DPAS), but the V load position change negated the benefit.

---

## Strategy 3: Interleaved in SxV DPAS (WINNER)

```
QK DPAS → V load → softmax → barrier → SxV DPAS + V CONVERT interleaved → SLM scatter
```

V load stays in its original position (after QK). V conversion moves to inside SxV DPAS.

**Result: 0.954-0.976x (FASTER than fp16!)**

### Why this works

1. **V conversion result is for the NEXT iteration**, not the current one:
   - Current iteration: SxV uses V data already in SLM (loaded in previous iteration)
   - V load in current iteration fetches data for NEXT iteration's SLM
   - So converting V during current SxV has no data dependency conflict

2. **SxV has 64 DPAS calls** — plenty of XMX latency to hide conversion:
   - First half: 32 DPAS calls + 32 conversion chunks (1 chunk per DPAS)
   - Second half: 32 DPAS calls, no conversion (already done)
   - Each chunk converts 16 bf16 elements to fp16 via fp32 intermediate

3. **V load stays after QK** — no extra blocking latency added to QK start

4. **Conversion finishes before SLM scatter** — scatter happens at end of SxV section

### Implementation detail

```cpp
// First half of SxV: 32 DPAS + 32 V conversion chunks
auto vAsBf16 = fp16VState.template bit_cast_view<bf16>();
#pragma unroll
for (int nn = 0; nn < 2; nn++) {        // 2 blocks of 16 V rows
  #pragma unroll
  for (int l = 0; l < 2; l++) {          // 2 SLM loads per block
    // SLM block loads for current SxV...
    #pragma unroll
    for (int ll = 0; ll < 8; ll++) {     // 8 DPAS per SLM load
      // fp16 DPAS (executes on XMX, takes ~8 cycles systolic)
      ccTile = dpas<8, 8, fp16, fp16, fp16, fp16>(...);

      // V conversion: 1 chunk of 16 elements (XVE, runs while XMX busy)
      {
        int32_t ci = nn * 16 + l * 8 + ll;
        simd<float, 16> cvt = vAsBf16.select<16, 1>(16 * ci);
        fp16VState.select<16, 1>(16 * ci) = cvt;
      }
    }
  }
}
// Second half: 32 DPAS, no conversion needed
```

### Chunk math
- 32 DPAS calls in first half × 1 conversion chunk each = 32 chunks
- 32 chunks × 16 elements = 512 elements = exactly one V tile per thread
- Second half has 32 DPAS calls with no extra work = full XMX throughput

---

## Strategy Comparison Summary

| Strategy | V load position | Conversion position | 8Kx8K | 16Kx16K | Why |
|----------|----------------|--------------------|---------|---------|----|
| Original | After QK | Between V-load and softmax | 1.112x | 1.120x | On critical path |
| QK interleave | Before QK | During QK DPAS | 1.092x | 1.139x | V load blocks QK start |
| **SxV interleave** | After QK | During SxV DPAS | **0.976x** | **0.964x** | Free — XVE ∥ XMX |

---

## Rules for XVE/XMX Overlap

1. **Only works during DPAS calls** — XMX must be actively processing a DPAS for XVE to run in parallel
2. **Keep XVE work short** per DPAS call — one conversion chunk (16 elements, ~2 instructions) fits easily in DPAS systolic latency (~8 cycles)
3. **No data dependency** between conversion input and DPAS input — conversion operates on V (next iteration), DPAS operates on SLM V (current iteration)
4. **Compiler must not serialize** — `#pragma unroll` ensures the compiler sees the full loop and can schedule XVE/XMX in parallel
5. **Don't move blocking loads** to enable interleaving — the load latency will often exceed the conversion savings
