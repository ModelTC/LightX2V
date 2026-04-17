# Barrier + Interleave Optimizations for SDP Kernels

Three micro-optimizations that improve all kernel variants (fp16, bf16, bf16io).
Combined effect: ~1.3-1.5% for fp16, ~3-5% for bf16 (bf16 benefits more due to expensive compensation).

---

## Optimization 1: Barrier Before Compensation

### Before (original)
```
softmax → pack VNNI → SxV: { compensate → SLM load → DPAS }
                              ↑ barrier here (inside SxV, after compensate)
```

### After (optimized)
```
softmax → pack VNNI → barrier → SxV: { compensate → SLM load → DPAS }
                       ↑ barrier moved before compensation
```

### Why it helps
- SLM V data was written by all threads in the PREVIOUS iteration's scatter
- The barrier ensures SLM is safe to read
- By placing barrier BEFORE compensation, the compensation ALU can overlap with SLM block_load latency
- SLM loads issue immediately after barrier; compensation runs on XVE while SLM loads are in flight

### Code pattern
```cpp
barrier();  // SLM safe — moved before compensation

// l=0: compensate, then SxV DPAS
for (int kk = 0; kk < 32; kk++)
  finalOutput.select<32,1>(32*kk) *= compensationTemp.select<32,1>(0);

// SLM loads + DPAS for l=0...
```

---

## Optimization 2: Early Type-Convert of CompensationTemp

### Before (original)
```
fp32SoftMaxCompensation = exp2(...)
fp32SoftMaxTemp *= fp32SoftMaxCompensation        // fp32 multiply
// ... sum reduction (8 iterations) ...
// ... VNNI packing ...
// barrier
compensationTemp = fp32SoftMaxCompensation         // fp32 → fp16/bf16 convert HERE (late)
finalOutput *= compensationTemp                     // compensation multiply
```

### After (optimized)
```
fp32SoftMaxCompensation = exp2(...)
fp32SoftMaxTemp *= fp32SoftMaxCompensation        // fp32 multiply
compensationTemp = fp32SoftMaxCompensation         // fp32 → fp16/bf16 convert HERE (early)
// ... sum reduction (8 iterations) ...            // conversion latency hidden by sum ALU
// ... VNNI packing ...
// barrier
finalOutput *= compensationTemp                     // compensationTemp already ready
```

### Why it helps
- fp32 → fp16/bf16 conversion has ~2-4 cycle latency
- Sum reduction is pure ALU that doesn't depend on compensationTemp
- Moving the conversion before sum reduction hides the latency entirely
- compensationTemp is ready by the time it's needed after the barrier

### Code pattern
```cpp
fp32SoftMaxCompensation = __ESIMD_NS::exp2<float, 16, float>(fp32SoftMaxCompensation);
fp32SoftMaxTemp *= fp32SoftMaxCompensation;

// EARLY convert — before sum reduction
compensationTemp.select<16, 1>(0) = fp32SoftMaxCompensation;   // fp32 → fp16/bf16
compensationTemp.select<16, 1>(16) = fp32SoftMaxCompensation;  // duplicate for 32-wide

// Sum reduction follows (hides conversion latency)
for (int kk = 0; kk < 4; kk++) {
  ttemp.select<32, 1>(32 * kk) = tempOutput.select<32, 1>(64 * kk)
                                + tempOutput.select<32, 1>(64 * kk + 32);
}
// ... more sum reduction ...
```

---

## Optimization 3: Interleave Compensation by L-Group

### Before (original)
```
barrier
compensate ALL 2048 elements of finalOutput
SxV DPAS for ALL output (l=0 and l=1 interleaved)
```

### After (optimized)
```
barrier
compensate l=0 (elements 0..1023)        ← XVE work
SxV DPAS l=0 (32 DPAS calls)             ← XMX busy; l=0 comp already consumed
compensate l=1 (elements 1024..2047)      ← XVE overlaps with l=0 DPAS pipeline drain
SxV DPAS l=1 (32 DPAS calls)
```

### Why it helps
- The SxV DPAS pipeline takes ~8 cycles per call to drain through the systolic array
- After issuing the last DPAS for l=0, the XMX pipeline is still draining
- During this drain time, XVE can compensate l=1 (1024 elements) for free
- Without interleaving, all 2048 compensations happen before any DPAS, creating a longer XVE stall

### SxV structure with l-group interleaving
```cpp
// --- l=0: output elements [0..1023] ---
// Compensate l=0 half
#pragma unroll
for (int kk = 0; kk < 32; kk++)
  finalOutput.select<32, 1>(32 * kk) *= compensationTemp.select<32, 1>(0);

// SxV Block 1 (V rows 0-31), output l=0
#pragma unroll
for (int nn = 0; nn < 2; nn++) {
  // SLM loads + 8 DPAS calls...
}
// SxV Block 2 (V rows 32-63), output l=0
#pragma unroll
for (int nn = 0; nn < 2; nn++) {
  // SLM loads + 8 DPAS calls...
}

// --- l=1: output elements [1024..2047] ---
// Compensate l=1 half (overlaps with l=0 DPAS pipeline drain)
#pragma unroll
for (int kk = 32; kk < 64; kk++)
  finalOutput.select<32, 1>(32 * kk) *= compensationTemp.select<32, 1>(0);

// SxV Block 1 (V rows 0-31), output l=1
// ... same pattern, different SLM offsets (+16*64*sizeof(fp16)) ...
// SxV Block 2 (V rows 32-63), output l=1
// ... same pattern ...
```

### L-group SLM offset pattern
- l=0 reads SLM at base offset (first 64 columns of V in VNNI layout)
- l=1 reads SLM at base offset + `16 * 64 * sizeof(fp16)` (second 64 columns)
- Both use the same ping-pong buffer selection

---

## Combined Impact

### fp16 kernel
| Optimization | Individual gain | Notes |
|-------------|----------------|-------|
| Barrier before comp | ~0.3% | Small because fp16 comp is fast |
| Early convert | ~0.2% | fp32→fp16 already fast |
| L-group interleave | ~0.8% | Main contributor |
| **Combined** | **~1.3-1.5%** | Measured across 8K-16K |

### bf16 kernel
| Optimization | Individual gain | Notes |
|-------------|----------------|-------|
| Barrier before comp | ~0.5% | Larger because bf16 comp takes longer |
| Early convert | ~0.5% | fp32→bf16 has more latency to hide |
| L-group interleave | ~2-4% | Much larger because bf16 comp is 3x cost |
| **Combined** | **~3-5%** | bf16 comp dominates, so overlap helps more |

### bf16io kernel
The bf16io kernel uses fp16 compensation (same as fp16 kernel), so the benefit is ~1.3-1.5%.
All three optimizations are already included in the bf16io kernel.

---

## Applicability

These optimizations apply to any flash attention kernel with:
1. Online softmax (running max + compensation)
2. SLM-based V tile sharing (barrier between iterations)
3. DPAS-based SxV matmul (XMX/XVE parallelism opportunity)

They do NOT apply to:
- Decode kernels (no SLM ping-pong, different structure)
- Non-DPAS attention (no XMX to overlap with)
