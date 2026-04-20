# Optimization History: FP16 GEMM on BMG

Complete record of every optimization experiment, with results and conclusions.

## Starting Point

**fp16_gemm_noif.cpp** — ~109 TFLOPS

Architecture: SYCL ESIMD, 256×256 WG tile, 32 threads per WG (doubleGRF), K-step=32, N_SUBS=2 double-buffer, XMX DPAS 8×8×16 tiles. Payload rebuilt from scratch each call (not CSE'd).

---

## Optimization 1: Payload CSE

**File**: `fp16_gemm_nopf.cpp`
**TFLOPS**: 117.10 (+7.4% over 109T baseline)

**What**: Construct `config_2d_mem_access` once outside K-loop. Only call `set_x()` / `set_y()` inside loop. Eliminates per-call descriptor rebuild.

**Why it worked**: The descriptor rebuild involves multiple XVE ALU ops (surface width/height/pitch packing). Hoisting to pre-loop eliminates ~6 XVE ops per load call → large aggregate savings across 128 K-step iterations × 8 loads per substep.

**Key finding**: XVE activity is the bottleneck (it competes with XMX for EU resources and causes frequency throttling). Reducing it from descriptor rebuild to only coordinate updates was the single largest gain.

---

## Optimization 2: Prefetch Removal

**Incorporated into**: `fp16_gemm_nopf.cpp` (no prefetch variant)
**vs original (with prefetch)**: +7.4%

**What**: Remove all `lsc_prefetch_2d` calls from the kernel.

**Why it worked**: GEMM is compute-bound, not memory-latency-bound. The A/B tiles are streamed from L2 (fits in L2 working set). Adding prefetch instructions adds XVE pressure and instruction count without providing latency benefit. Net result: prefetch hurts at all distances tested (PF_DIST=1..8: −8% to −24%).

---

## Experiment: Distributed Prefetch

**File**: `fp16_gemm_distpf.cpp`
**Results**:
- PF_DIST=0 (baseline): 118.33T
- PF_DIST=1: ~109T (−8%)
- PF_DIST=2..8: −10% to −24%

**What was tried**: Instead of prefetching all tiles at top of K-loop, spread one prefetch slice per substep to reduce instruction-issue spikes. Split into two loop phases to avoid `if` branching overhead.

**Conclusion**: Even distributed prefetch hurts. GEMM is definitively compute-bound on BMG at 4096×4096. Prefetch is ruled out permanently.

---

## Experiment: Walk Order (Z-Order / Hilbert)

**File**: `fp16_gemm_walk.cpp`
**Results**:
- LINEAR (row-major): 117.95T (baseline for this sweep)
- COL_SWIZZLE_2/4/8: −25% (catastrophic)
- Z_ORDER_2: −1.6%
- Z_ORDER_4: −3.4%
- Z_ORDER_8: −7.7%

**What was tried**: Remap WG index to Morton-curve (Z-order) tile order to improve L2 spatial reuse.

**Why it failed**: On BMG 4096×4096 with 256×256 tiles, there are exactly 256 WGs. ALL 256 WGs run concurrently on the GPU (32 DSS × 8 EUs). No WG waits for another → the WG execution order in the ND-range has zero effect on L2 reuse. Walk order optimizations only help when total WGs > GPU capacity so later WGs benefit from L2 warmup by earlier WGs.

**Conclusion**: Walk order tuning is irrelevant at this problem size on BMG.

---

## Experiment: K_STEP Sweep

**Files**: `fp16_gemm_kstep_nopf.cpp`, `fp16_gemm_kstep_nopf2.cpp`
**Results**:
- K_STEP=32: ~117T
- K_STEP=64: ~110T (−5.7%)
- K_STEP=128: ~110T (−6%)
- K_STEP=256: ~111T (−5%)

**What was tried**: Larger K_STEP → fewer loop iterations → less loop overhead.

**Why it failed**: EU instruction scheduler has a finite window. Larger K_STEP unrolls more loads+DPAS into the loop body, exceeding the scheduler window → instruction-issue stalls at substep boundaries. The loop overhead is NOT the bottleneck; the scheduler window exhaustion is.

**Lambda hypothesis (disproved)**: Initially thought lambda `int buf` param prevented GRF specialization of `bb[buf][m]` (dynamic register index). Rewrote with explicit inline code — identical results. Lambda was not the cause.

**Conclusion**: K_STEP=32 is optimal.

---

## Experiment: Induction Variable Optimization

**File**: `fp16_gemm_nopf3.cpp`
**Results**:
- CACHED (baseline): 117.44T (+0.3%)
- L1UC: 43.90T (−62.6% — catastrophic)

**What was tried**: Use `uint32_t ks = K_SUB; ks += K_STEP` as induction variable instead of `k + K_SUB` multiplication.

**Why marginal**: The compiler already performs LICM on `m_start + m*8` (constant within K-loop). The explicit increment form saved 1-2 XVE ops per iter but compiler already handled most cases.

**L1UC lesson**: Using `cache_hint::uncached` for L1 while keeping L2 cached is catastrophic for GEMM inner-loop loads. All tile loads must use `cached/cached`.

---

## Experiment: 2×payA + 2×payB with Alternating Order

**File**: `fp16_gemm_2pay.cpp`
**Result**: 113.79T (−2.8% vs nopf)

**What was tried**: Split A into payA0 (m=0,2) and payA1 (m=1,3), split B into payB0 (n=0,2) and payB1 (n=1,3). Use alternating even/odd load order to save 1 set_x + 1 set_y per substep (10 → 8 XVE ops/substep). Total savings: 512 ops over K=4096.

**Why it failed**:
1. 512 ops saved ≈ 0.01% of total ops — too small to measure.
2. Alternating interleaved access pattern (m=0,2,1,3 instead of 0,1,2,3) likely broke load-tracker prefetch prediction, causing additional latency.
3. 4 live payload objects vs 2 increases GRF pressure on payload descriptors even without spill.

**Note on 4×payA + 4×payB**: Attempted earlier (`fp16_gemm_nopf2.cpp`), caused 3840-byte spill and JIT kernel load failure. Do not attempt.

---

## Summary: What Works

| Technique | Impact | Status |
|-----------|--------|--------|
| Payload CSE (single payA + payB) | +7.4% | ✅ Use always |
| No prefetch | +7.4% (vs original) | ✅ Use always |
| K_STEP=32 | Optimal | ✅ Keep |
| N_SUBS=2 double-buffer | Baseline | ✅ Keep |
| `cached/cached` for all loads | Baseline | ✅ Mandatory |
| doubleGRF enabled | Baseline | ✅ Mandatory |
| WG threads=32 | Optimal | ✅ Keep |
| Index increment (+=) | +0.3% marginal | ✅ Use |
| Random non-zero init | Correctness/validity | ✅ Use always |
| NaN check before rel_rms | Correctness | ✅ Use always |

## Summary: What Doesn't Work

| Technique | Impact | Why |
|-----------|--------|-----|
| Any prefetch (PF_DIST≥1) | −8% to −24% | Compute-bound, adds XVE pressure |
| Z-order / COL_SWIZZLE walk | −2% to −25% | All WGs run concurrently |
| K_STEP > 32 | −6% | EU scheduler window exhaustion |
| L1UC cache hint | −63% | Kills L1 hit rate for streaming |
| 2×payA + 2×payB | −2.8% | Pattern change hurt > XVE savings |
| 4×payA + 4×payB | kernel crash | GRF spill → JIT load failure |
| Lambdas with int buf param | Neutral | Not actually an issue |

## Current Best: ~118 TFLOPS (~87% of 135T peak)

Estimated ceiling: ~120 TFLOPS. The ~4% XVE overhead is structural (loop counter updates, branch), irreducible without algorithm change. Further gains would require architectural changes (e.g., mixed-precision accumulation, pipelining with SLM, or hardware-specific ISA intrinsics not exposed by ESIMD).
