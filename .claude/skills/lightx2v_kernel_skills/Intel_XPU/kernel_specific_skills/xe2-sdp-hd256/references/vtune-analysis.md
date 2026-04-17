# VTune Performance Analysis: HD=256 Flash Attention SDP Kernel on Intel Battlemage

**Configuration:** N=16384, d=256, Hq=32, batch implied by grid
**FLOPS formula:** `4 * 16384 * 16384 * 256 * 32 + 2 * 16384 * 16384 * 32 = 8,813,272,891,392`
**Platform:** Intel Battlemage GPU | **Roofline Peak:** 135 TFLOPS
**Measurement:** 100 kernel instances per version, averaged

---

## 1. Overview Table

| # | Version | Kernel | LocalWS | AvgTime (s) | TFLOPS | XMX% | Active% | Stalled% | Top Stalls |
|---|---------|--------|---------|-------------|--------|------|---------|----------|------------|
| 01 | baseline | benchmark_reversed | 1x16 | 0.125037 | 70.5 | 25.6 | 50.1 | 48.3 | SBID 62.1%, Barrier 38.4% |
| 02 | xprefetch | benchmark_v2 | 1x32 | 0.130439 | 67.6 | 24.5 | 56.9 | 41.8 | SBID 22.0%, Barrier 5.1% |
| 03 | deferred | benchmark_v2 | 1x32 | 0.126074 | 69.9 | 25.3 | 57.3 | 41.5 | SBID 24.6%, Barrier 4.4% |
| 04 | 70tflops | benchmark_v2 | 1x32 | 0.126053 | 69.9 | 25.4 | 57.3 | 41.5 | SBID 24.6%, Barrier 4.4% |
| 05 | nbarrier | benchmark_v2 | 1x32 | 0.122386 | 72.0 | 26.1 | 58.7 | 39.8 | SBID 21.4%, Barrier 4.2% |
| 06 | 72tflops | benchmark_v2 | 1x32 | 0.123061 | 71.6 | 26.0 | 58.5 | 40.0 | SBID 21.6%, Barrier 4.2% |
| 07 | s_preload | benchmark_v2 | 1x32 | 0.121714 | 72.4 | 26.3 | 59.0 | 39.6 | SBID 21.4%, Barrier 4.4% |
| 08 | fp16_transpose | benchmark_v2 | 1x32 | 0.106186 | 83.0 | 30.1 | 51.1 | 47.5 | SBID 24.2%, Barrier 3.9% |
| 09 | simd_inv | benchmark_v2 | 1x32 | 0.106130 | 83.0 | 30.1 | 51.3 | 47.4 | SBID 22.9%, Barrier 3.9% |
| 10 | mad_fusion | benchmark_v2 | 1x32 | 0.104404 | 84.4 | 30.6 | 50.7 | 48.0 | SBID 21.3%, DistAcc 7.6% |
| 11 | 84tflops_pipelined | benchmark_v2 | 1x32 | 0.105605 | 83.5 | 30.3 | 51.5 | 47.2 | SBID 21.9%, Barrier 3.7% |
| 12 | 86tflops_pipeline | benchmark_v2 | 1x32 | 0.103596 | 85.1 | 30.8 | 52.1 | 46.5 | SBID 22.1%, IFetch 4.5% |
| 13 | s_scatter | benchmark_v2 | 1x32 | 0.100229 | 87.9 | 31.9 | 49.0 | 49.6 | SBID 25.5%, Send 4.0% |

**Note on TFLOPS calculation:** All values computed as `8,813,272,891,392 / AvgTime / 1e12`. Version names in the CSV (70tflops, 72tflops, etc.) are consistent with these calculations. The version 01 baseline already runs at ~70.5 TFLOPS under this formula; the "49 TFLOPS" referenced externally may correspond to a prior version not captured in this CSV.

---

## 2. Phase-by-Phase Analysis

### Phase 1: Architecture Redesign (01 -> 07, ~70.5 -> 72.4 TFLOPS)

#### 01 -> 02: Baseline to Extended Prefetch (+XPrefetch, WG 16->32 threads)

| Metric | 01 (baseline) | 02 (xprefetch) | Delta |
|--------|--------------|----------------|-------|
| AvgTime (s) | 0.125037 | 0.130439 | +4.3% slower |
| TFLOPS | 70.5 | 67.6 | -4.1% |
| Active% | 50.1 | 56.9 | +6.8pp |
| Stalled% | 48.3 | 41.8 | -6.5pp |
| XMX% | 25.6 | 24.5 | -1.1pp |
| Barrier stall% | 38.4 | 5.1 | -33.3pp |
| SBID stall% | 62.1 | 22.0 | -40.1pp |
| L3 Read GB/s | 284.9 | 898.0 | +215% |
| ALU0 instr | 970B | 1238B | +27.6% |

**Analysis:** This transition is extraordinary in its stall profile transformation. The move from 1x16 to 1x32 workgroup size and the switch to `benchmark_v2` with extended prefetch **demolished barrier stalls** from 38.4% to 5.1% and cut SBID stalls from 62.1% to 22.0%. The GPU threads are now dramatically more active (+6.8pp). However, throughput actually *decreased* by 4.3%. Why?

The L3 read bandwidth exploded from 285 to 898 GB/s — a 3.15x increase. The new architecture is prefetching aggressively, flooding the memory subsystem. ALU0 instructions rose 27.6% (more bookkeeping for prefetch management). Despite better utilization of execution slots, the kernel is doing more total work per invocation, and the XMX units are slightly less utilized (25.6% -> 24.5%). This version traded raw efficiency for a better pipeline structure — an investment that pays off in later versions.

**Why L3 bandwidth jumped 3x:** The 1x16 baseline likely used a tighter data reuse pattern within a smaller workgroup, keeping data in registers/SLM. The 1x32 version with extended prefetch issues far more L3 read requests to preload K/V tiles ahead of computation, resulting in much higher observed bandwidth. The L3 miss ratio simultaneously dropped from 5.6% to 1.2%, confirming that the prefetching is effective — almost all reads hit L3 rather than going to DRAM.

#### 02 -> 03: Deferred Compensation

| Metric | 02 | 03 | Delta |
|--------|----|----|-------|
| AvgTime (s) | 0.130439 | 0.126074 | -3.3% faster |
| TFLOPS | 67.6 | 69.9 | +3.4% |
| Active% | 56.9 | 57.3 | +0.4pp |
| SBID stall% | 22.0 | 24.6 | +2.6pp |
| Send stall% | 4.4 | 2.7 | -1.7pp |
| L3 Read GB/s | 898.0 | 972.2 | +8.3% |

**Analysis:** Deferred compensation reorganizes when numerical correction terms are applied, batching them rather than interleaving with main computation. This reduced Send stalls by 1.7pp (fewer synchronization points for intermediate results) and boosted L3 bandwidth by 8.3% through better access pattern locality. SBID stalls increased slightly, suggesting the deferred computation creates longer dependency chains in exchange for fewer pipeline bubbles overall. Net: 3.4% throughput gain.

#### 03 -> 04: Cleanup (70tflops_best)

| Metric | 03 | 04 | Delta |
|--------|----|----|-------|
| AvgTime (s) | 0.126074 | 0.126053 | -0.02% |
| ALU0 instr | 1234.7B | 1234.5B | -0.02% |

**Analysis:** Essentially identical. This version represents a clean snapshot of the code at 70 TFLOPS. Minor instruction count reduction suggests trivial cleanup (dead code removal, register reallocation). No meaningful metric changes.

#### 04 -> 05: Named Barriers

| Metric | 04 | 05 | Delta |
|--------|----|----|-------|
| AvgTime (s) | 0.126053 | 0.122386 | -2.9% faster |
| TFLOPS | 69.9 | 72.0 | +3.0% |
| SBID stall% | 24.6 | 21.4 | -3.2pp |
| Barrier stall% | 4.4 | 4.2 | -0.2pp |
| Send stall% | 2.7 | 1.8 | -0.9pp |
| L3 Read GB/s | 972.2 | 984.9 | +1.3% |

**Analysis:** Named barriers allow finer-grained synchronization — instead of all threads waiting on a global barrier, subsets of threads can synchronize independently. This reduced SBID stalls by 3.2pp and Send stalls by 0.9pp, indicating that threads waiting for data from specific producers can now proceed as soon as their specific dependency is met rather than waiting for all producers. Clean 3% improvement with no architectural downsides.

#### 05 -> 06: Tuning (72tflops)

| Metric | 05 | 06 | Delta |
|--------|----|----|-------|
| AvgTime (s) | 0.122386 | 0.123061 | +0.6% slower |
| ALU0 instr | 1234.4B | 1236.9B | +0.2% |
| L3 Write GB/s | 3.59 | 3.96 | +10.3% |

**Analysis:** Slight regression — possibly a tuning snapshot that traded something for increased L3 write bandwidth. The additional ALU0 instructions and slight slowdown suggest an experimental change that did not improve performance. Kept as a reference point.

#### 06 -> 07: S Tile Preload

| Metric | 06 | 07 | Delta |
|--------|----|----|-------|
| AvgTime (s) | 0.123061 | 0.121714 | -1.1% faster |
| TFLOPS | 71.6 | 72.4 | +1.1% |
| Active% | 58.5 | 59.0 | +0.5pp |
| SBID stall% | 21.6 | 21.4 | -0.2pp |
| Send stall% | 1.8 | 2.2 | +0.4pp |
| ALU0 instr | 1236.9B | 1234.3B | -0.2% |

**Analysis:** Preloading S tiles into SLM or registers before they are needed hides some latency. Modest 1.1% gain with slightly better activity and tiny SBID reduction. Send stalls increased slightly, likely because the preload issues additional send instructions. This was the end of what could be achieved by rearranging the existing computation — the kernel was bottlenecked on ALU operations that needed fundamental rethinking.

**Phase 1 Summary:** From 70.5 to 72.4 TFLOPS (+2.7%). The dramatic stall profile improvement (SBID 62.1% -> 21.4%, Barrier 38.4% -> 4.4%) was achieved through architectural changes (workgroup size, prefetch, named barriers), but this translated to only modest throughput gains because the ALU0 instruction count remained stubbornly at ~1234B — the real bottleneck was instruction volume, not scheduling.

---

### Phase 2: The Breakthrough (08, 72.4 -> 83.0 TFLOPS, +14.7%)

#### 07 -> 08: FP16 Transpose — Separating F32->FP16 Conversion from Strided Gather

| Metric | 07 | 08 | Delta |
|--------|----|----|-------|
| AvgTime (s) | 0.121714 | 0.106186 | -12.8% faster |
| TFLOPS | 72.4 | 83.0 | +14.7% |
| **ALU0 instr** | **1234.3B** | **429.4B** | **-65.2%** |
| **ALU1 instr** | **1291.3B** | **868.6B** | **-32.7%** |
| XMX% | 26.3 | 30.1 | +3.8pp |
| Active% | 59.0 | 51.1 | -7.9pp |
| Stalled% | 39.6 | 47.5 | +7.9pp |
| ALU0% | 18.9 | 7.5 | -11.4pp |
| L3 Read GB/s | 986.3 | 1143.3 | +15.9% |
| SBID stall% | 21.4 | 24.2 | +2.8pp |

**This is the single most impactful optimization in the entire sequence.**

**What happened:** In versions 01-07, the S tile (softmax attention scores, shape ~[seq, seq]) needed to be transposed and converted from F32 to FP16 before being fed into the second DPAS (S * V matmul). The old approach interleaved the F32->FP16 type conversion with a strided register gather to perform the transpose, requiring an enormous number of register-to-register mov instructions via ALU0.

Version 08 separated these two operations: first convert F32->FP16 in place (cheap, pipelined), then perform the transpose on the already-narrower FP16 data (half the register moves since each element is 2 bytes instead of 4). This single change eliminated **805 billion ALU0 instructions** (65.2% reduction) and **423 billion ALU1 instructions** (32.7% reduction).

**Why Active% dropped but TFLOPS rose:** This is the critical insight. Active% measures what fraction of cycles have *any* EU thread executing. With 1234B ALU0 instructions, the EUs were constantly busy doing register shuffles. Removing those instructions means:
1. The kernel finishes 12.8% faster (wall clock)
2. The EUs spend less time on non-productive ALU work
3. XMX% rose from 26.3% to 30.1% — the XMX units now occupy a larger fraction of the shorter runtime
4. But the "Active%" metric dropped because many of those eliminated ALU cycles were "active but useless"

The SBID stall% increased from 21.4% to 24.2% because with fewer ALU instructions to fill pipeline bubbles, the kernel now exposes more of its true memory latency. The stall was always there — it was just hidden by the ALU work.

L3 read bandwidth jumped 15.9% because the same data volume is now consumed in less time.

---

### Phase 3: Incremental Optimizations (09 -> 12, 83.0 -> 85.1 TFLOPS, +2.5%)

#### 08 -> 09: SIMD16 Reciprocal

| Metric | 08 | 09 | Delta |
|--------|----|----|-------|
| AvgTime (s) | 0.106186 | 0.106130 | -0.05% |
| ALU0 instr | 429.4B | 429.5B | +0.02% |
| ALU1 instr | 868.6B | 867.1B | -0.2% |
| SBID stall% | 24.2 | 22.9 | -1.3pp |
| L3 Read GB/s | 1143.3 | 1107.7 | -3.1% |

**Analysis:** Replacing a scalar or SIMD8 reciprocal with a SIMD16 version for the softmax normalization. Negligible throughput impact — the reciprocal is not on the critical path. Slight SBID reduction suggests marginally better scheduling. Essentially noise-level improvement.

#### 09 -> 10: MAD Instruction Fusion

| Metric | 09 | 10 | Delta |
|--------|----|----|-------|
| AvgTime (s) | 0.106130 | 0.104404 | -1.6% faster |
| TFLOPS | 83.0 | 84.4 | +1.7% |
| **ALU1 instr** | **867.1B** | **679.2B** | **-21.7%** |
| ALU0 instr | 429.5B | 483.0B | +12.5% |
| DistAcc stall% | 4.2 | 7.6 | +3.4pp |
| MultiPipe% | 4.6 | 3.5 | -1.1pp |

**Analysis:** Fusing separate multiply and add operations into MAD (multiply-add) instructions reduced ALU1 instructions by 21.7% (188B fewer). However, ALU0 instructions increased by 12.5%, suggesting some operations shifted between pipes. The DistAcc (distributed accumulator) stall increase of 3.4pp indicates the fused instructions create tighter register dependencies. Net throughput gain of 1.7% shows the instruction reduction outweighs the increased stalls.

#### 10 -> 11: 32-Thread WG / Pipelined QK (84tflops attempt)

| Metric | 10 | 11 | Delta |
|--------|----|----|-------|
| AvgTime (s) | 0.104404 | 0.105605 | +1.2% slower |
| TFLOPS | 84.4 | 83.5 | -1.1% |
| ALU1 instr | 679.2B | 845.1B | +24.4% |
| ALU0 instr | 483.0B | 431.0B | -10.8% |
| DistAcc stall% | 7.6 | 4.2 | -3.4pp |
| SBID stall% | 21.3 | 21.9 | +0.6pp |

**Analysis:** This version attempted a different pipelining strategy for the Q*K matmul, possibly with a 32-thread workgroup variant. ALU1 instructions jumped back up 24.4%, likely from loop overhead or index calculations for the new pipeline structure. DistAcc stalls improved but SBID stalls and overall throughput regressed. This version was a net negative — the pipelining overhead exceeded the latency hiding benefit.

#### 11 -> 12: Pipelined K Loads (86tflops)

| Metric | 11 | 12 | Delta |
|--------|----|----|-------|
| AvgTime (s) | 0.105605 | 0.103596 | -1.9% faster |
| TFLOPS | 83.5 | 85.1 | +1.9% |
| ALU0 instr | 431.0B | 454.7B | +5.5% |
| ALU1 instr | 845.1B | 870.3B | +3.0% |
| Barrier stall% | 3.7 | 3.0 | -0.7pp |
| Send stall% | 2.5 | 1.5 | -1.0pp |
| IFetch stall% | 3.3 | 4.5 | +1.2pp |
| L3 Miss% | 1.2 | 0.6 | -0.6pp |
| L3 Read GB/s | 1128.7 | 1204.5 | +6.7% |
| TLB Misses | 8.72M | 7.07M | -18.9% |

**Analysis:** Pipelining K tile loads — starting the next K tile load while the current Q*K matmul is still executing. This is classic double-buffering. The results are excellent:
- Send stalls dropped 1.0pp (loads complete before they're needed)
- L3 miss ratio halved (0.6% from 1.2%) — better prefetch patterns
- TLB misses dropped 18.9% — more sequential access pattern
- L3 read bandwidth up 6.7% — sustaining higher throughput
- IFetch stalls rose 1.2pp — the larger code footprint from pipeline unrolling causes more instruction cache pressure

This is a clean win: better memory latency hiding with minimal overhead.

**Phase 3 Summary:** Net gain of ~2 TFLOPS (83.0 -> 85.1). The optimizations at this stage target secondary effects — ALU pipe efficiency, memory pipelining. The diminishing returns indicate the kernel is approaching the efficiency frontier for its current algorithmic structure.

---

### Phase 4: S Transpose Elimination (13, 85.1 -> 87.9 TFLOPS, +3.3%)

#### 12 -> 13: lsc_slm_scatter Replaces Register Transpose

| Metric | 12 | 13 | Delta |
|--------|----|----|-------|
| AvgTime (s) | 0.103596 | 0.100229 | -3.2% faster |
| TFLOPS | 85.1 | 87.9 | +3.3% |
| **ALU1 instr** | **870.3B** | **575.1B** | **-33.9%** |
| XMX% | 30.8 | 31.9 | +1.1pp |
| Active% | 52.1 | 49.0 | -3.1pp |
| Stalled% | 46.5 | 49.6 | +3.1pp |
| SBID stall% | 22.1 | 25.5 | +3.4pp |
| Send stall% | 1.5 | 4.0 | +2.5pp |
| L3 Read GB/s | 1204.5 | 1249.6 | +3.7% |
| L3 Write GB/s | 3.95 | 4.00 | +1.3% |

**Analysis:** This optimization replaces ~270 register-to-register mov instructions (used for in-register S tile transpose) with `lsc_slm_scatter` — a hardware scatter operation that writes data to SLM (Shared Local Memory) in transposed order, then reads it back transposed. This offloads the transpose from the register file and ALU pipes to the SLM hardware.

**Key observations:**
- **ALU1 instructions dropped 33.9%** (295B fewer) — the mov instructions for transpose were executing on ALU1
- **Active% dropped 3.1pp** while **TFLOPS rose 3.3%** — same pattern as version 08. Removing ALU work that was "active but not productive" decreases the activity percentage while increasing real throughput
- **SBID stalls rose 3.4pp** — the SLM scatter/gather has latency; threads stall waiting for the SLM operation to complete (scoreboard dependency)
- **Send stalls rose 2.5pp** — the `lsc_slm_scatter` is a send instruction; more send traffic means more send-related stalls
- **XMX% rose to 31.9%** — highest in the entire sequence; the XMX units are now the dominant consumer of execution time, which is exactly what we want for a matmul-heavy kernel

The trade-off is explicit: replace ~270 register movs (cheap per instruction, but 295B total) with a handful of SLM scatter/gather operations (more latent per operation, but far fewer total). The net effect is 3.3% throughput gain.

---

## 3. Key Metric Trends

### XMX Utilization Path

```
Version:  01    02    03    04    05    06    07    08    09    10    11    12    13
XMX%:    25.6  24.5  25.3  25.4  26.1  26.0  26.3  30.1  30.1  30.6  30.3  30.8  31.9
          |___Phase 1: ~25-26%___|   |__Phase 2-3: ~30%__|   |__Phase 4: 32%__|
```

XMX utilization shows two distinct step functions:
- **Phase 1 (01-07):** ~25-26%. Limited by the massive ALU0/ALU1 instruction volume that dominates execution time.
- **Phase 2 (08):** Jumps to 30.1% when fp16_transpose eliminates 800B+ ALU instructions.
- **Phase 3-4 (09-13):** Gradual climb to 31.9% as further ALU instruction reductions yield incremental XMX share.

### ALU0 Instruction Count (Register Mov Indicator)

```
Version:  01      02      03      04      05      06      07      08     09     10     11     12     13
ALU0(B): 969.8  1237.6  1234.7  1234.5  1234.4  1236.9  1234.3  429.4  429.5  483.0  431.0  454.7  454.6
```

The massive drop at version 08 (1234B -> 429B, -65%) is the fp16_transpose optimization. Phase 1 actually *increased* ALU0 from 970B to 1234B due to the prefetch management overhead, but this was compensated by better pipelining. The slight increases in versions 10 and 12 reflect pipeline unrolling overhead.

### ALU1 Instruction Count

```
Version:  01      02      03      04      05      06      07      08     09     10     11     12     13
ALU1(B): 1186.7  1293.0  1295.1  1294.9  1291.4  1293.7  1291.3  868.6  867.1  679.2  845.1  870.3  575.1
```

Two major drops:
- Version 08: -32.7% (strided gather reduction)
- Version 13: -33.9% (register transpose mov elimination via SLM scatter)

Version 10 (MAD fusion) also produced a significant -21.7% drop, though version 11 partially reverted it.

### L3 Bandwidth and Miss Ratio

```
Version:   01     02     03     04     05     06     07     08     09     10     11     12     13
L3 R(GB/s): 285   898    972    972    985    980    986   1143   1108   1156   1129   1205   1250
L3 Miss%:   5.6   1.2    1.5    1.5    1.2    1.3    1.1    1.2    1.1    1.1    1.2    0.6    0.6
```

L3 read bandwidth tracks inversely with kernel runtime — as the kernel executes faster, the same data volume is consumed in less time, so bandwidth increases. The 01->02 jump (285->898) reflects the fundamental architecture change to aggressive prefetching. L3 miss ratio stabilized around 1.1-1.5% after the initial prefetch optimization, then dropped to 0.6% with the pipelined K loads in version 12.

### Stall Breakdown Evolution

```
Version:  01    02    03    04    05    06    07    08    09    10    11    12    13
SBID%:   62.1  22.0  24.6  24.6  21.4  21.6  21.4  24.2  22.9  21.3  21.9  22.1  25.5
Barrier%: 38.4   5.1   4.4   4.4   4.2   4.2   4.4   3.9   3.9   4.8   3.7   3.0   3.2
DistAcc%: 11.5   4.1   4.8   4.8   4.2   4.2   4.2   4.0   4.2   7.6   4.2   4.2   4.3
Send%:    0.1   4.4   2.7   2.7   1.8   1.8   2.2   3.0   3.0   2.4   2.5   1.5   4.0
IFetch%:  6.5   4.4   3.2   3.2   3.2   3.2   3.1   3.3   3.4   2.2   3.3   4.5   3.3
```

**SBID (Scoreboard ID):** Dominates throughout. The initial drop from 62.1% to 22% was the biggest single improvement (prefetch/WG restructure). It has remained stubbornly around 21-25% since — this is the fundamental memory latency floor for this kernel.

**Barrier:** Crushed from 38.4% to ~4% in version 02 by the WG restructure and prefetch. Named barriers (05) provided marginal further improvement. This is no longer a bottleneck.

**Send:** Initially negligible (0.1%), rose to 4.4% with prefetch (more send instructions), then oscillated between 1.5-4.0%. The rise to 4.0% in version 13 is the cost of SLM scatter operations.

**DistAcc:** Spiked to 7.6% in version 10 (MAD fusion) due to tighter register dependencies, otherwise stable at ~4%.

### TLB Misses

```
Version:  01     02     03     04     05     06     07     08     09     10     11     12     13
TLB(M):  8.74   9.02   10.13  9.96   9.45   9.71   8.97   8.72   8.60   8.65   8.72   7.07   7.13
```

TLB misses were relatively stable at 8.7-10.1M until version 12, when pipelined K loads reduced them by 18.9% to 7.07M. More sequential access patterns from double-buffering result in better TLB hit rates. Version 13 maintained the low TLB miss count.

---

## 4. Critical Insights

### Why did fp16_transpose (08) cause the biggest jump?

Version 08 delivered **+14.7% throughput** — more than all other optimizations combined. The reason is architectural: in versions 01-07, every S tile needed to be transposed via register-to-register moves before the S*V matmul. With F32 data, each 4-byte element required a mov through the ALU pipes. The S tile at HD=256 is substantial, and with 16384 sequence length processed across the grid, the total move count was enormous.

By separating the F32->FP16 conversion (done in-place, cheap) from the transpose (now operating on 2-byte FP16 values), the number of register mov bytes was cut roughly in half, and the conversion could use efficient vector instructions rather than strided scatter patterns. The ALU0 instruction count dropped from 1234B to 429B — **805 billion fewer instructions**. This removed the single largest non-productive workload from the kernel.

The key evidence: XMX instruction count remained virtually identical across all 13 versions (~1716-1717B), proving that the actual matrix multiply work is constant. All performance gains came from reducing the overhead *around* the XMX operations.

### Why does version 13 (s_scatter) show LOWER Active% but HIGHER TFLOPS?

Version 13: Active% = 49.0% (lowest of all versions), TFLOPS = 87.9 (highest of all versions).

This apparent paradox reveals a fundamental truth about GPU performance metrics: **Active% measures utilization, not productivity.**

In versions 01-07, the EUs were "active" 50-59% of the time, but much of that activity was register shuffling (movs for transpose). Eliminating those instructions in versions 08 and 13 means:

1. The useful work (XMX matmuls, softmax computation) takes less wall-clock time
2. The EU activity during that shorter time is more productive (XMX% = 31.9%, highest ever)
3. But between the productive bursts, the EUs stall waiting for memory (SBID 25.5%)
4. Since the unproductive ALU filler is gone, those stalls are now exposed and measured

Think of it as removing the "busy work" from a factory — total hours worked decrease, but output per hour increases. The 49.0% Active with 31.9% XMX is far better than 59.0% Active with 26.3% XMX.

### What does the stall profile tell us about remaining optimization potential?

The current stall profile for version 13:
- **SBID: 25.5%** — Scoreboard dependency stalls, meaning threads are waiting for data from memory or long-latency operations
- **Send: 4.0%** — Waiting for send (load/store) operations to complete
- **DistAcc: 4.3%** — Distributed accumulator bank conflicts
- **Barrier: 3.2%** — Synchronization overhead
- **IFetch: 3.3%** — Instruction fetch latency

**SBID at 25.5% is the dominant remaining bottleneck.** This indicates the kernel is fundamentally memory-latency-bound at this point. The XMX units finish their work and then wait for the next tile to arrive. Potential mitigations:
1. **Deeper software pipelining** — overlap more loads with compute (versions 12's K pipeline was a step in this direction)
2. **Larger tile sizes** — more compute per loaded tile, better arithmetic intensity
3. **Occupancy tuning** — at 98.5% occupancy, there's negligible room for more threads to hide latency
4. **Register pressure management** — deeper pipelining requires more registers for double/triple buffering

The 4.0% Send stall is the cost of the SLM scatter approach. This could potentially be reduced by using hardware transpose units if available, or by restructuring the data layout to avoid transpose entirely.

### Why did L3 read bandwidth jump from ~285 to ~900+ GB/s between versions 01 and 02?

Three factors converged:

1. **Workgroup size doubled (16->32 threads):** More threads per workgroup means more concurrent memory requests. With 32 threads issuing loads, the memory subsystem sees 2x the request rate.

2. **Extended prefetch added:** The `xprefetch` mechanism issues non-blocking prefetch instructions that bring data into L3 cache ahead of demand. This converts cache misses into cache hits but shows up as additional L3 read traffic.

3. **Different data access pattern:** The switch from `benchmark_reversed` to `benchmark_v2` likely changed the K/V tile traversal order. The L3 miss ratio dropping from 5.6% to 1.2% confirms that the new pattern has much better spatial locality — data prefetched is actually used before eviction.

The DRAM bandwidth actually *decreased* (6.86 -> 4.39 GB/s), confirming that the higher L3 bandwidth comes from better cache utilization, not more memory traffic. The kernel went from "cache-thrashing with frequent DRAM accesses" to "well-prefetched with almost everything served from L3."

---

## 5. Gap to Roofline

### Current Position

| Metric | Value |
|--------|-------|
| Peak XMX throughput | 135 TFLOPS |
| Current best (v13) | 87.9 TFLOPS |
| Efficiency | **65.1%** |
| Gap | **47.1 TFLOPS (34.9%)** |

### Where is the Remaining 35% Lost?

#### 1. XMX Utilization Ceiling (31.9% of cycles)

The XMX units are active only 31.9% of cycles. If they ran 100% of the time at peak throughput, we'd reach the 135 TFLOPS roofline. The 68.1% of cycles where XMX is idle represents three categories:

**a) Stall cycles (49.6% of total):**
- SBID stalls (25.5%) — the largest single source of lost XMX throughput. Threads are waiting for data that hasn't arrived yet. Each SBID stall cycle is a cycle the XMX could have been computing.
- Send stalls (4.0%) — waiting for SLM scatter/gather operations
- DistAcc stalls (4.3%) — register bank conflicts
- Barrier stalls (3.2%) — synchronization overhead
- IFetch stalls (3.3%) — instruction cache misses
- Other stalls (~9.3%) — miscellaneous

**b) Non-XMX active cycles (Active% - XMX% = 49.0 - 31.9 = 17.1%):**
These cycles are spent on necessary non-matmul work: softmax computation (exp, sum, reciprocal, multiply), address calculations, loop control, prefetch management, SLM scatter/gather for S transpose. This overhead is inherent to the Flash Attention algorithm and cannot be eliminated, only minimized.

**c) Idle cycles (1.5%):**
Negligible — the GPU is almost fully loaded.

#### 2. Quantifying the Loss Budget

| Loss Category | Estimated TFLOPS Lost | % of Gap |
|---------------|----------------------|----------|
| SBID stalls (memory latency) | ~18-20 | ~40% |
| Non-XMX computation (softmax, etc.) | ~12-15 | ~28% |
| Other stalls (barrier, send, dist) | ~8-10 | ~20% |
| Instruction overhead (control flow) | ~3-5 | ~8% |
| Idle/startup/drain | ~2 | ~4% |

#### 3. Actionable Optimization Directions

**High impact (targeting SBID stalls):**
- **Triple-buffered K/V pipeline:** Currently double-buffered for K loads. Adding a third buffer stage could hide more latency at the cost of register pressure.
- **Asynchronous copy with barrier:** Use hardware async copy mechanisms to decouple data movement from computation more aggressively.
- **Tile size exploration:** Larger K/V tiles would increase arithmetic intensity (more compute per byte loaded), reducing the ratio of memory-wait to compute cycles. However, register pressure is already likely near the limit at HD=256.

**Medium impact (targeting non-XMX overhead):**
- **Approximate softmax:** Use hardware special function units for exp() if not already, or explore linear attention approximations.
- **Fuse softmax with S*V matmul:** Overlap the softmax normalization with the start of the S*V DPAS chain.
- **Eliminate remaining register shuffles:** The ALU0 count is still 454.6B — profile which instructions these are and whether any can be eliminated or offloaded.

**Low impact (targeting minor stalls):**
- **Instruction cache optimization:** IFetch stalls of 3.3% suggest the kernel code is large. Loop unrolling may be excessive; consider selective re-rolling of cold paths.
- **SLM bank conflict reduction:** The 4.0% Send stall from SLM scatter could be reduced by padding SLM layouts to avoid bank conflicts.

#### 4. Realistic Achievable Target

Given the fundamental constraints of the Flash Attention algorithm (softmax requires a full pass over each row before proceeding, creating a serial dependency), and the HD=256 tile size requiring substantial register pressure, a realistic target is:

- **Short-term (pipeline optimization):** 92-95 TFLOPS (68-70% of peak)
- **Medium-term (algorithmic improvements):** 100-105 TFLOPS (74-78% of peak)
- **Theoretical limit for this algorithm:** ~110-115 TFLOPS (81-85% of peak), bounded by the irreducible softmax overhead and memory latency floor

The remaining 15-19% gap to the absolute 135 TFLOPS peak is likely unachievable for Flash Attention due to the inherent non-matmul computational requirements and the serial dependency through the softmax normalization.

---

## Appendix: XMX Instruction Count Stability

A remarkable constant across all 13 versions:

```
XMX Instructions (billions):
01: 1716.66  |  05: 1716.99  |  09: 1716.94  |  13: 1716.70
02: 1716.74  |  06: 1717.23  |  10: 1716.89  |
03: 1717.28  |  07: 1716.86  |  11: 1716.56  |
04: 1716.86  |  08: 1716.96  |  12: 1716.89  |
```

All versions execute within 0.04% of the same number of XMX instructions (~1716.9B +/- 0.3B). This confirms that:
1. The actual matrix multiply workload is fixed by the problem dimensions
2. **Every single optimization in this sequence reduced overhead, not computation**
3. The performance journey from 70.5 to 87.9 TFLOPS (+24.7%) was achieved entirely by removing wasted work around the fixed XMX core

This is a textbook demonstration of Amdahl's Law in action: the XMX matmul is the "parallel" portion that scales with hardware, while the ALU overhead is the "serial" portion that must be minimized. Reducing ALU0 from 1234B to 454B (-63%) and ALU1 from 1291B to 575B (-55%) is what delivered the 24.7% wall-clock improvement.
