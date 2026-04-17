# Optimization History — BMG SDP HD=256 Kernel

## Starting Point: Reversed-DPAS Architecture

The HD=256 kernel uses a fundamentally different approach from HD=128:
- **S^T (transposed scores)**: QK GEMM produces S in `[KV × Q]` layout because K is loaded in reversed/transposed format for DPAS compatibility with HD=256's larger tile sizes.
- **V through SLM**: V is cooperatively loaded to SLM, then consumed by VS GEMM.
- **Q through SLM**: Q is loaded once to SLM (64KB), shared by all subgroups.
- **Cross-SG softmax**: Max and sum reduction across 8 sg_i subgroups via SLM.

The baseline architecture (`rev256_kv32_exec_16t.h`) used KV_STEP=32 with 16 threads/WG.

---

## Phase 1: oneDNN-Inspired Redesign (rev256_onednn_v2)

**Goal**: Adopt oneDNN's KV_STEP=128 with 8 sg_i threads processing 16 KV rows each.

### Version: vnni_64tflops
- **Technique**: Initial VNNI-based QK approach with oneDNN-style KV_STEP=128
- **Result**: 64 TFLOPS (47% roofline)
- **Verdict**: NOT USEFUL — VNNI packing overhead ate the parallelism gains
- **Root cause**: Extra register moves for VNNI format conversion dominated the QK phase

### Version: xprefetch_68tflops
- **Technique**: Added extended prefetch hints (prefetch K and V tiles further ahead)
- **Result**: 68 TFLOPS (+4T from 64)
- **Verdict**: NOT USEFUL — marginal gain, added code complexity
- **Root cause**: Memory was not the primary bottleneck at this point; ALU/ISA overhead was

### Version: deferred_comp_70tflops
- **Technique**: Deferred compensation — apply `A_tile *= delta` before VS GEMM rather than interleaved inside the VS loop. Also added cross-phase prefetch (prefetch next K during VS, prefetch V during QK).
- **Result**: 70 TFLOPS (+2T from 68)
- **Verdict**: USEFUL — established the foundation pattern used by all later versions
- **Why it worked**: Separating compensation from VS GEMM loop allowed cleaner DPAS pipeline with fewer intervening instructions

### Version: 70tflops_best
- **Technique**: Combined all optimizations up to this point into a clean baseline
- **Result**: 70.4 TFLOPS (52% roofline)
- **Verdict**: USEFUL — stable reference point for all later experiments

### Version: nbarrier_71tflops
- **Technique**: Replaced `barrier()` with named barriers (`nbarrier_init` / `nbarrier_wait`)
- **Result**: 71 TFLOPS (+0.6T)
- **Verdict**: USEFUL — small but essentially free gain
- **Why**: Named barriers have slightly lower overhead than generic barriers on Xe2

### Version: 72tflops_best
- **Technique**: Fine-tuning of register allocation and loop ordering
- **Result**: 72 TFLOPS (+1T)
- **Verdict**: USEFUL — marginal improvement, became the pre-breakthrough baseline

### Version: s_preload_72tflops
- **Technique**: Pre-load S tiles from SLM into registers before the VS GEMM inner loop
- **Result**: 72 TFLOPS (no change)
- **Verdict**: NEUTRAL — same performance, added code complexity
- **Why it didn't help**: SLM reads were already well-pipelined; pre-loading didn't hide any latency

---

## Phase 2: The Breakthrough — fp16 Transpose Optimization

### Version: fp16_transpose_82tflops
- **Technique**: Separate the F32→FP16 conversion from the strided register gather.
  - **Before**: `fp16 result = (fp16)ST_tile.select<8,16>(...)` — converting while doing stride-16 gather. Compiler generates massive `mov (4|M0)` sequences because it must do the fp32→fp16 conversion interleaved with strided access.
  - **After**: First convert entire `ST_tile` from fp32 to fp16 in contiguous blocks, THEN do the strided gather on the fp16 data. The fp16 data is half the size, so the strided gathers move fewer bytes.
- **Result**: 82 TFLOPS (+10T, +14% from 72)
- **Verdict**: VERY USEFUL — single largest optimization in the entire journey
- **Why it worked**: By converting to fp16 first, the subsequent strided gather moves 2 bytes per element instead of 4, halving register traffic. The ISA showed significantly fewer `mov` instructions in the transpose section.
- **ISA evidence**: mov count dropped from ~180 to ~120 in the softmax transpose section

### Version: simd_inv_83tflops
- **Technique**: Replace scalar `1.0f / sum[i]` loop with SIMD16 `simd<float,16> inv = 1.0f / sum_vec`
- **Result**: 82.5 TFLOPS (+0.5T)
- **Verdict**: USEFUL — small gain, simple change
- **Why**: SIMD16 reciprocal uses the math unit more efficiently than 16 scalar divisions

### Version: mad_fusion_83tflops (CURRENT BEST)
- **Technique**: Restructure `fp32_sum = fp32_sum * delta + local_sum` to generate BMG's `mad` instruction (fused multiply-add) which has 2× XVE throughput compared to separate mul + add.
- **Result**: 82.7 TFLOPS (+0.2T)
- **Verdict**: USEFUL — small gain, leverages hardware MAD unit
- **ISA evidence**: Confirmed `mad` instructions in ISA dump where previously there were separate `mul` + `add`

---

## Phase 3: S Transpose Optimization — Scatter/Gather Approaches

### Version: s_scatter_88tflops (CURRENT BEST)
- **Technique**: Eliminate ~270 register transpose `mov` instructions by using `lsc_slm_scatter<uint32_t, 2, u32, 16>` to write S tiles directly to transposed SLM positions.
- **Implementation**:
  ```cpp
  // Pack adjacent KV rows into u32 via shift+OR
  simd<uint32_t, 16> packed0 =
    simd<uint32_t, 16>(ST_fp16_u16.select<16, 1>((kv_base + 0) * 16)) |
    (simd<uint32_t, 16>(ST_fp16_u16.select<16, 1>((kv_base + 1) * 16)) << 16);
  simd<uint32_t, 16> packed1 = /* kv_base+2 and kv_base+3 similarly */;

  // CRITICAL: Element-major layout (NOT address-major interleave!)
  simd<uint32_t, 32> data;
  data.select<16, 1>(0) = packed0;   // elem0 for all 16 addresses
  data.select<16, 1>(16) = packed1;  // elem1 for all 16 addresses

  lsc_slm_scatter<uint32_t, 2, u32, 16>(q_offsets + kv_base * 2, data);
  ```
- **Result**: 87.6 TFLOPS at 16Kx16K (**+1.7 TFLOPS** from 84)
- **Verdict**: VERY USEFUL — eliminates the #1 bottleneck (S register transpose)
- **Key discoveries**:
  1. **Element-major data layout**: `lsc_slm_scatter/gather` with NElts > 1 expects element-major layout: data[0..N-1] = elem0 for all addresses, data[N..2N-1] = elem1 for all addresses. Address-major interleave (`data.select<16,2>(0)`) produces ~0.12 rel_rms systematic error — a subtle, hard-to-diagnose bug.
  2. **NElts=2 u32 is the sweet spot**: Each scatter writes 4 fp16 values (2 u32) per address. 8 scatter calls per thread (2 qp × 4 kv_groups). NElts=1 has too much message overhead; higher NElts doesn't help.
  3. **u32 packing via shift+OR**: `packed = low_kv | (high_kv << 16)` to combine adjacent KV rows into u32. The u16→u32 widening conversion (`simd<uint32_t, 16>(simd<uint16_t, 16>)`) zero-extends correctly.
  4. **SLM read unchanged**: `slm_block_load<uint32_t, 64>` still works because scatter writes the same Q-major KV-fast format as the baseline register transpose.
- **Performance breakdown**: 86.6 (8Kx8K), 87.2 (8Kx16K), 87.6 (16Kx16K)

### Attempt: s_gather (block_store + gather read) — REVERTED
- **Idea**: Store ST_fp16 to SLM directly without transpose (KV-major Q-fast layout), then use `lsc_slm_gather<uint32_t, 4, u32, 16>` to read in transposed order.
- **Implementation**: 4 gathers per S tile (each gather: 16 addresses × 4 u32 NElts = 8 fp16 per address). Two-level deinterleave: stride-4 u32 source → bit_cast to u16 → stride-2 u16 select.
- **Result**: 73 TFLOPS (**-12.9 TFLOPS regression**)
- **Verdict**: NOT USEFUL — SLM gathers are much more expensive than block loads
- **Root cause**: 4 gather messages per S tile vs 1 block load each. The gather message overhead dominates — each gather sends 16 addresses to the SLM unit, while block_load sends a single contiguous request. Even with NElts=4 (reducing from 32 to 4 gathers per kv_blk), the per-message cost is too high.
- **Lesson**: SLM block loads are far more bandwidth-efficient than gathers. If the write side can arrange data in the needed format (via scatter), the read side should always use block loads.

### Attempt: SLM Scatter NElts=1 (REVERTED)
- **Idea**: Eliminate register transpose with per-element SLM scatter.
- **Implementation**: 64 `lsc_slm_scatter<uint16_t, 1, u16, 8>` calls
- **Result**: 69 TFLOPS (**-16% regression**)
- **Root cause**: 64 scatter messages with only 16 bytes each. SLM scatter message overhead dominates.
- **Lesson**: Use NElts > 1 (preferably NElts=2 u32) to amortize scatter message overhead.

### Attempt: VNNI Interleave Transpose (REVERTED)
- **Idea**: Pair adjacent KV into uint32 VNNI pairs with stride-2 dest, then 8×8 uint32 transpose.
- **Result**: 82.7 TFLOPS (neutral)
- **Root cause**: Xe2 compiler decomposes `select<8, 8>()` stride-8 into `mov(4|M0)`, producing even more movs.

---

## ISA Analysis Findings

### Our Kernel ISA (from dump_v2_new/)
- **Total DPAS per iteration**: 128 (64 QK + 64 VS)
- **DPAS grouping**: 2 consecutive DPAS, then sends/movs
- **mov instructions in softmax section**: ~120 `mov (4|M0)` for stride-16 fp16 transpose
- **Total movs between QK and VS**: ~566
- **send count**: Higher than oneDNN due to SLM V loads

### oneDNN ISA (from analysis/oneDNN_16kx16k.csv)
- **Total DPAS per iteration**: 144 (48 QK + 96 VS) — VS-heavy distribution
- **DPAS grouping**: 4 consecutive QK, 8 consecutive VS with `{Atomic}` tags
- **mov instructions in softmax section**: 74 total, all 16-wide `mov (16|M0)`
- **Key mov pattern**: `mov (16|M0) rdst.0<2>:hf rsrc:f` — stride-2 destination interleave, converts f32→fp16 and transposes in one 16-wide instruction
- **V access**: 2D block load from global memory (no SLM staging)

### Gap Summary
| Metric | Ours | oneDNN | Gap |
|--------|------|--------|-----|
| DPAS/iter | 128 | 144 | -11% |
| Max consecutive DPAS | 2 | 8 | 4× |
| Softmax movs | ~120 (4-wide) | ~74 (16-wide) | ~2.6× |
| Threads/WG | 16 | 32 | 2× |
| TFLOPS | 82.7 | 90+ | ~10% |

---

## Backup File Index

### Kernel Headers (.h)
| File | TFLOPS | Status |
|------|--------|--------|
| `rev256_onednn_v2_vnni_64tflops.h` | 64 | Archived — VNNI approach abandoned |
| `rev256_onednn_v2_xprefetch_68tflops.h` | 68 | Archived — marginal gain |
| `rev256_onednn_v2_deferred_comp_70tflops.h` | 70 | Archived — foundation pattern |
| `rev256_onednn_v2_70tflops_best.h` | 70 | Archived — clean baseline |
| `rev256_onednn_v2_71tflops_nbarrier.h` | 71 | Archived — named barriers |
| `rev256_onednn_v2_72tflops_best.h` | 72 | Archived — pre-breakthrough best |
| `rev256_onednn_v2_72tflops_s_preload.h` | 72 | Archived — S-preload experiment |
| `rev256_onednn_v2_82tflops_fp16_transpose.h` | 82 | Key milestone — transpose breakthrough |
| `rev256_onednn_v2_83tflops_simd_inv.h` | 82.5 | SIMD reciprocal |
| `rev256_onednn_v2_83tflops_mad_fusion.h` | 82.7 | mad fusion |
| `rev256_onednn_v2_86tflops_qk_pipeline.h` | 84 | 32-thread WG, pipelined K |
| **`rev256_onednn_v2_88tflops_s_scatter.h`** | **88** | **Current best — lsc_slm_scatter S transpose elimination** |

### Test Programs (.cpp)
| File | Purpose |
|------|---------|
| `test_rev256_onednn_v2_70tflops_best.cpp` | Full test for 70T version |
| `test_rev256_onednn_v2_71tflops_nbarrier.cpp` | Full test for 71T version |
| `test_rev256_onednn_v2_72tflops_s_preload.exe` | Built exe for 72T version |
| `test_rev256_onednn_v2_deferred_comp_70tflops.cpp` | Full test for deferred comp |
| `test_rev256_onednn_v2_xprefetch_68tflops.cpp` | Full test for xprefetch |
| `rev256_onednn_v2_83tflops_mad_fusion.cpp` | Mini benchmark (perf comparison only) |
| `test_v2_mini_*.cpp` | Mini benchmarks for various versions |

### Pre-v2 Files
| File | Purpose |
|------|---------|
| `flash.attn.b.mha256.fp16.reversed.kv32_exec.h` | Original reversed-DPAS HD=256 kernel |
| `flash_attn_test_reversed_hd256.kv32_exec.cpp` | Test for original kernel |
| `rev256_kv32_exec_16t.h` | KV32 exec variant with 16 threads |

---

## Performance Numbers (16Kx16K, 32Q/2KV heads, HD=256)

| Version | Time (ms) | TFLOPS | Roofline% |
|---------|-----------|--------|-----------|
| Baseline (opt) | 178.973 | 49.2 | 36.5% |
| v2 vnni | ~137 | 64 | 47% |
| v2 xprefetch | ~129 | 68 | 50% |
| v2 deferred_comp | ~125 | 70 | 52% |
| v2 nbarrier | ~123 | 71 | 53% |
| v2 72 best | ~122 | 72 | 53% |
| v2 fp16_transpose | ~107 | 82 | 61% |
| v2 simd_inv | ~106.8 | 82.5 | 61% |
| v2 mad_fusion | ~106.6 | 82.7 | 61.3% |
| v2 pipelined_qk | ~104 | 84 | 62% |
| **v2 s_scatter** | **~100.7** | **87.6** | **64.9%** |
| oneDNN reference | ~97 (est) | 90+ | 67%+ |

---

## Next Steps (Future Work)

1. **WG Redesign (8×4 = 32 threads)**: Full kernel rewrite to match oneDNN's dispatch pattern. 4 consecutive QK DPAS, 8 consecutive VS DPAS. Expected: 85-100+ TFLOPS. See `analysis/kernel_redesign_plan.md`.
2. **2D Block Load for K**: Replace reversed lsc_gather with standard 2D block loads. Requires WG redesign.
3. **V from Global Memory**: Remove SLM staging for V, load directly from global via 2D block load. Frees 32KB SLM for S storage.
4. **Inline Assembly for Transpose**: Use Xe2 inline asm to achieve 16-wide mov with stride-2 dest like oneDNN. Requires compiler support investigation.
