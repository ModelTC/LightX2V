---
name: xe2-sdp-hd256
description: Use this skill when writing, optimizing, benchmarking, or debugging Flash Attention SDP kernels with head dimension 256 (HD=256) targeting Intel Xe2 (Lunar Lake/LNL, Battlemage/BMG) GPU using SYCL ESIMD. Xe2 is the GPU architecture; LNL and BMG are product names. Covers the S^T (transposed scores) architecture, oneDNN-inspired v2 kernel design, GQA support, softmax optimization, lsc_slm_scatter S transpose elimination, ISA-level analysis, and the complete optimization journey from 64 to 88 TFLOPS. Use whenever the user mentions HD=256 SDP, head_dim=256 attention, rev256, onednn_v2 kernel, S transpose, s_scatter, s_gather, lsc_slm_scatter, lsc_slm_gather, or large head dimension flash attention on Intel GPU.
---

# Xe2 (Lunar Lake/LNL, Battlemage/BMG) ESIMD Flash Attention — HD=256 (S^T Architecture)

Specialized knowledge for GQA flash attention with head_dim=256 on Intel Xe2 (Lunar Lake/LNL, Battlemage/BMG).
Current best: **88 TFLOPS** (64.9% of 135 TFLOPS peak) at 16Kx16K, 32Q/2KV heads.
Reference files hold detailed history; this file holds critical rules and architecture.

---

## Quick-Reference Rules (must follow every time)

### HD=256 Architecture Differences (vs HD=128)
- **S^T approach**: QK GEMM produces scores in transposed layout `[KV_PER_SG=16][Q_ROWS*Q_GRPS=32]` per thread. This is because K is loaded in reversed/transposed format for DPAS efficiency.
- **Register transpose required**: Before VS GEMM, S must be transposed from `[KV×Q]` to `[Q×KV]` and written to SLM as fp16. This is the **#1 overhead source** (~20% of kernel time).
- **Larger SLM footprint**: Q_SLM=64KB + S_SLM=32KB + MAX_SLM=4KB + SUM_SLM=4KB = ~104KB (vs ~16-32KB for HD=128).
- **32 threads/WG** (8 sg_i × 4 sg_j) — matches oneDNN decomposition. doubleGRF required.

### Kernel Constants
```cpp
HD = 256;              // Head dimension
WG_Q_ROWS = 128;       // Q rows processed per WG
KV_CHUNK = 128;         // KV rows per outer iteration (8 sg_i × 16 KV_PER_SG)
KV_PER_SG = 16;         // KV rows per subgroup per iteration
Q_ROWS = 8;             // Q rows per subgroup tile
Q_PAIRS = 2;            // Q pairs per sg_j (2 × 16Q = 32Q per sg_j)
Q_GRPS = 4;             // Q groups per subgroup (4 × 8 = 32 Q rows per SG)
D_BLKS_PER_SG = 2;      // D blocks per SG (2 × 16D = 32D per sg_i)
```

### Thread Mapping
```cpp
sg_i = tid & 7;         // 0-7: KV/D dimension
sg_j = tid >> 3;        // 0-3: Q dimension
// sg_i handles: KV rows [sg_i*16 : sg_i*16+16], D cols [sg_i*32 : sg_i*32+32] (VS phase)
// sg_j handles: Q rows [sg_j*32 : sg_j*32+32] (4 groups × 8 rows)
```

### SLM Layout
```
0x00000 - 0x0FFFF:  Q_SLM    (64 KB) — Q tile [128 Q × 256 D] fp16
0x10000 - 0x17FFF:  S_SLM    (32 KB) — Scores [128 Q × 128 KV] fp16 (after transpose)
0x18000 - 0x18FFF:  MAX_SLM  (4 KB)  — Cross-SG max reduction
0x19000 - 0x19FFF:  SUM_SLM  (4 KB)  — Cross-SG sum reduction
```

### Build Commands
```powershell
# Full build (correctness + perf benchmark):
icpx test_rev256_onednn_v2.cpp -o test_rev256_onednn_v2.exe `
  -fsycl -fsycl-targets=spir64_gen `
  -Xs "-device bmg -options -doubleGRF" -O2 -w

# Mini build (perf only, faster compile):
icpx rev256_onednn_v2_83tflops_mad_fusion.cpp -o test.exe `
  -fsycl -fsycl-targets=spir64_gen `
  -Xs "-device bmg -options -doubleGRF" -O2 -w

# ISA dump build:
icpx test_rev256_onednn_v2.cpp -o test.exe `
  -fsycl -fsycl-targets=spir64_gen `
  -Xs "-device bmg -options -doubleGRF" -O2 -w -save-temps
# Then: ocloc disasm -file <spv> -device bmg -dump dump_v2_new
```

### S Transpose: Solved via lsc_slm_scatter
The register transpose from `[KV=16][Q=16]` fp32 → `[Q=8][KV=16]` fp16 row-major used to generate ~270 `mov (4|M0)` instructions in ISA. **Now eliminated** using `lsc_slm_scatter<uint32_t, 2, u32, 16>`:

**Working approach (Approach 2 — s_scatter, +1.7 TFLOPS):**
1. Pack adjacent KV rows into u32 via shift+OR: `packed = low_kv | (high_kv << 16)`
2. `lsc_slm_scatter<uint32_t, 2, u32, 16>` writes 2 u32 per Q address → 4 KV values per call
3. 4 scatter calls per qp (kv groups 0..3, 4..7, 8..11, 12..15), 8 total per thread
4. SLM layout after scatter is Q-major KV-fast — same as baseline after register transpose
5. Block read unchanged (slm_block_load)

**CRITICAL: Element-major data layout.** `lsc_slm_scatter` with NElts > 1 expects element-major layout:
```cpp
// CORRECT: data[0..15] = elem0 for all addrs, data[16..31] = elem1 for all addrs
data.select<16, 1>(0) = packed0;
data.select<16, 1>(16) = packed1;

// WRONG: address-major interleave (produces ~0.12 rel_rms systematic error)
data.select<16, 2>(0) = packed0;   // DO NOT USE
data.select<16, 2>(1) = packed1;   // DO NOT USE
```

**Failed alternatives:**
- Approach 1 (s_gather — block_store + gather read): **73 TFLOPS (-12.9)** — SLM gathers too expensive
- SLM scatter NElts=1 (64 D16 messages): **69 TFLOPS** — scatter bandwidth too low with single-element writes
- VNNI interleave (stride-2 uint32 pairs + 8×8 transpose): **neutral** (82.7 TFLOPS) — compiler decomposed stride-8 uint32 gather into even more mov(4|M0)

### Performance Testing
- **Correctness**: 9 test configs (128×128 to 1K×1K, GQA + MHA, multiple data ranges)
- **Benchmark sizes**: 8K×8K, 8K×16K, 16K×16K (compute-bound regime)
- **Warmup**: 5 iterations, then 100 timed iterations
- **FLOPS formula**: `4 * q_len * kv_len * HD * num_heads_q + 2 * q_len * kv_len * num_heads_q`
- **Peak**: 135 TFLOPS (BMG Xe2 XMX fp16)
- **Correctness threshold**: max_diff < 0.1, check NaN count first

### Backup Convention
All backups in `SDP/backup/` with naming: `rev256_onednn_v2_<TFLOPS>tflops_<technique>.{h,cpp,exe}`

---

## Performance Ladder (current)

| Version | TFLOPS | Roofline% | Technique | Useful? |
|---------|--------|-----------|-----------|---------|
| vnni | 64 | 47% | VNNI-based QK approach | NO — slower than reversed DPAS |
| xprefetch | 68 | 50% | Extended prefetch hints | NO — marginal gain, added complexity |
| deferred_comp_70 | 70 | 52% | Deferred compensation, cross-phase prefetch | YES — foundation for all later work |
| nbarrier_71 | 71 | 53% | Named barriers instead of regular barriers | YES — small but free gain |
| best_72 | 72 | 53% | Optimal baseline configuration | YES — stable reference point |
| s_preload_72 | 72 | 53% | S-preload in VS phase | NEUTRAL — same perf, more code complexity |
| **fp16_transpose_82** | **82** | **61%** | Separate F32→FP16 from strided gather | **YES** — major breakthrough (+14%) |
| simd_inv_83 | 82.5 | 61% | SIMD16 reciprocal instead of scalar div | YES — small gain |
| mad_fusion_83 | 82.7 | 61% | fp32_sum mad fusion (mul+add → mad) | YES — incremental gain |
| pipelined_qk_84 | 84 | 62% | 32-thread WG (8×4), pipelined K loads, split barrier B | YES — previous best |
| **s_scatter_88** | **88** | **65%** | Eliminate S register transpose via lsc_slm_scatter<u32,2> | **YES** — current best |
| s_gather (reverted) | 73 | 54% | block_store + lsc_slm_gather<u32,4> read | **NO** — gather reads too expensive (-12.9 TFLOPS) |
| SLM scatter NElts=1 (reverted) | 69 | 51% | Replace transpose with 64 SLM scatter D16 ops | **NO** — major regression, scatter BW too low |
| VNNI interleave (reverted) | 82.7 | 61% | Stride-2 interleave + 8×8 uint32 transpose | **NO** — neutral, compiler generated more movs |
| Early V loads (reverted) | 82.7 | 61% | V loads before S transpose | **NO** — increased register live range hurt scheduling |
| Double-buffered S (reverted) | 83.8 | 62% | Cross-iteration pipeline with 2 S buffers | **NO** — neutral, barrier_A already syncs S writes (see below) |
| S_VNNI + transposed acc (abandoned) | — | — | Eliminate S transpose via DPAS operand swap | **NO** — hw constraints: fp16 transpose max width=4, store height max=8 |

---

## Key Optimization Insights

### What Worked
1. **lsc_slm_scatter S transpose elimination** (+1.7 TFLOPS, 84→88): Replace ~270 register transpose `mov` instructions with `lsc_slm_scatter<uint32_t, 2, u32, 16>`. Pack adjacent KV rows into u32 via shift+OR, scatter directly to transposed SLM positions. 8 scatter calls per thread (2 qp × 4 kv_groups). Key discovery: NElts > 1 data must be in **element-major** layout.
2. **Separate F32→FP16 conversion from strided gather** (+14%): Convert entire ST_tile to fp16 first, then do strided gather on fp16 data. Halves the register traffic for the transpose.
3. **mad instruction fusion**: `fp32_sum = fp32_sum * delta + local_sum` generates SIMD32 `mad` instruction with 2× throughput on BMG XVE.
4. **SIMD16 reciprocal**: `simd<float,16> inv = 1.0f / sum` instead of scalar `1.0f / sum[i]` in a loop.
5. **Split barrier B (arrive/wait)**: Named barrier arrive after S SLM writes → V loads + K prefetch + compensation (all independent work) → named barrier wait before VS reads. The arrive/wait window spans V loads + K prefetch + deferred compensation = 100+ cycles of useful work between arrive and wait. See `xe2-nbarrier-pipelining` skill for the pattern.
6. **Deferred compensation**: Apply `A_tile *= delta` before VS GEMM, not interleaved inside VS loop — cleaner DPAS pipeline.
7. **Cross-phase prefetch**: Prefetch next K tile during VS phase, prefetch V during QK phase.
8. **32-thread WG + pipelined K loads** (+1.3 TFLOPS): Expanded from 8×2=16 to 8×4=32 threads matching oneDNN decomposition. K loads pipelined: load K[d+1] after DPAS[d], hiding K load latency behind compute.

### What Didn't Work
1. **SLM gather S read (Approach 1)**: `slm_block_store` of un-transposed data + `lsc_slm_gather<uint32_t, 4, u32, 16>` to read transposed. 73 TFLOPS (-12.9 from baseline). SLM gather reads are much more expensive than block loads — 4 gathers per S tile vs 1 block load each, and each gather has higher message overhead.
2. **SLM scatter NElts=1**: 64 `lsc_slm_scatter<u16, 1, u16, 8>` messages (one per KV × qh). 69 TFLOPS. Per-element scatter bandwidth is severely limited.
3. **VNNI interleave transpose**: Pairing adjacent KV into uint32 VNNI pairs with stride-2 dest writes — Xe2 compiler decomposes `select<8,8>()` stride-8 uint32 gather into `mov(4|M0)`, producing even more instructions.
3. **Extended prefetch (xprefetch)**: Adding more prefetch hints beyond the basic cross-phase pattern gave marginal improvement.
4. **S-preload optimization**: Pre-loading S tiles from SLM before VS GEMM — no measurable improvement.
5. **Early V loads**: Moving V loads before S transpose to overlap V latency with transpose ALU. Increased register live range, hurt compiler scheduling → 82.7 vs 84.1 baseline.
6. **Double-buffered S SLM** (83.8 vs 84.1 TFLOPS — neutral): Cross-iteration pipeline with 2 S buffers (S_A + S_B, 32KB each). Pipeline: QK[k+1] → softmax[k+1] → S_store[k+1]→buf_new → compensation[k] → VS[k]→buf_old. **Why it failed:**
   - barrier_A (full barrier for softmax max reduction) already synchronizes ALL SLM writes including S — the named barrier for S sync was already free (never blocks)
   - Q SLM reduced from 16→15 d_blks (SLM budget: 60KB Q + 32KB S_A + 32KB S_B + 4KB shared = 128KB max) → 16th d_blk loaded from global (extra LSC sends)
   - Extra registers: `delta_prev`, `local_sum_prev` carried across iterations
   - Code bloat from prologue/epilogue (3-phase structure duplicates compensation+VS)
   - **Key lesson**: The split barrier B arrive/wait window (S writes → V loads + K prefetch + compensation → VS reads) already hides S synchronization latency. Double-buffering removes a barrier that was never blocking.
7. **S_VNNI + transposed accumulator** (abandoned at build): Swap DPAS operands so S_vnni is src1 and V is src2, eliminating the S register transpose entirely. Accumulator becomes [8D×16Q] instead of [8Q×16D], requiring one-time output transpose. **Build constraints discovered:**
   - `lsc_load_2d` fp16 transpose: max width=4 (need 2 loads per 8D tile → 8 loads/kv_blk vs 2)
   - `lsc_store_2d` fp16: max height=8 (can't do height=16 for transposed output store)
   - These are hardware/ESIMD API constraints, not software limitations

### Remaining Gap vs oneDNN (90+ TFLOPS)
1. **DPAS density**: oneDNN achieves 4 consecutive QK DPAS and 8 consecutive VS DPAS with `{Atomic}` tags. Ours has 2 consecutive DPAS with intervening sends/movs.
2. **V from global**: oneDNN loads V via 2D block load from global memory. Ours loads V cooperatively to SLM first.
3. **S transpose is now solved**: ~270 register transpose movs eliminated via lsc_slm_scatter. Gap narrowed from ~6 to ~3 TFLOPS vs oneDNN.

---

## Architecture: oneDNN vs Ours

| Aspect | oneDNN SDPA | Our rev256_onednn_v2 |
|--------|-------------|----------------------|
| Threads/WG | 32 (8×4) | 32 (8×4) |
| DPAS/iter | 144 (48 QK + 96 VS) | 128 (64 QK + 64 VS) |
| QK DPAS density | 4 consecutive | 4 consecutive (pipelined K) |
| VS DPAS density | 8 consecutive | 8 consecutive |
| Q storage | SLM (shared by all SGs) | SLM (shared by all SGs) |
| K access | 2D block load | 2D block load (pipelined) |
| V access | 2D block load from global | 2D block load with VNNI |
| S storage | SLM (VNNI packed) | SLM (row-major fp16, via lsc_slm_scatter) |
| Softmax | Cross-SG atomic max | Cross-SG SLM max + barrier |
| S sync | Split barrier (arrive/wait) | Named barrier (arrive/wait) |
| Peak measured | ~90+ TFLOPS | 88 TFLOPS |

See `analysis/onednn_vs_ours_analysis.md` and `analysis/kernel_redesign_plan.md` for full WG redesign plan.

---

## Assets (ready to compile)

```powershell
icpx <file>.cpp -o <file>.exe -fsycl -fsycl-targets=spir64_gen -Xs "-device bmg -options -doubleGRF" -O2 -w
```

| Asset | Purpose | Perf |
|-------|---------|------|
| `assets/rev256_onednn_v2_88tflops_s_scatter.h` | **Current best** — lsc_slm_scatter S transpose elimination. 32-thread WG, pipelined K, split barrier B, all Phase 1-4 optimizations. Primary code reference. | **88 TFLOPS (65%)** |
| `assets/rev256_onednn_v2_88tflops_s_scatter.cpp` | **Test harness for 88T** — correctness (9 tests) + 3 perf benchmarks. Reproducible. | run directly |
| `assets/rev256_onednn_v2_86tflops_qk_pipeline.h` | **Pre-scatter** — pipelined K loads, 32-thread WG, split barrier B. Diff with 88T to see exactly what s_scatter changed. | 86 TFLOPS (64%) |
| `assets/rev256_onednn_v2_86tflops_qk_pipeline.cpp` | **Test harness for 86T** — reproducible perf benchmark. | run directly |
| `assets/rev256_onednn_v2_84tflops_pipelined_qk.h` | **Pre-pipeline** — 32-thread WG with pipelined K loads. Diff with 86T to see K pipeline refinements. | 84 TFLOPS (62%) |
| `assets/rev256_onednn_v2_84tflops_pipelined_qk.cpp` | **Test harness for 84T** — reproducible perf benchmark. | run directly |
| `assets/rev256_onednn_v2_83tflops_mad_fusion.h` | **Pre-32-thread** — 16-thread WG with mad fusion, simd_inv, fp16 transpose. Shows the Phase 2 architecture before WG expansion. | 83 TFLOPS (61%) |
| `assets/rev256_onednn_v2_83tflops_mad_fusion.cpp` | **Mini benchmark for 83T** — perf-only test harness. | run directly |
| `assets/rev256_onednn_v2_82tflops_fp16_transpose.h` | **fp16 transpose breakthrough** — the single optimization that gave +14.7%. Compare with 83T to see simd_inv + mad fusion changes. | 82 TFLOPS (61%) |
| `assets/test_rev256_onednn_v2.cpp` | **Full test harness** — 9 correctness configs (GQA+MHA, multiple ranges/seeds) + 3 perf benchmarks (8K-16K). | run directly |
| `assets/flash.attn.b.mha256.fp16.opt.h` | **Baseline kernel** — original HD=256 opt kernel (49 TFLOPS). Non-S^T architecture for contrast. | 49 TFLOPS (36%) |

**Key code patterns to study (diff across versions):**
- **88T vs 86T**: lsc_slm_scatter S transpose elimination (the only diff — S write section)
- **86T vs 84T**: Pipelined K loads refinement, split barrier B optimization
- **84T vs 83T**: 16→32 thread WG expansion, K load pipelining
- **83T vs 82T**: simd_inv (SIMD16 reciprocal) + mad fusion (fused multiply-add)
- **82T**: The fp16 transpose breakthrough — separate F32→FP16 from strided gather

---

## Workflow

1. **Edit kernel** in `rev256_onednn_v2.h` following rules above
2. **Build** with `do_build_full.ps1` (correctness+perf) or `do_build.ps1` (perf only)
3. **Run exe** — check 9/9 correctness PASS, then check TFLOPS at 8K×8K, 16K×16K
4. **If improvement**: backup to `backup/rev256_onednn_v2_<TFLOPS>tflops_<technique>.{h,cpp,exe}`
5. **If regression**: restore from latest backup in `backup/`
6. **ISA analysis**: build with `-save-temps`, disassemble with `ocloc`, compare mov/DPAS counts

---

## Key Files

| File | Purpose |
|------|---------|
| `rev256_onednn_v2_s_scatter.h` | Current best HD=256 kernel (88 TFLOPS, s_scatter) |
| `test_rev256_onednn_v2.cpp` | Full test harness (correctness + perf benchmark) |
| `rev256_onednn_v2_83tflops_mad_fusion.cpp` | Mini benchmark (perf only, baseline comparison) |
| `flash.attn.b.mha256.fp16.opt.h` | HD=256 baseline kernel for perf comparison |
| `do_build_full.ps1` | PowerShell build script (full test) |
| `do_build.ps1` | PowerShell build script (mini/perf only) |
| `do_isa_new.ps1` | PowerShell ISA dump + disassemble script |
| `analysis/onednn_vs_ours_analysis.md` | Detailed architecture comparison with oneDNN |
| `analysis/kernel_redesign_plan.md` | WG redesign plan (8×4 = 32 threads) |
| `analysis/oneDNN_16kx16k.csv` | oneDNN ISA dump for reference |

---

## Backup Files

| Backup | TFLOPS | Description |
|--------|--------|-------------|
| `rev256_onednn_v2_vnni_64tflops.h` | 64 | Initial VNNI approach |
| `rev256_onednn_v2_xprefetch_68tflops.h` | 68 | Extended prefetch experiment |
| `rev256_onednn_v2_deferred_comp_70tflops.h` | 70 | Deferred compensation baseline |
| `rev256_onednn_v2_70tflops_best.h` | 70 | Cross-phase prefetch optimized |
| `rev256_onednn_v2_71tflops_nbarrier.h` | 71 | Named barriers |
| `rev256_onednn_v2_72tflops_best.h` | 72 | Best pre-transpose-opt config |
| `rev256_onednn_v2_72tflops_s_preload.h` | 72 | S-preload experiment |
| `rev256_onednn_v2_82tflops_fp16_transpose.h` | 82 | F32→FP16 before transpose (breakthrough) |
| `rev256_onednn_v2_83tflops_simd_inv.h` | 82.5 | SIMD16 reciprocal |
| `rev256_onednn_v2_83tflops_mad_fusion.h` | 82.7 | mad fusion |
| `rev256_onednn_v2_84tflops_pipelined_qk.h` | 84 | 32-thread WG, pipelined K loads, split barrier B |
| **`rev256_onednn_v2_88tflops_s_scatter.h`** | **88** | **lsc_slm_scatter S transpose elimination (current best)** |
| `rev256_onednn_v2_double_buffer.h` | 83.8 | Double-buffered S cross-iteration pipeline (reverted) |

---

## Related Skills

| Skill | When to use |
|-------|------------|
| `xe2-nbarrier-pipelining` | Named barrier arrive/wait patterns for pipelining independent GPU operations |
| `xe2-sdp-kernels` | HD=128 flash attention — different architecture, higher roofline% |
| `xe2-sdp-bf16` | BF16 variants — bf16 ALU limits, V conversion interleaving |
| `xe2-gtpin-profiling` | GTPin setup and ISA analysis methodology |
| `intel-gpu-vtune-profiling` | VTune GPU Compute/Media Hotspots — capture and analyze HW metrics (XMX%, stalls, L3, occupancy) |
| `intel-gpu-kernel-opt` | General Xe2 optimization patterns |
| `sycl-esimd-build` | Compilation flags, doubleGRF, spill detection |
| `xe2-dpas-patterns` | DPAS tiling, VNNI packing, systolic pipeline |

---

## Reference Files

| File | Contents |
|------|----------|
| `references/optimization-history.md` | Full chronological journey from 64→88 TFLOPS with root cause analysis |
| `references/vtune-analysis.md` | VTune GPU hotspot analysis across all 13 versions — HW metrics, stall breakdown, phase-by-phase explanation |
| `references/vtune-summary.csv` | Raw VTune metrics CSV (13 rows × 30 columns) for programmatic analysis |
| `assets/rev256_onednn_v2_88tflops_s_scatter.h` | **Current best** kernel (88T) — lsc_slm_scatter S transpose elimination |
| `assets/rev256_onednn_v2_88tflops_s_scatter.cpp` | Test harness for 88T — reproducible |
| `assets/rev256_onednn_v2_86tflops_qk_pipeline.h` | Pre-scatter kernel (86T) — pipelined K, split barrier B |
| `assets/rev256_onednn_v2_86tflops_qk_pipeline.cpp` | Test harness for 86T — reproducible |
| `assets/rev256_onednn_v2_84tflops_pipelined_qk.h` | Pipelined QK kernel (84T) — 32-thread WG |
| `assets/rev256_onednn_v2_84tflops_pipelined_qk.cpp` | Test harness for 84T — reproducible |
| `assets/rev256_onednn_v2_83tflops_mad_fusion.h` | MAD fusion kernel (83T) — pre-32-thread |
| `assets/rev256_onednn_v2_83tflops_mad_fusion.cpp` | Mini benchmark for 83T |
| `assets/rev256_onednn_v2_82tflops_fp16_transpose.h` | fp16 transpose breakthrough (82T) |
| `assets/test_rev256_onednn_v2.cpp` | Full correctness + perf test harness |
| `assets/flash.attn.b.mha256.fp16.opt.h` | Baseline kernel (49T) for comparison |
