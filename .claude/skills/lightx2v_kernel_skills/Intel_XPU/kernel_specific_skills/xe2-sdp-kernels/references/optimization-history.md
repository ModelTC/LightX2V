# Optimization History — BMG SDP Kernels

## Starting Point: flash.attn.b.mha128.h (Legacy Non-GQA)

The first kernel (`flash.attn.b.mha128.h`) was a non-GQA flash attention implementation.

- ~60% roofline efficiency
- No GQA support: headQ == headKv required
- Had some runtime conditionals in the inner loop
- Scalar-loop decode (not two-phase)

Key lessons from this version:
- DPAS tile sizes (16×16×16) and VNNI packing were established
- SLM ping-pong for V buffering was introduced
- The causal mask branching was inside the loop → caused SPIR-V issues

---

## Phase 1: GQA Prefill (flash.attn.b.mha128.gqa.precomputed_yuchen.h)

**Goal:** support headQ=32, headKv=2 (group_size=16).

Changes:
1. Added GQA indexing: `kvHeadIdx = v; headIdx = v * group_size`
2. Fixed grid dimension assignment: `h = get_group(0)` (query blocks), `v = get_group(1)` (KV heads)
3. Pre-computed `kvSeqOutLoopCount` on device before main loop — eliminates runtime `if` for causal
4. Replaced scalar causal check with vectorized SIMD merge: `tempOutput.merge(FP32_MIN, mask)`
5. Maintained exact same barrier count for all threads (1 per KV iteration)

**Result:** 83–86% roofline (from ~60%) — the GQA sharing multiplied effective compute per KV load.

---

## Phase 2: Two-Phase Decode (decode_sdp_gqa.h)

**Problem:** Single-thread decode over kv_len=4K was sequential and slow.

**Solution:** Two-phase parallel decode:

### Phase 1: Chunked parallel
- Each WG handles one `chunk_size`-token chunk for one `kv_head`
- Threads within WG further split chunk into `sp_blk_size`-token sub-blocks
- SLM used to accumulate partial results, then reduce within WG
- Outputs: `[chunk_num, headQ, HD]` (fp32), `[chunk_num, headQ]` lse, max

### Phase 2: Reduce
- One thread per 16 Q-heads
- Two-pass reduction: find global max, then weighted accumulate
- Output: `[headQ, HD]` (fp16)

**Key parameters tuned:**
- `Q_HEAD_PER_T = 8` (optimal for SLM fit and parallelism)
- `chunk_size = 256` (chunk num = kv_len/256 for kv=4K → 16 chunks)
- `sp_blk_size = 32` (sp_blk_num_per_t = 256/32 = 8)

**WG thread count:** sp_blk_num_per_t × head_groups_per_g = 8 × 2 = 16 ≤ 32 ✓

**Result:** ~11.5 µs for kv=4K → ~349 GB/s → ~66% of 530 GB/s peak.

---

## Phase 3: Non-Aligned kv_len Support

**Problem:** Initial decode kernel required kv_len to be chunk_size-aligned.

**Fix:**
- Ceiling division: `chunk_num = (kv_len + chunk_size - 1) / chunk_size`
- `valid_t` clamping in the inner loop:
  ```cpp
  int valid_t = min(sp_blk_size, max(0, (int)kv_len - (int)kvLogicalOffset));
  for (int t = 0; t < valid_t; t++) { ... }
  ```
- Removed stale `assert(kv_len % KV_BLOCK == 0)` (was left from old alignment requirement)

---

## Phase 4: Sparse Prefill + Decode

Added sparse head-group-level masking (described in bmg-sdp-sparse-mask skill).

---

## Performance Ladder (Decode, kv=4K, headQ=32, headKv=2, HD=128)

| Configuration | Time | Notes |
|--------------|------|-------|
| Legacy sequential decode | ~80 µs | One thread, no parallelism |
| Two-phase, initial config | ~18 µs | Too many threads throttled |
| Q_HEAD_PER_T=8, chunk=256 | ~11.5 µs | **Production config** |
| Q_HEAD_PER_T=4, chunk=128 | ~13 µs | Fewer threads, lower util |
| Q_HEAD_PER_T=16, chunk=512 | ~14 µs | SLM overflow / frequency drop |

## Performance Ladder (Prefill, headQ=32, headKv=2, HD=128, non-causal)

| q_len × kv_len | Time | Roofline |
|----------------|------|----------|
| 512 × 1024 | ~0.08 ms | — (too small) |
| 4K × 4K | 2.52 ms | 83–86% |
| 8K × 8K | 10.0 ms | 83–86% |
| 16K × 16K | 39.04 ms | 83.8% |

---

## Phase 5: Barrier + Interleave Optimization (fp16 opt)

Three micro-optimizations applied to fp16 prefill kernel:

1. **Barrier before compensation** — move barrier() before `finalOutput *= compensationTemp` so SLM loads overlap with compensation ALU
2. **Early fp16 convert of compensationTemp** — convert fp32→fp16 before sum reduction to hide latency
3. **L-group interleaved compensation** — compensate l=0 half, SxV l=0, compensate l=1 (overlaps with l=0 DPAS pipeline), SxV l=1

**Result:** 1.3-1.5% faster for fp16. Larger benefit (3-5%) for bf16 due to expensive bf16 compensation.

File: `flash.attn.b.mha128.fp16.opt.h` (in `bmg-sdp-bf16` skill assets)

---

## Phase 6: BF16 Kernel Variants

See `bmg-sdp-bf16` skill for full details. Summary:

| Kernel | Strategy | Perf vs fp16 |
|--------|----------|-------------|
| bf16 (pure) | All bf16 DPAS + bf16 accumulator | 1.17x slower (bf16 ALU overhead) |
| bf16io (hybrid) | bf16 QK + fp16 SxV, V conv in DPAS | **0.96x faster** (V conversion free via XVE/XMX overlap) |

---

## Key Lessons

1. **No `if`/`?:` in inner loop** — runtime branches in SYCL ESIMD kernels cause SPIR-V linker errors or branch divergence. Move to `if constexpr` or host dispatch.

2. **Barrier count must match for all threads** — even one thread taking an extra `barrier()` inside a branch → GPU hang.

3. **Pre-compute loop bounds on host (or device preamble)** — causal `kvSeqOutLoopCount` computed before the loop, not checked each iteration.

4. **exp2 is faster than exp** — use `attnScoreMul = (1/sqrt(HD)) * log2e` and `exp2(score * attnScoreMul - max_scaled)`.

5. **GQA sharing** — 16× more Q-heads attend per K/V load → dramatically improves arithmetic intensity for prefill.

6. **Two-phase decode** — parallelizes kv_len dimension across WGs, then reduces. Essential for performance at kv>1K.

7. **SLM size constraints** — with Q_HEAD_PER_T=8, decode SLM is exactly 64 KB (fills the budget). Increasing parameters → SLM spill.

8. **Xe2 has no native bf16 ALU** — bf16 multiply emulated via fp32 (~3x cost). Use fp16 for all ALU-heavy paths (compensation, accumulation). See `bmg-sdp-bf16` skill.

9. **XVE/XMX parallelism** — type conversions can be hidden by interleaving them with DPAS calls. XVE converts one chunk while XMX processes the DPAS systolic pipeline.

10. **fp32 accumulator causes spilling** — `simd<float, 2048>` needs 128 GRFs, exceeding the 256 doubleGRF budget. Result: 4.5x slowdown. Use fp16/bf16 accumulators instead.

11. **Barrier placement matters** — moving barrier before compensation allows compensation ALU to overlap with SLM load latency (~1-5% gain depending on compensation cost).
