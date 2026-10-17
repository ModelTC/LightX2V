# Code Index — BMG SDP Kernels

The bundled `sdp_perf_all.cpp` and `sdp_correctness_all.cpp` also include `mask_convert.h`, `flash.attn.sparse.mha128.gqa.h`, and `decode_sdp_sparse.h`, which are absent from this checkout. They are historical reference harnesses, not directly buildable tests. Do not report the old 86-case pass count as current validation. For a self-contained dense HD=128 example, use [test_sdp_best.cpp](../../xe2-sdp-bf16/assets/test_sdp_best.cpp) with its sibling headers; it does not cover sparse/GQA decode.

## Asset Files

### flash.attn.b.mha128.gqa.precomputed_yuchen.h
**Production GQA Prefill**

| Item | Value |
|------|-------|
| Entry point | `flashAttnBMha128GQAPrecomputed<IS_CAUSAL>()` |
| Functor | `FlashAttnFunctor<IS_CAUSAL>` |
| Host launcher | `launch_flash_attention<IS_CAUSAL>()` |
| Template params | `IS_CAUSAL` (bool) |
| Key runtime args | `qState, kState, vState, normAlpha, out, activationLength, kvSeqLen, history_len, headQ, headKv` |
| Dispatch | `nd_range<2>({q_blocks, headKv*16}, {1, 16})` |
| WG threads | 16 |
| Tile: Q | `simd<fp16, 16*128>` — 16 heads × 128 dims |
| Tile: V (SLM) | 2 × 64 × 128 fp16 = 32 KB ping-pong |
| Tile: Scores | `simd<float, 16*64>` — 16 heads × 64 KV positions |
| KV block size | 64 tokens per iteration |
| Performance | 83–86% roofline |

Key patterns:
- `kvSeqOutLoopCount` pre-computed in preamble (causal shortcut)
- Main loop (iter 0..N-2): no boundary checks, always executes
- Last loop (iter N-1): boundary mask + causal mask applied after DPAS
- Exactly 1 `barrier()` per iteration (symmetry guaranteed)
- Output via `lsc_scatter` with `valid_q` predicate mask

---

### decode_sdp_gqa.h
**Production GQA Decode Full Dense**

| Item | Value |
|------|-------|
| Phase 1 entry | `flashAttnMha128GQAdecode<HD, headQ, headKv, Q_HEAD_PER_T, sp_blk_size, chunk_size>()` |
| Phase 2 entry | `flashAttnMha128GQADecodeReduce<HD, headQ>()` |
| Best config | Q_HEAD_PER_T=8, sp_blk_size=32, chunk_size=256 |
| Phase 1 dispatch | `nd_range<3>({1, chunk_num*8, headKv*2}, {1, 8, 2})` |
| Phase 2 dispatch | `nd_range<3>({1, 1, headQ/16}, {1, 1, 1})` |
| WG threads (phase 1) | sp_blk_num_per_t × head_groups_per_g = 8 × 2 = 16 |
| SLM layout | reduce + max + lse buffers; at HD=128 and the config above: 65536 + 512 + 512 = 66560 bytes (65 KiB) |
| Intermediate buffers | `[chunk_num, headQ, HD]` fp32 + lse + max |
| Performance | ~11.5 µs at kv=4K (66% BW) |

Key patterns:
- `valid_t` clamping handles non-aligned kv_len
- SLM intra-WG reduction after `barrier()`
- Only `sp_blk_idx==0` threads do the SLM reduction
- Phase 2 uses `pow(2.718f, ...)` for correction (not exp2 — only 16 ops so cost is minor)

---

### flash.attn.b.mha128.h
**Legacy Non-GQA Prefill (Historical Reference)**

| Item | Value |
|------|-------|
| Performance | ~60% roofline |
| Limitation | headQ must equal headKv |
| Status | Historical reference only; not used in production |

---

### sdp_perf_all.cpp
**Unified Performance Test**

Sections:
1. Mask convert perf
2. Prefill sparse perf
3. Prefill full perf
4. Decode full perf (N_BUF=4 cache-bust)
5. Decode sparse perf (N_BUF=4 cache-bust)

Historical compile template (requires the missing headers):
```bash
icpx sdp_perf_all.cpp -o sdp_perf_all.exe -fsycl -fsycl-targets=spir64_gen -Xs "-device bmg -options -doubleGRF" -O3
```

---

### sdp_correctness_all.cpp
**Unified Correctness Test (86 cases)**

Sections:
1. (19 cases) Mask convert correctness
2. (30 cases) Prefill sparse
3. (8 cases)  Prefill full
4. (15 cases) Decode full (various kv_len, aligned/non-aligned)
5. (12 cases) Decode sparse
6. (2 cases)  End-to-end: mask_convert → sparse SDP

Historical full-suite result: `86/86 PASSED -- ALL PASSED` (dependencies not bundled).

---

## Parameter Summary

| Parameter | Typical | Range |
|-----------|---------|-------|
| HD (head dim) | 128 | Fixed at 128 for these kernels |
| headQ | 32 | Must be multiple of 16 |
| headKv | 2 | headQ/headKv = group_size |
| group_size | 16 | headQ / headKv |
| Q_HEAD_PER_T (decode) | 8 | 4, 8, 16 tested |
| chunk_size (decode) | 256 | 128, 256 tested |
| sp_blk_size (decode) | 32 | Fixed |
| KV block size (prefill) | 64 | Fixed (1 iteration = 64 KV tokens) |
| attscale | 1/sqrt(128) ≈ 0.0884 | Application-specific |

## Compile Command

```bash
icpx <file>.cpp -o <file>.exe \
  -fsycl -fsycl-targets=spir64_gen \
  -Xs "-device bmg -options -doubleGRF" -O3
```

- Do NOT use `-doubleGRF` as a top-level flag
- Spill warning → reduce tile sizes or Q_HEAD_PER_T
