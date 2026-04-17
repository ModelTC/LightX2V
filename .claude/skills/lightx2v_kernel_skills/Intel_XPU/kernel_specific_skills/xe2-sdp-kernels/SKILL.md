---
name: xe2-sdp-kernels
description: Use this skill when writing, optimizing, benchmarking, or debugging Flash Attention SDP kernels (prefill full-dense or decode full-dense) targeting Intel Xe2 (Lunar Lake/LNL, Battlemage/BMG) GPU using SYCL ESIMD. Xe2 is the GPU architecture; LNL and BMG are product names. Covers the complete optimization journey from scalar loops to DPAS, GQA layout, two-phase decode, perf testing methodology, and all hardware constraints discovered through implementation.
---

# Xe2 (Lunar Lake/LNL, Battlemage/BMG) ESIMD Flash Attention — SDP Prefill & Decode (Full Dense)

Specialized knowledge for GQA flash attention on Intel Xe2 (Lunar Lake/LNL, Battlemage/BMG).
Reference files hold detail; this file holds critical rules and workflow.

---

## Quick-Reference Rules (must follow every time)

### Hardware limits
- **Max WG threads = 32** when `doubleGRF` is on. Never set nd_range WG size > 32.
- **Always compile with `doubleGRF`** — mandatory for SDP tile sizes. Do not remove.
- **Barriers**: every thread in WG must execute the same number of `barrier()` calls. Unequal counts = GPU hang (silent, infinite stall). See `references/hardware-constraints.md`.

### Inner-loop rules
- **No `if`, no `?:`** inside the KV iteration loop body. Move all runtime conditionals to host.
- **Pre-compute loop counts on host** (e.g. causal `kvSeqOutLoopCount` based on max q-pos in WG).
- **Use `index +=` increments** — never recompute full coordinate expressions each iteration.
- **More XVE + XMX parallelism lowers frequency** — minimize XVE ops in hot loop.
- **Separate last iteration** from main loop to handle boundary masking without branches.

### API namespace
- Use `sycl::ext::intel::experimental::esimd` for `lsc_load_2d`, `lsc_prefetch_2d`, `config_2d_mem_access`.
- Use `sycl::ext::intel::esimd::xmx::dpas` for DPAS.
- Use `sycl::ext::intel::esimd::block_load<T,N>` for 1D contiguous loads.
- See `references/lsc-memory-ops.md` for full API.

### DPAS / VNNI
- DPAS signature: `dpas<8, 8>(acc, B_vnni, A_rowmaj)` — B first, A second.
- For S×V: softmax weights must be VNNI-packed (interleave pairs of fp16 before DPAS).
- Use `exp2()` with `attnScoreMul = (1/sqrt(HD)) * log2e` — faster than `exp()`.
- See `references/kernel-patterns.md`.

### kv_len alignment
- **kv_len does NOT need to be chunk_size-aligned.** Use `valid_t` clamping:
  ```cpp
  int valid_t = min(sp_blk_size, max(0, kv_len - kvLogicalOffset));
  for (int t = 0; t < valid_t; t++) { ... }
  ```
- Use **ceiling** division: `chunk_num = (kv_len + chunk_size - 1) / chunk_size`.

### GQA layout
- Q: `[q_len, headQ, HD]` — stride between tokens: `headQ * HD * sizeof(fp16)`
- K/V: `[kv_len, headKv, HD]` — stride: `headKv * HD * sizeof(fp16)`
- Decode Q: `[headQ, HD]` (single token)
- `group_size = headQ / headKv`; each kv_head serves group_size q_heads.

### Performance testing
- **Cache-bust** (N_BUF=4) for memory-bound kernels (decode, mask_convert) — rotate buffer sets every iteration.
- **Warmup**: 5 iters minimum; 20 preferred for stable frequency.
- **Iteration count**: 100 iters minimum; 1000 for compute-bound kernels.
- **Random non-zero init** — avoid all-zero inputs (hide NaN bugs, unrealistic cache perf).
- See `references/perf-testing.md`.

### Correctness testing
- Check **NaN count** before computing max_diff (NaN denominator silently passes threshold).
- Common NaN source: `fp32SoftMaxTemp` (softmax denominator) → 0 when all scores are -inf.
- CPU reference threshold: `thresh = (scale <= 0.1f) ? 0.1f : scale * 1.5f`.
- See `references/correctness-testing.md`.

### Compile command
```bash
icpx <src>.cpp -o <out>.exe \
  -fsycl -fsycl-targets=spir64_gen \
  -Xs "-device bmg -options -doubleGRF" -O3
```
- Do **not** use `-doubleGRF` as a top-level flag — it must be inside `-Xs "..."`.
- Spill warning `warning: ... spilled ... bytes` → reduce tile sizes.

### SPIR-V linker errors
- Cause: runtime `if` in kernel that can be moved to host.
- Fix: `if constexpr` (template param) or dispatch multiple template instantiations from host.

---

## Workflow

1. **Write kernel** following inner-loop rules above.
2. **Compile** — check for spill warnings. Any spill > 0 causes significant regression.
3. **Correctness test** at small sizes (q=512, kv=1024). Check NaN. Check max_diff.
4. **Corner cases**: q not 16-aligned, kv not chunk-aligned, kv not 8-aligned, kv < chunk_size.
5. **Benchmark** at production sizes with 5 warmup + 100 iters, N_BUF=4 for BW-bound.
6. **Iterate**: reduce XVE ops, verify barrier symmetry, check frequency stability.

---

## Assets (ready to compile)

```bash
icpx <file>.cpp -o <file>.exe -fsycl -fsycl-targets=spir64_gen -Xs "-device bmg -options -doubleGRF" -O3
```

| Asset | Purpose | Perf |
|-------|---------|------|
| `assets/flash.attn.b.mha128.gqa.precomputed_yuchen.h` | **Production prefill** GQA, causal/non-causal template | 83–86% roofline |
| `assets/decode_sdp_gqa.h` | **Production decode** full-dense, two-phase | 65–78% roofline |
| `assets/flash.attn.b.mha128.h` | Legacy non-GQA prefill (historical reference) | ~60% roofline |
| `assets/sdp_perf_all.cpp` | **Unified perf test** — all 5 kernels, sections 1–5 | run directly |
| `assets/sdp_correctness_all.cpp` | **Unified correctness** — 86 test cases, 6 sections | run directly |

Expected: `sdp_correctness_all.exe` → `86/86 PASSED -- ALL PASSED`

---

## Related Skills

| Skill | When to use |
|-------|------------|
| `xe2-sdp-bf16` | BF16 and bf16io (hybrid) flash attention kernels — bf16 ALU limits, V conversion interleaving, mixed-precision strategies |
| `xe2-kernel-testing` | General correctness/perf testing patterns for any Xe2 ESIMD kernel |
| `sycl-esimd-build` | Compilation flags, doubleGRF, spill detection |

---

## Reference files

| File | Contents |
|------|----------|
| `references/hardware-constraints.md` | Xe2/BMG GRF, L1, SLM, WG, barrier, frequency limits |
| `references/kernel-patterns.md` | DPAS tile layout, VNNI packing, online softmax, SLM ping-pong, coord patterns |
| `references/lsc-memory-ops.md` | Full LSC API: `lsc_load_2d`, `lsc_prefetch_2d`, `block_load`, `lsc_scatter`, cache hints |
| `references/perf-testing.md` | Cache-bust boilerplate, timing harness, random init, NaN check |
| `references/correctness-testing.md` | CPU reference pattern, thresholds, corner case list |
| `references/optimization-history.md` | Full story from 0.0004T scalar → 86% roofline DPAS |
| `references/code-index.md` | Per-file annotations, dispatch shapes, parameter summary |
