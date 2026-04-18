# Hardware Constraints — Intel Battlemage (BMG / Xe2)

## GPU Specs

| Resource | Value | Notes |
|----------|-------|-------|
| Peak FP16 Throughput | 135 TFLOPS | XMX DPAS |
| Peak Memory Bandwidth | 530 GB/s | HBM |
| XVE (EU) count | 512 | 32 WGs × 16 threads/WG typical |
| GRF per thread (doubleGRF) | 256 × 32B = 8 KB | 256 registers, 32 bytes each |
| SLM per workgroup | 64 KB | Shared Local Memory |
| L1 cache | ~512 KB per XVE cluster | Effective for 2D block loads |
| L2 cache | ~16 MB | Shared across all XVEs |

---

## Work-Group Sizing

- **Max WG threads = 32** when `doubleGRF` is enabled (not 64).
- `doubleGRF` gives each thread 256 GRF registers (double the default 128).
- Always compile with `-Xs "-device bmg -options -doubleGRF"`.
- Setting `nd_range` local size > 32 with doubleGRF → compile warning or runtime hang.

**SDP prefill WG:**
- `LocalRange = {1, 16}` → 16 threads per WG. OK.
- Each thread handles 1 query position × 16 Q-heads.

**SDP decode WG:**
- `LocalRange = {1, sp_blk_num_per_t, head_groups_per_g}`
- With Q_HEAD_PER_T=8, chunk_size=256, sp_blk_size=32: sp_blk_num_per_t=8, head_groups_per_g=2 → 16 threads. OK.
- Total WG threads = sp_blk_num_per_t × head_groups_per_g ≤ 32.

---

## Barrier Rules

- **All threads in a WG must execute the same number of `barrier()` calls.**
- Unequal barrier counts → GPU hang (silent, infinite stall, no error).
- Common mistake: putting `barrier()` inside a conditional branch that not all threads take.
- **Fix**: restructure so barrier is always reached, or use `if constexpr` on compile-time conditions.
- Prefill uses exactly 1 barrier per KV iteration (between S×V and V→SLM store).
- Decode uses exactly 1 barrier (between per-sp_blk compute and SLM reduce).

---

## GRF Budget

Spill occurs when kernel uses more GRF than the 256-register budget.

- `icpx` will emit: `warning: ... spilled N bytes to stateless memory`
- Any spill → significant performance regression (register spill to memory = many extra loads/stores).
- Common causes in SDP: too many live `simd<float, HD>` accumulator variables.
- Fix: reduce tile sizes, reduce unroll factors, split phases.

**Tile sizes in production prefill (flash.attn.b.mha128.gqa.precomputed_yuchen.h):**
- Q tile: `simd<fp16, 16×128>` = 4 KB
- QK scores: `simd<float, 16×64>` = 4 KB
- SoftMax accumulator: `simd<fp16, 16×128>` = 4 KB
- Temp buffers: `simd<float, 16×32>` = 2 KB
- Total ≈ 14 KB / 8 KB GRF → spills some, but compiler manages OK with doubleGRF

---

## Frequency Throttling

- Adding more XVE + XMX parallelism (bigger unroll, more concurrent DPAS) → **lower operating frequency**.
- Target: minimize XVE scalar/vector ops in the inner loop. DPAS dominates; extra XVE work causes throttle.
- Observed DPAS frequency: ~1850 MHz for well-tuned SDP.
- Under-tuned kernels: ~1500 MHz or lower.

---

## SLM Layout (Decode)

```
SLM for decode (one WG):
  [0, slm_reduce_size)        : output accumulators [head_groups × sp_blks × Q_HEAD_PER_T × HD] fp32
  [slm_reduce_size, +max_sz)  : per-block max [head_groups × sp_blks × Q_HEAD_PER_T] fp32
  [+max_sz, +lse_sz)          : per-block lse [head_groups × sp_blks × Q_HEAD_PER_T] fp32
```

With Q_HEAD_PER_T=8, sp_blk_num_per_t=8, head_groups_per_g=2, HD=128:
- slm_reduce_size = 2×8×8×128×4 = 65536 bytes (64 KB) — exactly fills SLM!
- Decode SLM usage is tight; do not increase these parameters.

---

## SLM Layout (Prefill)

```
SLM for prefill (one WG):
  [0, slmSizeV)  : V ping-pong buffer = 2 × 64 × 128 × 2 bytes = 32 KB
```

Only 32 KB used for prefill → comfortable margin.
