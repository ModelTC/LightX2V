# Hardware Constraints — Intel Battlemage (BMG / Xe2)

## GPU Specs

Separate historical benchmark inputs from SKU limits. Verified B580 resource sizes follow the [hardware reference](../../../kernel-basic-skills/intel-gpu-hw-info/SKILL.md); query the target device and kernel before changing launch or SLM parameters.

| Resource | Value | Notes |
|----------|-------|-------|
| Historical FP16 Throughput Input | 135 TFLOPS | XMX DPAS; not a B580 specification |
| Historical Memory Bandwidth Input | 530 GB/s | BMG uses GDDR6, not HBM; remeasure on target |
| Example launched ESIMD work-items | 512 | 32 WGs × 16 work-items/WG; not the XVE count |
| B580 XVE count | 160 | 20 cores × 8 vector engines |
| GRF per thread (B580, large mode) | 256 × 64B = 16 KiB | Regular mode: 128 × 64B = 8 KiB |
| B580 SLM per core / max per workgroup | 128 KiB / 128 KiB | Resident WGs share the per-core capacity |
| B580 L1 cache | 256 KB per XE core | SKU-specific capacity |
| B580 L3 cache | 18 MB | Historical notes estimated ~16 MB and called it L2 |

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
- Listed vectors total ≈ 14 KiB against B580's 16 KiB large-register budget. Actual spills depend on live ranges and other values; inspect compiler output rather than inferring spills from this sum alone.

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
- slm_reduce_size = 2×8×8×128×4 = 65536 bytes (64 KiB).
- max and lse each add 2×8×8×4 = 512 bytes: the [bundled kernel](../assets/decode_sdp_gqa.h) requests 66560 bytes (65 KiB) total, not 64 KiB.
- Check the total allocation against the target's local memory limit and allocation granularity before increasing parameters; SLM capacity and occupancy both matter.

---

## SLM Layout (Prefill)

```
SLM for prefill (one WG):
  [0, slmSizeV)  : V ping-pong buffer = 2 × 64 × 128 × 2 bytes = 32 KB
```

Only 32 KB used for prefill → comfortable margin.
