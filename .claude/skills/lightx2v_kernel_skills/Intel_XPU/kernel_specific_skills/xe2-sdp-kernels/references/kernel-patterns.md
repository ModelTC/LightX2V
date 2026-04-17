# Kernel Patterns — BMG SDP ESIMD

## GQA Layout

```
Q:   [q_len, headQ,  HD]   stride between tokens: headQ  × HD × sizeof(fp16)
K:   [kv_len, headKv, HD]  stride between tokens: headKv × HD × sizeof(fp16)
V:   [kv_len, headKv, HD]  stride between tokens: headKv × HD × sizeof(fp16)
Out: [q_len, headQ,  HD]

Decode Q: [headQ, HD]  (single token, qlen=1)

group_size = headQ / headKv
each kv_head serves group_size consecutive q_heads
kv_head_idx = q_head_idx / group_size
```

**Production config:** headQ=32, headKv=2, HD=128, group_size=16.

---

## DPAS Call Signature

```cpp
// B first, A second — this is opposite of conventional notation
result = dpas<8, 8, AccT, AccT, BT, AT>(acc, B_vnni, A_rowmaj);
//                                        ^^^  ^^^^^^   ^^^^^^^^
//                                        acc  B first  A second
```

For Q×K (S×K scores, fp32 accumulate):
```cpp
auto ccTile = dpas<8, 8, float, float, fp16, fp16>(
    simd<float, 128>(ccTile.data()),    // acc
    simd<fp16, 256>(aaTile.data()),     // B = Q tile (VNNI-like via transpose load)
    simd<fp16, 128>(bbTile.data()));    // A = K tile
```

For S×V (fp16 accumulate):
```cpp
auto ccTile = dpas<8, 8, fp16, fp16, fp16, fp16>(
    simd<fp16, 128>(ccTile.data()),
    simd<fp16, 256>(aaTile.data()),     // B = softmax weights (VNNI-packed)
    simd<fp16, 128>(bbTile.data()));    // A = V tile from SLM
```

**VNNI packing for S×V weights:**
Softmax weights are fp32 → need to interleave pairs into fp16 layout for DPAS B operand:
```cpp
// Convert fp32 scores to fp16 and VNNI-interleave
#pragma unroll
for (int k = 0; k < 4; k++) {
    #pragma unroll
    for (int kk = 0; kk < 2; kk++) {
        tempBufferAsFp16.select<32, 2>(128 * k + 64 * kk)     = tempOutput.select<32, 1>(128 * k + 64 * kk);
        tempBufferAsFp16.select<32, 2>(128 * k + 64 * kk + 1) = tempOutput.select<32, 1>(128 * k + 64 * kk + 32);
    }
}
```

---

## Online Softmax (exp2 trick)

Use `exp2` with pre-multiplied scale for speed (avoids natural-log base conversion each element):

```cpp
constexpr float matMulQuantCoeff = 1.0f / sqrtf(128.0f);  // = 0.08838...
// Combine with log2e once, use in inner loop
constexpr float attnScoreMul = matMulQuantCoeff * sycl::ext::intel::esimd::detail::log2e;

// In softmax:
score_scaled = score * attnScoreMul - current_max_scaled;
exp2_score   = __ESIMD_NS::exp2<float, N, float>(score_scaled);
// This gives exp(score * scale - current_max * scale) correctly
```

Online softmax update pattern:
```cpp
simd<float, N> max_val = FP32_MIN;
simd<float, N> sum_exp = 0;
simd<float, N> output  = 0;

for each KV token t:
    score = Q · K[t]  // raw dot product
    old_max = max_val
    max_val = max(score, old_max)

    correction = exp2((old_max - max_val) * attnScoreMul)
    output = output * correction
    sum_exp = sum_exp * correction

    exp_score = exp2((score - max_val) * attnScoreMul)
    sum_exp += exp_score
    output += exp_score * V[t]

output /= sum_exp  // final normalize
```

---

## SLM Ping-Pong (Prefill V tiles)

Prefill overlaps V loading with QK compute using double-buffered SLM.

```
SLM = 2 × 64 × 128 × 2 bytes = 32 KB
  slot 0: [0,   32KB)  — "ping"
  slot 1: [32KB, 64KB) — "pong"  (NOT used — only 32 KB needed)

Actually: slmPingpongLoad alternates 0/1 × 64×128×sizeof(fp16)
```

```cpp
// Main loop: load V for next block while computing S×V for current block
uint32_t slmPingpongLoad  = (loopIdx & 1) * 64 * 128 * sizeof(fp16);
uint32_t slmPingpongStore = ((loopIdx+1) & 1) * 64 * 128 * sizeof(fp16);

// Compute S×V from SLM[slmPingpongLoad]
// Store next V tile to SLM[slmPingpongStore]
// barrier() — synchronize before next iteration reads the freshly stored V
```

---

## kv_len Clamping (valid_t)

For decode, the last chunk may have fewer than `chunk_size` tokens:

```cpp
// KV offset for this sp_blk:
uint32_t kvLogicalOffset = chunk_idx * chunk_size + sp_blk_idx * sp_blk_size;

// Clamp to actual kv_len (handles non-aligned kv_len):
int valid_t = (int)sp_blk_size;
int tok_end = (int)kv_len - (int)kvLogicalOffset;
if (tok_end < valid_t) valid_t = tok_end > 0 ? tok_end : 0;

// Only iterate over valid tokens:
for (int t = 0; t < valid_t; t++) {
    // load K[kvRealOffset + t*headKv*HD], V[same]
}
```

For prefill, kv_block boundary uses `kvSeqLen` as an OOB mask on scores:
```cpp
// In last loop iteration, apply boundary mask:
tempOutput.select<16, 1>(base_idx).merge(FP32_MIN,
    softmaxPositions.select<16, 0>(kk) >= kvSeqLen);
```

---

## Causal Mask (Prefill)

Pre-compute per-WG loop count on host to avoid runtime `if` in inner loop:

```cpp
if constexpr (IS_CAUSAL) {
    int max_q_in_wg = h * 16 + 15;  // h = WG's query block index
    max_q_in_wg = min(max_q_in_wg, activationLength - 1);
    int max_kv_pos = history_len + max_q_in_wg;
    kvSeqOutLoopCount = (max_kv_pos + 64) / 64;
    kvSeqOutLoopCount = min(kvSeqOutLoopCount, (kvSeqLen + 63) / 64);
} else {
    kvSeqOutLoopCount = (kvSeqLen + 63) / 64;
}
```

Apply mask inside loop using SIMD vectorized compare (not scalar if):
```cpp
simd<int, 16> v_kv_pos(kv_pos);
auto mask = v_kv_pos > causal_boundaries;
tempOutput.select<16, 1>(base_idx).merge(FP32_MIN, mask);
```

---

## Dispatch Shape Summary

### Prefill
```cpp
nd_range<2>({q_blocks * 1,  headKv * 16},
            {1,              16})
// q_blocks = (q_len + 15) / 16
// Each WG: 16 threads, each handling 1 q_pos × 16 heads
// One WG covers 16 query positions × headQ Q-heads (via GQA: headKv group)
```

### Decode Phase 1
```cpp
nd_range<3>({1, chunk_num * sp_blk_num_per_t, headKv * head_groups_per_g},
            {1, sp_blk_num_per_t, head_groups_per_g})
```

### Decode Phase 2 (Reduce)
```cpp
nd_range<3>({1, 1, headQ / 16},
            {1, 1, 1})
// Each thread handles 16 consecutive Q-heads
```
