# Correctness Testing — BMG SDP Kernels

## Test Infrastructure

All 86 test cases live in `assets/sdp_correctness_all.cpp`.

Expected result: `86/86 PASSED -- ALL PASSED`

| Section | Tests | What it covers |
|---------|-------|----------------|
| 1 | 19 | mask_convert: kv_len alignment, q_len, q_blocks boundary |
| 2 | 30 | Prefill sparse: causal/non-causal, various shapes |
| 3 | 8  | Prefill full (dense): causal/non-causal, alignment |
| 4 | 15 | Decode full: various kv_len including non-aligned |
| 5 | 12 | Decode sparse: kv_block lists, partial kv coverage |
| 6 | 2  | Combined: mask_convert → sparse SDP end-to-end |

---

## NaN Guard

**Always check for NaN before computing max_diff or rel_rms.**

```cpp
int nan_count = 0;
for (int i = 0; i < total; i++) {
    if (std::isnan((float)h_out[i])) nan_count++;
}
if (nan_count > 0) {
    printf("FAIL: %d NaN in output\n", nan_count);
    return false;
}
```

**NaN root cause in SDP:** `fp32SoftMaxTemp` (the softmax denominator) becomes 0 when all attention scores are -inf. This happens when:
- `kv_len = 0`
- Causal mask with `q_pos = 0` and `history_len = 0` → no valid KV positions
- Sparse mask with empty kv_block list

**Fix:** always have at least one valid KV token before normalizing. Or guard: `if (lse > 0) output /= lse; else output = 0;`

---

## Thresholds

```cpp
// For SDP output (fp16 vs CPU fp32 reference):
float scale = attscale;  // typically 1/sqrt(128) ≈ 0.0884
float thresh = (scale <= 0.1f) ? 0.1f : scale * 1.5f;
// thresh ≈ 0.1 for typical SDP scale

// Pass condition:
bool pass = (max_diff < thresh) && (nan_count == 0);
```

---

## CPU Reference Pattern

```cpp
// CPU reference: naive O(qlen × kv_len × HD) attention
void cpu_sdp_reference(
    const float* Q,   // [q_len, headQ, HD]
    const float* K,   // [kv_len, headKv, HD]
    const float* V,   // [kv_len, headKv, HD]
    float* out,       // [q_len, headQ, HD]
    int q_len, int kv_len, int headQ, int headKv, int HD,
    float scale, bool causal, int history_len)
{
    int group_size = headQ / headKv;
    for (int q = 0; q < q_len; q++) {
        for (int h = 0; h < headQ; h++) {
            int kv_h = h / group_size;
            // QK scores
            std::vector<float> scores(kv_len, -1e30f);
            for (int k = 0; k < kv_len; k++) {
                if (causal && k > history_len + q) continue;
                float s = 0;
                for (int d = 0; d < HD; d++)
                    s += Q[q*headQ*HD + h*HD + d] * K[k*headKv*HD + kv_h*HD + d];
                scores[k] = s * scale;
            }
            // Softmax
            float mx = *std::max_element(scores.begin(), scores.end());
            float sum = 0;
            for (auto& s : scores) { s = std::exp(s - mx); sum += s; }
            for (auto& s : scores) s /= sum;
            // Weighted V
            for (int d = 0; d < HD; d++) {
                float acc = 0;
                for (int k = 0; k < kv_len; k++)
                    acc += scores[k] * V[k*headKv*HD + kv_h*HD + d];
                out[q*headQ*HD + h*HD + d] = acc;
            }
        }
    }
}
```

---

## Corner Cases to Test

| Case | Why it matters |
|------|---------------|
| `kv_len = 1` | Minimum; softmax on single element |
| `kv_len` not 64-aligned | Last KV block partially valid |
| `kv_len` not 8-aligned | Fine-grained alignment check |
| `kv_len < chunk_size` | Decode: chunk_num=1, boundary valid_t < sp_blk_size |
| `q_len` not 16-aligned | Prefill: last q_block partially valid |
| Causal, `q_pos=0, history=0` | Only kv[0] is valid |
| `headQ = headKv` (GQA group=1) | No GQA sharing |
| `headQ/headKv = 16` | Full GQA (production config) |
| Decode with `kv_len = chunk_size × N` | Exact boundary |
| Decode with `kv_len = chunk_size × N + 1` | One extra token |

---

## Common Failures and Fixes

| Symptom | Cause | Fix |
|---------|-------|-----|
| NaN everywhere | All softmax scores -inf → denominator 0 | Check kv_len > 0; check causal boundary |
| Large max_diff at last q_block | Out-of-bounds Q read for padded positions | Use `valid_q` mask in scatter write |
| Wrong output for kv_len non-aligned | KV boundary not clamped | Use `valid_t = min(sp_blk_size, max(0, kv_len - kvLogicalOffset))` |
| Assertion fail `kv_len % KV_BLOCK == 0` | Stale assert left after ceiling-division fix | Remove the assert |
| Silent wrong result | DPAS argument order swapped | `dpas<8,8>(acc, B_vnni, A)` — B first, A second |
