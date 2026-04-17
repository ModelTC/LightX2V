# GEMV Kernel Patterns

## 1. W4A16 SIMD Dequantization (the critical pattern)

### Memory layout
```
weight:  [N, K/2]  uint8  — two uint4 packed per byte
           byte b = weight[n][k/2]
           even k: low  nibble = b & 0x0F
           odd  k: high nibble = (b >> 4) & 0x0F
scale:   [N, K/BLOCK_SIZE]  fp16  — BLOCK_SIZE = 128
formula: weight_fp = (uint4_val - 8) * scale    (zero_point = 8)
```

### Full dequant block loop (correct)
```cpp
constexpr int BLOCK_SIZE = 128;
constexpr int NUM_BLOCKS = VL / BLOCK_SIZE;   // e.g. 1024/128 = 8

// Load K_BLOCKS scales for this VL chunk
simd<sycl::half, NUM_BLOCKS> scale_vec =
    block_load<sycl::half, NUM_BLOCKS>(scale + n * K_BLOCKS + k_base / BLOCK_SIZE);

// Load VL/2 packed bytes
simd<uint8_t, VL/2> weight_packed =
    block_load<uint8_t, VL/2>(weight + n * (K/2) + k_base/2);

simd<float, VL> weight_f;

#pragma unroll
for (int blk = 0; blk < NUM_BLOCKS; blk++) {
    float sc = static_cast<float>(scale_vec[blk]);

    // Extract 64 packed bytes covering 128 nibbles for this block
    auto p = weight_packed.template select<64, 1>(blk * 64);

    // SIMD nibble extraction — all ops on full simd<float,64>
    simd<float, 64> lo = p & 0x0F;          // even positions
    simd<float, 64> hi = (p >> 4) & 0x0F;   // odd positions

    lo = (lo - 8.0f) * sc;
    hi = (hi - 8.0f) * sc;

    // CRITICAL: strided select — NOT a scalar loop
    int base = blk * BLOCK_SIZE;
    weight_f.template select<64, 2>(base + 0) = lo;   // writes positions base+0, base+2, base+4, ...
    weight_f.template select<64, 2>(base + 1) = hi;   // writes positions base+1, base+3, base+5, ...
}
```

### Why select<COUNT, STRIDE> works
`simd.template select<N, STRIDE>(OFFSET)` returns a reference to N elements
starting at OFFSET, stepping STRIDE. Assignment writes directly into those
positions. The compiler sees this as a single vector gather/scatter instruction,
not N scalar stores.

### Common mistake — scalar loop (NEVER do this)
```cpp
// This compiles but is scalar — each iteration is one store. 289 GB/s.
for (int i = 0; i < 64; i++) {
    weight_f[blk * BLOCK_SIZE + i * 2]     = lo[i];
    weight_f[blk * BLOCK_SIZE + i * 2 + 1] = hi[i];
}
```

---

## 2. W8A16 Dequantization Pattern

```cpp
// Per-row scale (no blocking)
float scale_f = static_cast<float>(scale[n]);

for (int k = 0; k < K; k += VL) {
    simd<sycl::half, VL> input_vec = block_load<sycl::half, VL>(input + k);
    simd<float, VL> input_f = input_vec;

    simd<int8_t, VL> weight_vec = block_load<int8_t, VL>(weight + n * K + k);
    simd<float, VL> weight_f = weight_vec;    // implicit int8→float conversion
    weight_f = weight_f * scale_f;

    sum += reduce<float>(input_f * weight_f, std::plus<>());
}
```

No interleaving needed — int8 is already 1-element-per-byte, no bit packing.

---

## 3. K-Split with SLM Reduction

### Thread layout
```
work-group of size (ROWS × K_SPLIT):
  thread local_id  →  row_thread_id = local_id / K_SPLIT
                       k_thread_id   = local_id % K_SPLIT

  row_thread_id ∈ [0, ROWS)    → which output row this thread works on
  k_thread_id   ∈ [0, K_SPLIT) → which K slice
```

### Complete K-split kernel structure
```cpp
template<int ROWS, int VL, int K_SPLIT>
struct W4A16_KSplit {
    /* data pointers, N, K */

    void operator()(sycl::nd_item<1> item) const SYCL_ESIMD_KERNEL {
        // Step 1: init SLM — MUST be first
        constexpr int SLM_SIZE = ROWS * K_SPLIT * sizeof(float);
        slm_init(SLM_SIZE);

        int local_id       = item.get_local_id(0);
        int group_id       = item.get_group(0);
        int n_start        = group_id * ROWS;

        int k_thread_id    = local_id % K_SPLIT;
        int row_thread_id  = local_id / K_SPLIT;

        int n = n_start + row_thread_id;

        // Step 2: compute partial dot product
        int K_PER_THREAD = K / K_SPLIT;
        int k_start = k_thread_id * K_PER_THREAD;

        simd<float, 8> partial_sums = 0.0f;
        int acc_idx = 0;

        if (row_thread_id < ROWS && n < N) {
            for (int k_base = k_start; k_base < k_start + K_PER_THREAD; k_base += VL) {
                // ... load input, dequant weight, compute dot product ...
                partial_sums[acc_idx] += dot_product;
                acc_idx = (acc_idx + 1) & 0x7;
            }
        }

        // Step 3: store partial sum to SLM
        float partial_sum = reduce<float>(partial_sums, std::plus<>());
        if (row_thread_id < ROWS) {
            uint32_t slm_off = (row_thread_id * K_SPLIT + k_thread_id) * sizeof(float);
            slm_block_store<float, 1>(slm_off, simd<float, 1>(partial_sum));
        }

        // Step 4: synchronize
        barrier();

        // Step 5: reduce and write output (k_thread_id == 0 only)
        if (k_thread_id == 0 && row_thread_id < ROWS && n < N) {
            uint32_t slm_base = row_thread_id * K_SPLIT * sizeof(float);
            float final_sum;

            if constexpr (K_SPLIT == 1) {
                simd<float, 1> r = slm_block_load<float, 1>(slm_base);
                final_sum = r[0];
            } else if constexpr (K_SPLIT == 2) {
                simd<float, 2> r = slm_block_load<float, 2>(slm_base);
                final_sum = r[0] + r[1];
            } else if constexpr (K_SPLIT == 4) {
                simd<float, 4> r = slm_block_load<float, 4>(slm_base);
                final_sum = reduce<float>(r, std::plus<>());
            } else if constexpr (K_SPLIT == 8) {
                simd<float, 8> r = slm_block_load<float, 8>(slm_base);
                final_sum = reduce<float>(r, std::plus<>());
            }

            output[n] = sycl::half(final_sum);
        }
    }
};
```

### Launch parameters
```cpp
int local_size  = ROWS * K_SPLIT;
int num_groups  = (N + ROWS - 1) / ROWS;
int global_size = num_groups * local_size;

q.submit([&](handler& h) {
    h.parallel_for(nd_range<1>(global_size, local_size), kernel{...});
});
```

---

## 4. Multiple Accumulator Pattern (latency hiding)

```cpp
simd<float, 8> partial_sums = 0.0f;
int acc_idx = 0;

for (int k_base = ...; k_base < ...; k_base += VL) {
    // ... compute dot_product ...
    partial_sums[acc_idx] += dot_product;
    acc_idx = (acc_idx + 1) & 0x7;   // rotate 0..7
}

float s = reduce<float>(partial_sums, std::plus<>());
```

Rotating through 8 accumulators hides FMA latency (~8 cycles on Xe2).
Without this, each accumulation stalls on the previous result.

---

## 5. Bandwidth Formulas

### W4A16
```
bytes = K × sizeof(fp16)                       // input vector
      + N × (K/2) × sizeof(uint8)              // packed 4-bit weights
      + N × (K/128) × sizeof(fp16)             // scales
      + N × sizeof(fp16)                        // output
      = K×2 + N×K/2 + N×K/64 + N×2
```

### W8A16
```
bytes = K × sizeof(fp16)                       // input vector
      + N × K × sizeof(int8)                   // 8-bit weights
      + N × sizeof(fp16)                        // scale (per-row)
      + N × sizeof(fp16)                        // output
      = K×2 + N×K + N×2 + N×2
```

W4A16 uses **2× less weight bandwidth** than W8A16 for the same N,K — this is why both achieve similar absolute GB/s (weight bandwidth dominates).

---

## 6. Output Write Pattern

```cpp
// Single fp16 scalar output per row
output[n] = sycl::half(final_sum);

// OR using block_store for consistency
simd<sycl::half, 1> result = sycl::half(final_sum);
block_store<sycl::half, 1>(output + n, result);
```

Both are correct. The scalar assignment avoids the simd wrapper overhead.
