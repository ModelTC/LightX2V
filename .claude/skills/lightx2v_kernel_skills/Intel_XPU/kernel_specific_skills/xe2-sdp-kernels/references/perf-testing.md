# Performance Testing Methodology — BMG SDP Kernels

## Core Principles

1. **Warmup before timing** — GPU needs steady-state frequency. Min 5 iters; 20 preferred.
2. **Enough iterations** — 100 min; 1000 for compute-bound to reduce noise.
3. **Cache busting (N_BUF=4)** — rotate input buffers every iteration for memory-bound kernels.
4. **Non-zero random init** — avoid all-zero inputs (hides NaN bugs; unrealistically fast cache results).
5. **Different buffer sets** — memory-bound SDP kernels must use N_BUF ≥ 4 rotating buffers.

---

## Cache-Bust Pattern (SDP kernels are memory-bound)

SDP decode and prefill are memory-bandwidth bound. Without cache-busting, the GPU L2 holds the entire KV dataset across iterations → measured BW appears much higher than reality.

```cpp
constexpr int N_BUF = 4;

// Allocate N_BUF independent KV buffer sets
std::vector<sycl::half*> d_K_bufs(N_BUF), d_V_bufs(N_BUF);
for (int i = 0; i < N_BUF; i++) {
    d_K_bufs[i] = sycl::malloc_device<sycl::half>(kv_len * headKv * HD, q);
    d_V_bufs[i] = sycl::malloc_device<sycl::half>(kv_len * headKv * HD, q);
    // ... random init and copy ...
}

// Timed loop — rotate buffer index each iteration
for (int iter = 0; iter < iters; iter++) {
    int buf_idx = iter % N_BUF;
    flashAttnMha128GQAdecode<HD, headQ, headKv, Q_HEAD_PER_T, sp_blk_size, chunk_size>(
        d_Q, d_K_bufs[buf_idx], d_V_bufs[buf_idx],
        d_out, d_lse, d_max, kv_len, attscale, dpcpp_queue);
    flashAttnMha128GQADecodeReduce<HD, headQ>(
        d_out, d_lse, d_max, d_final, chunk_num, dpcpp_queue);
    dpcpp_queue.wait();
}
```

---

## Random Non-Zero Init

```cpp
void rand_init_fp16(sycl::half* h_ptr, size_t n, float scale = 0.01f) {
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-scale, scale);
    for (size_t i = 0; i < n; i++) {
        float v = dist(rng);
        if (std::abs(v) < 1e-4f) v = (v >= 0.f) ? 1e-4f : -1e-4f;  // no exact zero
        h_ptr[i] = (sycl::half)v;
    }
}
```

Use scale 0.01f to avoid fp16 overflow in accumulated QK products.

---

## Timing Harness

```cpp
#include <chrono>

// Warmup
for (int i = 0; i < 20; i++) run_kernel(/* ... */);
dpcpp_queue.wait();

// Timed
auto t0 = std::chrono::high_resolution_clock::now();
for (int iter = 0; iter < iters; iter++) {
    int buf = iter % N_BUF;
    run_kernel(buf);
    dpcpp_queue.wait();
}
auto t1 = std::chrono::high_resolution_clock::now();

double ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / iters;
```

---

## Bandwidth Calculation (memory-bound decode)

```
BW = (bytes_read + bytes_written) / time_per_iter

For decode kv=4K, headQ=32, headKv=2, HD=128, fp16:
  Read Q:    32 × 128 × 2 bytes = 8 KB
  Read K:    4096 × 2 × 128 × 2 bytes = 2 MB
  Read V:    4096 × 2 × 128 × 2 bytes = 2 MB
  Write out: 32 × 128 × 2 bytes = 8 KB
  Total ≈ 4.016 MB per decode call (dominated by K+V reads)

At 11.5 µs: 4.016 MB / 11.5e-6 s = 349 GB/s ≈ 66% of 530 GB/s peak
```

---

## Expected Performance (BMG, headQ=32, headKv=2, HD=128)

### Decode Full Dense

| kv_len | Time (µs) | BW (GB/s) | % of 530 |
|--------|-----------|-----------|----------|
| 1K     | ~4.5      | ~200      | ~38%     |
| 4K     | ~11.5     | ~349      | ~66%     |
| 8K     | ~20       | ~400      | ~75%     |
| 16K    | ~38       | ~420      | ~79%     |

### Prefill Full Dense (kv=q, non-causal)

| q_len | kv_len | Time (ms) | % roofline |
|-------|--------|-----------|------------|
| 4K    | 4K     | 2.52      | 83–86%     |
| 8K    | 8K     | 10.0      | 83–86%     |
| 16K   | 16K    | 39.04     | 83.8%      |

---

## sdp_perf_all.cpp Sections

| Section | Kernel | Config |
|---------|--------|--------|
| 1 | mask_convert | Various kv_len, q_len |
| 2 | Prefill sparse | headQ=32, headKv=2, HD=128 |
| 3 | Prefill full | headQ=32, headKv=2, HD=128 |
| 4 | Decode full | headQ=32, headKv=2, HD=128, N_BUF=4 |
| 5 | Decode sparse | headQ=32, headKv=2, HD=128, N_BUF=4 |
