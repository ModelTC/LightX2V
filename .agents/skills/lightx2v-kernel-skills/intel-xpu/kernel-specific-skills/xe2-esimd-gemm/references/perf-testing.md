# Performance Testing Methodology for BMG ESIMD Kernels

## Core Principles

1. **Warmup before timing** — GPU needs to reach steady-state frequency.
2. **Enough iterations** — reduces noise; 1000 for compute-bound, 100 minimum.
3. **Cache busting** — rotate input buffers every iteration for memory-bound kernels.
4. **Non-zero random init** — avoid all-zero data (hides NaN bugs, unrealistic cache perf).
5. **Pre-sleep** — let other GPU activity settle before benchmark starts.

---

## Boilerplate Timing Harness

```cpp
#include <sycl/sycl.hpp>
#include <chrono>
#include <thread>
#include <cmath>
#include <random>
#include <cstdio>

// --- Random init (non-zero, half precision) ---
void rand_init(sycl::half* h_ptr, size_t n, float scale = 0.01f) {
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-scale, scale);
    for (size_t i = 0; i < n; i++) {
        float v = dist(rng);
        // Avoid exact zero and denormals
        if (std::abs(v) < 1e-4f) v = (v >= 0.f) ? 1e-4f : -1e-4f;
        h_ptr[i] = (sycl::half)v;
    }
}

// --- Cache bust buffer pool (for memory-bound kernels) ---
// For GEMM (compute-bound), single buffer is fine.
// For bandwidth-bound kernels, rotate through POOL_SIZE buffer sets.
constexpr int POOL_SIZE = 8;

// --- Benchmark harness ---
void benchmark(sycl::queue& q,
               sycl::half* A, sycl::half* B, sycl::half* C,
               int M, int N, int K,
               int warmup = 20, int iters = 1000) {

    // Pre-sleep: let GPU settle
    printf(">>> Sleeping 5 seconds - attach profiler now if needed <<<\n");
    std::this_thread::sleep_for(std::chrono::seconds(5));
    printf(">>> Starting <<<\n\n");

    auto run_kernel = [&]() {
        q.submit([&](sycl::handler& h) {
            h.parallel_for(
                sycl::nd_range<2>({(size_t)(M/256), (size_t)(N/256)}, {1, 1}),
                [=](sycl::nd_item<2> item) SYCL_ESIMD_KERNEL {
                    gemm_kernel(item, A, B, C, M, N, K);
                }
            );
        }).wait();
    };

    // Warmup
    for (int i = 0; i < warmup; i++) run_kernel();

    // Timed run
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iters; i++) run_kernel();
    auto t1 = std::chrono::high_resolution_clock::now();

    double elapsed_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    double ms_per_iter = elapsed_ms / iters;
    double flops = 2.0 * M * N * K;
    double tflops = flops / (ms_per_iter * 1e9);
    double peak = 135.0;

    printf("Results (%d iters, peak=%.0f TFLOPS):\n", iters, peak);
    printf("  Time/iter  : %.4f ms\n", ms_per_iter);
    printf("  TFLOPS     : %.2f\n", tflops);
    printf("  %% of %.0fT  : %.1f%%\n", peak, tflops / peak * 100.0);
}
```

## Cache-Bust Pattern (memory-bound kernels)

```cpp
// Allocate POOL_SIZE independent input arrays
std::vector<sycl::half*> pool_A(POOL_SIZE), pool_B(POOL_SIZE);
for (int i = 0; i < POOL_SIZE; i++) {
    pool_A[i] = sycl::malloc_device<sycl::half>(M * K, q);
    pool_B[i] = sycl::malloc_device<sycl::half>(K * N, q);
    // init with random data...
}

// In benchmark loop, rotate pointer each iteration
for (int i = 0; i < iters; i++) {
    int idx = i % POOL_SIZE;
    run_kernel(pool_A[idx], pool_B[idx], C);
}
```

**Why**: for compute-bound GEMM, single buffer is fine (L2 holds working set). For memory-latency-sensitive kernels, rotating buffers prevents the L2 from caching the entire dataset across iterations, giving a more realistic bandwidth measurement.

## Random Non-Zero Init

```cpp
// Host-side random fill, copy to device
std::vector<sycl::half> h_A(M * K), h_B(K * N);
rand_init(h_A.data(), h_A.size(), 0.01f);
rand_init(h_B.data(), h_B.size(), 0.01f);
sycl::half* d_A = sycl::malloc_device<sycl::half>(M * K, q);
q.memcpy(d_A, h_A.data(), M * K * sizeof(sycl::half)).wait();
```

Rationale: all-zero input triggers special-case fast-paths in hardware (zero × anything = zero, skipped by some multiplier implementations). Use small random values (scale ~0.01) to avoid fp16 overflow in GEMM output.

## Correctness Check with NaN Guard

```cpp
bool check_correctness(sycl::half* d_C, float* ref_C, int M, int N) {
    std::vector<sycl::half> h_C(M * N);
    // copy d_C to host...

    double rms = 0.0, ref_rms = 0.0;
    int nan_count = 0, outlier_count = 0;

    for (int i = 0; i < M * N; i++) {
        float gpu_val = (float)h_C[i];
        float ref_val = ref_C[i];

        // Check NaN first — NaN in denominator silently passes threshold
        if (std::isnan(gpu_val)) { nan_count++; continue; }

        float diff = gpu_val - ref_val;
        rms     += (double)(diff * diff);
        ref_rms += (double)(ref_val * ref_val);
    }

    if (nan_count > 0) {
        printf("FAIL: %d NaN values in output\n", nan_count);
        return false;
    }

    double rel_rms = std::sqrt(rms / (ref_rms + 1e-30));
    printf("rel RMS = %.4f%%\n", rel_rms * 100.0);

    // Count true outliers (ignore near-zero denominator)
    for (int i = 0; i < M * N; i++) {
        float ref_val = ref_C[i];
        if (std::abs(ref_val) < 1e-3f) continue;  // skip near-zero denom
        float rel_err = std::abs((float)h_C[i] - ref_val) / std::abs(ref_val);
        if (rel_err > 0.05f) outlier_count++;
    }

    printf("Outliers (>5%% rel, non-trivial denom): %d\n", outlier_count);
    return (rel_rms < 0.005 && outlier_count == 0);
}
```

## Frequency Measurement

VTune or `intel_gpu_top` to observe actual GPU frequency:

```bash
# During benchmark run, in another terminal:
intel_gpu_top -d 1000   # 1 second interval
# Look for "freq" column — should be ~1850 MHz for GEMM on BMG
```

Low frequency (~1500 MHz or below) indicates thermal throttling or excessive XVE/XMX parallelism.

## Expected Results (BMG, M=N=K=4096, FP16)

| Kernel | TFLOPS | % of 135T | Notes |
|--------|--------|-----------|-------|
| fp16_gemm_nopf | 117.10 | 86.7% | Payload CSE, no prefetch, baseline |
| fp16_gemm_nopf3 (induction vars) | 117.44 | 87.0% | +0.3% marginal gain |
| fp16_gemm_walk LINEAR | ~118.0 | 87.4% | Walk order sweep baseline |
| fp16_gemm_distpf PF_DIST=0 | 118.33 | 87.7% | Same as nopf effectively |
| fp16_gemm_2pay (2×payA+2×payB) | 113.79 | 84.3% | −2.8%: access pattern regression |
| fp16_gemm_distpf PF_DIST≥1 | <110 | <82% | Prefetch always hurts |
| fp16_gemm (original, bunched pf) | ~109 | 80.7% | Starting point |
| K_STEP=64+ | ~110 | 81.5% | EU scheduler window issue |
| L1UC cache hint | ~44 | 32.6% | Catastrophic — never use |
| COL_SWIZZLE walk order | ~88 | 65.2% | GPU parallelism starvation |
