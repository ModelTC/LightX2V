# GEMV Correctness Testing

## CPU reference pattern

### W4A16
```cpp
void cpu_gemv_w4a16(const std::vector<sycl::half>& input,
                    const std::vector<uint8_t>& weight_packed,
                    const std::vector<sycl::half>& scale,
                    std::vector<float>& output,
                    int N, int K) {
    constexpr int BLOCK_SIZE = 128;
    int K_BLOCKS = K / BLOCK_SIZE;

    for (int n = 0; n < N; n++) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            float x = static_cast<float>(input[k]);
            uint8_t packed = weight_packed[n * (K/2) + k/2];
            uint8_t w4 = (k % 2 == 0) ? (packed & 0x0F) : ((packed >> 4) & 0x0F);
            float sc = static_cast<float>(scale[n * K_BLOCKS + k / BLOCK_SIZE]);
            float w = (static_cast<float>(w4) - 8.0f) * sc;
            sum += x * w;
        }
        output[n] = sum;  // FP32 reference
    }
}
```

### W8A16
```cpp
void cpu_gemv_w8a16(const std::vector<sycl::half>& input,
                    const std::vector<int8_t>& weight,
                    const std::vector<sycl::half>& scale,
                    std::vector<float>& output,
                    int N, int K) {
    for (int n = 0; n < N; n++) {
        float sum = 0.0f;
        float sc = static_cast<float>(scale[n]);
        for (int k = 0; k < K; k++) {
            sum += static_cast<float>(input[k])
                 * (static_cast<float>(weight[n*K+k]) * sc);
        }
        output[n] = sum;
    }
}
```

## Verification function

```cpp
bool verify_results(const std::vector<float>& cpu_output,
                    const std::vector<sycl::half>& gpu_output,
                    int N) {
    int errors = 0;
    float max_diff = 0.0f;

    // Check NaN first — NaN comparisons silently pass threshold checks
    for (int i = 0; i < N; i++) {
        if (std::isnan(static_cast<float>(gpu_output[i]))) {
            std::cout << "NaN at index " << i << std::endl;
            errors++;
        }
    }
    if (errors > 0) {
        std::cout << "FAILED: " << errors << " NaN outputs" << std::endl;
        return false;
    }

    for (int i = 0; i < N; i++) {
        float cpu_val = cpu_output[i];
        float gpu_val = static_cast<float>(gpu_output[i]);
        float diff    = std::abs(cpu_val - gpu_val);
        float rel_err = std::abs(cpu_val) > 1e-6f ? diff / std::abs(cpu_val) : 0.0f;

        max_diff = std::max(max_diff, diff);

        // Threshold: absolute > 1.0 AND relative > 2%
        if (diff > 1.0f && rel_err > 0.02f) {
            if (errors < 3)
                std::cout << "  [" << i << "]: CPU=" << cpu_val
                          << " GPU=" << gpu_val << " diff=" << diff << std::endl;
            errors++;
        }
    }

    std::cout << "Verification: " << (errors == 0 ? "PASSED" : "FAILED")
              << " (" << errors << "/" << N << " errors)"
              << " max_diff=" << max_diff << std::endl;
    return errors == 0;
}
```

## Error thresholds

| Source | Expected error | Reason |
|--------|---------------|--------|
| fp16 accumulation vs fp32 | ~0.5–1% rel | fp16 has limited mantissa |
| W4 quantization | ~1–5% rel | 4-bit precision loss |
| W8 quantization | ~0.1–0.5% rel | 8-bit is more precise |

Use `abs_diff > 1.0 && rel_error > 2%` as the failure threshold. A result can have large absolute
difference but small relative error (e.g. large-magnitude outputs) — use both conditions.

## Corner cases to test

1. **Small N** (N=1, N=4): single workgroup, boundary conditions
2. **N not a multiple of ROWS**: last group has fewer rows than ROWS — guard `if (n < N)` required
3. **K not a multiple of VL**: kernel assumes K is VL-aligned; verify assumption or add tail handling
4. **Zero scale**: output should be zero — no NaN or inf
5. **Extreme weights**: all 0x0F nibbles (value 7 after subtracting 8 = -1) — check sign

## Test sizes

- Quick correctness: N=256, K=512 (fast CPU reference, < 1 s)
- Full correctness: N=4096, K=4096 (covers all edge cases, ~10 s CPU)
- Benchmark: N=8192–16384, K=4096–8192

## Correctness vs first weight copy only

When using 32 cache-busting weight copies, always verify against `d_weights[0]` (same seed as CPU reference):
```cpp
// Run verification with copy 0 (matches cpu_output computed from weights[0])
q.submit([&](handler& h) {
    h.parallel_for(nd_range, Kernel{d_input, d_weights[0], d_scale, d_output, N, K});
}).wait();
q.memcpy(gpu_output.data(), d_output, N * sizeof(sycl::half)).wait();
verify_results(cpu_output, gpu_output, N);
```
