// Pure C++ ULT for oneDNN matmul primitives on Intel XPU (SYCL)
//
// Tests:
//   1. FP16 x FP8_E4M3 GEMM with per-N scale
//   2. BF16 x FP8_E4M3 GEMM with per-N scale
//   3. FP16 x U4 (INT4) GEMM with block-wise scale + zero-point
//   4. FP16 x FP8_E4M3 GEMM with bias
//
// Build:
//   icpx test_onednn.cpp -fsycl -o test_onednn.exe
//        -I<torch_include> -L<dnnl_lib> -ldnnl
//
// Run:
//   test_onednn.exe

#include <sycl/sycl.hpp>
#include <oneapi/dnnl/dnnl.hpp>
#include <oneapi/dnnl/dnnl_sycl.hpp>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <vector>
#include <random>
#include <chrono>

// ── FP8 E4M3 conversion helpers ─────────────────────────────────────────────
// E4M3: 1 sign + 4 exponent + 3 mantissa, bias=7, no inf, nan=0x7f
static uint8_t fp32_to_fp8_e4m3(float val) {
    // Clamp to E4M3 range [-448, 448]
    const float FP8_MAX = 448.0f;
    val = std::max(-FP8_MAX, std::min(FP8_MAX, val));

    uint32_t bits;
    std::memcpy(&bits, &val, 4);

    uint8_t sign = (bits >> 31) & 1;
    int32_t exp_f32 = ((bits >> 23) & 0xFF) - 127;  // unbiased
    uint32_t mant_f32 = bits & 0x7FFFFF;

    if (val == 0.0f) return 0;

    // E4M3 bias = 7, exp range [-6, 8] for normal
    int32_t exp_e4m3 = exp_f32 + 7;  // biased

    if (exp_e4m3 <= 0) {
        // Subnormal or zero
        return (sign << 7);
    }
    if (exp_e4m3 >= 15) {
        // Saturate to max (no inf in E4M3)
        return (sign << 7) | 0x7E;  // max normal
    }

    // Round mantissa: 23 bits -> 3 bits
    uint8_t mant_e4m3 = (mant_f32 >> 20) & 0x7;
    // Round to nearest even
    uint32_t remainder = mant_f32 & 0xFFFFF;
    if (remainder > 0x80000 || (remainder == 0x80000 && (mant_e4m3 & 1))) {
        mant_e4m3++;
        if (mant_e4m3 > 7) {
            mant_e4m3 = 0;
            exp_e4m3++;
            if (exp_e4m3 >= 15) return (sign << 7) | 0x7E;
        }
    }

    return (sign << 7) | (exp_e4m3 << 3) | mant_e4m3;
}

static float fp8_e4m3_to_fp32(uint8_t val) {
    if (val == 0 || val == 0x80) return 0.0f;

    uint8_t sign = (val >> 7) & 1;
    uint8_t exp = (val >> 3) & 0xF;
    uint8_t mant = val & 0x7;

    float result;
    if (exp == 0) {
        // Subnormal
        result = std::ldexp((float)mant, -9);  // 2^(-6) * mant/8 = mant * 2^(-9)
    } else {
        result = std::ldexp(1.0f + mant / 8.0f, exp - 7);
    }
    return sign ? -result : result;
}

// ── Helpers ──────────────────────────────────────────────────────────────────

static float rel_rms(const float* a, const float* b, size_t n) {
    double sum_diff2 = 0.0, sum_ref2 = 0.0;
    for (size_t i = 0; i < n; i++) {
        double d = (double)a[i] - (double)b[i];
        sum_diff2 += d * d;
        sum_ref2 += (double)b[i] * (double)b[i];
    }
    return (float)(std::sqrt(sum_diff2 / n) / (std::sqrt(sum_ref2 / n) + 1e-8));
}

static void fill_random(float* buf, size_t n, float lo, float hi, unsigned seed = 42) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(lo, hi);
    for (size_t i = 0; i < n; i++) buf[i] = dist(rng);
}

// CPU reference: C[M,N] = A[M,K] * B_dequant[K,N]
// B stored as [N,K] FP8, logically transposed to [K,N]
static void gemm_ref_fp8(
    const sycl::half* A, const uint8_t* B_fp8, const float* scales,
    float* C_f32, int M, int K, int N)
{
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            double acc = 0.0;
            for (int k = 0; k < K; k++) {
                float a_val = (float)A[m * K + k];
                float b_val = fp8_e4m3_to_fp32(B_fp8[n * K + k]) * scales[n];
                acc += (double)a_val * (double)b_val;
            }
            C_f32[m * N + n] = (float)acc;
        }
    }
}

// CPU reference for INT4: C[M,N] = A[M,K] * dequant(B_u4[K,N])
// B_u4 stored as [N,K/2] packed uint8 (2 values per byte, low nibble first)
// scales: [K/block_size, N] FP16
// zero_point: scalar u8
static void gemm_ref_int4(
    const sycl::half* A, const uint8_t* B_packed, const sycl::half* scales,
    uint8_t zero_point, float* C_f32,
    int M, int K, int N, int block_size)
{
    int n_groups = K / block_size;
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            double acc = 0.0;
            for (int k = 0; k < K; k++) {
                float a_val = (float)A[m * K + k];

                // Unpack u4
                int byte_idx = n * (K / 2) + k / 2;
                uint8_t packed = B_packed[byte_idx];
                uint8_t q_val = (k % 2 == 0) ? (packed & 0xF) : (packed >> 4);

                // Dequant: (q_val - zero_point) * scale
                int group = k / block_size;
                float scale_val = (float)scales[group * N + n];  // scales[group, n]
                float b_val = ((float)q_val - (float)zero_point) * scale_val;
                acc += (double)a_val * (double)b_val;
            }
            C_f32[m * N + n] = (float)acc;
        }
    }
}

// ── Test 1: FP16 x FP8 GEMM ─────────────────────────────────────────────────

bool test_fp16_fp8(sycl::queue& q, int M, int N, int K, const char* label) {
    dnnl::engine eng = dnnl::sycl_interop::make_engine(q.get_device(), q.get_context());
    dnnl::stream s = dnnl::sycl_interop::make_stream(eng, q);

    // Allocate host buffers
    std::vector<float> A_f32(M * K), scales_f32(N);
    fill_random(A_f32.data(), M * K, -1.0f, 1.0f, 42);
    fill_random(scales_f32.data(), N, 0.001f, 0.01f, 123);

    // Create FP8 weights on host
    std::vector<float> B_f32(N * K);
    fill_random(B_f32.data(), N * K, -1.0f, 1.0f, 77);
    std::vector<uint8_t> B_fp8(N * K);
    for (size_t i = 0; i < N * K; i++) {
        B_fp8[i] = fp32_to_fp8_e4m3(B_f32[i] / scales_f32[i / K]);
    }

    // Convert A to FP16
    std::vector<sycl::half> A_fp16(M * K);
    for (size_t i = 0; i < M * K; i++) A_fp16[i] = sycl::half(A_f32[i]);

    // Allocate device buffers
    auto* d_A = sycl::malloc_device<sycl::half>(M * K, q);
    auto* d_B = sycl::malloc_device<uint8_t>(N * K, q);
    auto* d_scales = sycl::malloc_device<float>(N, q);
    auto* d_C = sycl::malloc_device<sycl::half>(M * N, q);

    q.memcpy(d_A, A_fp16.data(), M * K * sizeof(sycl::half));
    q.memcpy(d_B, B_fp8.data(), N * K * sizeof(uint8_t));
    q.memcpy(d_scales, scales_f32.data(), N * sizeof(float));
    q.wait();

    // oneDNN descriptors
    using dt = dnnl::memory::data_type;
    using ft = dnnl::memory::format_tag;

    dnnl::memory::desc a_md({M, K}, dt::f16, ft::ab);
    dnnl::memory::desc b_md({K, N}, dt::f8_e4m3, ft::ba);  // logical [K,N], physical [N,K]
    dnnl::memory::desc scale_md({N}, dt::f32, ft::a);
    dnnl::memory::desc c_md({M, N}, dt::f16, ft::ab);

    dnnl::primitive_attr attr;
    attr.set_scales_mask(DNNL_ARG_WEIGHTS, 2);  // per-N
    attr.set_fpmath_mode(dnnl::fpmath_mode::any, true);

    dnnl::matmul::primitive_desc pd(eng, a_md, b_md, c_md, attr);
    std::string impl = pd.impl_info_str();

    dnnl::matmul prim(pd);

    std::unordered_map<int, dnnl::memory> args = {
        {DNNL_ARG_SRC, dnnl::memory(a_md, eng, d_A)},
        {DNNL_ARG_WEIGHTS, dnnl::memory(b_md, eng, d_B)},
        {DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS, dnnl::memory(scale_md, eng, d_scales)},
        {DNNL_ARG_DST, dnnl::memory(c_md, eng, d_C)},
    };

    prim.execute(s, args);
    s.wait();

    // Copy result back
    std::vector<sycl::half> C_fp16(M * N);
    q.memcpy(C_fp16.data(), d_C, M * N * sizeof(sycl::half)).wait();

    // CPU reference
    std::vector<float> C_ref(M * N);
    gemm_ref_fp8(A_fp16.data(), B_fp8.data(), scales_f32.data(), C_ref.data(), M, K, N);

    // Convert GPU result to f32
    std::vector<float> C_gpu_f32(M * N);
    for (size_t i = 0; i < M * N; i++) C_gpu_f32[i] = (float)C_fp16[i];

    float err = rel_rms(C_gpu_f32.data(), C_ref.data(), M * N);
    bool pass = err < 0.05f;

    printf("  [%s] M=%d N=%d K=%d  impl=%s  rel_rms=%.4f  %s\n",
           label, M, N, K, impl.c_str(), err, pass ? "PASS" : "FAIL");

    // Benchmark
    const int warmup = 10, iters = 10;
    for (int i = 0; i < warmup; i++) prim.execute(s, args);
    s.wait();

    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iters; i++) prim.execute(s, args);
    s.wait();
    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / iters;
    double tflops = 2.0 * M * N * K / (ms / 1e3) / 1e12;
    printf("           perf: %.3f ms  %.2f TFLOPS\n", ms, tflops);

    sycl::free(d_A, q);
    sycl::free(d_B, q);
    sycl::free(d_scales, q);
    sycl::free(d_C, q);

    return pass;
}

// ── Test 2: BF16 x FP8 GEMM ─────────────────────────────────────────────────

bool test_bf16_fp8(sycl::queue& q, int M, int N, int K, const char* label) {
    dnnl::engine eng = dnnl::sycl_interop::make_engine(q.get_device(), q.get_context());
    dnnl::stream s = dnnl::sycl_interop::make_stream(eng, q);

    std::vector<float> A_f32(M * K), scales_f32(N);
    fill_random(A_f32.data(), M * K, -1.0f, 1.0f, 42);
    fill_random(scales_f32.data(), N, 0.001f, 0.01f, 123);

    std::vector<float> B_f32(N * K);
    fill_random(B_f32.data(), N * K, -1.0f, 1.0f, 77);
    std::vector<uint8_t> B_fp8(N * K);
    for (size_t i = 0; i < N * K; i++) {
        B_fp8[i] = fp32_to_fp8_e4m3(B_f32[i] / scales_f32[i / K]);
    }

    // For BF16 reference, we still use FP16 host conversion then truncate
    // (sycl::ext::oneapi::bfloat16 for proper BF16)
    // Use raw uint16_t for BF16 on host
    std::vector<uint16_t> A_bf16(M * K);
    for (size_t i = 0; i < M * K; i++) {
        uint32_t bits;
        std::memcpy(&bits, &A_f32[i], 4);
        A_bf16[i] = (uint16_t)(bits >> 16);  // truncate to BF16
    }

    auto* d_A = sycl::malloc_device<uint16_t>(M * K, q);
    auto* d_B = sycl::malloc_device<uint8_t>(N * K, q);
    auto* d_scales = sycl::malloc_device<float>(N, q);
    auto* d_C = sycl::malloc_device<uint16_t>(M * N, q);

    q.memcpy(d_A, A_bf16.data(), M * K * sizeof(uint16_t));
    q.memcpy(d_B, B_fp8.data(), N * K * sizeof(uint8_t));
    q.memcpy(d_scales, scales_f32.data(), N * sizeof(float));
    q.wait();

    using dt = dnnl::memory::data_type;
    using ft = dnnl::memory::format_tag;

    dnnl::memory::desc a_md({M, K}, dt::bf16, ft::ab);
    dnnl::memory::desc b_md({K, N}, dt::f8_e4m3, ft::ba);
    dnnl::memory::desc scale_md({N}, dt::f32, ft::a);
    dnnl::memory::desc c_md({M, N}, dt::bf16, ft::ab);

    dnnl::primitive_attr attr;
    attr.set_scales_mask(DNNL_ARG_WEIGHTS, 2);
    attr.set_fpmath_mode(dnnl::fpmath_mode::any, true);

    dnnl::matmul::primitive_desc pd(eng, a_md, b_md, c_md, attr);
    std::string impl = pd.impl_info_str();

    dnnl::matmul prim(pd);

    std::unordered_map<int, dnnl::memory> args = {
        {DNNL_ARG_SRC, dnnl::memory(a_md, eng, d_A)},
        {DNNL_ARG_WEIGHTS, dnnl::memory(b_md, eng, d_B)},
        {DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS, dnnl::memory(scale_md, eng, d_scales)},
        {DNNL_ARG_DST, dnnl::memory(c_md, eng, d_C)},
    };

    prim.execute(s, args);
    s.wait();

    // Copy back
    std::vector<uint16_t> C_bf16(M * N);
    q.memcpy(C_bf16.data(), d_C, M * N * sizeof(uint16_t)).wait();

    // CPU ref using fp16 (good enough for rel_rms check)
    std::vector<sycl::half> A_fp16(M * K);
    for (size_t i = 0; i < M * K; i++) A_fp16[i] = sycl::half(A_f32[i]);
    std::vector<float> C_ref(M * N);
    gemm_ref_fp8(A_fp16.data(), B_fp8.data(), scales_f32.data(), C_ref.data(), M, K, N);

    // Convert BF16 result to f32
    std::vector<float> C_gpu_f32(M * N);
    for (size_t i = 0; i < M * N; i++) {
        uint32_t bits = (uint32_t)C_bf16[i] << 16;
        std::memcpy(&C_gpu_f32[i], &bits, 4);
    }

    float err = rel_rms(C_gpu_f32.data(), C_ref.data(), M * N);
    bool pass = err < 0.05f;

    printf("  [%s] M=%d N=%d K=%d  impl=%s  rel_rms=%.4f  %s\n",
           label, M, N, K, impl.c_str(), err, pass ? "PASS" : "FAIL");

    // Benchmark
    const int warmup = 10, iters_bench = 10;
    for (int i = 0; i < warmup; i++) prim.execute(s, args);
    s.wait();

    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iters_bench; i++) prim.execute(s, args);
    s.wait();
    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / iters_bench;
    double tflops = 2.0 * M * N * K / (ms / 1e3) / 1e12;
    printf("           perf: %.3f ms  %.2f TFLOPS\n", ms, tflops);

    sycl::free(d_A, q);
    sycl::free(d_B, q);
    sycl::free(d_scales, q);
    sycl::free(d_C, q);

    return pass;
}

// ── Test 3: FP16 x INT4 (U4) GEMM ───────────────────────────────────────────

bool test_fp16_int4(sycl::queue& q, int M, int N, int K, int block_size, const char* label) {
    dnnl::engine eng = dnnl::sycl_interop::make_engine(q.get_device(), q.get_context());
    dnnl::stream s = dnnl::sycl_interop::make_stream(eng, q);

    int n_groups = K / block_size;

    // Host data
    std::vector<float> A_f32(M * K);
    fill_random(A_f32.data(), M * K, -1.0f, 1.0f, 42);

    std::vector<sycl::half> A_fp16(M * K);
    for (size_t i = 0; i < M * K; i++) A_fp16[i] = sycl::half(A_f32[i]);

    // Generate FP32 weights, quantize to U4
    std::vector<float> W_f32(N * K);
    fill_random(W_f32.data(), N * K, -1.0f, 1.0f, 77);

    // Quantize per-group: group along K dim
    // scales: [n_groups, N], zero_point: scalar u8 = 8
    uint8_t zero_point = 8;
    std::vector<sycl::half> scales_fp16(n_groups * N);
    std::vector<uint8_t> W_packed(N * K / 2);  // packed U4

    for (int n = 0; n < N; n++) {
        for (int g = 0; g < n_groups; g++) {
            // Find max in group
            float max_abs = 0.0f;
            int k_start = g * block_size;
            for (int k = k_start; k < k_start + block_size; k++) {
                float val = std::abs(W_f32[n * K + k]);
                if (val > max_abs) max_abs = val;
            }
            float scale = max_abs / 7.0f;  // map [-max, max] to [1, 15] with zp=8
            if (scale < 1e-12f) scale = 1e-12f;
            scales_fp16[g * N + n] = sycl::half(scale);

            // Quantize
            for (int k = k_start; k < k_start + block_size; k++) {
                float q_val = std::round(W_f32[n * K + k] / scale + (float)zero_point);
                q_val = std::max(0.0f, std::min(15.0f, q_val));
                uint8_t q = (uint8_t)q_val;

                int byte_idx = n * (K / 2) + k / 2;
                if (k % 2 == 0)
                    W_packed[byte_idx] = q;
                else
                    W_packed[byte_idx] |= (q << 4);
            }
        }
    }

    // Device buffers
    auto* d_A = sycl::malloc_device<sycl::half>(M * K, q);
    auto* d_W = sycl::malloc_device<uint8_t>(N * K / 2, q);
    auto* d_scales = sycl::malloc_device<sycl::half>(n_groups * N, q);
    auto* d_zp = sycl::malloc_device<uint8_t>(1, q);
    auto* d_C = sycl::malloc_device<sycl::half>(M * N, q);

    q.memcpy(d_A, A_fp16.data(), M * K * sizeof(sycl::half));
    q.memcpy(d_W, W_packed.data(), N * K / 2);
    q.memcpy(d_scales, scales_fp16.data(), n_groups * N * sizeof(sycl::half));
    q.memcpy(d_zp, &zero_point, 1);
    q.wait();

    using dt = dnnl::memory::data_type;
    using ft = dnnl::memory::format_tag;

    dnnl::memory::desc a_md({M, K}, dt::f16, ft::ab);
    dnnl::memory::desc w_md({K, N}, dt::u4, ft::ba);  // logical [K,N], physical [N,K]
    dnnl::memory::desc scale_md({n_groups, N}, dt::f16, ft::ba);
    dnnl::memory::desc zp_md({1}, dt::u8, ft::a);
    dnnl::memory::desc c_md({M, N}, dt::f16, ft::ab);

    dnnl::primitive_attr attr;
    // Block-wise scales: mask = (1<<0)|(1<<1) = 3, with group {block_size, 1}
    attr.set_scales(DNNL_ARG_WEIGHTS, (1 << 1) + (1 << 0), {block_size, 1}, dt::f16);
    attr.set_zero_points(DNNL_ARG_WEIGHTS, 0, {}, dt::u8);
    attr.set_fpmath_mode(dnnl::fpmath_mode::any, true);

    dnnl::matmul::primitive_desc pd(eng, a_md, w_md, c_md, attr);
    std::string impl = pd.impl_info_str();

    dnnl::matmul prim(pd);

    std::unordered_map<int, dnnl::memory> args = {
        {DNNL_ARG_SRC, dnnl::memory(a_md, eng, d_A)},
        {DNNL_ARG_WEIGHTS, dnnl::memory(w_md, eng, d_W)},
        {DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS, dnnl::memory(scale_md, eng, d_scales)},
        {DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_WEIGHTS, dnnl::memory(zp_md, eng, d_zp)},
        {DNNL_ARG_DST, dnnl::memory(c_md, eng, d_C)},
    };

    prim.execute(s, args);
    s.wait();

    // Copy back
    std::vector<sycl::half> C_fp16(M * N);
    q.memcpy(C_fp16.data(), d_C, M * N * sizeof(sycl::half)).wait();

    // CPU reference
    std::vector<float> C_ref(M * N);
    gemm_ref_int4(A_fp16.data(), W_packed.data(), scales_fp16.data(),
                  zero_point, C_ref.data(), M, K, N, block_size);

    std::vector<float> C_gpu_f32(M * N);
    for (size_t i = 0; i < M * N; i++) C_gpu_f32[i] = (float)C_fp16[i];

    float err = rel_rms(C_gpu_f32.data(), C_ref.data(), M * N);
    bool pass = err < 0.10f;  // INT4 quantization has higher error

    printf("  [%s] M=%d N=%d K=%d blk=%d  impl=%s  rel_rms=%.4f  %s\n",
           label, M, N, K, block_size, impl.c_str(), err, pass ? "PASS" : "FAIL");

    // Benchmark
    const int warmup = 10, iters_bench = 10;
    for (int i = 0; i < warmup; i++) prim.execute(s, args);
    s.wait();

    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iters_bench; i++) prim.execute(s, args);
    s.wait();
    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / iters_bench;
    double tflops = 2.0 * M * N * K / (ms / 1e3) / 1e12;
    printf("           perf: %.3f ms  %.2f TFLOPS\n", ms, tflops);

    sycl::free(d_A, q);
    sycl::free(d_W, q);
    sycl::free(d_scales, q);
    sycl::free(d_zp, q);
    sycl::free(d_C, q);

    return pass;
}

// ── Test 4: FP16 x FP8 with bias ────────────────────────────────────────────

bool test_fp16_fp8_bias(sycl::queue& q, int M, int N, int K, const char* label) {
    dnnl::engine eng = dnnl::sycl_interop::make_engine(q.get_device(), q.get_context());
    dnnl::stream s = dnnl::sycl_interop::make_stream(eng, q);

    std::vector<float> A_f32(M * K), scales_f32(N), bias_f32(N);
    fill_random(A_f32.data(), M * K, -1.0f, 1.0f, 42);
    fill_random(scales_f32.data(), N, 0.001f, 0.01f, 123);
    fill_random(bias_f32.data(), N, -0.5f, 0.5f, 99);

    std::vector<float> B_f32(N * K);
    fill_random(B_f32.data(), N * K, -1.0f, 1.0f, 77);
    std::vector<uint8_t> B_fp8(N * K);
    for (size_t i = 0; i < N * K; i++) {
        B_fp8[i] = fp32_to_fp8_e4m3(B_f32[i] / scales_f32[i / K]);
    }

    std::vector<sycl::half> A_fp16(M * K), bias_fp16(N);
    for (size_t i = 0; i < M * K; i++) A_fp16[i] = sycl::half(A_f32[i]);
    for (size_t i = 0; i < N; i++) bias_fp16[i] = sycl::half(bias_f32[i]);

    auto* d_A = sycl::malloc_device<sycl::half>(M * K, q);
    auto* d_B = sycl::malloc_device<uint8_t>(N * K, q);
    auto* d_scales = sycl::malloc_device<float>(N, q);
    auto* d_bias = sycl::malloc_device<sycl::half>(N, q);
    auto* d_C = sycl::malloc_device<sycl::half>(M * N, q);

    q.memcpy(d_A, A_fp16.data(), M * K * sizeof(sycl::half));
    q.memcpy(d_B, B_fp8.data(), N * K);
    q.memcpy(d_scales, scales_f32.data(), N * sizeof(float));
    q.memcpy(d_bias, bias_fp16.data(), N * sizeof(sycl::half));
    q.wait();

    using dt = dnnl::memory::data_type;
    using ft = dnnl::memory::format_tag;

    dnnl::memory::desc a_md({M, K}, dt::f16, ft::ab);
    dnnl::memory::desc b_md({K, N}, dt::f8_e4m3, ft::ba);
    dnnl::memory::desc scale_md({N}, dt::f32, ft::a);
    dnnl::memory::desc bias_md({N}, dt::f16, ft::a);
    dnnl::memory::desc c_md({M, N}, dt::f16, ft::ab);

    dnnl::primitive_attr attr;
    attr.set_scales_mask(DNNL_ARG_WEIGHTS, 2);
    attr.set_fpmath_mode(dnnl::fpmath_mode::any, true);

    dnnl::matmul::primitive_desc pd(eng, a_md, b_md, c_md, attr);
    dnnl::matmul prim(pd);

    std::unordered_map<int, dnnl::memory> args = {
        {DNNL_ARG_SRC, dnnl::memory(a_md, eng, d_A)},
        {DNNL_ARG_WEIGHTS, dnnl::memory(b_md, eng, d_B)},
        {DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS, dnnl::memory(scale_md, eng, d_scales)},
        {DNNL_ARG_BIAS, dnnl::memory(bias_md, eng, d_bias)},
        {DNNL_ARG_DST, dnnl::memory(c_md, eng, d_C)},
    };

    prim.execute(s, args);
    s.wait();

    std::vector<sycl::half> C_fp16(M * N);
    q.memcpy(C_fp16.data(), d_C, M * N * sizeof(sycl::half)).wait();

    // CPU ref: GEMM + bias
    std::vector<float> C_ref(M * N);
    gemm_ref_fp8(A_fp16.data(), B_fp8.data(), scales_f32.data(), C_ref.data(), M, K, N);
    for (int m = 0; m < M; m++)
        for (int n = 0; n < N; n++)
            C_ref[m * N + n] += bias_f32[n];

    std::vector<float> C_gpu_f32(M * N);
    for (size_t i = 0; i < M * N; i++) C_gpu_f32[i] = (float)C_fp16[i];

    float err = rel_rms(C_gpu_f32.data(), C_ref.data(), M * N);
    bool pass = err < 0.05f;

    printf("  [%s] M=%d N=%d K=%d  rel_rms=%.4f  %s\n",
           label, M, N, K, err, pass ? "PASS" : "FAIL");

    sycl::free(d_A, q);
    sycl::free(d_B, q);
    sycl::free(d_scales, q);
    sycl::free(d_bias, q);
    sycl::free(d_C, q);

    return pass;
}

// ── main ─────────────────────────────────────────────────────────────────────

int main() {
    sycl::queue q(sycl::gpu_selector_v);
    printf("Device: %s\n\n", q.get_device().get_info<sycl::info::device::name>().c_str());

    bool all_pass = true;

    printf("======================================================================\n");
    printf("Test 1: FP16 x FP8_E4M3  (per-N scale)\n");
    printf("======================================================================\n");
    all_pass &= test_fp16_fp8(q, 512, 4096, 4096, "attn");
    all_pass &= test_fp16_fp8(q, 512, 10240, 4096, "FFN up");
    all_pass &= test_fp16_fp8(q, 512, 4096, 10240, "FFN down");

    printf("\n======================================================================\n");
    printf("Test 2: BF16 x FP8_E4M3  (per-N scale)\n");
    printf("======================================================================\n");
    all_pass &= test_bf16_fp8(q, 512, 4096, 4096, "attn");
    all_pass &= test_bf16_fp8(q, 512, 10240, 4096, "FFN up");

    printf("\n======================================================================\n");
    printf("Test 3: FP16 x INT4 (U4)  (block-wise scale + zero-point)\n");
    printf("======================================================================\n");
    all_pass &= test_fp16_int4(q, 1, 4096, 2048, 64, "GEMV");
    all_pass &= test_fp16_int4(q, 512, 4096, 4096, 64, "attn");

    printf("\n======================================================================\n");
    printf("Test 4: FP16 x FP8 + bias\n");
    printf("======================================================================\n");
    all_pass &= test_fp16_fp8_bias(q, 512, 4096, 4096, "attn+bias");

    printf("\n%s\n", all_pass ? "ALL PASS" : "*** SOME TESTS FAILED ***");
    return all_pass ? 0 : 1;
}
