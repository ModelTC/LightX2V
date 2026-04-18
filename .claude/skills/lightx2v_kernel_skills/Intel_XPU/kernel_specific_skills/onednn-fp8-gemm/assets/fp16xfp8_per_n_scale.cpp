// =============================================================================
// FP32/BF16/FP16 input × FP8 weight (transposed, per-N scale) → FP32/BF16/FP16 output
// Using OneDNN MatMul on Intel PTL (Panther Lake) GPU
//
// Layout:  C[M,N] = A[M,K] * B^T[K,N]
//          where B is stored as [N,K] (row-major, each row is one output channel)
//          scale[N] = per-output-channel dequant scale
//
// Dequantization: B_dequant[k,n] = scale[n] * B_fp8[n,k]
//
// Supports: --dtype=f16 (default), --dtype=bf16, --dtype=f32
//
// For FP32 I/O: OneDNN internally uses FP16 compute with FP32 accumulation
// =============================================================================

#include <oneapi/dnnl/dnnl.hpp>
#include <oneapi/dnnl/dnnl_sycl.hpp>
#include <sycl/sycl.hpp>

#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <thread>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

using namespace dnnl;
using half = sycl::half;

// ─── BF16 type alias (use uint16_t storage, same as OneDNN) ─────────────────
using bf16_storage = uint16_t;

static bf16_storage fp32_to_bf16(float val) {
    uint32_t bits;
    std::memcpy(&bits, &val, 4);
    // Round to nearest even
    uint32_t rounding = 0x7FFF + ((bits >> 16) & 1);
    bits += rounding;
    return (bf16_storage)(bits >> 16);
}

static float bf16_to_fp32(bf16_storage val) {
    uint32_t bits = (uint32_t)val << 16;
    float result;
    std::memcpy(&result, &bits, 4);
    return result;
}

// ─── FP8 E4M3 conversion ────────────────────────────────────────────────────
static uint8_t fp32_to_fp8_e4m3(float val) {
    const float FP8_MAX = 448.0f;
    val = std::max(-FP8_MAX, std::min(FP8_MAX, val));

    uint32_t bits;
    std::memcpy(&bits, &val, 4);

    uint8_t sign = (bits >> 31) & 1;
    int32_t exp  = ((bits >> 23) & 0xFF) - 127;
    uint32_t frac = bits & 0x7FFFFF;

    if (exp == -127 && frac == 0) return (uint8_t)(sign << 7);
    if (exp < -9) return (uint8_t)(sign << 7);

    int32_t fp8_exp = exp + 7;

    if (fp8_exp >= 15) {
        return (uint8_t)((sign << 7) | 0x7E);
    }

    uint8_t fp8_frac;
    if (fp8_exp <= 0) {
        int shift = 1 - fp8_exp;
        uint32_t full_frac = (1 << 23) | frac;
        full_frac >>= (shift + 20);
        fp8_frac = (uint8_t)(full_frac & 0x7);
        fp8_exp = 0;
    } else {
        fp8_frac = (uint8_t)((frac + (1 << 19)) >> 20);
        if (fp8_frac > 7) { fp8_frac = 0; fp8_exp++; }
        if (fp8_exp >= 15) return (uint8_t)((sign << 7) | 0x7E);
    }

    return (uint8_t)((sign << 7) | (fp8_exp << 3) | fp8_frac);
}

static float fp8_e4m3_to_fp32(uint8_t val) {
    uint8_t sign = (val >> 7) & 1;
    uint8_t exp  = (val >> 3) & 0xF;
    uint8_t frac = val & 0x7;

    float result;
    if (exp == 0 && frac == 0) {
        result = 0.0f;
    } else if (exp == 0) {
        result = std::ldexp((float)frac / 8.0f, -6);
    } else {
        result = std::ldexp(1.0f + (float)frac / 8.0f, (int)exp - 7);
    }

    return sign ? -result : result;
}

// ─── PTL roofline constants ─────────────────────────────────────────────────
constexpr double PTL_FP16_XMX_TFLOPS = 55.0;
constexpr double PTL_DRAM_BW_GBS     = 112.0;

// ─── Random init ────────────────────────────────────────────────────────────
static void rand_init_f32(float* ptr, size_t n, float scale = 0.5f, int seed = 42) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-scale, scale);
    for (size_t i = 0; i < n; i++) {
        float v = dist(rng);
        if (std::abs(v) < 1e-4f) v = (v >= 0.f) ? 1e-4f : -1e-4f;
        ptr[i] = v;
    }
}

// ─── Quantize B[N,K] with per-N scale ───────────────────────────────────────
// B_fp8[n,k] = round(B_fp32[n,k] / scale[n])
// scale[n] = max(|B_fp32[n,:]|) / 448.0
static void quantize_weights_per_n_transposed(const float* B_fp32, uint8_t* B_fp8,
                                               float* scales, int N, int K) {
    for (int n = 0; n < N; n++) {
        // Find absmax for this N row (output channel)
        float amax = 0.0f;
        for (int k = 0; k < K; k++) {
            float v = std::abs(B_fp32[n * K + k]);
            if (v > amax) amax = v;
        }

        const float FP8_MAX = 448.0f;
        float s = (amax > 0.0f) ? (amax / FP8_MAX) : 1.0f;
        scales[n] = s;

        for (int k = 0; k < K; k++) {
            float v = B_fp32[n * K + k] / s;
            B_fp8[n * K + k] = fp32_to_fp8_e4m3(v);
        }
    }
}

// ─── CPU reference: C[M,N] = A[M,K] * B^T[K,N], B stored as [N,K] ───────────
// With per-N scale: B_dequant[n,k] = scale[n] * B_fp8[n,k]
static void cpu_gemm_transposed_ref(const float* A, const uint8_t* B_fp8,
                                     const float* scales, float* C,
                                     int M, int N, int K) {
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            double acc = 0.0;
            for (int k = 0; k < K; k++) {
                float a_val = A[m * K + k];
                // B is [N,K], so B[n,k] = B_fp8[n * K + k]
                float b_val = fp8_e4m3_to_fp32(B_fp8[n * K + k]) * scales[n];
                acc += (double)a_val * (double)b_val;
            }
            C[m * N + n] = (float)acc;
        }
    }
}

// ─── Correctness check ──────────────────────────────────────────────────────
static bool check_correctness(const void* gpu_out, memory::data_type dt,
                              const float* ref_out, int M, int N,
                              double threshold = 0.02) {
    int total = M * N;
    int nan_count = 0;
    double rms = 0.0, ref_rms = 0.0;
    int outlier_count = 0;

    for (int i = 0; i < total; i++) {
        float gpu_val;
        if (dt == memory::data_type::f16) {
            gpu_val = float(((const half*)gpu_out)[i]);
        } else if (dt == memory::data_type::bf16) {
            gpu_val = bf16_to_fp32(((const bf16_storage*)gpu_out)[i]);
        } else {  // f32
            gpu_val = ((const float*)gpu_out)[i];
        }
        float ref_val = ref_out[i];

        if (std::isnan(gpu_val)) { nan_count++; continue; }

        float diff = gpu_val - ref_val;
        rms     += (double)(diff * diff);
        ref_rms += (double)(ref_val * ref_val);
    }

    if (nan_count > 0) {
        printf("  FAIL: %d NaN values in GPU output\n", nan_count);
        return false;
    }

    double rel_rms = std::sqrt(rms / (ref_rms + 1e-30));
    printf("  rel RMS error = %.4f%%\n", rel_rms * 100.0);

    for (int i = 0; i < total; i++) {
        float gpu_val;
        if (dt == memory::data_type::f16) {
            gpu_val = float(((const half*)gpu_out)[i]);
        } else if (dt == memory::data_type::bf16) {
            gpu_val = bf16_to_fp32(((const bf16_storage*)gpu_out)[i]);
        } else {  // f32
            gpu_val = ((const float*)gpu_out)[i];
        }
        float ref_val = ref_out[i];
        if (std::abs(ref_val) < 1e-3f) continue;
        float rel_err = std::abs(gpu_val - ref_val) / std::abs(ref_val);
        if (rel_err > 0.10f) outlier_count++;
    }
    printf("  Outliers (>10%% rel): %d / %d\n", outlier_count, total);

    bool pass = (rel_rms < threshold) && (outlier_count < total / 10);
    printf("  Correctness: %s  (threshold: rel_rms < %.1f%%)\n",
           pass ? "PASS" : "FAIL", threshold * 100.0);
    return pass;
}

// ─── Roofline analysis ──────────────────────────────────────────────────────
static void print_roofline(int M, int N, int K, double time_ms,
                           int elem_bytes) {
    double flops  = 2.0 * M * N * K;
    double tflops = flops / (time_ms * 1e9);

    // Traffic: A[M,K]*elem + B[N,K]*1 (FP8) + scale[N]*4 + C[M,N]*elem
    double bytes_A     = (double)M * K * elem_bytes;
    double bytes_B     = (double)N * K * 1.0;
    double bytes_scale = (double)N * 4.0;
    double bytes_C     = (double)M * N * elem_bytes;
    double bytes       = bytes_A + bytes_B + bytes_scale + bytes_C;
    double bw_gbs      = bytes / (time_ms * 1e6);

    double oi = flops / bytes;
    double bw_bound_tflops = PTL_DRAM_BW_GBS * oi / 1e3;
    double roofline_tflops = std::min(PTL_FP16_XMX_TFLOPS, bw_bound_tflops);

    const char* dtype_str = (elem_bytes == 4) ? "FP32" : "FP16/BF16";

    printf("\n=== Roofline Analysis (PTL, %s × FP8 per-N) ===\n", dtype_str);
    printf("  M=%d  N=%d  K=%d\n", M, N, K);
    printf("  FLOPs           : %.2f GFLOP\n", flops / 1e9);
    printf("  Traffic breakdown:\n");
    printf("    A[M,K]        : %.2f MB (%s)\n", bytes_A / 1e6, dtype_str);
    printf("    B[N,K] FP8    : %.2f MB\n", bytes_B / 1e6);
    printf("    scale[N] FP32 : %.4f MB\n", bytes_scale / 1e6);
    printf("    C[M,N]        : %.2f MB (%s)\n", bytes_C / 1e6, dtype_str);
    printf("    Total         : %.2f MB\n", bytes / 1e6);
    printf("  OI              : %.1f FLOP/byte\n", oi);
    printf("  Time/iter       : %.4f ms\n", time_ms);
    printf("  Throughput      : %.2f TFLOPS\n", tflops);
    printf("  BW achieved     : %.1f GB/s\n", bw_gbs);
    printf("  ----- Roofline -----\n");
    printf("  Compute peak    : %.1f TFLOPS\n", PTL_FP16_XMX_TFLOPS);
    printf("  BW peak         : %.1f GB/s\n", PTL_DRAM_BW_GBS);
    printf("  BW-bound cap    : %.2f TFLOPS\n", bw_bound_tflops);
    printf("  Roofline cap    : %.2f TFLOPS (%s)\n", roofline_tflops,
           roofline_tflops < PTL_FP16_XMX_TFLOPS ? "BW-limited" : "COMPUTE-limited");
    printf("  Efficiency      : %.1f%% of roofline\n",
           tflops / roofline_tflops * 100.0);
    printf("  %% of %.0fT peak  : %.1f%%\n", PTL_FP16_XMX_TFLOPS,
           tflops / PTL_FP16_XMX_TFLOPS * 100.0);
}

// ─── Run matmul with transposed B[N,K] and per-N scale ──────────────────────
static void run_matmul_transposed(
    const char* label,
    engine& eng, stream& strm, sycl::queue& sycl_q,
    memory::data_type io_dt,  // f16 or bf16 for A and C
    int M, int N, int K,
    const void* h_A, const uint8_t* h_B_fp8, const float* h_scales,
    const float* ref_C,
    bool do_verify, bool do_perf)
{
    printf("\n══════════════════════════════════════════════════════════\n");
    printf(" Config: %s\n", label);
    printf("══════════════════════════════════════════════════════════\n");

    // Memory descriptors:
    // A: [M, K] row-major
    // B: logical [K, N] but stored as [N, K] → use format_tag::ba
    // C: [M, N] row-major
    auto a_md = memory::desc({M, K}, io_dt, memory::format_tag::ab);
    auto b_md = memory::desc({K, N}, memory::data_type::f8_e4m3, memory::format_tag::ba);
    auto c_md = memory::desc({M, N}, io_dt, memory::format_tag::ab);

    // Scale: per-N (per output channel, dim 1 of logical B[K,N])
    // mask = 2 (bit 1 set) means per dim-1
    int scale_mask = 2;
    auto scale_md = memory::desc({N}, memory::data_type::f32, memory::format_tag::a);

    // Primitive attributes
    primitive_attr attr;
    attr.set_scales_mask(DNNL_ARG_WEIGHTS, scale_mask);

    // Create primitive
    matmul::primitive_desc matmul_pd;
    try {
        matmul_pd = matmul::primitive_desc(eng, a_md, b_md, c_md, attr);
    } catch (const dnnl::error& e) {
        printf("  UNSUPPORTED: %s (status=%d)\n", e.what(), (int)e.status);
        printf("  This dtype + layout + scale combination not supported.\n");
        return;
    }
    auto matmul_prim = matmul(matmul_pd);

    printf("  Primitive created successfully.\n");
    std::string impl_str = matmul_pd.impl_info_str();
    printf("  Impl: %s\n", impl_str.c_str());

    // Check if fallback to reference implementation
    bool is_ref_impl = (impl_str.find("ref") != std::string::npos);
    if (is_ref_impl && do_perf) {
        printf("\n  WARNING: Fallback to reference implementation detected.\n");
        printf("  Skipping performance benchmark (would be extremely slow).\n");
        printf("  Only FP16 I/O has optimized JIT kernel on PTL.\n");
        do_perf = false;
    }

    int elem_bytes;
    if (io_dt == memory::data_type::f32) {
        elem_bytes = 4;
    } else {  // f16 or bf16
        elem_bytes = 2;
    }

    // Allocate device memory
    size_t a_bytes = (size_t)M * K * elem_bytes;
    size_t b_bytes = (size_t)N * K * 1;  // FP8
    size_t c_bytes = (size_t)M * N * elem_bytes;

    auto* d_A     = sycl::malloc_device(a_bytes, sycl_q);
    auto* d_B     = sycl::malloc_device(b_bytes, sycl_q);
    auto* d_C     = sycl::malloc_device(c_bytes, sycl_q);
    auto* d_scale = sycl::malloc_device<float>(N, sycl_q);

    if (!d_A || !d_B || !d_C || !d_scale) {
        printf("  ERROR: malloc_device failed (OOM?)\n");
        return;
    }

    sycl_q.memcpy(d_A, h_A, a_bytes).wait();
    sycl_q.memcpy(d_B, h_B_fp8, b_bytes).wait();
    sycl_q.memcpy(d_scale, h_scales, N * sizeof(float)).wait();
    sycl_q.memset(d_C, 0, c_bytes).wait();

    auto a_mem = sycl_interop::make_memory(a_md, eng,
                     sycl_interop::memory_kind::usm, d_A);
    auto b_mem = sycl_interop::make_memory(b_md, eng,
                     sycl_interop::memory_kind::usm, d_B);
    auto c_mem = sycl_interop::make_memory(c_md, eng,
                     sycl_interop::memory_kind::usm, d_C);
    auto scale_mem = sycl_interop::make_memory(scale_md, eng,
                         sycl_interop::memory_kind::usm, d_scale);

    std::unordered_map<int, memory> exec_args = {
        {DNNL_ARG_SRC, a_mem},
        {DNNL_ARG_WEIGHTS, b_mem},
        {DNNL_ARG_DST, c_mem},
        {DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS, scale_mem},
    };

    // Correctness
    if (do_verify && ref_C) {
        printf("\n  -- Correctness check --\n");
        matmul_prim.execute(strm, exec_args);
        strm.wait();

        std::vector<char> h_C(c_bytes);
        sycl_q.memcpy(h_C.data(), d_C, c_bytes).wait();
        check_correctness(h_C.data(), io_dt, ref_C, M, N, 0.02);
    }

    // Performance
    if (do_perf) {
        printf("\n  -- Performance benchmark --\n");
        printf("  Sleeping 5 seconds...\n");
        fflush(stdout);
        std::this_thread::sleep_for(std::chrono::seconds(5));

        constexpr int WARMUP = 20;
        for (int i = 0; i < WARMUP; i++)
            matmul_prim.execute(strm, exec_args);
        strm.wait();

        constexpr int NUM_ITERS = 1000;
        auto t0 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < NUM_ITERS; i++)
            matmul_prim.execute(strm, exec_args);
        strm.wait();
        auto t1 = std::chrono::high_resolution_clock::now();

        double total_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        double ms_per_iter = total_ms / NUM_ITERS;

        printf("  Warmup     : %d iters\n", WARMUP);
        printf("  Timed      : %d iters\n", NUM_ITERS);
        printf("  Time/iter  : %.4f ms\n", ms_per_iter);

        print_roofline(M, N, K, ms_per_iter, elem_bytes);
    }

    sycl::free(d_A, sycl_q);
    sycl::free(d_B, sycl_q);
    sycl::free(d_C, sycl_q);
    sycl::free(d_scale, sycl_q);
}

int main(int argc, char** argv) {
    int M = 4096, N = 4096, K = 4096;
    bool run_verify = true;
    bool run_perf   = true;
    memory::data_type io_dt = memory::data_type::f16;  // default FP16

    int pos = 0;
    for (int i = 1; i < argc; i++) {
        std::string arg(argv[i]);
        if (arg == "--no-verify") { run_verify = false; continue; }
        if (arg == "--no-perf")   { run_perf   = false; continue; }
        if (arg == "--dtype=f16" || arg == "--dtype=fp16") {
            io_dt = memory::data_type::f16; continue;
        }
        if (arg == "--dtype=bf16") {
            io_dt = memory::data_type::bf16; continue;
        }
        if (arg == "--dtype=f32" || arg == "--dtype=fp32") {
            io_dt = memory::data_type::f32; continue;
        }
        if (pos == 0) M = std::atoi(argv[i]);
        if (pos == 1) N = std::atoi(argv[i]);
        if (pos == 2) K = std::atoi(argv[i]);
        pos++;
    }

    const char* dtype_name;
    if (io_dt == memory::data_type::f16) dtype_name = "FP16";
    else if (io_dt == memory::data_type::bf16) dtype_name = "BF16";
    else dtype_name = "FP32";

    printf("════════════════════════════════════════════════════════════\n");
    printf(" %s × FP8(e4m3) GEMM — B[N,K] transposed, per-N scale\n", dtype_name);
    printf(" OneDNN / Intel PTL\n");
    printf("════════════════════════════════════════════════════════════\n");
    printf("M=%d  N=%d  K=%d\n", M, N, K);
    printf("A[M,K]=%s  B[N,K]=FP8_E4M3  scale[N]=FP32  C[M,N]=%s\n\n",
           dtype_name, dtype_name);

    try {
        sycl::queue sycl_q(sycl::gpu_selector_v, sycl::property::queue::in_order{});
        printf("Device: %s\n",
               sycl_q.get_device().get_info<sycl::info::device::name>().c_str());

        engine eng = sycl_interop::make_engine(sycl_q.get_device(),
                                                sycl_q.get_context());
        stream strm = sycl_interop::make_stream(eng, sycl_q);

        // ─── Prepare data (use FP32 on host, convert to BF16/FP16 as needed) ───
        std::vector<float> h_A_f32(M * K);
        std::vector<float> h_B_f32(N * K);
        rand_init_f32(h_A_f32.data(), h_A_f32.size(), 0.5f, 42);
        rand_init_f32(h_B_f32.data(), h_B_f32.size(), 0.5f, 137);

        // Quantize B[N,K] with per-N scale
        std::vector<uint8_t> h_B_fp8(N * K);
        std::vector<float>   h_scales(N);
        quantize_weights_per_n_transposed(h_B_f32.data(), h_B_fp8.data(),
                                           h_scales.data(), N, K);

        printf("  Quantization done. Sample scales: [%.4f, %.4f, %.4f, ...]\n",
               h_scales[0], h_scales[1], h_scales[2]);

        // Convert A to target dtype for GPU
        int elem_bytes = (io_dt == memory::data_type::f32) ? 4 : 2;
        std::vector<char> h_A_typed(M * K * elem_bytes);
        if (io_dt == memory::data_type::f16) {
            half* ptr = (half*)h_A_typed.data();
            for (size_t i = 0; i < h_A_f32.size(); i++)
                ptr[i] = half(h_A_f32[i]);
        } else if (io_dt == memory::data_type::bf16) {
            bf16_storage* ptr = (bf16_storage*)h_A_typed.data();
            for (size_t i = 0; i < h_A_f32.size(); i++)
                ptr[i] = fp32_to_bf16(h_A_f32[i]);
        } else {  // f32 - just copy
            std::memcpy(h_A_typed.data(), h_A_f32.data(), M * K * sizeof(float));
        }

        // ─── Correctness ULT (small size) ───────────────────────────────────
        std::vector<float> ref_C;
        if (run_verify) {
            int sM = 256, sN = 256, sK = 256;
            printf("\n── Correctness ULT (M=%d N=%d K=%d) ──\n", sM, sN, sK);

            std::vector<float>   s_A(sM * sK);
            std::vector<float>   s_B(sN * sK);
            std::vector<uint8_t> s_B_fp8(sN * sK);
            std::vector<float>   s_scales(sN);
            ref_C.resize(sM * sN);

            rand_init_f32(s_A.data(), s_A.size(), 0.5f, 42);
            rand_init_f32(s_B.data(), s_B.size(), 0.5f, 137);
            quantize_weights_per_n_transposed(s_B.data(), s_B_fp8.data(),
                                               s_scales.data(), sN, sK);

            printf("  Computing CPU FP32 reference (%s × FP8 dequant)...\n", dtype_name);
            cpu_gemm_transposed_ref(s_A.data(), s_B_fp8.data(), s_scales.data(),
                                     ref_C.data(), sM, sN, sK);

            // Convert small A to target dtype
            int s_elem_bytes = (io_dt == memory::data_type::f32) ? 4 : 2;
            std::vector<char> s_A_typed(sM * sK * s_elem_bytes);
            if (io_dt == memory::data_type::f16) {
                half* ptr = (half*)s_A_typed.data();
                for (size_t i = 0; i < s_A.size(); i++)
                    ptr[i] = half(s_A[i]);
            } else if (io_dt == memory::data_type::bf16) {
                bf16_storage* ptr = (bf16_storage*)s_A_typed.data();
                for (size_t i = 0; i < s_A.size(); i++)
                    ptr[i] = fp32_to_bf16(s_A[i]);
            } else {  // f32 - just copy
                std::memcpy(s_A_typed.data(), s_A.data(), sM * sK * sizeof(float));
            }

            char label[128];
            snprintf(label, sizeof(label), "%s×FP8_E4M3 B[N,K] per-N scale (ULT, %dx%dx%d)",
                     dtype_name, sM, sN, sK);

            run_matmul_transposed(
                label,
                eng, strm, sycl_q,
                io_dt,
                sM, sN, sK,
                s_A_typed.data(), s_B_fp8.data(), s_scales.data(),
                ref_C.data(),
                true, false
            );
        }

        // ─── Full-size performance ──────────────────────────────────────────
        if (run_perf) {
            char label[128];
            snprintf(label, sizeof(label), "%s×FP8_E4M3 B[N,K] per-N scale (PERF, %dx%dx%d)",
                     dtype_name, M, N, K);

            run_matmul_transposed(
                label,
                eng, strm, sycl_q,
                io_dt,
                M, N, K,
                h_A_typed.data(), h_B_fp8.data(), h_scales.data(),
                nullptr,
                false, true
            );
        }

        printf("\nDone.\n");
        return 0;

    } catch (const dnnl::error& e) {
        fprintf(stderr, "OneDNN error: %s (status=%d)\n", e.what(), (int)e.status);
        return 1;
    } catch (const sycl::exception& e) {
        fprintf(stderr, "SYCL error: %s\n", e.what());
        return 1;
    } catch (const std::exception& e) {
        fprintf(stderr, "Error: %s\n", e.what());
        return 1;
    }
}
