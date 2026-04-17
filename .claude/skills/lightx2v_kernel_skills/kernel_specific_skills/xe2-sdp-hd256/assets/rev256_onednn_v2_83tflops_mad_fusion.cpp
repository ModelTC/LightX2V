#include <sycl/sycl.hpp>
#include <sycl/ext/intel/esimd.hpp>
#include <iostream>
#include <chrono>
#include <random>
#include <iomanip>
#include <cmath>
#include <vector>

using namespace sycl;
using fp16 = sycl::half;
using bf16 = sycl::ext::oneapi::bfloat16;

#define __ESIMD_NS sycl::ext::intel::esimd
#define __ESIMD_ENS sycl::ext::intel::experimental::esimd
#undef ESIMD_INLINE
#define ESIMD_INLINE inline __attribute__((always_inline))
#define FP32_MIN -3.402823466e+38f

using namespace sycl::ext::intel::esimd;
using namespace sycl::ext::intel::esimd::xmx;
using namespace sycl::ext::intel::experimental::esimd;

#include "flash.attn.b.mha256.fp16.opt.h"       // baseline
#include "rev256_onednn_v2_83tflops_mad_fusion.h" // v2 under test

// ===== CPU Reference (fp32, non-causal) =====
void cpu_sdp_reference(
    const float* Q, const float* K, const float* V, const float* normAlpha,
    float* out,
    int q_len, int kv_len, int num_heads_q, int num_heads_kv, int head_dim, float scale)
{
    int groupSize = num_heads_q / num_heads_kv;
    for (int q = 0; q < q_len; q++) {
        for (int h = 0; h < num_heads_q; h++) {
            int kv_h = h / groupSize;
            std::vector<float> scores(kv_len);
            for (int k = 0; k < kv_len; k++) {
                float s = 0;
                for (int d = 0; d < head_dim; d++)
                    s += Q[q * num_heads_q * head_dim + h * head_dim + d]
                       * K[k * num_heads_kv * head_dim + kv_h * head_dim + d];
                scores[k] = s * scale;
            }
            float mx = -1e30f;
            for (auto s : scores) mx = std::max(mx, s);
            float sum = 0;
            for (auto& s : scores) { s = std::exp(s - mx); sum += s; }
            if (sum > 0) for (auto& s : scores) s /= sum;
            for (int d = 0; d < head_dim; d++) {
                float acc = 0;
                for (int k = 0; k < kv_len; k++)
                    acc += scores[k] * V[k * num_heads_kv * head_dim + kv_h * head_dim + d];
                out[q * num_heads_q * head_dim + h * head_dim + d] = acc * normAlpha[h * head_dim + d];
            }
        }
    }
}

// ===== Correctness Test =====
bool test_correctness(sycl::queue& q, int q_len, int kv_len, int num_heads_q, int num_heads_kv,
                      float data_range = 0.5f, int seed = 42) {
    constexpr int HD = 256;
    float scale = 1.0f / std::sqrt((float)HD);
    size_t q_size = q_len * num_heads_q * HD;
    size_t kv_size = kv_len * num_heads_kv * HD;

    std::vector<float> h_Q_f32(q_size), h_K_f32(kv_size), h_V_f32(kv_size);
    std::vector<float> h_normAlpha(num_heads_q * HD, 1.0f);
    std::vector<float> h_ref(q_size);

    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dis(-data_range, data_range);
    for (size_t i = 0; i < q_size; i++) h_Q_f32[i] = dis(gen);
    for (size_t i = 0; i < kv_size; i++) h_K_f32[i] = dis(gen);
    for (size_t i = 0; i < kv_size; i++) h_V_f32[i] = dis(gen);

    std::vector<fp16> h_Q(q_size), h_K(kv_size), h_V(kv_size);
    for (size_t i = 0; i < q_size; i++) h_Q[i] = fp16(h_Q_f32[i]);
    for (size_t i = 0; i < kv_size; i++) h_K[i] = fp16(h_K_f32[i]);
    for (size_t i = 0; i < kv_size; i++) h_V[i] = fp16(h_V_f32[i]);

    std::vector<float> h_Q_ref(q_size), h_K_ref(kv_size), h_V_ref(kv_size);
    for (size_t i = 0; i < q_size; i++) h_Q_ref[i] = (float)h_Q[i];
    for (size_t i = 0; i < kv_size; i++) h_K_ref[i] = (float)h_K[i];
    for (size_t i = 0; i < kv_size; i++) h_V_ref[i] = (float)h_V[i];

    cpu_sdp_reference(h_Q_ref.data(), h_K_ref.data(), h_V_ref.data(), h_normAlpha.data(),
                      h_ref.data(), q_len, kv_len, num_heads_q, num_heads_kv, HD, scale);

    fp16* d_Q = sycl::malloc_device<fp16>(q_size, q);
    fp16* d_K = sycl::malloc_device<fp16>(kv_size, q);
    fp16* d_V = sycl::malloc_device<fp16>(kv_size, q);
    fp16* d_O = sycl::malloc_device<fp16>(q_size, q);
    float* d_normAlpha = sycl::malloc_device<float>(num_heads_q * HD, q);

    q.memcpy(d_Q, h_Q.data(), q_size * sizeof(fp16)).wait();
    q.memcpy(d_K, h_K.data(), kv_size * sizeof(fp16)).wait();
    q.memcpy(d_V, h_V.data(), kv_size * sizeof(fp16)).wait();
    q.memcpy(d_normAlpha, h_normAlpha.data(), num_heads_q * HD * sizeof(float)).wait();

    int groupH = num_heads_q;
    int groupV = (q_len + 127) / 128;
    sycl::nd_range<2> nd_range({(size_t)(32 * groupH), (size_t)groupV}, {32, 1});

    q.submit([&](sycl::handler& cgh) {
        cgh.parallel_for(nd_range, [=](sycl::nd_item<2> ndi) SYCL_ESIMD_KERNEL {
            flashAttnBMha256Fp16Rev_onednn_v2(
                reinterpret_cast<uint8_t*>(d_Q), reinterpret_cast<uint8_t*>(d_K),
                reinterpret_cast<uint8_t*>(d_V), reinterpret_cast<uint8_t*>(d_normAlpha),
                reinterpret_cast<uint8_t*>(d_O),
                q_len, kv_len, num_heads_q, num_heads_kv, ndi);
        });
    }).wait();

    std::vector<fp16> h_O(q_size);
    q.memcpy(h_O.data(), d_O, q_size * sizeof(fp16)).wait();

    int nan_count = 0;
    float max_diff = 0;
    double sum_sq_diff = 0, sum_sq_ref = 0;
    for (size_t i = 0; i < q_size; i++) {
        float gpu_val = (float)h_O[i];
        if (std::isnan(gpu_val)) { nan_count++; continue; }
        float ref_val = h_ref[i];
        float diff = std::abs(gpu_val - ref_val);
        max_diff = std::max(max_diff, diff);
        sum_sq_diff += (double)(diff * diff);
        sum_sq_ref += (double)(ref_val * ref_val);
    }

    if (nan_count > 0) {
        std::cout << "FAIL: " << nan_count << " NaN values\n";
        sycl::free(d_Q, q); sycl::free(d_K, q); sycl::free(d_V, q); sycl::free(d_O, q); sycl::free(d_normAlpha, q);
        return false;
    }

    float rel_rms = (sum_sq_ref > 0) ? std::sqrt(sum_sq_diff / sum_sq_ref) : 0;
    bool pass = (max_diff < 0.1f);
    std::cout << (pass ? "PASS" : "FAIL")
              << " | max_diff=" << std::fixed << std::setprecision(4) << max_diff
              << " rel_rms=" << std::scientific << std::setprecision(3) << rel_rms << "\n";

    sycl::free(d_Q, q); sycl::free(d_K, q); sycl::free(d_V, q); sycl::free(d_O, q); sycl::free(d_normAlpha, q);
    return pass;
}

// ===== Benchmark v2 =====
double benchmark_v2(sycl::queue& q, int q_len, int kv_len, int num_heads_q, int num_heads_kv,
                    fp16* d_Q, fp16* d_K, fp16* d_V, fp16* d_O, float* d_normAlpha) {
    int groupH = num_heads_q;
    int groupV = (q_len + 127) / 128;
    sycl::nd_range<2> nd_range({(size_t)(32 * groupH), (size_t)groupV}, {32, 1});

    for (int i = 0; i < 5; i++) {
        q.submit([&](sycl::handler& cgh) {
            cgh.parallel_for(nd_range, [=](sycl::nd_item<2> ndi) SYCL_ESIMD_KERNEL {
                flashAttnBMha256Fp16Rev_onednn_v2(
                    reinterpret_cast<uint8_t*>(d_Q), reinterpret_cast<uint8_t*>(d_K),
                    reinterpret_cast<uint8_t*>(d_V), reinterpret_cast<uint8_t*>(d_normAlpha),
                    reinterpret_cast<uint8_t*>(d_O),
                    q_len, kv_len, num_heads_q, num_heads_kv, ndi);
            });
        }).wait();
    }

    constexpr int NUM_ITERS = 100;
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < NUM_ITERS; i++) {
        q.submit([&](sycl::handler& cgh) {
            cgh.parallel_for(nd_range, [=](sycl::nd_item<2> ndi) SYCL_ESIMD_KERNEL {
                flashAttnBMha256Fp16Rev_onednn_v2(
                    reinterpret_cast<uint8_t*>(d_Q), reinterpret_cast<uint8_t*>(d_K),
                    reinterpret_cast<uint8_t*>(d_V), reinterpret_cast<uint8_t*>(d_normAlpha),
                    reinterpret_cast<uint8_t*>(d_O),
                    q_len, kv_len, num_heads_q, num_heads_kv, ndi);
            });
        }).wait();
    }
    auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(end - start).count() / NUM_ITERS;
}

// ===== Benchmark baseline =====
double benchmark_opt(sycl::queue& q, int q_len, int kv_len, int num_heads_q,
                      fp16* d_Q, fp16* d_K, fp16* d_V, fp16* d_O, float* d_normAlpha) {
    int groupH = num_heads_q;
    int groupV = (q_len + 127) / 128;
    sycl::nd_range<2> nd_range({(size_t)(16 * groupH), (size_t)groupV}, {16, 1});

    for (int i = 0; i < 5; i++) {
        q.submit([&](sycl::handler& cgh) {
            cgh.parallel_for(nd_range, [=](sycl::nd_item<2> ndi) SYCL_ESIMD_KERNEL {
                flashAttnBMha256Fp16OptPrecomputed(
                    reinterpret_cast<uint8_t*>(d_Q), reinterpret_cast<uint8_t*>(d_K),
                    reinterpret_cast<uint8_t*>(d_V), reinterpret_cast<uint8_t*>(d_normAlpha),
                    reinterpret_cast<uint8_t*>(d_O),
                    q_len, kv_len, num_heads_q, num_heads_q, ndi);
            });
        }).wait();
    }

    constexpr int NUM_ITERS = 100;
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < NUM_ITERS; i++) {
        q.submit([&](sycl::handler& cgh) {
            cgh.parallel_for(nd_range, [=](sycl::nd_item<2> ndi) SYCL_ESIMD_KERNEL {
                flashAttnBMha256Fp16OptPrecomputed(
                    reinterpret_cast<uint8_t*>(d_Q), reinterpret_cast<uint8_t*>(d_K),
                    reinterpret_cast<uint8_t*>(d_V), reinterpret_cast<uint8_t*>(d_normAlpha),
                    reinterpret_cast<uint8_t*>(d_O),
                    q_len, kv_len, num_heads_q, num_heads_q, ndi);
            });
        }).wait();
    }
    auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(end - start).count() / NUM_ITERS;
}

void run_perf(sycl::queue& q, int q_len, int kv_len, int num_heads_q, int num_heads_kv, const char* label) {
    constexpr int HD = 256;
    constexpr double PEAK_TFLOPS = 135.0;

    size_t q_size = q_len * num_heads_q * HD;
    size_t kv_size_v2 = kv_len * num_heads_kv * HD;
    size_t kv_size_opt = kv_len * num_heads_q * HD;
    size_t kv_size = std::max(kv_size_v2, kv_size_opt);

    fp16* d_Q = sycl::malloc_device<fp16>(q_size, q);
    fp16* d_K = sycl::malloc_device<fp16>(kv_size, q);
    fp16* d_V = sycl::malloc_device<fp16>(kv_size, q);
    fp16* d_O = sycl::malloc_device<fp16>(q_size, q);
    float* d_normAlpha = sycl::malloc_device<float>(num_heads_q * HD, q);

    std::vector<fp16> h_fp16(std::max(q_size, kv_size));
    std::vector<float> h_alpha(num_heads_q * HD, 1.0f);
    std::mt19937 gen(12345);
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);

    for (size_t i = 0; i < q_size; i++) h_fp16[i] = fp16(dis(gen) * 0.1f);
    q.memcpy(d_Q, h_fp16.data(), q_size * sizeof(fp16)).wait();
    for (size_t i = 0; i < kv_size; i++) h_fp16[i] = fp16(dis(gen) * 0.1f);
    q.memcpy(d_K, h_fp16.data(), kv_size * sizeof(fp16)).wait();
    for (size_t i = 0; i < kv_size; i++) h_fp16[i] = fp16(dis(gen) * 0.1f);
    q.memcpy(d_V, h_fp16.data(), kv_size * sizeof(fp16)).wait();
    q.memcpy(d_normAlpha, h_alpha.data(), num_heads_q * HD * sizeof(float)).wait();

    double flops = 4.0 * q_len * kv_len * HD * num_heads_q + 2.0 * q_len * kv_len * num_heads_q;

    double time_opt = benchmark_opt(q, q_len, kv_len, num_heads_q, d_Q, d_K, d_V, d_O, d_normAlpha);
    double tflops_opt = flops / (time_opt * 1e9);
    double roof_opt = (tflops_opt / PEAK_TFLOPS) * 100.0;

    double time_v2 = benchmark_v2(q, q_len, kv_len, num_heads_q, num_heads_kv, d_Q, d_K, d_V, d_O, d_normAlpha);
    double tflops_v2 = flops / (time_v2 * 1e9);
    double roof_v2 = (tflops_v2 / PEAK_TFLOPS) * 100.0;

    double ratio = time_v2 / time_opt;

    std::cout << std::setw(12) << label
              << " | opt:  " << std::fixed << std::setprecision(3) << std::setw(8) << time_opt << "ms "
              << std::setprecision(1) << std::setw(6) << tflops_opt << " TFLOPS "
              << std::setw(5) << roof_opt << "%"
              << " | v2:   " << std::setprecision(3) << std::setw(8) << time_v2 << "ms "
              << std::setprecision(1) << std::setw(6) << tflops_v2 << " TFLOPS "
              << std::setw(5) << roof_v2 << "%"
              << " | ratio: " << std::setprecision(3) << ratio
              << (ratio < 1.0 ? " FASTER" : (ratio <= 1.01 ? " SAME" : " SLOWER")) << "\n";

    sycl::free(d_Q, q); sycl::free(d_K, q); sycl::free(d_V, q); sycl::free(d_O, q); sycl::free(d_normAlpha, q);
}

int main() {
    constexpr int NUM_Q_HEADS = 32;
    constexpr int NUM_KV_HEADS = 2;

    sycl::queue q(sycl::gpu_selector_v);
    std::cout << "================================================================\n";
    std::cout << "Variant: onednn_v2 (S^T approach, K block load, no hmax)\n";
    std::cout << "================================================================\n";
    std::cout << "Device: " << q.get_device().get_info<sycl::info::device::name>() << "\n";
    std::cout << "Q Heads: " << NUM_Q_HEADS << ", KV Heads: " << NUM_KV_HEADS << ", Head Dim: 256\n\n";

    // ===== Section 1: Correctness =====
    std::cout << "--- Correctness (v2 vs CPU fp32 ref) ---\n";
    int pass = 0, total = 0;
    struct CorrConfig { int q_len; int kv_len; int headQ; int headKv; float range; int seed; const char* name; };
    CorrConfig corr_tests[] = {
        {128,  128,  NUM_Q_HEADS, NUM_KV_HEADS, 0.5f, 42, "128x128 GQA 0.5"},
        {256,  256,  NUM_Q_HEADS, NUM_Q_HEADS,  0.5f, 42, "256x256 MHA 0.5"},
        {128,  512,  NUM_Q_HEADS, NUM_KV_HEADS, 0.5f, 11, "128x512 GQA 0.5"},
        {512,  1024, NUM_Q_HEADS, NUM_KV_HEADS, 0.5f, 42, "512x1K GQA 0.5"},
        {1024, 1024, NUM_Q_HEADS, NUM_KV_HEADS, 0.5f, 42, "1Kx1K GQA 0.5"},
        {512,  1024, NUM_Q_HEADS, NUM_KV_HEADS, 0.1f, 77, "512x1K GQA 0.1"},
        {512,  1024, NUM_Q_HEADS, NUM_KV_HEADS, 2.0f, 99, "512x1K GQA 2.0"},
        {256,  300,  NUM_Q_HEADS, NUM_KV_HEADS, 0.5f, 33, "256x300 GQA 0.5"},
        {256,  48,   NUM_Q_HEADS, NUM_KV_HEADS, 0.5f, 55, "256x48 GQA 0.5"},
    };
    for (auto& t : corr_tests) {
        std::cout << "  " << std::setw(20) << t.name << ": ";
        bool ok = test_correctness(q, t.q_len, t.kv_len, t.headQ, t.headKv, t.range, t.seed);
        if (ok) pass++;
        total++;
    }
    std::cout << "Correctness: " << pass << "/" << total << " PASSED\n\n";

    if (pass < total) {
        std::cout << "Correctness failures detected -- skipping perf tests.\n";
        return 1;
    }

    // ===== Section 2: Performance =====
    std::cout << "--- Performance (opt baseline vs v2, 100 iters) ---\n";
    std::cout << std::string(130, '-') << "\n";
    struct PerfConfig { int q_len; int kv_len; const char* name; };
    PerfConfig perf_tests[] = {
        {8192,  8192,  "8Kx8K"},
        {8192,  16384, "8Kx16K"},
        {16384, 16384, "16Kx16K"},
    };
    for (auto& t : perf_tests) {
        run_perf(q, t.q_len, t.kv_len, NUM_Q_HEADS, NUM_KV_HEADS, t.name);
    }

    std::cout << "\n================================================================\n";
    std::cout << "All tests complete!\n";
    std::cout << "================================================================\n";

    return 0;
}
