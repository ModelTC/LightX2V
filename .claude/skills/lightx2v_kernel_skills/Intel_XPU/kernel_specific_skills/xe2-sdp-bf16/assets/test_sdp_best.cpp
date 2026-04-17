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

// Best 3 kernels
#include "flash.attn.b.mha128.fp16.opt.h"   // fp16 optimized (barrier+interleave)
#include "flash.attn.b.mha128.bf16.h"        // bf16 (barrier+interleave, bf16 DPAS)
#include "flash.attn.b.mha128.bf16io.h"      // bf16io (hybrid bf16 QK + fp16 SxV, V conv in SxV)

// ===== CPU Reference (fp32, non-causal) =====
void cpu_sdp_reference(
    const float* Q, const float* K, const float* V, const float* normAlpha,
    float* out,
    int q_len, int kv_len, int num_heads, int head_dim, float scale)
{
    for (int q = 0; q < q_len; q++) {
        for (int h = 0; h < num_heads; h++) {
            std::vector<float> scores(kv_len);
            for (int k = 0; k < kv_len; k++) {
                float s = 0;
                for (int d = 0; d < head_dim; d++)
                    s += Q[q * num_heads * head_dim + h * head_dim + d]
                       * K[k * num_heads * head_dim + h * head_dim + d];
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
                    acc += scores[k] * V[k * num_heads * head_dim + h * head_dim + d];
                out[q * num_heads * head_dim + h * head_dim + d] = acc * normAlpha[h * head_dim + d];
            }
        }
    }
}

// ===== Correctness: test one kernel, return pass/fail =====
struct CorrResult { bool pass; float max_diff; float rel_rms; int nan_count; };

CorrResult check_output(const std::vector<float>& h_ref, const float* gpu_f32, size_t n, float thresh) {
    CorrResult r{true, 0, 0, 0};
    double sum_sq_diff = 0, sum_sq_ref = 0;
    for (size_t i = 0; i < n; i++) {
        float gv = gpu_f32[i];
        if (std::isnan(gv)) { r.nan_count++; continue; }
        float rv = h_ref[i];
        float d = std::abs(gv - rv);
        r.max_diff = std::max(r.max_diff, d);
        sum_sq_diff += (double)(d * d);
        sum_sq_ref += (double)(rv * rv);
    }
    r.rel_rms = (sum_sq_ref > 0) ? (float)std::sqrt(sum_sq_diff / sum_sq_ref) : 0;
    r.pass = (r.nan_count == 0) && (r.max_diff < thresh);
    return r;
}

void test_correctness_all(sycl::queue& q, int q_len, int kv_len, int num_heads, const char* label) {
    constexpr int HD = 128;
    float scale = 1.0f / std::sqrt((float)HD);
    size_t q_size = q_len * num_heads * HD;
    size_t kv_size = kv_len * num_heads * HD;

    // Generate fp32 random data
    std::vector<float> h_Q_f32(q_size), h_K_f32(kv_size), h_V_f32(kv_size);
    std::vector<float> h_normAlpha(num_heads * HD, 1.0f);
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dis(-0.5f, 0.5f);
    for (size_t i = 0; i < q_size; i++) h_Q_f32[i] = dis(gen);
    for (size_t i = 0; i < kv_size; i++) h_K_f32[i] = dis(gen);
    for (size_t i = 0; i < kv_size; i++) h_V_f32[i] = dis(gen);

    // fp16 quantized
    std::vector<fp16> h_Q_fp16(q_size), h_K_fp16(kv_size), h_V_fp16(kv_size);
    for (size_t i = 0; i < q_size; i++) h_Q_fp16[i] = fp16(h_Q_f32[i]);
    for (size_t i = 0; i < kv_size; i++) h_K_fp16[i] = fp16(h_K_f32[i]);
    for (size_t i = 0; i < kv_size; i++) h_V_fp16[i] = fp16(h_V_f32[i]);

    // bf16 quantized
    std::vector<bf16> h_Q_bf16(q_size), h_K_bf16(kv_size), h_V_bf16(kv_size);
    for (size_t i = 0; i < q_size; i++) h_Q_bf16[i] = bf16(h_Q_f32[i]);
    for (size_t i = 0; i < kv_size; i++) h_K_bf16[i] = bf16(h_K_f32[i]);
    for (size_t i = 0; i < kv_size; i++) h_V_bf16[i] = bf16(h_V_f32[i]);

    // CPU references (using quantized values matching each kernel's input type)
    std::vector<float> h_ref_fp16(q_size), h_ref_bf16(q_size);
    {
        std::vector<float> qr(q_size), kr(kv_size), vr(kv_size);
        for (size_t i = 0; i < q_size; i++) qr[i] = (float)h_Q_fp16[i];
        for (size_t i = 0; i < kv_size; i++) kr[i] = (float)h_K_fp16[i];
        for (size_t i = 0; i < kv_size; i++) vr[i] = (float)h_V_fp16[i];
        cpu_sdp_reference(qr.data(), kr.data(), vr.data(), h_normAlpha.data(),
                          h_ref_fp16.data(), q_len, kv_len, num_heads, HD, scale);
    }
    {
        std::vector<float> qr(q_size), kr(kv_size), vr(kv_size);
        for (size_t i = 0; i < q_size; i++) qr[i] = (float)h_Q_bf16[i];
        for (size_t i = 0; i < kv_size; i++) kr[i] = (float)h_K_bf16[i];
        for (size_t i = 0; i < kv_size; i++) vr[i] = (float)h_V_bf16[i];
        cpu_sdp_reference(qr.data(), kr.data(), vr.data(), h_normAlpha.data(),
                          h_ref_bf16.data(), q_len, kv_len, num_heads, HD, scale);
    }

    // Device buffers
    fp16* d_Q_fp16 = sycl::malloc_device<fp16>(q_size, q);
    fp16* d_K_fp16 = sycl::malloc_device<fp16>(kv_size, q);
    fp16* d_V_fp16 = sycl::malloc_device<fp16>(kv_size, q);
    fp16* d_O_fp16 = sycl::malloc_device<fp16>(q_size, q);

    bf16* d_Q_bf16 = sycl::malloc_device<bf16>(q_size, q);
    bf16* d_K_bf16 = sycl::malloc_device<bf16>(kv_size, q);
    bf16* d_V_bf16 = sycl::malloc_device<bf16>(kv_size, q);
    bf16* d_O_bf16 = sycl::malloc_device<bf16>(q_size, q);

    float* d_normAlpha = sycl::malloc_device<float>(num_heads * HD, q);

    q.memcpy(d_Q_fp16, h_Q_fp16.data(), q_size * sizeof(fp16)).wait();
    q.memcpy(d_K_fp16, h_K_fp16.data(), kv_size * sizeof(fp16)).wait();
    q.memcpy(d_V_fp16, h_V_fp16.data(), kv_size * sizeof(fp16)).wait();
    q.memcpy(d_Q_bf16, h_Q_bf16.data(), q_size * sizeof(bf16)).wait();
    q.memcpy(d_K_bf16, h_K_bf16.data(), kv_size * sizeof(bf16)).wait();
    q.memcpy(d_V_bf16, h_V_bf16.data(), kv_size * sizeof(bf16)).wait();
    q.memcpy(d_normAlpha, h_normAlpha.data(), num_heads * HD * sizeof(float)).wait();

    int groupH = num_heads;
    int groupV = (q_len + 255) / 256;
    sycl::nd_range<2> nd_range({(size_t)(16 * groupH), (size_t)groupV}, {16, 1});

    std::cout << "  " << std::setw(12) << label << ": ";

    // --- fp16 opt ---
    q.submit([&](sycl::handler& cgh) {
        cgh.parallel_for(nd_range, [=](sycl::nd_item<2> ndi) SYCL_ESIMD_KERNEL {
            flashAttnBMha128Fp16OptPrecomputed(
                reinterpret_cast<uint8_t*>(d_Q_fp16), reinterpret_cast<uint8_t*>(d_K_fp16),
                reinterpret_cast<uint8_t*>(d_V_fp16), reinterpret_cast<uint8_t*>(d_normAlpha),
                reinterpret_cast<uint8_t*>(d_O_fp16),
                q_len, kv_len, num_heads, num_heads, ndi);
        });
    }).wait();
    {
        std::vector<fp16> h_O(q_size);
        std::vector<float> h_O_f32(q_size);
        q.memcpy(h_O.data(), d_O_fp16, q_size * sizeof(fp16)).wait();
        for (size_t i = 0; i < q_size; i++) h_O_f32[i] = (float)h_O[i];
        auto r = check_output(h_ref_fp16, h_O_f32.data(), q_size, 0.05f);
        std::cout << (r.pass ? "PASS" : "FAIL") << "(fp16  ) md=" << std::fixed << std::setprecision(4) << r.max_diff
                  << " rms=" << std::scientific << std::setprecision(2) << r.rel_rms;
        if (r.nan_count > 0) std::cout << " NaN=" << r.nan_count;
    }

    // --- bf16 ---
    q.submit([&](sycl::handler& cgh) {
        cgh.parallel_for(nd_range, [=](sycl::nd_item<2> ndi) SYCL_ESIMD_KERNEL {
            flashAttnBMha128Bf16Precomputed(
                reinterpret_cast<uint8_t*>(d_Q_bf16), reinterpret_cast<uint8_t*>(d_K_bf16),
                reinterpret_cast<uint8_t*>(d_V_bf16), reinterpret_cast<uint8_t*>(d_normAlpha),
                reinterpret_cast<uint8_t*>(d_O_bf16),
                q_len, kv_len, num_heads, num_heads, ndi);
        });
    }).wait();
    {
        std::vector<bf16> h_O(q_size);
        std::vector<float> h_O_f32(q_size);
        q.memcpy(h_O.data(), d_O_bf16, q_size * sizeof(bf16)).wait();
        for (size_t i = 0; i < q_size; i++) h_O_f32[i] = (float)h_O[i];
        auto r = check_output(h_ref_bf16, h_O_f32.data(), q_size, 0.15f);
        std::cout << " | " << (r.pass ? "PASS" : "FAIL") << "(bf16  ) md=" << std::fixed << std::setprecision(4) << r.max_diff
                  << " rms=" << std::scientific << std::setprecision(2) << r.rel_rms;
        if (r.nan_count > 0) std::cout << " NaN=" << r.nan_count;
    }

    // --- bf16io ---
    q.submit([&](sycl::handler& cgh) {
        cgh.parallel_for(nd_range, [=](sycl::nd_item<2> ndi) SYCL_ESIMD_KERNEL {
            flashAttnBMha128Bf16IoPrecomputed(
                reinterpret_cast<uint8_t*>(d_Q_bf16), reinterpret_cast<uint8_t*>(d_K_bf16),
                reinterpret_cast<uint8_t*>(d_V_bf16), reinterpret_cast<uint8_t*>(d_normAlpha),
                reinterpret_cast<uint8_t*>(d_O_bf16),
                q_len, kv_len, num_heads, num_heads, ndi);
        });
    }).wait();
    {
        std::vector<bf16> h_O(q_size);
        std::vector<float> h_O_f32(q_size);
        q.memcpy(h_O.data(), d_O_bf16, q_size * sizeof(bf16)).wait();
        for (size_t i = 0; i < q_size; i++) h_O_f32[i] = (float)h_O[i];
        auto r = check_output(h_ref_bf16, h_O_f32.data(), q_size, 0.15f);
        std::cout << " | " << (r.pass ? "PASS" : "FAIL") << "(bf16io) md=" << std::fixed << std::setprecision(4) << r.max_diff
                  << " rms=" << std::scientific << std::setprecision(2) << r.rel_rms;
        if (r.nan_count > 0) std::cout << " NaN=" << r.nan_count;
    }
    std::cout << "\n";

    sycl::free(d_Q_fp16, q); sycl::free(d_K_fp16, q); sycl::free(d_V_fp16, q); sycl::free(d_O_fp16, q);
    sycl::free(d_Q_bf16, q); sycl::free(d_K_bf16, q); sycl::free(d_V_bf16, q); sycl::free(d_O_bf16, q);
    sycl::free(d_normAlpha, q);
}

// ===== Benchmark helpers =====
double bench_fp16(sycl::queue& q, int q_len, int kv_len, int num_heads,
                  fp16* d_Q, fp16* d_K, fp16* d_V, fp16* d_O, float* d_nA) {
    int gH = num_heads, gV = (q_len + 255) / 256;
    sycl::nd_range<2> nr({(size_t)(16 * gH), (size_t)gV}, {16, 1});
    for (int i = 0; i < 5; i++)
        q.submit([&](sycl::handler& c) { c.parallel_for(nr, [=](sycl::nd_item<2> n) SYCL_ESIMD_KERNEL {
            flashAttnBMha128Fp16OptPrecomputed(reinterpret_cast<uint8_t*>(d_Q), reinterpret_cast<uint8_t*>(d_K),
                reinterpret_cast<uint8_t*>(d_V), reinterpret_cast<uint8_t*>(d_nA),
                reinterpret_cast<uint8_t*>(d_O), q_len, kv_len, num_heads, num_heads, n);
        }); }).wait();
    constexpr int N = 100;
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < N; i++)
        q.submit([&](sycl::handler& c) { c.parallel_for(nr, [=](sycl::nd_item<2> n) SYCL_ESIMD_KERNEL {
            flashAttnBMha128Fp16OptPrecomputed(reinterpret_cast<uint8_t*>(d_Q), reinterpret_cast<uint8_t*>(d_K),
                reinterpret_cast<uint8_t*>(d_V), reinterpret_cast<uint8_t*>(d_nA),
                reinterpret_cast<uint8_t*>(d_O), q_len, kv_len, num_heads, num_heads, n);
        }); }).wait();
    auto t1 = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(t1 - t0).count() / N;
}

double bench_bf16(sycl::queue& q, int q_len, int kv_len, int num_heads,
                  bf16* d_Q, bf16* d_K, bf16* d_V, bf16* d_O, float* d_nA) {
    int gH = num_heads, gV = (q_len + 255) / 256;
    sycl::nd_range<2> nr({(size_t)(16 * gH), (size_t)gV}, {16, 1});
    for (int i = 0; i < 5; i++)
        q.submit([&](sycl::handler& c) { c.parallel_for(nr, [=](sycl::nd_item<2> n) SYCL_ESIMD_KERNEL {
            flashAttnBMha128Bf16Precomputed(reinterpret_cast<uint8_t*>(d_Q), reinterpret_cast<uint8_t*>(d_K),
                reinterpret_cast<uint8_t*>(d_V), reinterpret_cast<uint8_t*>(d_nA),
                reinterpret_cast<uint8_t*>(d_O), q_len, kv_len, num_heads, num_heads, n);
        }); }).wait();
    constexpr int N = 100;
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < N; i++)
        q.submit([&](sycl::handler& c) { c.parallel_for(nr, [=](sycl::nd_item<2> n) SYCL_ESIMD_KERNEL {
            flashAttnBMha128Bf16Precomputed(reinterpret_cast<uint8_t*>(d_Q), reinterpret_cast<uint8_t*>(d_K),
                reinterpret_cast<uint8_t*>(d_V), reinterpret_cast<uint8_t*>(d_nA),
                reinterpret_cast<uint8_t*>(d_O), q_len, kv_len, num_heads, num_heads, n);
        }); }).wait();
    auto t1 = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(t1 - t0).count() / N;
}

double bench_bf16io(sycl::queue& q, int q_len, int kv_len, int num_heads,
                    bf16* d_Q, bf16* d_K, bf16* d_V, bf16* d_O, float* d_nA) {
    int gH = num_heads, gV = (q_len + 255) / 256;
    sycl::nd_range<2> nr({(size_t)(16 * gH), (size_t)gV}, {16, 1});
    for (int i = 0; i < 5; i++)
        q.submit([&](sycl::handler& c) { c.parallel_for(nr, [=](sycl::nd_item<2> n) SYCL_ESIMD_KERNEL {
            flashAttnBMha128Bf16IoPrecomputed(reinterpret_cast<uint8_t*>(d_Q), reinterpret_cast<uint8_t*>(d_K),
                reinterpret_cast<uint8_t*>(d_V), reinterpret_cast<uint8_t*>(d_nA),
                reinterpret_cast<uint8_t*>(d_O), q_len, kv_len, num_heads, num_heads, n);
        }); }).wait();
    constexpr int N = 100;
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < N; i++)
        q.submit([&](sycl::handler& c) { c.parallel_for(nr, [=](sycl::nd_item<2> n) SYCL_ESIMD_KERNEL {
            flashAttnBMha128Bf16IoPrecomputed(reinterpret_cast<uint8_t*>(d_Q), reinterpret_cast<uint8_t*>(d_K),
                reinterpret_cast<uint8_t*>(d_V), reinterpret_cast<uint8_t*>(d_nA),
                reinterpret_cast<uint8_t*>(d_O), q_len, kv_len, num_heads, num_heads, n);
        }); }).wait();
    auto t1 = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(t1 - t0).count() / N;
}

void run_perf(sycl::queue& q, int q_len, int kv_len, int num_heads, const char* label) {
    constexpr int HD = 128;
    constexpr double PEAK_TFLOPS = 135.0;
    size_t q_size = q_len * num_heads * HD;
    size_t kv_size = kv_len * num_heads * HD;

    fp16* d_Qf = sycl::malloc_device<fp16>(q_size, q);
    fp16* d_Kf = sycl::malloc_device<fp16>(kv_size, q);
    fp16* d_Vf = sycl::malloc_device<fp16>(kv_size, q);
    fp16* d_Of = sycl::malloc_device<fp16>(q_size, q);

    bf16* d_Qb = sycl::malloc_device<bf16>(q_size, q);
    bf16* d_Kb = sycl::malloc_device<bf16>(kv_size, q);
    bf16* d_Vb = sycl::malloc_device<bf16>(kv_size, q);
    bf16* d_Ob = sycl::malloc_device<bf16>(q_size, q);

    float* d_nA = sycl::malloc_device<float>(num_heads * HD, q);

    std::vector<fp16> hf(std::max(q_size, kv_size));
    std::vector<bf16> hb(std::max(q_size, kv_size));
    std::vector<float> ha(num_heads * HD, 1.0f);
    std::mt19937 gen(12345);
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);

    for (size_t i = 0; i < q_size; i++) { float v = dis(gen) * 0.1f; hf[i] = fp16(v); hb[i] = bf16(v); }
    q.memcpy(d_Qf, hf.data(), q_size * sizeof(fp16)).wait();
    q.memcpy(d_Qb, hb.data(), q_size * sizeof(bf16)).wait();

    for (size_t i = 0; i < kv_size; i++) { float v = dis(gen) * 0.1f; hf[i] = fp16(v); hb[i] = bf16(v); }
    q.memcpy(d_Kf, hf.data(), kv_size * sizeof(fp16)).wait();
    q.memcpy(d_Kb, hb.data(), kv_size * sizeof(bf16)).wait();

    for (size_t i = 0; i < kv_size; i++) { float v = dis(gen) * 0.1f; hf[i] = fp16(v); hb[i] = bf16(v); }
    q.memcpy(d_Vf, hf.data(), kv_size * sizeof(fp16)).wait();
    q.memcpy(d_Vb, hb.data(), kv_size * sizeof(bf16)).wait();

    q.memcpy(d_nA, ha.data(), num_heads * HD * sizeof(float)).wait();

    double flops = 4.0 * q_len * kv_len * HD * num_heads + 2.0 * q_len * kv_len * num_heads;

    double t_fp16 = bench_fp16(q, q_len, kv_len, num_heads, d_Qf, d_Kf, d_Vf, d_Of, d_nA);
    double r_fp16 = (flops / (t_fp16 * 1e9)) / PEAK_TFLOPS * 100.0;

    double t_bf16 = bench_bf16(q, q_len, kv_len, num_heads, d_Qb, d_Kb, d_Vb, d_Ob, d_nA);
    double ratio_bf16 = t_bf16 / t_fp16;

    double t_bf16io = bench_bf16io(q, q_len, kv_len, num_heads, d_Qb, d_Kb, d_Vb, d_Ob, d_nA);
    double ratio_bf16io = t_bf16io / t_fp16;

    std::cout << std::setw(12) << label
              << " | fp16: " << std::fixed << std::setprecision(3) << std::setw(8) << t_fp16 << "ms "
              << std::setprecision(1) << std::setw(5) << r_fp16 << "%"
              << " | bf16: " << std::setprecision(3) << std::setw(8) << t_bf16 << "ms x" << std::setprecision(3) << ratio_bf16
              << " | bf16io: " << std::setprecision(3) << std::setw(8) << t_bf16io << "ms x" << std::setprecision(3) << ratio_bf16io
              << "\n";

    sycl::free(d_Qf, q); sycl::free(d_Kf, q); sycl::free(d_Vf, q); sycl::free(d_Of, q);
    sycl::free(d_Qb, q); sycl::free(d_Kb, q); sycl::free(d_Vb, q); sycl::free(d_Ob, q);
    sycl::free(d_nA, q);
}

int main() {
    constexpr int NUM_HEADS = 32;

    sycl::queue q(sycl::gpu_selector_v);
    std::cout << "================================================================\n";
    std::cout << "Best 3 SDP Kernels — Unified Test\n";
    std::cout << "================================================================\n";
    std::cout << "Device: " << q.get_device().get_info<sycl::info::device::name>() << "\n";
    std::cout << "Heads: " << NUM_HEADS << ", Head Dim: 128, Non-causal\n";
    std::cout << "Kernels:\n";
    std::cout << "  fp16   — fp16 I/O, fp16 internal, barrier+interleave opt\n";
    std::cout << "  bf16   — bf16 I/O, bf16 DPAS, barrier+interleave opt\n";
    std::cout << "  bf16io — bf16 I/O, hybrid bf16 QK + fp16 SxV, V conv in SxV\n\n";

    // ===== Correctness =====
    std::cout << "--- Correctness (vs CPU fp32 reference) ---\n";
    int pass_count = 0, total = 0;
    struct CC { int ql; int kvl; const char* n; };
    CC ctests[] = { {1024, 1024, "1Kx1K"}, {512, 1024, "512x1K"} };
    for (auto& t : ctests) {
        test_correctness_all(q, t.ql, t.kvl, NUM_HEADS, t.n);
        total++;
    }
    std::cout << "\n";

    // ===== Performance =====
    std::cout << "--- Performance (100 iters, non-causal) ---\n";
    std::cout << std::string(105, '-') << "\n";
    struct PC { int ql; int kvl; const char* n; };
    PC ptests[] = { {8192, 8192, "8Kx8K"}, {8192, 16384, "8Kx16K"}, {16384, 16384, "16Kx16K"} };
    for (auto& t : ptests) {
        run_perf(q, t.ql, t.kvl, NUM_HEADS, t.n);
    }

    std::cout << "\n================================================================\n";
    std::cout << "Done.\n";
    std::cout << "================================================================\n";
    return 0;
}
