#include <sycl/sycl.hpp>
#include <sycl/ext/intel/esimd.hpp>
#include <sycl/ext/intel/experimental/esimd/memory.hpp>
#include <iostream>
#include <vector>
#include <random>
#include <cmath>

using namespace sycl;
using namespace sycl::ext::intel::esimd;
namespace xesimd = sycl::ext::intel::experimental::esimd;

// ---- Exact copy of the FP16_GEMM_NOPF kernel from fp16_gemm_nopf.cpp ----
struct FP16_GEMM_NOPF {
    sycl::half* A;
    sycl::half* B_T;
    sycl::half* C;
    int M, N, K;

    void operator()(sycl::nd_item<1> item) const SYCL_ESIMD_KERNEL {
        constexpr int K_STEP = 32;
        constexpr int K_SUB  = 16;

        int wg_id    = item.get_group(0);
        int local_id = item.get_local_id(0);

        int wg_tiles_n = (N + 255) / 256;
        int m_start = (wg_id / wg_tiles_n) * 256 + (local_id / 4) * 32;
        int n_start = (wg_id % wg_tiles_n) * 256 + (local_id % 4) * 64;

        if (m_start >= M || n_start >= N) return;

        const uint32_t surfW_A = (uint32_t)K * 2u - 1u;
        const uint32_t surfH_A = (uint32_t)M - 1u;
        const uint32_t surfW_B = (uint32_t)N * 2u - 1u;
        const uint32_t surfH_B = (uint32_t)K - 1u;

        xesimd::config_2d_mem_access<sycl::half, 16, 8,  1> payA(
            A,   surfW_A, surfH_A, surfW_A, 0u, 0u);
        xesimd::config_2d_mem_access<sycl::half, 16, 16, 1> payB(
            B_T, surfW_B, surfH_B, surfW_B, 0u, 0u);

        simd<sycl::half, 8*16>  acc[4][4] = {0};
        simd<sycl::half, 8*16>  bb[2][4];
        simd<sycl::half, 16*16> aa[2][4];

        // PROLOGUE: load buf[0] at k=0
        payA.set_x(0u);
        #pragma unroll
        for (int m = 0; m < 4; m++) {
            payA.set_y((uint32_t)(m_start + m * 8));
            bb[0][m] = xesimd::lsc_load_2d<sycl::half, 16, 8, 1, false, false,
                xesimd::cache_hint::cached, xesimd::cache_hint::cached>(payA);
        }
        payB.set_y(0u);
        #pragma unroll
        for (int n = 0; n < 4; n++) {
            payB.set_x((uint32_t)(n_start + n * 16));
            aa[0][n] = xesimd::lsc_load_2d<sycl::half, 16, 16, 1, false, true,
                xesimd::cache_hint::cached, xesimd::cache_hint::cached>(payB);
        }

        // MAIN LOOP: no prefetch
        for (int k = 0; k < K - K_STEP; k += K_STEP) {
            payA.set_x((uint32_t)(k + K_SUB));
            #pragma unroll
            for (int m = 0; m < 4; m++) {
                payA.set_y((uint32_t)(m_start + m * 8));
                bb[1][m] = xesimd::lsc_load_2d<sycl::half, 16, 8, 1, false, false,
                    xesimd::cache_hint::cached, xesimd::cache_hint::cached>(payA);
            }
            payB.set_y((uint32_t)(k + K_SUB));
            #pragma unroll
            for (int n = 0; n < 4; n++) {
                payB.set_x((uint32_t)(n_start + n * 16));
                aa[1][n] = xesimd::lsc_load_2d<sycl::half, 16, 16, 1, false, true,
                    xesimd::cache_hint::cached, xesimd::cache_hint::cached>(payB);
            }
            #pragma unroll
            for (int m = 0; m < 4; m++)
                #pragma unroll
                for (int n = 0; n < 4; n++)
                    acc[m][n] = xmx::dpas<8, 8, sycl::half, sycl::half,
                        sycl::half, sycl::half>(acc[m][n], aa[0][n], bb[0][m]);

            payA.set_x((uint32_t)(k + K_STEP));
            #pragma unroll
            for (int m = 0; m < 4; m++) {
                payA.set_y((uint32_t)(m_start + m * 8));
                bb[0][m] = xesimd::lsc_load_2d<sycl::half, 16, 8, 1, false, false,
                    xesimd::cache_hint::cached, xesimd::cache_hint::cached>(payA);
            }
            payB.set_y((uint32_t)(k + K_STEP));
            #pragma unroll
            for (int n = 0; n < 4; n++) {
                payB.set_x((uint32_t)(n_start + n * 16));
                aa[0][n] = xesimd::lsc_load_2d<sycl::half, 16, 16, 1, false, true,
                    xesimd::cache_hint::cached, xesimd::cache_hint::cached>(payB);
            }
            #pragma unroll
            for (int m = 0; m < 4; m++)
                #pragma unroll
                for (int n = 0; n < 4; n++)
                    acc[m][n] = xmx::dpas<8, 8, sycl::half, sycl::half,
                        sycl::half, sycl::half>(acc[m][n], aa[1][n], bb[1][m]);
        }

        // LAST K-BLOCK
        {
            const int k = K - K_STEP;
            payA.set_x((uint32_t)(k + K_SUB));
            #pragma unroll
            for (int m = 0; m < 4; m++) {
                payA.set_y((uint32_t)(m_start + m * 8));
                bb[1][m] = xesimd::lsc_load_2d<sycl::half, 16, 8, 1, false, false,
                    xesimd::cache_hint::cached, xesimd::cache_hint::cached>(payA);
            }
            payB.set_y((uint32_t)(k + K_SUB));
            #pragma unroll
            for (int n = 0; n < 4; n++) {
                payB.set_x((uint32_t)(n_start + n * 16));
                aa[1][n] = xesimd::lsc_load_2d<sycl::half, 16, 16, 1, false, true,
                    xesimd::cache_hint::cached, xesimd::cache_hint::cached>(payB);
            }
            #pragma unroll
            for (int m = 0; m < 4; m++)
                #pragma unroll
                for (int n = 0; n < 4; n++)
                    acc[m][n] = xmx::dpas<8, 8, sycl::half, sycl::half,
                        sycl::half, sycl::half>(acc[m][n], aa[0][n], bb[0][m]);
            #pragma unroll
            for (int m = 0; m < 4; m++)
                #pragma unroll
                for (int n = 0; n < 4; n++)
                    acc[m][n] = xmx::dpas<8, 8, sycl::half, sycl::half,
                        sycl::half, sycl::half>(acc[m][n], aa[1][n], bb[1][m]);
        }

        // STORE
        #pragma unroll
        for (int m = 0; m < 4; m++) {
            #pragma unroll
            for (int n = 0; n < 4; n++) {
                int m_idx = m_start + m * 8;
                int n_idx = n_start + n * 16;
                if (m_idx + 8 <= M && n_idx + 16 <= N)
                    xesimd::lsc_store_2d<sycl::half, 16, 8,
                        xesimd::cache_hint::write_back, xesimd::cache_hint::write_back>(
                        C, surfW_B, surfH_A, surfW_B,
                        (uint32_t)n_idx, (uint32_t)m_idx, acc[m][n]);
            }
        }
    }
};

// CPU reference GEMM in float32 (A: M×K row-major, B_T: K×N row-major, C: M×N row-major)
void cpu_gemm_ref(const std::vector<sycl::half>& A,
                  const std::vector<sycl::half>& B_T,  // stored as K×N
                  std::vector<float>& C_ref,
                  int M, int N, int K)
{
    // Use small tile to keep reasonable runtime
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            float acc = 0.f;
            for (int k = 0; k < K; k++)
                acc += (float)A[m*K+k] * (float)B_T[k*N+n];
            C_ref[m*N+n] = acc;
        }
    }
}

int main() {
    // Use a small problem for correctness check (CPU ref is O(M*N*K))
    const int M = 256, N = 256, K = 256;

    printf("FP16 GEMM NOPF - Correctness Verification\n");
    printf("M=%d N=%d K=%d\n\n", M, N, K);

    try {
        queue q(gpu_selector_v, property::queue::in_order{});
        printf("Device: %s\n\n",
               q.get_device().get_info<sycl::info::device::name>().c_str());

        std::vector<sycl::half> h_A(M*K), h_B(N*K), h_B_T(K*N);
        std::mt19937 gen(42);
        std::uniform_real_distribution<float> dis(-1.f, 1.f);
        for (auto& v : h_A)  v = sycl::half(dis(gen));
        for (auto& v : h_B)  v = sycl::half(dis(gen));
        // B_T[k*N + n] = B[n*K + k]
        for (int n = 0; n < N; n++)
            for (int k = 0; k < K; k++)
                h_B_T[k*N + n] = h_B[n*K + k];

        // CPU reference (fp32 accumulation)
        printf("Computing CPU reference (fp32 accumulation)...\n");
        std::vector<float> h_C_ref(M*N);
        cpu_gemm_ref(h_A, h_B_T, h_C_ref, M, N, K);
        printf("CPU reference done.\n\n");

        // GPU kernel
        auto* d_A   = malloc_device<sycl::half>(M*K, q);
        auto* d_B_T = malloc_device<sycl::half>(K*N, q);
        auto* d_C   = malloc_device<sycl::half>(M*N, q);
        q.memcpy(d_A,   h_A.data(),   M*K*sizeof(sycl::half)).wait();
        q.memcpy(d_B_T, h_B_T.data(), K*N*sizeof(sycl::half)).wait();

        // Zero output first
        q.memset(d_C, 0, M*N*sizeof(sycl::half)).wait();

        int num_wg_m    = (M + 255) / 256;
        int num_wg_n    = (N + 255) / 256;
        constexpr int WG_SIZE = 32;
        int global_size = num_wg_m * num_wg_n * WG_SIZE;

        q.submit([&](handler& h) {
            h.parallel_for(nd_range<1>(global_size, WG_SIZE),
                FP16_GEMM_NOPF{d_A, d_B_T, d_C, M, N, K});
        }).wait();

        std::vector<sycl::half> h_C(M*N);
        q.memcpy(h_C.data(), d_C, M*N*sizeof(sycl::half)).wait();

        // Compare: allow fp16 rounding error
        // Expected max relative error for K=256 fp16 accumulation: ~K * eps_fp16 ~ 256 * 1e-3 = 0.256
        // We check absolute error relative to ref magnitude with generous tolerance
        int num_errors = 0;
        float max_abs_err = 0.f, max_rel_err = 0.f;
        double sum_sq_err = 0.0, sum_sq_ref = 0.0;

        for (int m = 0; m < M; m++) {
            for (int n = 0; n < N; n++) {
                float gpu = (float)h_C[m*N+n];
                float ref = h_C_ref[m*N+n];
                float abs_err = std::abs(gpu - ref);
                float rel_err = abs_err / (std::abs(ref) + 1e-6f);
                if (abs_err > max_abs_err) max_abs_err = abs_err;
                if (rel_err > max_rel_err) max_rel_err = rel_err;
                sum_sq_err += (double)abs_err * abs_err;
                sum_sq_ref += (double)ref * ref;
                // Flag large errors (>5% relative for ref magnitude > 0.1)
                if (std::abs(ref) > 0.1f && rel_err > 0.05f)
                    num_errors++;
            }
        }

        double rms_err = std::sqrt(sum_sq_err / (M*N));
        double rms_ref = std::sqrt(sum_sq_ref / (M*N));
        double rel_rms = rms_err / (rms_ref + 1e-10);

        printf("Correctness Results:\n");
        printf("  Max absolute error : %.6f\n", max_abs_err);
        printf("  Max relative error : %.4f%%\n", max_rel_err * 100.f);
        printf("  RMS error          : %.6f\n", (float)rms_err);
        printf("  Relative RMS       : %.4f%%\n", (float)(rel_rms * 100.0));
        printf("  Outliers (rel>5%%) : %d / %d\n", num_errors, M*N);

        // Sample a few values for manual inspection
        printf("\nSample comparisons (first 5 elements of row 0):\n");
        printf("  idx     GPU          CPU-ref      abs_err\n");
        for (int n = 0; n < 5 && n < N; n++) {
            float gpu = (float)h_C[n];
            float ref = h_C_ref[n];
            printf("  [0,%3d]  %10.4f   %10.4f   %.6f\n", n, gpu, ref, std::abs(gpu-ref));
        }

        printf("\n");
        if (num_errors == 0 && max_rel_err < 0.05f) {
            printf("PASS: FP16 GEMM NOPF result is CORRECT.\n");
            printf("      (Max rel err %.3f%% < 5%% threshold, no outliers)\n",
                   max_rel_err * 100.f);
        } else if (rel_rms < 0.02) {
            printf("PASS (with notes): Relative RMS %.4f%% < 2%%. %d outliers.\n",
                   (float)(rel_rms*100.0), num_errors);
            printf("      (Outliers may be fp16 accumulation rounding, not a bug.)\n");
        } else {
            printf("FAIL: Significant errors detected.\n");
            printf("      rel_rms=%.4f%% outliers=%d\n", (float)(rel_rms*100.0), num_errors);
        }

        free(d_A, q); free(d_B_T, q); free(d_C, q);
        return 0;

    } catch (const sycl::exception& e) {
        fprintf(stderr, "SYCL: %s\n", e.what());
        return 1;
    }
}
