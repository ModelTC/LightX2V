#include <sycl/sycl.hpp>
#include <sycl/ext/intel/esimd.hpp>
#include <sycl/ext/intel/experimental/esimd/memory.hpp>
#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <thread>

using namespace sycl;
using namespace sycl::ext::intel::esimd;
namespace xesimd = sycl::ext::intel::experimental::esimd;

// ---------------------------------------------------------------------------
// NOPF3: induction-variable XVE reduction, single payA/payB (no GRF increase)
//
// vs nopf (117.10T, 1.85 GHz, 4% XVE):
//   nopf:  loop variable k (int) → set_x(k + K_SUB) = ADD+MOV per substep
//                                  set_x(k + K_STEP) = ADD+MOV per substep
//          same for payB set_y: 4 ADDs per K_STEP total
//   nopf3: uint32_t ks = K_SUB, kn = K_STEP; ks += K_STEP, kn += K_STEP
//          → set_x(ks) / set_x(kn) = pure MOV (no ADD at call site)
//          2 fewer ADDs per K_STEP = 2 × 128 = 256 fewer XVE instructions total
//
// Additionally test L1=uncached variant (MULTI_L1UC).
// ---------------------------------------------------------------------------

// Sweep over two cache hint combos for loads
enum class LoadMode { CACHED, L1UC };

template<LoadMode LM>
struct FP16_GEMM_NOPF3 {
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

        // Single payload objects — same GRF as nopf
        xesimd::config_2d_mem_access<sycl::half, 16, 8,  1> payA(
            A,   surfW_A, surfH_A, surfW_A, 0u, 0u);
        xesimd::config_2d_mem_access<sycl::half, 16, 16, 1> payB(
            B_T, surfW_B, surfH_B, surfW_B, 0u, 0u);

        simd<sycl::half, 8*16>  acc[4][4] = {0};
        simd<sycl::half, 8*16>  bb[2][4];
        simd<sycl::half, 16*16> aa[2][4];

#define LOAD_A(dst, buf, m)                                                    \
    if constexpr (LM == LoadMode::CACHED)                                      \
        dst[buf][m] = xesimd::lsc_load_2d<sycl::half, 16, 8, 1, false, false, \
            xesimd::cache_hint::cached, xesimd::cache_hint::cached>(payA);     \
    else                                                                       \
        dst[buf][m] = xesimd::lsc_load_2d<sycl::half, 16, 8, 1, false, false, \
            xesimd::cache_hint::uncached, xesimd::cache_hint::cached>(payA)

#define LOAD_B(dst, buf, n)                                                    \
    if constexpr (LM == LoadMode::CACHED)                                      \
        dst[buf][n] = xesimd::lsc_load_2d<sycl::half, 16, 16, 1, false, true, \
            xesimd::cache_hint::cached, xesimd::cache_hint::cached>(payB);     \
    else                                                                       \
        dst[buf][n] = xesimd::lsc_load_2d<sycl::half, 16, 16, 1, false, true, \
            xesimd::cache_hint::uncached, xesimd::cache_hint::cached>(payB)

        // PROLOGUE: load buf[0] at k=0, x=0, y=m_start+m*8
        payA.set_x(0u);
        #pragma unroll
        for (int m = 0; m < 4; m++) {
            payA.set_y((uint32_t)(m_start + m * 8));
            LOAD_A(bb, 0, m);
        }
        payB.set_y(0u);
        #pragma unroll
        for (int n = 0; n < 4; n++) {
            payB.set_x((uint32_t)(n_start + n * 16));
            LOAD_B(aa, 0, n);
        }

        // MAIN LOOP: induction vars ks = K_SUB, kn = K_STEP
        // ks/kn are uint32_t → set_x/set_y become pure MOVs, no ADD at call site
        for (uint32_t ks = (uint32_t)K_SUB, kn = (uint32_t)K_STEP;
             kn < (uint32_t)K;
             ks += (uint32_t)K_STEP, kn += (uint32_t)K_STEP)
        {
            // substep 0: load buf[1] at ks, compute buf[0]
            payA.set_x(ks);
            #pragma unroll
            for (int m = 0; m < 4; m++) {
                payA.set_y((uint32_t)(m_start + m * 8));
                LOAD_A(bb, 1, m);
            }
            payB.set_y(ks);
            #pragma unroll
            for (int n = 0; n < 4; n++) {
                payB.set_x((uint32_t)(n_start + n * 16));
                LOAD_B(aa, 1, n);
            }
            #pragma unroll
            for (int m = 0; m < 4; m++)
                #pragma unroll
                for (int n = 0; n < 4; n++)
                    acc[m][n] = xmx::dpas<8, 8, sycl::half, sycl::half,
                        sycl::half, sycl::half>(acc[m][n], aa[0][n], bb[0][m]);

            // substep 1: load buf[0] at kn, compute buf[1]
            payA.set_x(kn);
            #pragma unroll
            for (int m = 0; m < 4; m++) {
                payA.set_y((uint32_t)(m_start + m * 8));
                LOAD_A(bb, 0, m);
            }
            payB.set_y(kn);
            #pragma unroll
            for (int n = 0; n < 4; n++) {
                payB.set_x((uint32_t)(n_start + n * 16));
                LOAD_B(aa, 0, n);
            }
            #pragma unroll
            for (int m = 0; m < 4; m++)
                #pragma unroll
                for (int n = 0; n < 4; n++)
                    acc[m][n] = xmx::dpas<8, 8, sycl::half, sycl::half,
                        sycl::half, sycl::half>(acc[m][n], aa[1][n], bb[1][m]);
        }

        // LAST K-BLOCK: k = K - K_STEP
        {
            const uint32_t ks_last = (uint32_t)(K - K_STEP + K_SUB);

            payA.set_x(ks_last);
            #pragma unroll
            for (int m = 0; m < 4; m++) {
                payA.set_y((uint32_t)(m_start + m * 8));
                LOAD_A(bb, 1, m);
            }
            payB.set_y(ks_last);
            #pragma unroll
            for (int n = 0; n < 4; n++) {
                payB.set_x((uint32_t)(n_start + n * 16));
                LOAD_B(aa, 1, n);
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

#undef LOAD_A
#undef LOAD_B

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

template<LoadMode LM>
double run_one(queue& q, sycl::half* d_A, sycl::half* d_B_T, sycl::half* d_C,
               int M, int N, int K)
{
    int num_wg_m    = (M + 255) / 256;
    int num_wg_n    = (N + 255) / 256;
    constexpr int WG_SIZE = 32;
    int global_size = num_wg_m * num_wg_n * WG_SIZE;

    for (int i = 0; i < 20; i++)
        q.submit([&](handler& h) {
            h.parallel_for(nd_range<1>(global_size, WG_SIZE),
                FP16_GEMM_NOPF3<LM>{d_A, d_B_T, d_C, M, N, K});
        });
    q.wait();

    constexpr int NUM_ITERS = 1000;
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < NUM_ITERS; i++)
        q.submit([&](handler& h) {
            h.parallel_for(nd_range<1>(global_size, WG_SIZE),
                FP16_GEMM_NOPF3<LM>{d_A, d_B_T, d_C, M, N, K});
        });
    q.wait();
    auto t1 = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double>(t1 - t0).count() / NUM_ITERS;
}

void row(const char* label, double t, double flops, double base) {
    double tf = flops / t / 1e12;
    printf("  %-48s  %6.2f TFLOPS  %5.1f%%  %+5.1f%%\n",
           label, tf, tf/135.0*100.0, (tf-base)/base*100.0);
    fflush(stdout);
}

int main(int argc, char** argv) {
    int M = 4096, N = 4096, K = 4096;
    if (argc > 1) M = std::atoi(argv[1]);
    if (argc > 2) N = std::atoi(argv[2]);
    if (argc > 3) K = std::atoi(argv[3]);

    printf("FP16 GEMM - NOPF3: induction-var XVE reduction (single payA/payB)\n");
    printf("M=%d N=%d K=%d   WG=256x256, 32 threads, K-step=32\n\n", M, N, K);
    printf("vs nopf: uint32_t ks/kn induction vars → set_x/set_y = pure MOV\n");
    printf("         4 fewer ADDs per K_STEP, zero GRF increase\n\n");

    printf(">>> Sleeping 5 seconds - attach VTune now if needed <<<\n");
    fflush(stdout);
    std::this_thread::sleep_for(std::chrono::seconds(5));
    printf(">>> Starting <<<\n\n");
    fflush(stdout);

    try {
        queue q(gpu_selector_v, property::queue::in_order{});
        printf("Device: %s\n\n",
               q.get_device().get_info<sycl::info::device::name>().c_str());

        std::vector<sycl::half> h_A(M*K), h_B(N*K), h_B_T(K*N);
        std::mt19937 gen(42);
        std::uniform_real_distribution<float> dis(-1.f, 1.f);
        for (auto& v : h_A)  v = sycl::half(dis(gen));
        for (auto& v : h_B)  v = sycl::half(dis(gen));
        for (int n = 0; n < N; n++)
            for (int k = 0; k < K; k++)
                h_B_T[k*N + n] = h_B[n*K + k];

        auto* d_A   = malloc_device<sycl::half>(M*K, q);
        auto* d_B_T = malloc_device<sycl::half>(K*N, q);
        auto* d_C   = malloc_device<sycl::half>(M*N, q);
        q.memcpy(d_A,   h_A.data(),   M*K*sizeof(sycl::half)).wait();
        q.memcpy(d_B_T, h_B_T.data(), K*N*sizeof(sycl::half)).wait();

        double flops = 2.0*M*N*K;

        printf("Reference: nopf.exe = 117.10 TFLOPS (1.85 GHz, 4%% XVE)\n\n");
        printf("Results (1000 iters, peak=135 TFLOPS):\n");
        printf("  %s\n", std::string(86, '-').c_str());

        double t_cached = run_one<LoadMode::CACHED>(q, d_A, d_B_T, d_C, M, N, K);
        double base     = flops / t_cached / 1e12;
        row("NOPF3 CACHED  (ks/kn induction, L1=cached)", t_cached, flops, base);

        double t_l1uc = run_one<LoadMode::L1UC>(q, d_A, d_B_T, d_C, M, N, K);
        row("NOPF3 L1UC    (ks/kn induction, L1=uncached)", t_l1uc, flops, base);

        printf("  %s\n\n", std::string(86, '-').c_str());

        printf("Best result:\n");
        double best_t  = (t_cached < t_l1uc) ? t_cached : t_l1uc;
        double best_tf = flops / best_t / 1e12;
        printf("  Time/iter : %.4f ms\n", best_t * 1000.0);
        printf("  TFLOPS    : %.2f\n", best_tf);
        printf("  %% of 135T : %.1f%%\n", best_tf / 135.0 * 100.0);

        free(d_A, q); free(d_B_T, q); free(d_C, q);
        return 0;

    } catch (const sycl::exception& e) {
        fprintf(stderr, "SYCL: %s\n", e.what());
        return 1;
    }
}
