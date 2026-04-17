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
using namespace sycl::ext::intel::experimental::esimd;

/*
 * Optimized FP16 GEMM - branch-free inner loops
 *
 * Key change vs fp16_gemm_final_best.cpp:
 *   - All "if" checks removed from the K-loop body
 *   - First iteration handled as an explicit prologue (load buf[0] before loop)
 *   - Main loop  : k = 0 .. K-2*K_STEP, no ifs at all
 *   - Last block : k = K-K_STEP, written out explicitly, no load on final substep
 *   - Store      : only place that keeps M/N boundary checks
 *
 * Assumes K is a multiple of K_STEP (32), M multiple of 32, N multiple of 64.
 */
struct FP16_GEMM_NOIF {
    const sycl::half* A;    // M x K row-major
    const sycl::half* B_T;  // K x N row-major (transposed from N x K)
    sycl::half* C;          // M x N row-major
    int M, N, K;

    void operator()(sycl::nd_item<1> item) const SYCL_ESIMD_KERNEL {
        constexpr int K_STEP  = 32;   // outer K-loop step
        constexpr int K_SUB   = 16;   // one DPAS K-width

        int wg_id    = item.get_group(0);
        int local_id = item.get_local_id(0);

        int wg_tiles_n = (N + 255) / 256;
        int m_start = (wg_id / wg_tiles_n) * 256 + (local_id / 4) * 32;
        int n_start = (wg_id % wg_tiles_n) * 256 + (local_id % 4) * 64;

        if (m_start >= M || n_start >= N) return;

        // Accumulators: 4 M-blocks x 4 N-blocks, each 8x16
        simd<sycl::half, 8*16> acc[4][4] = {0};

        // Double buffers: [buf][block]
        simd<sycl::half, 8*16>  bb[2][4];   // A tiles
        simd<sycl::half, 16*16> aa[2][4];   // B tiles (transposed)

        // ------------------------------------------------------------------
        // PROLOGUE: load buf[0] for (k=0, substep=0) - no if, assume aligned
        // ------------------------------------------------------------------
        #pragma unroll
        for (int m = 0; m < 4; m++)
            bb[0][m] = lsc_load_2d<sycl::half, 16, 8, 1, false, false,
                cache_hint::cached, cache_hint::cached>(
                A, K*sizeof(sycl::half)-1, M-1, K*sizeof(sycl::half)-1,
                0, m_start + m*8);

        #pragma unroll
        for (int n = 0; n < 4; n++)
            aa[0][n] = lsc_load_2d<sycl::half, 16, 16, 1, false, true,
                cache_hint::cached, cache_hint::cached>(
                B_T, N*sizeof(sycl::half)-1, K-1, N*sizeof(sycl::half)-1,
                n_start + n*16, 0);

        // ------------------------------------------------------------------
        // MAIN LOOP: k = 0 .. K-2*K_STEP  (at least one more K-block follows)
        // No "if" at all inside. Every load address is guaranteed in-range:
        //   - prefetch  pf_k = k+K_STEP,  pf_k+K_SUB: both < K  (k <= K-2*K_STEP)
        //   - load ks=1 k+K_SUB          : < K  (K_SUB=16 < K_STEP=32)
        //   - load next k+K_STEP         : < K  (k < K-K_STEP in this loop)
        // ------------------------------------------------------------------
        for (int k = 0; k < K - K_STEP; k += K_STEP) {

            // Prefetch next K-block (PF_DIST=2: pf_k = k + K_STEP)
            int pf_k = k + K_STEP;
            #pragma unroll
            for (int m = 0; m < 4; m++) {
                lsc_prefetch_2d<sycl::half, 16, 8, 1,
                    cache_hint::cached, cache_hint::cached>(
                    A, K*sizeof(sycl::half)-1, M-1, K*sizeof(sycl::half)-1,
                    pf_k,       m_start + m*8);
                lsc_prefetch_2d<sycl::half, 16, 8, 1,
                    cache_hint::cached, cache_hint::cached>(
                    A, K*sizeof(sycl::half)-1, M-1, K*sizeof(sycl::half)-1,
                    pf_k+K_SUB, m_start + m*8);
            }
            #pragma unroll
            for (int n = 0; n < 4; n++) {
                lsc_prefetch_2d<sycl::half, 16, 16, 1,
                    cache_hint::cached, cache_hint::cached>(
                    B_T, N*sizeof(sycl::half)-1, K-1, N*sizeof(sycl::half)-1,
                    n_start + n*16, pf_k);
                lsc_prefetch_2d<sycl::half, 16, 16, 1,
                    cache_hint::cached, cache_hint::cached>(
                    B_T, N*sizeof(sycl::half)-1, K-1, N*sizeof(sycl::half)-1,
                    n_start + n*16, pf_k+K_SUB);
            }

            // --- substep 0: issue loads for substep 1 into buf[1],
            //                then compute substep 0 from buf[0] ---
            #pragma unroll
            for (int m = 0; m < 4; m++)
                bb[1][m] = lsc_load_2d<sycl::half, 16, 8, 1, false, false,
                    cache_hint::cached, cache_hint::cached>(
                    A, K*sizeof(sycl::half)-1, M-1, K*sizeof(sycl::half)-1,
                    k + K_SUB, m_start + m*8);

            #pragma unroll
            for (int n = 0; n < 4; n++)
                aa[1][n] = lsc_load_2d<sycl::half, 16, 16, 1, false, true,
                    cache_hint::cached, cache_hint::cached>(
                    B_T, N*sizeof(sycl::half)-1, K-1, N*sizeof(sycl::half)-1,
                    n_start + n*16, k + K_SUB);

            #pragma unroll
            for (int m = 0; m < 4; m++)
                #pragma unroll
                for (int n = 0; n < 4; n++)
                    acc[m][n] = xmx::dpas<8, 8,
                        sycl::half, sycl::half, sycl::half, sycl::half>(
                        acc[m][n], aa[0][n], bb[0][m]);

            // --- substep 1: issue loads for (k+K_STEP, substep 0) into buf[0],
            //                then compute substep 1 from buf[1] ---
            #pragma unroll
            for (int m = 0; m < 4; m++)
                bb[0][m] = lsc_load_2d<sycl::half, 16, 8, 1, false, false,
                    cache_hint::cached, cache_hint::cached>(
                    A, K*sizeof(sycl::half)-1, M-1, K*sizeof(sycl::half)-1,
                    k + K_STEP, m_start + m*8);

            #pragma unroll
            for (int n = 0; n < 4; n++)
                aa[0][n] = lsc_load_2d<sycl::half, 16, 16, 1, false, true,
                    cache_hint::cached, cache_hint::cached>(
                    B_T, N*sizeof(sycl::half)-1, K-1, N*sizeof(sycl::half)-1,
                    n_start + n*16, k + K_STEP);

            #pragma unroll
            for (int m = 0; m < 4; m++)
                #pragma unroll
                for (int n = 0; n < 4; n++)
                    acc[m][n] = xmx::dpas<8, 8,
                        sycl::half, sycl::half, sycl::half, sycl::half>(
                        acc[m][n], aa[1][n], bb[1][m]);
        }

        // ------------------------------------------------------------------
        // LAST K-BLOCK: k = K - K_STEP
        //   substep 0: load substep-1 data into buf[1], compute from buf[0]
        //   substep 1: no more loads (would go out of range), compute from buf[1]
        // buf[0] still holds (K-K_STEP, substep=0) from end of main loop
        //   (or from prologue when K == K_STEP).
        // ------------------------------------------------------------------
        {
            const int k = K - K_STEP;

            // substep 0 - load buf[1] for k+K_SUB (= K-K_STEP+16 < K, always valid)
            #pragma unroll
            for (int m = 0; m < 4; m++)
                bb[1][m] = lsc_load_2d<sycl::half, 16, 8, 1, false, false,
                    cache_hint::cached, cache_hint::cached>(
                    A, K*sizeof(sycl::half)-1, M-1, K*sizeof(sycl::half)-1,
                    k + K_SUB, m_start + m*8);

            #pragma unroll
            for (int n = 0; n < 4; n++)
                aa[1][n] = lsc_load_2d<sycl::half, 16, 16, 1, false, true,
                    cache_hint::cached, cache_hint::cached>(
                    B_T, N*sizeof(sycl::half)-1, K-1, N*sizeof(sycl::half)-1,
                    n_start + n*16, k + K_SUB);

            #pragma unroll
            for (int m = 0; m < 4; m++)
                #pragma unroll
                for (int n = 0; n < 4; n++)
                    acc[m][n] = xmx::dpas<8, 8,
                        sycl::half, sycl::half, sycl::half, sycl::half>(
                        acc[m][n], aa[0][n], bb[0][m]);

            // substep 1 - no loads, just compute
            #pragma unroll
            for (int m = 0; m < 4; m++)
                #pragma unroll
                for (int n = 0; n < 4; n++)
                    acc[m][n] = xmx::dpas<8, 8,
                        sycl::half, sycl::half, sycl::half, sycl::half>(
                        acc[m][n], aa[1][n], bb[1][m]);
        }

        // ------------------------------------------------------------------
        // STORE: only place with M/N boundary checks
        // ------------------------------------------------------------------
        #pragma unroll
        for (int m = 0; m < 4; m++) {
            #pragma unroll
            for (int n = 0; n < 4; n++) {
                int m_idx = m_start + m * 8;
                int n_idx = n_start + n * 16;
                if (m_idx + 8 <= M && n_idx + 16 <= N)
                    lsc_store_2d<sycl::half, 16, 8,
                        cache_hint::write_back, cache_hint::write_back>(
                        C, N*sizeof(sycl::half)-1, M-1, N*sizeof(sycl::half)-1,
                        n_idx, m_idx, acc[m][n]);
            }
        }
    }
};

double benchmark(queue& q,
                 sycl::half* d_A, sycl::half* d_B_T, sycl::half* d_C,
                 int M, int N, int K)
{
    int num_wg_m = (M + 255) / 256;
    int num_wg_n = (N + 255) / 256;
    constexpr int WG_SIZE   = 32;
    int global_size = num_wg_m * num_wg_n * WG_SIZE;

    // Warmup: 20 iters to trigger freq drop
    for (int i = 0; i < 20; i++)
        q.submit([&](handler& h) {
            h.parallel_for(nd_range<1>(global_size, WG_SIZE),
                FP16_GEMM_NOIF{d_A, d_B_T, d_C, M, N, K});
        });
    q.wait();

    // Timed: 1000 iters at stable frequency, single wait at the end
    constexpr int NUM_ITERS = 1000;
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < NUM_ITERS; i++)
        q.submit([&](handler& h) {
            h.parallel_for(nd_range<1>(global_size, WG_SIZE),
                FP16_GEMM_NOIF{d_A, d_B_T, d_C, M, N, K});
        });
    q.wait();
    auto t1 = std::chrono::high_resolution_clock::now();

    return std::chrono::duration<double>(t1 - t0).count() / NUM_ITERS;
}

int main(int argc, char** argv) {
    int M = 4096, N = 4096, K = 4096;
    if (argc > 1) M = std::atoi(argv[1]);
    if (argc > 2) N = std::atoi(argv[2]);
    if (argc > 3) K = std::atoi(argv[3]);

    std::cout << "FP16 GEMM - branch-free inner loops\n";
    std::cout << "M=" << M << " N=" << N << " K=" << K << "\n";
    std::cout << "Config: K-step=32, PF_DIST=2, WG 256x256, Thread 32x64\n";
    std::cout << "Assumes K%32==0, M%32==0, N%64==0\n\n";

    // -----------------------------------------------------------------------
    // SLEEP 5 SECONDS - attach VTune now before GPU work starts
    // -----------------------------------------------------------------------
    std::cout << ">>> Sleeping 5 seconds - attach VTune GPU collection now <<<\n";
    std::cout.flush();
    std::this_thread::sleep_for(std::chrono::seconds(5));
    std::cout << ">>> Sleep done, starting GPU work <<<\n\n";
    std::cout.flush();

    try {
        queue q(gpu_selector_v, property::queue::in_order{});
        std::cout << "Device: " << q.get_device().get_info<sycl::info::device::name>() << "\n\n";

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

        double time_s = benchmark(q, d_A, d_B_T, d_C, M, N, K);
        double tflops = 2.0*M*N*K / time_s / 1e12;
        constexpr double PEAK = 135.0;

        std::cout << "Time/iter : " << time_s*1000.0 << " ms\n";
        std::cout << "TFLOPS    : " << tflops << "\n";
        std::cout << "% of 135T : " << (tflops/PEAK*100.0) << "%\n";
        std::cout << "vs old    : " << (tflops/88.85*100.0) << "%  (baseline 88.85 TFLOPS)\n\n";

        // Correctness: compare against CPU on a 256x256 sub-block
        std::cout << "Verifying...\n";
        std::vector<sycl::half> h_C(M*N);
        q.memcpy(h_C.data(), d_C, M*N*sizeof(sycl::half)).wait();

        int tsz = std::min(256, std::min(M, N));
        int errors = 0; float max_diff = 0.f;
        for (int m = 0; m < tsz && errors < 5; m++) {
            for (int n = 0; n < tsz && errors < 5; n++) {
                float ref = 0.f;
                for (int k = 0; k < K; k++)
                    ref += float(h_A[m*K+k]) * float(h_B[n*K+k]);
                float got  = float(h_C[m*N+n]);
                float diff = std::abs(got - ref);
                max_diff   = std::max(max_diff, diff);
                if (diff > 1.f && std::abs(ref) > 1e-6f && diff/std::abs(ref) > 0.05f)
                    errors++;
            }
        }
        std::cout << "Max diff: " << max_diff << "  Errors: " << errors << "\n";
        std::cout << (errors == 0 ? "PASSED\n" : "FAILED\n");

        free(d_A, q); free(d_B_T, q); free(d_C, q);
        return errors == 0 ? 0 : 1;

    } catch (const sycl::exception& e) {
        std::cerr << "SYCL: " << e.what() << "\n";
        return 1;
    }
}
