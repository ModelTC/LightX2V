// fp16_gemm_gather.cpp
//
// A[M×K] row-major (unchanged from nopf)
// B[N×K] row-major  ← NOT transposed; each row is one output neuron
//
// B VNNI loading: lsc_gather<uint32_t, 8> with SOA result layout.
// For DPAS VNNI tile (16 N × 8 k-pairs), one gather per n_blk:
//   byte_offset[nl] = base_B[nb][nl] + k_ab
// where base_B[nb][nl] = (n_start + nb×16 + nl) × K × 2  (row byte offset, precomputed)
//       k_ab = running byte offset into the row, incremented by K_SUB×2 = 32 each substep
// lsc_gather SOA layout: result[p × 16 + nl] = uint32 at (byte_off[nl] + p×4)
//   = { B[nl, k+p×2], B[nl, k+p×2+1] }  → VNNI uint32[k_pair×16 + n_col] directly ✓
// 4 gather calls per substep (vs 32 previously).
//
// Structure: identical double-buffer K-loop from fp16_gemm_nopf (prologue + main + last).
// Offset updates use += only (no recomputation from scratch).

#include <sycl/sycl.hpp>
#include <sycl/ext/intel/esimd.hpp>
#include <sycl/ext/intel/experimental/esimd/memory.hpp>
#include <vector>
#include <random>
#include <chrono>
#include <thread>
#include <cstdio>
#include <cmath>
#include <algorithm>

using namespace sycl;
using namespace sycl::ext::intel::esimd;
namespace xesimd = sycl::ext::intel::experimental::esimd;

// ---------------------------------------------------------------------------
struct FP16_GEMM_GATHER_V2 {
    const sycl::half* A;   // [M × K] row-major
    const sycl::half* B;   // [N × K] row-major  (not transposed)
    sycl::half*       C;   // [M × N] row-major
    int M, N, K;

    void operator()(sycl::nd_item<1> item) const SYCL_ESIMD_KERNEL {
        constexpr int K_STEP = 32;
        constexpr int K_SUB  = 16;

        const int wg_id    = item.get_group(0);
        const int local_id = item.get_local_id(0);

        const int wg_tiles_n = (N + 255) / 256;
        const int m_start = (wg_id / wg_tiles_n) * 256 + (local_id / 4) * 32;
        const int n_start = (wg_id % wg_tiles_n) * 256 + (local_id % 4) * 64;

        if (m_start >= M || n_start >= N) return;

        // ---- Surface descriptors ----
        const uint32_t surfW_A = (uint32_t)K * 2u - 1u;  // A row pitch in bytes - 1
        const uint32_t surfH_A = (uint32_t)M - 1u;
        const uint32_t surfW_C = (uint32_t)N * 2u - 1u;  // C row pitch in bytes - 1

        // ---- A payload (CSE: construct once outside K-loop) ----
        xesimd::config_2d_mem_access<sycl::half, 16, 8, 1> payA(
            A, surfW_A, surfH_A, surfW_A, 0u, 0u);

        // ---- B gather base byte offsets (precomputed once outside K-loop) ----
        // base_B[nb][nl] = (n_start + nb*16 + nl) * K * sizeof(half)
        //                = row start byte offset of B[n_start+nb*16+nl, 0]
        // In K-loop: byte_off[nl] = base_B[nb][nl] + k_a * sizeof(half)
        //   lsc_gather<uint32_t, 8> (SOA layout):
        //     result[p * 16 + nl] = uint32 at (byte_off[nl] + p * 4)
        //                        = { B[nl, k_a + p*2], B[nl, k_a + p*2 + 1] }
        //   → produces VNNI layout uint32[k_pair * 16 + n_col] directly ✓
        const uint32_t Kbytes = (uint32_t)K * 2u;   // row stride in bytes (fp16)
        const uint32_t* B_u32 = reinterpret_cast<const uint32_t*>(B);

        simd<uint32_t, 16> base_B[4];
        #pragma unroll
        for (int nb = 0; nb < 4; nb++) {
            #pragma unroll
            for (int nl = 0; nl < 16; nl++)
                base_B[nb][nl] = (uint32_t)(n_start + nb * 16 + nl) * Kbytes;
        }

        // ---- Accumulators / tile registers (same layout as nopf) ----
        
        simd<sycl::half, 8*16>  acc[4][4] = {0};
        simd<sycl::half, 8*16>  a_tile[2][4];    // A tiles:  a_tile[buf][m_blk]
        simd<sycl::half, 16*16> b_tile[2][4];    // B tiles:  b_tile[buf][n_blk]  (VNNI)

        // ---- Running k offset (incremented with += only) ----
        // k_a : column byte offset for B gather AND fp16-element index for payA.set_x
        //       incremented by K_SUB * sizeof(half) = 32 per substep
        uint32_t k_a  = 0u;   // fp16 element offset for payA.set_x
        uint32_t k_ab = 0u;   // byte offset for B gather = k_a * 2

        // Helper: load a_tile (A) at current k_a
#define LOAD_A(buf)                                                                \
        payA.set_x(k_a);                                                           \
        _Pragma("unroll")                                                          \
        for (int m = 0; m < 4; m++) {                                             \
            payA.set_y((uint32_t)(m_start + m * 8));                              \
            a_tile[(buf)][m] = xesimd::lsc_load_2d<sycl::half, 16, 8, 1, false, false,\
                xesimd::cache_hint::cached, xesimd::cache_hint::cached>(payA);    \
        }

        // Helper: load b_tile (B VNNI) at current k_ab
        // One lsc_gather<uint32_t, 8> per n_blk (SOA layout):
        //   result[p * 16 + nl] = uint32 at (base_B[nb][nl] + k_ab + p*4)
        //                       = { B[nl, k_a+p*2], B[nl, k_a+p*2+1] }
        //   → direct VNNI uint32[k_pair*16 + n_col] without any repack ✓
#define LOAD_B(buf)                                                                \
        _Pragma("unroll")                                                          \
        for (int nb = 0; nb < 4; nb++) {                                          \
            simd<uint32_t, 16> byte_off = base_B[nb] + k_ab;                      \
            b_tile[(buf)][nb].template bit_cast_view<uint32_t>() =                \
                xesimd::lsc_gather<uint32_t, 8,                                    \
                    xesimd::lsc_data_size::u32,                                    \
                    xesimd::cache_hint::cached,                                    \
                    xesimd::cache_hint::cached,                                    \
                    16, uint32_t>(B_u32, byte_off);                               \
        }

        // Helper: DPAS 4×4 = 16 calls
#define COMPUTE(buf)                                                               \
        _Pragma("unroll")                                                          \
        for (int m = 0; m < 4; m++)                                               \
            _Pragma("unroll")                                                      \
            for (int n = 0; n < 4; n++)                                           \
                acc[m][n] = xmx::dpas<8, 8, sycl::half, sycl::half,              \
                    sycl::half, sycl::half>(acc[m][n], b_tile[(buf)][n], a_tile[(buf)][m]);

        // ------ PROLOGUE: load buf[0] at k_a=0, k_ab=0 ------
        LOAD_A(0)
        LOAD_B(0)

        // ------ MAIN LOOP ------
        // Each substep: k_a += K_SUB (fp16 elements), k_ab += K_SUB*2 (bytes)
        for (int k = 0; k < K - K_STEP; k += K_STEP) {
            // substep 0: advance to k+K_SUB, load buf[1], compute buf[0]
            k_a  += (uint32_t)K_SUB;
            k_ab += (uint32_t)(K_SUB * 2);
            LOAD_A(1)
            LOAD_B(1)
            COMPUTE(0)

            // substep 1: advance to k+K_STEP, load buf[0], compute buf[1]
            k_a  += (uint32_t)K_SUB;
            k_ab += (uint32_t)(K_SUB * 2);
            LOAD_A(0)
            LOAD_B(0)
            COMPUTE(1)
        }

        // ------ LAST K-BLOCK ------
        {
            k_a  += (uint32_t)K_SUB;
            k_ab += (uint32_t)(K_SUB * 2);
            LOAD_A(1)
            LOAD_B(1)
            COMPUTE(0)
            COMPUTE(1)
        }

#undef LOAD_A
#undef LOAD_B
#undef COMPUTE

        // ------ STORE ------
        #pragma unroll
        for (int m = 0; m < 4; m++) {
            #pragma unroll
            for (int n = 0; n < 4; n++) {
                const int m_idx = m_start + m * 8;
                const int n_idx = n_start + n * 16;
                if (m_idx + 8 <= M && n_idx + 16 <= N)
                    xesimd::lsc_store_2d<sycl::half, 16, 8,
                        xesimd::cache_hint::write_back, xesimd::cache_hint::write_back>(
                        C, surfW_C, surfH_A, surfW_C,
                        (uint32_t)n_idx, (uint32_t)m_idx, acc[m][n]);
            }
        }
    }
};

// ---------------------------------------------------------------------------
// CPU reference: C[m,n] = sum_k A[m,k] * B[n,k]   (B is N×K)
// ---------------------------------------------------------------------------
static void cpu_gemm(const std::vector<sycl::half>& A,
                     const std::vector<sycl::half>& B,
                     std::vector<float>& C,
                     int M, int N, int K)
{
    for (int m = 0; m < M; m++)
        for (int n = 0; n < N; n++) {
            float s = 0.f;
            for (int k = 0; k < K; k++)
                s += (float)A[m*K+k] * (float)B[n*K+k];
            C[m*N+n] = s;
        }
}

// ---------------------------------------------------------------------------
int main(int argc, char** argv) {
    int M = 4096, N = 4096, K = 4096;
    if (argc > 1) M = std::atoi(argv[1]);
    if (argc > 2) N = std::atoi(argv[2]);
    if (argc > 3) K = std::atoi(argv[3]);

    printf("FP16 GEMM GATHER  A[%d×%d] × B^T[%d×%d] → C[%d×%d]\n",
           M, K, N, K, M, N);
    printf("B layout: [N×K] row-major, gather VNNI (no transpose needed)\n\n");

    printf(">>> Sleeping 5 seconds <<<\n");
    fflush(stdout);
    std::this_thread::sleep_for(std::chrono::seconds(5));

    try {
        queue q(gpu_selector_v, property::queue::in_order{});
        printf("Device: %s\n\n",
               q.get_device().get_info<sycl::info::device::name>().c_str());

        // ---- Correctness test at small size ----
        {
            constexpr int Mv = 256, Nv = 256, Kv = 256;
            printf("--- Correctness (M=N=K=%d) ---\n", Mv);

            std::mt19937 gen(42);
            std::uniform_real_distribution<float> dis(-0.1f, 0.1f);

            std::vector<sycl::half> h_A(Mv*Kv), h_B(Nv*Kv);
            for (auto& v : h_A) v = sycl::half(dis(gen));
            for (auto& v : h_B) v = sycl::half(dis(gen));

            std::vector<float> h_C_ref(Mv*Nv);
            cpu_gemm(h_A, h_B, h_C_ref, Mv, Nv, Kv);

            auto* d_A = malloc_device<sycl::half>(Mv*Kv, q);
            auto* d_B = malloc_device<sycl::half>(Nv*Kv, q);
            auto* d_C = malloc_device<sycl::half>(Mv*Nv, q);
            q.memcpy(d_A, h_A.data(), Mv*Kv*sizeof(sycl::half)).wait();
            q.memcpy(d_B, h_B.data(), Nv*Kv*sizeof(sycl::half)).wait();

            const int nwg = (Mv/256) * (Nv/256);
            q.submit([&](handler& h) {
                h.parallel_for(nd_range<1>(nwg*32, 32),
                    FP16_GEMM_GATHER_V2{d_A, d_B, d_C, Mv, Nv, Kv});
            }).wait();

            std::vector<sycl::half> h_C_gpu(Mv*Nv);
            q.memcpy(h_C_gpu.data(), d_C, Mv*Nv*sizeof(sycl::half)).wait();

            // Check NaN first
            bool has_nan = false;
            for (int i = 0; i < Mv*Nv; i++)
                if (std::isnan((float)h_C_gpu[i])) { has_nan = true; break; }
            if (has_nan) { printf("FAIL: NaN in output\n"); return 1; }

            // Relative RMS error
            double sum_err2 = 0, sum_ref2 = 0;
            for (int i = 0; i < Mv*Nv; i++) {
                double e = (double)(float)h_C_gpu[i] - (double)h_C_ref[i];
                sum_err2 += e * e;
                sum_ref2 += (double)h_C_ref[i] * h_C_ref[i];
            }
            double rel_rms = (sum_ref2 > 0) ? std::sqrt(sum_err2 / sum_ref2) * 100.0 : 0.0;
            printf("Rel RMS error: %.4f%%  %s\n\n",
                   rel_rms, rel_rms < 0.5 ? "PASS" : "FAIL");

            free(d_A, q); free(d_B, q); free(d_C, q);
            if (rel_rms >= 0.5) return 1;
        }

        // ---- Benchmark ----
        printf("--- Benchmark (M=%d N=%d K=%d) ---\n", M, N, K);

        std::mt19937 gen(99);
        std::uniform_real_distribution<float> dis(-0.01f, 0.01f);
        auto nz = [&](float f){ return f == 0.f ? 1e-4f : f; };

        std::vector<sycl::half> h_A(M*K), h_B(N*K);
        for (auto& v : h_A) v = sycl::half(nz(dis(gen)));
        for (auto& v : h_B) v = sycl::half(nz(dis(gen)));

        auto* d_A = malloc_device<sycl::half>(M*K, q);
        auto* d_B = malloc_device<sycl::half>(N*K, q);
        auto* d_C = malloc_device<sycl::half>(M*N, q);
        q.memcpy(d_A, h_A.data(), M*K*sizeof(sycl::half)).wait();
        q.memcpy(d_B, h_B.data(), N*K*sizeof(sycl::half)).wait();

        const int num_wg_m  = (M + 255) / 256;
        const int num_wg_n  = (N + 255) / 256;
        const int global_sz = num_wg_m * num_wg_n * 32;

        // Warmup
        for (int i = 0; i < 20; i++)
            q.submit([&](handler& h) {
                h.parallel_for(nd_range<1>(global_sz, 32),
                    FP16_GEMM_GATHER_V2{d_A, d_B, d_C, M, N, K});
            });
        q.wait();

        // Timed
        constexpr int ITERS = 1000;
        auto t0 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < ITERS; i++)
            q.submit([&](handler& h) {
                h.parallel_for(nd_range<1>(global_sz, 32),
                    FP16_GEMM_GATHER_V2{d_A, d_B, d_C, M, N, K});
            });
        q.wait();
        double sec = std::chrono::duration<double>(
            std::chrono::high_resolution_clock::now() - t0).count() / ITERS;

        double tflops = 2.0 * M * N * K / sec / 1e12;
        printf("Time/iter : %.4f ms\n", sec * 1e3);
        printf("TFLOPS    : %.2f\n", tflops);
        printf("%% of 135T : %.1f%%\n", tflops / 135.0 * 100.0);

        free(d_A, q); free(d_B, q); free(d_C, q);
        return 0;

    } catch (const sycl::exception& e) {
        fprintf(stderr, "SYCL: %s\n", e.what());
        return 1;
    }
}
