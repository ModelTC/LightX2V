// sdp_perf_all.cpp
// Merged performance test for all SDP kernels:
//   Section 1: mask_convert perf         (6 configs)
//   Section 2: prefill sparse SDP perf   (5 configs x noncausal+causal)
//   Section 3: prefill full SDP perf     (3 configs x noncausal+causal)
//   Section 4: decode full SDP perf      (4 kv_lens)
//   Section 5: decode sparse SDP perf    (4 kv_lens)
//
// GPU: BMG  headQ=32  headKv=2  headDim=128
// Peak BW:  530 GB/s (BMG HBM)
// Peak TF:  135 TFLOPS (FP16)

#include <sycl/sycl.hpp>
#include <sycl/ext/intel/esimd.hpp>
#include <iostream>
#include <iomanip>
#include <vector>
#include <random>
#include <numeric>
#include <algorithm>
#include <chrono>
#include <set>
#include <cassert>

using namespace sycl;
using fp16 = sycl::half;
using fp32 = float;

// Macros needed by prefill kernel headers
#define __ESIMD_NS  sycl::ext::intel::esimd
#define __ESIMD_ENS sycl::ext::intel::experimental::esimd
#undef ESIMD_INLINE
#define ESIMD_INLINE inline __attribute__((always_inline))
#define FP32_MIN -3.402823466e+38f

using namespace sycl::ext::intel::esimd;
using namespace sycl::ext::intel::esimd::xmx;
using namespace sycl::ext::intel::experimental::esimd;

// Kernel headers (order matters)
#include "mask_convert.h"
#include "flash.attn.sparse.mha128.gqa.h"
#include "flash.attn.b.mha128.gqa.precomputed_yuchen.h"
#include "decode_sdp_sparse.h"   // also brings in decode_sdp_gqa.h

// ============================================================
// Global constants
// ============================================================
static const int    WARMUP         = 5;
static const int    ITERS          = 100;
static const double PEAK_BW_GBps   = 530.0;   // BMG memory BW
static const double PEAK_TFLOPS    = 135.0;   // BMG FP16 TFLOPS
static const int    MC_N_BUF       = 4;        // mask_convert: rotate 4 buffers
static const int    DEC_N_BUF      = 4;        // decode: rotate 4 buffers
// prefill: N_BUF=1 (large data already busts L2)

// GPU config
static constexpr int NUM_Q_HEADS  = 32;
static constexpr int NUM_KV_HEADS = 2;
static constexpr int HEAD_DIM     = 128;
static constexpr int KV_BLOCK     = 64;

// Decode kernel config
static constexpr uint32_t DEC_HD               = 128;
static constexpr uint32_t DEC_HEADQ            = 32;
static constexpr uint32_t DEC_HEADKV           = 2;
static constexpr uint32_t DEC_Q_HEAD_PER_T     = 8;
static constexpr uint32_t DEC_SP_BLK_SIZE      = 64;
static constexpr uint32_t DEC_CHUNK_SIZE        = 256;
static constexpr uint32_t DEC_NUM_SPARSE        = 64;
static constexpr uint32_t DEC_CHUNK_NUM_SPARSE = DEC_NUM_SPARSE / (DEC_CHUNK_SIZE / DEC_SP_BLK_SIZE); // 16

// Sparse decode BW bytes (constant regardless of kv_len)
static const double SPARSE_KV_BYTES =
    2.0 * DEC_NUM_SPARSE * DEC_SP_BLK_SIZE * DEC_HEADKV * DEC_HD * sizeof(fp16);

// ============================================================
// Helpers
// ============================================================

static void fill_mask_orig_random(uint32_t* ptr, int num_kv_heads, int qlen,
                                  int total_kv_blocks, unsigned seed)
{
    std::mt19937 rng(seed);
    std::vector<uint32_t> pool((size_t)total_kv_blocks);
    std::iota(pool.begin(), pool.end(), 0u);
    for (int kh = 0; kh < num_kv_heads; kh++) {
        for (int qi = 0; qi < qlen; qi++) {
            std::shuffle(pool.begin(), pool.end(), rng);
            size_t base = (size_t)(kh * qlen + qi) * 64;
            std::copy(pool.begin(), pool.begin() + 64, ptr + base);
            std::sort(ptr + base, ptr + base + 64);
        }
    }
}

static void gen_uniform_sparse_mask(
    std::vector<uint32_t>& mask_flat,
    std::vector<uint32_t>& cnt_flat,
    int num_kv_heads, int q_blocks, int kv_len, int mask_cnt_per_block,
    std::mt19937& gen)
{
    int total_kv_blocks = (kv_len + KV_BLOCK - 1) / KV_BLOCK;
    int actual_cnt = std::min(mask_cnt_per_block, total_kv_blocks);

    mask_flat.assign((size_t)num_kv_heads * q_blocks * 1024, 0u);
    cnt_flat.assign((size_t)num_kv_heads * q_blocks, (uint32_t)actual_cnt);

    for (int kh = 0; kh < num_kv_heads; kh++) {
        for (int qb = 0; qb < q_blocks; qb++) {
            std::set<uint32_t> chosen;
            std::uniform_int_distribution<uint32_t> dist(0, total_kv_blocks - 1);
            while ((int)chosen.size() < actual_cnt) chosen.insert(dist(gen));
            int base = (kh * q_blocks + qb) * 1024;
            int i = 0;
            for (uint32_t idx : chosen) mask_flat[base + i++] = idx;
        }
    }
}

static std::vector<uint32_t> gen_decode_sparse_mask(int kv_len, unsigned seed)
{
    int total_kv_blocks = kv_len / (int)DEC_SP_BLK_SIZE;
    std::mt19937 rng(seed);
    std::vector<uint32_t> mask(DEC_HEADKV * DEC_NUM_SPARSE);
    for (uint32_t kh = 0; kh < DEC_HEADKV; kh++) {
        std::set<uint32_t> chosen;
        std::uniform_int_distribution<uint32_t> dist(0, total_kv_blocks - 1);
        while ((int)chosen.size() < (int)DEC_NUM_SPARSE) chosen.insert(dist(rng));
        int i = 0;
        for (uint32_t idx : chosen) mask[kh * DEC_NUM_SPARSE + i++] = idx;
    }
    return mask;
}

// ============================================================
// Section 1: mask_convert perf
// ============================================================
static void run_section_mask_convert(sycl::queue& q)
{
    std::cout << "============================================================\n";
    std::cout << "Section 1: mask_convert Performance\n";
    std::cout << "  N_BUF=" << MC_N_BUF << "  WARMUP=" << WARMUP << "  ITERS=" << ITERS << "\n";
    std::cout << "  Peak BW: " << PEAK_BW_GBps << " GB/s\n";
    std::cout << "============================================================\n";

    struct MCPerfCfg { int num_kv_heads; int qlen; int kv_len; const char* name; };
    std::vector<MCPerfCfg> configs = {
        {2,  1024,  4096, "kh=2  qlen=1K  kv=4K "},
        {2,  2048,  8192, "kh=2  qlen=2K  kv=8K "},
        {2,  4096, 16384, "kh=2  qlen=4K  kv=16K"},
        {2,  8192, 16384, "kh=2  qlen=8K  kv=16K"},
        {2, 16384, 16384, "kh=2  qlen=16K kv=16K"},
        {2, 32768, 32768, "kh=2  qlen=32K kv=32K"},
    };

    std::cout << std::left  << std::setw(26) << "Config"
              << std::right << std::setw(12) << "Time(us)"
              << std::setw(10) << "GB/s"
              << std::setw(10) << "Roof%"
              << "\n";
    std::cout << std::string(60, '-') << "\n";

    for (auto& cfg : configs) {
        int    q_blocks        = (cfg.qlen + 15) / 16;
        int    total_kv_blocks = (cfg.kv_len + 63) / 64;
        size_t orig_sz         = (size_t)cfg.num_kv_heads * cfg.qlen * 64;
        size_t out_sz          = (size_t)cfg.num_kv_heads * q_blocks * 1024;
        size_t cnt_sz          = (size_t)cfg.num_kv_heads * q_blocks;

        double total_bytes = (double)(orig_sz + out_sz + cnt_sz) * sizeof(uint32_t);

        // Allocate MC_N_BUF buffer sets to rotate
        std::vector<uint32_t*> d_orig(MC_N_BUF), d_out(MC_N_BUF), d_cnt(MC_N_BUF);
        for (int b = 0; b < MC_N_BUF; b++) {
            std::vector<uint32_t> h_orig(orig_sz);
            fill_mask_orig_random(h_orig.data(), cfg.num_kv_heads, cfg.qlen,
                                  total_kv_blocks, (unsigned)(b * 1234 + 42));
            d_orig[b] = sycl::malloc_device<uint32_t>(orig_sz, q);
            d_out[b]  = sycl::malloc_device<uint32_t>(out_sz,  q);
            d_cnt[b]  = sycl::malloc_device<uint32_t>(cnt_sz,  q);
            q.memcpy(d_orig[b], h_orig.data(), orig_sz * sizeof(uint32_t)).wait();
        }

        sycl::nd_range<1> nr({(size_t)(cfg.num_kv_heads * q_blocks)}, {1});

        // Warmup (with .wait() after each)
        for (int i = 0; i < WARMUP; i++) {
            int b = i % MC_N_BUF;
            MaskConvertFunctor f{d_orig[b], d_out[b], d_cnt[b],
                                 (uint32_t)cfg.qlen, (uint32_t)cfg.num_kv_heads,
                                 (uint32_t)total_kv_blocks};
            q.submit([&](sycl::handler& cgh){ cgh.parallel_for(nr, f); }).wait();
        }

        // Timed: submit all, single wait at end (no intermediate waits)
        auto t0 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < ITERS; i++) {
            int b = i % MC_N_BUF;
            MaskConvertFunctor f{d_orig[b], d_out[b], d_cnt[b],
                                 (uint32_t)cfg.qlen, (uint32_t)cfg.num_kv_heads,
                                 (uint32_t)total_kv_blocks};
            q.submit([&](sycl::handler& cgh){ cgh.parallel_for(nr, f); });
        }
        q.wait();
        auto t1 = std::chrono::high_resolution_clock::now();

        double us   = std::chrono::duration<double, std::micro>(t1 - t0).count() / ITERS;
        double gbps = total_bytes / (us * 1e3);
        double roof = gbps / PEAK_BW_GBps * 100.0;

        std::cout << std::left  << std::setw(26) << cfg.name
                  << std::right << std::fixed << std::setprecision(1)
                  << std::setw(12) << us
                  << std::setw(10) << gbps
                  << std::setw(9)  << roof << "%"
                  << "\n";

        for (int b = 0; b < MC_N_BUF; b++) {
            sycl::free(d_orig[b], q);
            sycl::free(d_out[b],  q);
            sycl::free(d_cnt[b],  q);
        }
    }
    std::cout << "\n";
}

// ============================================================
// Section 2: prefill sparse SDP perf
// ============================================================
static void run_section_prefill_sparse(sycl::queue& q)
{
    std::cout << "============================================================\n";
    std::cout << "Section 2: Prefill Sparse SDP Performance\n";
    std::cout << "  mask_cnt=64 (kv_active=64*64=4096 per q_block)\n";
    std::cout << "  WARMUP=" << WARMUP << "  ITERS=" << ITERS << "\n";
    std::cout << "  Peak TF: " << PEAK_TFLOPS << " TFLOPS FP16\n";
    std::cout << "============================================================\n";

    static const int MASK_CNT = 64;

    struct SPerfCfg { int q_len, kv_len; const char* name; };
    std::vector<SPerfCfg> configs = {
        {8192,  8192,  "8K x8K "},
        {8192,  16384, "8K x16K"},
        {8192,  32768, "8K x32K"},
        {16384, 16384, "16Kx16K"},
        {16384, 32768, "16Kx32K"},
    };

    // FLOPS: based on active kv tokens = MASK_CNT * KV_BLOCK (causal doesn't skip DPAS)
    auto compute_flops = [&](int q_len) -> double {
        double kv_active = (double)MASK_CNT * KV_BLOCK;
        return (4.0 * q_len * kv_active * HEAD_DIM
              + 2.0 * q_len * kv_active) * NUM_Q_HEADS;
    };

    std::cout << std::setw(14) << std::left  << "Config"
              << std::setw(11) << std::right << "Time(ms)"
              << std::setw(11) << "TFLOPS"
              << std::setw(11) << "Roofline%"
              << "\n";
    std::cout << "Non-Causal:\n";
    std::cout << std::string(47, '-') << "\n";

    for (auto& c : configs) {
        int    q_blocks    = (c.q_len + 15) / 16;
        size_t q_size      = (size_t)c.q_len  * NUM_Q_HEADS  * HEAD_DIM;
        size_t kv_size     = (size_t)c.kv_len * NUM_KV_HEADS * HEAD_DIM;
        uint32_t hist      = 0;

        std::mt19937 gen(99);
        std::uniform_real_distribution<float> dis(-1.0f, 1.0f);

        std::vector<fp16>  hQ(q_size), hK(kv_size), hV(kv_size);
        std::vector<float> hN(NUM_Q_HEADS * HEAD_DIM, 1.0f);
        for (auto& x : hQ) x = fp16(dis(gen) * 0.1f);
        for (auto& x : hK) x = fp16(dis(gen) * 0.1f);
        for (auto& x : hV) x = fp16(dis(gen) * 0.1f);

        std::vector<uint32_t> mask_flat, cnt_flat;
        gen_uniform_sparse_mask(mask_flat, cnt_flat, NUM_KV_HEADS, q_blocks, c.kv_len, MASK_CNT, gen);

        fp16*     dQ = sycl::malloc_device<fp16>(q_size, q);
        fp16*     dK = sycl::malloc_device<fp16>(kv_size, q);
        fp16*     dV = sycl::malloc_device<fp16>(kv_size, q);
        fp16*     dO = sycl::malloc_device<fp16>(q_size, q);
        float*    dN = sycl::malloc_device<float>(NUM_Q_HEADS * HEAD_DIM, q);
        uint32_t* dM = sycl::malloc_device<uint32_t>(mask_flat.size(), q);
        uint32_t* dC = sycl::malloc_device<uint32_t>(cnt_flat.size(), q);

        q.memcpy(dQ, hQ.data(), q_size  * sizeof(fp16)).wait();
        q.memcpy(dK, hK.data(), kv_size * sizeof(fp16)).wait();
        q.memcpy(dV, hV.data(), kv_size * sizeof(fp16)).wait();
        q.memcpy(dN, hN.data(), NUM_Q_HEADS * HEAD_DIM * sizeof(float)).wait();
        q.memcpy(dM, mask_flat.data(), mask_flat.size() * sizeof(uint32_t)).wait();
        q.memcpy(dC, cnt_flat.data(),  cnt_flat.size()  * sizeof(uint32_t)).wait();

        sycl::nd_range<2> nr({(size_t)q_blocks, (size_t)NUM_KV_HEADS * 16}, {1, 16});

        auto run_nc = [&]() {
            SparseFlashAttnFunctor<false> f{(uint8_t*)dQ, (uint8_t*)dK, (uint8_t*)dV,
                (uint8_t*)dN, (uint8_t*)dO, dM, dC,
                c.q_len, c.kv_len, (int)hist, NUM_Q_HEADS, NUM_KV_HEADS};
            q.submit([&](sycl::handler& h){ h.parallel_for(nr, f); }).wait();
        };

        // Warmup
        for (int i = 0; i < WARMUP; i++) run_nc();

        // Timed (wait after each — large kernels, measure per-iter latency)
        auto t0 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < ITERS; i++) run_nc();
        auto t1 = std::chrono::high_resolution_clock::now();

        double ms    = std::chrono::duration<double, std::milli>(t1 - t0).count() / ITERS;
        double flops = compute_flops(c.q_len);
        double tflops= flops / (ms * 1e9);
        double roof  = tflops / PEAK_TFLOPS * 100.0;

        std::cout << std::setw(14) << std::left << c.name
                  << std::right << std::fixed << std::setprecision(3)
                  << std::setw(11) << ms
                  << std::setw(11) << std::setprecision(2) << tflops
                  << std::setw(10) << std::setprecision(2) << roof << "%\n";

        sycl::free(dQ, q); sycl::free(dK, q); sycl::free(dV, q);
        sycl::free(dO, q); sycl::free(dN, q); sycl::free(dM, q); sycl::free(dC, q);
    }

    std::cout << "\nCausal:\n";
    std::cout << std::string(47, '-') << "\n";

    for (auto& c : configs) {
        int    q_blocks    = (c.q_len + 15) / 16;
        size_t q_size      = (size_t)c.q_len  * NUM_Q_HEADS  * HEAD_DIM;
        size_t kv_size     = (size_t)c.kv_len * NUM_KV_HEADS * HEAD_DIM;
        uint32_t hist      = (c.kv_len > c.q_len) ? (uint32_t)(c.kv_len - c.q_len) : 0u;

        std::mt19937 gen(99);
        std::uniform_real_distribution<float> dis(-1.0f, 1.0f);

        std::vector<fp16>  hQ(q_size), hK(kv_size), hV(kv_size);
        std::vector<float> hN(NUM_Q_HEADS * HEAD_DIM, 1.0f);
        for (auto& x : hQ) x = fp16(dis(gen) * 0.1f);
        for (auto& x : hK) x = fp16(dis(gen) * 0.1f);
        for (auto& x : hV) x = fp16(dis(gen) * 0.1f);

        std::vector<uint32_t> mask_flat, cnt_flat;
        gen_uniform_sparse_mask(mask_flat, cnt_flat, NUM_KV_HEADS, q_blocks, c.kv_len, MASK_CNT, gen);

        fp16*     dQ = sycl::malloc_device<fp16>(q_size, q);
        fp16*     dK = sycl::malloc_device<fp16>(kv_size, q);
        fp16*     dV = sycl::malloc_device<fp16>(kv_size, q);
        fp16*     dO = sycl::malloc_device<fp16>(q_size, q);
        float*    dN = sycl::malloc_device<float>(NUM_Q_HEADS * HEAD_DIM, q);
        uint32_t* dM = sycl::malloc_device<uint32_t>(mask_flat.size(), q);
        uint32_t* dC = sycl::malloc_device<uint32_t>(cnt_flat.size(), q);

        q.memcpy(dQ, hQ.data(), q_size  * sizeof(fp16)).wait();
        q.memcpy(dK, hK.data(), kv_size * sizeof(fp16)).wait();
        q.memcpy(dV, hV.data(), kv_size * sizeof(fp16)).wait();
        q.memcpy(dN, hN.data(), NUM_Q_HEADS * HEAD_DIM * sizeof(float)).wait();
        q.memcpy(dM, mask_flat.data(), mask_flat.size() * sizeof(uint32_t)).wait();
        q.memcpy(dC, cnt_flat.data(),  cnt_flat.size()  * sizeof(uint32_t)).wait();

        sycl::nd_range<2> nr({(size_t)q_blocks, (size_t)NUM_KV_HEADS * 16}, {1, 16});

        auto run_c = [&]() {
            SparseFlashAttnFunctor<true> f{(uint8_t*)dQ, (uint8_t*)dK, (uint8_t*)dV,
                (uint8_t*)dN, (uint8_t*)dO, dM, dC,
                c.q_len, c.kv_len, (int)hist, NUM_Q_HEADS, NUM_KV_HEADS};
            q.submit([&](sycl::handler& h){ h.parallel_for(nr, f); }).wait();
        };

        for (int i = 0; i < WARMUP; i++) run_c();

        auto t0 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < ITERS; i++) run_c();
        auto t1 = std::chrono::high_resolution_clock::now();

        double ms    = std::chrono::duration<double, std::milli>(t1 - t0).count() / ITERS;
        double flops = compute_flops(c.q_len);
        double tflops= flops / (ms * 1e9);
        double roof  = tflops / PEAK_TFLOPS * 100.0;

        std::cout << std::setw(14) << std::left << c.name
                  << std::right << std::fixed << std::setprecision(3)
                  << std::setw(11) << ms
                  << std::setw(11) << std::setprecision(2) << tflops
                  << std::setw(10) << std::setprecision(2) << roof << "%\n";

        sycl::free(dQ, q); sycl::free(dK, q); sycl::free(dV, q);
        sycl::free(dO, q); sycl::free(dN, q); sycl::free(dM, q); sycl::free(dC, q);
    }
    std::cout << "\n";
}

// ============================================================
// Section 3: prefill full SDP perf
// ============================================================
static void run_section_prefill_full(sycl::queue& q)
{
    std::cout << "============================================================\n";
    std::cout << "Section 3: Prefill Full SDP Performance\n";
    std::cout << "  WARMUP=" << WARMUP << "  ITERS=" << ITERS << "\n";
    std::cout << "  Peak TF: " << PEAK_TFLOPS << " TFLOPS FP16\n";
    std::cout << "============================================================\n";

    struct PFPerfCfg { int q_len, kv_len; const char* name; };
    std::vector<PFPerfCfg> configs = {
        {8192,  8192,  "8Kx8K"},
        {8192,  16384, "8Kx16K"},
        {16384, 16384, "16Kx16K"},
    };

    std::cout << std::setw(14) << std::left  << "Config"
              << std::setw(11) << std::right << "Time(ms)"
              << std::setw(11) << "TFLOPS"
              << std::setw(11) << "Roofline%"
              << "\n";
    std::cout << "Non-Causal:\n";
    std::cout << std::string(47, '-') << "\n";

    for (auto& c : configs) {
        int    q_blocks = (c.q_len + 15) / 16;
        size_t q_size   = (size_t)c.q_len  * NUM_Q_HEADS  * HEAD_DIM;
        size_t kv_size  = (size_t)c.kv_len * NUM_KV_HEADS * HEAD_DIM;

        std::mt19937 gen(12345);
        std::uniform_real_distribution<float> dis(-1.0f, 1.0f);

        std::vector<fp16>  hQ(q_size), hK(kv_size), hV(kv_size);
        std::vector<float> hN(NUM_Q_HEADS * HEAD_DIM, 1.0f);
        for (auto& x : hQ) x = fp16(dis(gen) * 0.1f);
        for (auto& x : hK) x = fp16(dis(gen) * 0.1f);
        for (auto& x : hV) x = fp16(dis(gen) * 0.1f);

        fp16*  dQ = sycl::malloc_device<fp16>(q_size, q);
        fp16*  dK = sycl::malloc_device<fp16>(kv_size, q);
        fp16*  dV = sycl::malloc_device<fp16>(kv_size, q);
        fp16*  dO = sycl::malloc_device<fp16>(q_size, q);
        float* dN = sycl::malloc_device<float>(NUM_Q_HEADS * HEAD_DIM, q);

        q.memcpy(dQ, hQ.data(), q_size  * sizeof(fp16)).wait();
        q.memcpy(dK, hK.data(), kv_size * sizeof(fp16)).wait();
        q.memcpy(dV, hV.data(), kv_size * sizeof(fp16)).wait();
        q.memcpy(dN, hN.data(), NUM_Q_HEADS * HEAD_DIM * sizeof(float)).wait();

        sycl::nd_range<2> nr({(size_t)q_blocks, (size_t)NUM_KV_HEADS * 16}, {1, 16});

        auto run_nc = [&]() {
            FlashAttnFunctor<false> f{(uint8_t*)dQ, (uint8_t*)dK, (uint8_t*)dV,
                (uint8_t*)dN, (uint8_t*)dO,
                (int)c.q_len, (int)c.kv_len, 0, NUM_Q_HEADS, NUM_KV_HEADS};
            q.submit([&](sycl::handler& h){ h.parallel_for(nr, f); }).wait();
        };

        for (int i = 0; i < WARMUP; i++) run_nc();

        auto t0 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < ITERS; i++) run_nc();
        auto t1 = std::chrono::high_resolution_clock::now();

        double ms    = std::chrono::duration<double, std::milli>(t1 - t0).count() / ITERS;
        double flops = (4.0 * c.q_len * c.kv_len * HEAD_DIM
                      + 2.0 * c.q_len * c.kv_len) * NUM_Q_HEADS;
        double tflops= flops / (ms * 1e9);
        double roof  = tflops / PEAK_TFLOPS * 100.0;

        std::cout << std::setw(14) << std::left << c.name
                  << std::right << std::fixed << std::setprecision(3)
                  << std::setw(11) << ms
                  << std::setw(11) << std::setprecision(2) << tflops
                  << std::setw(10) << std::setprecision(2) << roof << "%\n";

        sycl::free(dQ, q); sycl::free(dK, q); sycl::free(dV, q);
        sycl::free(dO, q); sycl::free(dN, q);
    }

    std::cout << "\nCausal:\n";
    std::cout << std::string(47, '-') << "\n";

    for (auto& c : configs) {
        int    q_blocks = (c.q_len + 15) / 16;
        size_t q_size   = (size_t)c.q_len  * NUM_Q_HEADS  * HEAD_DIM;
        size_t kv_size  = (size_t)c.kv_len * NUM_KV_HEADS * HEAD_DIM;
        int    hist     = (c.kv_len > c.q_len) ? c.kv_len - c.q_len : 0;

        std::mt19937 gen(12345);
        std::uniform_real_distribution<float> dis(-1.0f, 1.0f);

        std::vector<fp16>  hQ(q_size), hK(kv_size), hV(kv_size);
        std::vector<float> hN(NUM_Q_HEADS * HEAD_DIM, 1.0f);
        for (auto& x : hQ) x = fp16(dis(gen) * 0.1f);
        for (auto& x : hK) x = fp16(dis(gen) * 0.1f);
        for (auto& x : hV) x = fp16(dis(gen) * 0.1f);

        fp16*  dQ = sycl::malloc_device<fp16>(q_size, q);
        fp16*  dK = sycl::malloc_device<fp16>(kv_size, q);
        fp16*  dV = sycl::malloc_device<fp16>(kv_size, q);
        fp16*  dO = sycl::malloc_device<fp16>(q_size, q);
        float* dN = sycl::malloc_device<float>(NUM_Q_HEADS * HEAD_DIM, q);

        q.memcpy(dQ, hQ.data(), q_size  * sizeof(fp16)).wait();
        q.memcpy(dK, hK.data(), kv_size * sizeof(fp16)).wait();
        q.memcpy(dV, hV.data(), kv_size * sizeof(fp16)).wait();
        q.memcpy(dN, hN.data(), NUM_Q_HEADS * HEAD_DIM * sizeof(float)).wait();

        sycl::nd_range<2> nr({(size_t)q_blocks, (size_t)NUM_KV_HEADS * 16}, {1, 16});

        auto run_c = [&]() {
            FlashAttnFunctor<true> f{(uint8_t*)dQ, (uint8_t*)dK, (uint8_t*)dV,
                (uint8_t*)dN, (uint8_t*)dO,
                (int)c.q_len, (int)c.kv_len, (int)hist, NUM_Q_HEADS, NUM_KV_HEADS};
            q.submit([&](sycl::handler& h){ h.parallel_for(nr, f); }).wait();
        };

        for (int i = 0; i < WARMUP; i++) run_c();

        auto t0 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < ITERS; i++) run_c();
        auto t1 = std::chrono::high_resolution_clock::now();

        double ms    = std::chrono::duration<double, std::milli>(t1 - t0).count() / ITERS;
        // Causal: apply causal factor (average active fraction)
        double causal_pct = (c.q_len == c.kv_len)
            ? 0.5  // square: exactly half active on average
            : ((double)c.q_len * c.kv_len - 0.5 * c.q_len * c.q_len)
              / ((double)c.q_len * c.kv_len);
        double flops = (4.0 * c.q_len * c.kv_len * HEAD_DIM
                      + 2.0 * c.q_len * c.kv_len) * NUM_Q_HEADS * causal_pct;
        double tflops= flops / (ms * 1e9);
        double roof  = tflops / PEAK_TFLOPS * 100.0;

        std::cout << std::setw(14) << std::left << c.name
                  << std::right << std::fixed << std::setprecision(3)
                  << std::setw(11) << ms
                  << std::setw(11) << std::setprecision(2) << tflops
                  << std::setw(10) << std::setprecision(2) << roof << "%\n";

        sycl::free(dQ, q); sycl::free(dK, q); sycl::free(dV, q);
        sycl::free(dO, q); sycl::free(dN, q);
    }
    std::cout << "\n";
}

// ============================================================
// Section 4: decode full SDP perf
// ============================================================
static void run_section_decode_full(sycl::queue& q)
{
    std::cout << "============================================================\n";
    std::cout << "Section 4: Decode Full SDP Performance\n";
    std::cout << "  N_BUF=" << DEC_N_BUF << "  WARMUP=" << WARMUP << "  ITERS=" << ITERS << "\n";
    std::cout << "  Peak BW: " << PEAK_BW_GBps << " GB/s\n";
    std::cout << "  Config: headQ=" << DEC_HEADQ << " headKv=" << DEC_HEADKV
              << " HD=" << DEC_HD << " chunk=" << DEC_CHUNK_SIZE << "\n";
    std::cout << "============================================================\n";

    std::vector<int> kv_lens = {4096, 16384, 32768, 65536};

    std::cout << std::left  << std::setw(10) << "kv_len"
              << std::right << std::setw(10) << "chunks"
              << std::setw(12) << "Time(us)"
              << std::setw(10) << "GB/s"
              << std::setw(9)  << "Roof%"
              << std::setw(16) << "KV_bytes(MB)"
              << "\n";
    std::cout << std::string(70, '-') << "\n";

    for (int kv_len : kv_lens) {
        int    chunk_num     = kv_len / (int)DEC_CHUNK_SIZE;
        size_t Q_elems       = (size_t)DEC_HEADQ * DEC_HD;
        size_t KV_elems      = (size_t)kv_len * DEC_HEADKV * DEC_HD;
        size_t partial_elems = (size_t)chunk_num * DEC_HEADQ * DEC_HD;
        size_t lse_max_elems = (size_t)chunk_num * DEC_HEADQ;
        size_t out_elems     = (size_t)DEC_HEADQ * DEC_HD;

        double bw_bytes =
            (double)KV_elems      * 2 * sizeof(fp16)      // K + V read
          + (double)Q_elems        * sizeof(fp16)           // Q read (small)
          + (double)partial_elems  * sizeof(float) * 2      // partial write+read in reduce
          + (double)lse_max_elems  * sizeof(float) * 2 * 2  // lse+max write+read
          + (double)out_elems      * sizeof(fp16);           // final out write

        float attscale = 1.0f / std::sqrt((float)DEC_HD) * 0.1f;

        struct BufSet {
            fp16*  dQ; fp16*  dK; fp16*  dV;
            float* dPartial; float* dLse; float* dMax;
            fp16*  dOut;
        };

        std::vector<BufSet> bufs(DEC_N_BUF);
        for (int b = 0; b < DEC_N_BUF; b++) {
            bufs[b].dQ       = sycl::malloc_device<fp16>(Q_elems, q);
            bufs[b].dK       = sycl::malloc_device<fp16>(KV_elems, q);
            bufs[b].dV       = sycl::malloc_device<fp16>(KV_elems, q);
            bufs[b].dPartial = sycl::malloc_device<float>(partial_elems, q);
            bufs[b].dLse     = sycl::malloc_device<float>(lse_max_elems, q);
            bufs[b].dMax     = sycl::malloc_device<float>(lse_max_elems, q);
            bufs[b].dOut     = sycl::malloc_device<fp16>(out_elems, q);

            std::mt19937 rng((unsigned)(b * 1111 + 42));
            std::uniform_real_distribution<float> dis(-0.1f, 0.1f);
            std::vector<fp16> hQ(Q_elems), hKV(KV_elems);
            for (auto& x : hQ)  x = (fp16)dis(rng);
            for (auto& x : hKV) x = (fp16)dis(rng);
            q.memcpy(bufs[b].dQ, hQ.data(),  Q_elems  * sizeof(fp16)).wait();
            q.memcpy(bufs[b].dK, hKV.data(), KV_elems * sizeof(fp16)).wait();
            q.memcpy(bufs[b].dV, hKV.data(), KV_elems * sizeof(fp16)).wait();
        }

        // Warmup (no intermediate waits inside loop)
        for (int i = 0; i < WARMUP; i++) {
            int b = i % DEC_N_BUF;
            flashAttnMha128GQAdecode<DEC_HD, DEC_HEADQ, DEC_HEADKV,
                                     DEC_Q_HEAD_PER_T, DEC_SP_BLK_SIZE, DEC_CHUNK_SIZE>(
                bufs[b].dQ, bufs[b].dK, bufs[b].dV,
                bufs[b].dPartial, bufs[b].dLse, bufs[b].dMax,
                (uint32_t)kv_len, attscale, q);
            flashAttnMha128GQADecodeReduce<DEC_HD, DEC_HEADQ>(
                bufs[b].dPartial, bufs[b].dLse, bufs[b].dMax,
                bufs[b].dOut, (uint32_t)chunk_num, q);
            q.wait();
        }

        // Timed: submit all, single wait at end
        auto t0 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < ITERS; i++) {
            int b = i % DEC_N_BUF;
            flashAttnMha128GQAdecode<DEC_HD, DEC_HEADQ, DEC_HEADKV,
                                     DEC_Q_HEAD_PER_T, DEC_SP_BLK_SIZE, DEC_CHUNK_SIZE>(
                bufs[b].dQ, bufs[b].dK, bufs[b].dV,
                bufs[b].dPartial, bufs[b].dLse, bufs[b].dMax,
                (uint32_t)kv_len, attscale, q);
            flashAttnMha128GQADecodeReduce<DEC_HD, DEC_HEADQ>(
                bufs[b].dPartial, bufs[b].dLse, bufs[b].dMax,
                bufs[b].dOut, (uint32_t)chunk_num, q);
        }
        q.wait();
        auto t1 = std::chrono::high_resolution_clock::now();

        double us   = std::chrono::duration<double, std::micro>(t1 - t0).count() / ITERS;
        double gbps = bw_bytes / (us * 1e3);
        double roof = gbps / PEAK_BW_GBps * 100.0;
        double kv_mb= (double)KV_elems * 2 * sizeof(fp16) / (1024.0 * 1024.0);

        std::cout << std::left  << std::setw(10) << kv_len
                  << std::right << std::fixed << std::setprecision(1)
                  << std::setw(10) << chunk_num
                  << std::setw(12) << us
                  << std::setw(10) << gbps
                  << std::setw(8)  << roof << "%"
                  << std::setw(12) << kv_mb << " MB"
                  << "\n";

        for (int b = 0; b < DEC_N_BUF; b++) {
            sycl::free(bufs[b].dQ, q); sycl::free(bufs[b].dK, q);
            sycl::free(bufs[b].dV, q); sycl::free(bufs[b].dPartial, q);
            sycl::free(bufs[b].dLse, q); sycl::free(bufs[b].dMax, q);
            sycl::free(bufs[b].dOut, q);
        }
    }
    std::cout << "\n";
}

// ============================================================
// Section 5: decode sparse SDP perf
// ============================================================
static void run_section_decode_sparse(sycl::queue& q)
{
    std::cout << "============================================================\n";
    std::cout << "Section 5: Decode Sparse SDP Performance\n";
    std::cout << "  N_BUF=" << DEC_N_BUF << "  WARMUP=" << WARMUP << "  ITERS=" << ITERS << "\n";
    std::cout << "  Peak BW: " << PEAK_BW_GBps << " GB/s\n";
    std::cout << "  Sparse: num_sparse_blocks=" << DEC_NUM_SPARSE
              << "  sp_blk_size=" << DEC_SP_BLK_SIZE
              << "  chunk_size=" << DEC_CHUNK_SIZE
              << "  chunk_num=" << DEC_CHUNK_NUM_SPARSE << "\n";
    std::cout << "  Sparse KV tokens per head: " << DEC_NUM_SPARSE * DEC_SP_BLK_SIZE
              << "  (" << SPARSE_KV_BYTES / (1024.0 * 1024.0) << " MB K+V always)\n";
    std::cout << "============================================================\n";

    std::vector<int> kv_lens = {4096, 16384, 32768, 65536};

    std::cout << std::left  << std::setw(10) << "kv_len"
              << std::right << std::setw(9)  << "sparsity"
              << std::setw(12) << "Time(us)"
              << std::setw(10) << "GB/s"
              << std::setw(9)  << "Roof%"
              << std::setw(14) << "dense_est(us)"
              << "\n";
    std::cout << std::string(65, '-') << "\n";

    for (int kv_len : kv_lens) {
        float  attscale       = 1.0f / std::sqrt((float)DEC_HD) * 0.1f;
        double dense_kv_bytes = 2.0 * kv_len * DEC_HEADKV * DEC_HD * sizeof(fp16);

        struct SparseBufSet {
            fp16*     dQ; fp16*     dK; fp16*     dV;
            uint32_t* dMask;
            float*    dPartial; float* dLse; float* dMax;
            fp16*     dOut;
        };

        std::vector<SparseBufSet> bufs(DEC_N_BUF);
        for (int b = 0; b < DEC_N_BUF; b++) {
            bufs[b].dQ       = sycl::malloc_device<fp16>(DEC_HEADQ * DEC_HD, q);
            bufs[b].dK       = sycl::malloc_device<fp16>((size_t)kv_len * DEC_HEADKV * DEC_HD, q);
            bufs[b].dV       = sycl::malloc_device<fp16>((size_t)kv_len * DEC_HEADKV * DEC_HD, q);
            bufs[b].dMask    = sycl::malloc_device<uint32_t>(DEC_HEADKV * DEC_NUM_SPARSE, q);
            bufs[b].dPartial = sycl::malloc_device<float>((size_t)DEC_CHUNK_NUM_SPARSE * DEC_HEADQ * DEC_HD, q);
            bufs[b].dLse     = sycl::malloc_device<float>((size_t)DEC_CHUNK_NUM_SPARSE * DEC_HEADQ, q);
            bufs[b].dMax     = sycl::malloc_device<float>((size_t)DEC_CHUNK_NUM_SPARSE * DEC_HEADQ, q);
            bufs[b].dOut     = sycl::malloc_device<fp16>(DEC_HEADQ * DEC_HD, q);

            // Random K/V to defeat HW memory compression
            std::mt19937 rng((unsigned)(b * 1111 + 42));
            std::uniform_real_distribution<float> dis(-0.1f, 0.1f);
            std::vector<fp16> hQ(DEC_HEADQ * DEC_HD),
                              hKV((size_t)kv_len * DEC_HEADKV * DEC_HD);
            for (auto& x : hQ)  x = (fp16)dis(rng);
            for (auto& x : hKV) x = (fp16)dis(rng);
            q.memcpy(bufs[b].dQ, hQ.data(),  DEC_HEADQ * DEC_HD * sizeof(fp16)).wait();
            q.memcpy(bufs[b].dK, hKV.data(), (size_t)kv_len * DEC_HEADKV * DEC_HD * sizeof(fp16)).wait();
            q.memcpy(bufs[b].dV, hKV.data(), (size_t)kv_len * DEC_HEADKV * DEC_HD * sizeof(fp16)).wait();

            std::vector<uint32_t> hMask = gen_decode_sparse_mask(kv_len, (unsigned)(b * 31 + 7));
            q.memcpy(bufs[b].dMask, hMask.data(),
                     DEC_HEADKV * DEC_NUM_SPARSE * sizeof(uint32_t)).wait();
        }

        // Warmup (with wait after each)
        for (int i = 0; i < WARMUP; i++) {
            int b = i % DEC_N_BUF;
            flashAttnMha128GQAdecodeSparse<DEC_HD, DEC_HEADQ, DEC_HEADKV,
                                           DEC_Q_HEAD_PER_T, DEC_SP_BLK_SIZE,
                                           DEC_NUM_SPARSE, DEC_CHUNK_SIZE>(
                bufs[b].dQ, bufs[b].dK, bufs[b].dV, bufs[b].dMask,
                bufs[b].dPartial, bufs[b].dLse, bufs[b].dMax, attscale, (uint32_t)kv_len, q);
            flashAttnMha128GQADecodeReduce<DEC_HD, DEC_HEADQ>(
                bufs[b].dPartial, bufs[b].dLse, bufs[b].dMax,
                bufs[b].dOut, DEC_CHUNK_NUM_SPARSE, q);
            q.wait();
        }

        // Timed: submit all, single wait at end
        auto t0 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < ITERS; i++) {
            int b = i % DEC_N_BUF;
            flashAttnMha128GQAdecodeSparse<DEC_HD, DEC_HEADQ, DEC_HEADKV,
                                           DEC_Q_HEAD_PER_T, DEC_SP_BLK_SIZE,
                                           DEC_NUM_SPARSE, DEC_CHUNK_SIZE>(
                bufs[b].dQ, bufs[b].dK, bufs[b].dV, bufs[b].dMask,
                bufs[b].dPartial, bufs[b].dLse, bufs[b].dMax, attscale, (uint32_t)kv_len, q);
            flashAttnMha128GQADecodeReduce<DEC_HD, DEC_HEADQ>(
                bufs[b].dPartial, bufs[b].dLse, bufs[b].dMax,
                bufs[b].dOut, DEC_CHUNK_NUM_SPARSE, q);
        }
        q.wait();
        auto t1 = std::chrono::high_resolution_clock::now();

        double us       = std::chrono::duration<double, std::micro>(t1 - t0).count() / ITERS;
        double gbps     = SPARSE_KV_BYTES / (us * 1e3);
        double roof     = gbps / PEAK_BW_GBps * 100.0;
        double dense_us = dense_kv_bytes / (SPARSE_KV_BYTES / us);
        double sparsity = 100.0 * DEC_NUM_SPARSE * DEC_SP_BLK_SIZE / kv_len;

        std::cout << std::left  << std::setw(10) << kv_len
                  << std::right << std::fixed << std::setprecision(1)
                  << std::setw(8) << sparsity << "%"
                  << std::setw(12) << us
                  << std::setw(10) << gbps
                  << std::setw(8)  << roof << "%"
                  << "   (dense~" << std::setprecision(0) << dense_us << "us)"
                  << "\n";

        for (int b = 0; b < DEC_N_BUF; b++) {
            sycl::free(bufs[b].dQ,       q); sycl::free(bufs[b].dK,       q);
            sycl::free(bufs[b].dV,       q); sycl::free(bufs[b].dMask,    q);
            sycl::free(bufs[b].dPartial, q); sycl::free(bufs[b].dLse,     q);
            sycl::free(bufs[b].dMax,     q); sycl::free(bufs[b].dOut,     q);
        }
    }
    std::cout << "\n";
}

// ============================================================
// main
// ============================================================
int main()
{
    sycl::queue q(sycl::gpu_selector_v);
    std::cout << "============================================================\n";
    std::cout << "SDP Performance All — merged performance suite\n";
    std::cout << "============================================================\n";
    std::cout << "Device:    " << q.get_device().get_info<sycl::info::device::name>() << "\n";
    std::cout << "GQA:       Q_heads=" << NUM_Q_HEADS << "  KV_heads=" << NUM_KV_HEADS
              << "  head_dim=" << HEAD_DIM << "\n";
    std::cout << "Peak BW:   " << PEAK_BW_GBps << " GB/s\n";
    std::cout << "Peak TFLOPS: " << PEAK_TFLOPS << " FP16\n";
    std::cout << "WARMUP=" << WARMUP << "  ITERS=" << ITERS << "\n\n";

    run_section_mask_convert(q);
    run_section_prefill_sparse(q);
    run_section_prefill_full(q);
    run_section_decode_full(q);
    run_section_decode_sparse(q);

    std::cout << "============================================================\n";
    std::cout << "All sections complete.\n";
    std::cout << "============================================================\n";
    return 0;
}
