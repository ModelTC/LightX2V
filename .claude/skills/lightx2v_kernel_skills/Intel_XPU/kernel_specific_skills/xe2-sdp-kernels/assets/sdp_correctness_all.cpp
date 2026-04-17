// sdp_correctness_all.cpp
// Merged correctness test for all SDP kernels:
//   Section 1: mask_convert                         (16 test cases)
//   Section 2: prefill sparse SDP                   (25 test cases)
//   Section 3: prefill full SDP                     (4 test cases)
//   Section 4: decode full SDP                      (10 test cases)
//   Section 5: decode sparse SDP                    (10 test cases)
//   Section 6: mask_convert + prefill sparse SDP    (8 test cases)
//              (end-to-end pipeline: mask_orig -> GPU mask_convert -> GPU sparse SDP)
//
// GPU: BMG  headQ=32  headKv=2  headDim=128

#include <sycl/sycl.hpp>
#include <sycl/ext/intel/esimd.hpp>
#include <iostream>
#include <iomanip>
#include <vector>
#include <random>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <set>
#include <cassert>
#include <string>

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
// GPU / kernel constants
// ============================================================
static constexpr int NUM_Q_HEADS  = 32;
static constexpr int NUM_KV_HEADS = 2;
static constexpr int HEAD_DIM     = 128;
static constexpr int KV_BLOCK     = 64;

// Best decode config
static constexpr uint32_t DEC_HD               = 128;
static constexpr uint32_t DEC_HEADQ            = 32;
static constexpr uint32_t DEC_HEADKV           = 2;
static constexpr uint32_t DEC_Q_HEAD_PER_T     = 8;
static constexpr uint32_t DEC_SP_BLK_SIZE      = 64;
static constexpr uint32_t DEC_CHUNK_SIZE        = 256;
static constexpr uint32_t DEC_NUM_SPARSE        = 64;
// chunk_num for sparse decode = num_sparse_blocks / (chunk_size / sp_blk_size) = 64 / 4 = 16
static constexpr uint32_t DEC_CHUNK_NUM_SPARSE = DEC_NUM_SPARSE / (DEC_CHUNK_SIZE / DEC_SP_BLK_SIZE);

// ============================================================
// SECTION 1: mask_convert
// ============================================================

static void mc_cpu_mask_convert(
    const std::vector<uint32_t>& orig,
    std::vector<uint32_t>& out, std::vector<uint32_t>& cnt_out,
    int num_kv_heads, int qlen)
{
    int q_blocks = (qlen + 15) / 16;
    out.assign((size_t)num_kv_heads * q_blocks * 1024, 0u);
    cnt_out.assign((size_t)num_kv_heads * q_blocks, 0u);
    for (int kh = 0; kh < num_kv_heads; kh++) {
        for (int qb = 0; qb < q_blocks; qb++) {
            std::set<uint32_t> u;
            int q_start = qb * 16, q_end = std::min(q_start + 16, qlen);
            for (int qi = q_start; qi < q_end; qi++) {
                size_t base = (size_t)(kh * qlen + qi) * 64;
                for (int i = 0; i < 64; i++) u.insert(orig[base + i]);
            }
            int flat = kh * q_blocks + qb;
            uint32_t c = 0;
            for (uint32_t v : u) out[(size_t)flat * 1024 + c++] = v;
            cnt_out[flat] = c;
        }
    }
}

static void mc_gen_orig(std::vector<uint32_t>& orig, int num_kv_heads, int qlen,
                        int total_kv_blocks, std::mt19937& rng)
{
    assert(total_kv_blocks >= 1);
    orig.resize((size_t)num_kv_heads * qlen * 64);
    if (total_kv_blocks >= 64) {
        // Normal case: pick 64 unique sorted random indices from [0, total_kv_blocks)
        std::vector<uint32_t> pool(total_kv_blocks);
        std::iota(pool.begin(), pool.end(), 0u);
        for (int kh = 0; kh < num_kv_heads; kh++) {
            for (int qi = 0; qi < qlen; qi++) {
                std::shuffle(pool.begin(), pool.end(), rng);
                std::vector<uint32_t> chosen(pool.begin(), pool.begin() + 64);
                std::sort(chosen.begin(), chosen.end());
                size_t base = (size_t)(kh * qlen + qi) * 64;
                for (int i = 0; i < 64; i++) orig[base + i] = chosen[i];
            }
        }
    } else {
        // Small kv: total_kv_blocks < 64. Repeat valid indices 0..N-1 to fill 64 slots.
        // MaskConvertFunctor deduplicates via bitmap → output has total_kv_blocks unique entries.
        for (int kh = 0; kh < num_kv_heads; kh++) {
            for (int qi = 0; qi < qlen; qi++) {
                size_t base = (size_t)(kh * qlen + qi) * 64;
                for (int i = 0; i < 64; i++)
                    orig[base + i] = (uint32_t)(i % total_kv_blocks);
            }
        }
    }
}

struct MCfg { int qlen; int kv_len; unsigned seed; const char* name; };

static bool mc_run_test(sycl::queue& q, const MCfg& c)
{
    int total_kv   = c.kv_len / 64;
    int q_blocks   = (c.qlen + 15) / 16;
    size_t orig_sz = (size_t)NUM_KV_HEADS * c.qlen * 64;
    size_t out_sz  = (size_t)NUM_KV_HEADS * q_blocks * 1024;
    size_t cnt_sz  = (size_t)NUM_KV_HEADS * q_blocks;

    std::mt19937 rng(c.seed);
    std::vector<uint32_t> h_orig;
    mc_gen_orig(h_orig, NUM_KV_HEADS, c.qlen, total_kv, rng);

    std::vector<uint32_t> cpu_out, cpu_cnt;
    mc_cpu_mask_convert(h_orig, cpu_out, cpu_cnt, NUM_KV_HEADS, c.qlen);

    uint32_t* d_orig = sycl::malloc_device<uint32_t>(orig_sz, q);
    uint32_t* d_out  = sycl::malloc_device<uint32_t>(out_sz,  q);
    uint32_t* d_cnt  = sycl::malloc_device<uint32_t>(cnt_sz,  q);
    q.memcpy(d_orig, h_orig.data(), orig_sz * sizeof(uint32_t)).wait();

    MaskConvertFunctor f{d_orig, d_out, d_cnt,
                         (uint32_t)c.qlen, (uint32_t)NUM_KV_HEADS, (uint32_t)total_kv};
    q.submit([&](sycl::handler& h){
        h.parallel_for(sycl::nd_range<1>({(size_t)(NUM_KV_HEADS * q_blocks)}, {1}), f);
    }).wait();

    std::vector<uint32_t> gpu_out(out_sz), gpu_cnt(cnt_sz);
    q.memcpy(gpu_out.data(), d_out, out_sz * sizeof(uint32_t)).wait();
    q.memcpy(gpu_cnt.data(), d_cnt, cnt_sz * sizeof(uint32_t)).wait();
    sycl::free(d_orig, q); sycl::free(d_out, q); sycl::free(d_cnt, q);

    int cnt_fail = 0, ent_fail = 0;
    for (int kh = 0; kh < NUM_KV_HEADS; kh++) {
        for (int qb = 0; qb < q_blocks; qb++) {
            int flat = kh * q_blocks + qb;
            if (gpu_cnt[flat] != cpu_cnt[flat]) { cnt_fail++; continue; }
            for (uint32_t i = 0; i < cpu_cnt[flat]; i++) {
                if (gpu_out[(size_t)flat * 1024 + i] != cpu_out[(size_t)flat * 1024 + i])
                    ent_fail++;
            }
        }
    }
    bool pass = (cnt_fail == 0 && ent_fail == 0);

    double avg_union = 0;
    for (auto v : cpu_cnt) avg_union += v;
    avg_union /= (double)cpu_cnt.size();

    std::cout << "  " << std::setw(44) << std::left << c.name
              << std::right
              << "  avg_union=" << std::fixed << std::setprecision(1) << avg_union
              << "  " << (pass ? "PASSED" : "FAILED");
    if (!pass) std::cout << " (cnt_fail=" << cnt_fail << " ent_fail=" << ent_fail << ")";
    std::cout << "\n";
    return pass;
}

// ============================================================
// SECTION 2: prefill sparse SDP
// ============================================================

static void ps_cpu_noncausal(
    const std::vector<fp16>& Q, const std::vector<fp16>& K, const std::vector<fp16>& V,
    std::vector<fp16>& O, int q_len, int kv_len,
    const std::vector<std::vector<std::vector<uint32_t>>>& mask,
    const std::vector<std::vector<uint32_t>>& cnt)
{
    const float sc = 1.0f / std::sqrt((float)HEAD_DIM);
    const int gs   = NUM_Q_HEADS / NUM_KV_HEADS;
    for (int qh = 0; qh < NUM_Q_HEADS; qh++) {
        int kh = qh / gs;
        for (int qi = 0; qi < q_len; qi++) {
            int qb = qi / 16;
            std::vector<float> s(kv_len, -1e38f);
            for (uint32_t bi = 0; bi < cnt[kh][qb]; bi++) {
                int ks = mask[kh][qb][bi] * KV_BLOCK;
                int ke = std::min(ks + KV_BLOCK, kv_len);
                for (int k = ks; k < ke; k++) {
                    float v = 0;
                    for (int d = 0; d < HEAD_DIM; d++)
                        v += (float)Q[qi * NUM_Q_HEADS * HEAD_DIM + qh * HEAD_DIM + d]
                           * (float)K[k  * NUM_KV_HEADS * HEAD_DIM + kh * HEAD_DIM + d];
                    s[k] = v * sc;
                }
            }
            float mx = *std::max_element(s.begin(), s.end());
            float sum = 0;
            for (auto& x : s) { x = std::exp(x - mx); sum += x; }
            for (int d = 0; d < HEAD_DIM; d++) {
                float o = 0;
                for (int k = 0; k < kv_len; k++)
                    o += s[k] / sum * (float)V[k * NUM_KV_HEADS * HEAD_DIM + kh * HEAD_DIM + d];
                O[qi * NUM_Q_HEADS * HEAD_DIM + qh * HEAD_DIM + d] = fp16(o);
            }
        }
    }
}

static void ps_cpu_causal(
    const std::vector<fp16>& Q, const std::vector<fp16>& K, const std::vector<fp16>& V,
    std::vector<fp16>& O, int q_len, int kv_len,
    const std::vector<std::vector<std::vector<uint32_t>>>& mask,
    const std::vector<std::vector<uint32_t>>& cnt)
{
    const float sc   = 1.0f / std::sqrt((float)HEAD_DIM);
    const int gs     = NUM_Q_HEADS / NUM_KV_HEADS;
    const int hist   = (kv_len > q_len) ? kv_len - q_len : 0;
    for (int qh = 0; qh < NUM_Q_HEADS; qh++) {
        int kh = qh / gs;
        for (int qi = 0; qi < q_len; qi++) {
            int qb  = qi / 16;
            int bnd = hist + qi;
            std::vector<int> pos;
            for (uint32_t bi = 0; bi < cnt[kh][qb]; bi++) {
                int ks = mask[kh][qb][bi] * KV_BLOCK;
                int ke = std::min(ks + KV_BLOCK, kv_len);
                for (int k = ks; k < ke; k++)
                    if (k <= bnd) pos.push_back(k);
            }
            if (pos.empty()) {
                for (int d = 0; d < HEAD_DIM; d++)
                    O[qi * NUM_Q_HEADS * HEAD_DIM + qh * HEAD_DIM + d] = fp16(0.0f);
                continue;
            }
            std::vector<float> s(pos.size()); float mx = -1e38f;
            for (int i = 0; i < (int)pos.size(); i++) {
                float v = 0;
                for (int d = 0; d < HEAD_DIM; d++)
                    v += (float)Q[qi  * NUM_Q_HEADS  * HEAD_DIM + qh * HEAD_DIM + d]
                       * (float)K[pos[i] * NUM_KV_HEADS * HEAD_DIM + kh * HEAD_DIM + d];
                s[i] = v * sc; mx = std::max(mx, s[i]);
            }
            float sum = 0;
            for (auto& x : s) { x = std::exp(x - mx); sum += x; }
            for (int d = 0; d < HEAD_DIM; d++) {
                float o = 0;
                for (int i = 0; i < (int)pos.size(); i++)
                    o += s[i] / sum * (float)V[pos[i] * NUM_KV_HEADS * HEAD_DIM + kh * HEAD_DIM + d];
                O[qi * NUM_Q_HEADS * HEAD_DIM + qh * HEAD_DIM + d] = fp16(o);
            }
        }
    }
}

static void ps_gen_mask(std::vector<std::vector<std::vector<uint32_t>>>& sparse_mask,
                        std::vector<std::vector<uint32_t>>& sparse_cnt,
                        int q_blocks, int kv_len, int mask_cnt, std::mt19937& gen)
{
    int total_kv = (kv_len + KV_BLOCK - 1) / KV_BLOCK;
    sparse_mask.assign(NUM_KV_HEADS, std::vector<std::vector<uint32_t>>(q_blocks));
    sparse_cnt.assign(NUM_KV_HEADS, std::vector<uint32_t>(q_blocks));
    for (int kh = 0; kh < NUM_KV_HEADS; kh++) {
        for (int qb = 0; qb < q_blocks; qb++) {
            int c = std::min(mask_cnt, total_kv);
            std::set<uint32_t> chosen;
            std::uniform_int_distribution<uint32_t> dist(0, total_kv - 1);
            while ((int)chosen.size() < c) chosen.insert(dist(gen));
            sparse_mask[kh][qb] = {chosen.begin(), chosen.end()};
            sparse_cnt[kh][qb]  = c;
        }
    }
}

static void ps_flatten_mask(const std::vector<std::vector<std::vector<uint32_t>>>& sm,
                            const std::vector<std::vector<uint32_t>>& sc,
                            std::vector<uint32_t>& mf, std::vector<uint32_t>& cf,
                            int q_blocks)
{
    mf.assign((size_t)NUM_KV_HEADS * q_blocks * 1024, 0);
    cf.assign((size_t)NUM_KV_HEADS * q_blocks, 0);
    for (int kh = 0; kh < NUM_KV_HEADS; kh++)
        for (int qb = 0; qb < q_blocks; qb++) {
            int flat  = kh * q_blocks + qb;
            cf[flat]  = sc[kh][qb];
            for (uint32_t i = 0; i < sc[kh][qb]; i++)
                mf[(size_t)flat * 1024 + i] = sm[kh][qb][i];
        }
}

struct SCfg { int q_len, kv_len, mask_cnt; bool causal; float scale; unsigned seed; const char* name; };

static bool ps_run_test(sycl::queue& q, const SCfg& c)
{
    int q_blocks = (c.q_len + 15) / 16;
    size_t qs  = (size_t)c.q_len  * NUM_Q_HEADS  * HEAD_DIM;
    size_t kvs = (size_t)c.kv_len * NUM_KV_HEADS * HEAD_DIM;
    int hist   = (c.causal && c.kv_len > c.q_len) ? c.kv_len - c.q_len : 0;

    std::mt19937 gen(c.seed);
    std::uniform_real_distribution<float> dis(-c.scale, c.scale);

    std::vector<fp16>  hQ(qs), hK(kvs), hV(kvs), hOg(qs), hOc(qs);
    std::vector<float> hN(NUM_Q_HEADS * HEAD_DIM, 1.0f);
    for (auto& x : hQ) x = fp16(dis(gen));
    for (auto& x : hK) x = fp16(dis(gen));
    for (auto& x : hV) x = fp16(dis(gen));

    std::vector<std::vector<std::vector<uint32_t>>> sm;
    std::vector<std::vector<uint32_t>> sc;
    ps_gen_mask(sm, sc, q_blocks, c.kv_len, c.mask_cnt, gen);
    std::vector<uint32_t> mf, cf;
    ps_flatten_mask(sm, sc, mf, cf, q_blocks);

    fp16*     dQ = sycl::malloc_device<fp16>(qs, q);
    fp16*     dK = sycl::malloc_device<fp16>(kvs, q);
    fp16*     dV = sycl::malloc_device<fp16>(kvs, q);
    fp16*     dO = sycl::malloc_device<fp16>(qs, q);
    float*    dN = sycl::malloc_device<float>(NUM_Q_HEADS * HEAD_DIM, q);
    uint32_t* dM = sycl::malloc_device<uint32_t>(mf.size(), q);
    uint32_t* dC = sycl::malloc_device<uint32_t>(cf.size(), q);

    q.memcpy(dQ, hQ.data(), qs  * sizeof(fp16)).wait();
    q.memcpy(dK, hK.data(), kvs * sizeof(fp16)).wait();
    q.memcpy(dV, hV.data(), kvs * sizeof(fp16)).wait();
    q.memcpy(dN, hN.data(), NUM_Q_HEADS * HEAD_DIM * sizeof(float)).wait();
    q.memcpy(dM, mf.data(), mf.size() * sizeof(uint32_t)).wait();
    q.memcpy(dC, cf.data(), cf.size() * sizeof(uint32_t)).wait();

    sycl::nd_range<2> nr({(size_t)q_blocks, (size_t)NUM_KV_HEADS * 16}, {1, 16});
    if (c.causal) {
        SparseFlashAttnFunctor<true> f{(uint8_t*)dQ, (uint8_t*)dK, (uint8_t*)dV,
            (uint8_t*)dN, (uint8_t*)dO, dM, dC,
            c.q_len, c.kv_len, hist, NUM_Q_HEADS, NUM_KV_HEADS};
        q.submit([&](sycl::handler& h){ h.parallel_for(nr, f); }).wait();
    } else {
        SparseFlashAttnFunctor<false> f{(uint8_t*)dQ, (uint8_t*)dK, (uint8_t*)dV,
            (uint8_t*)dN, (uint8_t*)dO, dM, dC,
            c.q_len, c.kv_len, hist, NUM_Q_HEADS, NUM_KV_HEADS};
        q.submit([&](sycl::handler& h){ h.parallel_for(nr, f); }).wait();
    }
    q.memcpy(hOg.data(), dO, qs * sizeof(fp16)).wait();

    if (c.causal) ps_cpu_causal   (hQ, hK, hV, hOc, c.q_len, c.kv_len, sm, sc);
    else          ps_cpu_noncausal(hQ, hK, hV, hOc, c.q_len, c.kv_len, sm, sc);

    float mx = 0, avg = 0; int nan_c = 0, large_c = 0, valid = 0;
    for (size_t i = 0; i < qs; i++) {
        float g = (float)hOg[i], r = (float)hOc[i];
        if (std::isnan(g) || std::isnan(r)) { nan_c++; continue; }
        float d = std::abs(g - r); mx = std::max(mx, d); avg += d;
        if (d > 0.5f) large_c++; valid++;
    }
    if (valid) avg /= valid;
    float thresh = (c.scale <= 0.1f) ? 0.1f : c.scale * 1.5f;
    bool pass = mx < thresh && large_c == 0 && nan_c == 0;

    std::cout << "  [" << (c.causal ? "causal   " : "noncausal") << "] "
              << std::setw(24) << std::left << c.name << std::right
              << "  max=" << std::scientific << std::setprecision(2) << mx
              << " avg=" << avg
              << " NaN=" << nan_c << " large=" << large_c
              << "  " << (pass ? "PASSED" : "FAILED") << "\n";

    sycl::free(dQ, q); sycl::free(dK, q); sycl::free(dV, q);
    sycl::free(dO, q); sycl::free(dN, q); sycl::free(dM, q); sycl::free(dC, q);
    return pass;
}

// ============================================================
// SECTION 3: prefill full SDP
// ============================================================

static void pf_cpu_noncausal(
    const std::vector<fp16>& Q, const std::vector<fp16>& K, const std::vector<fp16>& V,
    std::vector<fp16>& O, int q_len, int kv_len, int num_q_heads, int num_kv_heads, int head_dim)
{
    const float scale      = 1.0f / std::sqrt((float)head_dim);
    const int   group_size = num_q_heads / num_kv_heads;
    for (int qh = 0; qh < num_q_heads; qh++) {
        int kh = qh / group_size;
        for (int qi = 0; qi < q_len; qi++) {
            std::vector<float> scores(kv_len);
            float max_score = -1e38f;
            for (int k = 0; k < kv_len; k++) {
                float score = 0;
                for (int d = 0; d < head_dim; d++) {
                    score += (float)Q[qi * num_q_heads * head_dim + qh * head_dim + d]
                           * (float)K[k  * num_kv_heads * head_dim + kh * head_dim + d];
                }
                score *= scale; scores[k] = score;
                max_score = std::max(max_score, score);
            }
            float sum = 0;
            for (int k = 0; k < kv_len; k++) { scores[k] = std::exp(scores[k] - max_score); sum += scores[k]; }
            for (int d = 0; d < head_dim; d++) {
                float out_val = 0;
                for (int k = 0; k < kv_len; k++)
                    out_val += (scores[k] / sum) * (float)V[k * num_kv_heads * head_dim + kh * head_dim + d];
                O[qi * num_q_heads * head_dim + qh * head_dim + d] = fp16(out_val);
            }
        }
    }
}

static void pf_cpu_causal(
    const std::vector<fp16>& Q, const std::vector<fp16>& K, const std::vector<fp16>& V,
    std::vector<fp16>& O, int q_len, int kv_len, int num_q_heads, int num_kv_heads, int head_dim)
{
    const float scale        = 1.0f / std::sqrt((float)head_dim);
    const int   history_len  = (kv_len > q_len) ? (kv_len - q_len) : 0;
    const int   group_size   = num_q_heads / num_kv_heads;
    for (int qh = 0; qh < num_q_heads; qh++) {
        int kh = qh / group_size;
        for (int qi = 0; qi < q_len; qi++) {
            int causal_boundary = history_len + qi;
            std::vector<float> scores(kv_len);
            float max_score = -1e38f;
            for (int k = 0; k < kv_len; k++) {
                float score;
                if (k > causal_boundary) {
                    score = -1e38f;
                } else {
                    score = 0;
                    for (int d = 0; d < head_dim; d++)
                        score += (float)Q[qi * num_q_heads * head_dim + qh * head_dim + d]
                               * (float)K[k  * num_kv_heads * head_dim + kh * head_dim + d];
                    score *= scale;
                }
                scores[k] = score; max_score = std::max(max_score, score);
            }
            float sum = 0;
            for (int k = 0; k < kv_len; k++) { scores[k] = std::exp(scores[k] - max_score); sum += scores[k]; }
            for (int d = 0; d < head_dim; d++) {
                float out_val = 0;
                for (int k = 0; k < kv_len; k++)
                    out_val += (scores[k] / sum) * (float)V[k * num_kv_heads * head_dim + kh * head_dim + d];
                O[qi * num_q_heads * head_dim + qh * head_dim + d] = fp16(out_val);
            }
        }
    }
}

static bool pf_run_test(sycl::queue& q, int q_len, int kv_len, bool is_causal, const char* name)
{
    size_t   q_size  = (size_t)q_len  * NUM_Q_HEADS  * HEAD_DIM;
    size_t   kv_size = (size_t)kv_len * NUM_KV_HEADS * HEAD_DIM;
    uint32_t hist    = (is_causal && kv_len > q_len) ? (uint32_t)(kv_len - q_len) : 0u;

    std::vector<fp16>  hQ(q_size), hK(kv_size), hV(kv_size);
    std::vector<fp16>  hO_gpu(q_size), hO_cpu(q_size);
    std::vector<float> hN(NUM_Q_HEADS * HEAD_DIM, 1.0f);

    std::mt19937 gen(12345);
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
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

    int groupH = (q_len + 15) / 16;
    sycl::nd_range<2> nr({(size_t)groupH, (size_t)NUM_KV_HEADS * 16}, {1, 16});

    if (is_causal) {
        FlashAttnFunctor<true> f{(uint8_t*)dQ, (uint8_t*)dK, (uint8_t*)dV,
            (uint8_t*)dN, (uint8_t*)dO,
            (int)q_len, (int)kv_len, (int)hist, (int)NUM_Q_HEADS, (int)NUM_KV_HEADS};
        q.submit([&](sycl::handler& h){ h.parallel_for(nr, f); }).wait();
    } else {
        FlashAttnFunctor<false> f{(uint8_t*)dQ, (uint8_t*)dK, (uint8_t*)dV,
            (uint8_t*)dN, (uint8_t*)dO,
            (int)q_len, (int)kv_len, (int)hist, (int)NUM_Q_HEADS, (int)NUM_KV_HEADS};
        q.submit([&](sycl::handler& h){ h.parallel_for(nr, f); }).wait();
    }
    q.memcpy(hO_gpu.data(), dO, q_size * sizeof(fp16)).wait();

    if (is_causal) pf_cpu_causal   (hQ, hK, hV, hO_cpu, q_len, kv_len, NUM_Q_HEADS, NUM_KV_HEADS, HEAD_DIM);
    else           pf_cpu_noncausal(hQ, hK, hV, hO_cpu, q_len, kv_len, NUM_Q_HEADS, NUM_KV_HEADS, HEAD_DIM);

    float max_diff = 0, avg_diff = 0;
    int   large_errors = 0;
    for (size_t i = 0; i < q_size; i++) {
        float diff = std::abs((float)hO_gpu[i] - (float)hO_cpu[i]);
        max_diff = std::max(max_diff, diff);
        avg_diff += diff;
        if (diff > 0.5f) large_errors++;
    }
    avg_diff /= (float)q_size;
    bool pass = (max_diff < 0.1f && large_errors == 0);

    std::cout << "  [" << (is_causal ? "causal   " : "noncausal") << "] "
              << std::setw(24) << std::left << name << std::right
              << "  max=" << std::scientific << std::setprecision(2) << max_diff
              << "  avg=" << avg_diff
              << "  large=" << large_errors
              << "  " << (pass ? "PASSED" : "FAILED") << "\n";

    sycl::free(dQ, q); sycl::free(dK, q); sycl::free(dV, q);
    sycl::free(dO, q); sycl::free(dN, q);
    return pass;
}

// ============================================================
// SECTION 4: decode full SDP
// ============================================================

static void df_cpu_decode(
    const std::vector<fp16>& Q,
    const std::vector<fp16>& K,
    const std::vector<fp16>& V,
    std::vector<fp16>&       out,
    int kv_len, float attscale)
{
    const int group_size = DEC_HEADQ / DEC_HEADKV;
    for (int qh = 0; qh < (int)DEC_HEADQ; qh++) {
        int kh = qh / group_size;
        std::vector<float> scores(kv_len);
        float max_s = -1e38f;
        for (int kv = 0; kv < kv_len; kv++) {
            float s = 0;
            for (int d = 0; d < (int)DEC_HD; d++)
                s += (float)Q[qh * DEC_HD + d] * (float)K[(kv * DEC_HEADKV + kh) * DEC_HD + d];
            s *= attscale; scores[kv] = s;
            max_s = std::max(max_s, s);
        }
        float sum = 0;
        for (int kv = 0; kv < kv_len; kv++) { scores[kv] = std::exp(scores[kv] - max_s); sum += scores[kv]; }
        for (int kv = 0; kv < kv_len; kv++) scores[kv] /= sum;
        std::vector<float> o(DEC_HD, 0.f);
        for (int kv = 0; kv < kv_len; kv++)
            for (int d = 0; d < (int)DEC_HD; d++)
                o[d] += scores[kv] * (float)V[(kv * DEC_HEADKV + kh) * DEC_HD + d];
        for (int d = 0; d < (int)DEC_HD; d++) out[qh * DEC_HD + d] = (fp16)o[d];
    }
}

struct DFCfg { int kv_len; float scale; unsigned seed; const char* name; };

static bool df_run_test(sycl::queue& q, const DFCfg& c)
{
    std::cout << "  [" << c.name << "]\n";

    // Ceiling division: handles kv_len not a multiple of chunk_size
    int   chunk_num = ((int)c.kv_len + (int)DEC_CHUNK_SIZE - 1) / (int)DEC_CHUNK_SIZE;
    float attscale  = 1.0f / std::sqrt((float)DEC_HD) * c.scale;

    std::mt19937 rng(c.seed);
    std::uniform_real_distribution<float> dis(-c.scale, c.scale);

    std::vector<fp16> hQ(DEC_HEADQ * DEC_HD), hK((size_t)c.kv_len * DEC_HEADKV * DEC_HD),
                      hV((size_t)c.kv_len * DEC_HEADKV * DEC_HD);
    for (auto& x : hQ) x = (fp16)dis(rng);
    for (auto& x : hK) x = (fp16)dis(rng);
    for (auto& x : hV) x = (fp16)dis(rng);

    std::vector<fp16> cpu_out(DEC_HEADQ * DEC_HD);
    df_cpu_decode(hQ, hK, hV, cpu_out, c.kv_len, attscale);

    fp16*  dQ  = sycl::malloc_device<fp16>(DEC_HEADQ * DEC_HD, q);
    fp16*  dK  = sycl::malloc_device<fp16>((size_t)c.kv_len * DEC_HEADKV * DEC_HD, q);
    fp16*  dV  = sycl::malloc_device<fp16>((size_t)c.kv_len * DEC_HEADKV * DEC_HD, q);
    fp16*  dOut= sycl::malloc_device<fp16>(DEC_HEADQ * DEC_HD, q);
    float* dPartial = sycl::malloc_device<float>((size_t)chunk_num * DEC_HEADQ * DEC_HD, q);
    float* dLse     = sycl::malloc_device<float>((size_t)chunk_num * DEC_HEADQ, q);
    float* dMax     = sycl::malloc_device<float>((size_t)chunk_num * DEC_HEADQ, q);

    q.memcpy(dQ, hQ.data(), DEC_HEADQ * DEC_HD * sizeof(fp16)).wait();
    q.memcpy(dK, hK.data(), (size_t)c.kv_len * DEC_HEADKV * DEC_HD * sizeof(fp16)).wait();
    q.memcpy(dV, hV.data(), (size_t)c.kv_len * DEC_HEADKV * DEC_HD * sizeof(fp16)).wait();

    flashAttnMha128GQAdecode<DEC_HD, DEC_HEADQ, DEC_HEADKV, DEC_Q_HEAD_PER_T,
                             DEC_SP_BLK_SIZE, DEC_CHUNK_SIZE>(
        dQ, dK, dV, dPartial, dLse, dMax, (uint32_t)c.kv_len, attscale, q);
    q.wait();

    flashAttnMha128GQADecodeReduce<DEC_HD, DEC_HEADQ>(
        dPartial, dLse, dMax, dOut, (uint32_t)chunk_num, q);
    q.wait();

    std::vector<fp16> gpu_out(DEC_HEADQ * DEC_HD);
    q.memcpy(gpu_out.data(), dOut, DEC_HEADQ * DEC_HD * sizeof(fp16)).wait();

    float max_diff = 0, avg_diff = 0;
    int   nan_cnt = 0, large_cnt = 0;
    float thresh = (c.scale <= 0.1f) ? 0.1f : c.scale * 1.5f;
    for (size_t i = 0; i < gpu_out.size(); i++) {
        if (std::isnan((float)gpu_out[i])) { nan_cnt++; continue; }
        float d = std::abs((float)gpu_out[i] - (float)cpu_out[i]);
        max_diff = std::max(max_diff, d); avg_diff += d;
        if (d > thresh) large_cnt++;
    }
    avg_diff /= (float)gpu_out.size();
    bool ok = (nan_cnt == 0 && large_cnt == 0);

    std::cout << "    decode+reduce"
              << "  max=" << std::scientific << std::setprecision(2) << max_diff
              << "  avg=" << avg_diff
              << "  NaN=" << nan_cnt << "  large=" << large_cnt
              << "  " << (ok ? "PASSED" : "FAILED") << "\n";

    sycl::free(dQ, q); sycl::free(dK, q); sycl::free(dV, q); sycl::free(dOut, q);
    sycl::free(dPartial, q); sycl::free(dLse, q); sycl::free(dMax, q);
    return ok;
}

// ============================================================
// SECTION 5: decode sparse SDP
// ============================================================

static std::vector<uint32_t> ds_gen_sparse_mask(int kv_len, unsigned seed)
{
    int total_kv_blocks = kv_len / (int)DEC_SP_BLK_SIZE;
    assert(total_kv_blocks >= (int)DEC_NUM_SPARSE);
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

static void ds_cpu_sparse_decode(
    const std::vector<fp16>&     Q,
    const std::vector<fp16>&     K,
    const std::vector<fp16>&     V,
    const std::vector<uint32_t>& mask,
    std::vector<fp16>&           out,
    int kv_len, float attscale)
{
    const int group_size = DEC_HEADQ / DEC_HEADKV;
    for (int qh = 0; qh < (int)DEC_HEADQ; qh++) {
        int kh = qh / group_size;
        std::vector<int> toks;
        toks.reserve(DEC_NUM_SPARSE * DEC_SP_BLK_SIZE);
        for (int sb = 0; sb < (int)DEC_NUM_SPARSE; sb++) {
            uint32_t blk_id = mask[kh * DEC_NUM_SPARSE + sb];
            for (int t = 0; t < (int)DEC_SP_BLK_SIZE; t++) {
                int tok = (int)(blk_id * DEC_SP_BLK_SIZE) + t;
                if (tok < kv_len) toks.push_back(tok);
            }
        }
        int N = (int)toks.size();
        std::vector<float> scores(N);
        float max_s = -1e38f;
        for (int i = 0; i < N; i++) {
            int kv = toks[i]; float s = 0;
            for (int d = 0; d < (int)DEC_HD; d++)
                s += (float)Q[qh * DEC_HD + d] * (float)K[(kv * DEC_HEADKV + kh) * DEC_HD + d];
            s *= attscale; scores[i] = s;
            if (s > max_s) max_s = s;
        }
        float sum = 0;
        for (int i = 0; i < N; i++) { scores[i] = std::exp(scores[i] - max_s); sum += scores[i]; }
        for (int i = 0; i < N; i++) scores[i] /= sum;
        std::vector<float> o(DEC_HD, 0.f);
        for (int i = 0; i < N; i++) {
            int kv = toks[i];
            for (int d = 0; d < (int)DEC_HD; d++)
                o[d] += scores[i] * (float)V[(kv * DEC_HEADKV + kh) * DEC_HD + d];
        }
        for (int d = 0; d < (int)DEC_HD; d++) out[qh * DEC_HD + d] = (fp16)o[d];
    }
}

// Generate sparse mask forcing the last (partially valid) kv_block to be included.
// Uses ceiling total_kv_blocks so the partial block is selectable.
static std::vector<uint32_t> ds_gen_mask_partial_last(int kv_len, unsigned seed)
{
    int ceil_total = ((int)kv_len + (int)DEC_SP_BLK_SIZE - 1) / (int)DEC_SP_BLK_SIZE;
    assert(ceil_total >= (int)DEC_NUM_SPARSE &&
           "kv_len too small: need at least DEC_NUM_SPARSE blocks even with partial");
    int last_blk = ceil_total - 1;  // this block is partial (some tokens > kv_len)

    std::mt19937 rng(seed);
    std::vector<uint32_t> mask(DEC_HEADKV * DEC_NUM_SPARSE);
    for (uint32_t kh = 0; kh < DEC_HEADKV; kh++) {
        // Pick DEC_NUM_SPARSE-1 unique blocks from [0, last_blk) then add last_blk
        std::set<uint32_t> chosen;
        std::uniform_int_distribution<uint32_t> dist(0, (uint32_t)(last_blk - 1));
        while ((int)chosen.size() < (int)DEC_NUM_SPARSE - 1) chosen.insert(dist(rng));
        chosen.insert((uint32_t)last_blk);
        int i = 0;
        for (uint32_t idx : chosen) mask[kh * DEC_NUM_SPARSE + i++] = idx;
    }
    return mask;
}

struct DSCfg { int kv_len; float scale; unsigned seed; bool partial_last; const char* name; };

static bool ds_run_test(sycl::queue& q, const DSCfg& c)
{
    std::cout << "  [" << c.name << "]\n";
    // kv_len may be non-multiple of sp_blk_size; integer-division gives total_kv_blocks,
    // and all selected sparse blocks fully lie within kv_len (no OOB).
    assert(c.kv_len >= (int)(DEC_NUM_SPARSE * DEC_SP_BLK_SIZE));

    float attscale = 1.0f / std::sqrt((float)DEC_HD) * c.scale;

    std::mt19937 rng(c.seed);
    std::uniform_real_distribution<float> dis(-c.scale, c.scale);

    std::vector<fp16> hQ(DEC_HEADQ * DEC_HD),
                      hK((size_t)c.kv_len * DEC_HEADKV * DEC_HD),
                      hV((size_t)c.kv_len * DEC_HEADKV * DEC_HD);
    for (auto& x : hQ) x = (fp16)dis(rng);
    for (auto& x : hK) x = (fp16)dis(rng);
    for (auto& x : hV) x = (fp16)dis(rng);

    std::vector<uint32_t> hMask = c.partial_last
        ? ds_gen_mask_partial_last(c.kv_len, c.seed + 7)
        : ds_gen_sparse_mask(c.kv_len, c.seed + 7);

    std::vector<fp16> cpu_out(DEC_HEADQ * DEC_HD);
    ds_cpu_sparse_decode(hQ, hK, hV, hMask, cpu_out, c.kv_len, attscale);

    fp16*     dQ    = sycl::malloc_device<fp16>(DEC_HEADQ * DEC_HD, q);
    fp16*     dK    = sycl::malloc_device<fp16>((size_t)c.kv_len * DEC_HEADKV * DEC_HD, q);
    fp16*     dV    = sycl::malloc_device<fp16>((size_t)c.kv_len * DEC_HEADKV * DEC_HD, q);
    uint32_t* dMask = sycl::malloc_device<uint32_t>(DEC_HEADKV * DEC_NUM_SPARSE, q);
    fp16*     dOut  = sycl::malloc_device<fp16>(DEC_HEADQ * DEC_HD, q);
    float*    dP    = sycl::malloc_device<float>((size_t)DEC_CHUNK_NUM_SPARSE * DEC_HEADQ * DEC_HD, q);
    float*    dLse  = sycl::malloc_device<float>((size_t)DEC_CHUNK_NUM_SPARSE * DEC_HEADQ, q);
    float*    dMax  = sycl::malloc_device<float>((size_t)DEC_CHUNK_NUM_SPARSE * DEC_HEADQ, q);

    q.memcpy(dQ,    hQ.data(),    DEC_HEADQ * DEC_HD * sizeof(fp16)).wait();
    q.memcpy(dK,    hK.data(),    (size_t)c.kv_len * DEC_HEADKV * DEC_HD * sizeof(fp16)).wait();
    q.memcpy(dV,    hV.data(),    (size_t)c.kv_len * DEC_HEADKV * DEC_HD * sizeof(fp16)).wait();
    q.memcpy(dMask, hMask.data(), DEC_HEADKV * DEC_NUM_SPARSE * sizeof(uint32_t)).wait();

    flashAttnMha128GQAdecodeSparse<DEC_HD, DEC_HEADQ, DEC_HEADKV, DEC_Q_HEAD_PER_T,
                                   DEC_SP_BLK_SIZE, DEC_NUM_SPARSE, DEC_CHUNK_SIZE>(
        dQ, dK, dV, dMask, dP, dLse, dMax, attscale, (uint32_t)c.kv_len, q);
    q.wait();

    flashAttnMha128GQADecodeReduce<DEC_HD, DEC_HEADQ>(
        dP, dLse, dMax, dOut, DEC_CHUNK_NUM_SPARSE, q);
    q.wait();

    std::vector<fp16> gpu_out(DEC_HEADQ * DEC_HD);
    q.memcpy(gpu_out.data(), dOut, DEC_HEADQ * DEC_HD * sizeof(fp16)).wait();

    float max_diff = 0, avg_diff = 0;
    int   nan_cnt = 0, large_cnt = 0;
    float thresh = (c.scale <= 0.1f) ? 0.1f : c.scale * 1.5f;
    for (size_t i = 0; i < gpu_out.size(); i++) {
        if (std::isnan((float)gpu_out[i])) { nan_cnt++; continue; }
        float d = std::abs((float)gpu_out[i] - (float)cpu_out[i]);
        if (d > max_diff) max_diff = d; avg_diff += d;
        if (d > thresh) large_cnt++;
    }
    avg_diff /= (float)gpu_out.size();
    bool ok = (nan_cnt == 0 && large_cnt == 0);

    std::cout << "    sparse decode+reduce"
              << "  max=" << std::scientific << std::setprecision(2) << max_diff
              << "  avg=" << avg_diff
              << "  NaN=" << nan_cnt << "  large=" << large_cnt
              << "  " << (ok ? "PASSED" : "FAILED") << "\n";

    sycl::free(dQ, q); sycl::free(dK, q); sycl::free(dV, q);
    sycl::free(dMask, q); sycl::free(dOut, q);
    sycl::free(dP, q); sycl::free(dLse, q); sycl::free(dMax, q);
    return ok;
}

// ============================================================
// SECTION 6: mask_convert + prefill sparse SDP (end-to-end pipeline)
// ============================================================

struct IPCfg { int q_len, kv_len; bool causal; float scale; unsigned seed; const char* name; };

static bool ip_run_test(sycl::queue& q, const IPCfg& c)
{
    // Use ceiling so partial last block is selectable (prefill kernel handles boundary via kvSeqLen mask)
    int total_kv = (c.kv_len + KV_BLOCK - 1) / KV_BLOCK;
    assert(total_kv >= 64 && "kv_len must yield >= 64 kv_blocks for mc_gen_orig");
    int q_blocks = (c.q_len + 15) / 16;
    int hist = (c.causal && c.kv_len > c.q_len) ? c.kv_len - c.q_len : 0;

    size_t orig_sz = (size_t)NUM_KV_HEADS * c.q_len * 64;
    size_t out_sz  = (size_t)NUM_KV_HEADS * q_blocks * 1024;
    size_t cnt_sz  = (size_t)NUM_KV_HEADS * q_blocks;
    size_t qs      = (size_t)c.q_len  * NUM_Q_HEADS  * HEAD_DIM;
    size_t kvs     = (size_t)c.kv_len * NUM_KV_HEADS * HEAD_DIM;

    std::mt19937 rng(c.seed);
    std::uniform_real_distribution<float> dis(-c.scale, c.scale);

    // Generate mask_orig [headKv, q_len, 64]
    std::vector<uint32_t> h_orig;
    mc_gen_orig(h_orig, NUM_KV_HEADS, c.q_len, total_kv, rng);

    // Generate Q/K/V
    std::vector<fp16>  hQ(qs), hK(kvs), hV(kvs);
    std::vector<float> hN(NUM_Q_HEADS * HEAD_DIM, 1.0f);
    for (auto& x : hQ) x = fp16(dis(rng));
    for (auto& x : hK) x = fp16(dis(rng));
    for (auto& x : hV) x = fp16(dis(rng));

    // CPU step 1: mask_convert
    std::vector<uint32_t> cpu_mask_out, cpu_mask_cnt;
    mc_cpu_mask_convert(h_orig, cpu_mask_out, cpu_mask_cnt, NUM_KV_HEADS, c.q_len);

    // Convert flat cpu mask to nested format for ps_cpu_* functions
    std::vector<std::vector<std::vector<uint32_t>>> sm(NUM_KV_HEADS,
        std::vector<std::vector<uint32_t>>(q_blocks));
    std::vector<std::vector<uint32_t>> sc(NUM_KV_HEADS, std::vector<uint32_t>(q_blocks));
    for (int kh = 0; kh < NUM_KV_HEADS; kh++) {
        for (int qb = 0; qb < q_blocks; qb++) {
            int flat      = kh * q_blocks + qb;
            uint32_t cnt  = cpu_mask_cnt[flat];
            sc[kh][qb]    = cnt;
            sm[kh][qb].resize(cnt);
            for (uint32_t i = 0; i < cnt; i++)
                sm[kh][qb][i] = cpu_mask_out[(size_t)flat * 1024 + i];
        }
    }

    // CPU step 2: sparse SDP
    std::vector<fp16> cpu_out(qs);
    if (c.causal) ps_cpu_causal   (hQ, hK, hV, cpu_out, c.q_len, c.kv_len, sm, sc);
    else          ps_cpu_noncausal(hQ, hK, hV, cpu_out, c.q_len, c.kv_len, sm, sc);

    // GPU allocations
    uint32_t* d_orig     = sycl::malloc_device<uint32_t>(orig_sz, q);
    uint32_t* d_mask_out = sycl::malloc_device<uint32_t>(out_sz,  q);
    uint32_t* d_mask_cnt = sycl::malloc_device<uint32_t>(cnt_sz,  q);
    fp16*     dQ         = sycl::malloc_device<fp16>(qs, q);
    fp16*     dK         = sycl::malloc_device<fp16>(kvs, q);
    fp16*     dV         = sycl::malloc_device<fp16>(kvs, q);
    fp16*     dO         = sycl::malloc_device<fp16>(qs, q);
    float*    dN         = sycl::malloc_device<float>(NUM_Q_HEADS * HEAD_DIM, q);

    q.memcpy(d_orig, h_orig.data(), orig_sz * sizeof(uint32_t)).wait();
    q.memcpy(dQ,     hQ.data(),     qs  * sizeof(fp16)).wait();
    q.memcpy(dK,     hK.data(),     kvs * sizeof(fp16)).wait();
    q.memcpy(dV,     hV.data(),     kvs * sizeof(fp16)).wait();
    q.memcpy(dN,     hN.data(),     NUM_Q_HEADS * HEAD_DIM * sizeof(float)).wait();

    // GPU step 1: mask_convert
    MaskConvertFunctor mc_f{d_orig, d_mask_out, d_mask_cnt,
                            (uint32_t)c.q_len, (uint32_t)NUM_KV_HEADS, (uint32_t)total_kv};
    q.submit([&](sycl::handler& h){
        h.parallel_for(sycl::nd_range<1>({(size_t)(NUM_KV_HEADS * q_blocks)}, {1}), mc_f);
    }).wait();

    // GPU step 2: sparse SDP using converted mask
    sycl::nd_range<2> nr({(size_t)q_blocks, (size_t)NUM_KV_HEADS * 16}, {1, 16});
    if (c.causal) {
        SparseFlashAttnFunctor<true> f{(uint8_t*)dQ, (uint8_t*)dK, (uint8_t*)dV,
            (uint8_t*)dN, (uint8_t*)dO, d_mask_out, d_mask_cnt,
            c.q_len, c.kv_len, hist, NUM_Q_HEADS, NUM_KV_HEADS};
        q.submit([&](sycl::handler& h){ h.parallel_for(nr, f); }).wait();
    } else {
        SparseFlashAttnFunctor<false> f{(uint8_t*)dQ, (uint8_t*)dK, (uint8_t*)dV,
            (uint8_t*)dN, (uint8_t*)dO, d_mask_out, d_mask_cnt,
            c.q_len, c.kv_len, hist, NUM_Q_HEADS, NUM_KV_HEADS};
        q.submit([&](sycl::handler& h){ h.parallel_for(nr, f); }).wait();
    }

    std::vector<fp16> gpu_out(qs);
    q.memcpy(gpu_out.data(), dO, qs * sizeof(fp16)).wait();

    // Compare
    float mx = 0, avg = 0;
    int   nan_c = 0, large_c = 0, valid = 0;
    float thresh = (c.scale <= 0.1f) ? 0.1f : c.scale * 1.5f;
    for (size_t i = 0; i < qs; i++) {
        float g = (float)gpu_out[i], r = (float)cpu_out[i];
        if (std::isnan(g) || std::isnan(r)) { nan_c++; continue; }
        float d = std::abs(g - r); mx = std::max(mx, d); avg += d;
        if (d > thresh) large_c++;
        valid++;
    }
    if (valid) avg /= valid;
    bool pass = (mx < thresh && large_c == 0 && nan_c == 0);

    std::cout << "  [" << (c.causal ? "causal   " : "noncausal") << "] "
              << std::setw(24) << std::left << c.name << std::right
              << "  max=" << std::scientific << std::setprecision(2) << mx
              << " avg=" << avg
              << " NaN=" << nan_c << " large=" << large_c
              << "  " << (pass ? "PASSED" : "FAILED") << "\n";

    sycl::free(d_orig, q); sycl::free(d_mask_out, q); sycl::free(d_mask_cnt, q);
    sycl::free(dQ, q); sycl::free(dK, q); sycl::free(dV, q);
    sycl::free(dO, q); sycl::free(dN, q);
    return pass;
}

// ============================================================
// main
// ============================================================
int main()
{
    sycl::queue q(sycl::gpu_selector_v);
    std::cout << "============================================================\n";
    std::cout << "SDP Correctness All — merged test suite\n";
    std::cout << "============================================================\n";
    std::cout << "Device: " << q.get_device().get_info<sycl::info::device::name>() << "\n";
    std::cout << "GQA: Q_heads=" << NUM_Q_HEADS << "  KV_heads=" << NUM_KV_HEADS
              << "  head_dim=" << HEAD_DIM << "\n\n";

    int total_pass = 0, total_fail = 0;

    // ----------------------------------------------------------
    // Section 1: mask_convert
    // ----------------------------------------------------------
    std::cout << "------------------------------------------------------------\n";
    std::cout << "Section 1: mask_convert correctness (19 cases)\n";
    std::cout << "------------------------------------------------------------\n";

    std::vector<MCfg> mc_cfgs = {
        // Corner cases: small kv_len (total_kv_blocks < 64)
        {1,   64,  1, "q=1   kv=64   (1 blk, min kv)"},
        {16,  64,  2, "q=16  kv=64   (1 blk)"},
        {16, 128,  3, "q=16  kv=128  (2 blks)"},
        // Normal cases
        {64,    4096,  1, "qlen=64    kv=4096   (union=64, trivial)"},
        {512,   4096,  1, "qlen=512   kv=4096   (union=64, trivial)"},
        {512,  16384,  1, "qlen=512   kv=16384  (64/256)"},
        {1024, 16384,  1, "qlen=1024  kv=16384  (64/256)"},
        {1024, 16384, 42, "qlen=1024  kv=16384  seed=42"},
        {2048, 16384,  7, "qlen=2048  kv=16384  seed=7"},
        {1024, 32768,  1, "qlen=1024  kv=32768  (64/512)"},
        {1024, 65536,  1, "qlen=1024  kv=65536  (64/1024, bitmap_full)"},
        {1024, 131072, 1, "qlen=1024  kv=131072 (64/2048, idx>1023 CRITICAL)"},
        {512,  131072, 2, "qlen=512   kv=131072 seed=2"},
        {1024, 524288, 1, "qlen=1024  kv=524288 (64/8192)"},
        {1024,2097152, 1, "qlen=1024  kv=2097152 (64/32768, max)"},
        {16,   16384,  5, "qlen=16    kv=16384  (single q_block)"},
        {16,  131072,  5, "qlen=16    kv=131072 (single q_block, large kv)"},
        {512,  16384, 99, "qlen=512   kv=16384  seed=99"},
        {512,  16384,123, "qlen=512   kv=16384  seed=123"},
    };
    int sec1_pass = 0, sec1_fail = 0;
    for (auto& c : mc_cfgs) {
        bool ok = mc_run_test(q, c);
        ok ? sec1_pass++ : sec1_fail++;
    }
    total_pass += sec1_pass; total_fail += sec1_fail;
    std::cout << "Section 1 result: " << sec1_pass << "/" << (sec1_pass + sec1_fail) << " passed\n\n";

    // ----------------------------------------------------------
    // Section 2: prefill sparse SDP
    // ----------------------------------------------------------
    std::cout << "------------------------------------------------------------\n";
    std::cout << "Section 2: prefill sparse SDP correctness (30 cases)\n";
    std::cout << "------------------------------------------------------------\n";

    std::vector<SCfg> sp_cfgs = {
        {512,  1024, 1,  false, 0.1f, 1, "512x1024  mc=1  s=0.1"},
        {512,  1024, 4,  false, 0.1f, 1, "512x1024  mc=4  s=0.1"},
        {512,  1024, 8,  false, 0.1f, 1, "512x1024  mc=8  s=0.1"},
        {512,  1024, 16, false, 0.1f, 1, "512x1024  mc=16 s=0.1"},
        {1024, 1024, 8,  false, 0.1f, 1, "1024x1024 mc=8  s=0.1"},
        {1024, 1024, 16, false, 0.1f, 1, "1024x1024 mc=16 s=0.1"},
        {512,  1024, 8,  false, 1.0f, 2, "512x1024  mc=8  s=1.0"},
        {512,  1024, 8,  false, 0.01f,3, "512x1024  mc=8  s=0.01"},
        {64,  16384, 64, false, 0.1f, 1, "64x16384  mc=64"},
        {64,  16384, 97, false, 0.1f, 1, "64x16384  mc=97"},
        {64,  16384,128, false, 0.1f, 1, "64x16384  mc=128"},
        {64,  16384,200, false, 0.1f, 1, "64x16384  mc=200"},
        {1024, 1024, 1,  false, 0.1f, 9, "1024x1024 mc=1  min_sparse"},
        {512,  1024, 1,  true,  0.1f, 1, "512x1024  mc=1  s=0.1"},
        {512,  1024, 8,  true,  0.1f, 1, "512x1024  mc=8  s=0.1"},
        {512,  1024, 16, true,  0.1f, 1, "512x1024  mc=16 s=0.1"},
        {1024, 1024, 8,  true,  0.1f, 1, "1024x1024 mc=8  s=0.1"},
        {1024, 1024, 16, true,  0.1f, 1, "1024x1024 mc=16 s=0.1"},
        {512,  1024, 8,  true,  1.0f, 2, "512x1024  mc=8  s=1.0"},
        {512,  1024, 8,  true,  0.01f,3, "512x1024  mc=8  s=0.01"},
        {64,  16384, 64, true,  0.1f, 1, "64x16384  mc=64"},
        {64,  16384, 97, true,  0.1f, 1, "64x16384  mc=97"},
        {64,  16384,128, true,  0.1f, 1, "64x16384  mc=128"},
        {512,  1024, 8,  true,  0.1f,99, "512x1024  mc=8  seed99"},
        {512,  1024, 8,  false, 0.1f,99, "512x1024  mc=8  seed99"},
        // Corner cases: q non-16-aligned
        { 17, 1024, 8, false, 0.1f, 1, "q=17  nc   (q non-16-align)"},
        { 33, 1024, 4, true,  0.1f, 2, "q=33  causal (q non-16-align)"},
        // Corner cases: small kv (< 64 blocks)
        { 64,  100, 1, false, 0.1f, 3, "kv=100 nc  (small kv, 2 blks)"},
        { 16,  128, 2, false, 0.1f, 4, "kv=128 nc  (small kv, 2 blks)"},
        // Corner cases: kv not 8-aligned (not small)
        { 64, 1029, 8, false, 0.1f, 5, "kv=1029 nc (kv non-8-align)"},
    };
    int sec2_pass = 0, sec2_fail = 0;
    for (auto& c : sp_cfgs) {
        bool ok = ps_run_test(q, c);
        ok ? sec2_pass++ : sec2_fail++;
    }
    total_pass += sec2_pass; total_fail += sec2_fail;
    std::cout << "Section 2 result: " << sec2_pass << "/" << (sec2_pass + sec2_fail) << " passed\n\n";

    // ----------------------------------------------------------
    // Section 3: prefill full SDP
    // ----------------------------------------------------------
    std::cout << "------------------------------------------------------------\n";
    std::cout << "Section 3: prefill full SDP correctness (8 cases)\n";
    std::cout << "------------------------------------------------------------\n";

    int sec3_pass = 0, sec3_fail = 0;
    struct PFCase { int q; int kv; bool causal; const char* name; };
    std::vector<PFCase> pf_cases = {
        {512,  1024, false, "512x1024 non-causal"},
        {1024, 1024, false, "1024x1024 non-causal"},
        {512,  1024, true,  "512x1024 causal"},
        {1024, 1024, true,  "1024x1024 causal"},
        // Corner cases
        { 17,  256, false, "q=17  nc  (q non-16-align)"},
        { 17,  256, true,  "q=17  causal (q non-16-align)"},
        {512,  100, false, "kv=100 nc (small kv, non-64-align)"},
        {512, 1025, false, "kv=1025 nc (kv non-8-align)"},
    };
    for (auto& c : pf_cases) {
        bool ok = pf_run_test(q, c.q, c.kv, c.causal, c.name);
        ok ? sec3_pass++ : sec3_fail++;
    }
    total_pass += sec3_pass; total_fail += sec3_fail;
    std::cout << "Section 3 result: " << sec3_pass << "/" << (sec3_pass + sec3_fail) << " passed\n\n";

    // ----------------------------------------------------------
    // Section 4: decode full SDP
    // ----------------------------------------------------------
    std::cout << "------------------------------------------------------------\n";
    std::cout << "Section 4: decode full SDP correctness (15 cases)\n";
    std::cout << "------------------------------------------------------------\n";

    std::vector<DFCfg> df_cfgs = {
        // Corner cases: kv_len not a multiple of chunk_size=256
        {  100, 0.1f,  1, "kv=100  (< chunk_size)"},
        {  300, 0.1f,  2, "kv=300  (non-chunk)"},
        { 4097, 0.1f,  3, "kv=4097 (non-chunk)"},
        // Corner cases: kv not 8-aligned (not small)
        { 1001, 0.1f,  4, "kv=1001 (non-8-align)"},
        { 8193, 0.1f,  5, "kv=8193 (non-8-align, large)"},
        // Normal cases
        { 1024, 0.01f, 11,  "kv=1K   scale=0.01 seed=11"},
        { 1024, 0.1f,  42,  "kv=1K   scale=0.1  seed=42"},
        { 1024, 1.0f,  42,  "kv=1K   scale=1.0  seed=42"},
        { 4096, 0.1f,  42,  "kv=4K   scale=0.1  seed=42"},
        { 4096, 1.0f,  99,  "kv=4K   scale=1.0  seed=99"},
        { 4096, 0.1f,   7,  "kv=4K   scale=0.1  seed=7 "},
        { 8192, 0.1f,  42,  "kv=8K   scale=0.1  seed=42"},
        { 8192, 0.5f, 123,  "kv=8K   scale=0.5  seed=123"},
        {16384, 0.1f,  42,  "kv=16K  scale=0.1  seed=42"},
        {16384, 1.0f,  42,  "kv=16K  scale=1.0  seed=42"},
    };
    int sec4_pass = 0, sec4_fail = 0;
    for (auto& c : df_cfgs) {
        bool ok = df_run_test(q, c);
        ok ? sec4_pass++ : sec4_fail++;
    }
    total_pass += sec4_pass; total_fail += sec4_fail;
    std::cout << "Section 4 result: " << sec4_pass << "/" << (sec4_pass + sec4_fail) << " passed\n\n";

    // ----------------------------------------------------------
    // Section 5: decode sparse SDP
    // ----------------------------------------------------------
    std::cout << "------------------------------------------------------------\n";
    std::cout << "Section 5: decode sparse SDP correctness (12 cases)\n";
    std::cout << "------------------------------------------------------------\n";

    std::vector<DSCfg> ds_cfgs = {
        // Corner case: kv_len not multiple of sp_blk_size, truncated mask (no partial block)
        { 4128,  0.1f,  5, false, "kv=4128 (non-sp-blk-align, trunc mask)"},
        // Corner case: kv_len not multiple of sp_blk_size, partial last block SELECTED
        { 4128,  0.1f,  6, true,  "kv=4128 (partial last blk selected)"},
        // Normal cases
        { 4096,  0.01f, 11, false, "kv=4K   scale=0.01 seed=11"},
        { 4096,  0.1f,  42, false, "kv=4K   scale=0.1  seed=42"},
        { 4096,  1.0f,  42, false, "kv=4K   scale=1.0  seed=42"},
        { 8192,  0.1f,  42, false, "kv=8K   scale=0.1  seed=42"},
        { 8192,  1.0f,  99, false, "kv=8K   scale=1.0  seed=99"},
        {16384,  0.1f,  42, false, "kv=16K  scale=0.1  seed=42"},
        {16384,  1.0f,  42, false, "kv=16K  scale=1.0  seed=42"},
        {32768,  0.1f,  77, false, "kv=32K  scale=0.1  seed=77"},
        {65536,  0.1f,  42, false, "kv=64K  scale=0.1  seed=42"},
        {65536,  0.5f, 123, false, "kv=64K  scale=0.5  seed=123"},
    };
    int sec5_pass = 0, sec5_fail = 0;
    for (auto& c : ds_cfgs) {
        bool ok = ds_run_test(q, c);
        ok ? sec5_pass++ : sec5_fail++;
    }
    total_pass += sec5_pass; total_fail += sec5_fail;
    std::cout << "Section 5 result: " << sec5_pass << "/" << (sec5_pass + sec5_fail) << " passed\n\n";

    // ----------------------------------------------------------
    // Section 6: mask_convert + prefill sparse SDP (pipeline)
    // ----------------------------------------------------------
    std::cout << "------------------------------------------------------------\n";
    std::cout << "Section 6: mask_convert + prefill sparse SDP pipeline (2 cases)\n";
    std::cout << "------------------------------------------------------------\n";

    // Two cases: fast CPU ref (q_len<=64)
    std::vector<IPCfg> ip_cfgs = {
        // 64x16384: union ~253, all aligned
        { 64, 16384, false, 0.1f,  1, "64x16384 nc  (union~253)"},
        // 61x16377: q non-16-align, kv non-8-align; partial last kv block (blk 255) selected
        { 61, 16377, false, 0.1f,  9, "61x16377 nc  (q-non16+kv-non8, partial blk255)"},
    };
    int sec6_pass = 0, sec6_fail = 0;
    for (auto& c : ip_cfgs) {
        bool ok = ip_run_test(q, c);
        ok ? sec6_pass++ : sec6_fail++;
    }
    total_pass += sec6_pass; total_fail += sec6_fail;
    std::cout << "Section 6 result: " << sec6_pass << "/" << (sec6_pass + sec6_fail) << " passed\n\n";

    // ----------------------------------------------------------
    // Grand total
    // ----------------------------------------------------------
    std::cout << "============================================================\n";
    std::cout << "GRAND TOTAL: " << total_pass << "/" << (total_pass + total_fail) << " PASSED";
    if (total_fail == 0) std::cout << " -- ALL PASSED";
    std::cout << "\n";
    std::cout << "  Section 1 (mask_convert):          " << sec1_pass << "/19\n";
    std::cout << "  Section 2 (prefill sparse):        " << sec2_pass << "/30\n";
    std::cout << "  Section 3 (prefill full):          " << sec3_pass << "/8\n";
    std::cout << "  Section 4 (decode full):           " << sec4_pass << "/15\n";
    std::cout << "  Section 5 (decode sparse):         " << sec5_pass << "/12\n";
    std::cout << "  Section 6 (mask_conv+sparse SDP):  " << sec6_pass << "/2\n";
    std::cout << "============================================================\n";
    return (total_fail == 0) ? 0 : 1;
}
