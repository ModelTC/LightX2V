// SageAttention prefill for gfx1201: INT8 Q.K^T (per-token Q scale, per-8-key K scale, 128-query Q smoothing)
// and FP8 P.V. Non-causal, head_dim 128, one sequence. Used for the layers whose Q/K outliers INT4 cannot hold.
//
// Same tiling as the INT4 kernel (8 waves x 32 queries, 32-key tiles, 3-buffer LDS ring), but QK -> softmax -> PV
// runs per 16-key block with an exact running max, and a block skips its PV WMMAs when the P of both fragments is 0.
#include "sage_common.h"

namespace {
constexpr int HD = 128;
constexpr int NW = 8;
constexpr int NQF = 2;
constexpr int BQW = NW * 16 * NQF;  // 256 queries per workgroup
constexpr int SUB = 128;            // Q-smoothing block; must match q_prep
constexpr int NSUB = BQW / SUB;
constexpr int NT = NW * 32;
constexpr int BK = 32;
constexpr int NKB = BK / 16;
constexpr int PK = HD;        // int8 bytes per token
constexpr int NKC = HD / 16;  // iu8 16x16x16 WMMAs per 16x16 score fragment
constexpr int KSTR = PK + 16;
constexpr int VSTR = BK + 16;
constexpr int NB = 3;
static_assert(BK * PK / 16 == NT, "K: every thread loads one 16B chunk");
static_assert(HD * BK / 16 == NT, "V: every thread loads one 16B chunk");
static_assert(NSUB * BK * 2 / 16 == 8 && BK / 8 == 4, "dS: 8 chunks, sK: 1 chunk");

struct Smem {
    uint8_t k[NB][BK * KSTR];
    uint8_t v[NB][HD * VSTR];
    _Float16 ds[NB][NSUB][BK];
    float sk[NB][BK / 8];
};

struct Stage {
    v4i k, v, x;  // x: dS chunk (wave 4 lanes 0-7) or sK chunk (wave 4 lane 8)
};
}  // namespace

// Qh/Kh: int8 [.., 128] per token. Vt: tile-contiguous [NkPad/32][128][32] fp8. dSh: [Hc, NqPad/128, NkPad] fp16.
__global__ __launch_bounds__(NT) void sage_attn_int8_kernel(const uint8_t* __restrict__ Qh, const float* __restrict__ sQ, const uint8_t* __restrict__ Kh,
                                                            const float* __restrict__ sK, const _Float16* __restrict__ dSh, const uint8_t* __restrict__ Vt,
                                                            const float* __restrict__ sV, uint16_t* __restrict__ out, int N, int NqPad, int NkPad, int H, int h0) {
    __shared__ __attribute__((aligned(16))) Smem sm;

    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int wave = __builtin_amdgcn_readfirstlane(tid >> 5);
    const int col = lane & 15, half = lane >> 4;
    const int hl = blockIdx.y;
    const int qblk = blockIdx.x;
    const int qw = qblk * BQW + wave * 16 * NQF;
    const int sub = (wave * 16 * NQF) / SUB;

    const bool x_ld = wave == NW / 2 && lane < 9;
    const bool is_sk = lane == 8;
    // uniform tile bases (advanced per tile) + per-thread 32-bit byte offsets
    const uint8_t* kt = Kh + (size_t)hl * NkPad * PK;
    const uint8_t* vt = Vt + (size_t)hl * HD * NkPad;
    const uint8_t* dt = (const uint8_t*)(dSh + ((size_t)hl * (NqPad / SUB) + qblk * NSUB) * NkPad);
    const uint8_t* st_ = (const uint8_t*)(sK + (size_t)hl * (NkPad / 8));
    const uint32_t k_go = tid * 16u, v_go = tid * 16u;
    const uint32_t x_go = is_sk ? 0u : (uint32_t)((lane >> 2) * NkPad * 2 + (lane & 3) * 16);
    const int k_ls = (tid >> 3) * KSTR + (tid & 7) * 16;
    const int v_ls = (tid >> 1) * VSTR + (tid & 1) * 16;

    auto gload = [&](Stage& s) {
        s.k = *(const v4i*)(kt + k_go);
        s.v = *(const v4i*)(vt + v_go);
        if (x_ld) s.x = *(const v4i*)((is_sk ? st_ : dt) + x_go);
        kt += BK * PK;
        vt += HD * BK;
        dt += BK * 2;
        st_ += BK / 8 * 4;
    };
    auto lstore = [&](auto bc, const Stage& s) {
        constexpr int b = decltype(bc)::value;
        *(v4i*)(&sm.k[b][k_ls]) = s.k;
        *(v4i*)(&sm.v[b][v_ls]) = s.v;
        if (x_ld) {
            if (is_sk)
                *(v4i*)(&sm.sk[b][0]) = s.x;
            else
                *(v4i*)(&sm.ds[b][lane >> 2][(lane & 3) * 8]) = s.x;
        }
    };

    v2i qf[NQF][NKC];
    float sq[NQF], m_run[NQF], lacc[NQF];
    v8f o[NQF][8];
#pragma unroll
    for (int f = 0; f < NQF; ++f) {
        const int q = qw + f * 16 + col;
        const uint8_t* qrow = Qh + ((size_t)hl * NqPad + q) * PK + half * 8;
#pragma unroll
        for (int c = 0; c < NKC; ++c) qf[f][c] = *(const v2i*)(qrow + c * 16);
        sq[f] = sQ[(size_t)hl * NqPad + q] * 8.f;
        m_run[f] = -INFINITY;
        lacc[f] = 0.f;
#pragma unroll
        for (int cb = 0; cb < 8; ++cb) o[f][cb] = v8f{};
    }

    auto compute = [&](auto bc) {
        constexpr int b = decltype(bc)::value;
#pragma unroll
        for (int kb = 0; kb < NKB; ++kb) {
            v8i acc[NQF];
            {
                v2i kf[NKC];
                const uint8_t* krow = &sm.k[b][(kb * 16 + col) * KSTR + half * 8];
#pragma unroll
                for (int c = 0; c < NKC; ++c) kf[c] = *(const v2i*)(krow + c * 16);
#pragma unroll
                for (int f = 0; f < NQF; ++f) {
                    acc[f] = v8i{};
#pragma unroll
                    for (int c = 0; c < NKC; ++c) acc[f] = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(true, kf[c], true, qf[f][c], acc[f], false);
                }
            }
            const float skb = sm.sk[b][kb * 2 + half];
            const h8 dsv = *(const h8*)(&sm.ds[b][sub][kb * 16 + half * 8]);
            v2i pf[NQF];
#pragma unroll
            for (int f = 0; f < NQF; ++f) {
                const float cs = sq[f] * skb;
                float u[8];
#pragma unroll
                for (int e = 0; e < 8; ++e) u[e] = __builtin_fmaf((float)acc[f][e], cs, (float)dsv[e]);
                float mx = u[0];
#pragma unroll
                for (int e = 1; e < 8; ++e) mx = fmaxf(mx, u[e]);
                mx = fmaxf(mx, other_half(mx));
                const float m_new = fmaxf(m_run[f], mx);
                if (__builtin_amdgcn_ballot_w32(m_new > m_run[f])) {
                    const float alpha = __builtin_amdgcn_exp2f((m_run[f] - m_new) * 0.125f);
#pragma unroll
                    for (int cb = 0; cb < 8; ++cb) o[f][cb] *= alpha;
                    lacc[f] *= alpha;
                    m_run[f] = m_new;
                }
                const float c = 126.f - m_run[f];
                pf[f] = v2i{(int)pack4(u[0] + c, u[1] + c, u[2] + c, u[3] + c), (int)pack4(u[4] + c, u[5] + c, u[6] + c, u[7] + c)};
                lacc[f] = __builtin_amdgcn_dot4_f32_fp8_fp8(pf[f][0], 0x38383838, lacc[f]);
                lacc[f] = __builtin_amdgcn_dot4_f32_fp8_fp8(pf[f][1], 0x38383838, lacc[f]);
            }
            // P == 0 exactly contributes o + 0 == o; skip only when both fragments are zero (one uniform branch)
            uint32_t zm = 0;
#pragma unroll
            for (int f = 0; f < NQF; ++f)
                if (!__builtin_amdgcn_ballot_w32((pf[f][0] | pf[f][1]) != 0)) zm |= 1u << f;
            zm = __builtin_amdgcn_readfirstlane(zm);
            if (zm != 3) {
#pragma unroll
                for (int cb = 0; cb < 8; ++cb) {
                    const v2i vf = *(const v2i*)(&sm.v[b][(cb * 16 + col) * VSTR + half * 8 + kb * 16]);
#pragma unroll
                    for (int f = 0; f < NQF; ++f) o[f][cb] = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(vf, pf[f], o[f][cb]);
                }
            }
        }
    };

    using I0 = std::integral_constant<int, 0>;
    using I1 = std::integral_constant<int, 1>;
    using I2 = std::integral_constant<int, 2>;
    const int ntiles = NkPad / BK;
    Stage st;
    gload(st);
    lstore(I0{}, st);
    bar_signal();

    auto step = [&](int t, auto bc, auto bn) {
        bar_wait();
        if (t + 1 < ntiles) {
            Stage s1;
            gload(s1);
            lstore(bn, s1);
        }
        bar_signal();
        compute(bc);
    };
    int t = 0;
    for (; t + 3 <= ntiles; t += 3) {
        step(t, I0{}, I1{});
        step(t + 1, I1{}, I2{});
        step(t + 2, I2{}, I0{});
    }
    if (t < ntiles) step(t, I0{}, I1{});
    if (t + 1 < ntiles) step(t + 1, I1{}, I2{});
    bar_wait();

    const float* sv = sV + (size_t)hl * HD;
#pragma unroll
    for (int f = 0; f < NQF; ++f) {
        const int q = qw + f * 16 + col;
        const float l_tot = lacc[f] + other_half(lacc[f]);
        if (q < N) {
            const float inv = 1.f / l_tot;
            uint16_t* orow = out + ((size_t)q * H + h0 + hl) * HD;
#pragma unroll
            for (int cb = 0; cb < 8; ++cb) {
                const int c0 = cb * 16 + half * 8;
                uint32_t w[4];
#pragma unroll
                for (int e = 0; e < 4; ++e) {
                    const float a = o[f][cb][2 * e] * sv[c0 + 2 * e] * inv;
                    const float bb = o[f][cb][2 * e + 1] * sv[c0 + 2 * e + 1] * inv;
                    w[e] = bf16_rne(a) | (bf16_rne(bb) << 16);
                }
                *(v4i*)(orow + c0) = v4i{(int)w[0], (int)w[1], (int)w[2], (int)w[3]};
            }
        }
    }
}

extern "C" int hip_sage_attn_int8(const void* Qh, const void* sQ, const void* Kh, const void* sK, const void* dSh, const void* Vt, const void* sV, void* out, int N, int NqPad,
                                  int NkPad, int H, int h0, int Hc, void* stream) {
    if (NqPad % BQW || NkPad % BK) return -1;
    hipLaunchKernelGGL(sage_attn_int8_kernel, dim3(NqPad / BQW, Hc), dim3(NT), 0, (hipStream_t)stream, (const uint8_t*)Qh, (const float*)sQ, (const uint8_t*)Kh,
                       (const float*)sK, (const _Float16*)dSh, (const uint8_t*)Vt, (const float*)sV, (uint16_t*)out, N, NqPad, NkPad, H, h0);
    return (int)hipGetLastError();
}
