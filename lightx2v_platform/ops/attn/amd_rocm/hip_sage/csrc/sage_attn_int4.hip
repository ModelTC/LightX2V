// SageAttention prefill for gfx1201: INT4 Q.K^T (per-token Q scale, per-8-key K scale, 128-query Q smoothing)
// and FP8 P.V. Non-causal, head_dim 128, one sequence.
//
// Workgroup: 8 waves x 32 queries; every wave holds two 16-query fragments that share each K/V LDS fragment.
// K/V stream in 32-key tiles through a 3-buffer LDS ring, prefetched into registers two tiles ahead.
// Softmax runs in "code" space: code = 8 * log2(P) + c_run is packed to a byte that, read as fp8 e4m3, is P.
// The row reference is lazy (re-anchored only when a code would exceed 126), and 16x16 P blocks that are
// all zero skip their PV WMMAs (bit-exact: o + 0 == o).
#include "sage_common.h"

namespace {
constexpr float TAU = 8.f;  // headroom in code units below 126 when a row (re-)anchors
constexpr int HD = 128;
constexpr int NW = 8;
constexpr int NQF = 2;
constexpr int BQW = NW * 16 * NQF;  // queries per workgroup
constexpr int SUB = 128;            // Q-smoothing block (queries per dS row); must match q_prep
constexpr int NSUB = BQW / SUB;
constexpr int NT = NW * 32;
constexpr int BK = 32;
constexpr int NKB = BK / 16;
constexpr int PK = HD / 2;  // int4 bytes per token
constexpr int KSTR = 80;
constexpr int VSTR = BK + 16;
constexpr int NB = 3;
constexpr int VT = HD * BK / 16;  // V 16-byte chunks per tile (one per thread)
constexpr int KT = BK * PK / 16;  // K 16-byte chunks per tile
constexpr int XW = NW / 2;        // wave that loads the dS rows and sK
constexpr int SKL = NSUB * 4;     // lane of XW that loads sK
static_assert(VT == NT && KT <= NT, "load roles");
static_assert(BQW % SUB == 0 && SKL < 32 && BK / 8 == 4, "dS: NSUB*4 chunks (lanes 0..), sK: 1 chunk");

struct Smem {
    uint8_t k[NB][BK * KSTR];
    uint8_t v[NB][HD * VSTR];
    _Float16 ds[NB][NSUB][BK];
    float sk[NB][BK / 8];
};

struct Stage {
    v4i k, v, x;  // x: dS chunk (wave XW lanes < SKL) or sK chunk (lane SKL)
};
}  // namespace

// Qh/Kh: packed int4 [.., 64] per token. Vt: tile-contiguous [NkPad/32][128][32] fp8. dSh: [Hc, NqPad/128, NkPad] fp16.
__global__ __launch_bounds__(NT) void sage_attn_int4_kernel(const uint8_t* __restrict__ Qh, const float* __restrict__ sQ, const uint8_t* __restrict__ Kh,
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

    const bool k_ld = tid < KT;
    const bool x_ld = wave == XW && (lane < NSUB * 4 || lane == SKL);
    const bool is_sk = lane == SKL;
    // uniform tile bases (advanced per tile) + per-thread 32-bit byte offsets
    const uint8_t* kt = Kh + (size_t)hl * NkPad * PK;
    const uint8_t* vt = Vt + (size_t)hl * HD * NkPad;
    const uint8_t* dt = (const uint8_t*)(dSh + ((size_t)hl * (NqPad / SUB) + qblk * NSUB) * NkPad);
    const uint8_t* st_ = (const uint8_t*)(sK + (size_t)hl * (NkPad / 8));
    const uint32_t k_go = tid * 16u, v_go = tid * 16u;
    const uint32_t x_go = is_sk ? 0u : (uint32_t)((lane >> 2) * NkPad * 2 + (lane & 3) * 16);
    const int k_ls = (tid >> 2) * KSTR + (tid & 3) * 16;
    const int v_ls = (tid >> 1) * VSTR + (tid & 1) * 16;

    auto gload = [&](Stage& s) {
        if (k_ld) s.k = *(const v4i*)(kt + k_go);
        s.v = *(const v4i*)(vt + v_go);
        if (x_ld) s.x = *(const v4i*)((is_sk ? st_ : dt) + x_go);
        kt += BK * PK;
        vt += HD * BK;
        dt += BK * 2;
        st_ += BK / 8 * 4;
    };
    auto lstore = [&](auto bc, const Stage& s) {
        constexpr int b = decltype(bc)::value;
        if (k_ld) *(v4i*)(&sm.k[b][k_ls]) = s.k;
        *(v4i*)(&sm.v[b][v_ls]) = s.v;
        if (x_ld) {
            if (is_sk)
                *(v4i*)(&sm.sk[b][0]) = s.x;
            else
                *(v4i*)(&sm.ds[b][lane >> 2][(lane & 3) * 8]) = s.x;
        }
    };

    v2i qf[NQF][4];
    float sq[NQF], m_run[NQF], lacc[NQF];
    v8f o[NQF][8];
#pragma unroll
    for (int f = 0; f < NQF; ++f) {
        const int q = qw + f * 16 + col;
        const uint8_t* qrow = Qh + ((size_t)hl * NqPad + q) * PK + half * 8;
#pragma unroll
        for (int c = 0; c < 4; ++c) qf[f][c] = *(const v2i*)(qrow + c * 16);
        sq[f] = sQ[(size_t)hl * NqPad + q] * 8.f;
        m_run[f] = 0.f;  // c_run: code offset of the row reference, set exactly on the first tile
        lacc[f] = 0.f;
#pragma unroll
        for (int cb = 0; cb < 8; ++cb) o[f][cb] = v8f{};
    }

    auto compute = [&](auto bc, bool first) {
        constexpr int b = decltype(bc)::value;
        v8i acc[NQF][NKB];
#pragma unroll
        for (int kb = 0; kb < NKB; ++kb) {
            v2i kf[4];
            const uint8_t* krow = &sm.k[b][(kb * 16 + col) * KSTR + half * 8];
#pragma unroll
            for (int c = 0; c < 4; ++c) kf[c] = *(const v2i*)(krow + c * 16);
#pragma unroll
            for (int f = 0; f < NQF; ++f) {
                acc[f][kb] = v8i{};
#pragma unroll
                for (int c = 0; c < 4; ++c) acc[f][kb] = __builtin_amdgcn_wmma_i32_16x16x32_iu4_w32_gfx12(true, kf[c], true, qf[f][c], acc[f][kb], false);
            }
        }
        v2i pf[NQF][NKB];
#pragma unroll
        for (int f = 0; f < NQF; ++f) {
            float u[NKB][8];
#pragma unroll
            for (int kb = 0; kb < NKB; ++kb) {
                const float cs = sq[f] * sm.sk[b][kb * 2 + half];
                // loaded as four half2 (not one h8) so each packed dS + c_run add stays within its own dword
                const h2* dp = (const h2*)(&sm.ds[b][sub][kb * 16 + half * 8]);
                const h2 c2 = {(_Float16)m_run[f], (_Float16)m_run[f]};
                h2 dd[4];
#pragma unroll
                for (int i = 0; i < 4; ++i) dd[i] = dp[i] + c2;
#pragma unroll
                for (int e = 0; e < 8; ++e) u[kb][e] = __builtin_fmaf((float)acc[f][kb][e], cs, (float)dd[e >> 1][e & 1]);
            }
            float mx = u[0][0];
#pragma unroll
            for (int kb = 0; kb < NKB; ++kb)
#pragma unroll
                for (int e = 0; e < 8; ++e) mx = fmaxf(mx, u[kb][e]);
            // u are codes already; rows whose codes exceed 126 (or the first tile) re-anchor to max - TAU
            if (first || __builtin_amdgcn_ballot_w32(mx > 126.f)) {
                const float rmx = fmaxf(mx, other_half(mx));
                float sh = (first || rmx > 126.f) ? rmx - (126.f - TAU) : 0.f;
                // keep c_run fp16-exact so every tile's dS + c_run carries the same offset the rescale assumes
                const float c_new = (float)(_Float16)(m_run[f] - sh);
                sh = m_run[f] - c_new;
                m_run[f] = c_new;
                if (!first) {
                    const float alpha = __builtin_amdgcn_exp2f(-sh * 0.125f);
#pragma unroll
                    for (int cb = 0; cb < 8; ++cb) o[f][cb] *= alpha;
                    lacc[f] *= alpha;
                }
#pragma unroll
                for (int kb = 0; kb < NKB; ++kb)
#pragma unroll
                    for (int e = 0; e < 8; ++e) u[kb][e] = fminf(u[kb][e] - sh, 126.f);  // |c_run| > 16K: fp16 ulp > TAU
            }
#pragma unroll
            for (int kb = 0; kb < NKB; ++kb) {
                pf[f][kb] = v2i{(int)pack4(u[kb][0], u[kb][1], u[kb][2], u[kb][3]), (int)pack4(u[kb][4], u[kb][5], u[kb][6], u[kb][7])};
                lacc[f] = __builtin_amdgcn_dot4_f32_fp8_fp8(pf[f][kb][0], 0x38383838, lacc[f]);
                lacc[f] = __builtin_amdgcn_dot4_f32_fp8_fp8(pf[f][kb][1], 0x38383838, lacc[f]);
            }
        }
        // P == 0 exactly contributes o + 0 == o, so all-zero 16x16 P blocks skip their PV WMMAs bit-exactly
        uint32_t zm = 0;
#pragma unroll
        for (int f = 0; f < NQF; ++f)
#pragma unroll
            for (int kb = 0; kb < NKB; ++kb)
                if (!__builtin_amdgcn_ballot_w32((pf[f][kb][0] | pf[f][kb][1]) != 0)) zm |= 1u << (f * NKB + kb);
        zm = __builtin_amdgcn_readfirstlane(zm);
#pragma unroll
        for (int kb = 0; kb < NKB; ++kb) {
            const uint32_t zk = zm >> kb;
            if ((zk & 1) && (zk >> NKB & 1)) continue;
            v2i vf[8];
#pragma unroll
            for (int cb = 0; cb < 8; ++cb) vf[cb] = *(const v2i*)(&sm.v[b][(cb * 16 + col) * VSTR + half * 8] + kb * 16);
#pragma unroll
            for (int f = 0; f < NQF; ++f) {
                if (zk >> (f * NKB) & 1) continue;
#pragma unroll
                for (int cb = 0; cb < 8; ++cb) o[f][cb] = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(vf[cb], pf[f][kb], o[f][cb]);
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
    if (ntiles > 1) gload(st);
    bar_signal();

    auto step = [&](int t, auto bc, auto bn, bool first) {
        bar_wait();
        if (t + 1 < ntiles) lstore(bn, st);
        bar_signal();
        if (t + 2 < ntiles) gload(st);
        compute(bc, first);
    };
    // tile 0 peeled so the first-tile anchoring is compile-time in the steady-state loop
    step(0, I0{}, I1{}, true);
    int t = 1;
    for (; t + 3 <= ntiles; t += 3) {
        step(t, I1{}, I2{}, false);
        step(t + 1, I2{}, I0{}, false);
        step(t + 2, I0{}, I1{}, false);
    }
    if (t < ntiles) step(t, I1{}, I2{}, false);
    if (t + 1 < ntiles) step(t + 1, I2{}, I0{}, false);
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

extern "C" int hip_sage_attn_int4(const void* Qh, const void* sQ, const void* Kh, const void* sK, const void* dSh, const void* Vt, const void* sV, void* out, int N, int NqPad,
                                  int NkPad, int H, int h0, int Hc, void* stream) {
    if (NqPad % BQW || NkPad % BK) return -1;
    hipLaunchKernelGGL(sage_attn_int4_kernel, dim3(NqPad / BQW, Hc), dim3(NT), 0, (hipStream_t)stream, (const uint8_t*)Qh, (const float*)sQ, (const uint8_t*)Kh,
                       (const float*)sK, (const _Float16*)dSh, (const uint8_t*)Vt, (const float*)sV, (uint16_t*)out, N, NqPad, NkPad, H, h0);
    return (int)hipGetLastError();
}
