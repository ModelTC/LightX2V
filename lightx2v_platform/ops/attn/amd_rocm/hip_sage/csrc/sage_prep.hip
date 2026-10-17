// Quantization / smoothing prep kernels and the attention entry point of the gfx1201 SageAttention library.
// q/k/v are [N, H, 128] bf16 with row stride H*128.
#include <hip/hip_runtime.h>
#include <math.h>
#include <stdint.h>

typedef int v4i __attribute__((ext_vector_type(4)));
typedef __bf16 bf8 __attribute__((ext_vector_type(8)));
typedef float v8f __attribute__((ext_vector_type(8)));

namespace {
constexpr int HD = 128;
constexpr int BQ = 128;
constexpr int BK = 32;

__device__ __forceinline__ float bf2f(uint16_t b) { return __uint_as_float((uint32_t)b << 16); }

__device__ __forceinline__ float wave_max(float x) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) x = fmaxf(x, __shfl_xor(x, o, 32));
    return x;
}

__device__ __forceinline__ uint32_t q8x4(float a, float b, float c, float d, float inv) {
    const int ia = (int)fminf(fmaxf(rintf(a * inv), -127.f), 127.f);
    const int ib = (int)fminf(fmaxf(rintf(b * inv), -127.f), 127.f);
    const int ic = (int)fminf(fmaxf(rintf(c * inv), -127.f), 127.f);
    const int id = (int)fminf(fmaxf(rintf(d * inv), -127.f), 127.f);
    return (ia & 0xff) | ((ib & 0xff) << 8) | ((ic & 0xff) << 16) | ((uint32_t)(id & 0xff) << 24);
}

// element 2i goes to the low nibble
__device__ __forceinline__ uint32_t q4x4(float a, float b, float c, float d, float inv) {
    const int ia = (int)fminf(fmaxf(rintf(a * inv), -7.f), 7.f);
    const int ib = (int)fminf(fmaxf(rintf(b * inv), -7.f), 7.f);
    const int ic = (int)fminf(fmaxf(rintf(c * inv), -7.f), 7.f);
    const int id = (int)fminf(fmaxf(rintf(d * inv), -7.f), 7.f);
    return (ia & 0xf) | ((ib & 0xf) << 4) | ((ic & 0xf) << 8) | ((id & 0xf) << 12);
}

template <int BITS> struct QT;
template <> struct QT<8> { static constexpr float QMAX = 127.f; };
template <> struct QT<4> { static constexpr float QMAX = 7.f; };

// Order-preserving float -> uint key so a uint atomicMax yields the float max (key 0 < any real float).
__device__ __forceinline__ uint32_t f2key(float f) {
    const uint32_t u = __float_as_uint(f);
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}
__device__ __forceinline__ float key2f(uint32_t k) {
    return __uint_as_float((k & 0x80000000u) ? (k & 0x7fffffffu) : ~k);
}
}  // namespace

// grid (NqPad/128, hc), block 256. Qh [hc,NqPad,128*BITS/8], sQ [hc,NqPad], qbar [hc,NqPad/128,128] (scaled).
template <int BITS>
__global__ __launch_bounds__(256) void q_prep_kernel(const uint16_t* __restrict__ q, int N, int H, int h0, float scale,
                                                     int NqPad, uint8_t* __restrict__ Qh, float* __restrict__ sQ,
                                                     float* __restrict__ qbar)
{
    __shared__ uint16_t tile[BQ][HD + 8];
    __shared__ float part[2][HD];
    const int tid = threadIdx.x, b = blockIdx.x, hl = blockIdx.y;
    const size_t rs = (size_t)H * HD;
    const int q0 = b * BQ;
    for (int i = tid; i < BQ * 16; i += 256) {
        const int r = i >> 4, c = (i & 15) * 8;
        v4i x = {0, 0, 0, 0};
        if (q0 + r < N) x = *(const v4i*)(q + (size_t)(q0 + r) * rs + (size_t)(h0 + hl) * HD + c);
        *(v4i*)&tile[r][c] = x;
    }
    __syncthreads();
    {
        const int c = tid & 127, r0 = (tid >> 7) * 64;
        float csum = 0.f;
        for (int r = r0; r < r0 + 64; ++r) csum += bf2f(tile[r][c]);
        part[tid >> 7][tid & 127] = csum;
    }
    __syncthreads();
    const int nvalid = max(1, min(BQ, N - q0));
    if (tid < HD) {
        const float m = (part[0][tid] + part[1][tid]) / (float)nvalid * scale;
        part[0][tid] = m;
        qbar[((size_t)hl * (NqPad / BQ) + b) * HD + tid] = m;
    }
    __syncthreads();
    const int lane = tid & 31, w = tid >> 5, c = lane * 4;
    const float m0 = part[0][c], m1 = part[0][c + 1], m2 = part[0][c + 2], m3 = part[0][c + 3];
    for (int r = w * 16; r < w * 16 + 16; ++r) {
        const bool ok = q0 + r < N;
        const float x0 = ok ? bf2f(tile[r][c]) * scale - m0 : 0.f;
        const float x1 = ok ? bf2f(tile[r][c + 1]) * scale - m1 : 0.f;
        const float x2 = ok ? bf2f(tile[r][c + 2]) * scale - m2 : 0.f;
        const float x3 = ok ? bf2f(tile[r][c + 3]) * scale - m3 : 0.f;
        const float amax = wave_max(fmaxf(fmaxf(fabsf(x0), fabsf(x1)), fmaxf(fabsf(x2), fabsf(x3))));
        const float s = fmaxf(amax, 1e-20f) / QT<BITS>::QMAX;
        const size_t row = (size_t)hl * NqPad + q0 + r;
        if constexpr (BITS == 8)
            *(uint32_t*)(Qh + row * HD + c) = q8x4(x0, x1, x2, x3, 1.f / s);
        else
            *(uint16_t*)(Qh + row * (HD / 2) + c / 2) = (uint16_t)q4x4(x0, x1, x2, x3, 1.f / s);
        if (lane == 0) sQ[row] = s;
    }
}

// grid (NkPad/64, hc), block 256; one wave per 8-token group. kmean [H,128] fp32.
template <int BITS>
__global__ __launch_bounds__(256) void k_prep_kernel(const uint16_t* __restrict__ k, const float* __restrict__ kmean,
                                                     int N, int H, int h0, int NkPad, uint8_t* __restrict__ Kh,
                                                     float* __restrict__ sK)
{
    const int tid = threadIdx.x, lane = tid & 31, hl = blockIdx.y;
    const int g = blockIdx.x * 8 + (tid >> 5);
    const int t = g * 8 + (lane >> 2), cb = (lane & 3) * 32;
    const float* km = kmean + (size_t)(h0 + hl) * HD + cb;
    float x[32];
    if (t < N) {
        const uint16_t* src = k + (size_t)t * H * HD + (size_t)(h0 + hl) * HD + cb;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const v4i u = *(const v4i*)(src + j * 8);
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                x[j * 8 + 2 * e] = __uint_as_float((uint32_t)u[e] << 16) - km[j * 8 + 2 * e];
                x[j * 8 + 2 * e + 1] = __uint_as_float((uint32_t)u[e] & 0xffff0000u) - km[j * 8 + 2 * e + 1];
            }
        }
    } else {
#pragma unroll
        for (int j = 0; j < 32; ++j) x[j] = 0.f;
    }
    float amax = 0.f;
#pragma unroll
    for (int j = 0; j < 32; ++j) amax = fmaxf(amax, fabsf(x[j]));
    amax = wave_max(amax);
    const float s = fmaxf(amax, 1e-20f) / QT<BITS>::QMAX, inv = 1.f / s;
    if constexpr (BITS == 8) {
        uint32_t w[8];
#pragma unroll
        for (int j = 0; j < 8; ++j) w[j] = q8x4(x[4 * j], x[4 * j + 1], x[4 * j + 2], x[4 * j + 3], inv);
        uint8_t* dst = Kh + ((size_t)hl * NkPad + t) * HD + cb;
        *(v4i*)dst = v4i{(int)w[0], (int)w[1], (int)w[2], (int)w[3]};
        *(v4i*)(dst + 16) = v4i{(int)w[4], (int)w[5], (int)w[6], (int)w[7]};
    } else {
        uint32_t w[4];
#pragma unroll
        for (int j = 0; j < 4; ++j)
            w[j] = q4x4(x[8 * j], x[8 * j + 1], x[8 * j + 2], x[8 * j + 3], inv) |
                   (q4x4(x[8 * j + 4], x[8 * j + 5], x[8 * j + 6], x[8 * j + 7], inv) << 16);
        *(v4i*)(Kh + ((size_t)hl * NkPad + t) * (HD / 2) + cb / 2) = v4i{(int)w[0], (int)w[1], (int)w[2], (int)w[3]};
    }
    if (lane == 0) sK[(size_t)hl * (NkPad / 8) + g] = s;
}

// ksum[H*128] += column sums of k, vmax[H*128] = max |v| (non-negative float bits, so uint atomicMax orders them).
// block (H*16, S): S row slices of `rows` rows each, reduced in LDS before the atomics.
// Outputs zero-initialised by the caller.
__global__ __launch_bounds__(1024) void kv_stats_kernel(const uint16_t* __restrict__ k, const uint16_t* __restrict__ v,
                                                        int N, int H, int rows, float* __restrict__ ksum,
                                                        uint32_t* __restrict__ vmax)
{
    __shared__ float red[2][1024 * 8 / 4];
    const int c8 = threadIdx.x * 8, S = blockDim.y, y = threadIdx.y;
    float s[8] = {}, m[8] = {};
    const int t0 = (blockIdx.x * S + y) * rows, t1 = min(N, t0 + rows);
    for (int t = t0; t < t1; ++t) {
        const v4i kx = *(const v4i*)(k + (size_t)t * H * HD + c8);
        const v4i vx = *(const v4i*)(v + (size_t)t * H * HD + c8);
#pragma unroll
        for (int e = 0; e < 4; ++e) {
            s[2 * e] += __uint_as_float((uint32_t)kx[e] << 16);
            s[2 * e + 1] += __uint_as_float((uint32_t)kx[e] & 0xffff0000u);
            m[2 * e] = fmaxf(m[2 * e], fabsf(__uint_as_float((uint32_t)vx[e] << 16)));
            m[2 * e + 1] = fmaxf(m[2 * e + 1], fabsf(__uint_as_float((uint32_t)vx[e] & 0xffff0000u)));
        }
    }
    const int W = blockDim.x * 8;
    for (int st = S / 2; st > 0; st >>= 1) {
        if (y >= st && y < 2 * st) {
#pragma unroll
            for (int e = 0; e < 8; ++e) {
                red[0][(y - st) * W + c8 + e] = s[e];
                red[1][(y - st) * W + c8 + e] = m[e];
            }
        }
        __syncthreads();
        if (y < st) {
#pragma unroll
            for (int e = 0; e < 8; ++e) {
                s[e] += red[0][y * W + c8 + e];
                m[e] = fmaxf(m[e], red[1][y * W + c8 + e]);
            }
        }
        __syncthreads();
    }
    if (y) return;
#pragma unroll
    for (int e = 0; e < 8; ++e) {
        atomicAdd(&ksum[c8 + e], s[e]);
        atomicMax(&vmax[c8 + e], __float_as_uint(m[e]));
    }
}

// grid (NkPad/32, hc), block 256. Vt [hc][NkPad/32][128][32] fp8 e4m3; sV [H,128].
__global__ __launch_bounds__(256) void v_prep_kernel(const uint16_t* __restrict__ v, const float* __restrict__ sV,
                                                     int N, int H, int h0, int NkPad, uint8_t* __restrict__ Vt)
{
    __shared__ uint16_t tile[BK][HD + 8];
    const int tid = threadIdx.x, tb = blockIdx.x, hl = blockIdx.y;
    for (int i = tid; i < BK * 16; i += 256) {
        const int r = i >> 4, c = (i & 15) * 8;
        const int t = tb * BK + r;
        v4i x = {0, 0, 0, 0};
        if (t < N) x = *(const v4i*)(v + (size_t)t * H * HD + (size_t)(h0 + hl) * HD + c);
        *(v4i*)&tile[r][c] = x;
    }
    __syncthreads();
    const int c = tid >> 1, j0 = (tid & 1) * 16;
    const float sv = sV[(size_t)(h0 + hl) * HD + c];
    uint32_t w[4];
#pragma unroll
    for (int e = 0; e < 4; ++e) {
        uint32_t p = 0;
        p = __builtin_amdgcn_cvt_pk_fp8_f32(bf2f(tile[j0 + 4 * e][c]) / sv, bf2f(tile[j0 + 4 * e + 1][c]) / sv, p, false);
        p = __builtin_amdgcn_cvt_pk_fp8_f32(bf2f(tile[j0 + 4 * e + 2][c]) / sv, bf2f(tile[j0 + 4 * e + 3][c]) / sv, p, true);
        w[e] = p;
    }
    *(v4i*)(Vt + (((size_t)hl * (NkPad / BK) + tb) * HD + c) * BK + j0) = v4i{(int)w[0], (int)w[1], (int)w[2], (int)w[3]};
}

// dS row shift: rkey[z][r] = max over every 4th 64-key block of qbar_hi[r] . k[j]. The shift only has to be near the
// row max to keep fp16 precise around the large scores. qs: [2][hc][nqbPad][128] bf16 split of qbar.
// The sampled keys of DSR consecutive 256-key ranges are staged in LDS once and every wave walks all row groups.
// grid (ceil(NkPad/(256*DSR)), hc), block 256.
constexpr int DSR = 2;
__global__ __launch_bounds__(256) void ds_max_kernel(const uint16_t* __restrict__ qs, const uint16_t* __restrict__ k,
                                                     int N, int H, int h0, int hc, int nqb, int nqbPad,
                                                     uint32_t* __restrict__ rkey)
{
    constexpr int KS = HD + 8, NKS = 64 * DSR;
    __shared__ __attribute__((aligned(16))) uint16_t kt[NKS * KS];
    const int tid = threadIdx.x, lane = tid & 31, w = tid >> 5;
    const int col = lane & 15, half = lane >> 4;
    const int z = blockIdx.y;
    const uint16_t* kh = k + (size_t)(h0 + z) * HD;
    auto key_of = [&](int s) { return (blockIdx.x * DSR + s / 64) * 256 + s % 64; };
    for (int idx = tid; idx < NKS * 16; idx += 256) {
        const int s = idx >> 4, c8 = (idx & 15) * 8, j = key_of(s);
        v4i x = {0, 0, 0, 0};
        if (j < N) x = *(const v4i*)(kh + (size_t)j * H * HD + c8);
        *(v4i*)&kt[s * KS + c8] = x;
    }
    __syncthreads();
    for (int rg = w; rg < nqbPad / 16; rg += 8) {
        const int r0 = rg * 16;
        if (r0 >= nqb) break;
        bf8 a[8];
#pragma unroll
        for (int c = 0; c < 8; ++c) a[c] = *(const bf8*)(qs + ((size_t)z * nqbPad + r0 + col) * HD + c * 16 + half * 8);
        float m[8];
#pragma unroll
        for (int e = 0; e < 8; ++e) m[e] = -INFINITY;
#pragma unroll
        for (int f = 0; f < NKS / 16; ++f) {
            const int j = key_of(f * 16 + col);
            v8f acc = {};
#pragma unroll
            for (int c = 0; c < 8; ++c) {
                const bf8 b = *(const bf8*)&kt[(f * 16 + col) * KS + c * 16 + half * 8];
                acc = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a[c], b, acc);
            }
            if (j < N) {
#pragma unroll
                for (int e = 0; e < 8; ++e) m[e] = fmaxf(m[e], acc[e]);
            }
        }
#pragma unroll
        for (int e = 0; e < 8; ++e) {
            float x = m[e];
#pragma unroll
            for (int o = 1; o < 16; o <<= 1) x = fmaxf(x, __shfl_xor(x, o, 32));
            const int r = r0 + half * 8 + e;
            if (col == 0 && r < nqb && x > -INFINITY) atomicMax(&rkey[(size_t)z * nqbPad + r], f2key(x));
        }
    }
}

// dS[z][r][j] = fp16 clamp(8 * (qbar[r] . k[j] - rowshift[r]), -60000), -inf for padded keys; qbar from the 2-way
// bf16 split. The 64-key K tile is loaded to LDS once per block and each wave stages its 16x64 tile for b128 stores.
// grid (NkPad/64, hc), block 256.
__global__ __launch_bounds__(256) void ds_write_kernel(const uint16_t* __restrict__ qs, const uint16_t* __restrict__ k,
                                                       int N, int NkPad, int H, int h0, int hc, int nqb, int nqbPad,
                                                       const uint32_t* __restrict__ rkey, _Float16* __restrict__ ds)
{
    constexpr int KS = HD + 8, OS = 64 + 8;
    __shared__ __attribute__((aligned(16))) uint16_t kt[64 * KS];
    __shared__ __attribute__((aligned(16))) _Float16 ot[8][16 * OS];
    const int tid = threadIdx.x, lane = tid & 31, w = tid >> 5;
    const int col = lane & 15, half = lane >> 4;
    const int z = blockIdx.y;
    const int j0 = blockIdx.x * 64;
    const uint16_t* kh = k + (size_t)(h0 + z) * HD;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int idx = tid + i * 256, key = idx >> 4, c8 = (idx & 15) * 8;
        v4i x = {0, 0, 0, 0};
        if (j0 + key < N) x = *(const v4i*)(kh + (size_t)(j0 + key) * H * HD + c8);
        *(v4i*)&kt[key * KS + c8] = x;
    }
    __syncthreads();
    _Float16* o = ot[w];
    for (int rg = w; rg < nqbPad / 16; rg += 8) {
        const int r0 = rg * 16;
        bf8 a[2][8];
#pragma unroll
        for (int p = 0; p < 2; ++p)
#pragma unroll
            for (int c = 0; c < 8; ++c)
                a[p][c] = *(const bf8*)(qs + (((size_t)p * hc + z) * nqbPad + r0 + col) * HD + c * 16 + half * 8);
        float m[8];
#pragma unroll
        for (int e = 0; e < 8; ++e) {
            const int r = r0 + half * 8 + e;
            m[e] = r < nqb ? key2f(rkey[(size_t)z * nqbPad + r]) : 0.f;
        }
#pragma unroll
        for (int f = 0; f < 4; ++f) {
            const int j = j0 + f * 16 + col;
            v8f acc = {};
#pragma unroll
            for (int c = 0; c < 8; ++c) {
                const bf8 b = *(const bf8*)&kt[(f * 16 + col) * KS + c * 16 + half * 8];
#pragma unroll
                for (int p = 1; p >= 0; --p) acc = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a[p][c], b, acc);
            }
#pragma unroll
            for (int e = 0; e < 8; ++e)
                o[(half * 8 + e) * OS + f * 16 + col] =
                    (_Float16)(j < N ? fmaxf((acc[e] - m[e]) * 8.f, -60000.f) : -INFINITY);
        }
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int idx = lane + i * 32, row = idx >> 3, c8 = (idx & 7) * 8;
            const v4i val = *(const v4i*)&o[row * OS + c8];
            if (r0 + row < nqb) *(v4i*)(ds + ((size_t)z * nqb + r0 + row) * NkPad + j0 + c8) = val;
        }
    }
}

extern "C" int hip_sage_kv_stats(const void* k, const void* v, int N, int H, void* ksum, void* vmax, void* stream)
{
    if (H * 16 > 1024) return -1;
    const int S = H * 16 <= 64 ? 8 : H * 16 <= 128 ? 4 : H * 16 <= 256 ? 2 : 1;
    const int rows = 64;
    hipLaunchKernelGGL(kv_stats_kernel, dim3((N + rows * S - 1) / (rows * S)), dim3(H * 16, S), 0, (hipStream_t)stream,
                       (const uint16_t*)k, (const uint16_t*)v, N, H, rows, (float*)ksum, (uint32_t*)vmax);
    return (int)hipGetLastError();
}

extern "C" int hip_sage_q_prep(int bits, const void* q, int N, int H, int h0, int hc, float scale, int NqPad, void* Qh,
                              void* sQ, void* qbar, void* stream)
{
    if ((bits != 4 && bits != 8) || NqPad % BQ) return -1;
    const dim3 g(NqPad / BQ, hc);
    const auto s = (hipStream_t)stream;
    if (bits == 8)
        hipLaunchKernelGGL(q_prep_kernel<8>, g, dim3(256), 0, s, (const uint16_t*)q, N, H, h0, scale, NqPad,
                           (uint8_t*)Qh, (float*)sQ, (float*)qbar);
    else
        hipLaunchKernelGGL(q_prep_kernel<4>, g, dim3(256), 0, s, (const uint16_t*)q, N, H, h0, scale, NqPad,
                           (uint8_t*)Qh, (float*)sQ, (float*)qbar);
    return (int)hipGetLastError();
}

extern "C" int hip_sage_k_prep(int bits, const void* k, const void* kmean, int N, int H, int h0, int hc, int NkPad,
                              void* Kh, void* sK, void* stream)
{
    if ((bits != 4 && bits != 8) || NkPad % 64) return -1;
    const dim3 g(NkPad / 64, hc);
    const auto s = (hipStream_t)stream;
    if (bits == 8)
        hipLaunchKernelGGL(k_prep_kernel<8>, g, dim3(256), 0, s, (const uint16_t*)k, (const float*)kmean, N, H, h0,
                           NkPad, (uint8_t*)Kh, (float*)sK);
    else
        hipLaunchKernelGGL(k_prep_kernel<4>, g, dim3(256), 0, s, (const uint16_t*)k, (const float*)kmean, N, H, h0,
                           NkPad, (uint8_t*)Kh, (float*)sK);
    return (int)hipGetLastError();
}

extern "C" int hip_sage_v_prep(const void* v, const void* sV, int N, int H, int h0, int hc, int NkPad, void* Vt,
                              void* stream)
{
    hipLaunchKernelGGL(v_prep_kernel, dim3(NkPad / BK, hc), dim3(256), 0, (hipStream_t)stream, (const uint16_t*)v,
                       (const float*)sV, N, H, h0, NkPad, (uint8_t*)Vt);
    return (int)hipGetLastError();
}

extern "C" int hip_sage_ds(const void* qs, const void* k, int N, int NkPad, int H, int h0, int hc, int nqb, int nqbPad,
                          void* rkey, void* ds, void* stream)
{
    if (nqbPad % 128 || NkPad % 64) return -1;
    const auto s = (hipStream_t)stream;
    if (hipMemsetAsync(rkey, 0, (size_t)hc * nqbPad * 4, s) != hipSuccess) return -2;
    hipLaunchKernelGGL(ds_max_kernel, dim3((NkPad + 256 * DSR - 1) / (256 * DSR), hc), dim3(256), 0, s,
                       (const uint16_t*)qs, (const uint16_t*)k, N, H, h0, hc, nqb, nqbPad, (uint32_t*)rkey);
    hipLaunchKernelGGL(ds_write_kernel, dim3(NkPad / 64, hc), dim3(256), 0, s, (const uint16_t*)qs,
                       (const uint16_t*)k, N, NkPad, H, h0, hc, nqb, nqbPad, (const uint32_t*)rkey, (_Float16*)ds);
    return (int)hipGetLastError();
}

extern "C" int hip_sage_attn_int4(const void* Qh, const void* sQ, const void* Kh, const void* sK, const void* dSh, const void* Vt, const void* sV, void* out, int N, int NqPad,
                                  int NkPad, int H, int h0, int Hc, void* stream);
extern "C" int hip_sage_attn_int8(const void* Qh, const void* sQ, const void* Kh, const void* sK, const void* dSh, const void* Vt, const void* sV, void* out, int N, int NqPad,
                                  int NkPad, int H, int h0, int Hc, void* stream);

extern "C" int hip_sage_attn(int bits, const void* Qh, const void* sQ, const void* Kh, const void* sK, const void* dSh, const void* Vt, const void* sV, void* out, int N,
                             int NqPad, int NkPad, int H, int h0, int Hc, void* stream)
{
    if (bits == 4) return hip_sage_attn_int4(Qh, sQ, Kh, sK, dSh, Vt, sV, out, N, NqPad, NkPad, H, h0, Hc, stream);
    if (bits == 8) return hip_sage_attn_int8(Qh, sQ, Kh, sK, dSh, Vt, sV, out, N, NqPad, NkPad, H, h0, Hc, stream);
    return -1;
}
