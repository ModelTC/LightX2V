// BF16 activations, per-channel INT8 weights, and a BF16 output on Apple M5.
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;

constant constexpr int K = {{K}};
constant constexpr int N = {{N}};
constant constexpr int NT = K % 2048 == 0 ? 512 : 256;

kernel void quantize_x(device const bfloat* src, device int8_t* dst, device float* scales,
    uint row [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
    bfloat4 values[K / (NT * 4)];
    threadgroup float maxima[NT / 32];
    float amax = 0;
    #pragma clang loop unroll(full)
    for (int i = 0; i < K / (NT * 4); ++i) {
        values[i] = *reinterpret_cast<device const bfloat4*>(src + row*K + tid*4 + i*NT*4);
        float4 v = abs(float4(values[i]));
        amax = max(amax, max(max(v.x, v.y), max(v.z, v.w)));
    }
    amax = simd_max(amax);
    if (lane == 0) maxima[sg] = amax;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    amax = simd_max(lane < NT/32 ? maxima[lane] : 0.0f);
    if (tid == 0) scales[row] = amax > 0 ? amax / 127.0f : 1.0f;
    #pragma clang loop unroll(full)
    for (int i = 0; i < K / (NT * 4); ++i) {
        float4 v = amax > 0 ? (float4(values[i]) / amax) * 127.0f : float4(0);
        *reinterpret_cast<device char4*>(dst + row*K + tid*4 + i*NT*4) = char4(rint(v));
    }
}

inline uint compact(uint x) {
    x &= 0x55555555u; x = (x | (x >> 1)) & 0x33333333u;
    x = (x | (x >> 2)) & 0x0f0f0f0fu;
    x = (x | (x >> 4)) & 0x00ff00ffu;
    return (x | (x >> 8)) & 0x0000ffffu;
}

template<int BN>
kernel void gemm(device int8_t* xq, device int8_t* wq,
    device const float* xs, device const float* ws, device bfloat* output, device const bfloat* bias,
    // shape = {padded M, has_bias}; K and N are compile-time constants.
    constant uint2& shape, uint2 group [[threadgroup_position_in_grid]]) {
    constexpr int BM = 128, BK = BN * 8;
    const uint gn = N / BN, gm = shape.x / BM;
    const uint pid = group.y * gn + group.x;
    if constexpr (BN == 64) {
        uint first = (pid / (8 * gn)) * 8;
        uint height = min(8u, gm - first), within = pid % (8 * gn);
        group = uint2(within / height, first + within % height);
    } else if (gn % 8 == 0 && gm % 8 == 0) {
        uint region = pid / 64, local = pid % 64;
        group = uint2((region % (gn / 8)) * 8 + compact(local),
                      (region / (gn / 8)) * 8 + compact(local >> 1));
    }
    // TensorOps uses fastest-first axes; W[K,N] has K contiguous.
    using T = tensor<device int8_t, dextents<int,2>, tensor_inline>;
    T A(xq, dextents<int,2>(K, shape.x), array<int,2>{1, K});
    T B(wq, dextents<int,2>(K, N), array<int,2>{1, K});
    constexpr auto desc = matmul2d_descriptor(BM, BN, BK, false, true, false,
                                             matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc, execution_simdgroups<8>> op;
    auto a = A.template slice<BK,BM>(0, int(group.y)*BM);
    auto b = B.template slice<BK,BN>(0, int(group.x)*BN);
    auto c = op.template get_destination_cooperative_tensor<decltype(a),decltype(b),int>();
    for (ushort i = 0; i < c.get_capacity(); ++i) if (c.is_valid_element(i)) c[i] = 0;
    for (uint k = 0; k < K; k += BK) {
        auto aa = A.template slice<BK,BM>(int(k), int(group.y)*BM);
        auto bb = B.template slice<BK,BN>(int(k), int(group.x)*BN);
        op.run(aa, bb, c);
        threadgroup_barrier(mem_flags::mem_none);
    }
    auto d = op.template get_destination_cooperative_tensor<decltype(a),decltype(b),bfloat>();
    for (ushort i = 0; i < c.get_capacity(); ++i) if (c.is_valid_element(i)) {
        auto coord = c.get_multidimensional_index(i);
        uint col = group.x*BN + coord[0], row = group.y*BM + coord[1];
        float value = c[i] * xs[row] * ws[col];
        if (shape.y) value += float(bias[col]);
        d[i] = bfloat(value);
    }
    auto dest = tensor(output, dextents<int,2>(N,shape.x), array<int,2>{1,N});
    d.store(dest.template slice<BN,BM>(int(group.x)*BN, int(group.y)*BM));
}

template [[host_name("gemm_64")]] kernel void gemm<64>(device int8_t*, device int8_t*, device const float*, device const float*, device bfloat*, device const bfloat*, constant uint2&, uint2);
template [[host_name("gemm_128")]] kernel void gemm<128>(device int8_t*, device int8_t*, device const float*, device const float*, device bfloat*, device const bfloat*, constant uint2&, uint2);
