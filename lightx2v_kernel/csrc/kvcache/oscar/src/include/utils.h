/******************************************************************************
 * Copyright (c) 2023, Tri Dao.
 ******************************************************************************/

#pragma once

#include <assert.h>
#include <stdint.h>
#include <stdlib.h>

#include <cuda_fp16.h>

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
#include <cuda_bf16.h>
#endif

#include <cute/tensor.hpp>

#include <cutlass/array.h>
#include <cutlass/cutlass.h>
#include <cutlass/numeric_conversion.h>
#include <cutlass/numeric_types.h>
#include "dequantize.h"

#ifndef FLASH_FORCE_2BIT_SCALAR_PV
#define FLASH_FORCE_2BIT_SCALAR_PV 0
#endif

#define PRINT(name, content) \
    print(name);             \
    print(" : ");            \
    print(content);          \
    print("\n");

#define PRINTTENSOR(name, content) \
    print(name);                   \
    print(" : ");                  \
    print_tensor(content);         \
    print("\n");

////////////////////////////////////////////////////////////////////////////////////////////////////

namespace flash {

////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename T>
__forceinline__ __device__ uint32_t relu2(const uint32_t x);

template<>
__forceinline__ __device__ uint32_t relu2<cutlass::half_t>(const uint32_t x) {
    uint32_t res;
    const uint32_t zero = 0u;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("max.f16x2 %0, %1, %2;\n" : "=r"(res) : "r"(x), "r"(zero));
#else
    asm volatile( \
        "{\n" \
        "\t .reg .f16x2 sela;\n" \
        "\t set.gtu.u32.f16x2 sela, %1, %2;\n" \
        "\t and.b32 %0, sela, %1;\n"
        "}\n" : "=r"(res) : "r"(x), "r"(zero));
#endif
    return res;
}

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
template<>
__forceinline__ __device__ uint32_t relu2<cutlass::bfloat16_t>(const uint32_t x) {
    uint32_t res;
    const uint32_t zero = 0u;
    asm volatile("max.bf16x2 %0, %1, %2;\n" : "=r"(res) : "r"(x), "r"(zero));
    return res;
}
#endif

////////////////////////////////////////////////////////////////////////////////////////////////////

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800

template<typename T>
__forceinline__ __device__ uint32_t convert_relu2(const float2 x);

template<>
__forceinline__ __device__ uint32_t convert_relu2<cutlass::half_t>(const float2 x) {
    uint32_t res;
    const uint32_t a = reinterpret_cast<const uint32_t&>(x.x);
    const uint32_t b = reinterpret_cast<const uint32_t&>(x.y);
    asm volatile("cvt.rn.relu.f16x2.f32 %0, %1, %2;\n" : "=r"(res) : "r"(b), "r"(a));
    return res;
}

template<>
__forceinline__ __device__ uint32_t convert_relu2<cutlass::bfloat16_t>(const float2 x) {
    uint32_t res;
    const uint32_t a = reinterpret_cast<const uint32_t&>(x.x);
    const uint32_t b = reinterpret_cast<const uint32_t&>(x.y);
    asm volatile("cvt.rn.relu.bf16x2.f32 %0, %1, %2;\n" : "=r"(res) : "r"(b), "r"(a));
    return res;
}

#endif

////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename T>
struct MaxOp {
__device__ __forceinline__ T operator()(T const & x, T const & y) { return x > y ? x : y; }
};

template <>
struct MaxOp<float> {
// This is slightly faster
__device__ __forceinline__ float operator()(float const &x, float const &y) { return max(x, y); }
};

template<typename T>
struct MinOp {
__device__ __forceinline__ T operator()(T const & x, T const & y) { return x < y ? x : y; }
};

template <>
struct MinOp<float> {
// This is slightly faster
__device__ __forceinline__ float operator()(float const &x, float const &y) { return min(x, y); }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename T>
struct SumOp {
__device__ __forceinline__ T operator()(T const & x, T const & y) { return x + y; }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<int THREADS>
struct Allreduce {
    static_assert(THREADS == 32 || THREADS == 16 || THREADS == 8 || THREADS == 4);
    template<typename T, typename Operator>
    static __device__ __forceinline__ T run(T x, Operator &op) {
        constexpr int OFFSET = THREADS / 2;
        x = op(x, __shfl_xor_sync(uint32_t(-1), x, OFFSET));
        return Allreduce<OFFSET>::run(x, op);
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<>
struct Allreduce<2> {
template<typename T, typename Operator>
static __device__ __forceinline__ T run(T x, Operator &op) {
    x = op(x, __shfl_xor_sync(uint32_t(-1), x, 1));
    return x;
}
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<bool A_in_regs=false, bool B_in_regs=false, typename Tensor0, typename Tensor1,
         typename Tensor2, typename Tensor3, typename Tensor4,
         typename TiledMma, typename TiledCopyA, typename TiledCopyB,
         typename ThrCopyA, typename ThrCopyB>
__forceinline__ __device__ void gemm(Tensor0 &acc, Tensor1 &tCrA, Tensor2 &tCrB, Tensor3 const& tCsA,
                            Tensor4 const& tCsB, TiledMma tiled_mma,
                            TiledCopyA smem_tiled_copy_A, TiledCopyB smem_tiled_copy_B,
                            ThrCopyA smem_thr_copy_A, ThrCopyB smem_thr_copy_B) {
    CUTE_STATIC_ASSERT_V(size<1>(tCrA) == size<1>(acc));                     // MMA_M
    CUTE_STATIC_ASSERT_V(size<1>(tCrB) == size<2>(acc));                     // MMA_N
    CUTE_STATIC_ASSERT_V(size<2>(tCrA) == size<2>(tCrB));                     // MMA_K
    Tensor tCrA_copy_view = smem_thr_copy_A.retile_D(tCrA);
    CUTE_STATIC_ASSERT_V(size<1>(tCsA) == size<1>(tCrA_copy_view));            // M
    Tensor tCrB_copy_view = smem_thr_copy_B.retile_D(tCrB);
    CUTE_STATIC_ASSERT_V(size<1>(tCsB) == size<1>(tCrB_copy_view));            // N
    if (!A_in_regs) { cute::copy(smem_tiled_copy_A, tCsA(_, _, _0{}), tCrA_copy_view(_, _, _0{})); }
    if (!B_in_regs) { cute::copy(smem_tiled_copy_B, tCsB(_, _, _0{}), tCrB_copy_view(_, _, _0{})); }
    #pragma unroll
    for (int i = 0; i < size<2>(tCrA); ++i) {
        if (i < size<2>(tCrA) - 1) {
            if (!A_in_regs) { cute::copy(smem_tiled_copy_A, tCsA(_, _, i + 1), tCrA_copy_view(_, _, i + 1)); }
            if (!B_in_regs) { cute::copy(smem_tiled_copy_B, tCsB(_, _, i + 1), tCrB_copy_view(_, _, i + 1)); }
        }
        cute::gemm(tiled_mma, tCrA(_, _, i), tCrB(_, _, i), acc);
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template<int kBlockN, bool CheckBounds, typename Tensor0, typename Tensor1, typename Tensor2, typename Tensor3>
__forceinline__ __device__ void load_Vtensor_2bit_gather_fragment(
        Tensor0 &tCrB_dequant,
        Tensor1 const& sV_pack,
        Tensor2 const& sV_params,
        Tensor3 const& tOcVt,
        const int k_tile,
        const int n_valid) {
    static_assert(kBlockN == 256, "2bit V gather assumes the 256-token split-K tile layout");
    CUTE_STATIC_ASSERT_V(size<0>(tCrB_dequant) == Int<4>{});
    CUTE_STATIC_ASSERT_V(size<1>(tCrB_dequant) == Int<4>{});
    CUTE_STATIC_ASSERT_V(size<2>(tCrB_dequant) == Int<16>{});
    CUTE_STATIC_ASSERT_V(size<0>(tCrB_dequant) == size<0>(tOcVt));
    CUTE_STATIC_ASSERT_V(size<1>(tCrB_dequant) == size<1>(tOcVt));
    CUTE_STATIC_ASSERT_V(size<2>(tCrB_dequant) == size<2>(tOcVt));
    using Element = cute::remove_cvref_t<decltype(tCrB_dequant(0, 0, 0))>;
    cutlass::NumericConverter<Element, float> convert_op;

    const int dim_base = threadIdx.x >> 2;
    const int pack_col = dim_base & 15;
    const int slot_base = dim_base >> 4;
    const int token_base = (k_tile << 4) + ((threadIdx.x & 3) << 1);
    const int token0 = token_base;
    const int token1 = token_base + 1;
    const int token2 = token_base + 8;
    const int token3 = token_base + 9;

    uint16_t word0 = 0;
    uint16_t word1 = 0;
    uint16_t word2 = 0;
    uint16_t word3 = 0;
    if constexpr (CheckBounds) {
        if (token0 < n_valid) { word0 = uint16_t(sV_pack(token0, pack_col)); }
        if (token1 < n_valid) { word1 = uint16_t(sV_pack(token1, pack_col)); }
        if (token2 < n_valid) { word2 = uint16_t(sV_pack(token2, pack_col)); }
        if (token3 < n_valid) { word3 = uint16_t(sV_pack(token3, pack_col)); }
    } else {
        word0 = uint16_t(sV_pack(token0, pack_col));
        word1 = uint16_t(sV_pack(token1, pack_col));
        word2 = uint16_t(sV_pack(token2, pack_col));
        word3 = uint16_t(sV_pack(token3, pack_col));
    }

    CUTE_UNROLL
    for (int ni = 0; ni < 4; ++ni) {
        const int slot = slot_base + (ni << 1);
        const int group = ni;

        float val0 = 0.0f;
        if constexpr (CheckBounds) {
            if (token0 < n_valid) {
                const float q0 = float((word0 >> (2 * slot)) & 0x3u);
                const float2 scale_zero0 = quant::pair2_to_float2(sV_params(token0, group));
                val0 = q0 * scale_zero0.x + scale_zero0.y;
            }
        } else {
            const float q0 = float((word0 >> (2 * slot)) & 0x3u);
            const float2 scale_zero0 = quant::pair2_to_float2(sV_params(token0, group));
            val0 = q0 * scale_zero0.x + scale_zero0.y;
        }
        tCrB_dequant(0, ni, k_tile) = convert_op(val0);

        float val1 = 0.0f;
        if constexpr (CheckBounds) {
            if (token1 < n_valid) {
                const float q1 = float((word1 >> (2 * slot)) & 0x3u);
                const float2 scale_zero1 = quant::pair2_to_float2(sV_params(token1, group));
                val1 = q1 * scale_zero1.x + scale_zero1.y;
            }
        } else {
            const float q1 = float((word1 >> (2 * slot)) & 0x3u);
            const float2 scale_zero1 = quant::pair2_to_float2(sV_params(token1, group));
            val1 = q1 * scale_zero1.x + scale_zero1.y;
        }
        tCrB_dequant(1, ni, k_tile) = convert_op(val1);

        float val2 = 0.0f;
        if constexpr (CheckBounds) {
            if (token2 < n_valid) {
                const float q2 = float((word2 >> (2 * slot)) & 0x3u);
                const float2 scale_zero2 = quant::pair2_to_float2(sV_params(token2, group));
                val2 = q2 * scale_zero2.x + scale_zero2.y;
            }
        } else {
            const float q2 = float((word2 >> (2 * slot)) & 0x3u);
            const float2 scale_zero2 = quant::pair2_to_float2(sV_params(token2, group));
            val2 = q2 * scale_zero2.x + scale_zero2.y;
        }
        tCrB_dequant(2, ni, k_tile) = convert_op(val2);

        float val3 = 0.0f;
        if constexpr (CheckBounds) {
            if (token3 < n_valid) {
                const float q3 = float((word3 >> (2 * slot)) & 0x3u);
                const float2 scale_zero3 = quant::pair2_to_float2(sV_params(token3, group));
                val3 = q3 * scale_zero3.x + scale_zero3.y;
            }
        } else {
            const float q3 = float((word3 >> (2 * slot)) & 0x3u);
            const float2 scale_zero3 = quant::pair2_to_float2(sV_params(token3, group));
            val3 = q3 * scale_zero3.x + scale_zero3.y;
        }
        tCrB_dequant(3, ni, k_tile) = convert_op(val3);
    }
}

template<int kBlockN,
         bool A_in_regs=false,
         typename Tensor0, typename Tensor1, typename Tensor2,
         typename Tensor3, typename Tensor4, typename Tensor5, typename Tensor6,
         typename TiledMma, typename TiledCopyA, typename ThrCopyA>
__forceinline__ __device__ void gemm_Vtensor_2bit_gather(
        Tensor0 &acc,
        Tensor1 &tCrA,
        Tensor2 &tCrB_dequant,
        Tensor3 const& sV_pack,
        Tensor4 const& sV_params,
        Tensor5 const& tCsA,
        Tensor6 const& tOcVt,
        TiledMma tiled_mma,
        TiledCopyA smem_tiled_copy_A,
        ThrCopyA smem_thr_copy_A,
        const int n_valid) {
    CUTE_STATIC_ASSERT_V(size<1>(tCrA) == size<1>(acc));          // MMA_M
    CUTE_STATIC_ASSERT_V(size<1>(tCrB_dequant) == size<2>(acc));  // MMA_N
    CUTE_STATIC_ASSERT_V(size<2>(tCrA) == size<2>(tCrB_dequant)); // MMA_K
    CUTE_STATIC_ASSERT_V(size<0>(tCrB_dequant) == size<0>(tOcVt));
    CUTE_STATIC_ASSERT_V(size<1>(tCrB_dequant) == size<1>(tOcVt));
    CUTE_STATIC_ASSERT_V(size<2>(tCrB_dequant) == size<2>(tOcVt));

    Tensor tCrA_copy_view = smem_thr_copy_A.retile_D(tCrA);
    CUTE_STATIC_ASSERT_V(size<1>(tCsA) == size<1>(tCrA_copy_view)); // M

    if (!A_in_regs) {
        cute::copy(smem_tiled_copy_A, tCsA(_, _, _0{}), tCrA_copy_view(_, _, _0{}));
    }
    const bool check_bounds = n_valid != kBlockN;

    if (check_bounds) {
        flash::load_Vtensor_2bit_gather_fragment<kBlockN, true>(
            tCrB_dequant, sV_pack, sV_params, tOcVt, 0, n_valid
        );
        CUTE_UNROLL
        for (int i = 0; i < size<2>(tCrA); ++i) {
            if (i < size<2>(tCrA) - 1) {
                if (!A_in_regs) {
                    cute::copy(smem_tiled_copy_A, tCsA(_, _, i + 1), tCrA_copy_view(_, _, i + 1));
                }
                flash::load_Vtensor_2bit_gather_fragment<kBlockN, true>(
                    tCrB_dequant, sV_pack, sV_params, tOcVt, i + 1, n_valid
                );
            }
            cute::gemm(tiled_mma, tCrA(_, _, i), tCrB_dequant(_, _, i), acc);
        }
    } else {
        flash::load_Vtensor_2bit_gather_fragment<kBlockN, false>(
            tCrB_dequant, sV_pack, sV_params, tOcVt, 0, kBlockN
        );
        CUTE_UNROLL
        for (int i = 0; i < size<2>(tCrA); ++i) {
            if (i < size<2>(tCrA) - 1) {
                if (!A_in_regs) {
                    cute::copy(smem_tiled_copy_A, tCsA(_, _, i + 1), tCrA_copy_view(_, _, i + 1));
                }
                flash::load_Vtensor_2bit_gather_fragment<kBlockN, false>(
                    tCrB_dequant, sV_pack, sV_params, tOcVt, i + 1, kBlockN
                );
            }
            cute::gemm(tiled_mma, tCrA(_, _, i), tCrB_dequant(_, _, i), acc);
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template<int kBlockN, typename Tensor0, typename Tensor1, typename Tensor2, typename Tensor3, typename Tensor4>
__forceinline__ __device__ void gemm_Vtensor_2bit_scalar(
        Tensor0 &acc,
        Tensor1 const& sAcc,
        Tensor2 const& sV_pack,
        Tensor3 const& sV_params,
        Tensor4 const& taccOcO,
        const int n_valid) {
    static_assert(kBlockN % 16 == 0, "2bit scalar V path expects a tensor-core K multiple");

    CUTE_UNROLL
    for (int mma = 0; mma < size<0>(acc); ++mma) {
        CUTE_UNROLL
        for (int mi = 0; mi < size<1>(acc); ++mi) {
            CUTE_UNROLL
            for (int ni = 0; ni < size<2>(acc); ++ni) {
                const auto coord = taccOcO(mma, mi, ni);
                const int row = get<0>(coord);
                const int dim = get<1>(coord);
                const int pack_col = dim & 15;
                const int slot = dim >> 4;
                const int group = dim >> 5;

                float sum = 0.0f;
                for (int n = 0; n < kBlockN; ++n) {
                    if (n < n_valid) {
                        const uint16_t word = uint16_t(sV_pack(n, pack_col));
                        const float q = float((word >> (2 * slot)) & 0x3u);
                        const float2 scale_zero = quant::pair2_to_float2(sV_params(n, group));
                        sum += float(sAcc(row, n)) * (q * scale_zero.x + scale_zero.y);
                    }
                }
                acc(mma, mi, ni) += sum;
            }
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template<int num_bits,
         bool A_in_regs=false, bool B_in_regs=false,
         typename Tensor0, typename Tensor1,
         typename Tensor2_i4, typename Tensor2_dequant,
         typename Tensor2_scales, typename Tensor2_zeros, typename Tensor2_params,
         typename Tensor3,
         typename Tensor4_i4,
         typename TiledMma,
         typename TiledCopyA,
         typename TiledCopyB_i4,
         typename ThrCopyA,
         typename ThrCopyB_i4>
__forceinline__ __device__ void gemm_Vtensor(Tensor0 &acc, Tensor1 &tCrA,
                            Tensor2_i4 &tCrB_i4, Tensor2_dequant &tCrB_dequant,
                            Tensor2_scales &tCrB_scales, Tensor2_zeros &tCrB_zeros, Tensor2_params &sV_params,
                            Tensor3 const& tCsA,
                            Tensor4_i4 const& tCsB_i4,
                            TiledMma tiled_mma,
                            TiledCopyA smem_tiled_copy_A,
                            TiledCopyB_i4 smem_tiled_copy_B_i4,
                            ThrCopyA smem_thr_copy_A,
                            ThrCopyB_i4 smem_thr_copy_B_i4,
                            const int num_params) {
    CUTE_STATIC_ASSERT_V(size<1>(tCrA) == size<1>(acc));                     // MMA_M
    Tensor tCrA_copy_view = smem_thr_copy_A.retile_D(tCrA);
    CUTE_STATIC_ASSERT_V(size<1>(tCsA) == size<1>(tCrA_copy_view));            // M
    Tensor tCrB_i4_copy_view = smem_thr_copy_B_i4.retile_D(tCrB_i4);
    if (!A_in_regs) { cute::copy(smem_tiled_copy_A, tCsA(_, _, _0{}), tCrA_copy_view(_, _, _0{})); }
    if (!B_in_regs) {
        cute::copy(smem_tiled_copy_B_i4, tCsB_i4(_, _, _0{}), tCrB_i4_copy_view(_, _, _0{}));
        quant::load_params_Vtensor<num_bits>(tCrB_scales, tCrB_zeros, sV_params, threadIdx.x, 0, num_params);
        quant::dequant_Kchannel_Vtensor<num_bits>(tCrB_i4(_,_,_0{}), tCrB_dequant(_,_,_0{}), tCrB_scales(_,_0{}), tCrB_zeros(_,_0{}), num_params);
    }

    #pragma unroll
    for (int i = 0; i < size<2>(tCrA); ++i) {
        if (i < size<2>(tCrA) - 1) {
            if (!A_in_regs) { cute::copy(smem_tiled_copy_A, tCsA(_, _, i + 1), tCrA_copy_view(_, _, i + 1)); }
            if (!B_in_regs) {
                cute::copy(smem_tiled_copy_B_i4, tCsB_i4(_, _, i + 1), tCrB_i4_copy_view(_, _, i + 1));
                quant::load_params_Vtensor<num_bits>(tCrB_scales, tCrB_zeros, sV_params, threadIdx.x, i + 1, num_params);
                quant::dequant_Kchannel_Vtensor<num_bits>(tCrB_i4(_,_, i + 1), tCrB_dequant(_,_, i + 1), tCrB_scales(_,i + 1), tCrB_zeros(_, i + 1), num_params);
            }
        }
        cute::gemm(tiled_mma, tCrA(_, _, i), tCrB_dequant(_, _, i), acc);
    }

}

////////////////////////////////////////////////////////////////////////////////////////////////////

template<int num_bits,
         bool A_in_regs=false, bool B_in_regs=false,
         typename Tensor0, typename Tensor1,
         typename Tensor2_i4, typename Tensor2_dequant,
         typename Tensor2_scales, typename Tensor2_zeros, typename Tensor2_params,
         typename Tensor3,
         typename Tensor4_i4,
         typename TiledMma,
         typename TiledCopyA,
         typename TiledCopyB_i4,
         typename ThrCopyA,
         typename ThrCopyB_i4>
__forceinline__ __device__ void gemm_Kchannel(Tensor0 &acc, Tensor1 &tCrA,
                            Tensor2_i4 &tCrB_i4, Tensor2_dequant &tCrB_dequant,
                            Tensor2_scales &tCrB_scales, Tensor2_zeros &tCrB_zeros, Tensor2_params &sK_params,
                            Tensor3 const& tCsA,
                            Tensor4_i4 const& tCsB_i4,
                            TiledMma tiled_mma,
                            TiledCopyA smem_tiled_copy_A,
                            TiledCopyB_i4 smem_tiled_copy_B_i4,
                            ThrCopyA smem_thr_copy_A,
                            ThrCopyB_i4 smem_thr_copy_B_i4,
                            const int num_params) {
    CUTE_STATIC_ASSERT_V(size<1>(tCrA) == size<1>(acc));                     // MMA_M
    Tensor tCrA_copy_view = smem_thr_copy_A.retile_D(tCrA);
    CUTE_STATIC_ASSERT_V(size<1>(tCsA) == size<1>(tCrA_copy_view));            // M
    Tensor tCrB_i4_copy_view = smem_thr_copy_B_i4.retile_D(tCrB_i4);

    #pragma unroll
    for (int i = 0; i < size<2>(tCrA); ++i) {
        quant::load_params_Kchannel(tCrB_scales, tCrB_zeros, sK_params, threadIdx.x, i, num_params);
    }

    if (!A_in_regs) { cute::copy(smem_tiled_copy_A, tCsA(_, _, _0{}), tCrA_copy_view(_, _, _0{})); }
    if (!B_in_regs) {
        cute::copy(smem_tiled_copy_B_i4, tCsB_i4(_, _, _0{}), tCrB_i4_copy_view(_, _, _0{}));
        quant::dequant_Kchannel_Vtensor<num_bits>(tCrB_i4(_,_,_0{}), tCrB_dequant(_,_,_0{}), tCrB_scales(_,_,_0{}), tCrB_zeros(_,_,_0{}), num_params);
    }

    #pragma unroll
    for (int i = 0; i < size<2>(tCrA); ++i) {
        if (i < size<2>(tCrA) - 1) {
            if (!A_in_regs) { cute::copy(smem_tiled_copy_A, tCsA(_, _, i + 1), tCrA_copy_view(_, _, i + 1)); }
            if (!B_in_regs) {
                cute::copy(smem_tiled_copy_B_i4, tCsB_i4(_, _, i + 1), tCrB_i4_copy_view(_, _, i + 1));
                quant::dequant_Kchannel_Vtensor<num_bits>(tCrB_i4(_, _, i + 1), tCrB_dequant(_, _, i + 1), tCrB_scales(_, _, i + 1), tCrB_zeros(_, _, i + 1), num_params);
            }
        }
        cute::gemm(tiled_mma, tCrA(_, _, i), tCrB_dequant(_, _, i), acc);
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template<bool A_in_regs=false, bool B_in_regs=false,
         typename Tensor0, typename Tensor1,
         typename Tensor2_i4, typename Tensor2_dequant,
         typename Tensor2_scales, typename Tensor2_zeros,
         typename Tensor3,
         typename Tensor4_i4,
         typename TiledMma,
         typename TiledCopyA,
         typename TiledCopyB_i4,
         typename ThrCopyA,
         typename ThrCopyB_i4>
__forceinline__ __device__ void gemm_Ktensor(Tensor0 &acc, Tensor1 &tCrA,
                            Tensor2_i4 &tCrB_i4, Tensor2_dequant &tCrB_dequant,
                            Tensor2_scales &tCrB_scales, Tensor2_zeros &tCrB_zeros,
                            Tensor3 const& tCsA,
                            Tensor4_i4 const& tCsB_i4,
                            TiledMma tiled_mma,
                            TiledCopyA smem_tiled_copy_A,
                            TiledCopyB_i4 smem_tiled_copy_B_i4,
                            ThrCopyA smem_thr_copy_A,
                            ThrCopyB_i4 smem_thr_copy_B_i4,
                            const int group_size) {
    CUTE_STATIC_ASSERT_V(size<1>(tCrA) == size<1>(acc));                     // MMA_M
    Tensor tCrA_copy_view = smem_thr_copy_A.retile_D(tCrA);
    CUTE_STATIC_ASSERT_V(size<1>(tCsA) == size<1>(tCrA_copy_view));            // M
    Tensor tCrB_i4_copy_view = smem_thr_copy_B_i4.retile_D(tCrB_i4);
    if (!A_in_regs) { cute::copy(smem_tiled_copy_A, tCsA(_, _, _0{}), tCrA_copy_view(_, _, _0{})); }
    if (!B_in_regs) {
        // cute::copy(smem_tiled_copy_B_i4, tCsB_i4(_, _, _0{}), tCrB_i4_copy_view(_, _, _0{}));
        quant::dequantize_Ktensor(tCrB_i4, tCrB_dequant, tCrB_scales, tCrB_zeros, 4, group_size, 0);
    }
    #pragma unroll
    for (int i = 0; i < size<2>(tCrA); ++i) {
        if (i < size<2>(tCrA) - 1) {
            if (!A_in_regs) { cute::copy(smem_tiled_copy_A, tCsA(_, _, i + 1), tCrA_copy_view(_, _, i + 1)); }
            if (!B_in_regs) {
                // cute::copy(smem_tiled_copy_B_i4, tCsB_i4(_, _, i + 1), tCrB_i4_copy_view(_, _, i + 1));
                quant::dequantize_Ktensor(tCrB_i4, tCrB_dequant, tCrB_scales, tCrB_zeros, 4, group_size, i + 1);
            }
        }
        cute::gemm(tiled_mma, tCrA(_, _, i), tCrB_dequant(_, _, i), acc);
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template<bool A_in_regs=false, bool B_in_regs=false, typename Tensor0, typename Tensor1,
         typename Tensor2, typename Tensor3, typename Tensor4,
         typename TiledMma, typename TiledCopyA, typename TiledCopyB,
         typename ThrCopyA, typename ThrCopyB>
__forceinline__ __device__ void gemm_residual(Tensor0 &acc, Tensor1 &tCrA, Tensor2 &tCrB, Tensor3 const& tCsA,
                            Tensor4 const& tCsB, TiledMma tiled_mma,
                            TiledCopyA smem_tiled_copy_A, TiledCopyB smem_tiled_copy_B,
                            ThrCopyA smem_thr_copy_A, ThrCopyB smem_thr_copy_B) {
    CUTE_STATIC_ASSERT_V(size<1>(tCrA) == size<1>(acc));                      // MMA_M
    CUTE_STATIC_ASSERT_V(size<1>(tCrB) == size<2>(acc));                      // MMA_N
    CUTE_STATIC_ASSERT_V(size<2>(tCrA) == size<2>(tCrB));                     // MMA_K
    Tensor tCrA_copy_view = smem_thr_copy_A.retile_D(tCrA);
    CUTE_STATIC_ASSERT_V(size<1>(tCsA) == size<1>(tCrA_copy_view));            // M
    Tensor tCrB_copy_view = smem_thr_copy_B.retile_D(tCrB);
    CUTE_STATIC_ASSERT_V(size<1>(tCsB) == size<1>(tCrB_copy_view));            // N
    if (!A_in_regs) { cute::copy(smem_tiled_copy_A, tCsA(_, _, _0{}), tCrA_copy_view(_, _, _0{})); }
    if (!B_in_regs) { cute::copy(smem_tiled_copy_B, tCsB(_, _, _0{}), tCrB_copy_view(_, _, _0{})); }

    #pragma unroll
    for (int i = 0; i < size<2>(tCrA); ++i) {
        if (i < size<2>(tCrA) - 1) {
            if (!A_in_regs) { cute::copy(smem_tiled_copy_A, tCsA(_, _, i + 1), tCrA_copy_view(_, _, i + 1)); }
            if (!B_in_regs) { cute::copy(smem_tiled_copy_B, tCsB(_, _, i + 1), tCrB_copy_view(_, _, i + 1)); }
        }
        cute::gemm(tiled_mma, tCrA(_, _, i), tCrB(_, _, i), acc);
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename Tensor0, typename Tensor1, typename Tensor2, typename Tensor3,
         typename TiledMma, typename TiledCopy, typename ThrCopy>
__forceinline__ __device__ void gemm_rs(Tensor0 &acc, Tensor1 &tCrA, Tensor2 &tCrB, Tensor3 const& tCsB,
                               TiledMma tiled_mma, TiledCopy smem_tiled_copy_B,
                               ThrCopy smem_thr_copy_B) {
    CUTE_STATIC_ASSERT_V(size<1>(tCrA) == size<1>(acc));                     // MMA_M
    CUTE_STATIC_ASSERT_V(size<1>(tCrB) == size<2>(acc));                     // MMA_N
    CUTE_STATIC_ASSERT_V(size<2>(tCrA) == size<2>(tCrB));                     // MMA_K
    Tensor tCrB_copy_view = smem_thr_copy_B.retile_D(tCrB);
    CUTE_STATIC_ASSERT_V(size<1>(tCsB) == size<1>(tCrB_copy_view));            // N
    cute::copy(smem_tiled_copy_B, tCsB(_, _, _0{}), tCrB_copy_view(_, _, _0{}));

    #pragma unroll
    for (int i = 0; i < size<2>(tCrA); ++i) {
        if (i < size<2>(tCrA) - 1) {
            cute::copy(smem_tiled_copy_B, tCsB(_, _, i + 1), tCrB_copy_view(_, _, i + 1));
        }
        cute::gemm(tiled_mma, tCrA(_, _, i), tCrB(_, _, i), acc);
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// Convert acc_layout from (MMA=4, MMA_M, MMA_N) to (nrow=(2, MMA_M), ncol=(2, MMA_N))
template<typename Layout>
__forceinline__ __device__ auto convert_layout_acc_rowcol(Layout acc_layout) {
    static_assert(decltype(size<0>(acc_layout))::value == 4);
    static_assert(decltype(rank(acc_layout))::value == 3);
    auto l = logical_divide(acc_layout, Shape<_2>{});  // ((2, 2), MMA_M, MMA_N)
    return make_layout(make_layout(get<0, 1>(l), get<1>(l)), make_layout(get<0, 0>(l), get<2>(l)));
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// Convert acc_layout from (MMA=4, MMA_M, MMA_N) to ((4, 2), MMA_M, MMA_N / 2)
// if using m16n8k16, or to (4, MMA_M, MMA_N) if using m16n8k8.
template<typename MMA_traits, typename Layout>
__forceinline__ __device__ auto convert_layout_acc_Aregs(Layout acc_layout) {
    using X = Underscore;
    static_assert(decltype(size<0>(acc_layout))::value == 4);
    static_assert(decltype(rank(acc_layout))::value == 3);
    constexpr int mma_shape_K = get<2>(typename MMA_traits::Shape_MNK{});
    static_assert(mma_shape_K == 8 || mma_shape_K == 16);
    if constexpr (mma_shape_K == 8) {
        return acc_layout;
    } else {
        auto l = logical_divide(acc_layout, Shape<X, X, _2>{});  // (4, MMA_M, (2, MMA_N / 2)))
        return make_layout(make_layout(get<0>(l), get<2, 0>(l)), get<1>(l), get<2, 1>(l));
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// Convert acc_layout from (MMA=4, MMA_M, MMA_N) to ((4, 2), MMA_M, MMA_N / 2)
template<typename Layout>
__forceinline__ __device__ auto convert_layout_acc_dropout(Layout acc_layout) {
    using X = Underscore;
    static_assert(decltype(size<0>(acc_layout))::value == 4);
    static_assert(decltype(rank(acc_layout))::value == 3);
    auto l = logical_divide(acc_layout, Shape<X, X, _2>{});  // (4, MMA_M, (2, MMA_N / 2)))
    return make_layout(make_layout(get<0>(l), get<2, 0>(l)), get<1>(l), get<2, 1>(l));
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename To_type, typename Engine, typename Layout>
__forceinline__ __device__ auto convert_type(Tensor<Engine, Layout> const &tensor) {
    using From_type = typename Engine::value_type;
    constexpr int numel = decltype(size(tensor))::value;
    cutlass::NumericArrayConverter<To_type, From_type, numel> convert_op;
    // HACK: this requires tensor to be "contiguous"
    auto frag = convert_op(*reinterpret_cast<const cutlass::Array<From_type, numel> *>(tensor.data()));
    return make_tensor(make_rmem_ptr<To_type>(&frag), tensor.layout());
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Engine, typename Layout>
__forceinline__ __device__ void relu_(Tensor<Engine, Layout> &tensor) {
    constexpr int numel = decltype(size(tensor))::value;
    static_assert(numel % 2 == 0);
    using value_t = typename Engine::value_type;
    // HACK: this requires tensor to be "contiguous"
    Tensor tensor_uint32 = recast<uint32_t>(tensor);
    #pragma unroll
    for (int i = 0; i < size(tensor_uint32); ++i) {
        tensor_uint32(i) = relu2<value_t>(tensor_uint32(i));
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// On SM80 and above, we can fuse fp32 -> fp16/bf16 conversion and relu into 1 instruction
template <typename To_type, typename Engine, typename Layout>
__forceinline__ __device__ auto convert_type_relu(Tensor<Engine, Layout> const &tensor) {
    using From_type = typename Engine::value_type;
    static_assert(std::is_same_v<To_type, cutlass::half_t> || std::is_same_v<To_type, cutlass::bfloat16_t>);
    static_assert(std::is_same_v<float, From_type>);
    constexpr int numel = decltype(size(tensor))::value;
    static_assert(numel % 2 == 0);
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    // HACK: this requires tensor to be "contiguous"
    Tensor tensor_float2 = recast<float2>(tensor);
    Tensor out_uint32 = make_tensor<uint32_t>(tensor_float2.layout());
    #pragma unroll
    for (int i = 0; i < size(out_uint32); ++i) {
        out_uint32(i) = convert_relu2<To_type>(tensor_float2(i));
    }
    Tensor out = make_tensor(make_rmem_ptr<To_type>(out_uint32.data()), tensor.layout());
#else
    Tensor out = flash::convert_type<To_type>(tensor);
    flash::relu_(out);
#endif
    return out;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// Blocks until all but N previous cp.async.commit_group operations have committed.
// This differs from cute::cp_async_wait in that when N = 0 we don't call cp.async.wait_all
// (which is equivalent to commit_group then wait_group 0).
// Instead we just call cp.async.wait_group 0, which is slightly faster.
// https://github.com/NVIDIA/cutlass/blob/master/include/cute/arch/copy_sm80.hpp#L113
template <int N>
CUTE_HOST_DEVICE
void cp_async_wait() {
#if defined(CUTE_ARCH_CP_ASYNC_SM80_ENABLED)
    asm volatile("cp.async.wait_group %0;\n" :: "n"(N));
#endif
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <bool Is_even_MN=true, bool Is_even_K=true, bool Clear_OOB_MN=false, bool Clear_OOB_K=true,
          typename TiledCopy, typename Engine0, typename Layout0, typename Engine1, typename Layout1,
          typename Engine2, typename Layout2, typename Engine3, typename Layout3>
__forceinline__ __device__ void copy(TiledCopy tiled_copy, Tensor<Engine0, Layout0> const &S,
                            Tensor<Engine1, Layout1> &D, Tensor<Engine2, Layout2> const &identity_MN,
                            Tensor<Engine3, Layout3> const &predicate_K, const int max_MN=0) {
    CUTE_STATIC_ASSERT_V(rank(S) == Int<3>{});
    CUTE_STATIC_ASSERT_V(rank(D) == Int<3>{});
    CUTE_STATIC_ASSERT_V(size<0>(S) == size<0>(D));                     // MMA
    CUTE_STATIC_ASSERT_V(size<1>(S) == size<1>(D));                     // MMA_M
    CUTE_STATIC_ASSERT_V(size<2>(S) == size<2>(D));                     // MMA_K
    // There's no case where !Clear_OOB_K && Clear_OOB_MN
    static_assert(!(Clear_OOB_MN && !Clear_OOB_K));
    #pragma unroll
    for (int m = 0; m < size<1>(S); ++m) {
        if (Is_even_MN || get<0>(identity_MN(0, m, 0)) < max_MN) {
            #pragma unroll
            for (int k = 0; k < size<2>(S); ++k) {
                if (Is_even_K || predicate_K(k)) {
                    cute::copy(tiled_copy, S(_, m, k), D(_, m, k));
                } else if (Clear_OOB_K) {
                    cute::clear(D(_, m, k));
                }
            }
        } else if (Clear_OOB_MN) {
            cute::clear(D(_, m, _));
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <bool Is_even_K=true,
          typename Engine0, typename Layout0, typename Engine1, typename Layout1,
          typename Engine2, typename Layout2, typename Engine3, typename Layout3>
__forceinline__ __device__ void copy_w_min_idx(Tensor<Engine0, Layout0> const &S,
                                      Tensor<Engine1, Layout1> &D, Tensor<Engine2, Layout2> const &identity_MN,
                                      Tensor<Engine3, Layout3> const &predicate_K,
                                      const int max_MN=0, const int min_MN=0) {
    CUTE_STATIC_ASSERT_V(rank(S) == Int<3>{});
    CUTE_STATIC_ASSERT_V(rank(D) == Int<3>{});
    CUTE_STATIC_ASSERT_V(size<0>(S) == size<0>(D));                     // MMA
    CUTE_STATIC_ASSERT_V(size<1>(S) == size<1>(D));                     // MMA_M
    CUTE_STATIC_ASSERT_V(size<2>(S) == size<2>(D));                     // MMA_K
    // if (threadIdx.x == 0 && blockIdx.z == 0) { printf("blockIdx.y = %d, max_MN = %d, min_MN = %d\n", blockIdx.y, max_MN, min_MN); }
    #pragma unroll
    for (int m = 0; m < size<1>(S); ++m) {
        // if (threadIdx.x == 0 && blockIdx.z == 0) { printf("blockIdx.y = %d, m = %d\n", blockIdx.y, get<0>(identity_MN(0, m, 0))); }
        if (get<0>(identity_MN(0, m, 0)) >= min_MN && get<0>(identity_MN(0, m, 0)) < max_MN) {
            // if (threadIdx.x == 0 && blockIdx.z == 0) { printf("Inner loop, blockIdx.y = %d, m = %d\n", blockIdx.y, get<0>(identity_MN(0, m, 0))); }
            #pragma unroll
            for (int k = 0; k < size<2>(S); ++k) {
                if (Is_even_K || predicate_K(k)) {
                    cute::copy(S(_, m, k), D(_, m, k));
                }
            }
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace flash
