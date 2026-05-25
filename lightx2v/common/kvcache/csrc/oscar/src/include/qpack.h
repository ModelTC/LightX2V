#pragma once

#include <cute/tensor.hpp>
#include <cutlass/numeric_types.h>
#include "utils.h"

namespace quant {

using namespace cute;

__device__ __forceinline__ void compute_affine_asym_params(
    float min_i,
    float max_i,
    float max_val,
    float &scale_inv,
    float &zero_q,
    float &zero_dequant
) {
    if (min_i == 0.0f && max_i == 0.0f) {
        min_i = -1.0f;
        max_i = 1.0f;
    }
    const float range = max_i - min_i;
    if (range > 0.0f) {
        const float scale = range / max_val;
        scale_inv = max_val / range;
        zero_q = nearbyintf(-min_i / scale);
        zero_dequant = -zero_q * scale;
    } else {
        scale_inv = 0.0f;
        zero_q = 0.0f;
        zero_dequant = min_i;
    }
}

template<typename Tensor0, typename Tensor1, typename Operator>
CUTE_DEVICE
void thread_reduce_(Tensor0 const& tensor, Tensor1& summary, Operator& op, const int num_params) {
    const int pack_num = size<1>(tensor) / num_params;

    CUTE_UNROLL
    for (int mi = 0; mi < size<0>(summary); ++mi) {
        int col_start = (mi / 4) * pack_num;
        summary(mi) = tensor(mi % 4, col_start);

        CUTE_UNROLL
        for (int ni = col_start; ni < col_start + pack_num; ++ni) {
            summary(mi) = op(summary(mi), tensor(mi % 4, ni));
        }

    }

}

template<typename T, typename Operator>
__device__ __forceinline__ T warp_reduce(T val, Operator op) {
    // Get the thread's position within its group of 4
    const int lane_id = threadIdx.x % 32;  // Lane ID within warp
    const int group_pos = lane_id % 4;     // Position within group of 4

    // Only reduce with threads that have the same position in their group of 4
    // Using butterfly pattern with xor
    for (int mask = 16; mask > 0; mask >>= 1) {
        T other = __shfl_xor_sync(0xffffffff, val, mask);
        // Only combine if the other thread has the same group_pos
        if ((lane_id ^ mask) < 32 && ((lane_id ^ mask) % 4 == group_pos)) {
            val = op(val, other);
        }
    }
    return val;
}

template<typename Tensor0, typename Tensor1, typename Tensor2, typename Operator>
CUTE_DEVICE
void allreduce_(Tensor0 &dst, Tensor1 &src, Tensor2 &reduce_tmp, Operator &op) {
    CUTE_STATIC_ASSERT_V(size(dst) == size(src));

    const int warp_id = threadIdx.x / 32;
    const int lane_id = threadIdx.x % 32;

    #pragma unroll
    for (int i = 0; i < size(dst); i++) {
        // First do reduction within each group of 4 threads
        float val = quant::warp_reduce(src(i), op);
        // Write the result to shared memory for each group's leader
        if (lane_id < 4) {
            auto &slot = reduce_tmp(i, warp_id * 4 + lane_id);
            using SlotType = cute::remove_cvref_t<decltype(slot)>;
            slot = SlotType(val);
        }
        __syncthreads();

        // First thread in the first group reads all values and reduces them
        if (lane_id < 4) {
            float final_val = float(reduce_tmp(i, 0 + lane_id));
            #pragma unroll
            for (int w = 1; w < 4; w++) {  // For 4 warps
                final_val = op(final_val, float(reduce_tmp(i, w * 4 + lane_id)));
            }
            // Write back the final result
            auto &slot = reduce_tmp(i, 0 + lane_id);
            using SlotType = cute::remove_cvref_t<decltype(slot)>;
            slot = SlotType(final_val);
        }
        __syncthreads();

        // All threads read the final result
        dst(i) = reduce_tmp(i,0 + lane_id % 4);

    }


}

template<typename Engine0, typename Layout0, typename Engine1, typename Layout1, typename Tensor2, typename Operator>
CUTE_DEVICE
void reduce_(Tensor<Engine0, Layout0> const& tensor, Tensor<Engine1, Layout1>& summary, Tensor2 &reduce_tmp, Operator& op, const int num_params) {
    quant::thread_reduce_(tensor, summary, op, num_params);
    quant::allreduce_(summary, summary, reduce_tmp, op);
}

template<typename Engine0, typename Layout0, typename Engine1, typename Layout1, typename Tensor2>
CUTE_DEVICE
void reduce_max(Tensor<Engine0, Layout0> const& tensor, Tensor<Engine1, Layout1> &max, Tensor2 &reduce_tmp, const int num_params) {
    flash::MaxOp<float> max_op;
    quant::reduce_(tensor, max, reduce_tmp, max_op, num_params);  // Use the existing reduce_q function
}

template<typename Engine0, typename Layout0, typename Engine1, typename Layout1, typename Tensor2>
CUTE_DEVICE
void reduce_min(Tensor<Engine0, Layout0> const& tensor, Tensor<Engine1, Layout1> &min, Tensor2 &reduce_tmp, const int num_params) {
    flash::MinOp<float> min_op;
    quant::reduce_(tensor, min, reduce_tmp, min_op, num_params);  // Use the existing reduce_q function
}

template<int num_bits, typename Tensor1, typename Tensor2, typename Tensor3, typename Tensor4, typename Tensor5>
struct qpack_kc_vt;

template<typename Tensor1, typename Tensor2, typename Tensor3, typename Tensor4, typename Tensor5>
struct qpack_kc_vt<2, Tensor1, Tensor2, Tensor3, Tensor4, Tensor5> {
    static constexpr int num_bits = 2;  // Add this line
    CUTE_DEVICE static
    void apply(Tensor1 &src, Tensor2 &dst, Tensor3 &scales_k, Tensor4 &zeros_k, Tensor5 &reduce_tmp, const int num_params) {
        const float max_val      = float((1 << num_bits) - 1);
        const int num_params_2   = size<1>(src) == 4 && num_params > 1 ? num_params / 2 : num_params;
        const int pack_num       = size<1>(src) / num_params_2;
        const int channel_stride = size<0>(src);

        // Declare per-channel tensors
        using TensorChannel = decltype(make_fragment_like<float>(scales_k(_, 0)));
        TensorChannel channel_max, channel_min, channel_range, channel_scales_inv, channel_zeros, channel_zero_q;

        CUTE_UNROLL
        for (int k = 0; k < size<2>(src); ++k) {
            // Perform per-channel max and min reductions
            quant::reduce_max(src(_, _, k), channel_max, reduce_tmp, num_params_2);
            quant::reduce_min(src(_, _, k), channel_min, reduce_tmp, num_params_2);

            // Compute per-channel scale inverses and zeros
            CUTE_UNROLL
            for (int i = 0; i < size(channel_max); ++i) {
                float max_i = float(channel_max(i));
                float min_i = float(channel_min(i));
                float scale_inv = 0.0f;
                float zero_q = 0.0f;
                float zero_dequant = 0.0f;
                compute_affine_asym_params(min_i, max_i, max_val, scale_inv, zero_q, zero_dequant);
                channel_scales_inv(i) = scale_inv;
                channel_zero_q(i) = zero_q;
                channel_zeros(i) = zero_dequant;
                // Store scales and zeros
                {
                    auto &s_slot = scales_k(i, k);
                    using ST = cute::remove_cvref_t<decltype(s_slot)>;
                    s_slot = ST(scale_inv == 0 ? 0.0f : 1.0f / scale_inv);
                }
                {
                    auto &z_slot = zeros_k(i, k);
                    using ZT = cute::remove_cvref_t<decltype(z_slot)>;
                    z_slot = ZT(zero_dequant);
                }
            }

            // Quantize and pack the tensor
            CUTE_UNROLL
            for (int i = 0; i < size<0>(src); ++i) {

                CUTE_UNROLL
                for (int jj = 0; jj < size<1>(src); jj += 8) {
                    // float val0 = float(src(i, jj,     k));
                    // float val1 = float(src(i, jj + 1, k));
                    // float val2 = float(src(i, jj + 2, k));
                    // float val3 = float(src(i, jj + 3, k));
                    // float val4 = float(src(i, jj + 4, k));
                    // float val5 = float(src(i, jj + 5, k));
                    // float val6 = float(src(i, jj + 6, k));
                    // float val7 = float(src(i, jj + 7, k));

                    // Load 4 values and convert to float
                    float val0 = float(src(i, jj,     k));
                    float val1 = float(src(i, jj + 1, k));
                    float val2 = float(src(i, jj + 2, k));
                    float val3 = float(src(i, jj + 3, k));
                    float val4 = float(src(i, jj + 4, k));
                    float val5 = float(src(i, jj + 5, k));
                    float val6 = float(src(i, jj + 6, k));
                    float val7 = float(src(i, jj + 7, k));

                    // Apply scale inverses
                    val0 = val0 * channel_scales_inv(i + (jj    ) / pack_num * channel_stride) + channel_zero_q(i + (jj    ) / pack_num * channel_stride);
                    val1 = val1 * channel_scales_inv(i + (jj + 1) / pack_num * channel_stride) + channel_zero_q(i + (jj + 1) / pack_num * channel_stride);
                    val2 = val2 * channel_scales_inv(i + (jj + 2) / pack_num * channel_stride) + channel_zero_q(i + (jj + 2) / pack_num * channel_stride);
                    val3 = val3 * channel_scales_inv(i + (jj + 3) / pack_num * channel_stride) + channel_zero_q(i + (jj + 3) / pack_num * channel_stride);
                    val4 = val4 * channel_scales_inv(i + (jj + 4) / pack_num * channel_stride) + channel_zero_q(i + (jj + 4) / pack_num * channel_stride);
                    val5 = val5 * channel_scales_inv(i + (jj + 5) / pack_num * channel_stride) + channel_zero_q(i + (jj + 5) / pack_num * channel_stride);
                    val6 = val6 * channel_scales_inv(i + (jj + 6) / pack_num * channel_stride) + channel_zero_q(i + (jj + 6) / pack_num * channel_stride);
                    val7 = val7 * channel_scales_inv(i + (jj + 7) / pack_num * channel_stride) + channel_zero_q(i + (jj + 7) / pack_num * channel_stride);

                    // Round and clamp the values
                    val0 = fminf(fmaxf(nearbyintf(val0), 0.0f), max_val);
                    val1 = fminf(fmaxf(nearbyintf(val1), 0.0f), max_val);
                    val2 = fminf(fmaxf(nearbyintf(val2), 0.0f), max_val);
                    val3 = fminf(fmaxf(nearbyintf(val3), 0.0f), max_val);
                    val4 = fminf(fmaxf(nearbyintf(val4), 0.0f), max_val);
                    val5 = fminf(fmaxf(nearbyintf(val5), 0.0f), max_val);
                    val6 = fminf(fmaxf(nearbyintf(val6), 0.0f), max_val);
                    val7 = fminf(fmaxf(nearbyintf(val7), 0.0f), max_val);

                    // Pack 8 values (2-bit each) into a 16-bit integer
                    uint16_t packed = 0;
                    packed |= (static_cast<uint16_t>(static_cast<uint8_t>(val7)) & 0x3);  // 2 bits
                    packed <<= 2;
                    packed |= (static_cast<uint16_t>(static_cast<uint8_t>(val6)) & 0x3);
                    packed <<= 2;
                    packed |= (static_cast<uint16_t>(static_cast<uint8_t>(val5)) & 0x3);
                    packed <<= 2;
                    packed |= (static_cast<uint16_t>(static_cast<uint8_t>(val4)) & 0x3);
                    packed <<= 2;
                    packed |= (static_cast<uint16_t>(static_cast<uint8_t>(val3)) & 0x3);
                    packed <<= 2;
                    packed |= (static_cast<uint16_t>(static_cast<uint8_t>(val2)) & 0x3);
                    packed <<= 2;
                    packed |= (static_cast<uint16_t>(static_cast<uint8_t>(val1)) & 0x3);
                    packed <<= 2;
                    packed |= (static_cast<uint16_t>(static_cast<uint8_t>(val0)) & 0x3);

                    // Store the packed value
                    dst(i, jj / 8, k) = packed;
                }
            }
        }

    }


};

template<typename Tensor1, typename Tensor2, typename Tensor3, typename Tensor4, typename Tensor5>
struct qpack_kc_vt<4, Tensor1, Tensor2, Tensor3, Tensor4, Tensor5> {
    static constexpr int num_bits = 4;  // Add this line
    CUTE_DEVICE static
    void apply(Tensor1 &src, Tensor2 &dst, Tensor3 &scales_k, Tensor4 &zeros_k, Tensor5 &reduce_tmp, const int num_params) {
        const float max_val      = float((1 << num_bits) - 1);
        const int pack_num       = size<1>(src) / num_params;
        const int channel_stride = size<0>(src);

        // Declare per-channel tensors
        using TensorChannel = decltype(make_fragment_like<float>(scales_k(_, 0)));
        TensorChannel channel_max, channel_min, channel_range, channel_scales_inv, channel_zeros, channel_zero_q;


        CUTE_UNROLL
        for (int k = 0; k < size<2>(src); ++k) {
            // Perform per-channel max and min reductions
            quant::reduce_max(src(_, _, k), channel_max, reduce_tmp, num_params);
            quant::reduce_min(src(_, _, k), channel_min, reduce_tmp, num_params);

            // Compute per-channel scale inverses and zeros
            CUTE_UNROLL
            for (int i = 0; i < size(channel_max); ++i) {
                float max_i = float(channel_max(i));
                float min_i = float(channel_min(i));
                float scale_inv = 0.0f;
                float zero_q = 0.0f;
                float zero_dequant = 0.0f;
                compute_affine_asym_params(min_i, max_i, max_val, scale_inv, zero_q, zero_dequant);
                channel_scales_inv(i) = scale_inv;
                channel_zero_q(i) = zero_q;
                channel_zeros(i) = zero_dequant;
                // Store scales and zeros
                {
                    auto &s_slot = scales_k(i, k);
                    using ST = cute::remove_cvref_t<decltype(s_slot)>;
                    s_slot = ST(scale_inv == 0 ? 0.0f : 1.0f / scale_inv);
                }
                {
                    auto &z_slot = zeros_k(i, k);
                    using ZT = cute::remove_cvref_t<decltype(z_slot)>;
                    z_slot = ZT(zero_dequant);
                }
            }

            // Quantize and pack the tensor
            CUTE_UNROLL
            for (int i = 0; i < size<0>(src); ++i) {

                CUTE_UNROLL
                for (int jj = 0; jj < size<1>(src); jj += 4) {
                    // float val0 = float(src(i, jj,     k));
                    // float val1 = float(src(i, jj + 1, k));
                    // float val2 = float(src(i, jj + 2, k));
                    // float val3 = float(src(i, jj + 3, k));

                    // Load 4 values and convert to float
                    float val0 = float(src(i, jj,     k));
                    float val1 = float(src(i, jj + 1, k));
                    float val2 = float(src(i, jj + 2, k));
                    float val3 = float(src(i, jj + 3, k));

                    // Apply scale inverses
                    val0 = val0 * channel_scales_inv(i + (jj    ) / pack_num * channel_stride) + channel_zero_q(i + (jj    ) / pack_num * channel_stride);
                    val1 = val1 * channel_scales_inv(i + (jj + 1) / pack_num * channel_stride) + channel_zero_q(i + (jj + 1) / pack_num * channel_stride);
                    val2 = val2 * channel_scales_inv(i + (jj + 2) / pack_num * channel_stride) + channel_zero_q(i + (jj + 2) / pack_num * channel_stride);
                    val3 = val3 * channel_scales_inv(i + (jj + 3) / pack_num * channel_stride) + channel_zero_q(i + (jj + 3) / pack_num * channel_stride);

                    // Round and clamp the values
                    val0 = fminf(fmaxf(nearbyintf(val0), 0.0f), max_val);
                    val1 = fminf(fmaxf(nearbyintf(val1), 0.0f), max_val);
                    val2 = fminf(fmaxf(nearbyintf(val2), 0.0f), max_val);
                    val3 = fminf(fmaxf(nearbyintf(val3), 0.0f), max_val);

                    // Pack the 4 quantized values into a 16-bit integer
                    uint16_t packed = 0;
                    packed |= (static_cast<uint16_t>(static_cast<uint8_t>(val3)) & 0xF);
                    packed <<= 4;
                    packed |= (static_cast<uint16_t>(static_cast<uint8_t>(val2)) & 0xF);
                    packed <<= 4;
                    packed |= (static_cast<uint16_t>(static_cast<uint8_t>(val1)) & 0xF);
                    packed <<= 4;
                    packed |= (static_cast<uint16_t>(static_cast<uint8_t>(val0)) & 0xF);

                    // Store the packed value
                    dst(i, jj / 4, k) = packed;
                }
            }
        }

    }
};

template<int num_bits, typename Tensor1, typename Tensor2, typename Tensor3, typename Tensor4, typename Tensor5>
CUTE_DEVICE
void qpack_Kchannel_Vtensor(Tensor1 &src, Tensor2 &dst,
                                 Tensor3 &scales_k, Tensor4 &zeros_k, Tensor5 &reduce_tmp,
                                 const int num_params = 1) {

    qpack_kc_vt<num_bits, Tensor1, Tensor2, Tensor3, Tensor4, Tensor5>::apply(src, dst, scales_k, zeros_k, reduce_tmp, num_params);

}

template<int kHeadDim, typename TensorV, typename TensorVPack, typename TensorVParams>
CUTE_DEVICE
void pack_Vtensor_2bit_interleaved(TensorV const& src, TensorVPack &dst, TensorVParams &params) {
    static_assert(kHeadDim == 128, "2bit interleaved V pack currently expects headdim 128");
    constexpr int kHeadDimPack = kHeadDim / 8;
    constexpr int kGroupSize = 32;
    constexpr int kNumGroups = kHeadDim / kGroupSize;
    constexpr float max_val = 3.0f;

    for (int idx = threadIdx.x; idx < size<0>(dst) * kNumGroups; idx += blockDim.x) {
        const int row = idx / kNumGroups;
        const int group = idx - row * kNumGroups;
        const int group_start = group * kGroupSize;

        float min_i = float(src(row, group_start));
        float max_i = min_i;
        CUTE_UNROLL
        for (int g = 1; g < kGroupSize; ++g) {
            const float x = float(src(row, group_start + g));
            min_i = fminf(min_i, x);
            max_i = fmaxf(max_i, x);
        }

        float scale_inv = 0.0f;
        float zero_q = 0.0f;
        float zero_dequant = 0.0f;
        compute_affine_asym_params(min_i, max_i, max_val, scale_inv, zero_q, zero_dequant);
        const float scale = scale_inv == 0.0f ? 0.0f : 1.0f / scale_inv;
        using ParamsT = cute::remove_cvref_t<decltype(params(0, 0))>;
        params(row, group) = make_pair2_vals<ParamsT>(scale, zero_dequant);
    }

    for (int idx = threadIdx.x; idx < size<0>(dst) * kHeadDimPack; idx += blockDim.x) {
        const int row = idx / kHeadDimPack;
        const int col = idx - row * kHeadDimPack;
        uint16_t packed = 0;

        CUTE_UNROLL
        for (int slot = 0; slot < 8; ++slot) {
            const int dim = col + 16 * slot;
            const int group_start = (dim / 32) * 32;

            float min_i = float(src(row, group_start));
            float max_i = min_i;
            CUTE_UNROLL
            for (int g = 1; g < 32; ++g) {
                const float x = float(src(row, group_start + g));
                min_i = fminf(min_i, x);
                max_i = fmaxf(max_i, x);
            }

            float scale_inv = 0.0f;
            float zero_q = 0.0f;
            float zero_dequant = 0.0f;
            compute_affine_asym_params(min_i, max_i, max_val, scale_inv, zero_q, zero_dequant);

            float q = float(src(row, dim)) * scale_inv + zero_q;
            q = fminf(fmaxf(nearbyintf(q), 0.0f), max_val);
            packed |= (uint16_t(q) & 0x3u) << (2 * slot);
        }

        dst(row, col) = packed;
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename TensorParamsG0, typename Tensor1, typename Operator>
CUTE_DEVICE
void quad_allreduce_g(TensorParamsG0 &dst, Tensor1 &src, Operator &op, int k, int num_params) {
    CUTE_UNROLL
    for (int i = k * num_params; i < (k + 1) * num_params; i++) {

        // Calculate which group of 4 this thread belongs to
        const int group_id = threadIdx.x / 4;
        const int group_base = group_id * 4;

        // Start with the value from the first thread in our group
        auto val = __shfl_sync(uint32_t(-1), src(i), group_base);

        // Reduce with the other 3 threads in our group
        #pragma unroll
        for (int offset = 1; offset < 4; offset++) {
            val = op(val, __shfl_sync(uint32_t(-1), src(i), group_base + offset));
        }

        // Broadcast the final result back to all threads in the group
        dst(i) = val;

    }
}

template<typename Tensor0, typename TensorParamsG0, typename Operator>
CUTE_DEVICE
void thread_reduce_g(Tensor0 const& tensor, TensorParamsG0& summary, Operator& op, int k, int num_params) {
    CUTE_UNROLL
    for (int i = k * num_params, j = 0; i < (k + 1) * num_params; i++, j++) {
        int ii = size<1>(tensor) / num_params;
        summary(i) = tensor(0, j * ii);

        CUTE_UNROLL
        for (int mi = 0; mi < size<0>(tensor); ++mi) {
            CUTE_UNROLL
            for (int ni = j * ii; ni < (j + 1) * ii; ++ni) {
                summary(i) = op(summary(i), tensor(mi, ni));
            }
        }
    }
}

template<typename Engine0, typename Layout0, typename TensorParamsG0, typename Operator>
CUTE_DEVICE
void reduce_g(Tensor<Engine0, Layout0> const& tensor, TensorParamsG0& summary, Operator& op, int k, int num_params) {
    quant::thread_reduce_g(tensor, summary, op, k, num_params);
    quant::quad_allreduce_g(summary, summary, op, k, num_params);
}

template<typename Engine0, typename Layout0, typename TensorParamsG0>
CUTE_DEVICE
void reduce_max_g(Tensor<Engine0, Layout0> const& tensor, TensorParamsG0 &max, int k, int num_params) {
    flash::MaxOp<float> max_op;
    quant::reduce_g(tensor, max, max_op, k, num_params);  // Use the existing reduce_q function
}

template<typename Engine0, typename Layout0, typename TensorParamsG0>
CUTE_DEVICE
void reduce_min_g(Tensor<Engine0, Layout0> const& tensor, TensorParamsG0 &min, int k, int num_params) {
    flash::MinOp<float> min_op;
    quant::reduce_g(tensor, min, min_op, k, num_params);  // Use the existing reduce_q function
}

template<typename Tensor1, typename Tensor2, typename TensorParamsG1, typename TensorParamsG2>
CUTE_DEVICE
void quant_Ktensor(Tensor1 &src, Tensor2 &dst,
                   TensorParamsG1 &scales_k_g, TensorParamsG2 &zeros_k_g,
                   const int num_params) {

    const int num_bits = 4;

    const float max_val  = float((1 << num_bits) - 1);
    // const int num_params = 128 / group_size;
    const int ki         = size<2>(src) / num_params;

    // Declare per-channel tensors
    using TensorChannel = decltype(make_fragment_like<float>(scales_k_g));
    TensorChannel channel_max, channel_min, channel_range, channel_scales_inv, channel_zeros, channel_zero_q;

    CUTE_UNROLL
    for (int k = 0; k < size<1>(src); ++k) {
        quant::reduce_max_g(src(_, k, _), channel_max, k, num_params); // TODO:check 128
        quant::reduce_min_g(src(_, k, _), channel_min, k, num_params);
    }

    // Compute per-channel scale inverses and zeros
    CUTE_UNROLL
    for (int i = 0; i < size(channel_max); ++i) {
        float max_i = float(channel_max(i));
        float min_i = float(channel_min(i));
        float scale_inv = 0.0f;
        float zero_q = 0.0f;
        float zero_dequant = 0.0f;
        compute_affine_asym_params(min_i, max_i, max_val, scale_inv, zero_q, zero_dequant);
        channel_scales_inv(i) = scale_inv;
        channel_zero_q(i)     = zero_q;
        channel_zeros(i)      = zero_dequant;
        // Store scales and zeros
        {
            auto &s_slot = scales_k_g(i);
            using ST = cute::remove_cvref_t<decltype(s_slot)>;
            s_slot = ST(scale_inv == 0 ? 0.0f : 1.0f / scale_inv);
        }
        {
            auto &z_slot = zeros_k_g(i);
            using ZT = cute::remove_cvref_t<decltype(z_slot)>;
            z_slot = ZT(zero_dequant);
        }
    }

    // Pack the tensor
    CUTE_UNROLL
    for (int k = 0; k < size<2>(src); ++k) {

        CUTE_UNROLL
        for (int i = 0; i < size<0>(src); ++i) {

            CUTE_UNROLL
            for (int jj = 0; jj < size<1>(src); jj += 4) {
                float zero_q0 = float(channel_zero_q(k / ki + jj + 0 * num_params));
                float zero_q1 = float(channel_zero_q(k / ki + jj + 1 * num_params));
                float zero_q2 = float(channel_zero_q(k / ki + jj + 2 * num_params));
                float zero_q3 = float(channel_zero_q(k / ki + jj + 3 * num_params));

                float scale_inv0 = float(channel_scales_inv(k / ki + jj + 0 * num_params));
                float scale_inv1 = float(channel_scales_inv(k / ki + jj + 1 * num_params));
                float scale_inv2 = float(channel_scales_inv(k / ki + jj + 2 * num_params));
                float scale_inv3 = float(channel_scales_inv(k / ki + jj + 3 * num_params));

                // float val0 = float(src(i, jj, k));
                // float val1 = float(src(i, jj + 1, k));
                // float val2 = float(src(i, jj + 2, k));
                // float val3 = float(src(i, jj + 3, k));

                float val0 = float(src(i, jj, k)) * scale_inv0 + zero_q0;
                float val1 = float(src(i, jj + 1, k)) * scale_inv1 + zero_q1;
                float val2 = float(src(i, jj + 2, k)) * scale_inv2 + zero_q2;
                float val3 = float(src(i, jj + 3, k)) * scale_inv3 + zero_q3;

                // Round and clamp the values
                val0 = fminf(fmaxf(nearbyintf(val0), 0.0f), max_val);
                val1 = fminf(fmaxf(nearbyintf(val1), 0.0f), max_val);
                val2 = fminf(fmaxf(nearbyintf(val2), 0.0f), max_val);
                val3 = fminf(fmaxf(nearbyintf(val3), 0.0f), max_val);

                // Pack the 4 quantized values into a 16-bit integer
                uint16_t packed = 0;
                packed |= (static_cast<uint16_t>(static_cast<uint8_t>(val3)) & 0xF);
                packed <<= 4;
                packed |= (static_cast<uint16_t>(static_cast<uint8_t>(val2)) & 0xF);
                packed <<= 4;
                packed |= (static_cast<uint16_t>(static_cast<uint8_t>(val1)) & 0xF);
                packed <<= 4;
                packed |= (static_cast<uint16_t>(static_cast<uint8_t>(val0)) & 0xF);

                // Store the packed value
                dst(i, jj / 4, k) = packed;
            }
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename TiledCopyRS, typename Tensor0, typename Tensor1,
         typename TiledCopySG, typename Tensor2, typename Tensor3,
         typename Tensor4, typename Tensor5, typename Tensor6>
CUTE_DEVICE
void pack_Ktensor_store(TiledCopyRS smem_tiled_copy, Tensor0 &src_r2s, Tensor1 &dst_r2s,
                        TiledCopySG gmem_tiled_copy, Tensor2 &src_s2g, Tensor3 &dst_g2s,
                        Tensor4 &scales, Tensor5 &zeros, Tensor6 &params,
                        const int num_params) {
    // copy from register to shared memory
    cute::copy(smem_tiled_copy, src_r2s, dst_r2s);
    __syncthreads();

    // copy from shared memory to global memory
    cute::copy(gmem_tiled_copy, src_s2g, dst_g2s);
    __syncthreads();

    // copy params from register to global memory
    CUTE_UNROLL
    for (int j = 0; j < size<0>(scales); ++j) {
        params(0  + 32 * (j / num_params) + threadIdx.x / 4, j % num_params) = scales(j);
        params(64 + 32 * (j / num_params) + threadIdx.x / 4, j % num_params) = zeros(j);
    }
    __syncthreads();
}

template<typename TiledCopyRS, typename Tensor0, typename Tensor1,
         typename TiledCopySG, typename Tensor2, typename Tensor3,
         typename Tensor4, typename Tensor5, typename Tensor6>
CUTE_DEVICE
void pack_Kchannel_store(TiledCopyRS smem_tiled_copy, Tensor0 &src_r2s, Tensor1 &dst_r2s,
                        TiledCopySG gmem_tiled_copy, Tensor2 &src_s2g, Tensor3 &dst_g2s,
                        Tensor4 &scales, Tensor5 &zeros, Tensor6 &params,
                        const int num_params) {
    // copy from register to shared memory
    cute::copy(smem_tiled_copy, src_r2s, dst_r2s);
    __syncthreads();

    // copy from shared memory to global memory
    cute::copy(gmem_tiled_copy, src_s2g, dst_g2s);
    __syncthreads();

    // copy params from register to global memory
    CUTE_UNROLL
    for (int i = 0; i < size<1>(scales); ++i) {
        CUTE_UNROLL
        for (int j = 0; j < size<0>(scales); ++j) {
            params(j % num_params, 0  + 8 * i + 4 * (j / num_params) + threadIdx.x % 4) = scales(j, i);
            params(j % num_params, 64 + 8 * i + 4 * (j / num_params) + threadIdx.x % 4) = zeros(j, i);
        }
    }
    __syncthreads();
}

template<int num_bits, int kHeadDim,
         typename TiledCopyRS, typename Tensor0, typename Tensor1,
         typename TiledCopySG, typename Tensor2, typename Tensor3,
         typename Tensor4, typename Tensor5, typename Tensor6>
CUTE_DEVICE
void pack_Vtensor_store(TiledCopyRS smem_tiled_copy, Tensor0 &src_r2s, Tensor1 &dst_r2s,
                        TiledCopySG gmem_tiled_copy, Tensor2 &src_s2g, Tensor3 &dst_g2s,
                        Tensor4 &scales, Tensor5 &zeros, Tensor6 &params,
                        const int num_params) {
    if (kHeadDim == 128 && num_bits == 2) {
        if (threadIdx.x < 64) {
            cute::copy(smem_tiled_copy, src_r2s, dst_r2s);
        }
    } else {
        cute::copy(smem_tiled_copy, src_r2s, dst_r2s);
    }
    __syncthreads();

    // copy from shared memory to global memory
    cute::copy(gmem_tiled_copy, src_s2g, dst_g2s);
    __syncthreads();

    if constexpr (num_bits == 2) {
        using ParamType = cute::remove_cvref_t<decltype(params(_0{}, _0{}))>;
        for (int idx = threadIdx.x; idx < size(params); idx += blockDim.x) {
            params(idx) = ParamType{};
        }
        __syncthreads();
    }

    // copy params from register to global memory
    const int params_dim = size<1>(params);
    const int num_params_2 = num_bits == 2 && num_params > params_dim ? params_dim : num_params;
    CUTE_UNROLL
    for (int i = 0; i < size<1>(scales); ++i) {
        CUTE_UNROLL
        for (int j = 0; j < size<0>(scales); ++j) {
            params(128 * (i / 8) + 0  + 8 * (i % 8) + 4 * (j / num_params_2) + threadIdx.x % 4, j % num_params_2) = scales(j, i);
            params(128 * (i / 8) + 64 + 8 * (i % 8) + 4 * (j / num_params_2) + threadIdx.x % 4, j % num_params_2) = zeros(j, i);
        }
    }
    __syncthreads();
}

} // namespace quant
