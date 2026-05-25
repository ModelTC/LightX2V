/******************************************************************************
 * Copyright (c) 2024, Tri Dao.
 ******************************************************************************/

#pragma once

#include <cmath>

#include <cute/tensor.hpp>

#include <cutlass/numeric_types.h>

#include "philox.cuh"
#include "utils.h"

namespace flash {

using namespace cute;

////////////////////////////////////////////////////////////////////////////////////////////////////

template<bool zero_init=true, typename Engine0, typename Layout0, typename Engine1, typename Layout1, typename Operator>
__device__ __forceinline__ void thread_reduce_(Tensor<Engine0, Layout0> const &tensor, Tensor<Engine1, Layout1> &summary, Operator &op) {
    static_assert(Layout0::rank == 2, "Only support 2D Tensor");
    static_assert(Layout1::rank == 1, "Only support 1D Tensor");
    CUTE_STATIC_ASSERT_V(size<0>(summary) == size<0>(tensor));
    #pragma unroll
    for (int mi = 0; mi < size<0>(tensor); mi++) {
        summary(mi) = zero_init ? tensor(mi, 0) : op(summary(mi), tensor(mi, 0));
        #pragma unroll
        for (int ni = 1; ni < size<1>(tensor); ni++) {
            summary(mi) = op(summary(mi), tensor(mi, ni));
        }
    }

}

// template<typename Engine0, typename Layout0, typename Engine1, typename Layout1, typename Operator>
// __device__ __forceinline__ void quad_allreduce_(Tensor<Engine0, Layout0> &dst, Tensor<Engine1, Layout1> &src, Operator &op) {
//     CUTE_STATIC_ASSERT_V(size(dst) == size(src));
//     #pragma unroll
//     for (int i = 0; i < size(dst); i++){
//         dst(i) = Allreduce<4>::run(src(i), op);
//     }
// }

template<typename Operator>
__device__ __forceinline__ float warp_reduce_acc(float &val, Operator &op) {
    // Get the thread's position within its group of 4
    const int group_id = threadIdx.x / 4;     // Which group of 4 this thread belongs to
    const int local_id = threadIdx.x % 4;     // Position within group of 4 (0-3)

    // Only reduce within groups of 4 threads
    // Using butterfly pattern
    #pragma unroll
    for (int offset = 2; offset > 0; offset >>= 1) {
        float other = __shfl_down_sync(0xffffffff, val, offset);
        if (local_id < offset) {
            val = op(val, other);
        }
    }

    // Broadcast the result from thread 0 to all threads in the group
    val = __shfl_sync(0xffffffff, val, group_id * 4);

    return val;
}

template<typename Engine0, typename Layout0, typename Engine1, typename Layout1, typename Tensor2, typename Operator>
__device__ __forceinline__ void quad_allreduce_2(Tensor<Engine0, Layout0> &dst, Tensor<Engine1, Layout1> &src, Tensor2 &reduce_tmp, Operator &op) {
    // __shared__ float smem[4];  // For 4 warps, we need 4 elements
    CUTE_STATIC_ASSERT_V(size(dst) == size(src));

    const int warp_id = threadIdx.x / 32;
    const int lane_id = threadIdx.x % 32;
    const int row = (threadIdx.x % 32) / 4;

    // #if DEBUG
    // if (threadIdx.x == 103 && threadIdx.y == 0 && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 7) {
    //     PRINTTENSOR("dst", dst);
    // }
    // #endif

    #pragma unroll
    for (int i = 0; i < 1; i++) {
        // First do reduction within each group of 4 threads
        float val = warp_reduce_acc(src(i), op);

        // Write the result to shared memory for each group's leader
        if (lane_id % 4 == 0) {
            auto &slot = reduce_tmp(row, warp_id);
            using SlotType = cute::remove_cvref_t<decltype(slot)>;
            slot = SlotType(val);
        }
        __syncthreads();

        // Check if thread is one of the first threads in each group of 4 (0,4,8,12,16,20,24,28)
        if ((lane_id % 4) == 0) {
            // This thread is responsible for reducing its group's values
            float group_val = reduce_tmp(row, 0);
            #pragma unroll
            for (int w = 1; w < 4; w++) {
                group_val = op(group_val, reduce_tmp(row, w));
            }
            auto &slot = reduce_tmp(row, 0);
            using SlotType = cute::remove_cvref_t<decltype(slot)>;
            slot = SlotType(group_val);
        }
        __syncthreads();

        // All threads read the final result
        dst(i) = reduce_tmp(row,0);

        // #if DEBUG
        // if (threadIdx.x == 103 && threadIdx.y == 0 && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 7) {
        //     printf("val: %f\n", val);
        //     PRINTTENSOR("reduce_tmp", reduce_tmp);
        // }
        // #endif

    }
}

template<typename Engine0, typename Layout0, typename Engine1, typename Layout1, typename Tensor2, typename Operator>
__device__ __forceinline__ void quad_allreduce_(Tensor<Engine0, Layout0> &dst, Tensor<Engine1, Layout1> &src, Tensor2 &reduce_tmp, Operator &op) {
    // __shared__ float smem[4];  // For 4 warps, we need 4 elements
    CUTE_STATIC_ASSERT_V(size(dst) == size(src));

    const int warp_id = threadIdx.x / 32;
    const int lane_id = threadIdx.x % 32;

    #pragma unroll
    for (int i = 0; i < size(dst); i++) {
        // First do reduction within each group of 4 threads
        float val = Allreduce<4>::run(src(i), op);

        // Write the result to shared memory for each group's leader
        if (lane_id % 4 == 0) {
            reduce_tmp(i,warp_id) = val;
        }
        __syncthreads();

        // First thread in the first group reads all values and reduces them
        if (lane_id == 0) {
            float final_val = reduce_tmp(0,0);
            #pragma unroll
            for (int w = 1; w < 4; w++) {  // For 4 warps
                final_val = op(final_val, reduce_tmp(i,w));
            }
            // Write back the final result
            reduce_tmp(i,0) = final_val;
        }
        __syncthreads();

        // All threads read the final result
        // cute::copy(reduce_tmp(0,0), dst(i));
        dst(i) = reduce_tmp(i,0);
    }
}

template<bool zero_init=true, typename Engine0, typename Layout0, typename Engine1, typename Layout1, typename Tensor2, typename Operator>
__device__ __forceinline__ void reduce_(Tensor<Engine0, Layout0> const& tensor, Tensor<Engine1, Layout1> &summary, Tensor2 &reduce_tmp, Operator &op) {
    thread_reduce_<zero_init>(tensor, summary, op);
    quad_allreduce_(summary, summary, reduce_tmp, op);
}

template<bool zero_init=true, typename Engine0, typename Layout0, typename Engine1, typename Layout1, typename Tensor2>
__device__ __forceinline__ void reduce_max(Tensor<Engine0, Layout0> const& tensor, Tensor<Engine1, Layout1> &max, Tensor2 &reduce_tmp){
    MaxOp<float> max_op;
    reduce_<zero_init>(tensor, max, reduce_tmp, max_op);
}

template<bool zero_init=true, typename Engine0, typename Layout0, typename Engine1, typename Layout1, typename Tensor2, typename Operator>
__device__ __forceinline__ void reduce_2(Tensor<Engine0, Layout0> const& tensor, Tensor<Engine1, Layout1> &summary, Tensor2 &reduce_tmp, Operator &op) {
    thread_reduce_<zero_init>(tensor, summary, op);
    quad_allreduce_2(summary, summary, reduce_tmp, op);
}

template<bool zero_init=true, typename Engine0, typename Layout0, typename Engine1, typename Layout1, typename Tensor2>
__device__ __forceinline__ void reduce_max_2(Tensor<Engine0, Layout0> const& tensor, Tensor<Engine1, Layout1> &max, Tensor2 &reduce_tmp){
    MaxOp<float> max_op;
    reduce_2<zero_init>(tensor, max, reduce_tmp, max_op);
}

template<bool zero_init=true, typename Engine0, typename Layout0, typename Engine1, typename Layout1>
__device__ __forceinline__ void reduce_sum(Tensor<Engine0, Layout0> const& tensor, Tensor<Engine1, Layout1> &sum){
    SumOp<float> sum_op;
    thread_reduce_<zero_init>(tensor, sum, sum_op);
}

// Apply the exp to all the elements.
template <bool Scale_max=true, typename Engine0, typename Layout0, typename Engine1, typename Layout1>
__forceinline__ __device__ void scale_apply_exp2(Tensor<Engine0, Layout0> &tensor, Tensor<Engine1, Layout1> const &max, const float scale) {
    static_assert(Layout0::rank == 2, "Only support 2D Tensor");
    static_assert(Layout1::rank == 1, "Only support 1D Tensor");
    CUTE_STATIC_ASSERT_V(size<0>(max) == size<0>(tensor));
    #pragma unroll
    for (int mi = 0; mi < size<0>(tensor); ++mi) {
        // If max is -inf, then all elements must have been -inf (possibly due to masking).
        // We don't want (-inf - (-inf)) since that would give NaN.
        // If we don't have float around M_LOG2E the multiplication is done in fp64.
        const float max_scaled = max(mi) == -INFINITY ? 0.f : max(mi) * (Scale_max ? scale : float(M_LOG2E));
        #pragma unroll
        for (int ni = 0; ni < size<1>(tensor); ++ni)  {
            // Instead of computing exp(x - max), we compute exp2(x * log_2(e) -
            // max * log_2(e)) This allows the compiler to use the ffma
            // instruction instead of fadd and fmul separately.
            // The following macro will disable the use of fma.
            // See: https://github.com/pytorch/pytorch/issues/121558 for more details
            // This macro is set in PyTorch and not FlashAttention
            #ifdef UNFUSE_FMA
                tensor(mi, ni) = exp2f(__fmul_rn(tensor(mi, ni), scale) - max_scaled);
            #else
                tensor(mi, ni) = exp2f(tensor(mi, ni) * scale - max_scaled);
            #endif
        }
    }
}

// Apply the exp to all the elements.
template <bool zero_init=true, typename Engine0, typename Layout0, typename Engine1, typename Layout1>
__forceinline__ __device__ void max_scale_exp2_sum(Tensor<Engine0, Layout0> &tensor, Tensor<Engine1, Layout1> &max, Tensor<Engine1, Layout1> &sum, const float scale) {
    static_assert(Layout0::rank == 2, "Only support 2D Tensor");
    static_assert(Layout1::rank == 1, "Only support 1D Tensor");
    CUTE_STATIC_ASSERT_V(size<0>(max) == size<0>(tensor));
    #pragma unroll
    for (int mi = 0; mi < size<0>(tensor); ++mi) {
        MaxOp<float> max_op;
        max(mi) = zero_init ? tensor(mi, 0) : max_op(max(mi), tensor(mi, 0));
        #pragma unroll
        for (int ni = 1; ni < size<1>(tensor); ni++) {
            max(mi) = max_op(max(mi), tensor(mi, ni));
        }
        max(mi) = Allreduce<4>::run(max(mi), max_op);
        // If max is -inf, then all elements must have been -inf (possibly due to masking).
        // We don't want (-inf - (-inf)) since that would give NaN.
        const float max_scaled = max(mi) == -INFINITY ? 0.f : max(mi) * scale;
        sum(mi) = 0;
        #pragma unroll
        for (int ni = 0; ni < size<1>(tensor); ++ni)  {
            // Instead of computing exp(x - max), we compute exp2(x * log_2(e) -
            // max * log_2(e)) This allows the compiler to use the ffma
            // instruction instead of fadd and fmul separately.
            tensor(mi, ni) = exp2f(tensor(mi, ni) * scale - max_scaled);
            sum(mi) += tensor(mi, ni);
        }
        SumOp<float> sum_op;
        sum(mi) = Allreduce<4>::run(sum(mi), sum_op);
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <int kNRows>
struct Softmax {

    using TensorT = decltype(make_tensor<float>(Shape<Int<kNRows>>{}));
    TensorT row_max, row_sum;

    __forceinline__ __device__ Softmax() {};

    template<bool Is_first, bool Check_inf=false, typename Tensor0, typename Tensor1, typename Tensor2>
    __forceinline__ __device__ void softmax_rescale_o(Tensor0 &acc_s, Tensor1 &acc_o, Tensor2 &reduce_tmp, float softmax_scale_log2) {
        // Reshape acc_s from (MMA=4, MMA_M, MMA_N) to (nrow=(2, MMA_M), ncol=(2, MMA_N))
        Tensor scores = make_tensor(acc_s.data(), flash::convert_layout_acc_rowcol(acc_s.layout()));
        static_assert(decltype(size<0>(scores))::value == kNRows);
        if (Is_first) {
            flash::template reduce_max_2</*zero_init=*/true>(scores, row_max, reduce_tmp);
            flash::scale_apply_exp2(scores, row_max, softmax_scale_log2);
            flash::reduce_sum</*zero_init=*/true>(scores, row_sum);
        } else {
            Tensor scores_max_prev = make_fragment_like(row_max);
            cute::copy(row_max, scores_max_prev);
            flash::template reduce_max_2</*zero_init=*/false>(scores, row_max, reduce_tmp);
            // Reshape acc_o from (MMA=4, MMA_M, MMA_K) to (nrow=(2, MMA_M), ncol=(2, MMA_K))
            Tensor acc_o_rowcol = make_tensor(acc_o.data(), flash::convert_layout_acc_rowcol(acc_o.layout()));
            static_assert(decltype(size<0>(acc_o_rowcol))::value == kNRows);
            #pragma unroll
            for (int mi = 0; mi < size(row_max); ++mi) {
                float scores_max_cur = !Check_inf
                    ? row_max(mi)
                    : (row_max(mi) == -INFINITY ? 0.0f : row_max(mi));
                float scores_scale = exp2f((scores_max_prev(mi) - scores_max_cur) * softmax_scale_log2);
                row_sum(mi) *= scores_scale;
                #pragma unroll
                for (int ni = 0; ni < size<1>(acc_o_rowcol); ++ni) { acc_o_rowcol(mi, ni) *= scores_scale; }
            }

            flash::scale_apply_exp2(scores, row_max, softmax_scale_log2);
            // We don't do the reduce across threads here since we don't need to use the row_sum.
            // We do that reduce at the end when we need to normalize the softmax.
            flash::reduce_sum</*zero_init=*/false>(scores, row_sum);
        }
    };

    template<bool Is_dropout=false, bool Split=false, typename Tensor0, typename Tensor1>
    __forceinline__ __device__ TensorT normalize_softmax_lse(Tensor0 &acc_o, Tensor1 &reduce_tmp, float softmax_scale, float rp_dropout=1.0) {
        SumOp<float> sum_op;
        quad_allreduce_2(row_sum, row_sum, reduce_tmp, sum_op);
        TensorT lse = make_fragment_like(row_sum);
        Tensor acc_o_rowcol = make_tensor(acc_o.data(), flash::convert_layout_acc_rowcol(acc_o.layout()));
        static_assert(decltype(size<0>(acc_o_rowcol))::value == kNRows);
        #pragma unroll
        for (int mi = 0; mi < size<0>(acc_o_rowcol); ++mi) {
            float sum = row_sum(mi);
            float inv_sum = (sum == 0.f || sum != sum) ? 1.f : 1.f / sum;
            lse(mi) = (sum == 0.f || sum != sum) ? (Split ? -INFINITY : INFINITY) : row_max(mi) * softmax_scale + __logf(sum);
            float scale = !Is_dropout ? inv_sum : inv_sum * rp_dropout;
            #pragma unroll
            for (int ni = 0; ni < size<1>(acc_o_rowcol); ++ni) { acc_o_rowcol(mi, ni) *= scale; }
        }
        return lse;
    };
};

}  // namespace flash
