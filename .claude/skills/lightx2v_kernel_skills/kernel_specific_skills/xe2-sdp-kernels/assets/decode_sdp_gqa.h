// decode_sdp_gqa.h
// Non-paged GQA decode SDP kernels for full (non-sparse) attention. qlen=1.
//
// flashAttnMha128GQAdecode:
//   Adapted from infllm_decode.h — paged KV-cache logic removed.
//   Each workgroup processes one chunk (chunk_size tokens) for one kv_head.
//   Threads: (sp_blk_num_per_t, head_groups_per_g) per workgroup.
//   Writes per-chunk partial output/lse/max to intermediate buffers.
//
// flashAttnMha128GQADecodeReduce:
//   Adapted from kernel_func1 in sage.decode.kernel.h with QLEN=1.
//   Reduces partial chunk results -> final fp16 output.
//   Each thread processes 16 heads. GlobalRange=(1,1,headQ/16).

#pragma once
#include <sycl/sycl.hpp>
#include <sycl/ext/intel/esimd.hpp>

using namespace sycl::ext::intel::esimd;
using namespace sycl::ext::intel::experimental::esimd;

using fp16 = sycl::half;
using fp32 = float;

#define FP32_MIN_DEC (-1.7e+38f)

// ---- Main decode kernel ----
// Inputs (fp16):
//   qState  : [headQ, HD]
//   kState  : [kv_len, headKv, HD]
//   vState  : [kv_len, headKv, HD]
// Intermediate outputs (fp32):
//   out     : [chunk_num, headQ, HD]
//   out_lse : [chunk_num, headQ]
//   out_max : [chunk_num, headQ]
template<uint32_t HD, uint32_t headQ, uint32_t headKv,
         uint32_t Q_HEAD_PER_T, uint32_t sp_blk_size, uint32_t chunk_size>
void flashAttnMha128GQAdecode(
    fp16*    qState,
    fp16*    kState,
    fp16*    vState,
    fp32*    out,
    fp32*    out_lse,
    fp32*    out_max,
    uint32_t kv_len,
    fp32     attscale,
    sycl::queue& dpcpp_queue)
{
    constexpr int q_head_num_per_kv_head = headQ / headKv;
    constexpr int head_groups_per_g      = q_head_num_per_kv_head / Q_HEAD_PER_T;
    constexpr int sp_blk_num_per_t       = chunk_size / sp_blk_size;
    static_assert(chunk_size % sp_blk_size == 0, "chunk_size must be multiple of sp_blk_size");

    int chunk_num = (kv_len + chunk_size - 1) / chunk_size;

    // GlobalRange: (1, chunk_num, headKv * head_groups_per_g)
    // LocalRange:  (1, sp_blk_num_per_t, head_groups_per_g)
    sycl::range<3> GlobalRange(1, (size_t)(chunk_num * sp_blk_num_per_t),
                               (size_t)(headKv * head_groups_per_g));
    sycl::range<3> LocalRange(1, (size_t)sp_blk_num_per_t, (size_t)head_groups_per_g);

    dpcpp_queue.submit([&](sycl::handler& cgh) {
        cgh.parallel_for(sycl::nd_range<3>(GlobalRange, LocalRange),
            [=](sycl::nd_item<3> ndi) SYCL_ESIMD_KERNEL {

            // SLM layout:
            //   [0,           slm_reduce_size)  : output accumulators  [head_groups_per_g, sp_blk_num_per_t, Q_HEAD_PER_T, HD] float
            //   [slm_reduce_size, +slm_max_size) : per-block max        [head_groups_per_g, sp_blk_num_per_t, Q_HEAD_PER_T] float
            //   [+slm_max_size,   +slm_lse_size) : per-block lse        [head_groups_per_g, sp_blk_num_per_t, Q_HEAD_PER_T] float
            constexpr int slm_reduce_size = head_groups_per_g * sp_blk_num_per_t * Q_HEAD_PER_T * HD * (int)sizeof(fp32);
            constexpr int slm_max_size    = head_groups_per_g * sp_blk_num_per_t * Q_HEAD_PER_T * (int)sizeof(fp32);
            constexpr int slm_lse_size    = head_groups_per_g * sp_blk_num_per_t * Q_HEAD_PER_T * (int)sizeof(fp32);
            slm_init(slm_reduce_size + slm_max_size + slm_lse_size);

            int chunk_idx      = ndi.get_group(1);
            int kv_head_idx    = ndi.get_group(2);
            int sp_blk_idx     = ndi.get_local_id(1);
            int head_group_idx = ndi.get_local_id(2);
            int q_head_idx     = kv_head_idx * q_head_num_per_kv_head + head_group_idx * Q_HEAD_PER_T;

            // Direct (non-paged) KV offset — removed kvCacheBlockTable indirection
            uint32_t kvLogicalOffset = (uint32_t)(chunk_idx * chunk_size + sp_blk_idx * sp_blk_size);
            uint32_t kvRealOffset    = kvLogicalOffset * headKv * HD + kv_head_idx * HD;

            // Read Q for Q_HEAD_PER_T heads
            simd<fp16, Q_HEAD_PER_T * HD> qIn =
                block_load<fp16, Q_HEAD_PER_T * HD>(qState + q_head_idx * HD);
            qIn = qIn * attscale;

            simd<fp32, Q_HEAD_PER_T>      maxKq         = FP32_MIN_DEC;
            simd<fp32, Q_HEAD_PER_T>      old_maxKq     = FP32_MIN_DEC;
            simd<fp32, Q_HEAD_PER_T>      max_correction = FP32_MIN_DEC;
            simd<fp32, Q_HEAD_PER_T>      lse            = 0;
            simd<fp32, Q_HEAD_PER_T * HD> output         = 0;

            // Compute valid tokens for this sp_blk (handles partial last block when kv_len
            // is not a multiple of chunk_size — avoids OOB reads beyond kv_len)
            int valid_t = (int)sp_blk_size;
            {
                int tok_end = (int)kv_len - (int)kvLogicalOffset;
                if (tok_end < valid_t) valid_t = tok_end > 0 ? tok_end : 0;
            }

            // Iterate over valid KV tokens in this sp_blk
            for (int t = 0; t < valid_t; t++) {
                simd<fp16, HD> kIn = block_load<fp16, HD>(kState + kvRealOffset + t * headKv * HD);
                simd<fp16, HD> vIn = block_load<fp16, HD>(vState + kvRealOffset + t * headKv * HD);

                // QK dot product
                simd<fp32, Q_HEAD_PER_T> kq_out;
                #pragma unroll
                for (int h = 0; h < Q_HEAD_PER_T; h++) {
                    kq_out[h] = sycl::ext::intel::esimd::detail::sum<fp32, fp16, HD>(
                        qIn.template select<HD, 1>(h * HD) * kIn);
                }

                // Online softmax update
                old_maxKq = maxKq;
                maxKq = __ESIMD_NS::max<fp32, Q_HEAD_PER_T, fp32>(kq_out, old_maxKq);
                kq_out = kq_out - maxKq;
                kq_out = __ESIMD_NS::exp2<fp32, Q_HEAD_PER_T, fp32>(
                    kq_out * sycl::ext::intel::esimd::detail::log2e);

                if (t >= 1) {
                    max_correction = old_maxKq - maxKq;
                    max_correction = __ESIMD_NS::exp2<fp32, Q_HEAD_PER_T, fp32>(
                        max_correction * sycl::ext::intel::esimd::detail::log2e);
                    #pragma unroll
                    for (int h = 0; h < Q_HEAD_PER_T; h++) {
                        output.template select<HD, 1>(h * HD) =
                            output.template select<HD, 1>(h * HD) * max_correction[h];
                    }
                    lse = lse * max_correction;
                }
                lse = lse + kq_out;

                // Weighted V accumulation
                #pragma unroll
                for (int h = 0; h < Q_HEAD_PER_T; h++) {
                    simd<fp32, HD> vIn2 = vIn.template select<HD, 1>(0) * kq_out[h];
                    output.template select<HD, 1>(h * HD) =
                        output.template select<HD, 1>(h * HD) + vIn2;
                }
            }

            // Store this sp_blk's result to SLM
            int idx_slm = head_group_idx * sp_blk_num_per_t + sp_blk_idx;
            slm_block_store<fp32, Q_HEAD_PER_T>(
                slm_reduce_size + idx_slm * Q_HEAD_PER_T * (int)sizeof(fp32), maxKq);
            slm_block_store<fp32, Q_HEAD_PER_T>(
                slm_reduce_size + slm_max_size + idx_slm * Q_HEAD_PER_T * (int)sizeof(fp32), lse);
            slm_block_store<fp32, Q_HEAD_PER_T * HD>(
                idx_slm * Q_HEAD_PER_T * HD * (int)sizeof(fp32), output);
            output = 0;

            barrier();

            // Intra-workgroup reduction: only sp_blk_idx==0 threads reduce across sp_blk_num_per_t blocks
            if (sp_blk_idx == 0) {
                int slm_r_h_offset   = (head_group_idx * sp_blk_num_per_t) * Q_HEAD_PER_T * (int)sizeof(fp32);
                int slm_r_o_h_offset = (head_group_idx * sp_blk_num_per_t) * Q_HEAD_PER_T * HD * (int)sizeof(fp32);

                simd<fp32, Q_HEAD_PER_T> max_final = FP32_MIN_DEC;
                simd<fp32, Q_HEAD_PER_T> lse_final = 0;

                // Pass 1: find max across sp_blks
                for (int c = 0; c < sp_blk_num_per_t; c++) {
                    simd<fp32, Q_HEAD_PER_T> cur_max = slm_block_load<fp32, Q_HEAD_PER_T>(
                        slm_reduce_size + slm_r_h_offset + c * Q_HEAD_PER_T * (int)sizeof(fp32));
                    max_final = max<fp32, Q_HEAD_PER_T, fp32>(cur_max, max_final);
                }

                // Pass 2: accumulate with correction
                for (int c = 0; c < sp_blk_num_per_t; c++) {
                    simd<fp32, Q_HEAD_PER_T> cur_max = slm_block_load<fp32, Q_HEAD_PER_T>(
                        slm_reduce_size + slm_r_h_offset + c * Q_HEAD_PER_T * (int)sizeof(fp32));
                    simd<fp32, Q_HEAD_PER_T> cur_lse = slm_block_load<fp32, Q_HEAD_PER_T>(
                        slm_reduce_size + slm_max_size + slm_r_h_offset + c * Q_HEAD_PER_T * (int)sizeof(fp32));
                    simd<fp32, Q_HEAD_PER_T> correction =
                        pow<fp32, Q_HEAD_PER_T, fp32>(2.718f, cur_max - max_final);
                    lse_final = lse_final + cur_lse * correction;

                    #pragma unroll
                    for (int h = 0; h < Q_HEAD_PER_T; h++) {
                        simd<fp32, HD> cur_o = slm_block_load<fp32, HD>(
                            slm_r_o_h_offset + (c * Q_HEAD_PER_T + h) * HD * (int)sizeof(fp32));
                        output.template select<HD, 1>(h * HD) =
                            cur_o.template select<HD, 1>(0) * correction[h] +
                            output.template select<HD, 1>(h * HD);
                    }
                }

                // Write chunk partial results to global memory
                #pragma unroll
                for (int h = 0; h < Q_HEAD_PER_T; h++) {
                    block_store<fp32, 1>(out_max + chunk_idx * headQ + (q_head_idx + h), max_final[h]);
                    block_store<fp32, 1>(out_lse + chunk_idx * headQ + (q_head_idx + h), lse_final[h]);
                }
                #pragma unroll
                for (int h = 0; h < Q_HEAD_PER_T; h++) {
                    block_store<fp32, HD>(out + chunk_idx * headQ * HD + (q_head_idx + h) * HD,
                        output.template select<HD, 1>(h * HD));
                }
            }
        });
    });
}

// ---- Reduce kernel ----
// Reduces per-chunk partial results to final fp16 output.
// Adapted from kernel_func1 (sage.decode.kernel.h) with QLEN=1 (decode).
// Each thread handles 16 consecutive Q heads.
// GlobalRange = (1, 1, headQ/16),  LocalRange = (1,1,1)
//
// Inputs (fp32, from flashAttnMha128GQAdecode):
//   out     : [chunk_num, headQ, HD]
//   out_lse : [chunk_num, headQ]
//   out_max : [chunk_num, headQ]
// Output (fp16):
//   out_final : [headQ, HD]
template<uint32_t HD, uint32_t headQ>
void flashAttnMha128GQADecodeReduce(
    fp32*    out,
    fp32*    out_lse,
    fp32*    out_max,
    fp16*    out_final,
    uint32_t chunk_num,
    sycl::queue& dpcpp_queue)
{
    static_assert(headQ % 16 == 0, "headQ must be multiple of 16");

    sycl::range<3> GlobalRange(1, 1, headQ / 16);
    sycl::range<3> LocalRange(1, 1, 1);

    dpcpp_queue.submit([&](sycl::handler& cgh) {
        cgh.parallel_for(sycl::nd_range<3>(GlobalRange, LocalRange),
            [=](sycl::nd_item<3> ndi) SYCL_ESIMD_KERNEL {

            // group(2) indexes 16-head tiles; each thread handles 16 consecutive Q heads
            int q_head_base = (int)ndi.get_group(2) << 4;  // 0, 16, 32, ...

            // Offsets into intermediate buffers for this thread's 16 heads
            // out layout:     [chunk_num, headQ, HD]  -> base at chunk*headQ*HD + q_head_base*HD
            // lse/max layout: [chunk_num, headQ]       -> base at chunk*headQ + q_head_base
            int32_t base_out    = q_head_base * HD;
            int32_t base_lsmax  = q_head_base;

            simd<fp32, 16>       max_final    = FP32_MIN_DEC;
            simd<fp32, HD * 16>  reduce_final = 0;
            simd<fp32, HD * 16>  cur_reduce;
            simd<fp32, 16>       lse_final    = 0;

            // Pass 1: global max across all chunks for these 16 heads
            for (int ck = 0; ck < (int)chunk_num; ck++) {
                simd<fp32, 16> cur_max =
                    block_load<fp32, 16>(out_max + base_lsmax + ck * headQ);
                max_final = __ESIMD_NS::max<fp32, 16, fp32>(cur_max, max_final);
            }

            // Pass 2: weighted accumulate
            for (int ck = 0; ck < (int)chunk_num; ck++) {
                int32_t out_ck   = base_out   + ck * headQ * HD;
                int32_t lsmax_ck = base_lsmax + ck * headQ;

                simd<fp32, 16> cur_max = block_load<fp32, 16>(out_max + lsmax_ck);
                simd<fp32, 16> cur_lse = block_load<fp32, 16>(out_lse + lsmax_ck);
                simd<fp32, 16> correction =
                    __ESIMD_NS::pow<fp32, 16, fp32>(2.718f, cur_max - max_final);
                lse_final = lse_final + cur_lse * correction;

                #pragma unroll
                for (int j = 0; j < 16; j++) {
                    cur_reduce.template select<HD, 1>(HD * j) =
                        block_load<fp32, HD>(out + out_ck + j * HD);
                    reduce_final.template select<HD, 1>(HD * j) =
                        cur_reduce.template select<HD, 1>(HD * j) * correction[j] +
                        reduce_final.template select<HD, 1>(HD * j);
                }
            }

            // Normalize by lse and write fp16 final output
            #pragma unroll
            for (int i = 0; i < 16; i++) {
                reduce_final.template select<HD, 1>(i * HD) =
                    reduce_final.template select<HD, 1>(i * HD) / lse_final[i];
                block_store<fp16, HD>(
                    out_final + q_head_base * HD + i * HD,
                    reduce_final.template select<HD, 1>(i * HD));
            }
        });
    });
}
