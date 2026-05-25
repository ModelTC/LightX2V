/******************************************************************************
 * Copyright (c) 2024, Tri Dao.
 ******************************************************************************/

// Include these 2 headers instead of torch/extension.h since we don't need all of the torch headers.
#pragma once

#include <torch/nn/functional.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <cutlass/numeric_types.h>

#include "include/flash.h"
#include "include/static_switch.h"

#define CHECK_DEVICE(x) TORCH_CHECK(x.is_cuda(), #x " must be on CUDA")
#define CHECK_SHAPE(x, ...) TORCH_CHECK(x.sizes() == torch::IntArrayRef({__VA_ARGS__}), #x " must have shape (" #__VA_ARGS__ ")")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")

void set_params_fprop(Flash_fwd_params &params,
                      // sizes
                      const size_t b,
                      const size_t seqlen_q,
                      const size_t seqlen_k,
                      const size_t seqlen_q_rounded,
                      const size_t seqlen_k_rounded,
                      const size_t h,
                      const size_t h_k,
                      const size_t d,
                      const size_t d_rounded,
                      // device pointers
                      const at::Tensor q,
                      const at::Tensor k, const at::Tensor k_pack, const at::Tensor k_params,
                      const at::Tensor v, const at::Tensor v_pack, const at::Tensor v_params,
                      at::Tensor out,
                      void *cu_seqlens_q_d,
                      void *cu_seqlens_k_d,
                      void *seqused_k,
                      void *p_d,
                      void *softmax_lse_d,
                      float p_dropout,
                      float softmax_scale,
                      int window_size_left,
                      int window_size_right,
                      const float softcap,
                      const std::string quant_mode,
                      const int group_size,
                      bool seqlenq_ngroups_swapped=false,
                      const bool unpadded_lse=false) {

    // Reset the parameters
    params = {};

    params.quant_mode = quant_mode;
    params.group_size = group_size;
    params.is_bf16 = q.dtype() == torch::kBFloat16;

    params.q_ptr = q.data_ptr();
    // params.k_ptr = k.data_ptr();
    params.K_pack_ptr = k_pack.data_ptr();
    params.k_params_ptr = k_params.data_ptr();
    // params.v_ptr = v.data_ptr();
    params.v_pack_ptr = v_pack.data_ptr();
    params.v_params_ptr = v_params.data_ptr();
    // All stride are in elements, not bytes.
    params.q_row_stride = q.stride(-3);
    // params.k_row_stride = k.stride(-3);
    params.K_pack_row_stride = k_pack.stride(-3);
    params.k_params_row_stride = k_params.stride(-3);
    // params.v_row_stride = v.stride(-3);
    params.v_pack_row_stride = v_pack.stride(-3);
    params.v_params_row_stride = v_params.stride(-1);

    params.k_params_dim_stride = k_params.stride(-1);
    params.v_params_dim_stride = v_params.stride(-3);

    params.q_head_stride = q.stride(-2);
    // params.k_head_stride = k.stride(-2);
    params.K_pack_head_stride = k_pack.stride(-2);
    params.k_params_head_stride = k_params.stride(-2);
    // params.v_head_stride = v.stride(-2);
    params.v_pack_head_stride = v_pack.stride(-2);
    params.v_params_head_stride = v_params.stride(-2);

    params.o_ptr = out.data_ptr();
    params.o_row_stride = out.stride(-3);
    params.o_head_stride = out.stride(-2);

    if (cu_seqlens_q_d == nullptr) {
        params.q_batch_stride = q.stride(0);
        // params.k_batch_stride = k.stride(0);
        params.K_pack_batch_stride = k_pack.stride(0);
        params.k_params_batch_stride = k_params.stride(0);
        // params.v_batch_stride = v.stride(0);
        params.v_pack_batch_stride = v_pack.stride(0);
        params.v_params_batch_stride = v_params.stride(0);
        params.o_batch_stride = out.stride(0);

        if (seqlenq_ngroups_swapped) {
            params.q_batch_stride *= seqlen_q;
            params.o_batch_stride *= seqlen_q;
        }
    }

    params.cu_seqlens_q = static_cast<int *>(cu_seqlens_q_d);
    params.cu_seqlens_k = static_cast<int *>(cu_seqlens_k_d);
    params.seqused_k = static_cast<int *>(seqused_k);

    // P = softmax(QK^T)
    params.p_ptr = p_d;

    // Softmax sum
    params.softmax_lse_ptr = softmax_lse_d;

    // Set the dimensions.
    params.b = b;
    params.h = h;
    params.h_k = h_k;
    params.h_h_k_ratio = h / h_k;
    params.seqlen_q = seqlen_q;
    params.seqlen_k = seqlen_k;
    params.seqlen_q_rounded = seqlen_q_rounded;
    params.seqlen_k_rounded = seqlen_k_rounded;
    params.d = d;
    params.d_rounded = d_rounded;

    // Set the different scale values.
    #ifdef FLASHATTENTION_DISABLE_SOFTCAP
    TORCH_CHECK(softcap <= 0.0, "This flash attention build does not support softcap.");
    #endif
    if (softcap > 0.0) {
        params.softcap = softmax_scale / softcap;
        params.scale_softmax = softcap;
        params.scale_softmax_log2 = softcap * M_LOG2E;
    } else{
        // Remove potential NaN
        params.softcap = 0.0;
        params.scale_softmax = softmax_scale;
        params.scale_softmax_log2 = softmax_scale * M_LOG2E;
    }

    // Set this to probability of keeping an element to simplify things.
    params.p_dropout = 1.f - p_dropout;
    // Convert p from float to int so we don't have to convert the random uint to float to compare.
    // [Minor] We want to round down since when we do the comparison we use <= instead of <
    // params.p_dropout_in_uint = uint32_t(std::floor(params.p_dropout * 4294967295.0));
    // params.p_dropout_in_uint16_t = uint16_t(std::floor(params.p_dropout * 65535.0));
    params.p_dropout_in_uint8_t = uint8_t(std::floor(params.p_dropout * 255.0));
    params.rp_dropout = 1.f / params.p_dropout;
    params.scale_softmax_rp_dropout = params.rp_dropout * params.scale_softmax;
    TORCH_CHECK(p_dropout < 1.f);
    #ifdef FLASHATTENTION_DISABLE_DROPOUT
    TORCH_CHECK(p_dropout == 0.0f, "This flash attention build does not support dropout.");
    #endif

    // Causal is the special case where window_size_right == 0 and window_size_left < 0.
    // Local is the more general case where window_size_right >= 0 or window_size_left >= 0.
    params.is_causal = window_size_left < 0 && window_size_right == 0;

    if (window_size_left < 0 && window_size_right >= 0) { window_size_left = seqlen_k; }
    if (window_size_left >= 0 && window_size_right < 0) { window_size_right = seqlen_k; }
    params.window_size_left = window_size_left;
    params.window_size_right = window_size_right;

    #ifdef FLASHATTENTION_DISABLE_LOCAL
    TORCH_CHECK(params.is_causal || (window_size_left < 0 && window_size_right < 0),
    "This flash attention build does not support local attention.");
    #endif

    params.is_seqlens_k_cumulative = true;

    #ifdef FLASHATTENTION_DISABLE_UNEVEN_K
    TORCH_CHECK(d == d_rounded, "This flash attention build does not support headdim not being a multiple of 32.");
    #endif

    params.unpadded_lse = unpadded_lse;
    params.seqlenq_ngroups_swapped = seqlenq_ngroups_swapped;
}

template<int num_bits>
void run_mha_fwd(Flash_fwd_params &params, cudaStream_t stream, bool force_split_kernel=false) {
    // FP16_SWITCH(!params.is_bf16, [&] {
    //     HEADDIM_SWITCH(params.d, [&] {
    //         BOOL_SWITCH(params.is_causal, Is_causal, [&] {
    //             if (params.num_splits <= 1 && !force_split_kernel) {  // If we don't set it num_splits == 0
    //                 run_mha_fwd_<elem_type, kHeadDim, Is_causal>(params, stream);
    //             } else {
    //                 run_mha_fwd_splitkv_dispatch<elem_type, kHeadDim, Is_causal>(params, stream);
    //             }
    //         });
    //     });
    // });
    if (params.num_splits <= 1 && !force_split_kernel) {  // If we don't set it num_splits == 0
        run_mha_fwd_<cutlass::half_t, 128, false>(params, stream);
    } else {
        if (params.quant_mode == "k-channel") {
            if (params.group_size == 128) {
                run_mha_fwd_splitkv_dispatch<cutlass::half_t, 128, false, 1, num_bits, 128>(params, stream);
            } else if (params.group_size == 64) {
                // run_mha_fwd_splitkv_dispatch<cutlass::half_t, 128, false, 1, num_bits, 64>(params, stream);
            } else if (params.group_size == 32) {
                run_mha_fwd_splitkv_dispatch<cutlass::half_t, 128, false, 1, num_bits, 32>(params, stream);
            }
        } else {
            if (params.group_size == 128) {
                // run_mha_fwd_splitkv_dispatch<cutlass::half_t, 128, false, 0, num_bits, 128>(params, stream);
            } else if (params.group_size == 64) {
                // run_mha_fwd_splitkv_dispatch<cutlass::half_t, 128, false, 0, num_bits, 64>(params, stream);
            } else if (params.group_size == 32) {
                // run_mha_fwd_splitkv_dispatch<cutlass::half_t, 128, false, 0, num_bits, 32>(params, stream);
            }
        }
    }
}

template <int num_bits>
void run_kvcache_qpack(Flash_fwd_params &params, cudaStream_t stream) {
    if (params.quant_mode == "k-channel") {
        if (params.group_size == 32) {
            run_kvcache_qpack_<cutlass::half_t, 128, 1, num_bits, 32>(params, stream);
        } else if (params.group_size == 64) {
            // run_kvcache_qpack_<cutlass::half_t, 128, 1, num_bits, 64>(params, stream);
        } else if (params.group_size == 128) {
            run_kvcache_qpack_<cutlass::half_t, 128, 1, num_bits, 128>(params, stream);
        }
    } else {
        if (params.group_size == 32) {
            // run_kvcache_qpack_<cutlass::half_t, 128, 0, num_bits, 32>(params, stream);
        } else if (params.group_size == 64) {
            // run_kvcache_qpack_<cutlass::half_t, 128, 0, num_bits, 64>(params, stream);
        } else if (params.group_size == 128) {
            // run_kvcache_qpack_<cutlass::half_t, 128, 0, num_bits, 128>(params, stream);
        }
    }
}

// Find the number of splits that maximizes the occupancy. For example, if we have
// batch * n_heads = 48 and we have 108 SMs, having 2 splits (efficiency = 0.89) is
// better than having 3 splits (efficiency = 0.67). However, we also don't want too many
// splits as that would incur more HBM reads/writes.
// So we find the best efficiency, then find the smallest number of splits that gets 85%
// of the best efficiency.
inline int num_splits_heuristic(int batch_nheads_mblocks, int num_SMs, int num_n_blocks, int max_splits) {
    // If we have enough to almost fill the SMs, then just use 1 split
    if (batch_nheads_mblocks >= 0.8f * num_SMs) { return 1; }
    max_splits = std::min({max_splits, num_SMs, num_n_blocks});
    float max_efficiency = 0.f;
    std::vector<float> efficiency;
    efficiency.reserve(max_splits);
    auto ceildiv = [](int a, int b) { return (a + b - 1) / b; };
    // Some splits are not eligible. For example, if we have 64 blocks and choose 11 splits,
    // we'll have 6 * 10 + 4 blocks. If we choose 12 splits, we'll have 6 * 11 + (-2) blocks
    // (i.e. it's 11 splits anyway).
    // So we check if the number of blocks per split is the same as the previous num_splits.
    auto is_split_eligible = [&ceildiv, &num_n_blocks](int num_splits) {
        return num_splits == 1 || ceildiv(num_n_blocks, num_splits) != ceildiv(num_n_blocks, num_splits - 1);
    };
    for (int num_splits = 1; num_splits <= max_splits; num_splits++) {
        if (!is_split_eligible(num_splits)) {
            efficiency.push_back(0.f);
        } else {
            float n_waves = float(batch_nheads_mblocks * num_splits) / num_SMs;
            float eff = n_waves / ceil(n_waves);
            // printf("num_splits = %d, eff = %f\n", num_splits, eff);
            if (eff > max_efficiency) { max_efficiency = eff; }
            efficiency.push_back(eff);
        }
    }
    for (int num_splits = 1; num_splits <= max_splits; num_splits++) {
        if (!is_split_eligible(num_splits)) { continue; }
        if (efficiency[num_splits - 1] >= 0.85 * max_efficiency) {
            // printf("num_splits chosen = %d\n", num_splits);
            return num_splits;
        }
    }
    return 1;
}

void set_params_splitkv(Flash_fwd_params &params, const int batch_size,
    const int num_heads, const int head_size, const int max_seqlen_k, const int max_seqlen_q,
    const int head_size_rounded, const float p_dropout,
    const int num_splits, const int block_n, cudaDeviceProp *dprops, struct c10::TensorOptions opts,
    at::Tensor &softmax_lse_accum, at::Tensor &out_accum) {

    // This needs to match with run_mha_fwd_splitkv_dispatch
    const int num_n_blocks = (max_seqlen_k + block_n - 1) / block_n;
    // Technically kBlockM = 64 only for the splitKV kernels, not the standard kernel.
    // In any case we don't expect seqlen_q to be larger than 64 for inference.
    const int num_m_blocks = (max_seqlen_q + 64 - 1) / 64;
    params.num_splits = num_splits;
    if (p_dropout == 0.0f) {  // SplitKV is not implemented for dropout
        if (num_splits < 1) {
            // We multiply number of SMs by 2 to hard-code the fact that we're using 128 threads per block.
            params.num_splits = num_splits_heuristic(batch_size * num_heads * num_m_blocks, dprops->multiProcessorCount * 2, num_n_blocks, 128);
            // printf("num_splits = %d\n", params.num_splits);
            // params.num_splits= 1;
        }
        const bool has_residual = params.new_lens > 0;
        if (!has_residual && params.num_splits < 2) {
            params.num_splits = 2;
        }
        params.num_splits += has_residual ? 1 : 0;  // Residual kernel owns the last split.
        // printf("num_splits = %d\n", params.num_splits);
        if (params.num_splits > 1) {
            softmax_lse_accum = torch::empty({params.num_splits, batch_size, num_heads, params.seqlen_q_rounded}, opts.dtype(at::kFloat));
            out_accum = torch::empty({params.num_splits, batch_size, num_heads, params.seqlen_q_rounded, head_size_rounded}, opts.dtype(at::kFloat));
            params.softmax_lseaccum_ptr = softmax_lse_accum.data_ptr();
            params.oaccum_ptr = out_accum.data_ptr();
        }
        TORCH_CHECK(params.num_splits <= 128, "num_splits > 128 not supported");
    }
}

template<int num_bits>
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
mha_fwd_kvcache(at::Tensor &q,                       // batch_size x seqlen_q x num_heads x head_size
                // const at::Tensor &kcache,                       // batch_size_c x seqlen_k x num_heads_k x head_size or num_blocks x page_block_size x num_heads_k x head_size if there's a block_table.
                const at::Tensor &k_pack,                       // batch_size_c x seqlen_k / 4 x num_heads_k x head_size or num_blocks x page_block_size x num_heads_k x head_size if there's a block_table.
                const at::Tensor &k_params,                     // batch_size_c x num_groups x num_heads_k x head_size
                // const at::Tensor &vcache,                       // batch_size_c x seqlen_k x num_heads_k x head_size or num_blocks x page_block_size x num_heads_k x head_size if there's a block_table.
                const at::Tensor &v_pack,                       // batch_size_c x seqlen_k / 4 x num_heads_k x head_size or num_blocks x page_block_size x num_heads_k x head_size if there's a block_table.
                const at::Tensor &v_params,                     // batch_size_c x num_groups x num_heads_k x head_size
                c10::optional<const at::Tensor> &k_,            // batch_size x seqlen_knew x num_heads_k x head_size
                c10::optional<const at::Tensor> &v_,            // batch_size x seqlen_knew x num_heads_k x head_size
                c10::optional<const at::Tensor> &seqlens_k_,    // batch_size
                at::Tensor &k_pack_new,
                at::Tensor &k_params_new,
                at::Tensor &v_pack_new,
                at::Tensor &v_params_new,
                c10::optional<at::Tensor> &block_table_,        // batch_size x max_num_blocks_per_seq
                const float softmax_scale=1.0,
                const std::string quant_mode="k-tensor",
                const int group_size=128,
                const int residual_block_size=128,
                const int new_lens=0,
                bool is_causal=false,
                int window_size_left=-1,
                int window_size_right=-1,
                const float softcap=0.0,
                bool is_rotary_interleaved=true,                // if true, rotary combines indices 0 & 1, else indices 0 & rotary_dim / 2
                int num_splits=0
                ) {

    auto dprops = at::cuda::getCurrentDeviceProperties();
    // bool is_sm75 = dprops->major == 7 && dprops->minor == 5;
    bool is_sm8x = dprops->major == 8 && dprops->minor >= 0;
    bool is_sm90 = dprops->major == 9 && dprops->minor == 0;
    TORCH_CHECK(is_sm90 || is_sm8x, "FlashAttention only supports Ampere GPUs or newer.");

    CHECK_DEVICE(q); // CHECK_DEVICE(kcache); CHECK_DEVICE(vcache);

    at::Tensor block_table;
    const bool paged_KV = block_table_.has_value();
    if (paged_KV) {
        block_table = block_table_.value();
        CHECK_DEVICE(block_table);
        TORCH_CHECK(block_table.dtype() == torch::kInt32, "block_table must have dtype torch.int32");
        TORCH_CHECK(block_table.stride(-1) == 1, "block_table must have contiguous last dimension");
    }

    auto q_dtype = q.dtype();
    const auto sizes = q.sizes();

    const int batch_size = sizes[0];
    int seqlen_q = sizes[1];
    int num_heads = sizes[2];
    const int head_size_og = sizes[3]; // dim

    const int max_num_blocks_per_seq = !paged_KV ? 0 : block_table.size(1);
    const int num_blocks             = !paged_KV ? 0 : v_pack.size(0);
    const int page_block_size        = !paged_KV ? 1 : v_pack.size(1);
    const int page_block_size_pack   = !paged_KV ? 1 : k_pack.size(1);
    const int seqlen_k               = !paged_KV ? v_pack.size(1) : max_num_blocks_per_seq * page_block_size;
    const int num_heads_k            = k_pack.size(2);
    const int batch_size_c           = !paged_KV ? k_pack.size(0) : batch_size;
    TORCH_CHECK(!paged_KV || page_block_size % 256 == 0, "Paged KV cache block size must be divisible by 256");
    TORCH_CHECK(batch_size > 0, "batch size must be postive");
    TORCH_CHECK(head_size_og <= 256, "FlashAttention forward only supports head dimension at most 256");
    TORCH_CHECK(num_heads % num_heads_k == 0, "Number of heads in key/value must divide number of heads in query");

    // causal=true is the same as causal=false in this case
    if (seqlen_q == 1) { is_causal = false; }

    // Faster to transpose q from (b, 1, (nheads_kv ngroups), d) to (b, ngroups, nheads_kv, d) in this case
    // H/t Daniel Haziza
    const int seqlenq_ngroups_swapped = seqlen_q == 1 && num_heads > num_heads_k && window_size_left < 0 && window_size_right < 0 && head_size_og % 8 == 0;
    if (seqlenq_ngroups_swapped) {
        const int ngroups = num_heads / num_heads_k;
        q = q.reshape({batch_size, num_heads_k, ngroups, head_size_og}).transpose(1, 2);
        seqlen_q = ngroups;
        num_heads = num_heads_k;
    }

    CHECK_SHAPE(q, batch_size, seqlen_q, num_heads, head_size_og);

    at::Tensor q_padded, kcache_padded, vcache_padded;
    q_padded = q;

    at::Tensor out;
    out = torch::empty_like(q_padded);

    auto round_multiple = [](int x, int m) { return (x + m - 1) / m * m; };
    const int head_size = round_multiple(head_size_og, 8);
    const int head_size_rounded = round_multiple(head_size, 32);
    const int seqlen_q_rounded = round_multiple(seqlen_q, 128);
    const int seqlen_k_rounded = round_multiple(seqlen_k, 128);

    // Otherwise the kernel will be launched from cuda:0 device
    // Cast to char to avoid compiler warning about narrowing
    at::cuda::CUDAGuard device_guard{(char)q.get_device()};

    auto opts = q.options();

    auto softmax_lse = torch::empty({batch_size, num_heads, seqlen_q}, opts.dtype(at::kFloat));

    Flash_fwd_params params;
    set_params_fprop(params,
                     batch_size,
                     seqlen_q, seqlen_k,
                     seqlen_q_rounded, seqlen_k_rounded,
                     num_heads, num_heads_k,
                     head_size, head_size_rounded,
                     q_padded,
                     kcache_padded, k_pack, k_params,
                     vcache_padded, v_pack, v_params,
                     out,
                     /*cu_seqlens_q_d=*/nullptr,
                     /*cu_seqlens_k_d=*/nullptr,
                     /*seqused_k=*/nullptr,
                     /*p_ptr=*/nullptr,
                     softmax_lse.data_ptr(),
                     /*p_dropout=*/0.f,
                     softmax_scale,
                     window_size_left,
                     window_size_right,
                     softcap,
                     quant_mode,
                     group_size
                     );

    at::Tensor k, v;
    if (k_.has_value()) {
        TORCH_CHECK(v_.has_value(), "If key is supplied, value must also be passed in");
        TORCH_CHECK(seqlens_k_.has_value(), "If key is supplied, seqlens_k must also be passed in");

        k = k_.value();
        v = v_.value();
        int seqlen_knew = k.size(1);
        auto seqlens_k = seqlens_k_.value();

        TORCH_CHECK(k.dtype() == q_dtype, "Key must have the same dtype as query");
        TORCH_CHECK(v.dtype() == q_dtype, "Value must have the same dtype as query");
        TORCH_CHECK(k.stride(-1) == 1, "Key tensor must have contiguous last dimension");
        TORCH_CHECK(v.stride(-1) == 1, "Value tensor must have contiguous last dimension");
        CHECK_SHAPE(k, batch_size, seqlen_knew, num_heads_k, head_size_og);
        CHECK_SHAPE(v, batch_size, seqlen_knew, num_heads_k, head_size_og);
        CHECK_DEVICE(k); CHECK_DEVICE(v);
        TORCH_CHECK(seqlens_k.dtype() == torch::kInt32, "seqlens_k must have dtype int32");
        CHECK_DEVICE(seqlens_k);
        CHECK_CONTIGUOUS(seqlens_k);
        CHECK_SHAPE(seqlens_k, batch_size);

        params.new_lens          = new_lens;

        params.seqlen_knew       = seqlen_knew;
        params.knew_ptr          = k.data_ptr();
        params.vnew_ptr          = v.data_ptr();
        params.knew_batch_stride = k.stride(0);
        params.vnew_batch_stride = v.stride(0);
        params.knew_row_stride   = k.stride(-3);
        params.vnew_row_stride   = v.stride(-3);
        params.knew_head_stride  = k.stride(-2);
        params.vnew_head_stride  = v.stride(-2);
        params.cu_seqlens_k      = static_cast<int *>(seqlens_k.data_ptr());

        const int pack_nums = 16 / num_bits;

        params.k_pack_new_ptr            = k_pack_new.data_ptr();
        params.k_params_new_ptr          = k_params_new.data_ptr();
        params.v_pack_new_ptr            = v_pack_new.data_ptr();
        params.v_params_new_ptr          = v_params_new.data_ptr();

        params.k_pack_new_batch_stride   = k_pack_new.stride(0);
        params.k_params_new_batch_stride = k_params_new.stride(0);
        params.v_pack_new_batch_stride   = v_pack_new.stride(0);
        params.v_params_new_batch_stride = v_params_new.stride(0);

        params.k_pack_new_row_stride     = k_pack_new.stride(-3);
        params.k_params_new_row_stride   = k_params_new.stride(-3);
        params.v_pack_new_row_stride     = v_pack_new.stride(-3);
        params.v_params_new_row_stride   = v_params_new.stride(-1);

        params.k_pack_new_head_stride    = k_pack_new.stride(-2);
        params.k_params_new_head_stride  = k_params_new.stride(-2);
        params.v_pack_new_head_stride    = v_pack_new.stride(-2);
        params.v_params_new_head_stride  = v_params_new.stride(-2);

        params.k_params_new_dim_stride   = k_params_new.stride(-1);
        params.v_params_new_dim_stride   = v_params_new.stride(-3);

    }

    params.is_seqlens_k_cumulative = !(seqlens_k_.has_value());
    params.rotary_dim = 0;

    if (paged_KV) {
        params.block_table = block_table.data_ptr<int>();
        params.block_table_batch_stride = block_table.stride(0);
    }
    params.page_block_size      = page_block_size;
    params.page_block_size_pack = page_block_size_pack;

    at::Tensor softmax_lse_accum;
    at::Tensor out_accum;
    set_params_splitkv(params, batch_size, num_heads,
                       head_size, seqlen_k, seqlen_q,
                       head_size_rounded, /*dropout*/0.f, num_splits,
                       256, dprops, opts,
                       softmax_lse_accum, out_accum);

    auto stream = at::cuda::getCurrentCUDAStream().stream();
    // Only split kernel supports appending to KV cache, or indexing to the cache with cache_batch_idx,
    // or paged KV cache
    run_mha_fwd<num_bits>(params, stream, /*force_split_kernel=*/true);

    if (seqlenq_ngroups_swapped) {
        out = out.transpose(1, 2).reshape({batch_size, 1, num_heads_k * seqlen_q, head_size_og});
    }

    return std::make_tuple(out, k_pack_new, k_params_new, v_pack_new, v_params_new);
}



///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// QPacking
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void set_params_fprop_qpack(Flash_fwd_params &params,
    // sizes
    const size_t b,
    const size_t seqlen_k,
    const size_t h, const size_t h_k,
    const size_t d,
    // device pointers
    const at::Tensor k, at::Tensor k_pack, at::Tensor k_params,
    const at::Tensor v, at::Tensor v_pack, at::Tensor v_params,
    void *cu_seqlens_k_d,
    const std::string quant_mode,
    const int group_size,
    bool page_kv
    ) {

    // Reset the parameters
    params = {};

    params.is_bf16 = k.dtype() == torch::kBFloat16;

    // Set the pointers and strides.
    params.k_ptr = k.data_ptr();
    params.K_pack_ptr = k_pack.data_ptr();
    params.k_params_ptr = k_params.data_ptr();
    params.v_ptr = v.data_ptr();
    params.v_pack_ptr = v_pack.data_ptr();
    params.v_params_ptr = v_params.data_ptr();
    // All stride are in elements, not bytes.
    params.k_row_stride = k.stride(-3);
    params.K_pack_row_stride = k_pack.stride(-3);
    params.k_params_row_stride = k_params.stride(-3);
    params.v_row_stride = v.stride(-3);
    params.v_pack_row_stride = v_pack.stride(-3);
    params.v_params_row_stride = v_params.stride(-1);

    params.k_params_dim_stride = k_params.stride(-1);
    params.v_params_dim_stride = v_params.stride(-3);

    params.k_head_stride = k.stride(-2);
    params.K_pack_head_stride = k_pack.stride(-2);
    params.k_params_head_stride = k_params.stride(-2);
    params.v_head_stride = v.stride(-2);
    params.v_pack_head_stride = v_pack.stride(-2);
    params.v_params_head_stride = v_params.stride(-2);

    if (page_kv) params.k_batch_stride = k.stride(0);
    else params.k_batch_stride = seqlen_k * k.size(-2) * k.size(-1);
    params.K_pack_batch_stride = k_pack.stride(0);
    params.k_params_batch_stride = k_params.stride(0);
    if (page_kv) params.v_batch_stride = v.stride(0);
    else params.v_batch_stride = seqlen_k * v.size(-2) * v.size(-1);
    params.v_pack_batch_stride = v_pack.stride(0);
    params.v_params_batch_stride = v_params.stride(0);

    params.cu_seqlens_k = static_cast<int *>(cu_seqlens_k_d);

    // Set the dimensions.
    params.b = b;
    params.h = h;
    params.h_k = h_k;
    params.h_h_k_ratio = h / h_k;
    params.seqlen_k = seqlen_k;
    params.d = d;

    params.quant_mode = quant_mode;
    params.group_size = group_size;
}

template <int num_bits>
void kvcache_qpack(const at::Tensor &k,
              at::Tensor &k_pack,
              at::Tensor &k_params,
              const at::Tensor &v,
              at::Tensor &v_pack,
              at::Tensor &v_params,
              c10::optional<at::Tensor> &block_table_,
              const at::Tensor &cu_seqlens_k,
              const int max_seqlen_k,
              const std::string quant_mode,
              const int group_size
             ) {

    auto k_dtype = k.dtype();
    TORCH_CHECK(k_dtype == torch::kFloat16 || k_dtype == torch::kBFloat16,
                "FlashAttention only support fp16 and bf16 data type");

    TORCH_CHECK(cu_seqlens_k.dtype() == torch::kInt32, "cu_seqlens_k must have dtype int32");

    CHECK_DEVICE(k); CHECK_DEVICE(v);
    CHECK_DEVICE(cu_seqlens_k);
    TORCH_CHECK(k.stride(-1) == 1, "Input tensor must have contiguous last dimension");
    TORCH_CHECK(v.stride(-1) == 1, "Input tensor must have contiguous last dimension");
    CHECK_CONTIGUOUS(cu_seqlens_k);

    at::Tensor block_table;
    const bool paged_KV = block_table_.has_value();
    if (paged_KV) {
        block_table = block_table_.value();
        CHECK_DEVICE(block_table);
        TORCH_CHECK(block_table.dtype() == torch::kInt32, "block_table must have dtype torch.int32");
        TORCH_CHECK(block_table.stride(-1) == 1, "block_table must have contiguous last dimension");
    }

    const auto sizes = k.sizes();

    const int batch_size  = cu_seqlens_k.numel() - 1;
    int num_heads         = paged_KV ? sizes[2] : sizes[1];
    const int head_size   = paged_KV ? sizes[3] : sizes[2];
    const int num_heads_k = paged_KV ? k.size(2) : k.size(1);

    const int max_num_blocks_per_seq = !paged_KV ? 0 : block_table.size(1);
    const int num_blocks             = !paged_KV ? 0 : k.size(0);
    const int page_block_size        = !paged_KV ? 1 : k.size(1);
    const int page_block_size_pack   = !paged_KV ? 0 : k_pack.size(1);
    const int seqlen_k               = !paged_KV ? k.size(1) : max_num_blocks_per_seq * page_block_size;
    const int batch_size_c           = !paged_KV ? k.size(0) : batch_size;

    TORCH_CHECK(!paged_KV || page_block_size % 256 == 0, "Paged KV cache block size must be divisible by 256");
    TORCH_CHECK(batch_size > 0, "batch size must be positive");
    TORCH_CHECK(head_size <= 256, "FlashAttention forward only supports head dimension at most 256");
    TORCH_CHECK(head_size % 8 == 0, "query, key, value, and out_ must have a head_size that is a multiple of 8");
    TORCH_CHECK(num_heads % num_heads_k == 0, "Number of heads in key/value must divide number of heads in query");
    CHECK_SHAPE(cu_seqlens_k, batch_size + 1);

    // Otherwise the kernel will be launched from cuda:0 device
    // Cast to char to avoid compiler warning about narrowing
    at::cuda::CUDAGuard device_guard{(char)k.get_device()};

    Flash_fwd_params params;
    set_params_fprop_qpack(params,
                           batch_size,
                           max_seqlen_k,
                           num_heads, num_heads_k,
                           head_size,
                           k, k_pack, k_params,
                           v, v_pack, v_params,
                           /*cu_seqlens_k_d=*/nullptr,
                           quant_mode,
                           group_size,
                           paged_KV
                           );

    if (paged_KV) {
        params.block_table = block_table.data_ptr<int>();
        params.block_table_batch_stride = block_table.stride(0);
    }
    params.page_block_size      = page_block_size;
    params.page_block_size_pack = page_block_size_pack;

    if (max_seqlen_k > 0) {
        auto stream = at::cuda::getCurrentCUDAStream().stream();
        run_kvcache_qpack<num_bits>(params, stream);
    }

    return;
}
