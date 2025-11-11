/*
 * Wrapper functions that bridge PyTorch tensors to the extracted Nunchaku SVDQuant kernels.
 */

#include <torch/extension.h>

#include "svdq/common.h"
#include "svdq/Tensor.h"
#include "svdq/interop/torch.h"
#include "svdq/kernels/zgemm/zgemm.h"

namespace lightx2v::svdq {

using OptionalTensor = c10::optional<torch::Tensor>;

namespace {

inline Tensor to_internal(OptionalTensor &tensor) {
    if (tensor.has_value()) {
        return from_torch(tensor.value());
    }
    return {};
}

} // namespace

void svdq_gemm_w4a4_cuda(OptionalTensor act,
                         OptionalTensor wgt,
                         OptionalTensor out,
                         OptionalTensor qout,
                         OptionalTensor ascales,
                         OptionalTensor wscales,
                         OptionalTensor oscales,
                         OptionalTensor poolout,
                         OptionalTensor lora_act_in,
                         OptionalTensor lora_up,
                         OptionalTensor lora_down,
                         OptionalTensor lora_act_out,
                         OptionalTensor norm_q,
                         OptionalTensor norm_k,
                         OptionalTensor rotary_emb,
                         OptionalTensor bias,
                         OptionalTensor smooth_factor,
                         OptionalTensor out_vk,
                         OptionalTensor out_linearattn,
                         bool act_unsigned,
                         std::vector<float> lora_scales,
                         bool fuse_silu,
                         bool fp4,
                         double alpha,
                         OptionalTensor wcscales,
                         OptionalTensor out_q,
                         OptionalTensor out_k,
                         OptionalTensor out_v,
                         int64_t attn_tokens) {
    TorchOpContext ctx;

    float alpha_f = static_cast<float>(alpha);

    nunchaku::kernels::gemm_w4a4(to_internal(act),
                                 to_internal(wgt),
                                 to_internal(out),
                                 to_internal(qout),
                                 to_internal(ascales),
                                 to_internal(wscales),
                                 to_internal(oscales),
                                 to_internal(poolout),
                                 to_internal(lora_act_in),
                                 to_internal(lora_up),
                                 to_internal(lora_down),
                                 to_internal(lora_act_out),
                                 to_internal(norm_q),
                                 to_internal(norm_k),
                                 to_internal(rotary_emb),
                                 to_internal(bias),
                                 to_internal(smooth_factor),
                                 to_internal(out_vk),
                                 to_internal(out_linearattn),
                                 act_unsigned,
                                 lora_scales,
                                 fuse_silu,
                                 fp4,
                                 alpha_f,
                                 to_internal(wcscales),
                                 to_internal(out_q),
                                 to_internal(out_k),
                                 to_internal(out_v),
                                 static_cast<int>(attn_tokens));
}

void svdq_quantize_w4a4_act_fuse_lora_cuda(OptionalTensor input,
                                           OptionalTensor output,
                                           OptionalTensor oscales,
                                           OptionalTensor lora_down,
                                           OptionalTensor lora_act_out,
                                           OptionalTensor smooth,
                                           bool fuse_glu,
                                           bool fp4) {
    TorchOpContext ctx;
    nunchaku::kernels::quantize_w4a4_act_fuse_lora(to_internal(input),
                                                   to_internal(output),
                                                   to_internal(oscales),
                                                   to_internal(lora_down),
                                                   to_internal(lora_act_out),
                                                   to_internal(smooth),
                                                   fuse_glu,
                                                   fp4);
}

} // namespace lightx2v::svdq
