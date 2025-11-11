#include <ATen/core/dispatch/Dispatcher.h>
#include <torch/all.h>
#include <torch/library.h>

#include "lightx2v_kernel_ops.h"

TORCH_LIBRARY_FRAGMENT(lightx2v_kernel, m) {
  m.def(
      "cutlass_scaled_nvfp4_mm_sm120(Tensor! out, Tensor mat_a, Tensor mat_b, Tensor scales_a, Tensor scales_b, Tensor "
      "alpha, Tensor? bias) -> ()");
  m.impl("cutlass_scaled_nvfp4_mm_sm120", torch::kCUDA, &cutlass_scaled_nvfp4_mm_sm120);

  m.def(
      "scaled_nvfp4_quant_sm120(Tensor! output, Tensor! input,"
      "                 Tensor! output_scale, Tensor! input_scale) -> ()");
  m.impl("scaled_nvfp4_quant_sm120", torch::kCUDA, &scaled_nvfp4_quant_sm120);

  m.def(
    "scaled_mxfp4_quant_sm120(Tensor! output, Tensor! input,"
    "                 Tensor! output_scale) -> ()");
  m.impl("scaled_mxfp4_quant_sm120", torch::kCUDA, &scaled_mxfp4_quant_sm120);

  m.def(
      "scaled_mxfp8_quant_sm120(Tensor! output, Tensor! input,"
      "                 Tensor! output_scale) -> ()");
  m.impl("scaled_mxfp8_quant_sm120", torch::kCUDA, &scaled_mxfp8_quant_sm120);

  m.def(
      "scaled_mxfp6_quant_sm120(Tensor! output, Tensor! input,"
      "                 Tensor! output_scale) -> ()");
  m.impl("scaled_mxfp6_quant_sm120", torch::kCUDA, &scaled_mxfp6_quant_sm120);

  m.def(
    "cutlass_scaled_mxfp4_mm_sm120(Tensor! out, Tensor mat_a, Tensor mat_b, Tensor scales_a, Tensor scales_b, Tensor "
    "alpha, Tensor? bias) -> ()");
  m.impl("cutlass_scaled_mxfp4_mm_sm120", torch::kCUDA, &cutlass_scaled_mxfp4_mm_sm120);

  m.def(
      "cutlass_scaled_mxfp6_mxfp8_mm_sm120(Tensor! out, Tensor mat_a, Tensor mat_b, Tensor scales_a, Tensor scales_b, Tensor "
      "alpha, Tensor? bias) -> ()");
  m.impl("cutlass_scaled_mxfp6_mxfp8_mm_sm120", torch::kCUDA, &cutlass_scaled_mxfp6_mxfp8_mm_sm120);

  m.def(
      "cutlass_scaled_mxfp8_mm_sm120(Tensor! out, Tensor mat_a, Tensor mat_b, Tensor scales_a, Tensor scales_b, Tensor "
      "alpha, Tensor? bias) -> ()");
  m.impl("cutlass_scaled_mxfp8_mm_sm120", torch::kCUDA, &cutlass_scaled_mxfp8_mm_sm120);

  m.def(
      "svdq_quantize_w4a4_act_fuse_lora_cuda(Tensor? input, Tensor? output, Tensor? oscales, Tensor? lora_down, Tensor? "
      "lora_act_out, Tensor? smooth, bool fuse_glu, bool fp4) -> ()");
  m.impl("svdq_quantize_w4a4_act_fuse_lora_cuda", torch::kCUDA, &lightx2v::svdq::svdq_quantize_w4a4_act_fuse_lora_cuda);

  m.def(
      "svdq_gemm_w4a4_cuda(Tensor? act, Tensor? wgt, Tensor? out, Tensor? qout, Tensor? ascales, Tensor? wscales, Tensor? "
      "oscales, Tensor? poolout, Tensor? lora_act_in, Tensor? lora_up, Tensor? lora_down, Tensor? lora_act_out, Tensor? "
      "norm_q, Tensor? norm_k, Tensor? rotary_emb, Tensor? bias, Tensor? smooth_factor, Tensor? out_vk, Tensor? "
      "out_linearattn, bool act_unsigned, float[] lora_scales, bool fuse_silu, bool fp4, float alpha, Tensor? wcscales, "
      "Tensor? out_q, Tensor? out_k, Tensor? out_v, int attn_tokens) -> ()");
  m.impl("svdq_gemm_w4a4_cuda", torch::kCUDA, &lightx2v::svdq::svdq_gemm_w4a4_cuda);
}

REGISTER_EXTENSION(common_ops)
