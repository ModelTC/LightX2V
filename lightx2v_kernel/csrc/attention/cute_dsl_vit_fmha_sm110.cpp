#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <cmath>
#include <cstdint>
#include <limits>
#include <mutex>

#include "lightx2v_kernel_ops.h"
#include "vit_fmha_d64.h"
#include "vit_fmha_d72.h"
#include "vit_fmha_d80.h"
#include "vit_fmha_d128.h"

namespace {

vit_fmha_d64_Kernel_Module_t g_vit_fmha_d64{};
vit_fmha_d72_Kernel_Module_t g_vit_fmha_d72{};
vit_fmha_d80_Kernel_Module_t g_vit_fmha_d80{};
vit_fmha_d128_Kernel_Module_t g_vit_fmha_d128{};

std::once_flag g_vit_fmha_d64_once;
std::once_flag g_vit_fmha_d72_once;
std::once_flag g_vit_fmha_d80_once;
std::once_flag g_vit_fmha_d128_once;

void check_cuda(cudaError_t error, char const* operation) {
  TORCH_CHECK(error == cudaSuccess, operation, " failed: ", cudaGetErrorString(error));
}

void load_vit_fmha_d64() {
  vit_fmha_d64_Kernel_Module_Load(&g_vit_fmha_d64);
  check_cuda(cudaGetLastError(), "Loading vit_fmha_d64");
}

void load_vit_fmha_d72() {
  vit_fmha_d72_Kernel_Module_Load(&g_vit_fmha_d72);
  check_cuda(cudaGetLastError(), "Loading vit_fmha_d72");
}

void load_vit_fmha_d80() {
  vit_fmha_d80_Kernel_Module_Load(&g_vit_fmha_d80);
  check_cuda(cudaGetLastError(), "Loading vit_fmha_d80");
}

void load_vit_fmha_d128() {
  vit_fmha_d128_Kernel_Module_Load(&g_vit_fmha_d128);
  check_cuda(cudaGetLastError(), "Loading vit_fmha_d128");
}

}  // namespace

torch::Tensor cute_dsl_vit_fmha_sm110(
    torch::Tensor const& q,
    torch::Tensor const& k,
    torch::Tensor const& v,
    torch::Tensor const& cu_seqlens,
    int64_t max_seqlen) {
  TORCH_CHECK(q.is_cuda() && k.is_cuda() && v.is_cuda(), "Q, K, and V must be CUDA tensors.");
  TORCH_CHECK(cu_seqlens.is_cuda(), "cu_seqlens must be a CUDA tensor.");
  TORCH_CHECK(q.device() == k.device() && q.device() == v.device(), "Q, K, and V must be on the same device.");
  TORCH_CHECK(q.device() == cu_seqlens.device(), "Q and cu_seqlens must be on the same device.");
  TORCH_CHECK(q.scalar_type() == torch::kHalf, "Q must use FP16.");
  TORCH_CHECK(k.scalar_type() == torch::kHalf, "K must use FP16.");
  TORCH_CHECK(v.scalar_type() == torch::kHalf, "V must use FP16.");
  TORCH_CHECK(cu_seqlens.scalar_type() == torch::kInt32, "cu_seqlens must use INT32.");
  TORCH_CHECK(q.dim() == 3 && k.dim() == 3 && v.dim() == 3, "Q, K, and V must have shape [total_S, H, D].");
  TORCH_CHECK(q.sizes().equals(k.sizes()) && q.sizes().equals(v.sizes()), "ViT FMHA requires Q, K, and V to have identical shapes.");
  TORCH_CHECK(q.is_contiguous() && k.is_contiguous() && v.is_contiguous(), "Q, K, and V must be contiguous.");
  TORCH_CHECK(cu_seqlens.dim() == 1 && cu_seqlens.is_contiguous(), "cu_seqlens must be a contiguous 1D tensor.");
  TORCH_CHECK(cu_seqlens.numel() >= 2, "cu_seqlens must contain at least one sequence.");
  TORCH_CHECK(q.size(0) > 0 && q.size(1) > 0, "Q must contain at least one token and one head.");
  TORCH_CHECK(max_seqlen > 0 && max_seqlen <= q.size(0), "max_seqlen must be in (0, total_S].");
  TORCH_CHECK(q.size(0) <= std::numeric_limits<int32_t>::max(), "total_S exceeds the CuTe DSL INT32 ABI.");
  TORCH_CHECK(q.size(1) <= std::numeric_limits<int32_t>::max(), "The number of heads exceeds the CuTe DSL INT32 ABI.");
  TORCH_CHECK(q.size(2) <= std::numeric_limits<int32_t>::max(), "head_dim exceeds the CuTe DSL INT32 ABI.");
  TORCH_CHECK(cu_seqlens.numel() - 1 <= std::numeric_limits<int32_t>::max(), "The batch size exceeds the CuTe DSL INT32 ABI.");
  TORCH_CHECK(max_seqlen <= std::numeric_limits<int32_t>::max(), "max_seqlen exceeds the CuTe DSL INT32 ABI.");

  int32_t const total_seq_len = static_cast<int32_t>(q.size(0));
  int32_t const num_heads = static_cast<int32_t>(q.size(1));
  int32_t const head_dim = static_cast<int32_t>(q.size(2));
  int32_t const batch_size = static_cast<int32_t>(cu_seqlens.numel() - 1);
  int32_t const max_seq_len = static_cast<int32_t>(max_seqlen);
  TORCH_CHECK(
      head_dim == 64 || head_dim == 72 || head_dim == 80 || head_dim == 128,
      "Unsupported CuTe DSL ViT FMHA head dimension: ",
      head_dim,
      ". Expected 64, 72, 80, or 128.");

  at::cuda::CUDAGuard device_guard{static_cast<char>(q.get_device())};
  cudaDeviceProp device_properties{};
  check_cuda(cudaGetDeviceProperties(&device_properties, q.get_device()), "Querying CUDA device properties");
  int32_t const sm_version = device_properties.major * 10 + device_properties.minor;
  TORCH_CHECK(sm_version == 110, "The linked CuTe DSL artifact targets SM110, but the current device is SM", sm_version, ".");

  auto output = torch::empty_like(q);
  cudaStream_t const stream = at::cuda::getCurrentCUDAStream(q.get_device());
  float const softmax_scale = 1.0F / std::sqrt(static_cast<float>(head_dim));
  float const softmax_scale_log2 = softmax_scale * 1.4426950408889634F;
  float const output_scale = 1.0F;
  int32_t result = -1;

#define CALL_VIT_FMHA(PREFIX, MODULE)                                                                                \
  do {                                                                                                               \
    PREFIX##_Tensor_q_tensor_t q_tensor{};                                                                           \
    q_tensor.data = q.data_ptr();                                                                                    \
    q_tensor.dynamic_shapes[0] = total_seq_len;                                                                      \
    q_tensor.dynamic_shapes[1] = num_heads;                                                                          \
    q_tensor.dynamic_shapes[2] = head_dim;                                                                           \
    q_tensor.dynamic_strides[0] = static_cast<int64_t>(num_heads) * head_dim;                                        \
    q_tensor.dynamic_strides[1] = head_dim;                                                                           \
                                                                                                                      \
    PREFIX##_Tensor_k_tensor_t k_tensor{};                                                                           \
    k_tensor.data = k.data_ptr();                                                                                    \
    k_tensor.dynamic_shapes[0] = total_seq_len;                                                                      \
    k_tensor.dynamic_shapes[1] = num_heads;                                                                          \
    k_tensor.dynamic_shapes[2] = head_dim;                                                                           \
    k_tensor.dynamic_strides[0] = static_cast<int64_t>(num_heads) * head_dim;                                        \
    k_tensor.dynamic_strides[1] = head_dim;                                                                           \
                                                                                                                      \
    PREFIX##_Tensor_v_tensor_t v_tensor{};                                                                           \
    v_tensor.data = v.data_ptr();                                                                                    \
    v_tensor.dynamic_shapes[0] = total_seq_len;                                                                      \
    v_tensor.dynamic_shapes[1] = num_heads;                                                                          \
    v_tensor.dynamic_shapes[2] = head_dim;                                                                           \
    v_tensor.dynamic_strides[0] = static_cast<int64_t>(num_heads) * head_dim;                                        \
    v_tensor.dynamic_strides[1] = head_dim;                                                                           \
                                                                                                                      \
    PREFIX##_Tensor_o_tensor_t o_tensor{};                                                                           \
    o_tensor.data = output.data_ptr();                                                                               \
    o_tensor.dynamic_shapes[0] = total_seq_len;                                                                      \
    o_tensor.dynamic_shapes[1] = num_heads;                                                                          \
    o_tensor.dynamic_shapes[2] = head_dim;                                                                           \
    o_tensor.dynamic_strides[0] = static_cast<int64_t>(num_heads) * head_dim;                                        \
    o_tensor.dynamic_strides[1] = head_dim;                                                                           \
                                                                                                                      \
    PREFIX##_Tensor_cu_seqlens_t cu_seqlens_tensor{};                                                                \
    cu_seqlens_tensor.data = cu_seqlens.data_ptr();                                                                   \
    cu_seqlens_tensor.dynamic_shapes[0] = batch_size + 1;                                                            \
                                                                                                                      \
    result = cute_dsl_##PREFIX##_wrapper(                                                                            \
        &(MODULE),                                                                                                    \
        &q_tensor,                                                                                                    \
        &k_tensor,                                                                                                    \
        &v_tensor,                                                                                                    \
        &o_tensor,                                                                                                    \
        &cu_seqlens_tensor,                                                                                           \
        max_seq_len,                                                                                                  \
        softmax_scale_log2,                                                                                           \
        softmax_scale,                                                                                                \
        output_scale,                                                                                                 \
        stream);                                                                                                      \
  } while (false)

  if (head_dim == 64) {
    std::call_once(g_vit_fmha_d64_once, load_vit_fmha_d64);
    CALL_VIT_FMHA(vit_fmha_d64, g_vit_fmha_d64);
  } else if (head_dim == 72) {
    std::call_once(g_vit_fmha_d72_once, load_vit_fmha_d72);
    CALL_VIT_FMHA(vit_fmha_d72, g_vit_fmha_d72);
  } else if (head_dim == 80) {
    std::call_once(g_vit_fmha_d80_once, load_vit_fmha_d80);
    CALL_VIT_FMHA(vit_fmha_d80, g_vit_fmha_d80);
  } else {
    std::call_once(g_vit_fmha_d128_once, load_vit_fmha_d128);
    CALL_VIT_FMHA(vit_fmha_d128, g_vit_fmha_d128);
  }

#undef CALL_VIT_FMHA

  TORCH_CHECK(result == 0, "CuTe DSL ViT FMHA failed with error code ", result, ".");
  return output;
}
