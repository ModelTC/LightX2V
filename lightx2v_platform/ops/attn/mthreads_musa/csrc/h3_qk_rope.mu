#include <torch/extension.h>
#include <musa_bf16.h>

#include "torch_musa/csrc/core/MUSAException.h"
#include "torch_musa/csrc/core/MUSAGuard.h"
#include "torch_musa/csrc/core/MUSAStream.h"

namespace {

constexpr int kHeadSize = 128;
constexpr int kRotarySize = 96;
constexpr int kRotaryHalf = 48;

__device__ __forceinline__ __mt_bfloat16 multiply(
    __mt_bfloat16 left,
    __mt_bfloat16 right) {
  return __float2bfloat16_rn(__bfloat162float(left) * __bfloat162float(right));
}

__device__ __forceinline__ __mt_bfloat16 add(
    __mt_bfloat16 left,
    __mt_bfloat16 right) {
  return __float2bfloat16_rn(__bfloat162float(left) + __bfloat162float(right));
}

__device__ __forceinline__ __mt_bfloat16 subtract(
    __mt_bfloat16 left,
    __mt_bfloat16 right) {
  return __float2bfloat16_rn(__bfloat162float(left) - __bfloat162float(right));
}

__global__ void h3_qk_rope_kernel(
    const __mt_bfloat16* query,
    const __mt_bfloat16* key,
    const __mt_bfloat16* cos,
    const __mt_bfloat16* sin,
    __mt_bfloat16* query_out,
    __mt_bfloat16* key_out,
    int heads,
    int query_token_stride,
    int query_head_stride,
    int key_token_stride,
    int key_head_stride) {
  const int token = blockIdx.x;
  const int head = blockIdx.y;
  const int column = threadIdx.x;
  const int output_offset = (token * heads + head) * kHeadSize + column;
  const int query_offset = token * query_token_stride + head * query_head_stride;
  const int key_offset = token * key_token_stride + head * key_head_stride;

  if (column >= kRotarySize) {
    query_out[output_offset] = query[query_offset + column];
    key_out[output_offset] = key[key_offset + column];
    return;
  }

  const int left_column = column < kRotaryHalf ? column : column - kRotaryHalf;
  const int right_column = left_column + kRotaryHalf;
  const __mt_bfloat16 frequency_cos = cos[token * kRotarySize + column];
  const __mt_bfloat16 frequency_sin = sin[token * kRotarySize + column];
  const __mt_bfloat16 query_left = query[query_offset + left_column];
  const __mt_bfloat16 query_right = query[query_offset + right_column];
  const __mt_bfloat16 key_left = key[key_offset + left_column];
  const __mt_bfloat16 key_right = key[key_offset + right_column];

  if (column < kRotaryHalf) {
    query_out[output_offset] = subtract(
        multiply(query_left, frequency_cos),
        multiply(query_right, frequency_sin));
    key_out[output_offset] = subtract(
        multiply(key_left, frequency_cos),
        multiply(key_right, frequency_sin));
  } else {
    query_out[output_offset] = add(
        multiply(query_left, frequency_sin),
        multiply(query_right, frequency_cos));
    key_out[output_offset] = add(
        multiply(key_left, frequency_sin),
        multiply(key_right, frequency_cos));
  }
}

__global__ void h3_qk_rope_fp32_kernel(
    const __mt_bfloat16* query,
    const __mt_bfloat16* key,
    const float* cos,
    const float* sin,
    __mt_bfloat16* query_out,
    __mt_bfloat16* key_out,
    int heads,
    int query_token_stride,
    int query_head_stride,
    int key_token_stride,
    int key_head_stride) {
  const int token = blockIdx.x;
  const int head = blockIdx.y;
  const int column = threadIdx.x;
  const int output_offset = (token * heads + head) * kHeadSize + column;
  const int query_offset = token * query_token_stride + head * query_head_stride;
  const int key_offset = token * key_token_stride + head * key_head_stride;

  if (column >= kRotarySize) {
    query_out[output_offset] = query[query_offset + column];
    key_out[output_offset] = key[key_offset + column];
    return;
  }

  const int partner_column = column < kRotaryHalf ? column + kRotaryHalf : column - kRotaryHalf;
  const float frequency_cos = cos[token * kRotarySize + column];
  const float frequency_sin = sin[token * kRotarySize + column];
  const float query_value = __bfloat162float(query[query_offset + column]);
  const float query_partner = __bfloat162float(query[query_offset + partner_column]);
  const float key_value = __bfloat162float(key[key_offset + column]);
  const float key_partner = __bfloat162float(key[key_offset + partner_column]);
  const float sign = column < kRotaryHalf ? -1.0f : 1.0f;
  const float query_cos = query_value * frequency_cos;
  const float query_sin = sign * query_partner * frequency_sin;
  const float key_cos = key_value * frequency_cos;
  const float key_sin = sign * key_partner * frequency_sin;
  query_out[output_offset] = __float2bfloat16_rn(query_cos + query_sin);
  key_out[output_offset] = __float2bfloat16_rn(key_cos + key_sin);
}

}  // namespace

std::vector<torch::Tensor> h3_qk_rope(
    torch::Tensor query,
    torch::Tensor key,
    torch::Tensor cos,
    torch::Tensor sin) {
  TORCH_CHECK(query.is_musa() && key.is_musa(), "query and key must be MUSA tensors");
  TORCH_CHECK(query.scalar_type() == torch::kBFloat16, "query must be BF16");
  TORCH_CHECK(key.scalar_type() == torch::kBFloat16, "key must be BF16");
  TORCH_CHECK(cos.scalar_type() == torch::kBFloat16, "cos must be BF16");
  TORCH_CHECK(sin.scalar_type() == torch::kBFloat16, "sin must be BF16");
  TORCH_CHECK(query.dim() == 3 && query.size(2) == kHeadSize, "query must be [T,H,128]");
  TORCH_CHECK(key.sizes() == query.sizes(), "key shape must match query");
  TORCH_CHECK(
      cos.dim() == 2 && cos.size(0) == query.size(0) && cos.size(1) == kRotarySize,
      "cos must be [T,96]");
  TORCH_CHECK(sin.sizes() == cos.sizes(), "sin shape must match cos");
  TORCH_CHECK(query.stride(2) == 1 && key.stride(2) == 1, "head dimension must be contiguous");
  TORCH_CHECK(cos.is_contiguous() && sin.is_contiguous(), "cos and sin must be contiguous");

  c10::musa::MUSAGuard device_guard(query.device());
  auto query_out = torch::empty_like(query, torch::MemoryFormat::Contiguous);
  auto key_out = torch::empty_like(key, torch::MemoryFormat::Contiguous);
  const int rows = query.size(0);
  const int heads = query.size(1);
  h3_qk_rope_kernel<<<
      dim3(rows, heads, 1),
      dim3(kHeadSize, 1, 1),
      0,
      c10::musa::getCurrentMUSAStream()>>>(
          reinterpret_cast<const __mt_bfloat16*>(query.data_ptr<at::BFloat16>()),
          reinterpret_cast<const __mt_bfloat16*>(key.data_ptr<at::BFloat16>()),
          reinterpret_cast<const __mt_bfloat16*>(cos.data_ptr<at::BFloat16>()),
          reinterpret_cast<const __mt_bfloat16*>(sin.data_ptr<at::BFloat16>()),
          reinterpret_cast<__mt_bfloat16*>(query_out.data_ptr<at::BFloat16>()),
          reinterpret_cast<__mt_bfloat16*>(key_out.data_ptr<at::BFloat16>()),
          heads,
          query.stride(0),
          query.stride(1),
          key.stride(0),
          key.stride(1));
  TORCH_MUSA_CHECK(musaGetLastError());
  return {query_out, key_out};
}

std::vector<torch::Tensor> h3_qk_rope_fp32(
    torch::Tensor query,
    torch::Tensor key,
    torch::Tensor cos,
    torch::Tensor sin) {
  TORCH_CHECK(query.is_musa() && key.is_musa(), "query and key must be MUSA tensors");
  TORCH_CHECK(query.scalar_type() == torch::kBFloat16, "query must be BF16");
  TORCH_CHECK(key.scalar_type() == torch::kBFloat16, "key must be BF16");
  TORCH_CHECK(cos.scalar_type() == torch::kFloat32, "cos must be FP32");
  TORCH_CHECK(sin.scalar_type() == torch::kFloat32, "sin must be FP32");
  TORCH_CHECK(query.dim() == 3 && query.size(2) == kHeadSize, "query must be [T,H,128]");
  TORCH_CHECK(key.sizes() == query.sizes(), "key shape must match query");
  TORCH_CHECK(
      cos.dim() == 2 && cos.size(0) == query.size(0) && cos.size(1) == kRotarySize,
      "cos must be [T,96]");
  TORCH_CHECK(sin.sizes() == cos.sizes(), "sin shape must match cos");
  TORCH_CHECK(query.stride(2) == 1 && key.stride(2) == 1, "head dimension must be contiguous");
  TORCH_CHECK(cos.is_contiguous() && sin.is_contiguous(), "cos and sin must be contiguous");

  c10::musa::MUSAGuard device_guard(query.device());
  auto query_out = torch::empty_like(query, torch::MemoryFormat::Contiguous);
  auto key_out = torch::empty_like(key, torch::MemoryFormat::Contiguous);
  const int rows = query.size(0);
  const int heads = query.size(1);
  h3_qk_rope_fp32_kernel<<<
      dim3(rows, heads, 1),
      dim3(kHeadSize, 1, 1),
      0,
      c10::musa::getCurrentMUSAStream()>>>(
          reinterpret_cast<const __mt_bfloat16*>(query.data_ptr<at::BFloat16>()),
          reinterpret_cast<const __mt_bfloat16*>(key.data_ptr<at::BFloat16>()),
          cos.data_ptr<float>(),
          sin.data_ptr<float>(),
          reinterpret_cast<__mt_bfloat16*>(query_out.data_ptr<at::BFloat16>()),
          reinterpret_cast<__mt_bfloat16*>(key_out.data_ptr<at::BFloat16>()),
          heads,
          query.stride(0),
          query.stride(1),
          key.stride(0),
          key.stride(1));
  TORCH_MUSA_CHECK(musaGetLastError());
  return {query_out, key_out};
}
