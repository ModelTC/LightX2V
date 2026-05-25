// Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES
//
// Licensed under the Apache License, Version 2.0 (the "License").
// You may not use this file except in compliance with the License.
// To view a copy of this license, visit http://www.apache.org/licenses/LICENSE-2.0
//
// No warranties are given. The work is provided "AS IS", without warranty of any kind, express or implied.
//
// SPDX-License-Identifier: Apache-2.0
#include <ATen/Dispatch.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include <cstdint>
#include <vector>

namespace {

#define CHECK_CUDA_TENSOR(x) TORCH_CHECK((x).is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK((x).is_contiguous(), #x " must be contiguous")

__device__ __constant__ float kE2M1ToFloat[16] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f,
};

__device__ __forceinline__ int64_t blocked_scale_index(
    const int row,
    const int scale_col,
    const int scale_cols)
{
    // Inverse of fouroversix.quantize.utils.to_blocked for a scale matrix
    // shaped [rows_padded, scale_cols].
    const int row_block = row / 128;
    const int row_in_block = row - row_block * 128;
    const int scale_col_block = scale_col / 4;
    const int scale_col_in_block = scale_col - scale_col_block * 4;
    const int scale_col_blocks = scale_cols / 4;

    const int logical_block = row_block * scale_col_blocks + scale_col_block;
    return (((int64_t)logical_block * 32 + (row_in_block & 31)) * 16
            + (row_in_block >> 5) * 4 + scale_col_in_block);
}

template <typename scalar_t>
__global__ void fp4_kv_dequant_kernel(
    const uint64_t* __restrict__ value_ptrs,
    const uint64_t* __restrict__ scale_ptrs,
    const uint64_t* __restrict__ amax_ptrs,
    scalar_t* __restrict__ output,
    const int64_t total_packed_values,
    const int block_token_size,
    const int num_heads,
    const int packed_cols,
    const int scale_cols,
    const float inv_global_scale_denom)
{
    const int64_t packed_idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (packed_idx >= total_packed_values) {
        return;
    }

    const int col_pair = packed_idx % packed_cols;
    const int64_t global_row = packed_idx / packed_cols;
    const int rows_per_cache_block = block_token_size * num_heads;
    const int cache_block = global_row / rows_per_cache_block;
    const int row_in_cache_block = global_row - (int64_t)cache_block * rows_per_cache_block;

    const int token_in_block = row_in_cache_block / num_heads;
    const int head = row_in_cache_block - token_in_block * num_heads;
    const int out_token = cache_block * block_token_size + token_in_block;

    const auto* values = reinterpret_cast<const uint8_t*>(value_ptrs[cache_block]);
    const auto* scales = reinterpret_cast<const __nv_fp8_e4m3*>(scale_ptrs[cache_block]);
    const auto* amax = reinterpret_cast<const float*>(amax_ptrs[cache_block]);

    const uint8_t packed = values[(int64_t)row_in_cache_block * packed_cols + col_pair];
    const int scale_col = (col_pair * 2) / 16;
    const int64_t scale_idx = blocked_scale_index(row_in_cache_block, scale_col, scale_cols);
    const float scale = static_cast<float>(scales[scale_idx]);
    const float global_scale = amax[0] * inv_global_scale_denom;

    const float low = kE2M1ToFloat[packed & 0x0F] * scale * global_scale;
    const float high = kE2M1ToFloat[(packed >> 4) & 0x0F] * scale * global_scale;

    const int out_col = col_pair * 2;
    const int64_t out_base = (((int64_t)out_token * num_heads + head) * (packed_cols * 2)) + out_col;
    output[out_base] = static_cast<scalar_t>(low);
    output[out_base + 1] = static_cast<scalar_t>(high);
}

at::ScalarType dtype_code_to_scalar_type(const int64_t dtype_code)
{
    switch (dtype_code) {
    case 0:
        return at::ScalarType::BFloat16;
    case 1:
        return at::ScalarType::Half;
    case 2:
        return at::ScalarType::Float;
    default:
        TORCH_CHECK(false, "Unsupported KV dequant dtype code: ", dtype_code);
    }
    return at::ScalarType::Float;
}

at::Tensor make_device_pointer_tensor(at::TensorList tensors)
{
    auto options = at::TensorOptions()
                       .dtype(at::ScalarType::Long)
                       .device(tensors.front().device());
    at::Tensor ptrs = at::empty({static_cast<int64_t>(tensors.size())}, options);

    std::vector<int64_t> host_ptrs(tensors.size());
    for (size_t i = 0; i < tensors.size(); ++i) {
        host_ptrs[i] = reinterpret_cast<int64_t>(tensors[i].data_ptr());
    }

    // The pointer table is tiny; use a synchronous copy so the temporary host
    // vector cannot outlive an async H2D transfer.
    C10_CUDA_CHECK(cudaMemcpy(
        ptrs.data_ptr<int64_t>(),
        host_ptrs.data(),
        host_ptrs.size() * sizeof(int64_t),
        cudaMemcpyHostToDevice));
    return ptrs;
}

}  // namespace

at::Tensor dequantize_kv_cache_fp4_cuda(
    at::TensorList values,
    at::TensorList scale_factors,
    at::TensorList amax,
    int64_t num_heads,
    int64_t block_token_size,
    int64_t dtype_code,
    double e2m1_max,
    double e4m3_max)
{
    TORCH_CHECK(!values.empty(), "values must contain at least one cache block");
    TORCH_CHECK(values.size() == scale_factors.size(),
                "values and scale_factors must have the same length");
    TORCH_CHECK(values.size() == amax.size(),
                "values and amax must have the same length");
    TORCH_CHECK(num_heads > 0, "num_heads must be positive");
    TORCH_CHECK(block_token_size > 0, "block_token_size must be positive");
    TORCH_CHECK(e2m1_max > 0.0 && e4m3_max > 0.0,
                "e2m1_max and e4m3_max must be positive");

    const auto device = values.front().device();
    c10::cuda::CUDAGuard device_guard(device);
    const int64_t max_blocks = static_cast<int64_t>(values.size());
    const int64_t packed_cols = values.front().size(1);
    const int64_t head_dim = packed_cols * 2;
    const int64_t rows_padded = values.front().size(0);
    const int64_t logical_rows = block_token_size * num_heads;
    const int64_t scale_cols = head_dim / 16;

    TORCH_CHECK(head_dim == 128, "KV dequant currently expects head_dim=128, got ", head_dim);
    TORCH_CHECK(scale_cols % 4 == 0, "scale column count must be a multiple of 4");
    TORCH_CHECK(rows_padded >= logical_rows,
                "values rows are smaller than logical KV block rows");
    TORCH_CHECK(rows_padded % 128 == 0, "values rows must be padded to a multiple of 128");

    for (int64_t i = 0; i < max_blocks; ++i) {
        CHECK_CUDA_TENSOR(values[i]);
        CHECK_CUDA_TENSOR(scale_factors[i]);
        CHECK_CUDA_TENSOR(amax[i]);
        CHECK_CONTIGUOUS(values[i]);
        CHECK_CONTIGUOUS(scale_factors[i]);
        CHECK_CONTIGUOUS(amax[i]);
        TORCH_CHECK(values[i].device() == device, "all values tensors must be on the same device");
        TORCH_CHECK(scale_factors[i].device() == device,
                    "all scale_factors tensors must be on the same device");
        TORCH_CHECK(amax[i].device() == device, "all amax tensors must be on the same device");
        TORCH_CHECK(values[i].scalar_type() == at::ScalarType::Byte,
                    "values tensors must be uint8");
        TORCH_CHECK(amax[i].scalar_type() == at::ScalarType::Float,
                    "amax tensors must be float32");
        TORCH_CHECK(values[i].dim() == 2, "values tensors must be 2D");
        TORCH_CHECK(values[i].size(0) == rows_padded && values[i].size(1) == packed_cols,
                    "all values tensors must have the same shape");
    }

    const auto out_dtype = dtype_code_to_scalar_type(dtype_code);
    at::Tensor output = at::empty(
        {1, max_blocks * block_token_size, num_heads, head_dim},
        values.front().options().dtype(out_dtype));

    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    at::Tensor value_ptrs = make_device_pointer_tensor(values);
    at::Tensor scale_ptrs = make_device_pointer_tensor(scale_factors);
    at::Tensor amax_ptrs = make_device_pointer_tensor(amax);

    const int64_t total_packed_values = max_blocks * logical_rows * packed_cols;
    const int threads = 256;
    const dim3 blocks((total_packed_values + threads - 1) / threads);
    const float inv_global_scale_denom = static_cast<float>(1.0 / (e2m1_max * e4m3_max));

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        output.scalar_type(),
        "fp4_kv_dequant_kernel",
        [&] {
            fp4_kv_dequant_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
                reinterpret_cast<const uint64_t*>(value_ptrs.data_ptr<int64_t>()),
                reinterpret_cast<const uint64_t*>(scale_ptrs.data_ptr<int64_t>()),
                reinterpret_cast<const uint64_t*>(amax_ptrs.data_ptr<int64_t>()),
                output.data_ptr<scalar_t>(),
                total_packed_values,
                static_cast<int>(block_token_size),
                static_cast<int>(num_heads),
                static_cast<int>(packed_cols),
                static_cast<int>(scale_cols),
                inv_global_scale_denom);
        });
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return output;
}
