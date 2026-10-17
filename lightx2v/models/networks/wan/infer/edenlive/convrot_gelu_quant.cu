/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 Comfy Org. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Derived from Comfy-Org/comfy-kitchen, commit
 * b614102f8f0fcbba00d7fe570c31eeeda4f0061d,
 * comfy_kitchen/backends/cuda/ops/int8_linear.cu:
 * convrot64_quantize_row and quantize_int8_rowwise_convrot64_kernel.
 * https://github.com/Comfy-Org/comfy-kitchen/blob/b614102f8f0fcbba00d7fe570c31eeeda4f0061d/comfy_kitchen/backends/cuda/ops/int8_linear.cu
 *
 * Local changes specialize BF16, K=14336, group=256, deterministic quantization
 * and add GELU with its original BF16 rounding boundary. GEMM stays in the
 * installed comfy-kitchen backend. No torch C++ headers or Python bindings are
 * needed: the lazy-loaded extension exposes only CUDA launchers through C ABI.
 */

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>
#include <cmath>

extern "C" __device__ float __nv_tanhf(float);

namespace {

constexpr int kWidth = 14336;
constexpr int kGroup = 256;
constexpr int kThreads = 512;
constexpr int kWarps = kThreads / 32;
constexpr int kGroupsInFlight = kThreads / 64;
constexpr int kSharedBytes = kWidth * sizeof(__nv_bfloat16) + kGroupsInFlight * 2 * kGroup * sizeof(float);

// Explicit PTX rounding modes retain subnormal values even under the backend's
// --use_fast_math. The precise libdevice tanh also avoids its fast-tanh rewrite.
__device__ __forceinline__ float precise_mul(float a, float b) {
    float result;
    asm("mul.rn.f32 %0, %1, %2;" : "=f"(result) : "f"(a), "f"(b));
    return result;
}

__device__ __forceinline__ float precise_add(float a, float b) {
    float result;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(result) : "f"(a), "f"(b));
    return result;
}

__device__ __forceinline__ float bf16_round(float value) {
    return __bfloat162float(__float2bfloat16_rn(value));
}

__device__ __forceinline__ float rounded_gelu(float value) {
    const float half_value = precise_mul(value, 0.5f);
    const float cubic = precise_mul(precise_mul(value, value), value);
    const float inner = precise_mul(
        precise_add(value, precise_mul(cubic, 0.044715f)),
        0.7978845608028654f);
    return bf16_round(precise_mul(half_value, precise_add(1.0f, __nv_tanhf(inner))));
}

__device__ __forceinline__ float warp_reduce_max(float value) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        value = fmaxf(value, __shfl_down_sync(0xffffffff, value, offset));
    }
    return value;
}

__device__ __forceinline__ float block_reduce_max(float value, float* warp_smem, float* block_smem) {
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    value = warp_reduce_max(value);
    if (lane == 0) warp_smem[warp] = value;
    __syncthreads();
    if (warp == 0) {
        float total = lane < kWarps ? warp_smem[lane] : 0.0f;
        total = warp_reduce_max(total);
        if (lane == 0) *block_smem = total;
    }
    __syncthreads();
    return *block_smem;
}

template<int Stride>
__device__ __forceinline__ void fht_stage(const float* source, float* output, int lane) {
    const int base = (lane % Stride) + (lane / Stride) * (4 * Stride);
    const float x0 = source[base];
    const float x1 = source[base + Stride];
    const float x2 = source[base + 2 * Stride];
    const float x3 = source[base + 3 * Stride];
    output[base] = 0.5f * (x0 + x1 + x2 - x3);
    output[base + Stride] = 0.5f * (x0 + x1 - x2 + x3);
    output[base + 2 * Stride] = 0.5f * (x0 - x1 + x2 + x3);
    output[base + 3 * Stride] = 0.5f * (-x0 + x1 + x2 + x3);
}

__device__ __forceinline__ float fht_last_stage(const float* source, __nv_bfloat16* output, int lane) {
    const float x0 = source[lane];
    const float x1 = source[lane + 64];
    const float x2 = source[lane + 128];
    const float x3 = source[lane + 192];
    const float y0 = 0.5f * (x0 + x1 + x2 - x3);
    const float y1 = 0.5f * (x0 + x1 - x2 + x3);
    const float y2 = 0.5f * (x0 - x1 + x2 + x3);
    const float y3 = 0.5f * (-x0 + x1 + x2 + x3);
    // Scale reduction still consumes the unrounded FP32 registers. Quantization
    // immediately rounds these values to BF16, so that boundary can move here.
    output[lane] = __float2bfloat16_rn(y0);
    output[lane + 64] = __float2bfloat16_rn(y1);
    output[lane + 128] = __float2bfloat16_rn(y2);
    output[lane + 192] = __float2bfloat16_rn(y3);
    return fmaxf(fmaxf(fabsf(y0), fabsf(y1)), fmaxf(fabsf(y2), fabsf(y3)));
}

__global__ void gelu_convrot_quant_kernel(const __nv_bfloat16* input, int8_t* quantized, float* scales) {
    extern __shared__ float shared[];
    __shared__ float warp_smem[kWarps];
    __shared__ float block_smem;
    __nv_bfloat16* row_buffer = reinterpret_cast<__nv_bfloat16*>(shared);
    float* temporary = shared + kWidth / 2;
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int subgroup = tid / 64;
    const int lane = tid % 64;
    const int64_t row_offset = static_cast<int64_t>(row) * kWidth;
    float* buffer0 = temporary + subgroup * 2 * kGroup;
    float* buffer1 = buffer0 + kGroup;
    float abs_max = 0.0f;
    constexpr int iterations = (kWidth / kGroup + kGroupsInFlight - 1) / kGroupsInFlight;
    for (int iteration = 0; iteration < iterations; ++iteration) {
        const int group = iteration * kGroupsInFlight + subgroup;
        const bool active = group < kWidth / kGroup;
        const int base = lane * 4;
        const int group_col = group * kGroup;
        const int col = group_col + base;
        const float x0 = active ? rounded_gelu(__bfloat162float(input[row_offset + col])) : 0.0f;
        const float x1 = active ? rounded_gelu(__bfloat162float(input[row_offset + col + 1])) : 0.0f;
        const float x2 = active ? rounded_gelu(__bfloat162float(input[row_offset + col + 2])) : 0.0f;
        const float x3 = active ? rounded_gelu(__bfloat162float(input[row_offset + col + 3])) : 0.0f;
        buffer1[base] = 0.5f * (x0 + x1 + x2 - x3);
        buffer1[base + 1] = 0.5f * (x0 + x1 - x2 + x3);
        buffer1[base + 2] = 0.5f * (x0 - x1 + x2 + x3);
        buffer1[base + 3] = 0.5f * (-x0 + x1 + x2 + x3);
        __syncthreads();
        fht_stage<4>(buffer1, buffer0, lane);
        __syncthreads();
        fht_stage<16>(buffer0, buffer1, lane);
        __syncthreads();
        if (active) abs_max = fmaxf(abs_max, fht_last_stage(buffer1, row_buffer + group_col, lane));
        __syncthreads();
    }
    abs_max = block_reduce_max(abs_max, warp_smem, &block_smem);
    const float scale = fmaxf(fminf(abs_max, 3.38953139e38f) * (1.0f / 127.0f), 1.0e-30f);
    if (tid == 0) scales[row] = scale;
    const float rounded_scale = bf16_round(scale);
    for (int col = tid; col < kWidth; col += kThreads) {
        // The rotated-value BF16 rounding already happened at its shared store.
        const float scaled = bf16_round(__bfloat162float(row_buffer[col]) / rounded_scale);
        const float value = fminf(127.0f, fmaxf(-128.0f, nearbyintf(scaled)));
        quantized[row_offset + col] = static_cast<int8_t>(value);
    }
}

// This narrow export also permits exhaustive BF16 GELU validation without
// entangling overflow in the subsequent ConvRot transform.
__global__ void rounded_gelu_kernel(const __nv_bfloat16* input, __nv_bfloat16* output, int64_t count) {
    const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < count) output[index] = __float2bfloat16_rn(rounded_gelu(__bfloat162float(input[index])));
}

} // namespace

extern "C" int edenlive_prepare_convrot_gelu() {
    return static_cast<int>(cudaFuncSetAttribute(gelu_convrot_quant_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kSharedBytes));
}

extern "C" int edenlive_gelu_convrot_quant(const void* input, void* quantized, void* scales, int rows, void* stream) {
    if (rows == 0) return 0;
    gelu_convrot_quant_kernel<<<rows, kThreads, kSharedBytes, static_cast<cudaStream_t>(stream)>>>(
        static_cast<const __nv_bfloat16*>(input), static_cast<int8_t*>(quantized), static_cast<float*>(scales));
    return static_cast<int>(cudaGetLastError());
}

extern "C" int edenlive_rounded_gelu(const void* input, void* output, int64_t count, void* stream) {
    if (count == 0) return 0;
    rounded_gelu_kernel<<<(count + 255) / 256, 256, 0, static_cast<cudaStream_t>(stream)>>>(
        static_cast<const __nv_bfloat16*>(input), static_cast<__nv_bfloat16*>(output), count);
    return static_cast<int>(cudaGetLastError());
}
