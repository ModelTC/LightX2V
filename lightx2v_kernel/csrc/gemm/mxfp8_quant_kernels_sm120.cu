#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <torch/all.h>

#include <array>
#include <mutex>

#include "sm120_utils.h"
#include "utils.h"

// Get type2 from type or vice versa (applied to half and bfloat16)
template <typename T>
struct TypeConverter {
  using Type = half2;
};  // keep for generality

template <>
struct TypeConverter<half2> {
  using Type = half;
};

template <>
struct TypeConverter<half> {
  using Type = half2;
};

template <>
struct TypeConverter<__nv_bfloat162> {
  using Type = __nv_bfloat16;
};

template <>
struct TypeConverter<__nv_bfloat16> {
  using Type = __nv_bfloat162;
};

#define ELTS_PER_THREAD 8

constexpr int CVT_FP8_ELTS_PER_THREAD = 8;
constexpr int CVT_FP8_SF_VEC_SIZE = 32;


// Convert 4 float2 values into 8 e4m3 values (represented as one uint64_t).
inline __device__ uint64_t fp32_vec_to_e4m3(float2 (&array)[4]) {
  uint64_t val;
  asm volatile(
      "{\n"
      ".reg .b16 pack0;\n"
      ".reg .b16 pack1;\n"
      ".reg .b16 pack2;\n"
      ".reg .b16 pack3;\n"
      "cvt.rn.satfinite.e4m3x2.f32   pack0, %2, %1;\n"
      "cvt.rn.satfinite.e4m3x2.f32   pack1, %4, %3;\n"
      "cvt.rn.satfinite.e4m3x2.f32   pack2, %6, %5;\n"
      "cvt.rn.satfinite.e4m3x2.f32   pack3, %8, %7;\n"
      "mov.b64 %0, {pack0, pack1, pack2, pack3};\n"
      "}"
      : "=l"(val)
      : "f"(array[0].x),
        "f"(array[0].y),
        "f"(array[1].x),
        "f"(array[1].y),
        "f"(array[2].x),
        "f"(array[2].y),
        "f"(array[3].x),
        "f"(array[3].y));
  return val;
}

// Fast reciprocal.
inline __device__ float reciprocal_approximate_ftz(float a) {
  float b;
  asm volatile("rcp.approx.ftz.f32 %0, %1;\n" : "=f"(b) : "f"(a));
  return b;
}

__device__ __forceinline__ float gelu_tanh_approx_mxfp8_quant(float x) {
  constexpr float kSqrt2OverPi = 0.7978845608028654f;
  constexpr float kCoeff = 0.044715f;
  float x3 = x * x * x;
  return 0.5f * x * (1.0f + tanhf(kSqrt2OverPi * (x + kCoeff * x3)));
}

template <class Type>
__device__ __forceinline__ float round_gelu_output_for_dtype(float x) {
  if constexpr (std::is_same_v<Type, half>) {
    return __half2float(__float2half(x));
  } else {
    return __bfloat162float(__float2bfloat16(x));
  }
}

template <class SFType, int CVT_FP8_NUM_THREADS_PER_SF>
__device__ uint8_t* get_sf_out_address(int rowIdx, int colIdx, int numCols, SFType* SFout) {
// #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  static_assert(CVT_FP8_NUM_THREADS_PER_SF == 4);

  // one of 4 threads write one SF to global memory.
  // TODO: stage through smem for packed STG.32
  // is it better than STG.8 from 4 threads ?
  if (threadIdx.x % CVT_FP8_NUM_THREADS_PER_SF == 0) {
    // SF vector index (16 elements share one SF in the K dimension).
    int32_t kIdx = colIdx / CVT_FP8_NUM_THREADS_PER_SF;
    int32_t mIdx = rowIdx;

    // SF layout [numMTiles, numKTiles, 32 (mTile), 4 (mTile), 4(kTile)]
    // --> index [mTileIdx, kTileIdx, outerMIdx, innerMIdx, innerKIdx]

    int32_t mTileIdx = mIdx / (32 * 4);
    // SF vector size 32.
    int factor = CVT_FP8_SF_VEC_SIZE * 4;
    int32_t numKTiles = (numCols + factor - 1) / factor;
    int64_t mTileStride = numKTiles * 32 * 4 * 4;

    int32_t kTileIdx = (kIdx / 4);
    int64_t kTileStride = 32 * 4 * 4;

    // M tile layout [32, 4] is column-major.
    int32_t outerMIdx = (mIdx % 32);    // same as (mIdx % 128) % 32
    int64_t outerMStride = 4 * 4;

    int32_t innerMIdx = (mIdx % (32 * 4)) / 32;
    int64_t innerMStride = 4;

    int32_t innerKIdx = (kIdx % 4);
    int64_t innerKStride = 1;

    // Compute the global offset.
    int64_t SFOffset = mTileIdx * mTileStride + kTileIdx * kTileStride + outerMIdx * outerMStride +
                       innerMIdx * innerMStride + innerKIdx * innerKStride;

    return reinterpret_cast<uint8_t*>(SFout) + SFOffset;
  } else {
    // Other threads do not write to SFout.
    return nullptr;
  }
}

// Define a 16 bytes packed data type.
template <class Type>
struct PackedVec {
  typename TypeConverter<Type>::Type elts[4];
};

template <>
struct PackedVec<__nv_fp8_e4m3> {
  __nv_fp8x2_e4m3 elts[8];
};

// Quantizes the provided PackedVec into the uint64_t output
template <class Type> // Type can be half or bfloat16
__device__ uint64_t cvt_warp_fp16_to_fp8(PackedVec<Type>& vec, uint8_t* SFout) {
// #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  // Get absolute maximum values among the local 8 values.
  auto localMax = __habs2(vec.elts[0]);

// Local maximum value.
#pragma unroll
  for (int i = 1; i < CVT_FP8_ELTS_PER_THREAD / 2; i++) {
    localMax = __hmax2(localMax, __habs2(vec.elts[i]));
  }

  // Get the absolute maximum among all 32 values (four threads).
  localMax = __hmax2(__shfl_xor_sync(uint32_t(-1), localMax, 1), localMax);
  localMax = __hmax2(__shfl_xor_sync(uint32_t(-1), localMax, 2), localMax);
  // Get the final absolute maximum values.
  float vecMax = float(__hmax(localMax.x, localMax.y));

  // Get the SF (max value of the vector / max value of e4m3).
  // maximum value of e4m3 = 448.0.
  // TODO: use half as compute data type.
  float SFValue = (vecMax / 448.0f);
  // 8 bits representation of the SF.
  uint8_t fp8SFVal;
  // Write the SF to global memory (STG.8).
  __nv_fp8_e8m0 tmp;
  tmp.__x = __nv_cvt_float_to_e8m0(SFValue, __NV_SATFINITE, cudaRoundPosInf);
  SFValue = static_cast<float>(tmp);
  fp8SFVal = tmp.__x;


  float outputScale =
      SFValue != 0 ? reciprocal_approximate_ftz(SFValue) : 0.0f;

  if (SFout) {
    // Write the SF to global memory (STG.8).
    *SFout = fp8SFVal;
  }

  // Convert the input to float.
  float2 fp2Vals[CVT_FP8_ELTS_PER_THREAD / 2];

#pragma unroll
  for (int i = 0; i < CVT_FP8_ELTS_PER_THREAD / 2; i++) {
    if constexpr (std::is_same_v<Type, half>) {
      fp2Vals[i] = __half22float2(vec.elts[i]);
    } else {
      fp2Vals[i] = __bfloat1622float2(vec.elts[i]);
    }
    fp2Vals[i].x *= outputScale;
    fp2Vals[i].y *= outputScale;
  }

  // Convert to e4m3 values.
  uint64_t e4m3Vec = fp32_vec_to_e4m3(fp2Vals);

  return e4m3Vec;
}

template <class Type> // Type can be half or bfloat16
__device__ uint64_t cvt_warp_gelu_fp16_to_fp8(PackedVec<Type>& vec, uint8_t* SFout) {
  float2 fp2Vals[CVT_FP8_ELTS_PER_THREAD / 2];

#pragma unroll
  for (int i = 0; i < CVT_FP8_ELTS_PER_THREAD / 2; i++) {
    if constexpr (std::is_same_v<Type, half>) {
      fp2Vals[i] = __half22float2(vec.elts[i]);
    } else {
      fp2Vals[i] = __bfloat1622float2(vec.elts[i]);
    }
    fp2Vals[i].x = round_gelu_output_for_dtype<Type>(gelu_tanh_approx_mxfp8_quant(fp2Vals[i].x));
    fp2Vals[i].y = round_gelu_output_for_dtype<Type>(gelu_tanh_approx_mxfp8_quant(fp2Vals[i].y));
  }

  float2 localMax2;
  localMax2.x = fmaxf(fabsf(fp2Vals[0].x), fabsf(fp2Vals[0].y));
  localMax2.y = fmaxf(fabsf(fp2Vals[1].x), fabsf(fp2Vals[1].y));

#pragma unroll
  for (int i = 2; i < CVT_FP8_ELTS_PER_THREAD / 2; i++) {
    localMax2.x = fmaxf(localMax2.x, fabsf(fp2Vals[i].x));
    localMax2.y = fmaxf(localMax2.y, fabsf(fp2Vals[i].y));
  }

  localMax2.x = fmaxf(__shfl_xor_sync(uint32_t(-1), localMax2.x, 1), localMax2.x);
  localMax2.y = fmaxf(__shfl_xor_sync(uint32_t(-1), localMax2.y, 1), localMax2.y);
  localMax2.x = fmaxf(__shfl_xor_sync(uint32_t(-1), localMax2.x, 2), localMax2.x);
  localMax2.y = fmaxf(__shfl_xor_sync(uint32_t(-1), localMax2.y, 2), localMax2.y);
  float vecMax = fmaxf(localMax2.x, localMax2.y);

  float SFValue = vecMax / 448.0f;
  __nv_fp8_e8m0 tmp;
  tmp.__x = __nv_cvt_float_to_e8m0(SFValue, __NV_SATFINITE, cudaRoundPosInf);
  SFValue = static_cast<float>(tmp);

  if (SFout) {
    *SFout = tmp.__x;
  }

  float outputScale = SFValue != 0 ? reciprocal_approximate_ftz(SFValue) : 0.0f;
#pragma unroll
  for (int i = 0; i < CVT_FP8_ELTS_PER_THREAD / 2; i++) {
    fp2Vals[i].x *= outputScale;
    fp2Vals[i].y *= outputScale;
  }

  return fp32_vec_to_e4m3(fp2Vals);
}

template <class Type>
__device__ __forceinline__ float convert_modulate_input(float x) {
  if constexpr (std::is_same_v<Type, half>) {
    return __half2float(__float2half(x));
  } else {
    return __bfloat162float(__float2bfloat16(x));
  }
}

template <class Type>
__device__ uint64_t cvt_warp_modulate_fp16_to_fp8(
    PackedVec<Type>& vec,
    Type const* scale,
    Type const* shift,
    int rowIdx,
    int colIdx,
    int numCols,
    bool scale_is_2d,
    bool shift_is_2d,
    uint8_t* SFout) {
  float2 fp2Vals[CVT_FP8_ELTS_PER_THREAD / 2];

#pragma unroll
  for (int i = 0; i < CVT_FP8_ELTS_PER_THREAD / 2; i++) {
    if constexpr (std::is_same_v<Type, half>) {
      fp2Vals[i] = __half22float2(vec.elts[i]);
    } else {
      fp2Vals[i] = __bfloat1622float2(vec.elts[i]);
    }

    const int col0 = colIdx * CVT_FP8_ELTS_PER_THREAD + i * 2;
    const int col1 = col0 + 1;
    const int64_t row_offset = static_cast<int64_t>(rowIdx) * numCols;
    const int64_t scale_offset0 = (scale_is_2d ? row_offset : 0) + col0;
    const int64_t scale_offset1 = (scale_is_2d ? row_offset : 0) + col1;
    const int64_t shift_offset0 = (shift_is_2d ? row_offset : 0) + col0;
    const int64_t shift_offset1 = (shift_is_2d ? row_offset : 0) + col1;

    float scale0;
    float scale1;
    float shift0;
    float shift1;
    if constexpr (std::is_same_v<Type, half>) {
      scale0 = __half2float(scale[scale_offset0]);
      scale1 = __half2float(scale[scale_offset1]);
      shift0 = __half2float(shift[shift_offset0]);
      shift1 = __half2float(shift[shift_offset1]);
    } else {
      scale0 = __bfloat162float(scale[scale_offset0]);
      scale1 = __bfloat162float(scale[scale_offset1]);
      shift0 = __bfloat162float(shift[shift_offset0]);
      shift1 = __bfloat162float(shift[shift_offset1]);
    }

    fp2Vals[i].x = convert_modulate_input<Type>(fp2Vals[i].x * (1.0f + scale0) + shift0);
    fp2Vals[i].y = convert_modulate_input<Type>(fp2Vals[i].y * (1.0f + scale1) + shift1);
  }

  float2 localMax2;
  localMax2.x = fmaxf(fabsf(fp2Vals[0].x), fabsf(fp2Vals[0].y));
  localMax2.y = fmaxf(fabsf(fp2Vals[1].x), fabsf(fp2Vals[1].y));

#pragma unroll
  for (int i = 2; i < CVT_FP8_ELTS_PER_THREAD / 2; i++) {
    localMax2.x = fmaxf(localMax2.x, fabsf(fp2Vals[i].x));
    localMax2.y = fmaxf(localMax2.y, fabsf(fp2Vals[i].y));
  }

  localMax2.x = fmaxf(__shfl_xor_sync(uint32_t(-1), localMax2.x, 1), localMax2.x);
  localMax2.y = fmaxf(__shfl_xor_sync(uint32_t(-1), localMax2.y, 1), localMax2.y);
  localMax2.x = fmaxf(__shfl_xor_sync(uint32_t(-1), localMax2.x, 2), localMax2.x);
  localMax2.y = fmaxf(__shfl_xor_sync(uint32_t(-1), localMax2.y, 2), localMax2.y);
  float vecMax = fmaxf(localMax2.x, localMax2.y);

  float SFValue = vecMax / 448.0f;
  __nv_fp8_e8m0 tmp;
  tmp.__x = __nv_cvt_float_to_e8m0(SFValue, __NV_SATFINITE, cudaRoundPosInf);
  SFValue = static_cast<float>(tmp);

  if (SFout) {
    *SFout = tmp.__x;
  }

  float outputScale = SFValue != 0 ? reciprocal_approximate_ftz(SFValue) : 0.0f;
#pragma unroll
  for (int i = 0; i < CVT_FP8_ELTS_PER_THREAD / 2; i++) {
    fp2Vals[i].x *= outputScale;
    fp2Vals[i].y *= outputScale;
  }

  return fp32_vec_to_e4m3(fp2Vals);
}


template <class Type> // Type can be half or bfloat16
__global__ void
// #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
__launch_bounds__(256, 6) cvt_fp16_to_fp8(
// #else
// cvt_fp16_to_fp8(
// #endif
    int32_t numRows, int32_t numCols, Type const* in, uint64_t* out, uint32_t* SFout) {
// #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  using PackedVec = PackedVec<Type>;
  static constexpr int CVT_FP8_NUM_THREADS_PER_SF = (CVT_FP8_SF_VEC_SIZE / CVT_FP8_ELTS_PER_THREAD);
  static_assert(sizeof(PackedVec) == sizeof(Type) * CVT_FP8_ELTS_PER_THREAD, "Vec size is not matched.");

  // Input tensor row/col loops.
  for (int rowIdx = blockIdx.x; rowIdx < numRows; rowIdx += gridDim.x) {
    for (int colIdx = threadIdx.x; colIdx < numCols / CVT_FP8_ELTS_PER_THREAD; colIdx += blockDim.x) {
      int64_t inOffset = rowIdx * (numCols / CVT_FP8_ELTS_PER_THREAD) + colIdx;
      PackedVec in_vec = reinterpret_cast<PackedVec const*>(in)[inOffset];
      // Get the output tensor offset.
      // Same as inOffset because 8 elements(E4M3) are packed into one uint64_t.
      int64_t outOffset = inOffset;
      auto& out_pos = out[outOffset];

      auto sf_out =
          get_sf_out_address<uint32_t, CVT_FP8_NUM_THREADS_PER_SF>(rowIdx, colIdx, numCols, SFout);

      out_pos = cvt_warp_fp16_to_fp8<Type>(in_vec, sf_out);
    }
  }
// #endif
}

template <class Type> // Type can be half or bfloat16
__global__ void __launch_bounds__(256, 6) cvt_gelu_fp16_to_fp8(
    int32_t numRows, int32_t numCols, Type const* in, uint64_t* out, uint32_t* SFout) {
  using PackedVec = PackedVec<Type>;
  static constexpr int CVT_FP8_NUM_THREADS_PER_SF = (CVT_FP8_SF_VEC_SIZE / CVT_FP8_ELTS_PER_THREAD);
  static_assert(sizeof(PackedVec) == sizeof(Type) * CVT_FP8_ELTS_PER_THREAD, "Vec size is not matched.");

  for (int rowIdx = blockIdx.x; rowIdx < numRows; rowIdx += gridDim.x) {
    for (int colIdx = threadIdx.x; colIdx < numCols / CVT_FP8_ELTS_PER_THREAD; colIdx += blockDim.x) {
      int64_t inOffset = rowIdx * (numCols / CVT_FP8_ELTS_PER_THREAD) + colIdx;
      PackedVec in_vec = reinterpret_cast<PackedVec const*>(in)[inOffset];
      int64_t outOffset = inOffset;
      auto& out_pos = out[outOffset];

      auto sf_out =
          get_sf_out_address<uint32_t, CVT_FP8_NUM_THREADS_PER_SF>(rowIdx, colIdx, numCols, SFout);

      out_pos = cvt_warp_gelu_fp16_to_fp8<Type>(in_vec, sf_out);
    }
  }
}

template <class Type> // Type can be half or bfloat16
__global__ void __launch_bounds__(256, 6) cvt_modulate_fp16_to_fp8(
    int32_t numRows,
    int32_t numCols,
    Type const* in,
    Type const* scale,
    Type const* shift,
    bool scale_is_2d,
    bool shift_is_2d,
    uint64_t* out,
    uint32_t* SFout) {
  using PackedVec = PackedVec<Type>;
  static constexpr int CVT_FP8_NUM_THREADS_PER_SF = (CVT_FP8_SF_VEC_SIZE / CVT_FP8_ELTS_PER_THREAD);
  static_assert(sizeof(PackedVec) == sizeof(Type) * CVT_FP8_ELTS_PER_THREAD, "Vec size is not matched.");

  for (int rowIdx = blockIdx.x; rowIdx < numRows; rowIdx += gridDim.x) {
    for (int colIdx = threadIdx.x; colIdx < numCols / CVT_FP8_ELTS_PER_THREAD; colIdx += blockDim.x) {
      int64_t inOffset = rowIdx * (numCols / CVT_FP8_ELTS_PER_THREAD) + colIdx;
      PackedVec in_vec = reinterpret_cast<PackedVec const*>(in)[inOffset];
      int64_t outOffset = inOffset;
      auto& out_pos = out[outOffset];

      auto sf_out =
          get_sf_out_address<uint32_t, CVT_FP8_NUM_THREADS_PER_SF>(rowIdx, colIdx, numCols, SFout);

      out_pos = cvt_warp_modulate_fp16_to_fp8<Type>(
          in_vec, scale, shift, rowIdx, colIdx, numCols, scale_is_2d, shift_is_2d, sf_out);
    }
  }
}

template <typename T>
void invokeFP8Quantization(
    int m,
    int n,
    T const* input,
    int64_t* output,
    int32_t* SFOuput,
    int multiProcessorCount,
    cudaStream_t stream) {
  // Grid, Block size.
  // Each thread converts 8 values.
  dim3 block(std::min(int(n / ELTS_PER_THREAD), 256));
  // Get number of blocks per SM (assume we can fully utilize the SM).
  int const numBlocksPerSM = 1536 / block.x;
  dim3 grid(std::min(int(m), multiProcessorCount * numBlocksPerSM));

  // Launch the cvt kernel.
    cvt_fp16_to_fp8<T>
    <<<grid, block, 0, stream>>>(
        m, n, input, reinterpret_cast<uint64_t*>(output), reinterpret_cast<uint32_t*>(SFOuput));
}

template <typename T>
void invokeGeluFP8Quantization(
    int m,
    int n,
    T const* input,
    int64_t* output,
    int32_t* SFOuput,
    int multiProcessorCount,
    cudaStream_t stream) {
  dim3 block(std::min(int(n / ELTS_PER_THREAD), 256));
  int const numBlocksPerSM = 1536 / block.x;
  dim3 grid(std::min(int(m), multiProcessorCount * numBlocksPerSM));

  cvt_gelu_fp16_to_fp8<T>
      <<<grid, block, 0, stream>>>(
          m, n, input, reinterpret_cast<uint64_t*>(output), reinterpret_cast<uint32_t*>(SFOuput));
}

template <typename T>
void invokeModulateFP8Quantization(
    int m,
    int n,
    T const* input,
    T const* scale,
    T const* shift,
    bool scale_is_2d,
    bool shift_is_2d,
    int64_t* output,
    int32_t* SFOuput,
    int multiProcessorCount,
    cudaStream_t stream) {
  dim3 block(std::min(int(n / ELTS_PER_THREAD), 256));
  int const numBlocksPerSM = 1536 / block.x;
  dim3 grid(std::min(int(m), multiProcessorCount * numBlocksPerSM));

  cvt_modulate_fp16_to_fp8<T>
      <<<grid, block, 0, stream>>>(
          m,
          n,
          input,
          scale,
          shift,
          scale_is_2d,
          shift_is_2d,
          reinterpret_cast<uint64_t*>(output),
          reinterpret_cast<uint32_t*>(SFOuput));
}

// Instantiate the function.
template void invokeFP8Quantization(
    int m,
    int n,
    half const* input,
    int64_t* output,
    int32_t* SFOuput,
    int multiProcessorCount,
    cudaStream_t stream);

template void invokeGeluFP8Quantization(
    int m,
    int n,
    half const* input,
    int64_t* output,
    int32_t* SFOuput,
    int multiProcessorCount,
    cudaStream_t stream);

template void invokeGeluFP8Quantization(
    int m,
    int n,
    __nv_bfloat16 const* input,
    int64_t* output,
    int32_t* SFOuput,
    int multiProcessorCount,
    cudaStream_t stream);

template void invokeModulateFP8Quantization(
    int m,
    int n,
    half const* input,
    half const* scale,
    half const* shift,
    bool scale_is_2d,
    bool shift_is_2d,
    int64_t* output,
    int32_t* SFOuput,
    int multiProcessorCount,
    cudaStream_t stream);

template void invokeModulateFP8Quantization(
    int m,
    int n,
    __nv_bfloat16 const* input,
    __nv_bfloat16 const* scale,
    __nv_bfloat16 const* shift,
    bool scale_is_2d,
    bool shift_is_2d,
    int64_t* output,
    int32_t* SFOuput,
    int multiProcessorCount,
    cudaStream_t stream);

template void invokeFP8Quantization(
    int m,
    int n,
    __nv_bfloat16 const* input,
    int64_t* output,
    int32_t* SFOuput,
    int multiProcessorCount,
    cudaStream_t stream);

namespace {

inline int64_t round_up_mxfp8_m_tile(int64_t m) {
  return ((m + 127) / 128) * 128;
}

void check_mxfp8_quant_io(
    torch::Tensor const& output,
    torch::Tensor const& input,
    torch::Tensor const& output_sf,
    char const* op_name) {
  TORCH_CHECK(input.dim() == 2, op_name, " expects a 2D input tensor.");
  TORCH_CHECK(input.is_cuda(), op_name, " input must be a CUDA tensor.");
  TORCH_CHECK(input.is_contiguous(), op_name, " input must be contiguous.");
  TORCH_CHECK(
      input.scalar_type() == torch::kHalf || input.scalar_type() == torch::kBFloat16,
      op_name,
      " input dtype must be float16 or bfloat16, got ",
      input.scalar_type());

  TORCH_CHECK(output.dim() == 2, op_name, " output must be a 2D tensor.");
  TORCH_CHECK(output.is_cuda(), op_name, " output must be a CUDA tensor.");
  TORCH_CHECK(output.is_contiguous(), op_name, " output must be contiguous.");
  TORCH_CHECK(output.scalar_type() == torch::kUInt8, op_name, " output dtype must be uint8, got ", output.scalar_type());
  TORCH_CHECK(output.get_device() == input.get_device(), op_name, " output must be on the same CUDA device as input.");
  TORCH_CHECK(
      output.size(0) == input.size(0) && output.size(1) == input.size(1),
      op_name,
      " output shape must match input shape, got output=(",
      output.size(0),
      ", ",
      output.size(1),
      ") input=(",
      input.size(0),
      ", ",
      input.size(1),
      ")");

  int64_t m = input.size(0);
  int64_t n = input.size(1);
  TORCH_CHECK(n % 32 == 0, op_name, " N dimension must be multiple of 32.");

  int64_t expected_sf_m = round_up_mxfp8_m_tile(m);
  int64_t expected_sf_n = (n / 32 + 3) / 4;
  TORCH_CHECK(output_sf.dim() == 2, op_name, " output_sf must be a 2D tensor.");
  TORCH_CHECK(output_sf.is_cuda(), op_name, " output_sf must be a CUDA tensor.");
  TORCH_CHECK(output_sf.is_contiguous(), op_name, " output_sf must be contiguous.");
  TORCH_CHECK(
      output_sf.scalar_type() == torch::kInt32,
      op_name,
      " output_sf dtype must be int32 storage, got ",
      output_sf.scalar_type());
  TORCH_CHECK(
      output_sf.get_device() == input.get_device(),
      op_name,
      " output_sf must be on the same CUDA device as input.");
  TORCH_CHECK(
      output_sf.size(0) == expected_sf_m && output_sf.size(1) == expected_sf_n,
      op_name,
      " output_sf shape must be (",
      expected_sf_m,
      ", ",
      expected_sf_n,
      "), got (",
      output_sf.size(0),
      ", ",
      output_sf.size(1),
      ")");
}

}  // namespace

void scaled_mxfp8_quant_sm120(
    torch::Tensor& output, torch::Tensor const& input, torch::Tensor& output_sf) {
  char const* op_name = "scaled_mxfp8_quant_sm120";
  check_mxfp8_quant_io(output, input, output_sf, op_name);
  c10::cuda::CUDAGuard device_guard(input.device());
  lightx2v_kernel::check_sm120_or_throw(input, op_name);

  int32_t m = input.size(0);
  int32_t n = input.size(1);

  int multiProcessorCount = lightx2v_kernel::getMultiProcessorCount(input.get_device());

  auto sf_out = static_cast<int32_t*>(output_sf.data_ptr());
  auto output_ptr = static_cast<int64_t*>(output.data_ptr());
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream(input.get_device());

  switch (input.scalar_type()) {
    case torch::kHalf: {
      auto input_ptr = reinterpret_cast<half const*>(input.data_ptr());
      invokeFP8Quantization(m, n, input_ptr, output_ptr, sf_out, multiProcessorCount, stream);
      break;
    }
    case torch::kBFloat16: {
      auto input_ptr = reinterpret_cast<__nv_bfloat16 const*>(input.data_ptr());
      invokeFP8Quantization(m, n, input_ptr, output_ptr, sf_out, multiProcessorCount, stream);
      break;
    }
    default: {
      TORCH_CHECK(false, "Unsupported input data type for quantize_to_fp8: ", input.scalar_type());
    }
  }
}

void scaled_mxfp8_gelu_quant_sm120(
    torch::Tensor& output, torch::Tensor const& input, torch::Tensor& output_sf) {
  char const* op_name = "scaled_mxfp8_gelu_quant_sm120";
  check_mxfp8_quant_io(output, input, output_sf, op_name);
  c10::cuda::CUDAGuard device_guard(input.device());
  lightx2v_kernel::check_sm120_or_throw(input, op_name);

  int32_t m = input.size(0);
  int32_t n = input.size(1);

  int multiProcessorCount = lightx2v_kernel::getMultiProcessorCount(input.get_device());

  auto sf_out = static_cast<int32_t*>(output_sf.data_ptr());
  auto output_ptr = static_cast<int64_t*>(output.data_ptr());
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream(input.get_device());

  switch (input.scalar_type()) {
    case torch::kHalf: {
      auto input_ptr = reinterpret_cast<half const*>(input.data_ptr());
      invokeGeluFP8Quantization(m, n, input_ptr, output_ptr, sf_out, multiProcessorCount, stream);
      break;
    }
    case torch::kBFloat16: {
      auto input_ptr = reinterpret_cast<__nv_bfloat16 const*>(input.data_ptr());
      invokeGeluFP8Quantization(m, n, input_ptr, output_ptr, sf_out, multiProcessorCount, stream);
      break;
    }
    default: {
      TORCH_CHECK(false, "Unsupported input data type for gelu_quantize_to_fp8: ", input.scalar_type());
    }
  }
}

namespace {

bool is_mxfp8_modulate_param_2d(torch::Tensor const& tensor, int32_t m, int32_t n, char const* name) {
  TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor.");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous.");
  if (tensor.dim() == 1) {
    TORCH_CHECK(tensor.size(0) == n, name, " must have shape (N,) for per-column MXFP8 modulate quant.");
    return false;
  }
  if (tensor.dim() == 2) {
    TORCH_CHECK(
        tensor.size(0) == m && tensor.size(1) == n,
        name,
        " must have shape (M, N) for per-token MXFP8 modulate quant.");
    return true;
  }
  TORCH_CHECK(false, name, " must be either 1D (N,) or 2D (M, N) for MXFP8 modulate quant.");
  return false;
}

}  // namespace

void scaled_mxfp8_modulate_quant_sm120(
    torch::Tensor& output,
    torch::Tensor const& input,
    torch::Tensor const& scale,
    torch::Tensor const& shift,
    torch::Tensor& output_sf) {
  char const* op_name = "scaled_mxfp8_modulate_quant_sm120";
  check_mxfp8_quant_io(output, input, output_sf, op_name);
  c10::cuda::CUDAGuard device_guard(input.device());
  lightx2v_kernel::check_sm120_or_throw(input, op_name);

  int32_t m = input.size(0);
  int32_t n = input.size(1);

  TORCH_CHECK(scale.scalar_type() == input.scalar_type(), "scale dtype must match input dtype.");
  TORCH_CHECK(shift.scalar_type() == input.scalar_type(), "shift dtype must match input dtype.");
  TORCH_CHECK(scale.get_device() == input.get_device(), "scale must be on the same CUDA device as input.");
  TORCH_CHECK(shift.get_device() == input.get_device(), "shift must be on the same CUDA device as input.");

  bool scale_is_2d = is_mxfp8_modulate_param_2d(scale, m, n, "scale");
  bool shift_is_2d = is_mxfp8_modulate_param_2d(shift, m, n, "shift");

  int multiProcessorCount = lightx2v_kernel::getMultiProcessorCount(input.get_device());

  auto sf_out = static_cast<int32_t*>(output_sf.data_ptr());
  auto output_ptr = static_cast<int64_t*>(output.data_ptr());
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream(input.get_device());

  switch (input.scalar_type()) {
    case torch::kHalf: {
      auto input_ptr = reinterpret_cast<half const*>(input.data_ptr());
      auto scale_ptr = reinterpret_cast<half const*>(scale.data_ptr());
      auto shift_ptr = reinterpret_cast<half const*>(shift.data_ptr());
      invokeModulateFP8Quantization(
          m, n, input_ptr, scale_ptr, shift_ptr, scale_is_2d, shift_is_2d, output_ptr, sf_out, multiProcessorCount, stream);
      break;
    }
    case torch::kBFloat16: {
      auto input_ptr = reinterpret_cast<__nv_bfloat16 const*>(input.data_ptr());
      auto scale_ptr = reinterpret_cast<__nv_bfloat16 const*>(scale.data_ptr());
      auto shift_ptr = reinterpret_cast<__nv_bfloat16 const*>(shift.data_ptr());
      invokeModulateFP8Quantization(
          m, n, input_ptr, scale_ptr, shift_ptr, scale_is_2d, shift_is_2d, output_ptr, sf_out, multiProcessorCount, stream);
      break;
    }
    default: {
      TORCH_CHECK(false, "Unsupported input data type for modulate_quantize_to_fp8: ", input.scalar_type());
    }
  }
}
