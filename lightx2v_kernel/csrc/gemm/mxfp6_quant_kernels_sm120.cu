#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda.h>
#include <cuda_fp8.h>
#include <cuda_fp6.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <torch/all.h>

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

constexpr int CVT_FP6_ELTS_PER_THREAD = 8;
constexpr int CVT_FP6_SF_VEC_SIZE = 32;

struct uint8x6_t {
  uint8_t elts[6];
};

// Convert 4 float2 values into 8 e3m2 values (represented as one uint8x6_t).
inline __device__ uint8x6_t fp32_vec_to_e3m2(float2 (&array)[4]) {
  uint64_t val;
  asm volatile(
      "{\n"
      ".reg .b16 pack0;\n"
      ".reg .b16 pack1;\n"
      ".reg .b16 pack2;\n"
      ".reg .b16 pack3;\n"
      "cvt.rn.satfinite.e3m2x2.f32   pack0, %2, %1;\n"
      "cvt.rn.satfinite.e3m2x2.f32   pack1, %4, %3;\n"
      "cvt.rn.satfinite.e3m2x2.f32   pack2, %6, %5;\n"
      "cvt.rn.satfinite.e3m2x2.f32   pack3, %8, %7;\n"
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

  uint8x6_t result;

  // pack 8 uint8_t into 6 uint8_t
  // here is how to pack:
  // 4个fp6 a b c d. a:[a5 a4 a3 a2 a1 a0], b..., c..., d...
  // 3个unint8 pack0 pack1 pack2
  // packed0: [b1 b0][a5 a4 a3 a2 a1 a0]
  // packed1: [c3 c2 c1 c0][b5 b4 b3 b2]
  // packed2: [d5 d4 d3 d2 d1 d0][c5 c4]

  // lower 4 uint8_t
  uint8_t l_val_0 = val & 0xFF;
  uint8_t l_val_1 = (val >> 8) & 0xFF;
  uint8_t l_val_2 = (val >> 16) & 0xFF;
  uint8_t l_val_3 = (val >> 24) & 0xFF;
  // higher 4 uint8_t
  uint8_t h_val_0 = (val >> 32) & 0xFF;
  uint8_t h_val_1 = (val >> 40) & 0xFF;
  uint8_t h_val_2 = (val >> 48) & 0xFF;
  uint8_t h_val_3 = (val >> 56) & 0xFF;

  // pack result
  result.elts[0] = (l_val_1 << 6) | l_val_0;
  result.elts[1] = (l_val_2 << 4) | (l_val_1 >> 2);
  result.elts[2] = (l_val_3 << 2) | (l_val_2 >> 4);
  result.elts[3] = (h_val_1 << 6) | h_val_0;
  result.elts[4] = (h_val_2 << 4) | (h_val_1 >> 2);
  result.elts[5] = (h_val_3 << 2) | (h_val_2 >> 4);

  return result;
}

// Fast reciprocal.
inline __device__ float reciprocal_approximate_ftz(float a) {
  float b;
  asm volatile("rcp.approx.ftz.f32 %0, %1;\n" : "=f"(b) : "f"(a));
  return b;
}

template <class SFType, int CVT_FP6_NUM_THREADS_PER_SF>
__device__ uint8_t* get_sf_out_address(int rowIdx, int colIdx, int numCols, SFType* SFout) {
// #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  static_assert(CVT_FP6_NUM_THREADS_PER_SF == 4);

  // one of 4 threads write one SF to global memory.
  // TODO: stage through smem for packed STG.32
  // is it better than STG.8 from 4 threads ?
  if (threadIdx.x % CVT_FP6_NUM_THREADS_PER_SF == 0) {
    // SF vector index (32 elements share one SF in the K dimension).
    int32_t kIdx = colIdx / CVT_FP6_NUM_THREADS_PER_SF;
    int32_t mIdx = rowIdx;

    // SF layout [numMTiles, numKTiles, 32 (mTile), 4 (mTile), 4(kTile)]
    // --> index [mTileIdx, kTileIdx, outerMIdx, innerMIdx, innerKIdx]

    int32_t mTileIdx = mIdx / (32 * 4);
    // SF vector size 32.
    int factor = CVT_FP6_SF_VEC_SIZE * 4;
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

// template <>
// struct PackedVec<__nv_fp8_e4m3> {
//   __nv_fp8x2_e4m3 elts[8];
// };

template <class Type> // Type can be half or bfloat16
__device__ uint8x6_t cvt_warp_fp16_to_fp6(PackedVec<Type>& vec, uint8_t* SFout) {
// #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  // Get absolute maximum values among the local 8 values.
  auto localMax = __habs2(vec.elts[0]);

// Local maximum value.
#pragma unroll
  for (int i = 1; i < CVT_FP6_ELTS_PER_THREAD / 2; i++) {
    localMax = __hmax2(localMax, __habs2(vec.elts[i]));
  }

  // Get the absolute maximum among all 32 values (four threads).
  localMax = __hmax2(__shfl_xor_sync(uint32_t(-1), localMax, 1), localMax);
  localMax = __hmax2(__shfl_xor_sync(uint32_t(-1), localMax, 2), localMax);
  // Get the final absolute maximum values.
  float vecMax = float(__hmax(localMax.x, localMax.y));

  // Get the SF (max value of the vector / max value of e3m2).
  // maximum value of e3m2 = 28.0.
  // TODO: use half as compute data type.
  float SFValue = (vecMax / 28.0f);
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
  float2 fp2Vals[CVT_FP6_ELTS_PER_THREAD / 2];

#pragma unroll
  for (int i = 0; i < CVT_FP6_ELTS_PER_THREAD / 2; i++) {
    if constexpr (std::is_same_v<Type, half>) {
      fp2Vals[i] = __half22float2(vec.elts[i]);
    } else {
      fp2Vals[i] = __bfloat1622float2(vec.elts[i]);
    }
    fp2Vals[i].x *= outputScale;
    fp2Vals[i].y *= outputScale;
  }

  // Convert to e3m2 values.
  uint8x6_t e3m2Vec = fp32_vec_to_e3m2(fp2Vals);

  return e3m2Vec;
}


template <class Type> // Type can be half or bfloat16
__global__ void
// #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
__launch_bounds__(256, 6) cvt_fp16_to_fp6(
// #else
// cvt_fp16_to_fp6(
// #endif
    int32_t numRows, int32_t numCols, Type const* in, uint8x6_t* out, uint32_t* SFout) {
// #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  using PackedVec = PackedVec<Type>;
  static constexpr int CVT_FP6_NUM_THREADS_PER_SF = (CVT_FP6_SF_VEC_SIZE / CVT_FP6_ELTS_PER_THREAD);
  static_assert(sizeof(PackedVec) == sizeof(Type) * CVT_FP6_ELTS_PER_THREAD, "Vec size is not matched.");

  // Input tensor row/col loops.
  for (int rowIdx = blockIdx.x; rowIdx < numRows; rowIdx += gridDim.x) {
    for (int colIdx = threadIdx.x; colIdx < numCols / CVT_FP6_ELTS_PER_THREAD; colIdx += blockDim.x) {
      int64_t inOffset = rowIdx * (numCols / CVT_FP6_ELTS_PER_THREAD) + colIdx;
      PackedVec in_vec = reinterpret_cast<PackedVec const*>(in)[inOffset];
      // Get the output tensor offset.
      // Same as inOffset because 8 elements(E3M2) are packed into one uint8x6_t.
      int64_t outOffset = inOffset;
      auto& out_pos = out[outOffset];

      auto sf_out =
          get_sf_out_address<uint32_t, CVT_FP6_NUM_THREADS_PER_SF>(rowIdx, colIdx, numCols, SFout);

      out_pos = cvt_warp_fp16_to_fp6<Type>(in_vec, sf_out);
    }
  }
// #endif
}

template <typename T>
void invokeFP6Quantization(
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
    cvt_fp16_to_fp6<T>
    <<<grid, block, 0, stream>>>(
        m, n, input, reinterpret_cast<uint8x6_t*>(output), reinterpret_cast<uint32_t*>(SFOuput));
}

// Instantiate the function.
template void invokeFP6Quantization(
    int m,
    int n,
    half const* input,
    int64_t* output,
    int32_t* SFOuput,
    int multiProcessorCount,
    cudaStream_t stream);

template void invokeFP6Quantization(
    int m,
    int n,
    __nv_bfloat16 const* input,
    int64_t* output,
    int32_t* SFOuput,
    int multiProcessorCount,
    cudaStream_t stream);

inline int getMultiProcessorCount() {
  static int multi_processor_count = []() {
    int device_id = 0;
    int count = 0;

    // Get the current CUDA device ID
    CHECK_CUDA_SUCCESS(cudaGetDevice(&device_id));

    // Get the number of multiprocessors for the current device
    CHECK_CUDA_SUCCESS(cudaDeviceGetAttribute(&count, cudaDevAttrMultiProcessorCount, device_id));

    return count;  // Initialize the static variable
  }();

  return multi_processor_count;  // Return the cached value on subsequent calls
}

void scaled_mxfp6_quant_sm120(
    torch::Tensor& output, torch::Tensor const& input, torch::Tensor& output_sf) {
  int32_t m = input.size(0);
  int32_t n = input.size(1);

  TORCH_CHECK(n % 32 == 0, "The N dimension must be multiple of 32.");

  int multiProcessorCount = getMultiProcessorCount();

  auto sf_out = static_cast<int32_t*>(output_sf.data_ptr());
  auto output_ptr = static_cast<int64_t*>(output.data_ptr());
  at::cuda::CUDAGuard device_guard{(char)input.get_device()};
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream(input.get_device());

  switch (input.scalar_type()) {
    case torch::kHalf: {
      auto input_ptr = reinterpret_cast<half const*>(input.data_ptr());
      invokeFP6Quantization(m, n, input_ptr, output_ptr, sf_out, multiProcessorCount, stream);
      break;
    }
    case torch::kBFloat16: {
      auto input_ptr = reinterpret_cast<__nv_bfloat16 const*>(input.data_ptr());
      invokeFP6Quantization(m, n, input_ptr, output_ptr, sf_out, multiProcessorCount, stream);
      break;
    }
    default: {
      std::cerr << "Observing: " << input.scalar_type() << " for the input datatype which is invalid";
      throw std::runtime_error("Unsupported input data type for quantize_to_fp6.");
    }
  }
}
