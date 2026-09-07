#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <torch/all.h>

#include <cstdlib>

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

#ifndef EXPERIMENT_BLOCK_THREADS
#define EXPERIMENT_BLOCK_THREADS 256
#endif
#ifndef EXPERIMENT_LAUNCH_THREADS
#define EXPERIMENT_LAUNCH_THREADS 192
#endif
#ifndef EXPERIMENT_BLOCKS_PER_SM
#define EXPERIMENT_BLOCKS_PER_SM 4
#endif
#ifndef EXPERIMENT_TILE4_LOAD_MODE
#define EXPERIMENT_TILE4_LOAD_MODE 3
#endif
#ifndef EXPERIMENT_TILE4_MAX_REGISTERS
#define EXPERIMENT_TILE4_MAX_REGISTERS 38
#endif
#ifndef EXPERIMENT_TILE4_OUTER_M
#define EXPERIMENT_TILE4_OUTER_M 1
#endif
#ifndef EXPERIMENT_PRECOMPUTE_SF
#define EXPERIMENT_PRECOMPUTE_SF 0
#endif
#ifndef EXPERIMENT_TILE4_PIPELINE
#define EXPERIMENT_TILE4_PIPELINE 0
#endif
#ifndef EXPERIMENT_TILE4_DIRECT_SCALE
#define EXPERIMENT_TILE4_DIRECT_SCALE 0
#endif
#ifndef EXPERIMENT_TILE4_DOUBLE_BUFFER
#define EXPERIMENT_TILE4_DOUBLE_BUFFER 0
#endif
#ifndef EXPERIMENT_SCALE_PACK_MODE
#define EXPERIMENT_SCALE_PACK_MODE 0
#endif
#ifndef EXPERIMENT_INCREMENTAL_TASK
#define EXPERIMENT_INCREMENTAL_TASK 0
#endif

constexpr int TILE4_BLOCK_THREADS = 320;
constexpr int TILE4_MAX_BLOCK_THREADS = 512;
constexpr int TILE4_GRID_BLOCKS_NUMERATOR = 19;
constexpr int TILE4_GRID_BLOCKS_DENOMINATOR = 5;
constexpr int CVT_FP4_ELTS_PER_THREAD = 8;
constexpr int CVT_FP4_SF_VEC_SIZE = 16;

// Convert 4 float2 values into 8 e2m1 values (represented as one uint32_t).
inline __device__ uint32_t fp32_vec_to_e2m1(float2 (&array)[4]) {
  // PTX instructions used here requires sm100a.
// #if CUDA_VERSION >= 12080
// #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000) && __CUDA_ARCH_HAS_FEATURE__(SM100_ALL)
  uint32_t val;
  asm volatile(
      "{\n"
      ".reg .b8 byte0;\n"
      ".reg .b8 byte1;\n"
      ".reg .b8 byte2;\n"
      ".reg .b8 byte3;\n"
      "cvt.rn.satfinite.e2m1x2.f32   byte0, %2, %1;\n"
      "cvt.rn.satfinite.e2m1x2.f32   byte1, %4, %3;\n"
      "cvt.rn.satfinite.e2m1x2.f32   byte2, %6, %5;\n"
      "cvt.rn.satfinite.e2m1x2.f32   byte3, %8, %7;\n"
      "mov.b32 %0, {byte0, byte1, byte2, byte3};\n"
      "}"
      : "=r"(val)
      : "f"(array[0].x),
        "f"(array[0].y),
        "f"(array[1].x),
        "f"(array[1].y),
        "f"(array[2].x),
        "f"(array[2].y),
        "f"(array[3].x),
        "f"(array[3].y));
  return val;
// #else
//   return 0;
// #endif
// #endif
}

// Fast reciprocal.
inline __device__ float reciprocal_approximate_ftz(float a) {
  float b;
  asm volatile("rcp.approx.ftz.f32 %0, %1;\n" : "=f"(b) : "f"(a));
  return b;
}

// Define a 16 bytes packed data type.
template <class Type>
struct alignas(16) PackedVec {
  typename TypeConverter<Type>::Type elts[4];
};

template <>
struct PackedVec<__nv_fp8_e4m3> {
  __nv_fp8x2_e4m3 elts[8];
};

template <class Type>
inline __device__ PackedVec<Type> load_packed_vec(Type const* ptr) {
  uint4 raw;
  asm volatile(
      "ld.global.v4.u32 {%0, %1, %2, %3}, [%4];"
      : "=r"(raw.x), "=r"(raw.y), "=r"(raw.z), "=r"(raw.w)
      : "l"(ptr));
  union {
    uint4 raw;
    PackedVec<Type> vec;
  } value;
  value.raw = raw;
  return value.vec;
}

template <class Type>
inline __device__ PackedVec<Type> load_packed_vec_v2u64(Type const* ptr) {
  unsigned long long lo;
  unsigned long long hi;
  asm volatile(
      "ld.global.v2.u64 {%0, %1}, [%2];"
      : "=l"(lo), "=l"(hi)
      : "l"(ptr));
  union {
    struct {
      unsigned long long lo;
      unsigned long long hi;
    } raw;
    PackedVec<Type> vec;
  } value;
  value.raw.lo = lo;
  value.raw.hi = hi;
  return value.vec;
}

template <class Type>
inline __device__ PackedVec<Type> load_packed_vec_streaming(Type const* ptr) {
  uint4 raw;
  asm volatile(
      "ld.global.cs.v4.u32 {%0, %1, %2, %3}, [%4];"
      : "=r"(raw.x), "=r"(raw.y), "=r"(raw.z), "=r"(raw.w)
      : "l"(ptr));
  union {
    uint4 raw;
    PackedVec<Type> vec;
  } value;
  value.raw = raw;
  return value.vec;
}

template <class Type>
inline __device__ PackedVec<Type> load_packed_vec_global(Type const* ptr) {
  uint4 raw;
  asm volatile(
      "ld.global.cg.v4.u32 {%0, %1, %2, %3}, [%4];"
      : "=r"(raw.x), "=r"(raw.y), "=r"(raw.z), "=r"(raw.w)
      : "l"(ptr));
  union {
    uint4 raw;
    PackedVec<Type> vec;
  } value;
  value.raw = raw;
  return value.vec;
}

template <class Type>
inline __device__ PackedVec<Type> load_tile4_vec(Type const* ptr) {
#if EXPERIMENT_TILE4_LOAD_MODE == 1
  return load_packed_vec(ptr);
#elif EXPERIMENT_TILE4_LOAD_MODE == 2
  return load_packed_vec_v2u64(ptr);
#elif EXPERIMENT_TILE4_LOAD_MODE == 3
  return load_packed_vec_streaming(ptr);
#elif EXPERIMENT_TILE4_LOAD_MODE == 4
  return load_packed_vec_global(ptr);
#else
  return *reinterpret_cast<PackedVec<Type> const*>(ptr);
#endif
}

inline __device__ uint32_t pack_scale_byte(uint8_t sfValue) {
  uint32_t sf = uint32_t(sfValue);
#if EXPERIMENT_SCALE_PACK_MODE == 1
  uint32_t pair = 0;
  uint32_t packed = 0;
  if ((threadIdx.x & 1) == 0) {
    uint32_t next = __shfl_down_sync(0x55555555, sf, 2, 8);
    pair = sf | (next << 8);
  }
  if ((threadIdx.x & 3) == 0) {
    uint32_t nextPair = __shfl_down_sync(0x11111111, pair, 4, 8);
    packed = pair | (nextPair << 16);
  }
  return packed;
#elif EXPERIMENT_SCALE_PACK_MODE == 2
  uint32_t pair = sf | (__shfl_down_sync(0xffffffff, sf, 2, 8) << 8);
  return pair | (__shfl_down_sync(0xffffffff, pair, 4, 8) << 16);
#else
  int32_t groupLane = (threadIdx.x & 31) & ~7;
  uint32_t sf1 = __shfl_sync(0xffffffff, sf, groupLane + 2);
  uint32_t sf2 = __shfl_sync(0xffffffff, sf, groupLane + 4);
  uint32_t sf3 = __shfl_sync(0xffffffff, sf, groupLane + 6);
  return sf | (sf1 << 8) | (sf2 << 16) | (sf3 << 24);
#endif
}

inline __device__ void stage_scale_byte(uint8_t sfValue, int32_t colIdx, uint32_t* sfStageRow) {
  uint32_t packedSF = pack_scale_byte(sfValue);
  if ((threadIdx.x & 7) == 0) {
    sfStageRow[colIdx / 8] = packedSF;
  }
}

// Quantizes the provided PackedVec into the uint32_t output
template <class Type, bool UE8M0_SF = false>
__device__ uint32_t cvt_warp_fp16_to_fp4(
    PackedVec<Type>& vec,
    float SFScaleVal,
#if EXPERIMENT_PRECOMPUTE_SF
    float SFScaleForMax,
#endif
    uint8_t* SFout,
    uint8_t* SFValueOut = nullptr) {
// #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  // Get absolute maximum values among the local 8 values.
  auto localMax = __habs2(vec.elts[0]);

// Local maximum value.
#pragma unroll
  for (int i = 1; i < CVT_FP4_ELTS_PER_THREAD / 2; i++) {
    localMax = __hmax2(localMax, __habs2(vec.elts[i]));
  }

  // Get the absolute maximum among all 16 values (two threads).
  localMax = __hmax2(__shfl_xor_sync(uint32_t(-1), localMax, 1), localMax);
  // Get the final absolute maximum values.
  float vecMax = float(__hmax(localMax.x, localMax.y));

#if EXPERIMENT_PRECOMPUTE_SF
  float SFValue = vecMax * SFScaleForMax;
#else
  float SFValue = SFScaleVal * (vecMax * 0.16666666666666666f);
#endif
  uint8_t fp8SFVal;
  if constexpr (UE8M0_SF) {
    __nv_fp8_e8m0 tmp;
    tmp.__x = __nv_cvt_float_to_e8m0(SFValue, __NV_SATFINITE, cudaRoundPosInf);
    SFValue = static_cast<float>(tmp);
    fp8SFVal = tmp.__x;
  } else {
    __nv_fp8_e4m3 tmp = __nv_fp8_e4m3(SFValue);
    fp8SFVal = tmp.__x;
    SFValue = static_cast<float>(tmp);
  }
  float outputScale = SFValue != 0 ? SFScaleVal * reciprocal_approximate_ftz(SFValue) : 0.0f;

  if (SFout) {
    // Write the SF to global memory (STG.8).
    *SFout = fp8SFVal;
  }
  if (SFValueOut) {
    *SFValueOut = fp8SFVal;
  }

  // Convert the input to float.
  float2 fp2Vals[CVT_FP4_ELTS_PER_THREAD / 2];

#pragma unroll
  for (int i = 0; i < CVT_FP4_ELTS_PER_THREAD / 2; i++) {
    if constexpr (std::is_same_v<Type, half>) {
      fp2Vals[i] = __half22float2(vec.elts[i]);
    } else {
      fp2Vals[i] = __bfloat1622float2(vec.elts[i]);
    }
    fp2Vals[i].x *= outputScale;
    fp2Vals[i].y *= outputScale;
  }

  // Convert to e2m1 values.
  uint32_t e2m1Vec = fp32_vec_to_e2m1(fp2Vals);

  // Write the e2m1 values to global memory.
  return e2m1Vec;
// #else
//   return 0;
// #endif
}

template <class Type, bool UE8M0_SF = false>
__global__ __launch_bounds__(EXPERIMENT_BLOCK_THREADS, EXPERIMENT_BLOCKS_PER_SM)
void cvt_fp16_to_fp4_incremental_sf(
    int32_t numRows, int32_t numCols, Type const* in, float const* SFScale, uint32_t* out, uint32_t* SFout) {
  using InputVec = PackedVec<Type>;
  float const SFScaleVal = SFScale == nullptr ? 1.0f : SFScale[0];
#if EXPERIMENT_PRECOMPUTE_SF
  float const SFScaleForMax = SFScaleVal * 0.16666666666666666f;
#endif
  int32_t numColVecs = numCols / CVT_FP4_ELTS_PER_THREAD;
  int32_t numKTiles = (numCols + CVT_FP4_SF_VEC_SIZE * 4 - 1) / (CVT_FP4_SF_VEC_SIZE * 4);

  for (int32_t rowIdx = blockIdx.x; rowIdx < numRows; rowIdx += gridDim.x) {
    int64_t rowVecBase = int64_t(rowIdx) * numColVecs;
    int64_t sfRowBase = int64_t(rowIdx / 128) * numKTiles * 512 +
                        int64_t(rowIdx % 32) * 16 + int64_t((rowIdx % 128) / 32) * 4;
    int32_t colIdx = threadIdx.x;
    int64_t sfOffset = sfRowBase + int64_t(threadIdx.x / 8) * 512 + ((threadIdx.x / 2) & 3);
    int32_t colStride = blockDim.x;
    int64_t sfStride = int64_t(blockDim.x / 8) * 512;
    for (; colIdx < numColVecs; colIdx += colStride, sfOffset += sfStride) {
      int64_t vecOffset = rowVecBase + colIdx;
      InputVec inVec = reinterpret_cast<InputVec const*>(in)[vecOffset];
      uint8_t sfValue;
      out[vecOffset] = cvt_warp_fp16_to_fp4<Type, UE8M0_SF>(
          inVec,
          SFScaleVal,
#if EXPERIMENT_PRECOMPUTE_SF
          SFScaleForMax,
#endif
          nullptr,
          &sfValue);

      uint32_t sf = uint32_t(sfValue);
      int32_t groupLane = threadIdx.x & ~7;
      uint32_t sf1 = __shfl_sync(0xffffffff, sf, groupLane + 2);
      uint32_t sf2 = __shfl_sync(0xffffffff, sf, groupLane + 4);
      uint32_t sf3 = __shfl_sync(0xffffffff, sf, groupLane + 6);
      if ((threadIdx.x & 7) == 0) {
        uint32_t packedSF = sf | (sf1 << 8) | (sf2 << 16) | (sf3 << 24);
        *reinterpret_cast<uint32_t*>(reinterpret_cast<uint8_t*>(SFout) + sfOffset) = packedSF;
      }
    }
  }
}

template <class Type, bool UE8M0_SF = false>
__global__ __maxnreg__(EXPERIMENT_TILE4_MAX_REGISTERS)
void cvt_fp16_to_fp4_tile4_sf(
    int32_t numRows, Type const* in, float const* SFScale, uint32_t* out, uint32_t* SFout) {
  using InputVec = PackedVec<Type>;
  constexpr int32_t numColVecs = 5120 / CVT_FP4_ELTS_PER_THREAD;
  constexpr int32_t numKTiles = 5120 / (CVT_FP4_SF_VEC_SIZE * 4);
  constexpr int32_t outerMPerTask = EXPERIMENT_TILE4_OUTER_M;
  constexpr int32_t scaleBuffers = EXPERIMENT_TILE4_DOUBLE_BUFFER ? 2 : 1;
  static_assert(outerMPerTask == 1 || outerMPerTask == 2 || outerMPerTask == 4 || outerMPerTask == 8);
  __shared__ uint32_t sfStageStorage[scaleBuffers][outerMPerTask][4][numKTiles];
  float const SFScaleVal = SFScale == nullptr ? 1.0f : SFScale[0];
#if EXPERIMENT_PRECOMPUTE_SF
  float const SFScaleForMax = SFScaleVal * 0.16666666666666666f;
#endif
  int32_t numMTiles = (numRows + 127) / 128;
  constexpr int32_t tasksPerMTile = 32 / outerMPerTask;
  int32_t numTasks = numMTiles * tasksPerMTile;
#if EXPERIMENT_INCREMENTAL_TASK
  int32_t taskInMTile = blockIdx.x % tasksPerMTile;
  int32_t mTileIdx = blockIdx.x / tasksPerMTile;
  int32_t gridMTileStride = gridDim.x / tasksPerMTile;
  int32_t gridTaskRemainder = gridDim.x % tasksPerMTile;
#endif

  for (int32_t taskIdx = blockIdx.x, taskIteration = 0; taskIdx < numTasks;
       taskIdx += gridDim.x, ++taskIteration) {
#if EXPERIMENT_INCREMENTAL_TASK
    int32_t outerMBase = taskInMTile * outerMPerTask;
#else
    int32_t mTileIdx = taskIdx / tasksPerMTile;
    int32_t outerMBase = (taskIdx % tasksPerMTile) * outerMPerTask;
#endif
    uint32_t (*sfStage)[4][numKTiles] = sfStageStorage[taskIteration % scaleBuffers];
#if EXPERIMENT_TILE4_DIRECT_SCALE
    static_assert(outerMPerTask == 1);
    int64_t sfTileBase = int64_t(mTileIdx) * numKTiles * 512 + int64_t(outerMBase) * 16;
    for (int32_t colIdx = threadIdx.x; colIdx < numColVecs; colIdx += blockDim.x) {
      uint32_t packedSF[4];
#pragma unroll
      for (int32_t innerM = 0; innerM < 4; ++innerM) {
        int32_t rowIdx = mTileIdx * 128 + innerM * 32 + outerMBase;
        if (rowIdx < numRows) {
          int64_t vecOffset = int64_t(rowIdx) * numColVecs + colIdx;
          InputVec inVec = load_tile4_vec(in + vecOffset * CVT_FP4_ELTS_PER_THREAD);
          uint8_t sfValue;
          out[vecOffset] = cvt_warp_fp16_to_fp4<Type, UE8M0_SF>(
              inVec,
              SFScaleVal,
#if EXPERIMENT_PRECOMPUTE_SF
              SFScaleForMax,
#endif
              nullptr,
              &sfValue);
          packedSF[innerM] = pack_scale_byte(sfValue);
        } else {
          packedSF[innerM] = 0;
        }
      }
      if ((threadIdx.x & 7) == 0) {
        uint4 packed = make_uint4(packedSF[0], packedSF[1], packedSF[2], packedSF[3]);
        *reinterpret_cast<uint4*>(reinterpret_cast<uint8_t*>(SFout) + sfTileBase +
                                  int64_t(colIdx / 8) * 512) = packed;
      }
    }
#else
#if EXPERIMENT_TILE4_PIPELINE == 2
    static_assert(outerMPerTask == 2);
#pragma unroll
    for (int32_t innerM = 0; innerM < 4; ++innerM) {
      int32_t rowIdx0 = mTileIdx * 128 + innerM * 32 + outerMBase;
      int32_t rowIdx1 = rowIdx0 + 1;
      bool valid0 = rowIdx0 < numRows;
      bool valid1 = rowIdx1 < numRows;
      if (valid0 && valid1) {
        int64_t rowVecBase0 = int64_t(rowIdx0) * numColVecs;
        int64_t rowVecBase1 = int64_t(rowIdx1) * numColVecs;
        for (int32_t colIdx = threadIdx.x; colIdx < numColVecs; colIdx += blockDim.x) {
          int64_t vecOffset0 = rowVecBase0 + colIdx;
          int64_t vecOffset1 = rowVecBase1 + colIdx;
          InputVec inVec0 = load_tile4_vec(in + vecOffset0 * CVT_FP4_ELTS_PER_THREAD);
          InputVec inVec1 = load_tile4_vec(in + vecOffset1 * CVT_FP4_ELTS_PER_THREAD);
          uint8_t sfValue0;
          uint8_t sfValue1;
          out[vecOffset0] = cvt_warp_fp16_to_fp4<Type, UE8M0_SF>(
              inVec0,
              SFScaleVal,
#if EXPERIMENT_PRECOMPUTE_SF
              SFScaleForMax,
#endif
              nullptr,
              &sfValue0);
          out[vecOffset1] = cvt_warp_fp16_to_fp4<Type, UE8M0_SF>(
              inVec1,
              SFScaleVal,
#if EXPERIMENT_PRECOMPUTE_SF
              SFScaleForMax,
#endif
              nullptr,
              &sfValue1);
          stage_scale_byte(sfValue0, colIdx, sfStage[0][innerM]);
          stage_scale_byte(sfValue1, colIdx, sfStage[1][innerM]);
        }
      } else {
#pragma unroll
        for (int32_t outerMOffset = 0; outerMOffset < 2; ++outerMOffset) {
          int32_t rowIdx = rowIdx0 + outerMOffset;
          if (rowIdx < numRows) {
            int64_t rowVecBase = int64_t(rowIdx) * numColVecs;
            for (int32_t colIdx = threadIdx.x; colIdx < numColVecs; colIdx += blockDim.x) {
              int64_t vecOffset = rowVecBase + colIdx;
              InputVec inVec = load_tile4_vec(in + vecOffset * CVT_FP4_ELTS_PER_THREAD);
              uint8_t sfValue;
              out[vecOffset] = cvt_warp_fp16_to_fp4<Type, UE8M0_SF>(
                  inVec,
                  SFScaleVal,
#if EXPERIMENT_PRECOMPUTE_SF
                  SFScaleForMax,
#endif
                  nullptr,
                  &sfValue);
              stage_scale_byte(sfValue, colIdx, sfStage[outerMOffset][innerM]);
            }
          } else {
            for (int32_t kTile = threadIdx.x; kTile < numKTiles; kTile += blockDim.x) {
              sfStage[outerMOffset][innerM][kTile] = 0;
            }
          }
        }
      }
    }
#else
#pragma unroll
    for (int32_t outerMOffset = 0; outerMOffset < outerMPerTask; ++outerMOffset) {
      int32_t outerM = outerMBase + outerMOffset;
#pragma unroll
      for (int32_t innerM = 0; innerM < 4; ++innerM) {
        int32_t rowIdx = mTileIdx * 128 + innerM * 32 + outerM;
        if (rowIdx < numRows) {
          int64_t rowVecBase = int64_t(rowIdx) * numColVecs;
          for (int32_t colIdx = threadIdx.x; colIdx < numColVecs;
#if EXPERIMENT_TILE4_PIPELINE == 1
               colIdx += blockDim.x * 2
#else
               colIdx += blockDim.x
#endif
          ) {
            int64_t vecOffset = rowVecBase + colIdx;
            InputVec inVec = load_tile4_vec(in + vecOffset * CVT_FP4_ELTS_PER_THREAD);
#if EXPERIMENT_TILE4_PIPELINE == 1
            int32_t colIdx1 = colIdx + blockDim.x;
            int64_t vecOffset1 = rowVecBase + colIdx1;
            InputVec inVec1;
            if (colIdx1 < numColVecs) {
              inVec1 = load_tile4_vec(in + vecOffset1 * CVT_FP4_ELTS_PER_THREAD);
            }
#endif
            uint8_t sfValue;
            out[vecOffset] = cvt_warp_fp16_to_fp4<Type, UE8M0_SF>(
                inVec,
                SFScaleVal,
#if EXPERIMENT_PRECOMPUTE_SF
                SFScaleForMax,
#endif
                nullptr,
                &sfValue);

            stage_scale_byte(sfValue, colIdx, sfStage[outerMOffset][innerM]);
#if EXPERIMENT_TILE4_PIPELINE == 1
            if (colIdx1 < numColVecs) {
              uint8_t sfValue1;
              out[vecOffset1] = cvt_warp_fp16_to_fp4<Type, UE8M0_SF>(
                  inVec1,
                  SFScaleVal,
#if EXPERIMENT_PRECOMPUTE_SF
                  SFScaleForMax,
#endif
                  nullptr,
                  &sfValue1);
              stage_scale_byte(sfValue1, colIdx1, sfStage[outerMOffset][innerM]);
            }
#endif
          }
        } else {
          for (int32_t kTile = threadIdx.x; kTile < numKTiles; kTile += blockDim.x) {
            sfStage[outerMOffset][innerM][kTile] = 0;
          }
        }
      }
    }
#endif
    __syncthreads();

    int64_t sfTileBase = int64_t(mTileIdx) * numKTiles * 512 + int64_t(outerMBase) * 16;
    for (int32_t storeIdx = threadIdx.x; storeIdx < numKTiles * outerMPerTask;
         storeIdx += blockDim.x) {
      int32_t kTile = storeIdx / outerMPerTask;
      int32_t outerMOffset = storeIdx % outerMPerTask;
      uint4 packed = make_uint4(
          sfStage[outerMOffset][0][kTile],
          sfStage[outerMOffset][1][kTile],
          sfStage[outerMOffset][2][kTile],
          sfStage[outerMOffset][3][kTile]);
      *reinterpret_cast<uint4*>(reinterpret_cast<uint8_t*>(SFout) + sfTileBase +
                                int64_t(kTile) * 512 + outerMOffset * 16) = packed;
    }
#if !EXPERIMENT_TILE4_DOUBLE_BUFFER
    __syncthreads();
#endif
#endif
#if EXPERIMENT_INCREMENTAL_TASK
    mTileIdx += gridMTileStride;
    taskInMTile += gridTaskRemainder;
    if (taskInMTile >= tasksPerMTile) {
      taskInMTile -= tasksPerMTile;
      ++mTileIdx;
    }
#endif
  }
}

template <typename T>
void invokeFP4Quantization(
    int m,
    int n,
    T const* input,
    float const* SFScale,
    int64_t* output,
    int32_t* SFOuput,
    bool useUE8M0,
    int multiProcessorCount,
    cudaStream_t stream) {
  // Grid, Block size.
  // Each thread converts 8 values.
  int blockThreads = EXPERIMENT_LAUNCH_THREADS;
  if (char const* value = std::getenv("FP4_QUANT_BLOCK_THREADS")) {
    blockThreads = std::atoi(value);
  }
  int gridBlocks = multiProcessorCount * EXPERIMENT_BLOCKS_PER_SM;
  if (char const* value = std::getenv("FP4_QUANT_GRID_BLOCKS")) {
    gridBlocks = std::atoi(value);
  }
  TORCH_CHECK(gridBlocks > 0);
  if (n == 5120) {
    if (std::getenv("FP4_QUANT_BLOCK_THREADS") == nullptr) {
      blockThreads = TILE4_BLOCK_THREADS;
    }
    if (std::getenv("FP4_QUANT_GRID_BLOCKS") == nullptr) {
      gridBlocks =
          multiProcessorCount * TILE4_GRID_BLOCKS_NUMERATOR / TILE4_GRID_BLOCKS_DENOMINATOR;
    }
    TORCH_CHECK(blockThreads > 0 && blockThreads <= TILE4_MAX_BLOCK_THREADS && blockThreads % 32 == 0);
    dim3 block(blockThreads);
    dim3 grid(std::min(int(m), gridBlocks));
    if (useUE8M0) {
      cvt_fp16_to_fp4_tile4_sf<T, true><<<grid, block, 0, stream>>>(
          m, input, SFScale, reinterpret_cast<uint32_t*>(output), reinterpret_cast<uint32_t*>(SFOuput));
    } else {
      cvt_fp16_to_fp4_tile4_sf<T, false><<<grid, block, 0, stream>>>(
          m, input, SFScale, reinterpret_cast<uint32_t*>(output), reinterpret_cast<uint32_t*>(SFOuput));
    }
    return;
  }

  TORCH_CHECK(blockThreads > 0 && blockThreads <= EXPERIMENT_BLOCK_THREADS && blockThreads % 32 == 0);
  dim3 block(std::min(int(n / ELTS_PER_THREAD), blockThreads));
  dim3 grid(std::min(int(m), gridBlocks));

  // Launch the generic conversion kernel.
  if (useUE8M0) {
    cvt_fp16_to_fp4_incremental_sf<T, true><<<grid, block, 0, stream>>>(
        m, n, input, SFScale, reinterpret_cast<uint32_t*>(output), reinterpret_cast<uint32_t*>(SFOuput));
  } else {
    cvt_fp16_to_fp4_incremental_sf<T, false><<<grid, block, 0, stream>>>(
        m, n, input, SFScale, reinterpret_cast<uint32_t*>(output), reinterpret_cast<uint32_t*>(SFOuput));
  }
}

// Instantiate the function.
template void invokeFP4Quantization(
    int m,
    int n,
    half const* input,
    float const* SFScale,
    int64_t* output,
    int32_t* SFOuput,
    bool useUE8M0,
    int multiProcessorCount,
    cudaStream_t stream);

template void invokeFP4Quantization(
    int m,
    int n,
    __nv_bfloat16 const* input,
    float const* SFScale,
    int64_t* output,
    int32_t* SFOuput,
    bool useUE8M0,
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

void scaled_nvfp4_quant_sm120(
    torch::Tensor& output, torch::Tensor const& input, torch::Tensor& output_sf, torch::Tensor const& input_sf) {
  int32_t m = input.size(0);
  int32_t n = input.size(1);

  TORCH_CHECK(n % 16 == 0, "The N dimension must be multiple of 16.");

  int multiProcessorCount = getMultiProcessorCount();

  auto input_sf_ptr = static_cast<float const*>(input_sf.data_ptr());
  auto sf_out = static_cast<int32_t*>(output_sf.data_ptr());
  auto output_ptr = static_cast<int64_t*>(output.data_ptr());
  at::cuda::CUDAGuard device_guard{input.get_device()};
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream(input.get_device());

  // We don't support e8m0 scales at this moment.
  bool useUE8M0 = false;

  switch (input.scalar_type()) {
    case torch::kHalf: {
      auto input_ptr = reinterpret_cast<half const*>(input.data_ptr());
      invokeFP4Quantization(m, n, input_ptr, input_sf_ptr, output_ptr, sf_out, useUE8M0, multiProcessorCount, stream);
      break;
    }
    case torch::kBFloat16: {
      auto input_ptr = reinterpret_cast<__nv_bfloat16 const*>(input.data_ptr());
      invokeFP4Quantization(m, n, input_ptr, input_sf_ptr, output_ptr, sf_out, useUE8M0, multiProcessorCount, stream);
      break;
    }
    default: {
      std::cerr << "Observing: " << input.scalar_type() << " for the input datatype which is invalid";
      throw std::runtime_error("Unsupported input data type for quantize_to_fp4.");
    }
  }
}
