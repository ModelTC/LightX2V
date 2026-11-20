// Shared device helpers for the gfx1201 SageAttention kernels (RDNA4, wave32 WMMA).
#pragma once

#include <hip/hip_runtime.h>
#include <math.h>
#include <stdint.h>

#include <type_traits>

typedef int v2i __attribute__((ext_vector_type(2)));
typedef int v4i __attribute__((ext_vector_type(4)));
typedef int v8i __attribute__((ext_vector_type(8)));
typedef float v8f __attribute__((ext_vector_type(8)));
typedef _Float16 h8 __attribute__((ext_vector_type(8)));
typedef _Float16 h2 __attribute__((ext_vector_type(2)));

namespace {

__device__ __forceinline__ uint32_t bf16_rne(float f) {
    const uint32_t u = __float_as_uint(f);
    return (u + 0x7FFFu + ((u >> 16) & 1u)) >> 16;
}

// value held by the lane 16 positions away (the other half of a 16x16 WMMA fragment column)
__device__ __forceinline__ float other_half(float x) {
    return __int_as_float(__builtin_amdgcn_permlanex16(__float_as_int(x), __float_as_int(x), 0x76543210, 0xfedcba98, false, false));
}

// Four P codes -> four fp8 e4m3 bytes. A code c in [0, 126] read as e4m3 bits is ~2^(c/8 - 7): the exp2 of the softmax.
__device__ __forceinline__ uint32_t pack4(float a, float b, float c, float d) {
    uint32_t w = __builtin_amdgcn_cvt_pk_u8_f32(a, 0, 0u);
    w = __builtin_amdgcn_cvt_pk_u8_f32(b, 1, w);
    w = __builtin_amdgcn_cvt_pk_u8_f32(c, 2, w);
    return __builtin_amdgcn_cvt_pk_u8_f32(d, 3, w);
}

__device__ __forceinline__ void bar_signal() {
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup", "local");
    __builtin_amdgcn_s_barrier_signal(-1);
}

__device__ __forceinline__ void bar_wait() {
    __builtin_amdgcn_s_barrier_wait(-1);
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup", "local");
}

}  // namespace
