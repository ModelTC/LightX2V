#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <cutlass/numeric_types.h>

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>

#include <tuple>

#define CHECK_DEVICE(x) TORCH_CHECK(x.is_cuda(), #x " must be on CUDA")

namespace {

using namespace nvcuda;

constexpr int kWarpSize = 32;
constexpr int kWarpCount = 4;
constexpr int kThreadsPerBlock = kWarpSize * kWarpCount;
constexpr int kHadamardHeadDim = 128;
constexpr int kHadamardRows = 8;
constexpr int kHadamardCols = 16;
constexpr int kWmmaTile = 16;

template <typename T>
struct WmmaInputType;

template <>
struct WmmaInputType<cutlass::half_t> {
    using type = half;
};

template <>
struct WmmaInputType<cutlass::bfloat16_t> {
    using type = __nv_bfloat16;
};

template <typename T>
struct NativeInputType;

template <>
struct NativeInputType<cutlass::half_t> {
    using type = half;
};

template <>
struct NativeInputType<cutlass::bfloat16_t> {
    using type = __nv_bfloat16;
};

template <typename T>
__device__ __forceinline__ float to_float_device(T v) {
    return static_cast<float>(v);
}

template <typename T>
__device__ __forceinline__ T from_float_device(float v) {
    return T(v);
}

template <typename T>
__device__ __forceinline__ typename NativeInputType<T>::type from_float_native(float v);

template <>
__device__ __forceinline__ half from_float_native<cutlass::half_t>(float v) {
    return __float2half_rn(v);
}

template <>
__device__ __forceinline__ __nv_bfloat16 from_float_native<cutlass::bfloat16_t>(float v) {
    return __float2bfloat16(v);
}

template <typename scalar_t>
__global__ void preprocess_k_cache_kernel_generic(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    float* __restrict__ norm_output,
    int total_tokens,
    int num_heads,
    int head_dim,
    bool apply_hadamard,
    bool apply_norm
) {
    extern __shared__ float smem[];

    const int token_idx = blockIdx.x;
    const int tid = threadIdx.x;

    if (token_idx >= total_tokens) {
        return;
    }

    const int token_stride = num_heads * head_dim;
    const scalar_t* token_input = input + static_cast<int64_t>(token_idx) * token_stride;
    scalar_t* token_output = output + static_cast<int64_t>(token_idx) * token_stride;

    float local_sum = 0.0f;
    const float hadamard_scale = rsqrtf(static_cast<float>(head_dim));

    for (int head_idx = 0; head_idx < num_heads; ++head_idx) {
        if (tid < head_dim) {
            smem[tid] = to_float_device(token_input[head_idx * head_dim + tid]);
        }
        __syncthreads();

        if (apply_hadamard) {
            for (int stride = 1; stride < head_dim; stride <<= 1) {
                if (tid < head_dim / 2) {
                    const int pair_idx = (tid / stride) * (stride * 2) + (tid % stride);
                    const float a = smem[pair_idx];
                    const float b = smem[pair_idx + stride];
                    smem[pair_idx] = a + b;
                    smem[pair_idx + stride] = a - b;
                }
                __syncthreads();
            }
        }

        if (tid < head_dim) {
            float value = smem[tid];
            if (apply_hadamard) {
                value *= hadamard_scale;
            }
            local_sum += value * value;
            token_output[head_idx * head_dim + tid] = from_float_device<scalar_t>(value);
        }
        __syncthreads();
    }

    smem[tid] = local_sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            smem[tid] += smem[tid + stride];
        }
        __syncthreads();
    }

    float token_norm = 1.0f;
    if (apply_norm) {
        token_norm = sqrtf(smem[0]);
        if (!(token_norm > 0.0f)) {
            token_norm = 1.0f;
        }
    }

    if (tid == 0 && norm_output != nullptr) {
        norm_output[token_idx] = token_norm;
    }

    if (apply_norm) {
        for (int idx = tid; idx < token_stride; idx += blockDim.x) {
            const float value = to_float_device(token_output[idx]) / token_norm;
            token_output[idx] = from_float_device<scalar_t>(value);
        }
    }
}

template <typename scalar_t>
__global__ void preprocess_k_cache_kernel_tc128(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    float* __restrict__ norm_output,
    int total_tokens,
    int num_heads,
    bool apply_norm
) {
    using wmma_input_t = typename WmmaInputType<scalar_t>::type;
    using native_input_t = typename NativeInputType<scalar_t>::type;
    extern __shared__ unsigned char shared_raw[];

    native_input_t* had_matrix = reinterpret_cast<native_input_t*>(shared_raw);
    native_input_t* warp_input_tiles = reinterpret_cast<native_input_t*>(had_matrix + (kWmmaTile * kWmmaTile));
    float* warp_output_tiles = reinterpret_cast<float*>(warp_input_tiles + kWarpCount * (kWmmaTile * kWmmaTile));
    float* warp_partial_sums = warp_output_tiles + kWarpCount * (kWmmaTile * kWmmaTile);
    __shared__ float token_norm_shared;

    const int token_idx = blockIdx.x;
    const int tid = threadIdx.x;
    const int warp_id = tid / kWarpSize;
    const int lane = tid % kWarpSize;
    const int token_stride = num_heads * kHadamardHeadDim;
    const float hadamard_scale = rsqrtf(static_cast<float>(kHadamardHeadDim));

    if (token_idx >= total_tokens) {
        return;
    }

    for (int idx = tid; idx < kWmmaTile * kWmmaTile; idx += blockDim.x) {
        const int row = idx / kWmmaTile;
        const int col = idx % kWmmaTile;
        const float sign = (__popc(row & col) & 1) ? -1.0f : 1.0f;
        had_matrix[idx] = from_float_native<scalar_t>(sign);
    }
    __syncthreads();

    const scalar_t* token_input = input + static_cast<int64_t>(token_idx) * token_stride;
    scalar_t* token_output = output + static_cast<int64_t>(token_idx) * token_stride;

    float local_sum = 0.0f;

    for (int head_idx = warp_id; head_idx < num_heads; head_idx += kWarpCount) {
        native_input_t* input_tile = warp_input_tiles + warp_id * (kWmmaTile * kWmmaTile);
        float* output_tile = warp_output_tiles + warp_id * (kWmmaTile * kWmmaTile);

        for (int idx = lane; idx < kWmmaTile * kWmmaTile; idx += kWarpSize) {
            const int row = idx / kWmmaTile;
            const int col = idx % kWmmaTile;
            float value = 0.0f;
            if (row < kHadamardRows) {
                value = to_float_device(token_input[head_idx * kHadamardHeadDim + row * kHadamardCols + col]);
            }
            input_tile[idx] = from_float_native<scalar_t>(value);
        }
        __syncwarp();

        wmma::fragment<wmma::matrix_a, kWmmaTile, kWmmaTile, kWmmaTile, wmma_input_t, wmma::row_major> a_frag;
        wmma::fragment<wmma::matrix_b, kWmmaTile, kWmmaTile, kWmmaTile, wmma_input_t, wmma::col_major> b_frag;
        wmma::fragment<wmma::accumulator, kWmmaTile, kWmmaTile, kWmmaTile, float> c_frag;
        wmma::fill_fragment(c_frag, 0.0f);
        wmma::load_matrix_sync(a_frag, input_tile, kWmmaTile);
        wmma::load_matrix_sync(b_frag, had_matrix, kWmmaTile);
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        wmma::store_matrix_sync(output_tile, c_frag, kWmmaTile, wmma::mem_row_major);
        __syncwarp();

        for (int stride = 1; stride < kHadamardRows; stride <<= 1) {
            for (int idx = lane; idx < kHadamardHeadDim; idx += kWarpSize) {
                const int row = idx / kHadamardCols;
                const int col = idx % kHadamardCols;
                if ((row & stride) == 0) {
                    const int pair_idx = (row + stride) * kHadamardCols + col;
                    const int self_idx = row * kHadamardCols + col;
                    const float a = output_tile[self_idx];
                    const float b = output_tile[pair_idx];
                    output_tile[self_idx] = a + b;
                    output_tile[pair_idx] = a - b;
                }
            }
            __syncwarp();
        }

        for (int idx = lane; idx < kHadamardHeadDim; idx += kWarpSize) {
            const float value = output_tile[idx] * hadamard_scale;
            local_sum += value * value;
            token_output[head_idx * kHadamardHeadDim + idx] = from_float_device<scalar_t>(value);
        }
        __syncwarp();
    }

    for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
        local_sum += __shfl_down_sync(0xffffffff, local_sum, offset);
    }

    if (lane == 0) {
        warp_partial_sums[warp_id] = local_sum;
    }
    __syncthreads();

    if (warp_id == 0) {
        float block_sum = (lane < kWarpCount) ? warp_partial_sums[lane] : 0.0f;
        for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
            block_sum += __shfl_down_sync(0xffffffff, block_sum, offset);
        }
        if (lane == 0) {
            float token_norm = 1.0f;
            if (apply_norm) {
                token_norm = sqrtf(block_sum);
                if (!(token_norm > 0.0f)) {
                    token_norm = 1.0f;
                }
            }
            token_norm_shared = token_norm;
            if (norm_output != nullptr) {
                norm_output[token_idx] = token_norm;
            }
        }
    }
    __syncthreads();

    if (apply_norm) {
        for (int idx = tid; idx < token_stride; idx += blockDim.x) {
            const float value = to_float_device(token_output[idx]) / token_norm_shared;
            token_output[idx] = from_float_device<scalar_t>(value);
        }
    }
}

template <typename scalar_t>
void launch_preprocess_k_cache(
    const at::Tensor& key_states,
    at::Tensor& key_states_out,
    at::Tensor& key_norm_out,
    bool apply_hadamard,
    bool apply_norm
) {
    const auto batch_size = static_cast<int>(key_states.size(0));
    const auto seqlen = static_cast<int>(key_states.size(1));
    const auto num_heads = static_cast<int>(key_states.size(2));
    const auto head_dim = static_cast<int>(key_states.size(3));
    const int total_tokens = batch_size * seqlen;

    const scalar_t* input_ptr = reinterpret_cast<const scalar_t*>(key_states.data_ptr());
    scalar_t* output_ptr = reinterpret_cast<scalar_t*>(key_states_out.data_ptr());
    float* norm_ptr = apply_norm ? reinterpret_cast<float*>(key_norm_out.data_ptr()) : nullptr;

    if (apply_hadamard && head_dim == kHadamardHeadDim) {
        const size_t shared_bytes =
            (kWmmaTile * kWmmaTile) * sizeof(typename NativeInputType<scalar_t>::type) +
            kWarpCount * (kWmmaTile * kWmmaTile) * sizeof(typename NativeInputType<scalar_t>::type) +
            kWarpCount * (kWmmaTile * kWmmaTile) * sizeof(float) +
            kWarpCount * sizeof(float);

        preprocess_k_cache_kernel_tc128<scalar_t><<<
            total_tokens,
            kThreadsPerBlock,
            shared_bytes,
            at::cuda::getCurrentCUDAStream()
        >>>(
            input_ptr,
            output_ptr,
            norm_ptr,
            total_tokens,
            num_heads,
            apply_norm
        );
    } else {
        constexpr int kGenericThreads = 128;
        preprocess_k_cache_kernel_generic<scalar_t><<<
            total_tokens,
            kGenericThreads,
            kGenericThreads * sizeof(float),
            at::cuda::getCurrentCUDAStream()
        >>>(
            input_ptr,
            output_ptr,
            norm_ptr,
            total_tokens,
            num_heads,
            head_dim,
            apply_hadamard,
            apply_norm
        );
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace

std::tuple<at::Tensor, at::Tensor> preprocess_k_cache_cuda(
    const at::Tensor& key_states,
    bool apply_hadamard,
    bool apply_norm
) {
    CHECK_DEVICE(key_states);
    TORCH_CHECK(key_states.dtype() == torch::kFloat16 || key_states.dtype() == torch::kBFloat16,
                "K preprocess only supports fp16 and bf16");
    TORCH_CHECK(key_states.dim() == 4, "Expected key_states with shape [batch, seqlen, heads, head_dim]");
    TORCH_CHECK(key_states.stride(-1) == 1, "key_states must have contiguous last dimension");
    TORCH_CHECK(key_states.is_contiguous(), "key_states must be contiguous");
    TORCH_CHECK(!apply_hadamard || key_states.size(-1) == 128,
                "Current K preprocess kernel only supports head_dim=128 when Hadamard is enabled");
    TORCH_CHECK(key_states.size(-1) <= 128,
                "Current K preprocess kernel expects head_dim <= 128");

    if (!apply_hadamard && !apply_norm) {
        return std::make_tuple(key_states, at::empty({0}, key_states.options()));
    }

    at::cuda::CUDAGuard device_guard{static_cast<char>(key_states.get_device())};

    auto key_states_out = torch::empty_like(key_states);
    auto key_norm_out = apply_norm
        ? torch::empty({key_states.size(0), key_states.size(1)}, key_states.options().dtype(torch::kFloat32))
        : torch::empty({0}, key_states.options().dtype(torch::kFloat32));

    if (key_states.dtype() == torch::kFloat16) {
        launch_preprocess_k_cache<cutlass::half_t>(key_states, key_states_out, key_norm_out, apply_hadamard, apply_norm);
    } else {
        launch_preprocess_k_cache<cutlass::bfloat16_t>(key_states, key_states_out, key_norm_out, apply_hadamard, apply_norm);
    }

    return std::make_tuple(key_states_out, key_norm_out);
}
