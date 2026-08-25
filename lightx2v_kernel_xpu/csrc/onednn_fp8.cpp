//
// Copyright 2016 The BigDL Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//

// W8A16 GEMM via oneDNN.
//
// Layout
//   A (activations) : [M, K]  FP16/BF16/FP32, format_tag::ab
//   B (weights)     : [N, K]  FP8, stored physically as [N, K],
//                           logical [K, N] -> format_tag::ba
//   scale           : [N]     FP32, per-output-channel
//   C (output)      : [M, N]  same dtype as A, format_tag::ab

#include <algorithm>
#include <cstdio>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <tuple>
#include <unordered_map>
#include <unordered_set>

#include <oneapi/dnnl/dnnl.hpp>
#include <oneapi/dnnl/dnnl_sycl.hpp>
#include <torch/extension.h>

#include "utils.h"

using ST = torch::ScalarType;
using DT = dnnl::memory::data_type;

namespace {

constexpr const char* kUnsupportedMarker =
    "LIGHTX2V_FP8_PRIMITIVE_UNSUPPORTED:";

struct CacheKey {
    int device_index;
    int input_type;
    int weight_type;
    int64_t M;
    int64_t K;
    int64_t N;
    bool has_bias;

    bool operator==(const CacheKey& other) const {
        return device_index == other.device_index &&
            input_type == other.input_type &&
            weight_type == other.weight_type && M == other.M && K == other.K &&
            N == other.N && has_bias == other.has_bias;
    }
};

struct CacheKeyHash {
    size_t operator()(const CacheKey& key) const {
        size_t seed = 0;
        auto combine = [&](size_t value) {
            seed ^= value + 0x9e3779b97f4a7c15ULL + (seed << 6) +
                (seed >> 2);
        };
        combine(std::hash<int>{}(key.device_index));
        combine(std::hash<int>{}(key.input_type));
        combine(std::hash<int>{}(key.weight_type));
        combine(std::hash<int64_t>{}(key.M));
        combine(std::hash<int64_t>{}(key.K));
        combine(std::hash<int64_t>{}(key.N));
        combine(std::hash<bool>{}(key.has_bias));
        return seed;
    }
};

struct PrimitiveState {
    dnnl::engine engine;
    dnnl::memory::desc input_desc;
    dnnl::memory::desc weight_desc;
    dnnl::memory::desc scale_desc;
    dnnl::memory::desc bias_desc;
    dnnl::memory::desc output_desc;
    dnnl::matmul primitive;
};

struct CacheCounters {
    int64_t hits = 0;
    int64_t misses = 0;
    int64_t failures = 0;
    int64_t negative_hits = 0;
};

std::mutex cache_mutex;
std::unordered_map<CacheKey, std::shared_ptr<PrimitiveState>, CacheKeyHash>
    primitive_cache;
std::unordered_set<CacheKey, CacheKeyHash> failed_primitive_cache;
CacheCounters cache_counters;

std::optional<int64_t> select_chunk_n_for_shape(
    int64_t M,
    int64_t K,
    int64_t N,
    ST input_type,
    const torch::Device& device) {
    // Known-good chunk sizes for MiniMax-H3 projection shapes. Some full-N
    // oneDNN primitives exhaust the requested register bundle, while these
    // smaller primitives create and execute reliably.
    if (M == 4096 && K == 4096) {
        if (N == 12288) {
            namespace syclex = sycl::ext::oneapi::experimental;
            const auto architecture = utils::get_queue(device)
                                          .get_device()
                                          .get_info<
                                              syclex::info::device::architecture>();
            if (input_type == ST::Half &&
                architecture == syclex::architecture::intel_gpu_ptl_h) {
                return int64_t{2048};
            }
            return int64_t{4096};
        }

        if (input_type == ST::BFloat16 && N == 24576) {
            return int64_t{4096};
        }
    }

    if (M == 4608 && K == 4096) {
        if (N == 16384) {
            // On BMG, the f16 primitive is reliable at N=256; bf16 can use
            // the larger chunk used by the remaining H3 projections.
            return input_type == ST::Half ? int64_t{256} : int64_t{4096};
        }

        if (input_type == ST::BFloat16 && N == 36864) {
            return int64_t{4096};
        }
    }

    if (input_type == ST::BFloat16) {
        if (M == 4096 && K == 12288 && N == 4096) {
            return int64_t{2048};
        }

        if (M == 4608 && K == 16384 && N == 4096) {
            return int64_t{1024};
        }
    }

    return std::nullopt;
}

std::shared_ptr<PrimitiveState> get_primitive(
    int64_t M,
    int64_t K,
    int64_t N,
    DT input_type,
    DT weight_type,
    bool has_bias,
    const torch::Device& device) {
    const CacheKey key{
        device.index(), static_cast<int>(input_type),
        static_cast<int>(weight_type), M, K, N, has_bias};

    std::lock_guard<std::mutex> lock(cache_mutex);
    auto found = primitive_cache.find(key);
    if (found != primitive_cache.end()) {
        ++cache_counters.hits;
        return found->second;
    }
    if (failed_primitive_cache.find(key) != failed_primitive_cache.end()) {
        ++cache_counters.negative_hits;
        TORCH_CHECK(
            false, kUnsupportedMarker, "cached: device=", device, " M=", M,
            " K=", K, " N=", N, " bias=", has_bias);
    }

    sycl::queue& queue = utils::get_queue(device);
    auto state = std::make_shared<PrimitiveState>();
    state->engine = dnnl::sycl_interop::make_engine(
        queue.get_device(), queue.get_context());
    state->input_desc = dnnl::memory::desc(
        {M, K}, input_type, dnnl::memory::format_tag::ab);
    state->weight_desc = dnnl::memory::desc(
        {K, N}, weight_type, dnnl::memory::format_tag::ba);
    state->scale_desc = dnnl::memory::desc(
        {N}, DT::f32, dnnl::memory::format_tag::a);
    state->output_desc = dnnl::memory::desc(
        {M, N}, input_type, dnnl::memory::format_tag::ab);
    if (has_bias) {
        state->bias_desc = dnnl::memory::desc(
            {1, N}, input_type, dnnl::memory::format_tag::ab);
    }

    dnnl::primitive_attr attributes;
    // Bit 1 of logical weight dimensions [K, N]: one scale per output.
    attributes.set_scales_mask(DNNL_ARG_WEIGHTS, 1 << 1);
    attributes.set_fpmath_mode(dnnl::fpmath_mode::any, true);

    try {
        auto primitive_desc = has_bias
            ? dnnl::matmul::primitive_desc(
                  state->engine, state->input_desc, state->weight_desc,
                  state->bias_desc, state->output_desc, attributes)
            : dnnl::matmul::primitive_desc(
                  state->engine, state->input_desc, state->weight_desc,
                  state->output_desc, attributes);
        const std::string implementation = primitive_desc.impl_info_str();
        if (implementation.find("ref") != std::string::npos) {
            std::fprintf(
                stderr,
                "[onednn_w8a16_fp8] WARNING: reference implementation "
                "selected for M=%lld K=%lld N=%lld: %s\n",
                static_cast<long long>(M), static_cast<long long>(K),
                static_cast<long long>(N), implementation.c_str());
        }
        state->primitive = dnnl::matmul(primitive_desc);
        primitive_cache.emplace(key, state);
        ++cache_counters.misses;
        return state;
    } catch (const dnnl::error& error) {
        const bool unsupported = error.status == dnnl_unimplemented ||
            (error.status == dnnl_runtime_error &&
             std::string(error.what()) == "could not create a primitive");
        if (unsupported) {
            if (failed_primitive_cache.emplace(key).second) {
                ++cache_counters.failures;
            }
            TORCH_CHECK(
                false, kUnsupportedMarker, "new: device=", device, " M=", M,
                " K=", K, " N=", N, " bias=", has_bias,
                "; oneDNN: ", error.what());
        }
        throw;
    }
}

void execute_fp8_matmul(
    const std::shared_ptr<PrimitiveState>& state,
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const torch::Tensor& scales,
    const std::optional<torch::Tensor>& bias,
    torch::Tensor& output) {
    sycl::queue& queue = utils::get_queue(input.device());
    dnnl::stream stream = dnnl::sycl_interop::make_stream(
        state->engine, queue);
    std::unordered_map<int, dnnl::memory> arguments = {
        {DNNL_ARG_SRC,
         dnnl::memory(state->input_desc, state->engine, input.data_ptr())},
        {DNNL_ARG_WEIGHTS,
         dnnl::memory(state->weight_desc, state->engine, weight.data_ptr())},
        {DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS,
         dnnl::memory(state->scale_desc, state->engine, scales.data_ptr())},
        {DNNL_ARG_DST,
         dnnl::memory(state->output_desc, state->engine, output.data_ptr())},
    };
    if (bias.has_value()) {
        arguments.emplace(
            DNNL_ARG_BIAS,
            dnnl::memory(
                state->bias_desc, state->engine, bias->data_ptr()));
    }
    state->primitive.execute(stream, arguments);
}

}  // namespace

void fp8_cache_clear() {
    std::lock_guard<std::mutex> lock(cache_mutex);
    primitive_cache.clear();
    failed_primitive_cache.clear();
    cache_counters = {};
}

std::tuple<int64_t, int64_t, int64_t> fp8_cache_stats() {
    std::lock_guard<std::mutex> lock(cache_mutex);
    return {
        cache_counters.hits, cache_counters.misses,
        static_cast<int64_t>(primitive_cache.size())};
}

std::tuple<int64_t, int64_t, int64_t> fp8_failure_cache_stats() {
    std::lock_guard<std::mutex> lock(cache_mutex);
    return {
        cache_counters.failures, cache_counters.negative_hits,
        static_cast<int64_t>(failed_primitive_cache.size())};
}

torch::Tensor onednn_w8a16_fp8(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor scales,
    std::optional<torch::Tensor> bias) {
    TORCH_CHECK(x.device().is_xpu(), "x must be an XPU tensor");
    TORCH_CHECK(
        weight.device() == x.device(),
        "weight must be on the same XPU device as x");
    TORCH_CHECK(
        scales.device() == x.device(),
        "scales must be on the same XPU device as x");
    TORCH_CHECK(x.dim() == 2, "x must be 2-D [M, K]");
    TORCH_CHECK(weight.dim() == 2, "weight must be 2-D [N, K]");
    TORCH_CHECK(
        x.scalar_type() == ST::Half || x.scalar_type() == ST::BFloat16 ||
            x.scalar_type() == ST::Float,
        "x must have dtype torch.float16, torch.bfloat16, or torch.float32");
    TORCH_CHECK(
        weight.scalar_type() == ST::Float8_e4m3fn ||
            weight.scalar_type() == ST::Float8_e5m2,
        "weight must have dtype torch.float8_e4m3fn or torch.float8_e5m2");
    TORCH_CHECK(
        scales.scalar_type() == ST::Float,
        "scales must have dtype torch.float32");

    const int64_t M = x.size(0);
    const int64_t K = x.size(1);
    const int64_t N = weight.size(0);
    TORCH_CHECK(
        weight.size(1) == K, "weight K dimension (", weight.size(1),
        ") must equal x K dimension (", K, ")");
    TORCH_CHECK(
        scales.numel() == N, "scales must contain N=", N,
        " values, got ", scales.numel());

    if (bias.has_value()) {
        TORCH_CHECK(
            bias->device() == x.device(),
            "bias must be on the same XPU device as x");
        TORCH_CHECK(
            bias->dim() == 1 && bias->numel() == N,
            "bias must have shape [N] = [", N, "]");
        TORCH_CHECK(
            bias->scalar_type() == x.scalar_type(),
            "bias dtype must match x dtype");
    }

    // oneDNN descriptors below describe dense [M,K], [N,K], and [N] storage.
    // Avoid silent wrong results for views while keeping the common path zero-copy.
    auto x_contiguous = x.contiguous();
    auto weight_contiguous = weight.contiguous();
    // Normalize both supported scale layouts ([N] and [N, 1]) so N-chunking
    // always slices along the output-channel dimension.
    auto scales_contiguous = scales.contiguous().view({N});
    std::optional<torch::Tensor> bias_contiguous;
    if (bias.has_value()) {
        bias_contiguous = bias->contiguous();
    }

    if (M == 0 || N == 0) {
        return torch::empty({M, N}, x.options());
    }

    const DT input_type = x.scalar_type() == ST::Half
        ? DT::f16
        : (x.scalar_type() == ST::BFloat16 ? DT::bf16 : DT::f32);
    const DT weight_type = weight.scalar_type() == ST::Float8_e5m2
        ? DT::f8_e5m2
        : DT::f8_e4m3;
    auto output = torch::empty({M, N}, x.options());
    const auto chunk_n = select_chunk_n_for_shape(
        M, K, N, x.scalar_type(), x.device());
    if (chunk_n.has_value()) {
        for (int64_t offset = 0; offset < N; offset += *chunk_n) {
            const int64_t current_n = std::min(*chunk_n, N - offset);
            auto weight_chunk =
                weight_contiguous.slice(0, offset, offset + current_n);
            auto scales_chunk =
                scales_contiguous.slice(0, offset, offset + current_n);
            std::optional<torch::Tensor> bias_chunk;
            if (bias_contiguous.has_value()) {
                bias_chunk =
                    bias_contiguous->slice(0, offset, offset + current_n);
            }

            auto output_chunk = torch::empty({M, current_n}, x.options());
            auto state = get_primitive(
                M, K, current_n, input_type, weight_type,
                bias_chunk.has_value(), x.device());
            execute_fp8_matmul(
                state, x_contiguous, weight_chunk, scales_chunk, bias_chunk,
                output_chunk);
            output.slice(1, offset, offset + current_n).copy_(output_chunk);
        }
    } else {
        auto state = get_primitive(
            M, K, N, input_type, weight_type, bias.has_value(), x.device());
        execute_fp8_matmul(
            state, x_contiguous, weight_contiguous, scales_contiguous,
            bias_contiguous, output);
    }
    return output;
}
