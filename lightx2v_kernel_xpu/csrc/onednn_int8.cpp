#include <cstdio>
#include <memory>
#include <mutex>
#include <optional>
#include <tuple>
#include <unordered_map>

#include <oneapi/dnnl/dnnl.hpp>
#include <oneapi/dnnl/dnnl_sycl.hpp>
#include <torch/extension.h>

#include "utils.h"

using ST = torch::ScalarType;
using DT = dnnl::memory::data_type;

namespace {

constexpr int kQuantizeWorkgroupSize = 256;

template <typename input_t>
class QuantizeInt8RowwiseKernel;

template <typename input_t>
std::tuple<torch::Tensor, torch::Tensor> quantize_int8_rowwise(
    const torch::Tensor& x) {
    const int64_t M = x.size(0);
    const int64_t K = x.size(1);
    auto qx = torch::empty_like(x, x.options().dtype(torch::kInt8));
    auto scales = torch::empty({M}, x.options().dtype(torch::kFloat32));

    const auto* input = reinterpret_cast<const input_t*>(x.data_ptr());
    auto* output = qx.data_ptr<int8_t>();
    auto* row_scales = scales.data_ptr<float>();
    sycl::queue& queue = utils::get_queue(x.device());

    queue.submit([&](sycl::handler& handler) {
        sycl::local_accessor<float, 1> maxima(
            sycl::range<1>(kQuantizeWorkgroupSize), handler);
        handler.parallel_for<QuantizeInt8RowwiseKernel<input_t>>(
            sycl::nd_range<1>(
                sycl::range<1>(M * kQuantizeWorkgroupSize),
                sycl::range<1>(kQuantizeWorkgroupSize)),
            [=](sycl::nd_item<1> item) {
                const int64_t row = item.get_group(0);
                const int lane = item.get_local_id(0);
                const int64_t offset = row * K;
                float local_max = 0.0f;
                for (int64_t col = lane; col < K;
                     col += kQuantizeWorkgroupSize) {
                    local_max = sycl::fmax(
                        local_max,
                        sycl::fabs(static_cast<float>(input[offset + col])));
                }
                maxima[lane] = local_max;
                item.barrier(sycl::access::fence_space::local_space);
                for (int stride = kQuantizeWorkgroupSize / 2; stride > 0;
                     stride /= 2) {
                    if (lane < stride) {
                        maxima[lane] = sycl::fmax(
                            maxima[lane], maxima[lane + stride]);
                    }
                    item.barrier(sycl::access::fence_space::local_space);
                }

                const float scale = sycl::fmax(maxima[0] / 127.0f, 1.0e-30f);
                if (lane == 0) {
                    row_scales[row] = scale;
                }
                for (int64_t col = lane; col < K;
                     col += kQuantizeWorkgroupSize) {
                    float value = sycl::rint(
                        static_cast<float>(input[offset + col]) / scale);
                    value = sycl::fmin(127.0f, sycl::fmax(-127.0f, value));
                    output[offset + col] = static_cast<int8_t>(value);
                }
            });
    });
    return {qx, scales};
}

struct CacheKey {
    int device;
    int output_type;
    int64_t M;
    int64_t K;
    int64_t N;
    bool has_bias;

    bool operator==(const CacheKey& other) const {
        return device == other.device && output_type == other.output_type &&
            M == other.M && K == other.K && N == other.N &&
            has_bias == other.has_bias;
    }
};

struct CacheKeyHash {
    size_t operator()(const CacheKey& key) const {
        size_t seed = 0;
        auto combine = [&](size_t value) {
            seed ^= value + 0x9e3779b97f4a7c15ULL + (seed << 6) + (seed >> 2);
        };
        combine(std::hash<int>{}(key.device));
        combine(std::hash<int>{}(key.output_type));
        combine(std::hash<int64_t>{}(key.M));
        combine(std::hash<int64_t>{}(key.K));
        combine(std::hash<int64_t>{}(key.N));
        combine(std::hash<bool>{}(key.has_bias));
        return seed;
    }
};

struct PrimitiveState {
    dnnl::engine engine;
    dnnl::memory::desc src_desc;
    dnnl::memory::desc weight_desc;
    dnnl::memory::desc src_scale_desc;
    dnnl::memory::desc weight_scale_desc;
    dnnl::memory::desc bias_desc;
    dnnl::memory::desc output_desc;
    dnnl::matmul primitive;
};

std::mutex primitive_cache_mutex;
std::unordered_map<CacheKey, std::shared_ptr<PrimitiveState>, CacheKeyHash>
    primitive_cache;

std::shared_ptr<PrimitiveState> get_primitive(
    int64_t M,
    int64_t K,
    int64_t N,
    DT output_type,
    bool has_bias,
    const torch::Device& device) {
    CacheKey key{
        device.index(), static_cast<int>(output_type), M, K, N, has_bias};
    std::lock_guard<std::mutex> lock(primitive_cache_mutex);
    auto found = primitive_cache.find(key);
    if (found != primitive_cache.end()) {
        return found->second;
    }

    sycl::queue& queue = utils::get_queue(device);
    auto state = std::make_shared<PrimitiveState>();
    state->engine = dnnl::sycl_interop::make_engine(
        queue.get_device(), queue.get_context());
    state->src_desc = dnnl::memory::desc(
        {M, K}, DT::s8, dnnl::memory::format_tag::ab);
    state->weight_desc = dnnl::memory::desc(
        {K, N}, DT::s8, dnnl::memory::format_tag::ba);
    state->src_scale_desc = dnnl::memory::desc(
        {M}, DT::f32, dnnl::memory::format_tag::a);
    state->weight_scale_desc = dnnl::memory::desc(
        {N}, DT::f32, dnnl::memory::format_tag::a);
    state->output_desc = dnnl::memory::desc(
        {M, N}, output_type, dnnl::memory::format_tag::ab);
    if (has_bias) {
        state->bias_desc = dnnl::memory::desc(
            {1, N}, DT::f32, dnnl::memory::format_tag::ab);
    }

    dnnl::primitive_attr attr;
    attr.set_scales(DNNL_ARG_SRC, (1 << 0) | (1 << 1), {1, K}, DT::f32);
    attr.set_scales(DNNL_ARG_WEIGHTS, 1 << 1, {}, DT::f32);
    attr.set_fpmath_mode(dnnl::fpmath_mode::any, true);
    auto primitive_desc = has_bias
        ? dnnl::matmul::primitive_desc(
              state->engine, state->src_desc, state->weight_desc,
              state->bias_desc, state->output_desc, attr)
        : dnnl::matmul::primitive_desc(
              state->engine, state->src_desc, state->weight_desc,
              state->output_desc, attr);
    const std::string implementation = primitive_desc.impl_info_str();
    if (implementation.find("ref") != std::string::npos) {
        std::fprintf(
            stderr,
            "[onednn_w8a8_int8] WARNING: reference implementation selected: %s\n",
            implementation.c_str());
    }
    state->primitive = dnnl::matmul(primitive_desc);
    primitive_cache.emplace(key, state);
    return state;
}

}  // namespace

torch::Tensor onednn_w8a8_int8(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor weight_scales,
    std::optional<torch::Tensor> bias) {
    TORCH_CHECK(x.device().is_xpu(), "x must be an XPU tensor");
    TORCH_CHECK(weight.device() == x.device(),
                "weight must be on the same XPU device as x");
    TORCH_CHECK(weight_scales.device() == x.device(),
                "weight_scales must be on the same XPU device as x");
    TORCH_CHECK(x.dim() == 2, "x must be 2-D [M, K]");
    TORCH_CHECK(weight.dim() == 2, "weight must be 2-D [N, K]");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(weight.is_contiguous(), "weight must be contiguous");
    TORCH_CHECK(weight_scales.is_contiguous(),
                "weight_scales must be contiguous");
    TORCH_CHECK(weight.scalar_type() == ST::Char,
                "weight must have dtype torch.int8");
    TORCH_CHECK(weight_scales.scalar_type() == ST::Float,
                "weight_scales must have dtype torch.float32");
    TORCH_CHECK(x.scalar_type() == ST::Half || x.scalar_type() == ST::BFloat16,
                "x must have dtype torch.float16 or torch.bfloat16");

    const int64_t M = x.size(0);
    const int64_t K = x.size(1);
    const int64_t N = weight.size(0);
    TORCH_CHECK(weight.size(1) == K,
                "weight K dimension (", weight.size(1),
                ") must equal x K dimension (", K, ")");
    TORCH_CHECK(weight_scales.numel() == N,
                "weight_scales must contain N=", N,
                " values, got ", weight_scales.numel());

    torch::Tensor bias_f32;
    if (bias.has_value()) {
        TORCH_CHECK(bias->device() == x.device(),
                    "bias must be on the same XPU device as x");
        TORCH_CHECK(bias->dim() == 1 && bias->numel() == N,
                    "bias must have shape [N] = [", N, "]");
        bias_f32 = bias->to(torch::kFloat32).reshape({1, N}).contiguous();
    }
    if (M == 0) {
        return torch::empty({M, N}, x.options());
    }

    torch::Tensor qx;
    torch::Tensor x_scales;
    if (x.scalar_type() == ST::Half) {
        std::tie(qx, x_scales) = quantize_int8_rowwise<sycl::half>(x);
    } else {
        std::tie(qx, x_scales) =
            quantize_int8_rowwise<sycl::ext::oneapi::bfloat16>(x);
    }

    const DT output_type = x.scalar_type() == ST::Half ? DT::f16 : DT::bf16;
    auto state = get_primitive(
        M, K, N, output_type, bias.has_value(), x.device());
    auto output = torch::empty({M, N}, x.options());
    sycl::queue& queue = utils::get_queue(x.device());
    dnnl::stream stream = dnnl::sycl_interop::make_stream(
        state->engine, queue);
    std::unordered_map<int, dnnl::memory> arguments = {
        {DNNL_ARG_SRC,
         dnnl::memory(state->src_desc, state->engine, qx.data_ptr())},
        {DNNL_ARG_WEIGHTS,
         dnnl::memory(state->weight_desc, state->engine, weight.data_ptr())},
        {DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC,
         dnnl::memory(
             state->src_scale_desc, state->engine, x_scales.data_ptr())},
        {DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS,
         dnnl::memory(
             state->weight_scale_desc, state->engine,
             weight_scales.data_ptr())},
        {DNNL_ARG_DST,
         dnnl::memory(state->output_desc, state->engine, output.data_ptr())},
    };
    if (bias.has_value()) {
        arguments.emplace(
            DNNL_ARG_BIAS,
            dnnl::memory(
                state->bias_desc, state->engine, bias_f32.data_ptr()));
    }
    state->primitive.execute(stream, arguments);
    return output;
}
