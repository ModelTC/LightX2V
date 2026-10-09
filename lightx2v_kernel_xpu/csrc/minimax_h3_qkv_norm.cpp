#include <cmath>
#include <cstdint>
#include <tuple>

#include <c10/xpu/XPUStream.h>
#include <sycl/ext/intel/esimd.hpp>
#include <sycl/sycl.hpp>
#include <torch/extension.h>

using fp16 = sycl::half;
using bf16 = sycl::ext::oneapi::bfloat16;
using namespace sycl::ext::intel::esimd;

namespace {

constexpr int kHeadDim = 128;
constexpr int kBlockSize = 32;
constexpr int kBlocks = kHeadDim / kBlockSize;

template <typename T>
class MiniMaxH3QKVNormRopeCombinedKernel;

template <typename T>
void launch_qkv_norm_rope_combined(
        const T* q_input, const T* k_input, const T* v_input,
        const T* q_weight, const T* k_weight,
        T* q, T* k, T* v, float q_eps, float k_eps, int64_t tokens,
        int64_t heads, int64_t q_stride, int64_t k_stride,
        int64_t v_stride, const c10::Device& device,
        const float* cos, const float* sin, int64_t cos_stride,
        int64_t sin_stride, bool low_precision) {
    const int64_t rows = tokens * heads;
    auto& queue = c10::xpu::getCurrentXPUStream(device.index()).queue();
    queue.submit([&](sycl::handler& handler) {
        handler.parallel_for<MiniMaxH3QKVNormRopeCombinedKernel<T>>(
            sycl::nd_range<1>(rows, 1),
            [=](sycl::nd_item<1> item) SYCL_ESIMD_KERNEL {
                const int64_t row = item.get_global_id(0);
                const int64_t token = row / heads;
                const int64_t head = row - token * heads;
                simd<float, 96> cos_values;
                simd<float, 96> sin_values;
#pragma unroll
                for (int block = 0; block < 3; ++block) {
                    simd<float, kBlockSize> c = block_load<float, kBlockSize>(
                        cos + token * cos_stride + block * kBlockSize);
                    simd<float, kBlockSize> s = block_load<float, kBlockSize>(
                        sin + token * sin_stride + block * kBlockSize);
                    if (low_precision) {
                        c = simd<T, kBlockSize>(c);
                        s = simd<T, kBlockSize>(s);
                    }
                    cos_values.template select<kBlockSize, 1>(block * kBlockSize) = c;
                    sin_values.template select<kBlockSize, 1>(block * kBlockSize) = s;
                }

#pragma unroll
                for (int component = 0; component < 2; ++component) {
                    const T* component_input = component == 0 ? q_input : k_input;
                    const int64_t component_stride = component == 0 ? q_stride : k_stride;
                    const T* input = component_input + token * component_stride +
                        head * kHeadDim;
                    T* output = (component == 0 ? q : k) + row * kHeadDim;
                    simd<float, kHeadDim> values;
#pragma unroll
                    for (int block = 0; block < kBlocks; ++block) {
                        values.template select<kBlockSize, 1>(block * kBlockSize) =
                            block_load<T, kBlockSize>(input + block * kBlockSize);
                    }
                    const float sum = sycl::ext::intel::esimd::detail::sum<
                        float, float, kHeadDim>(values * values);
                    const float scale = rsqrt(
                        sum / static_cast<float>(kHeadDim) +
                        (component == 0 ? q_eps : k_eps));
                    const T* weight = component == 0 ? q_weight : k_weight;
#pragma unroll
                    for (int block = 0; block < kBlocks; ++block) {
                        simd<float, kBlockSize> weights =
                            block_load<T, kBlockSize>(weight + block * kBlockSize);
                        simd<T, kBlockSize> normalized =
                            values.template select<kBlockSize, 1>(block * kBlockSize) *
                            scale * weights;
                        values.template select<kBlockSize, 1>(block * kBlockSize) = normalized;
                    }
#pragma unroll
                    for (int offset = 0; offset < 48; offset += 16) {
                        simd<float, 16> first = values.template select<16, 1>(offset);
                        simd<float, 16> second = values.template select<16, 1>(offset + 48);
                        auto c0 = cos_values.template select<16, 1>(offset);
                        auto c1 = cos_values.template select<16, 1>(offset + 48);
                        auto s0 = sin_values.template select<16, 1>(offset);
                        auto s1 = sin_values.template select<16, 1>(offset + 48);
                        simd<float, 16> a0 = first * c0, b0 = second * s0;
                        simd<float, 16> a1 = second * c1, b1 = first * s1;
                        if (low_precision) {
                            a0 = simd<T, 16>(a0); b0 = simd<T, 16>(b0);
                            a1 = simd<T, 16>(a1); b1 = simd<T, 16>(b1);
                        }
                        block_store<T, 16>(output + offset, simd<T, 16>(a0 - b0));
                        block_store<T, 16>(output + offset + 48, simd<T, 16>(a1 + b1));
                    }
                    block_store<T, 32>(output + 96,
                        simd<T, 32>(values.template select<32, 1>(96)));
                }

                const T* v_source = v_input + token * v_stride + head * kHeadDim;
                T* v_output = v + row * kHeadDim;
#pragma unroll
                for (int block = 0; block < kBlocks; ++block) {
                    block_store<T, kBlockSize>(v_output + block * kBlockSize,
                        block_load<T, kBlockSize>(v_source + block * kBlockSize));
                }
            });
    });
}

}  // namespace

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
minimax_h3_qkv_norm_rope_impl(const torch::Tensor& q_input,
                              const torch::Tensor& k_input,
                              const torch::Tensor& v_input,
                              const torch::Tensor& q_weight,
                              const torch::Tensor& k_weight, double q_eps,
                              double k_eps, const torch::Tensor& cos,
                              const torch::Tensor& sin, bool low_precision) {
    TORCH_CHECK(q_input.is_xpu() && k_input.is_xpu() && v_input.is_xpu() &&
                    q_weight.is_xpu() && k_weight.is_xpu(),
                "Q/K/V and norm weights must be XPU tensors");
    TORCH_CHECK(k_input.device() == q_input.device() && v_input.device() == q_input.device() &&
                    q_weight.device() == q_input.device() && k_weight.device() == q_input.device(),
                "Q/K/V and norm weights must be on the same XPU device");
    TORCH_CHECK(q_input.dim() == 2 && k_input.sizes() == q_input.sizes() &&
                    v_input.sizes() == q_input.sizes() && q_input.stride(1) == 1 &&
                    k_input.stride(1) == 1 && v_input.stride(1) == 1,
                "Q/K/V must be equally shaped [tokens, heads * 128] tensors with contiguous channels");
    TORCH_CHECK(q_input.size(1) > 0 && q_input.size(1) % kHeadDim == 0,
                "Q/K/V width must be a positive multiple of 128");
    TORCH_CHECK(q_weight.dim() == 1 && q_weight.numel() == kHeadDim &&
                    k_weight.sizes() == q_weight.sizes(),
                "Q/K RMSNorm weights must have shape [128]");
    TORCH_CHECK(q_weight.is_contiguous() && k_weight.is_contiguous(),
                "Q/K RMSNorm weights must be contiguous");
    TORCH_CHECK(k_input.scalar_type() == q_input.scalar_type() &&
                    v_input.scalar_type() == q_input.scalar_type() &&
                    q_input.scalar_type() == q_weight.scalar_type() &&
                    q_input.scalar_type() == k_weight.scalar_type(),
                "Q/K/V and norm weight dtypes must match");
    TORCH_CHECK(std::isfinite(q_eps) && q_eps > 0.0 &&
                    std::isfinite(k_eps) && k_eps > 0.0,
                "Q/K RMSNorm eps values must be positive and finite");

    const int64_t tokens = q_input.size(0);
    TORCH_CHECK(cos.device() == q_input.device() && sin.device() == q_input.device(),
                "cos/sin must be on the same XPU as Q/K/V");
    TORCH_CHECK(cos.scalar_type() == torch::kFloat32 && sin.scalar_type() == torch::kFloat32,
                "cos/sin must be FP32");
    TORCH_CHECK(cos.dim() == 2 && cos.size(0) == tokens && cos.size(1) == 96 && sin.sizes() == cos.sizes(),
                "cos/sin must have shape [tokens, 96]");
    TORCH_CHECK(cos.stride(1) == 1 && sin.stride(1) == 1,
                "cos/sin must have contiguous channels");
    const float* cos_ptr = cos.data_ptr<float>();
    const float* sin_ptr = sin.data_ptr<float>();
    const int64_t cos_stride = cos.stride(0);
    const int64_t sin_stride = sin.stride(0);
    const int64_t heads = q_input.size(1) / kHeadDim;
    const auto shape = std::vector<int64_t>{tokens, heads, kHeadDim};
    auto q = torch::empty(shape, q_input.options());
    auto k = torch::empty(shape, q_input.options());
    auto v = torch::empty(shape, q_input.options());
    if (tokens == 0) return {q, k, v};

    if (q_input.scalar_type() == torch::kBFloat16) {
        launch_qkv_norm_rope_combined<bf16>(
            reinterpret_cast<const bf16*>(q_input.data_ptr()),
            reinterpret_cast<const bf16*>(k_input.data_ptr()),
            reinterpret_cast<const bf16*>(v_input.data_ptr()),
            reinterpret_cast<const bf16*>(q_weight.data_ptr()),
            reinterpret_cast<const bf16*>(k_weight.data_ptr()),
            reinterpret_cast<bf16*>(q.data_ptr()),
            reinterpret_cast<bf16*>(k.data_ptr()),
            reinterpret_cast<bf16*>(v.data_ptr()),
            static_cast<float>(q_eps), static_cast<float>(k_eps), tokens,
            heads, q_input.stride(0), k_input.stride(0), v_input.stride(0), q_input.device(), cos_ptr, sin_ptr, cos_stride, sin_stride, low_precision);
    } else if (q_input.scalar_type() == torch::kFloat16) {
        launch_qkv_norm_rope_combined<fp16>(
            reinterpret_cast<const fp16*>(q_input.data_ptr()),
            reinterpret_cast<const fp16*>(k_input.data_ptr()),
            reinterpret_cast<const fp16*>(v_input.data_ptr()),
            reinterpret_cast<const fp16*>(q_weight.data_ptr()),
            reinterpret_cast<const fp16*>(k_weight.data_ptr()),
            reinterpret_cast<fp16*>(q.data_ptr()),
            reinterpret_cast<fp16*>(k.data_ptr()),
            reinterpret_cast<fp16*>(v.data_ptr()),
            static_cast<float>(q_eps), static_cast<float>(k_eps), tokens,
            heads, q_input.stride(0), k_input.stride(0), v_input.stride(0), q_input.device(), cos_ptr, sin_ptr, cos_stride, sin_stride, low_precision);
    } else if (q_input.scalar_type() == torch::kFloat32) {
        launch_qkv_norm_rope_combined<float>(
            q_input.data_ptr<float>(), k_input.data_ptr<float>(), v_input.data_ptr<float>(),
            q_weight.data_ptr<float>(),
            k_weight.data_ptr<float>(), q.data_ptr<float>(), k.data_ptr<float>(),
            v.data_ptr<float>(), static_cast<float>(q_eps),
            static_cast<float>(k_eps), tokens, heads, q_input.stride(0),
            k_input.stride(0), v_input.stride(0), q_input.device(), cos_ptr, sin_ptr, cos_stride, sin_stride, low_precision);
    } else {
        TORCH_CHECK(false, "MiniMax-H3 QKV norm supports fp32, fp16, and bf16");
    }
    return {q, k, v};
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
minimax_h3_qkv_norm_rope_xpu(const torch::Tensor& q, const torch::Tensor& k,
                             const torch::Tensor& v, const torch::Tensor& qw,
                             const torch::Tensor& kw, const torch::Tensor& cos,
                             const torch::Tensor& sin, double q_eps, double k_eps,
                             bool low_precision) {
    return minimax_h3_qkv_norm_rope_impl(q, k, v, qw, kw, q_eps, k_eps, cos, sin, low_precision);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
minimax_h3_qkv_norm_meta(const torch::Tensor& q,
                         const torch::Tensor& q_weight,
                         const torch::Tensor& k_weight, double q_eps,
                         double k_eps) {
    TORCH_CHECK(q.dim() == 2 && q.size(1) % kHeadDim == 0,
                "Q must have shape [tokens, heads * 128]");
    const auto shape = std::vector<int64_t>{
        q.size(0), q.size(1) / kHeadDim, kHeadDim};
    return {torch::empty(shape, q.options()),
            torch::empty(shape, q.options()),
            torch::empty(shape, q.options())};
}

TORCH_LIBRARY(sycl_kernels_minimax_h3_qkv, m) {
    m.def("qkv_norm_rope(Tensor q, Tensor k, Tensor v, Tensor q_weight, Tensor k_weight, Tensor cos, Tensor sin, float q_eps, float k_eps, bool low_precision_rope=False) -> (Tensor, Tensor, Tensor)");
}

TORCH_LIBRARY_IMPL(sycl_kernels_minimax_h3_qkv, XPU, m) {
    m.impl("qkv_norm_rope", &minimax_h3_qkv_norm_rope_xpu);
}

TORCH_LIBRARY_IMPL(sycl_kernels_minimax_h3_qkv, Meta, m) {
    m.impl("qkv_norm_rope", [](const torch::Tensor& q, const torch::Tensor& k,
                             const torch::Tensor& v, const torch::Tensor& qw,
                             const torch::Tensor& kw, const torch::Tensor& cos,
                             const torch::Tensor& sin, double q_eps, double k_eps,
                             bool low_precision) {
        TORCH_CHECK(k.sizes() == q.sizes() && v.sizes() == q.sizes(),
                    "Q/K/V must have matching shapes");
        TORCH_CHECK(cos.dim() == 2 && cos.size(0) == q.size(0) && cos.size(1) == 96 && sin.sizes() == cos.sizes(),
                    "cos/sin must have shape [tokens, 96]");
        return minimax_h3_qkv_norm_meta(q, qw, kw, q_eps, k_eps);
    });
}
