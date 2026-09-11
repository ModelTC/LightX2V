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

template <typename T, bool FuseRope>
class MiniMaxH3QKVNormKernel;

template <typename T>
class MiniMaxH3QKVNormRopeCombinedKernel;

template <typename T, bool FuseRope>
void launch_qkv_norm(const T* packed, const T* q_weight,
                     const T* k_weight, T* q, T* k, T* v, float q_eps,
                     float k_eps, int64_t tokens, int64_t heads,
                     int64_t row_stride, const c10::Device& device,
                     const float* cos, const float* sin,
                     int64_t cos_stride, int64_t sin_stride, bool low_precision) {
    const int64_t rows = tokens * heads;
    auto& queue = c10::xpu::getCurrentXPUStream(device.index()).queue();
    queue.submit([&](sycl::handler& handler) {
        handler.parallel_for<MiniMaxH3QKVNormKernel<T, FuseRope>>(
            sycl::nd_range<1>(rows * 3, 1),
            [=](sycl::nd_item<1> item) SYCL_ESIMD_KERNEL {
                const int64_t work = item.get_global_id(0);
                const int component = work % 3;
                const int64_t row = work / 3;
                const int64_t token = row / heads;
                const int64_t head = row - token * heads;
                const T* input = packed + token * row_stride +
                    (component * heads + head) * kHeadDim;
                T* output = (component == 0 ? q : component == 1 ? k : v) +
                    row * kHeadDim;

                if (component == 2) {
#pragma unroll
                    for (int block = 0; block < kBlocks; ++block) {
                        auto values = block_load<T, kBlockSize>(
                            input + block * kBlockSize);
                        block_store<T, kBlockSize>(
                            output + block * kBlockSize, values);
                    }
                    return;
                }

                // Keep the complete Q/K row in registers. RMSNorm needs the
                // values both for the reduction and for the final scaling;
                // retaining them avoids reading every Q/K element twice.
                simd<float, kHeadDim> values;
#pragma unroll
                for (int block = 0; block < kBlocks; ++block) {
                    values.template select<kBlockSize, 1>(
                        block * kBlockSize) =
                        block_load<T, kBlockSize>(
                            input + block * kBlockSize);
                }
                const float sum =
                    sycl::ext::intel::esimd::detail::sum<
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
                    if constexpr (FuseRope) {
                        // Preserve norm's output rounding before rotation.
                        values.template select<kBlockSize, 1>(block * kBlockSize) = normalized;
                    } else {
                        block_store<T, kBlockSize>(output + block * kBlockSize, normalized);
                    }
                }
                if constexpr (FuseRope) {
                    // Partial split-half RoPE: 96 rotating dimensions, 48 apart.
                    // Process both halves together so each pair stays in registers.
#pragma unroll
                    for (int offset = 0; offset < 48; offset += 16) {
                        simd<float, 16> first = values.template select<16, 1>(offset);
                        simd<float, 16> second = values.template select<16, 1>(offset + 48);
                        auto c0 = block_load<float, 16>(cos + token * cos_stride + offset);
                        auto c1 = block_load<float, 16>(cos + token * cos_stride + offset + 48);
                        auto s0 = block_load<float, 16>(sin + token * sin_stride + offset);
                        auto s1 = block_load<float, 16>(sin + token * sin_stride + offset + 48);
                        if (low_precision) {
                            c0 = simd<T, 16>(c0); c1 = simd<T, 16>(c1);
                            s0 = simd<T, 16>(s0); s1 = simd<T, 16>(s1);
                        }
                        simd<float, 16> a0 = first * c0, b0 = second * s0;
                        simd<float, 16> a1 = second * c1, b1 = first * s1;
                        if (low_precision) {
                            a0 = simd<T, 16>(a0); b0 = simd<T, 16>(b0);
                            a1 = simd<T, 16>(a1); b1 = simd<T, 16>(b1);
                        }
                        block_store<T, 16>(output + offset, simd<T, 16>(a0 - b0));
                        block_store<T, 16>(output + offset + 48, simd<T, 16>(a1 + b1));
                    }
                    block_store<T, 32>(output + 96, simd<T, 32>(values.template select<32, 1>(96)));
                }
            });
    });
}

template <typename T>
void launch_qkv_norm_rope_combined(
        const T* packed, const T* q_weight, const T* k_weight,
        T* q, T* k, T* v, float q_eps, float k_eps, int64_t tokens,
        int64_t heads, int64_t row_stride, const c10::Device& device,
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
                    const T* input = packed + token * row_stride +
                        (component * heads + head) * kHeadDim;
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

                const T* v_input = packed + token * row_stride +
                    (2 * heads + head) * kHeadDim;
                T* v_output = v + row * kHeadDim;
#pragma unroll
                for (int block = 0; block < kBlocks; ++block) {
                    block_store<T, kBlockSize>(v_output + block * kBlockSize,
                        block_load<T, kBlockSize>(v_input + block * kBlockSize));
                }
            });
    });
}

}  // namespace

template <bool FuseRope>
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
minimax_h3_qkv_norm_impl(const torch::Tensor& packed,
                        const torch::Tensor& q_weight,
                        const torch::Tensor& k_weight, double q_eps,
                        double k_eps, const torch::Tensor& cos = {},
                        const torch::Tensor& sin = {}, bool low_precision = false) {
    TORCH_CHECK(packed.is_xpu() && q_weight.is_xpu() && k_weight.is_xpu(),
                "packed QKV and norm weights must be XPU tensors");
    TORCH_CHECK(packed.device() == q_weight.device() &&
                    packed.device() == k_weight.device(),
                "packed QKV and norm weights must be on the same XPU device");
    TORCH_CHECK(packed.dim() == 2 && packed.stride(1) == 1,
                "packed QKV must be [tokens, 3 * heads * 128] with contiguous channels");
    TORCH_CHECK(packed.size(1) > 0 && packed.size(1) % (3 * kHeadDim) == 0,
                "packed QKV width must be a positive multiple of 3 * 128");
    TORCH_CHECK(q_weight.dim() == 1 && q_weight.numel() == kHeadDim &&
                    k_weight.sizes() == q_weight.sizes(),
                "Q/K RMSNorm weights must have shape [128]");
    TORCH_CHECK(q_weight.is_contiguous() && k_weight.is_contiguous(),
                "Q/K RMSNorm weights must be contiguous");
    TORCH_CHECK(packed.scalar_type() == q_weight.scalar_type() &&
                    packed.scalar_type() == k_weight.scalar_type(),
                "packed QKV and norm weight dtypes must match");
    TORCH_CHECK(std::isfinite(q_eps) && q_eps > 0.0 &&
                    std::isfinite(k_eps) && k_eps > 0.0,
                "Q/K RMSNorm eps values must be positive and finite");

    const int64_t tokens = packed.size(0);
    const float* cos_ptr = nullptr;
    const float* sin_ptr = nullptr;
    int64_t cos_stride = 0, sin_stride = 0;
    if constexpr (FuseRope) {
        TORCH_CHECK(cos.device() == packed.device() && sin.device() == packed.device(),
                    "cos/sin must be on the same XPU as packed QKV");
        TORCH_CHECK(cos.scalar_type() == torch::kFloat32 && sin.scalar_type() == torch::kFloat32,
                    "cos/sin must be FP32");
        TORCH_CHECK(cos.dim() == 2 && cos.size(0) == tokens && cos.size(1) == 96 && sin.sizes() == cos.sizes(),
                    "cos/sin must have shape [tokens, 96]");
        TORCH_CHECK(cos.stride(1) == 1 && sin.stride(1) == 1,
                    "cos/sin must have contiguous channels");
        cos_ptr = cos.data_ptr<float>(); sin_ptr = sin.data_ptr<float>();
        cos_stride = cos.stride(0); sin_stride = sin.stride(0);
    }
    const int64_t heads = packed.size(1) / (3 * kHeadDim);
    const auto shape = std::vector<int64_t>{tokens, heads, kHeadDim};
    auto q = torch::empty(shape, packed.options());
    auto k = torch::empty(shape, packed.options());
    auto v = torch::empty(shape, packed.options());
    if (tokens == 0) return {q, k, v};

    if (packed.scalar_type() == torch::kBFloat16) {
        auto launch = FuseRope ? launch_qkv_norm_rope_combined<bf16> :
            launch_qkv_norm<bf16, false>;
        launch(
            reinterpret_cast<const bf16*>(packed.data_ptr()),
            reinterpret_cast<const bf16*>(q_weight.data_ptr()),
            reinterpret_cast<const bf16*>(k_weight.data_ptr()),
            reinterpret_cast<bf16*>(q.data_ptr()),
            reinterpret_cast<bf16*>(k.data_ptr()),
            reinterpret_cast<bf16*>(v.data_ptr()),
            static_cast<float>(q_eps), static_cast<float>(k_eps), tokens,
            heads, packed.stride(0), packed.device(), cos_ptr, sin_ptr, cos_stride, sin_stride, low_precision);
    } else if (packed.scalar_type() == torch::kFloat16) {
        auto launch = FuseRope ? launch_qkv_norm_rope_combined<fp16> :
            launch_qkv_norm<fp16, false>;
        launch(
            reinterpret_cast<const fp16*>(packed.data_ptr()),
            reinterpret_cast<const fp16*>(q_weight.data_ptr()),
            reinterpret_cast<const fp16*>(k_weight.data_ptr()),
            reinterpret_cast<fp16*>(q.data_ptr()),
            reinterpret_cast<fp16*>(k.data_ptr()),
            reinterpret_cast<fp16*>(v.data_ptr()),
            static_cast<float>(q_eps), static_cast<float>(k_eps), tokens,
            heads, packed.stride(0), packed.device(), cos_ptr, sin_ptr, cos_stride, sin_stride, low_precision);
    } else if (packed.scalar_type() == torch::kFloat32) {
        auto launch = FuseRope ? launch_qkv_norm_rope_combined<float> :
            launch_qkv_norm<float, false>;
        launch(
            packed.data_ptr<float>(), q_weight.data_ptr<float>(),
            k_weight.data_ptr<float>(), q.data_ptr<float>(), k.data_ptr<float>(),
            v.data_ptr<float>(), static_cast<float>(q_eps),
            static_cast<float>(k_eps), tokens, heads, packed.stride(0),
            packed.device(), cos_ptr, sin_ptr, cos_stride, sin_stride, low_precision);
    } else {
        TORCH_CHECK(false, "MiniMax-H3 QKV norm supports fp32, fp16, and bf16");
    }
    return {q, k, v};
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
minimax_h3_qkv_norm_xpu(const torch::Tensor& packed, const torch::Tensor& qw,
                      const torch::Tensor& kw, double q_eps, double k_eps) {
    return minimax_h3_qkv_norm_impl<false>(packed, qw, kw, q_eps, k_eps);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
minimax_h3_qkv_norm_rope_xpu(const torch::Tensor& packed, const torch::Tensor& qw,
                           const torch::Tensor& kw, const torch::Tensor& cos,
                           const torch::Tensor& sin, double q_eps, double k_eps,
                           bool low_precision) {
    return minimax_h3_qkv_norm_impl<true>(packed, qw, kw, q_eps, k_eps, cos, sin, low_precision);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
minimax_h3_qkv_norm_meta(const torch::Tensor& packed,
                         const torch::Tensor& q_weight,
                         const torch::Tensor& k_weight, double q_eps,
                         double k_eps) {
    TORCH_CHECK(packed.dim() == 2 && packed.size(1) % (3 * kHeadDim) == 0,
                "packed QKV must have shape [tokens, 3 * heads * 128]");
    const auto shape = std::vector<int64_t>{
        packed.size(0), packed.size(1) / (3 * kHeadDim), kHeadDim};
    return {torch::empty(shape, packed.options()),
            torch::empty(shape, packed.options()),
            torch::empty(shape, packed.options())};
}

TORCH_LIBRARY(sycl_kernels_minimax_h3_qkv, m) {
    m.def("qkv_norm(Tensor packed, Tensor q_weight, Tensor k_weight, float q_eps, float k_eps) -> (Tensor, Tensor, Tensor)");
    m.def("qkv_norm_rope(Tensor packed, Tensor q_weight, Tensor k_weight, Tensor cos, Tensor sin, float q_eps, float k_eps, bool low_precision_rope=False) -> (Tensor, Tensor, Tensor)");
}

TORCH_LIBRARY_IMPL(sycl_kernels_minimax_h3_qkv, XPU, m) {
    m.impl("qkv_norm", &minimax_h3_qkv_norm_xpu);
    m.impl("qkv_norm_rope", &minimax_h3_qkv_norm_rope_xpu);
}

TORCH_LIBRARY_IMPL(sycl_kernels_minimax_h3_qkv, Meta, m) {
    m.impl("qkv_norm", &minimax_h3_qkv_norm_meta);
    m.impl("qkv_norm_rope", [](const torch::Tensor& packed, const torch::Tensor& qw,
                             const torch::Tensor& kw, const torch::Tensor& cos,
                             const torch::Tensor& sin, double q_eps, double k_eps,
                             bool low_precision) {
        TORCH_CHECK(cos.dim() == 2 && cos.size(0) == packed.size(0) && cos.size(1) == 96 && sin.sizes() == cos.sizes(),
                    "cos/sin must have shape [tokens, 96]");
        return minimax_h3_qkv_norm_meta(packed, qw, kw, q_eps, k_eps);
    });
}
