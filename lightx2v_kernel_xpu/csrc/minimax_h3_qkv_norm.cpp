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
class MiniMaxH3QKVNormKernel;

template <typename T>
void launch_qkv_norm(const T* packed, const T* q_weight,
                     const T* k_weight, T* q, T* k, T* v, float q_eps,
                     float k_eps, int64_t tokens, int64_t heads,
                     int64_t row_stride, const c10::Device& device) {
    const int64_t rows = tokens * heads;
    auto& queue = c10::xpu::getCurrentXPUStream(device.index()).queue();
    queue.submit([&](sycl::handler& handler) {
        handler.parallel_for<MiniMaxH3QKVNormKernel<T>>(
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
                    block_store<T, kBlockSize>(
                        output + block * kBlockSize,
                        simd<T, kBlockSize>(
                            values.template select<kBlockSize, 1>(
                                block * kBlockSize) *
                            scale * weights));
                }
            });
    });
}

}  // namespace

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
minimax_h3_qkv_norm_xpu(const torch::Tensor& packed,
                        const torch::Tensor& q_weight,
                        const torch::Tensor& k_weight, double q_eps,
                        double k_eps) {
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
    const int64_t heads = packed.size(1) / (3 * kHeadDim);
    const auto shape = std::vector<int64_t>{tokens, heads, kHeadDim};
    auto q = torch::empty(shape, packed.options());
    auto k = torch::empty(shape, packed.options());
    auto v = torch::empty(shape, packed.options());
    if (tokens == 0) return {q, k, v};

    if (packed.scalar_type() == torch::kBFloat16) {
        launch_qkv_norm<bf16>(
            reinterpret_cast<const bf16*>(packed.data_ptr()),
            reinterpret_cast<const bf16*>(q_weight.data_ptr()),
            reinterpret_cast<const bf16*>(k_weight.data_ptr()),
            reinterpret_cast<bf16*>(q.data_ptr()),
            reinterpret_cast<bf16*>(k.data_ptr()),
            reinterpret_cast<bf16*>(v.data_ptr()),
            static_cast<float>(q_eps), static_cast<float>(k_eps), tokens,
            heads, packed.stride(0), packed.device());
    } else if (packed.scalar_type() == torch::kFloat16) {
        launch_qkv_norm<fp16>(
            reinterpret_cast<const fp16*>(packed.data_ptr()),
            reinterpret_cast<const fp16*>(q_weight.data_ptr()),
            reinterpret_cast<const fp16*>(k_weight.data_ptr()),
            reinterpret_cast<fp16*>(q.data_ptr()),
            reinterpret_cast<fp16*>(k.data_ptr()),
            reinterpret_cast<fp16*>(v.data_ptr()),
            static_cast<float>(q_eps), static_cast<float>(k_eps), tokens,
            heads, packed.stride(0), packed.device());
    } else if (packed.scalar_type() == torch::kFloat32) {
        launch_qkv_norm<float>(
            packed.data_ptr<float>(), q_weight.data_ptr<float>(),
            k_weight.data_ptr<float>(), q.data_ptr<float>(), k.data_ptr<float>(),
            v.data_ptr<float>(), static_cast<float>(q_eps),
            static_cast<float>(k_eps), tokens, heads, packed.stride(0),
            packed.device());
    } else {
        TORCH_CHECK(false, "MiniMax-H3 QKV norm supports fp32, fp16, and bf16");
    }
    return {q, k, v};
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
}

TORCH_LIBRARY_IMPL(sycl_kernels_minimax_h3_qkv, XPU, m) {
    m.impl("qkv_norm", &minimax_h3_qkv_norm_xpu);
}

TORCH_LIBRARY_IMPL(sycl_kernels_minimax_h3_qkv, Meta, m) {
    m.impl("qkv_norm", &minimax_h3_qkv_norm_meta);
}
