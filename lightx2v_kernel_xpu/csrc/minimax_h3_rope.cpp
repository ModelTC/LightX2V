#include <c10/xpu/XPUStream.h>
#include <sycl/sycl.hpp>
#include <torch/extension.h>
#include <cstdint>
#include <limits>

using bf16 = sycl::ext::oneapi::bfloat16;

namespace {
constexpr int64_t kHeadDim = 128;
constexpr int64_t kRotaryDim = 96;
constexpr int64_t kRotaryHalf = 48;
constexpr uint32_t kWorkGroupSize = 256;
class MiniMaxH3RopeKernel;

void launch_rope(const torch::Tensor& input, const torch::Tensor& cos_or_freqs,
                 const torch::Tensor* sin, torch::Tensor& output) {
    const auto* x = reinterpret_cast<const bf16*>(input.data_ptr());
    const auto* cos_or_freqs_ptr = cos_or_freqs.data_ptr<float>();
    const auto* sin_ptr = sin == nullptr ? nullptr : sin->data_ptr<float>();
    auto* out = reinterpret_cast<bf16*>(output.data_ptr());
    const uint32_t rows = input.size(0), heads = input.size(1);
    const uint32_t x_stride = input.stride(0);
    const uint32_t cos_or_freqs_stride = cos_or_freqs.stride(0);
    const uint32_t sin_stride = sin == nullptr ? 0 : sin->stride(0);
    const bool use_cache = sin != nullptr;
    const uint32_t row_values = heads * kHeadDim;
    auto& queue = c10::xpu::getCurrentXPUStream(input.device().index()).queue();
    queue.submit([&](sycl::handler& h) {
        sycl::local_accessor<bf16, 1> cos_cache(kRotaryDim, h);
        sycl::local_accessor<bf16, 1> sin_cache(kRotaryDim, h);
        h.parallel_for<MiniMaxH3RopeKernel>(
            sycl::nd_range<1>(rows * kWorkGroupSize, kWorkGroupSize),
            [=](sycl::nd_item<1> item) {
                const uint32_t row = item.get_group(0);
                const uint32_t lane = item.get_local_id(0);
                if (lane < kRotaryDim) {
                    const float value =
                        cos_or_freqs_ptr[row * cos_or_freqs_stride + lane];
                    cos_cache[lane] = static_cast<bf16>(
                        use_cache ? value : sycl::cos(value));
                    sin_cache[lane] = static_cast<bf16>(use_cache
                        ? sin_ptr[row * sin_stride + lane]
                        : sycl::sin(value));
                }
                item.barrier(sycl::access::fence_space::local_space);
                const bf16* x_row = x + row * x_stride;
                bf16* out_row = out + row * row_values;
                for (uint32_t index = lane; index < row_values;
                     index += kWorkGroupSize) {
                    const uint32_t dim = index % kHeadDim;
                    if (dim >= kRotaryDim) {
                        out_row[index] = x_row[index];
                        continue;
                    }
                    const uint32_t pair = dim < kRotaryHalf
                        ? dim + kRotaryHalf : dim - kRotaryHalf;
                    const uint32_t head_base = index - dim;
                    const float value = static_cast<float>(x_row[index]);
                    const float paired = static_cast<float>(x_row[head_base + pair]);
                    const float cosine = static_cast<float>(cos_cache[dim]);
                    const float sine = static_cast<float>(sin_cache[dim]);
                    const bf16 value_cos = static_cast<bf16>(value * cosine);
                    const bf16 paired_sin = static_cast<bf16>(paired * sine);
                    out_row[index] = static_cast<bf16>(dim < kRotaryHalf
                        ? static_cast<float>(value_cos) - static_cast<float>(paired_sin)
                        : static_cast<float>(value_cos) + static_cast<float>(paired_sin));
                }
            });
    });
}
}  // namespace

torch::Tensor minimax_h3_rope_xpu(const torch::Tensor& input,
                                  const torch::Tensor& freqs) {
    TORCH_CHECK(input.is_xpu() && freqs.is_xpu(), "input and freqs must be XPU tensors");
    TORCH_CHECK(input.device() == freqs.device(), "input and freqs must be on the same XPU device");
    TORCH_CHECK(input.scalar_type() == torch::kBFloat16, "input must be BF16");
    TORCH_CHECK(input.dim() == 3 && input.size(0) > 0 && input.size(1) > 0 && input.size(2) == kHeadDim,
                "input must have shape [rows, heads, 128]");
    TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
    TORCH_CHECK(freqs.scalar_type() == torch::kFloat32, "freqs must be FP32");
    TORCH_CHECK(freqs.dim() == 2 && freqs.size(0) == input.size(0) && freqs.size(1) == kRotaryDim,
                "freqs must have shape [rows, 96]");
    TORCH_CHECK(freqs.stride(1) == 1 && freqs.stride(0) >= kRotaryDim,
                "freqs must have a contiguous last dimension");
    const int64_t max_x = (input.size(0) - 1) * input.stride(0) + input.size(1) * kHeadDim - 1;
    const int64_t max_f = (freqs.size(0) - 1) * freqs.stride(0) + kRotaryDim - 1;
    TORCH_CHECK(max_x <= std::numeric_limits<uint32_t>::max() &&
                    max_f <= std::numeric_limits<uint32_t>::max() &&
                    input.numel() <= std::numeric_limits<uint32_t>::max(),
                "inputs are too large for the MiniMax-H3 RoPE kernel");
    auto output = torch::empty_like(input);
    launch_rope(input, freqs, nullptr, output);
    return output;
}

torch::Tensor minimax_h3_rope_cached_xpu(const torch::Tensor& input,
                                         const torch::Tensor& cos,
                                         const torch::Tensor& sin) {
    TORCH_CHECK(input.is_xpu() && cos.is_xpu() && sin.is_xpu(),
                "input, cos, and sin must be XPU tensors");
    TORCH_CHECK(input.device() == cos.device() && input.device() == sin.device(),
                "input, cos, and sin must be on the same XPU device");
    TORCH_CHECK(input.scalar_type() == torch::kBFloat16,
                "input must be BF16");
    TORCH_CHECK(input.dim() == 3 && input.size(0) > 0 && input.size(1) > 0 &&
                    input.size(2) == kHeadDim && input.is_contiguous(),
                "input must be contiguous with shape [rows, heads, 128]");
    TORCH_CHECK(cos.scalar_type() == torch::kFloat32 &&
                    sin.scalar_type() == torch::kFloat32,
                "cos and sin must be FP32");
    TORCH_CHECK(cos.dim() == 2 && cos.size(0) == input.size(0) &&
                    cos.size(1) == kRotaryDim && sin.sizes() == cos.sizes(),
                "cos and sin must have shape [rows, 96]");
    TORCH_CHECK(cos.stride(1) == 1 && sin.stride(1) == 1,
                "cos and sin must have contiguous last dimensions");
    auto output = torch::empty_like(input);
    launch_rope(input, cos, &sin, output);
    return output;
}

TORCH_LIBRARY(sycl_kernels_minimax_h3, m) {
    m.def("rope(Tensor input, Tensor freqs) -> Tensor");
    m.def("rope_cached(Tensor input, Tensor cos, Tensor sin) -> Tensor");
}
TORCH_LIBRARY_IMPL(sycl_kernels_minimax_h3, XPU, m) {
    m.impl("rope", &minimax_h3_rope_xpu);
    m.impl("rope_cached", &minimax_h3_rope_cached_xpu);
}
