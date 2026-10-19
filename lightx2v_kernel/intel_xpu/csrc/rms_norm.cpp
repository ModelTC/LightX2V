#include <algorithm>
#include <cmath>
#include <torch/extension.h>
#include <sycl/ext/intel/esimd.hpp>
#include <sycl/sycl.hpp>
#include <c10/xpu/XPUStream.h>

using fp16 = sycl::half;
using bf16 = sycl::ext::oneapi::bfloat16;
using namespace sycl::ext::intel::esimd;

namespace {

template <typename T, int GroupSize, int BlockSize>
void launch_rms_norm(const T* weight, const T* input, T* output,
                     float eps, int rows, int hidden_size,
                     const c10::Device& device) {
    const int blocks = hidden_size / BlockSize;
    const int blocks_per_item = blocks / GroupSize;
    const int extra_blocks = blocks % GroupSize;
    constexpr int partial_bytes =
        ((GroupSize * static_cast<int>(sizeof(float)) + 15) / 16) * 16;
    const int partial_offset = hidden_size * sizeof(T);

    auto& queue = c10::xpu::getCurrentXPUStream(device.index()).queue();
    queue.submit([&](sycl::handler& handler) {
        handler.parallel_for(
            sycl::nd_range<2>(sycl::range<2>(rows, GroupSize),
                              sycl::range<2>(1, GroupSize)),
            [=](sycl::nd_item<2> item) SYCL_ESIMD_KERNEL {
                slm_init<8192 * sizeof(T) + partial_bytes>();
                const int row = item.get_global_id(0);
                const int lane = item.get_local_id(1);
                const T* input_row = input + static_cast<size_t>(row) * hidden_size;
                T* output_row = output + static_cast<size_t>(row) * hidden_size;
                const int first = blocks_per_item * lane + std::min(lane, extra_blocks);
                const int last = first + blocks_per_item + (lane < extra_blocks);
                simd<float, BlockSize> accumulator = 0;

                for (int block = first; block < last; ++block) {
                    simd<T, BlockSize> values =
                        block_load<T, BlockSize>(input_row + block * BlockSize);
                    slm_block_store<T, BlockSize>(
                        block * BlockSize * sizeof(T), values);
                    simd<float, BlockSize> values_fp32 = values;
                    accumulator += values_fp32 * values_fp32;
                }
                const float partial =
                    sycl::ext::intel::esimd::detail::sum<
                        float, float, BlockSize>(accumulator) /
                    static_cast<float>(hidden_size);

                float scale;
                if constexpr (GroupSize == 1) {
                    scale = rsqrt(partial + eps);
                } else {
                    slm_block_store<float, 1>(
                        partial_offset + lane * sizeof(float), partial);
                    barrier();
                    simd<float, GroupSize> partials =
                        slm_block_load<float, GroupSize>(partial_offset);
                    const float mean =
                        sycl::ext::intel::esimd::detail::sum<
                            float, float, GroupSize>(partials);
                    scale = rsqrt(mean + eps);
                }

                for (int block = first; block < last; ++block) {
                    simd<float, BlockSize> values =
                        slm_block_load<T, BlockSize>(
                            block * BlockSize * sizeof(T));
                    simd<float, BlockSize> weights =
                        block_load<T, BlockSize>(weight + block * BlockSize);
                    block_store<T, BlockSize>(
                        output_row + block * BlockSize,
                        simd<T, BlockSize>(values * scale * weights));
                }
            });
    });
}

template <typename T, int BlockSize>
void dispatch_rms_norm(const T* weight, const T* input, T* output,
                       float eps, int rows, int hidden_size,
                       const c10::Device& device) {
    const int blocks = hidden_size / BlockSize;
    if (blocks <= 1) return launch_rms_norm<T, 1, BlockSize>(weight, input, output, eps, rows, hidden_size, device);
    if (blocks <= 2) return launch_rms_norm<T, 2, BlockSize>(weight, input, output, eps, rows, hidden_size, device);
    if (blocks <= 4) return launch_rms_norm<T, 4, BlockSize>(weight, input, output, eps, rows, hidden_size, device);
    if (blocks <= 8) return launch_rms_norm<T, 8, BlockSize>(weight, input, output, eps, rows, hidden_size, device);
    if (blocks <= 16) return launch_rms_norm<T, 16, BlockSize>(weight, input, output, eps, rows, hidden_size, device);
    return launch_rms_norm<T, 32, BlockSize>(weight, input, output, eps, rows, hidden_size, device);
}

}  // namespace

torch::Tensor rms_norm_xpu(torch::Tensor weight, torch::Tensor input,
                           double eps) {
    TORCH_CHECK(input.is_xpu() && weight.is_xpu(),
                "input and weight must be XPU tensors");
    TORCH_CHECK(input.device() == weight.device(),
                "input and weight must be on the same XPU device");
    TORCH_CHECK(input.dim() == 2, "input must be [rows, hidden_size]");
    TORCH_CHECK(input.size(0) > 0, "input must contain at least one row");
    TORCH_CHECK(weight.dim() == 1 && weight.size(0) == input.size(1),
                "weight must be [hidden_size]");
    TORCH_CHECK(input.scalar_type() == weight.scalar_type(),
                "input and weight dtype must match");
    TORCH_CHECK(input.is_contiguous() && weight.is_contiguous(),
                "input and weight must be contiguous");
    TORCH_CHECK(std::isfinite(eps) && eps > 0.0,
                "eps must be positive and finite");
    const auto hidden_size = input.size(1);
    TORCH_CHECK(hidden_size > 0 && hidden_size <= 8192 && hidden_size % 32 == 0,
                "hidden_size must be positive, <= 8192, and divisible by 32");

    auto output = torch::empty_like(input);
    const int rows = static_cast<int>(input.size(0));
    const int hidden = static_cast<int>(hidden_size);
    if (input.scalar_type() == torch::kBFloat16) {
        dispatch_rms_norm<bf16, 32>(
            reinterpret_cast<const bf16*>(weight.data_ptr()),
            reinterpret_cast<const bf16*>(input.data_ptr()),
            reinterpret_cast<bf16*>(output.data_ptr()),
            static_cast<float>(eps), rows, hidden, input.device());
    } else if (input.scalar_type() == torch::kFloat16) {
        dispatch_rms_norm<fp16, 32>(
            reinterpret_cast<const fp16*>(weight.data_ptr()),
            reinterpret_cast<const fp16*>(input.data_ptr()),
            reinterpret_cast<fp16*>(output.data_ptr()),
            static_cast<float>(eps), rows, hidden, input.device());
    } else if (input.scalar_type() == torch::kFloat32) {
        dispatch_rms_norm<float, 32>(
            weight.data_ptr<float>(), input.data_ptr<float>(),
            output.data_ptr<float>(), static_cast<float>(eps), rows, hidden,
            input.device());
    } else {
        TORCH_CHECK(false, "rms_norm supports fp32, fp16, and bf16");
    }
    return output;
}

TORCH_LIBRARY(sycl_kernels_rms, m) {
    m.def("rms_norm(Tensor weight, Tensor input, float eps=0.000001) -> Tensor");
}

TORCH_LIBRARY_IMPL(sycl_kernels_rms, XPU, m) {
    m.impl("rms_norm", &rms_norm_xpu);
}
