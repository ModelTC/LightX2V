#include <optional>
#include <torch/extension.h>
#include <oneapi/dnnl/dnnl.hpp>
#include <oneapi/dnnl/dnnl_sycl.hpp>

#include "utils.h"

using ST = torch::ScalarType;

namespace {

template <dnnl::memory::data_type input_type>
void onednn_w8a16_int8_impl(
    void* x,
    void* weight,
    void* scales,
    void* output,
    int64_t M,
    int64_t K,
    int64_t N,
    const torch::Device& device) {
    sycl::queue& queue = utils::get_queue(device);
    dnnl::engine engine = dnnl::sycl_interop::make_engine(
        queue.get_device(), queue.get_context());
    dnnl::stream stream = dnnl::sycl_interop::make_stream(engine, queue);

    // PyTorch stores weight as [N, K]. oneDNN sees the same allocation as a
    // logical [K, N] matrix with transposed (ba) strides.
    dnnl::memory::desc x_desc(
        {M, K}, input_type, dnnl::memory::format_tag::ab);
    dnnl::memory::desc weight_desc(
        {K, N}, dnnl::memory::data_type::s8, dnnl::memory::format_tag::ba);
    dnnl::memory::desc scale_desc(
        {N}, dnnl::memory::data_type::f32, dnnl::memory::format_tag::a);
    dnnl::memory::desc output_desc(
        {M, N}, input_type, dnnl::memory::format_tag::ab);

    dnnl::primitive_attr attr;
    // Bit 1 corresponds to N in the logical [K, N] weight matrix.
    attr.set_scales_mask(DNNL_ARG_WEIGHTS, 1 << 1);
    attr.set_fpmath_mode(dnnl::fpmath_mode::any, true);

    dnnl::matmul::primitive_desc primitive_desc(
        engine, x_desc, weight_desc, output_desc, attr);
    dnnl::matmul primitive(primitive_desc);

    std::unordered_map<int, dnnl::memory> arguments = {
        {DNNL_ARG_SRC, dnnl::memory(x_desc, engine, x)},
        {DNNL_ARG_WEIGHTS, dnnl::memory(weight_desc, engine, weight)},
        {DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS,
         dnnl::memory(scale_desc, engine, scales)},
        {DNNL_ARG_DST, dnnl::memory(output_desc, engine, output)},
    };

    primitive.execute(stream, arguments);
}

} // namespace

torch::Tensor onednn_w8a16_int8(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor scales,
    std::optional<torch::Tensor> bias) {
    TORCH_CHECK(x.device().is_xpu(), "x must be an XPU tensor");
    TORCH_CHECK(weight.device() == x.device(),
                "weight must be on the same XPU device as x");
    TORCH_CHECK(scales.device() == x.device(),
                "scales must be on the same XPU device as x");
    TORCH_CHECK(x.dim() == 2, "x must be 2-D [M, K]");
    TORCH_CHECK(weight.dim() == 2, "weight must be 2-D [N, K]");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(weight.is_contiguous(), "weight must be contiguous");
    TORCH_CHECK(scales.is_contiguous(), "scales must be contiguous");
    TORCH_CHECK(weight.scalar_type() == ST::Char,
                "weight must have dtype torch.int8");
    TORCH_CHECK(scales.scalar_type() == ST::Float,
                "scales must have dtype torch.float32");

    const int64_t M = x.size(0);
    const int64_t K = x.size(1);
    const int64_t N = weight.size(0);
    TORCH_CHECK(weight.size(1) == K,
                "weight K dimension (", weight.size(1),
                ") must equal x K dimension (", K, ")");
    TORCH_CHECK(scales.numel() == N,
                "scales must contain N=", N,
                " values, got ", scales.numel());

    if (bias.has_value()) {
        TORCH_CHECK(bias->device() == x.device(),
                    "bias must be on the same XPU device as x");
        TORCH_CHECK(bias->scalar_type() == x.scalar_type(),
                    "bias dtype must match x dtype");
        TORCH_CHECK(bias->is_contiguous(), "bias must be contiguous");
        TORCH_CHECK(bias->dim() == 1 && bias->numel() == N,
                    "bias must have shape [N] = [", N, "]");
    }

    torch::Tensor output = torch::empty(
        {M, N}, torch::device(x.device()).dtype(x.dtype()));

    using DT = dnnl::memory::data_type;
#define DISPATCH_INT8(INPUT_TYPE)                                            \
    onednn_w8a16_int8_impl<INPUT_TYPE>(                                     \
        x.data_ptr(), weight.data_ptr(), scales.data_ptr(),                  \
        output.data_ptr(), M, K, N, x.device())

    switch (x.scalar_type()) {
        case ST::Half:
            DISPATCH_INT8(DT::f16);
            break;
        case ST::BFloat16:
            DISPATCH_INT8(DT::bf16);
            break;
        default:
            TORCH_CHECK(false, "x must have dtype torch.float16 or torch.bfloat16");
    }
#undef DISPATCH_INT8

    // oneDNN GPU does not support a bias descriptor for this mixed floating
    // activation + S8 weight configuration. Preserve Linear semantics with a
    // separate XPU add when bias is requested.
    if (bias.has_value()) {
        output.add_(*bias);
    }

    return output;
}
