#include <torch/extension.h>
#include <ext/intel/esimd.hpp>
#include <oneapi/dnnl/dnnl.hpp>
#include <oneapi/dnnl/dnnl_sycl.hpp>

#include "utils.h"

using fp16 = sycl::half;
using ST = torch::ScalarType;

template <const dnnl::memory::data_type IT>
void onednn_w4a16_kernel(
    void * x,
    void * weight,
    void * scales,
    void * zeros,
    void * bias,
    void * output,
    int64_t input_size,
    int64_t state_size,
    int64_t output_size,
    int64_t block_size,
    const torch::Device& device
) {
    sycl::queue &q = utils::get_queue(device);

    dnnl::engine eng = dnnl::sycl_interop::make_engine(q.get_device(), q.get_context());
    dnnl::stream s = dnnl::sycl_interop::make_stream(eng, q);

    dnnl::memory::desc x_desc({input_size, state_size}, IT, dnnl::memory::format_tag::ab);
    dnnl::memory::desc weight_desc({state_size, output_size}, dnnl::memory::data_type::u4, dnnl::memory::format_tag::ba);
    // scale's `format_tag` setting has no effect
    dnnl::memory::desc scale_desc({state_size / block_size, output_size}, dnnl::memory::data_type::f16, dnnl::memory::format_tag::ba);
    dnnl::memory::desc zero_desc({1}, dnnl::memory::data_type::u8, dnnl::memory::format_tag::a);
    dnnl::memory::desc output_desc({input_size, output_size}, IT, dnnl::memory::format_tag::ab);

    dnnl::primitive_attr attr;
    attr.set_scales(DNNL_ARG_WEIGHTS, (1 << 1) + (1 << 0), {block_size, 1}, dnnl::memory::data_type::f16);
    attr.set_zero_points(DNNL_ARG_WEIGHTS, 0, {}, dnnl::memory::data_type::u8);
    attr.set_fpmath_mode(dnnl::fpmath_mode::any, true);
    // attr.set_accumulation_mode(dnnl::accumulation_mode::f16);
    dnnl::matmul::primitive_desc matmul_pd(eng, x_desc, weight_desc, output_desc, attr);
    dnnl::matmul matmul_prim(matmul_pd);

    std::unordered_map<int, dnnl::memory> args = {
        {DNNL_ARG_SRC, dnnl::memory(x_desc, eng, x)},
        {DNNL_ARG_WEIGHTS, dnnl::memory(weight_desc, eng, weight)},
        {DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS, dnnl::memory(scale_desc, eng, scales)},
        {DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_WEIGHTS, dnnl::memory(zero_desc, eng, zeros)},
        {DNNL_ARG_DST, dnnl::memory(output_desc, eng, output)}
    };

    if (bias != nullptr) {
        dnnl::memory::desc bias_desc({output_size}, IT, dnnl::memory::format_tag::a);
        args.insert(std::make_pair(DNNL_ARG_BIAS, dnnl::memory(bias_desc, eng, bias)));
    }

    matmul_prim.execute(s, args);
}

torch::Tensor onednn_w4a16(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor scales,
    torch::Tensor zeros,
    std::optional<torch::Tensor> bias
) {
    TORCH_CHECK(x.dim() == 2 && weight.dim() == 2);
    int64_t input_size = x.size(0);
    int64_t state_size = x.size(1);
    int64_t output_size = weight.size(0);
    int64_t block_size = weight.numel() * 2 / scales.numel();

    torch::Tensor output = torch::empty({input_size, output_size},
                                        torch::device(x.device()).dtype(x.dtype()));

    auto func = [=] () {
        switch (x.scalar_type()) {
            case ST::Float: return onednn_w4a16_kernel<dnnl::memory::data_type::f32>;
            case ST::Half: return onednn_w4a16_kernel<dnnl::memory::data_type::f16>;
            case ST::BFloat16: return onednn_w4a16_kernel<dnnl::memory::data_type::bf16>;
            default: throw std::runtime_error("unsupported dtype, only fp32, fp16 and bf16 are supported");
        }
    } ();

    func(
        x.data_ptr(), weight.data_ptr(), scales.data_ptr(), zeros.data_ptr(),
        bias ? bias->data_ptr() : nullptr, output.data_ptr(),
        input_size, state_size, output_size, block_size, x.device()
    );

    return output;
}
