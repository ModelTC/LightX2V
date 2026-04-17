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

// FP16 activations × FP8_E4M3 weights GEMM via OneDNN
//
// Layout
//   A (activations) : [M, K]  FP16,       format_tag::ab
//   B (weights)     : [N, K]  FP8_E4M3,   stored physically as [N,K],
//                             logical [K,N] → format_tag::ba
//   scale           : [N]     FP32,        per-output-channel
//   C (output)      : [M, N]  FP16,       format_tag::ab
//
// Per-N scale → mask = 2  (bit-1 of logical B dims [K, N])
//
// On PTL this hits the optimised jit:gemm:any kernel and reaches ~80 % of
// the 55 TFLOPS FP16 peak.  BF16 / FP32 inputs fall back to a slow
// reference kernel; the function prints a warning in that case.

#include <torch/extension.h>
#include <oneapi/dnnl/dnnl.hpp>
#include <oneapi/dnnl/dnnl_sycl.hpp>

#include "utils.h"

using ST = torch::ScalarType;

// ── internal kernel ──────────────────────────────────────────────────────────

template <dnnl::memory::data_type IT>   // IT = f16 | bf16 | f32
static void onednn_w8a16_fp8_impl(
    void*  x,
    void*  weight,
    void*  scales,
    void*  bias,
    void*  output,
    int64_t M,
    int64_t K,
    int64_t N,
    const torch::Device& device
) {
    sycl::queue& q = utils::get_queue(device);

    dnnl::engine eng = dnnl::sycl_interop::make_engine(q.get_device(), q.get_context());
    dnnl::stream  s  = dnnl::sycl_interop::make_stream(eng, q);

    // A [M, K]
    dnnl::memory::desc x_md(
        {M, K}, IT, dnnl::memory::format_tag::ab);

    // B logical [K, N], physical [N, K] → format_tag::ba
    dnnl::memory::desc w_md(
        {K, N}, dnnl::memory::data_type::f8_e4m3, dnnl::memory::format_tag::ba);

    // scale [N] FP32 – provided per execution
    dnnl::memory::desc scale_md(
        {N}, dnnl::memory::data_type::f32, dnnl::memory::format_tag::a);

    // C [M, N]
    dnnl::memory::desc c_md(
        {M, N}, IT, dnnl::memory::format_tag::ab);

    dnnl::primitive_attr attr;
    // mask = 2: per dim-1 of logical B[K,N] = per output-channel (per-N)
    // set_scales_mask uses the implicit f32 scale path that the FP8 JIT kernel
    // (jit:gemm:any) requires on PTL; set_scales with explicit f32 dtype
    // unexpectedly forces the slow ocl:ref:any fallback.
    attr.set_scales_mask(DNNL_ARG_WEIGHTS, 2);
    attr.set_fpmath_mode(dnnl::fpmath_mode::any, /*apply_to_int=*/true);

    dnnl::matmul::primitive_desc pd(eng, x_md, w_md, c_md, attr);

    // Detect reference fallback so the caller isn't surprised by slow perf
    std::string impl = pd.impl_info_str();
    if (impl.find("ref") != std::string::npos) {
        printf("[onednn_w8a16_fp8] WARNING: slow reference impl selected: %s\n"
               "                   Only FP16×FP8 has an optimised JIT kernel "
               "on PTL.\n", impl.c_str());
    }

    dnnl::matmul prim(pd);

    std::unordered_map<int, dnnl::memory> args = {
        {DNNL_ARG_SRC,                            dnnl::memory(x_md,     eng, x)},
        {DNNL_ARG_WEIGHTS,                        dnnl::memory(w_md,     eng, weight)},
        {DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS, dnnl::memory(scale_md, eng, scales)},
        {DNNL_ARG_DST,                            dnnl::memory(c_md,     eng, output)},
    };

    if (bias) {
        // Bias dtype matches I/O dtype
        dnnl::memory::desc bias_md({N}, IT, dnnl::memory::format_tag::a);
        args.insert({DNNL_ARG_BIAS, dnnl::memory(bias_md, eng, bias)});
    }

    prim.execute(s, args);
}

// ── public entry point ───────────────────────────────────────────────────────

// onednn_w8a16_fp8(x, weight, scales, bias=None) → output
//
//   x      : [M, K]  FP16 / BF16 / FP32  (XPU)
//   weight : [N, K]  torch.float8_e4m3fn  (XPU)
//   scales : [N]     FP32                 (XPU)
//   bias   : [N]     same dtype as x      (XPU, optional)
//   returns: [M, N]  same dtype as x

torch::Tensor onednn_w8a16_fp8(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor scales,
    std::optional<torch::Tensor> bias
) {
    TORCH_CHECK(x.dim() == 2 && weight.dim() == 2,
                "x and weight must be 2-D");
    TORCH_CHECK(scales.scalar_type() == torch::kFloat,
                "scales must be FP32");

    const int64_t M = x.size(0);
    const int64_t K = x.size(1);
    const int64_t N = weight.size(0);

    TORCH_CHECK(weight.size(1) == K,
                "weight K-dim (", weight.size(1), ") != x K-dim (", K, ")");
    TORCH_CHECK(scales.numel() == N,
                "scales must have N=", N, " elements, got ", scales.numel());

    torch::Tensor out = torch::empty(
        {M, N}, torch::device(x.device()).dtype(x.dtype()));

    using DT = dnnl::memory::data_type;
    auto dispatch = [&](auto fn) {
        fn(x.data_ptr(), weight.data_ptr(), scales.data_ptr(),
           bias ? bias->data_ptr() : nullptr,
           out.data_ptr(), M, K, N, x.device());
    };

    switch (x.scalar_type()) {
        case ST::Half:
            dispatch(onednn_w8a16_fp8_impl<DT::f16>); break;
        case ST::BFloat16:
            dispatch(onednn_w8a16_fp8_impl<DT::bf16>); break;
        case ST::Float:
            dispatch(onednn_w8a16_fp8_impl<DT::f32>); break;
        default:
            TORCH_CHECK(false,
                "unsupported dtype: only FP16, BF16, FP32 are supported");
    }

    return out;
}
