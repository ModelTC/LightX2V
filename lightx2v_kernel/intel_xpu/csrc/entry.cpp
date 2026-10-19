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

#include <optional>
#include <tuple>
#include <torch/extension.h>

torch::Tensor onednn_w4a16(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor scales,
    torch::Tensor zeros,
    std::optional<torch::Tensor> bias
);

torch::Tensor onednn_w8a8_int8(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor scales,
    std::optional<torch::Tensor> bias
);

torch::Tensor onednn_w8a16_fp8(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor scales,
    std::optional<torch::Tensor> bias
);

void fp8_cache_clear();
std::tuple<int64_t, int64_t, int64_t> fp8_cache_stats();
std::tuple<int64_t, int64_t, int64_t> fp8_failure_cache_stats();

torch::Tensor sdp_torch(
    torch::Tensor Q,
    torch::Tensor K,
    torch::Tensor V
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("onednn_w4a16", &onednn_w4a16, "onednn w4a16 gemm");
    m.def("onednn_w8a8_int8", &onednn_w8a8_int8,
          "oneDNN dynamically quantized W8A8 INT8 GEMM",
          py::arg("x"), py::arg("weight"), py::arg("scales"),
          py::arg("bias") = py::none());
    m.def("onednn_w8a16_fp8", &onednn_w8a16_fp8,
          "oneDNN W8A16 FP8 per-N-scale GEMM",
          py::arg("x"), py::arg("weight"), py::arg("scales"),
          py::arg("bias") = py::none());
    m.def("fp8_cache_clear", &fp8_cache_clear,
          "Clear cached oneDNN FP8 primitives and counters");
    m.def("fp8_cache_stats", &fp8_cache_stats,
          "Return FP8 primitive cache (hits, misses, size)");
    m.def("fp8_failure_cache_stats", &fp8_failure_cache_stats,
          "Return FP8 failure cache (failures, negative hits, size)");
    m.def("sdp", &sdp_torch,
          "ESIMD Flash Attention SDP [B,L,H,128] PTL-H (fp16/bf16)",
          py::arg("Q"), py::arg("K"), py::arg("V"));
}
