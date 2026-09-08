#include <torch/extension.h>

std::vector<torch::Tensor> h3_qk_rope(
    torch::Tensor query,
    torch::Tensor key,
    torch::Tensor cos,
    torch::Tensor sin);

std::vector<torch::Tensor> h3_qk_rope_fp32(
    torch::Tensor query,
    torch::Tensor key,
    torch::Tensor cos,
    torch::Tensor sin);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("h3_qk_rope", &h3_qk_rope);
  module.def("h3_qk_rope_fp32", &h3_qk_rope_fp32);
}
