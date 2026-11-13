// Copyright (c) 2026 LightX2V contributors. MIT license, see LICENSE.
#include <torch/extension.h>
#include <c10/core/DeviceGuard.h>
#include <cnrt.h>
#include <framework/core/MLUStream.h>

extern "C" void launch_mesh(int, cnrtQueue_t, void**, int, int, float);

// Private launch entry. Buffer shapes are constructed by mesh.py. The Python
// public API validates vertex/face shapes and indices before constructing them.
void run(int op, const std::vector<torch::Tensor>& tensors, int n, int arg, float threshold) {
    TORCH_CHECK(op >= 0 && op <= 24, "Unknown mesh kernel");
    TORCH_CHECK(n >= 0 && tensors.size() > 0 && tensors.size() <= 6, "Invalid launch");
    static const std::vector<std::string> schemas = {
        "il", "li", "liii", "liib", "lii", "liiii", "iiib", "iiii", "iii", "i",
        "liiiii", "flif", "flif", "liii", "i", "i", "iii", "iii", "ii", "ii",
        "fb", "fi", "ii", "ff", "fif"
    };
    TORCH_CHECK(tensors.size() == schemas[op].size(), "Wrong mesh kernel buffer count");
    const auto device = tensors[0].device();
    TORCH_CHECK(device.type() == c10::DeviceType::PrivateUse1, "MLU tensors required");
    c10::DeviceGuard guard(device);
    void* pointers[6] = {};
    for (size_t i = 0; i < tensors.size(); ++i) {
        TORCH_CHECK(tensors[i].device() == device && tensors[i].is_contiguous(),
                    "Mesh kernel buffers must be contiguous and on the same MLU");
        const auto code = schemas[op][i];
        const auto expected = code == 'i' ? at::kInt : code == 'l' ? at::kLong : code == 'f' ? at::kFloat : at::kByte;
        TORCH_CHECK(tensors[i].scalar_type() == expected, "Wrong mesh kernel buffer dtype");
        pointers[i] = tensors[i].data_ptr();
    }
    if (n == 0) return;
    launch_mesh(op, torch_mlu::getCurrentMLUStream(device.index()).stream(), pointers, n, arg, threshold);
    TORCH_CHECK(cnrtGetLastError() == cnrtSuccess, "MLU mesh kernel launch failed");
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("_run", &run, "Private CuMesh-compatible MLU kernel launcher");
}
