#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include "down_dram_opt.cuh"

// Selected v1_dram_opt wave order; no experimental environment overrides.
static const int kSchedule[70] = {1024,1025,1026,1027,1028,1280,1281,1284,1283,1282,65536,65537,65538,65539,65540,65792,65793,65794,65795,65796,512,513,514,515,516,768,769,771,770,772,67072,67073,67074,67075,67076,67328,67329,67330,67331,67332,2048,2049,2050,2051,2052,2304,2305,2306,2307,2309,67846,67590,67589,67844,67333,67334,67077,67078,66821,66822,1029,1030,773,774,517,518,261,262,5,6};

// The same tile order with every tile accumulating K in ascending order (bit 16 cleared), which makes the result
// bitwise identical to a forward-K GEMM such as cuBLAS addmm / CUTLASS. Used for row counts other than the
// 1785-row production shape, whose callers previously ran cuBLAS.
static bool forward_k_schedule(int* out) {
  for (int i = 0; i < 70; ++i) out[i] = kSchedule[i] & 0xffff;
  return true;
}

class DownPlan {
  at::Tensor schedule_;
  Handle handle_;
  int device_;
  cudaStream_t stream_;
 public:
  explicit DownPlan(at::Tensor x, bool forward_k = false) : device_(x.get_device()) {
    c10::cuda::CUDAGuard guard(x.device());
    stream_ = at::cuda::getCurrentCUDAStream(device_).stream();
    static int fwd[70];
    static bool init = forward_k_schedule(fwd);
    (void)init;
    schedule_ = at::empty({70}, x.options().dtype(at::kInt));
    TORCH_CHECK(cudaMemcpyAsync(schedule_.data_ptr(), forward_k ? fwd : kSchedule, sizeof(kSchedule), cudaMemcpyHostToDevice, stream_) == cudaSuccess, "down schedule upload failed");
    TORCH_CHECK(cudaStreamSynchronize(stream_) == cudaSuccess, "down schedule upload failed");
    handle_.dtable = schedule_.data_ptr<int>();
  }
  void run_with(at::Tensor x, at::Tensor w, at::Tensor b, at::Tensor y) {
    for (auto const& t : {x, w, b, y}) {
      TORCH_CHECK(t.is_cuda() && t.get_device() == device_ && t.scalar_type() == at::kBFloat16 && t.is_contiguous(), "down requires contiguous BF16 tensors on the plan device");
      TORCH_CHECK(!t.requires_grad(), "down supports inference only");
      TORCH_CHECK(reinterpret_cast<uintptr_t>(t.data_ptr()) % 16 == 0, "down requires 16-byte alignment");
    }
    const int64_t rows = x.size(0);
    // The 70-tile schedule covers 7 row tiles of 256: 1536 < rows <= 1792.
    TORCH_CHECK(x.dim() == 2 && rows > 1536 && rows <= 1792 && x.size(1) == 13824 && w.sizes() == at::IntArrayRef({5120,13824}) && b.sizes() == at::IntArrayRef({5120}) && y.sizes() == at::IntArrayRef({rows,5120}), "unsupported down shape");
    c10::cuda::CUDAGuard guard(x.device());
    TORCH_CHECK(at::cuda::getCurrentCUDAStream(device_).stream() == stream_, "down plan stream changed");
    auto prop = at::cuda::getDeviceProperties(device_);
    TORCH_CHECK(prop->major == 11 && prop->minor == 0, "down requires Thor SM110");
    Sched sp{handle_.dtable, hint_value(0), hint_value(1)};
    TORCH_CHECK(make_plan(&handle_, reinterpret_cast<TA const*>(x.data_ptr()), reinterpret_cast<TA const*>(w.data_ptr()), reinterpret_cast<TA const*>(b.data_ptr()), reinterpret_cast<TA*>(y.data_ptr()), int(rows),5120,13824,sp), "down TMA plan failed");
    TORCH_CHECK(handle_.run(stream_) == cudaSuccess, "down kernel launch failed");
  }
};
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  pybind11::class_<DownPlan, std::shared_ptr<DownPlan>>(m, "DownPlan")
    .def(pybind11::init<at::Tensor, bool>(), pybind11::arg("x"), pybind11::arg("forward_k") = false)
    .def("run_with", &DownPlan::run_with);
}
