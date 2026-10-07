#include <torch/extension.h>
#include <memory>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include "up_gelu.cuh"

struct FfnPlan {
  virtual ~FfnPlan() = default;
  virtual void run_with(at::Tensor x, at::Tensor w, at::Tensor b, at::Tensor y) = 0;
};

using BF = cutlass::bfloat16_t;
void check(cutlass::Status s) {
  TORCH_CHECK(s == cutlass::Status::kSuccess, cutlassGetStatusString(s));
}
void validate(at::Tensor x, at::Tensor w, at::Tensor b, at::Tensor y);

template<class Gemm, class Op>
struct Plan : FfnPlan {
  at::Tensor workspace;
  c10::Device device;
  int64_t rows;
  cudaStream_t stream;
  Gemm op;
  Plan(at::Tensor x_, at::Tensor w_, at::Tensor b_, at::Tensor y_,
       typename Gemm::Arguments args) : device(x_.device()), rows(x_.size(0)) {
    c10::cuda::CUDAGuard guard(device);
    stream = at::cuda::getCurrentCUDAStream();
    workspace = at::empty({int64_t(Gemm::get_workspace_size(args))},
                          x_.options().dtype(at::kByte));
    check(op.can_implement(args));
    check(op.initialize(args,workspace.data_ptr(),at::cuda::getCurrentCUDAStream()));
  }
  void run_with(at::Tensor x_, at::Tensor w_, at::Tensor b_, at::Tensor y_) override {
    validate(x_, w_, b_, y_);
    TORCH_CHECK(x_.device() == device && x_.size(0) == rows, "Plan device/shape mismatch");
    c10::cuda::CUDAGuard guard(device);
    TORCH_CHECK(at::cuda::getCurrentCUDAStream().stream() == stream, "Plan stream mismatch");
    auto args = Op::make_arguments(reinterpret_cast<BF const*>(x_.data_ptr()), reinterpret_cast<BF const*>(w_.data_ptr()), reinterpret_cast<BF const*>(b_.data_ptr()), reinterpret_cast<BF*>(y_.data_ptr()), int(x_.size(0)));
    check(op.update(args, workspace.data_ptr()));
    check(op.run(at::cuda::getCurrentCUDAStream()));
  }
};
template<class Op>
std::shared_ptr<FfnPlan> make(at::Tensor x, at::Tensor w, at::Tensor b, at::Tensor y) {
  return std::make_shared<Plan<typename Op::Gemm, Op>>(x,w,b,y,Op::make_arguments(
      reinterpret_cast<BF const*>(x.data_ptr()), reinterpret_cast<BF const*>(w.data_ptr()),
      reinterpret_cast<BF const*>(b.data_ptr()), reinterpret_cast<BF*>(y.data_ptr()), int(x.size(0))));
}
struct UpGeluRow {
  using Gemm = up1785_gelu_row::Gemm;
  static auto make_arguments(BF const* x, BF const* w, BF const* b, BF* y, int rows) {
    return up1785_gelu_row::make_arguments(x,w,b,y,rows);
  }
};
void validate(at::Tensor x, at::Tensor w, at::Tensor b, at::Tensor y) {
  for (auto const& t : {x,w,b,y}) {
    TORCH_CHECK(t.is_cuda() && t.scalar_type() == at::kBFloat16 &&
                t.device() == x.device(), "Expected same-device CUDA BF16 tensors");
    TORCH_CHECK(!t.requires_grad(), "Inference-only operator");
    TORCH_CHECK(t.is_contiguous(), "Expected contiguous tensors");
    TORCH_CHECK(reinterpret_cast<uintptr_t>(t.data_ptr()) % 16 == 0,
                "16-byte alignment required");
  }
  c10::cuda::CUDAGuard guard(x.device());
  auto prop = at::cuda::getCurrentDeviceProperties();
  TORCH_CHECK(prop->major == 11 && prop->minor == 0, "SM110a build requires Thor");
  TORCH_CHECK(x.dim() == 2 && x.size(0) > 0 && x.size(1) == 5120 &&
              w.sizes() == at::IntArrayRef({13824,5120}) &&
              b.sizes() == at::IntArrayRef({13824}) &&
              y.sizes() == at::IntArrayRef({x.size(0),13824}), "Unsupported up FFN shape");
}

std::shared_ptr<FfnPlan> prepare(at::Tensor x, at::Tensor w, at::Tensor b,
                               at::Tensor y) {
  validate(x, w, b, y);
  return make<UpGeluRow>(x,w,b,y);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  pybind11::class_<FfnPlan, std::shared_ptr<FfnPlan>>(m, "Plan")
      .def("run_with", &FfnPlan::run_with);
  m.def("prepare", &prepare);
}
