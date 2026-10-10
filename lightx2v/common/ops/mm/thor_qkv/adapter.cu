#include "rowmajor_config.cuh"
#include <cstdio>
#include <memory>

// No padded input or intermediate output: A[M,K] and D[M,N] are exact allocations.
struct Handle {
  Gemm op;
  void* workspace = nullptr;
  size_t workspace_bytes = 0;
  ~Handle() { if (workspace) cudaFree(workspace); }
};

static Gemm::Arguments arguments(void* x, void* weight, void* bias, void* output) {
  Gemm::Arguments args{
    cutlass::gemm::GemmUniversalMode::kGemm, {1785, 5120, 5120, 1},
    {static_cast<cutlass::bfloat16_t*>(x), make_stride(int64_t(5120), _1{}, int64_t(0)),
     static_cast<cutlass::bfloat16_t*>(weight), make_stride(int64_t(5120), _1{}, int64_t(0))},
    {{}, nullptr, make_stride(int64_t(5120), _1{}, int64_t(0)),
     static_cast<cutlass::bfloat16_t*>(output), make_stride(int64_t(5120), _1{}, int64_t(0))}};
  args.epilogue.thread = {{}, {static_cast<cutlass::bfloat16_t*>(bias), cutlass::bfloat16_t(0), {}}, {}};
  return args;
}

extern "C" void* create(void* x, void* weight, void* bias, void* output, void* stream_ptr) {
  auto h = std::make_unique<Handle>();
  auto stream = static_cast<cudaStream_t>(stream_ptr);
  auto args = arguments(x, weight, bias, output);
  auto status = h->op.can_implement(args);
  if (status != cutlass::Status::kSuccess) {
    std::fprintf(stderr, "rowmajor can_implement: %s\n", cutlassGetStatusString(status));
    return nullptr;
  }
  h->workspace_bytes = h->op.get_workspace_size(args);
  if (h->workspace_bytes && cudaMalloc(&h->workspace, h->workspace_bytes) != cudaSuccess)
    return nullptr;
  status = h->op.initialize(args, h->workspace, stream);
  if (status != cutlass::Status::kSuccess) {
    std::fprintf(stderr, "rowmajor initialize: %s\n", cutlassGetStatusString(status));
    return nullptr;
  }
  return h.release();
}

extern "C" void destroy(void* ptr) { delete static_cast<Handle*>(ptr); }

extern "C" int launch_with(void* ptr, void* x, void* weight, void* bias, void* output, void* stream_ptr) {
  auto h = static_cast<Handle*>(ptr);
  auto stream = static_cast<cudaStream_t>(stream_ptr);
  auto args = arguments(x, weight, bias, output);
  auto status = h->op.initialize(args, h->workspace, stream);
  if (status != cutlass::Status::kSuccess) return int(status);
  status = h->op.run(stream);
  if (status != cutlass::Status::kSuccess) return int(status);
  return int(cudaGetLastError());
}
