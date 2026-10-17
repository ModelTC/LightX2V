#include "rowmajor_config.cuh"
#include <cstdio>
#include <cuda_runtime.h>
#include <map>
#include <memory>
#include <mutex>
#include <utility>

// D[M,N] = X[M,K] @ W[N,K]^T + bias[N] (BF16 in/out, FP32 accumulate, fused bias).
// A[M,K], W[N,K] and D[M,N] are exact row-major allocations (D row pitch ldd >= N); any M >= 1.
// One adapter object per (device, stream) is reused for every launch: only the TMA descriptors
// and the per-launch parameters are rebuilt by initialize(), which is host-only work. Keying by
// stream keeps concurrent streams off each other's workspace; the workspace is allocated on the
// device that owns the key.
namespace {
struct State {
  Gemm op;
  void* workspace = nullptr;
  size_t workspace_bytes = 0;
};

std::mutex g_mutex;  // ctypes releases the GIL, so launches can arrive from several threads.
std::map<std::pair<int, cudaStream_t>, std::unique_ptr<State>> g_states;

Gemm::Arguments arguments(void* x, void* weight, void* bias, void* output, int m, int n, int k, int64_t ldd) {
  Gemm::Arguments args{
    cutlass::gemm::GemmUniversalMode::kGemm, {m, n, k, 1},
    {static_cast<cutlass::bfloat16_t*>(x), make_stride(int64_t(k), _1{}, int64_t(0)),
     static_cast<cutlass::bfloat16_t*>(weight), make_stride(int64_t(k), _1{}, int64_t(0))},
    {{}, nullptr, make_stride(ldd, _1{}, int64_t(0)),
     static_cast<cutlass::bfloat16_t*>(output), make_stride(ldd, _1{}, int64_t(0))}};
  args.epilogue.thread = {{}, {static_cast<cutlass::bfloat16_t*>(bias), cutlass::bfloat16_t(0), {}}, {}};
  return args;
}
}  // namespace

// Returns 0 on success, a cutlass::Status (+1000 can_implement, +2000 initialize, +3000 run) or a cudaError_t.
// The caller must make the tensors' device current (the Python side wraps the call in torch.cuda.device).
extern "C" int thor_linear_bias(void* x, void* weight, void* bias, void* output, int m, int n, int k, long long ldd,
                                void* stream_ptr) {
  auto stream = static_cast<cudaStream_t>(stream_ptr);
  int device = 0;
  cudaError_t err = cudaGetDevice(&device);
  if (err != cudaSuccess) return int(err);
  std::lock_guard<std::mutex> lock(g_mutex);
  auto& slot = g_states[{device, stream}];
  if (!slot) slot = std::make_unique<State>();
  State& s = *slot;
  auto args = arguments(x, weight, bias, output, m, n, k, ldd);
  auto status = s.op.can_implement(args);
  if (status != cutlass::Status::kSuccess) return 1000 + int(status);
  size_t bytes = Gemm::get_workspace_size(args);
  if (bytes > s.workspace_bytes) {
    if (s.workspace) {
      // Earlier launches on this stream may still read the old workspace.
      err = cudaStreamSynchronize(stream);
      if (err != cudaSuccess) return int(err);
      cudaFree(s.workspace);
    }
    s.workspace = nullptr;
    s.workspace_bytes = 0;
    if (cudaMalloc(&s.workspace, bytes) != cudaSuccess) return int(cudaErrorMemoryAllocation);
    s.workspace_bytes = bytes;
  }
  status = s.op.initialize(args, s.workspace, stream);
  if (status != cutlass::Status::kSuccess) return 2000 + int(status);
  status = s.op.run(stream);
  if (status != cutlass::Status::kSuccess) return 3000 + int(status);
  return int(cudaGetLastError());
}
