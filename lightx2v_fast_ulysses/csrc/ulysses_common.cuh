// Copied and adapted from https://github.com/triple-mu/fast-ulysses
#pragma once
#include <c10/util/Exception.h>
#include <cstdint>
#include <cuda_runtime.h>
#include <functional>
#include <vector>

namespace ulysses {

// Check a CUDA runtime call's status and throw (TORCH_CHECK) on failure, including the call text and the
// driver error string. Use for real (non-autotune-probe) runtime calls; for kernel launches pass
// cudaGetLastError(). The autotune OOR probe deliberately inspects cudaGetLastError() itself, so it does
// NOT use this macro.
#define ULYSSES_CUDA_CHECK(expr)                                                                                       \
    do {                                                                                                               \
        cudaError_t err_ = (expr);                                                                                     \
        TORCH_CHECK(err_ == cudaSuccess, "CUDA error (" #expr "): ", cudaGetErrorString(err_));                        \
    } while (0)

// Shared host/device dim descriptor (int32 fields; kernels use int64 for addressing).
struct Ulysses4DDims {
    int32_t b, s_local, s_global, n_local, n_global, d, rank;
};

struct A2AConfig;  // a2a_config.cuh

template<int WS>
struct PeerPtrs {
    void* p[WS];
};

// A2A copies raw bytes; this no-op keeps the generic copy kernel simple.
struct EpilogueIdentity {
    __device__ __forceinline__ void operator()(uint8_t* /*row_bytes*/, int /*row_off*/) const {}
};

// Free function: pure-CUDA vectorized direct write to peer symmetric memory (host already
// resolved peer_ptrs via nvshmem_ptr). Launch with the given config (no env, no autotune).
void launch_a2a(const void*                  src,
                const std::vector<uint64_t>& peer_ptrs,
                const Ulysses4DDims&         dims,
                int                          mode,
                int                          elem_size,
                const A2AConfig&             cfg,
                cudaStream_t                 stream);

// Local microbench shared by both resolve paths: warmup + time 10 iters, return us/call. Defined in
// all_to_all.cu. run_once is the REAL per-call op (launch + finish, where finish = quiet + fast_barrier),
// so its ranking matches steady-state perf. The fast_barrier inside is hang-safe: under pure-lazy SPMD all
// ranks miss the same (shape,mode,tma) on the first call together, so they call equal barriers in lockstep.
float microbench_us(const std::function<void()>& run_once, cudaStream_t stream);

// SM count of the current device, queried once per process and cached. Defined in all_to_all.cu.
int sm_count_cached();

// non-TMA config resolution: micro-benchmark sweep over threads x unroll x blocks, keep the fastest (result
// held by the group's cfg_cache_). finish: the per-call quiet+barrier, appended to each timed run so the
// measurement reflects real per-call cost. verbose: print the chosen config (rank 0 only).
A2AConfig resolve_config_nontma(const void*                  src,
                                const std::vector<uint64_t>& peer_ptrs,
                                const Ulysses4DDims&         dims,
                                int                          mode,
                                int                          elem_size,
                                cudaStream_t                 stream,
                                bool                         verbose,
                                const std::function<void()>& finish);

// TMA version (fewer SMs, better comm/compute overlap); uniform mode0/mode1. See all_to_all_tma.cu.
// Launch with the given config (build maps + launch only; no team, no autotune, no env).
void launch_a2a_tma(const void*                  src,
                    const std::vector<uint64_t>& peer_ptrs,
                    const Ulysses4DDims&         dims,
                    int                          mode,
                    int                          elem_size,
                    const A2AConfig&             cfg,
                    cudaStream_t                 stream);

// TMA config resolution: pick the best candidate by micro-benchmark (result cached in the group's
// cfg_cache_; a hit skips the micro-benchmark). finish: per-call quiet+barrier appended to each timed run.
// verbose: rank-0 debug printing.
A2AConfig resolve_config_tma(const void*                  src,
                             const std::vector<uint64_t>& peer_ptrs,
                             const Ulysses4DDims&         dims,
                             int                          mode,
                             int                          elem_size,
                             bool                         verbose,
                             cudaStream_t                 stream,
                             const std::function<void()>& finish);

}  // namespace ulysses
