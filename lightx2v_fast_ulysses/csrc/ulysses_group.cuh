// Copied and adapted from https://github.com/triple-mu/fast-ulysses
#pragma once
#include "a2a_config.cuh"
#include "symmetric_pool.cuh"
#include <cstdint>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <map>
#include <memory>
#include <nvshmem.h>
#include <torch/custom_class.h>
#include <vector>

namespace ulysses {

class UlyssesGroup: public torch::CustomClassHolder {
public:
    static int64_t              uniqueid_nints();  // ceil(sizeof(nvshmemx_uniqueid_t)/8)
    static std::vector<int64_t> get_uniqueid();    // rank0 only
    static void init_world(std::vector<int64_t> uid_ints, int64_t rank, int64_t nranks);  // idempotent

    UlyssesGroup(std::vector<int64_t> peer_global_pes, int64_t my_rank, int64_t device_id, int64_t reserved_bytes);
    ~UlyssesGroup() override;

    int64_t rank() const
    {
        return my_rank_;
    }
    int64_t world_size() const
    {
        return world_size_;
    }
    void destroy();

    SymmetricHeapPool& pool()
    {
        return *pool_;
    }
    nvshmem_team_t team() const
    {
        return team_;
    }

    // Device compute capability major (cached via cudaDeviceGetAttribute at construction). resolve_config uses
    // it for the auto path choice (sm<9 has no TMA).
    int sm_major() const
    {
        return sm_major_;
    }

    struct PathConfig {
        bool      tma;
        A2AConfig cfg;
    };

    // Resolve the launch path + config for (dims,mode). use_tma_i is the tri-state -1 auto / 0 non-TMA /
    // 1 TMA (the sm<9 error for an explicit 1 is enforced by the caller). Returns the path to launch and
    // its config; a cache hit returns directly.
    //
    // Explicit 0/1: micro-benchmark that path's candidates, keep the fastest. Auto (-1): on sm<9 -> non-TMA;
    // on sm90+ micro-benchmark BOTH paths and pick the faster (the runtime replacement of the old static DiT
    // table -- "tune the best path+config and cache it"), memoised in auto_path_cache_.
    //
    // The microbench times the REAL per-call op (launch + quiet + fast_barrier) so its ranking matches steady
    // state, and the two paths' times are directly comparable. The fast_barrier makes the miss branch
    // cross-rank, but it is HANG-SAFE: pure-lazy SPMD (no tune()) means all ranks issue the same
    // (shape,mode,use_tma) sequence and miss the same entry on the first call together -> equal barrier calls
    // in lockstep (the candidate count is a function of rank-invariant dims, so it is identical on every rank).
    // A cache-hit rank and a miss rank never coexist for the same call, so no rank blocks alone.
    //
    // The auto path choice is a per-rank local timing decision, so on a near-tie shape different ranks may pick
    // different kernels. This is harmless: the two paths are functionally equivalent P2P writes (each rank
    // writes its own region correctly either way), and per-call barrier counts stay equal regardless of path.
    //
    // Thread safety: resolve_config/all_to_all on the same group instance must be called serially (SPMD
    // single-threaded; the caches are lock-free, so concurrent multi-stream use is not thread-safe).
    PathConfig resolve_config(const Ulysses4DDims&         dims,
                              int                          mode,
                              int                          use_tma_i,
                              const void*                  src,
                              const std::vector<uint64_t>& peers,
                              int                          elem,
                              cudaStream_t                 stream);

    // Custom single-node NVLink flag barrier: replaces the slow nvshmem sync (~280us) that falls back on
    // hardware without NVLS fabric. Call nvshmemx_quiet_on_stream first (so this rank's writes are globally
    // visible). No-op when world_size==1.
    void fast_barrier(cudaStream_t stream);

private:
    int                                my_rank_, world_size_, device_id_;
    int                                sm_major_ = 0;  // cached cudaDeviceGetAttribute(major) at construction
    std::vector<int>                   peer_global_pes_;
    nvshmem_team_t                     team_;
    bool                               owns_team_ = false;
    bool                               destroyed_ = false;
    std::unique_ptr<SymmetricHeapPool> pool_;

    // cfg_cache_ holds the best config per (ws,mode,tma,n_local,s_local,d). Lock-free std::map,
    // must be accessed serially (see resolve_config comment).
    std::map<ConfigKey, A2AConfig> cfg_cache_;
    // auto_path_cache_ memoises the best path for the auto (use_tma=None) case per (mode,n_local,s_local,d)
    // -- true=TMA -- so a repeat auto call skips the two-path micro-benchmark.
    std::map<std::tuple<int, int, int, int>, bool> auto_path_cache_;

    // fast_barrier state: symmetric flag buffer (uint64[ws]) + monotonic epoch (incremented lockstep per rank).
    bool                  bar_ready_ = false;
    uint64_t              bar_epoch_ = 0;
    void*                 bar_local_ = nullptr;  // this rank's flag base
    std::vector<uint64_t> bar_peers_;            // per-peer flag base (including self)
};

}  // namespace ulysses
