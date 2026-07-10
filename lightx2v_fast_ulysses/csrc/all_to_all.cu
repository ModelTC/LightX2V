// Copied and adapted from https://github.com/triple-mu/fast-ulysses
#include "a2a_config.cuh"
#include "ulysses_common.cuh"
#include <algorithm>
#include <cuda_runtime.h>
#include <functional>
#include <iomanip>
#include <iostream>

namespace ulysses {

// Each (peer, b, s) unit copies one contiguous n_local*d block (contiguous in both src and dst under
// mode0/mode1): inner (nl,v) is contiguous, so consecutive threads fill a whole block, giving coalesced
// large (typically 4KB) remote bursts. Far better NVLink efficiency than the old scattered writes
// ("write a 256B d-row then jump 4KB"). UNROLL register prefetch pipelines read/write (local reads hidden behind remote
// writes).
template<int WORLD_SIZE, int MODE, int UNROLL, typename Epilogue>
__global__ void a2a_copy_generic(
    const uint8_t* __restrict__ src, PeerPtrs<WORLD_SIZE> peers, Ulysses4DDims dims, int elem_size, Epilogue)
{
    const int     row_bytes = dims.d * elem_size;   // 16B aligned (guaranteed by Global Constraints)
    const int     vecs      = row_bytes >> 4;       // uint4 count per d-row
    const int     blk_vecs  = dims.n_local * vecs;  // uint4 count per contiguous n_local*d block
    const int64_t units     = static_cast<int64_t>(WORLD_SIZE) * dims.b * dims.s_local;
    const int64_t total     = units * blk_vecs;
    const int64_t stride    = static_cast<int64_t>(gridDim.x) * blockDim.x;
    const int64_t tid       = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x;

    for (int64_t base = tid; base < total; base += stride * UNROLL) {
        const uint4* sp[UNROLL];
        uint4*       dp[UNROLL];
        uint4        reg[UNROLL];
        // Read phase (local HBM): prefetch UNROLL uint4s into registers
#pragma unroll
        for (int k = 0; k < UNROLL; ++k) {
            int64_t idx = base + static_cast<int64_t>(k) * stride;
            sp[k]       = nullptr;
            if (idx >= total)
                continue;
            int     inner = static_cast<int>(idx % blk_vecs);  // uint4 offset within block (over nl and d)
            int64_t u     = idx / blk_vecs;                    // unit index over [WS, b, s_local]
            int     s     = static_cast<int>(u % dims.s_local);
            u /= dims.s_local;
            int b_idx = static_cast<int>(u % dims.b);
            u /= dims.b;
            int peer = static_cast<int>(u);  // 0..WS-1

            int64_t src_base_row, dst_base_row;  // block base address in units of d-rows
            if (MODE == 0) {
                src_base_row = (static_cast<int64_t>(b_idx) * dims.s_local + s) * dims.n_global + peer * dims.n_local;
                dst_base_row =
                    (static_cast<int64_t>(b_idx) * dims.s_global + (dims.rank * dims.s_local + s)) * dims.n_local;
            }
            else {
                src_base_row = (static_cast<int64_t>(b_idx) * dims.s_global + (peer * dims.s_local + s)) * dims.n_local;
                dst_base_row =
                    (static_cast<int64_t>(b_idx) * dims.s_local + s) * dims.n_global + dims.rank * dims.n_local;
            }
            sp[k]  = reinterpret_cast<const uint4*>(src + src_base_row * row_bytes) + inner;
            dp[k]  = reinterpret_cast<uint4*>(static_cast<uint8_t*>(peers.p[peer]) + dst_base_row * row_bytes) + inner;
            reg[k] = *sp[k];
        }
        // Write phase (remote NVLink): issued in bulk, local read latency hidden behind it
#pragma unroll
        for (int k = 0; k < UNROLL; ++k)
            if (sp[k])
                *dp[k] = reg[k];
    }
    __threadfence_system();  // system-scope visibility of P2P direct writes to other GPUs
}

template<int WS, int UNROLL>
static void launch_ws_u(const PeerPtrs<WS>&  pp,
                        const uint8_t*       src,
                        const Ulysses4DDims& dims,
                        int                  mode,
                        int                  elem_size,
                        int                  blocks,
                        int                  threads,
                        cudaStream_t         stream)
{
    if (mode == 0)
        a2a_copy_generic<WS, 0, UNROLL, EpilogueIdentity>
            <<<blocks, threads, 0, stream>>>(src, pp, dims, elem_size, EpilogueIdentity{});
    else
        a2a_copy_generic<WS, 1, UNROLL, EpilogueIdentity>
            <<<blocks, threads, 0, stream>>>(src, pp, dims, elem_size, EpilogueIdentity{});
}

template<int WS>
static void launch_ws(const uint8_t*               src,
                      const std::vector<uint64_t>& peer_ptrs,
                      const Ulysses4DDims&         dims,
                      int                          mode,
                      int                          elem_size,
                      int                          blocks,
                      int                          threads,
                      int                          unroll,
                      cudaStream_t                 stream)
{
    PeerPtrs<WS> pp;
    for (int i = 0; i < WS; ++i)
        pp.p[i] = reinterpret_cast<void*>(peer_ptrs[i]);
    // unroll candidates are {4, 8} (resolve_config_nontma); only these two are instantiated.
    if (unroll == 8)
        launch_ws_u<WS, 8>(pp, src, dims, mode, elem_size, blocks, threads, stream);
    else
        launch_ws_u<WS, 4>(pp, src, dims, mode, elem_size, blocks, threads, stream);
}

// Dispatch by ws to the matching launch_ws<WS> (folds the duplicated switch(ws) in launch_a2a /
// resolve_config_nontma). Caller already TORCH_CHECKs ws in [1, 8], so default is a no-op.
static void nontma_dispatch(int                          ws,
                            const uint8_t*               s,
                            const std::vector<uint64_t>& peers,
                            const Ulysses4DDims&         dims,
                            int                          mode,
                            int                          elem,
                            int                          blocks,
                            int                          threads,
                            int                          unroll,
                            cudaStream_t                 stream)
{
    switch (ws) {
        case 1:
            launch_ws<1>(s, peers, dims, mode, elem, blocks, threads, unroll, stream);
            break;
        case 2:
            launch_ws<2>(s, peers, dims, mode, elem, blocks, threads, unroll, stream);
            break;
        case 3:
            launch_ws<3>(s, peers, dims, mode, elem, blocks, threads, unroll, stream);
            break;
        case 4:
            launch_ws<4>(s, peers, dims, mode, elem, blocks, threads, unroll, stream);
            break;
        case 5:
            launch_ws<5>(s, peers, dims, mode, elem, blocks, threads, unroll, stream);
            break;
        case 6:
            launch_ws<6>(s, peers, dims, mode, elem, blocks, threads, unroll, stream);
            break;
        case 7:
            launch_ws<7>(s, peers, dims, mode, elem, blocks, threads, unroll, stream);
            break;
        case 8:
            launch_ws<8>(s, peers, dims, mode, elem, blocks, threads, unroll, stream);
            break;
        default:
            break;
    }
}

// SM count of the current device: queried once per process and cached (one device bound per process).
// Declared in ulysses_common.cuh for the A2A config resolver.
int sm_count_cached()
{
    static const int sm = [] {
        int d = 0, s = 0;
        ULYSSES_CUDA_CHECK(cudaGetDevice(&d));
        ULYSSES_CUDA_CHECK(cudaDeviceGetAttribute(&s, cudaDevAttrMultiProcessorCount, d));
        return s > 0 ? s : 132;  // 132 = H100 SM count, defensive fallback for an unexpected 0
    }();
    return sm;
}

static int clamp_blocks(int64_t needed, int64_t want)
{
    return static_cast<int>(std::max<int64_t>(1, std::min<int64_t>(needed, want)));
}

// Launch the non-TMA direct-write kernel with the given config (blocks/threads/unroll all set by
// resolve_config_nontma). threads/unroll set the number of in-flight remote write transactions (Little's
// law); 512 threads measured to raise NVLink BW from ~210-260 (256 threads) to a steady ~310 across all N
// (close to TMA), hence the default 512.
void launch_a2a(const void*                  src,
                const std::vector<uint64_t>& peer_ptrs,
                const Ulysses4DDims&         dims,
                int                          mode,
                int                          elem_size,
                const A2AConfig&             cfg,
                cudaStream_t                 stream)
{
    const int ws = static_cast<int>(peer_ptrs.size());
    nontma_dispatch(ws,
                    static_cast<const uint8_t*>(src),
                    peer_ptrs,
                    dims,
                    mode,
                    elem_size,
                    cfg.blocks,
                    cfg.threads,
                    cfg.unroll,
                    stream);
}

// Local microbench shared by both resolve_config paths: warmup then time 10 iters, return us/call.
// No collective primitive -- under SPMD all ranks miss the same (shape,mode,tma) on the first call
// and run this concurrently, so the remote writes contend just as in steady state. Correctness is
// guaranteed by the post-launch fast_barrier, not by any timing-side lockstep.
float microbench_us(const std::function<void()>& run_once, cudaStream_t stream)
{
    cudaEvent_t s, e;
    ULYSSES_CUDA_CHECK(cudaEventCreate(&s));
    ULYSSES_CUDA_CHECK(cudaEventCreate(&e));
    for (int i = 0; i < 3; ++i)
        run_once();  // warm up
    ULYSSES_CUDA_CHECK(cudaEventRecord(s, stream));
    for (int i = 0; i < 10; ++i)
        run_once();
    ULYSSES_CUDA_CHECK(cudaEventRecord(e, stream));
    ULYSSES_CUDA_CHECK(cudaEventSynchronize(e));
    float ms = 0.f;
    ULYSSES_CUDA_CHECK(cudaEventElapsedTime(&ms, s, e));
    ULYSSES_CUDA_CHECK(cudaEventDestroy(s));
    ULYSSES_CUDA_CHECK(cudaEventDestroy(e));
    return ms * 100.f;  // ms(10 iters) -> us/call: /10 iters * 1000 (ms->us)
}

// non-TMA config resolution: autotune over the three launch knobs -- threads (in-flight remote writes,
// Little's law), unroll (register-prefetch pipeline depth), and grid size (blocks). Sweeps a converged
// threads x unroll x grid grid by micro-benchmark and keeps the fastest. No cache of its own: the returned
// cfg is held by UlyssesGroup::cfg_cache_ (so the sweep runs once per shape).
A2AConfig resolve_config_nontma(const void*                  src,
                                const std::vector<uint64_t>& peer_ptrs,
                                const Ulysses4DDims&         dims,
                                int                          mode,
                                int                          elem_size,
                                cudaStream_t                 stream,
                                bool                         verbose,
                                const std::function<void()>& finish)
{
    const int      ws        = static_cast<int>(peer_ptrs.size());
    const int64_t  row_bytes = static_cast<int64_t>(dims.d) * elem_size;
    const int      vecs      = static_cast<int>(row_bytes >> 4);
    const int64_t  total     = static_cast<int64_t>(ws) * dims.b * dims.n_local * dims.s_local * vecs;
    const uint8_t* s         = static_cast<const uint8_t*>(src);
    const int      sm        = sm_count_cached();

    // Converged sweep (30 candidates): threads {256,512,1024} x unroll {4,8} x grid factor {8,12,16,24,32}.
    // unroll matches launch_ws_u's compile-time instantiations; the factor spread covers the measured DiT
    // optima (n_local=5 wins at factor 12; the table's non-TMA cells span factors 8..32).
    const int    threads_cand[] = {256, 512, 1024};
    const int    unroll_cand[]  = {4, 8};
    const double factors[]      = {8.0, 12.0, 16.0, 24.0, 32.0};

    A2AConfig best;
    best.threads  = 512;
    best.unroll   = 4;
    best.blocks   = clamp_blocks((total + 511) / 512, static_cast<int64_t>(sm) * 16);
    float best_us = 1e30f;
    for (int threads : threads_cand) {
        const int64_t needed = (total + threads - 1) / threads;
        for (int unroll : unroll_cand) {
            for (double f : factors) {
                const int blocks = clamp_blocks(needed, static_cast<int64_t>(sm * f));
                auto      launch = [&] {
                    nontma_dispatch(ws, s, peer_ptrs, dims, mode, elem_size, blocks, threads, unroll, stream);
                };
                // Probe (bare launch, not timed): large threads x large unroll can exceed the per-block
                // register budget. A failed launch (cudaErrorLaunchOutOfResources) returns synchronously and
                // times near-zero, so it would be falsely picked as fastest; skip it (cudaGetLastError
                // clears the error too).
                launch();
                if (cudaGetLastError() != cudaSuccess)
                    continue;
                // Timed run is the real per-call op (launch + finish=quiet+barrier) so the ranking matches
                // steady state.
                const float us = microbench_us(
                    [&] {
                        launch();
                        finish();
                    },
                    stream);
                if (us < best_us) {
                    best_us      = us;
                    best.threads = threads;
                    best.unroll  = unroll;
                    best.blocks  = blocks;
                }
            }
        }
    }
    // tile_n/tile_s are TMA-only fields; carry default_config's values so the struct stays consistent.
    const A2AConfig dc = default_config(mode, dims.n_local);
    best.tile_n        = dc.tile_n;
    best.tile_s        = dc.tile_s;
    if (verbose && dims.rank == 0) {
        const double per_iter_us  = best_us;  // microbench_us already returns us/call
        const double remote_bytes = static_cast<double>(total) * 16.0 * (ws - 1) / ws;
        const double gbps         = remote_bytes / (per_iter_us * 1e3);
        std::cout << "[ulysses non-TMA tune] ws=" << ws << " mode=" << mode << " n_local=" << dims.n_local
                  << " s_local=" << dims.s_local << " d=" << dims.d << " -> threads=" << best.threads
                  << " unroll=" << best.unroll << " blocks=" << best.blocks << " | " << std::fixed
                  << std::setprecision(1) << per_iter_us << " us/iter " << std::setprecision(0) << gbps << " GB/s"
                  << std::endl;
    }
    return best;
}

}  // namespace ulysses
