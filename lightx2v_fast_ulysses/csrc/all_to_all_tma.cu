// Copied and adapted from https://github.com/triple-mu/fast-ulysses
// TMA (cp.async.bulk.tensor) A2A: few SMs issue, TMA engine moves data (src gmem->smem->peer gmem).
// One launch covers all peers with many blocks (one tile per block) -> high concurrency saturates NVLink. mode0/mode1
// unified. Mechanism/coordinates verified by tma_p2p_probe / tma_a2a_test. Raw PTX (mbar/TMA/wait_group/fence)
// extracted into named device functions in tma_ptx.cuh, behavior-equivalent.
#include "a2a_config.cuh"
#include "tma_ptx.cuh"
#include "ulysses_common.cuh"
#include <algorithm>
#include <c10/util/Exception.h>
#include <cstdint>
#include <cuda.h>
#include <cuda_runtime.h>
#include <functional>
#include <iomanip>
#include <iostream>
#include <nvshmemx.h>
#include <vector>

namespace ulysses {

struct TmaMaps {
    CUtensorMap m[8];  // per-peer dst tensormap (world_size <= 8)
};

// Global tile index -> (peer, n, s, bi).
__device__ __forceinline__ void
tma_decode(int g, int per_peer, int n_ntiles, int n_stiles, int tile_n, int tile_s, int& peer, int& n, int& s, int& bi)
{
    peer    = g / per_peer;
    int t   = g % per_peer;
    int nt  = t % n_ntiles;
    int tmp = t / n_ntiles;
    int st  = tmp % n_stiles;
    bi      = tmp / n_stiles;
    n       = nt * tile_n;
    s       = st * tile_s;
}

// Software-pipelined: single thread issues, B smem buffers rotate, prefetch-1-ahead + wait_group(B-1)
// keeps B-1 TMA stores in flight -> remote NVLink write pipeline stays full (removes the single-stage serial bubble).
// grid covers the full (peer, b, s-tile, n-tile) set; each block grid-strides over its own run of tiles.
template<int STAGES>
__global__ void a2a_tma_kernel(const __grid_constant__ CUtensorMap src_map,
                               const __grid_constant__ TmaMaps     dst,
                               int                                 ws,
                               int                                 mode,
                               int                                 rank,
                               int                                 s_local,
                               int                                 n_local,
                               int                                 b,
                               int                                 tile_s,
                               int                                 tile_n,
                               uint32_t                            tile_bytes)
{
    if (threadIdx.x != 0)
        return;

    extern __shared__ uint8_t smem_raw[];
    const uint32_t            tb_al = (tile_bytes + 127u) & ~127u;  // 128B-align each buffer
    uintptr_t                 base  = (reinterpret_cast<uintptr_t>(smem_raw) + 127) & ~static_cast<uintptr_t>(127);
    uint64_t*                 mbar  = reinterpret_cast<uint64_t*>(base + (uintptr_t)STAGES * tb_al);

    uint32_t buf_a[STAGES], mbar_a[STAGES];
    int      parity[STAGES];
    for (int k = 0; k < STAGES; ++k) {
        buf_a[k]  = (uint32_t)__cvta_generic_to_shared(reinterpret_cast<void*>(base + (uintptr_t)k * tb_al));
        mbar_a[k] = (uint32_t)__cvta_generic_to_shared(mbar + k);
        parity[k] = 0;
        mbar_init(mbar_a[k]);
    }

    const int n_ntiles = (n_local + tile_n - 1) / tile_n;
    const int n_stiles = (s_local + tile_s - 1) / tile_s;
    const int per_peer = b * n_stiles * n_ntiles;
    const int total    = ws * per_peer;

    // mode0: src head-dim offset peer*n_local / dst seq-dim offset rank*s_local; mode1 swaps them.
    const int src_n_pp  = (mode == 0) ? n_local : 0;  // multiplied by peer
    const int src_s_pp  = (mode == 0) ? 0 : s_local;
    const int dst_n_off = (mode == 0) ? 0 : rank * n_local;
    const int dst_s_off = (mode == 0) ? rank * s_local : 0;

    // Tiles owned by this block (grid-stride run): g = blockIdx.x + j*gridDim.x, M total
    const int M = (total - (int)blockIdx.x + (int)gridDim.x - 1) / (int)gridDim.x;
    if (M <= 0)
        return;

    // prologue: issue loads for the first min(STAGES, M) tiles
    for (int k = 0; k < STAGES && k < M; ++k) {
        int g = blockIdx.x + k * gridDim.x;
        int peer, n, s, bi;
        tma_decode(g, per_peer, n_ntiles, n_stiles, tile_n, tile_s, peer, n, s, bi);
        mbar_arrive_expect(mbar_a[k], tile_bytes);
        tma_load_4d(buf_a[k], &src_map, 0, n + peer * src_n_pp, s + peer * src_s_pp, bi, mbar_a[k]);
    }

    for (int j = 0; j < M; ++j) {
        int cur = j % STAGES;
        int g   = blockIdx.x + j * gridDim.x;
        int peer, n, s, bi;
        tma_decode(g, per_peer, n_ntiles, n_stiles, tile_n, tile_s, peer, n, s, bi);

        // wait for this tile's load to complete
        mbar_wait(mbar_a[cur], parity[cur]);
        parity[cur] ^= 1;
        async_proxy_fence();

        // store this tile -> peer dst
        tma_store_4d(&dst.m[peer], 0, n + dst_n_off, s + dst_s_off, bi, buf_a[cur]);
        tma_commit_group();

        // prefetch next tile: first drain to <= STAGES-1 stores in flight so the target buffer was read by an old store
        int nl = j + STAGES;
        if (nl < M) {
            tma_wait_group<STAGES - 1>();  // keep <=STAGES-1 stores in flight
            int slot = nl % STAGES;
            int g2   = blockIdx.x + nl * gridDim.x;
            int peer2, n2, s2, bi2;
            tma_decode(g2, per_peer, n_ntiles, n_stiles, tile_n, tile_s, peer2, n2, s2, bi2);
            mbar_arrive_expect(mbar_a[slot], tile_bytes);
            tma_load_4d(buf_a[slot], &src_map, 0, n2 + peer2 * src_n_pp, s2 + peer2 * src_s_pp, bi2, mbar_a[slot]);
        }
    }
    tma_wait_group<0>();  // drain all stores before exit (correctness)
}

// 4D tensormap: dims(innermost first)=[d, ndim, sdim, b]; box=[d, tile_n, tile_s, 1].
static CUtensorMap tma_make_map(void* base, int d, int ndim, int sdim, int b, int tile_n, int tile_s, int elem_size)
{
    CUtensorMap         m;
    uint64_t            es       = (uint64_t)elem_size;
    uint64_t            gdims[4] = {(uint64_t)d, (uint64_t)ndim, (uint64_t)sdim, (uint64_t)b};
    uint64_t            gstr[3]  = {(uint64_t)d * es, (uint64_t)ndim * d * es, (uint64_t)sdim * ndim * d * es};
    uint32_t            box[4]   = {(uint32_t)d, (uint32_t)tile_n, (uint32_t)tile_s, 1u};
    uint32_t            estr[4]  = {1, 1, 1, 1};
    CUtensorMapDataType dt       = (elem_size == 2) ? CU_TENSOR_MAP_DATA_TYPE_UINT16 : CU_TENSOR_MAP_DATA_TYPE_UINT8;
    CUresult            r        = cuTensorMapEncodeTiled(&m,
                                        dt,
                                        4,
                                        base,
                                        gdims,
                                        gstr,
                                        box,
                                        estr,
                                        CU_TENSOR_MAP_INTERLEAVE_NONE,
                                        CU_TENSOR_MAP_SWIZZLE_NONE,
                                        CU_TENSOR_MAP_L2_PROMOTION_NONE,
                                        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    TORCH_CHECK(r == CUDA_SUCCESS, "cuTensorMapEncodeTiled failed: ", (int)r);
    return m;
}

// Build src + per-peer dst tensormaps for the uniform path. mode0/mode1 src/dst layouts are dual:
// mode0 src=[b,s_local,n_global,d], dst=[b,s_global,n_local,d]; mode1 swaps n/s in both.
static void build_tma_maps(int                          mode,
                           const Ulysses4DDims&         dims,
                           const void*                  src,
                           const std::vector<uint64_t>& peers,
                           int                          elem,
                           int                          tile_n,
                           int                          tile_s,
                           CUtensorMap&                 src_map,
                           TmaMaps&                     dst)
{
    const int ws = (int)peers.size();
    const int d = dims.d, b = dims.b;
    const int s_local = dims.s_local, n_local = dims.n_local;
    const int s_global = dims.s_global, n_global = dims.n_global;
    if (mode == 0) {
        src_map = tma_make_map((void*)src, d, n_global, s_local, b, tile_n, tile_s, elem);
        for (int p = 0; p < ws; ++p)
            dst.m[p] = tma_make_map((void*)peers[p], d, n_local, s_global, b, tile_n, tile_s, elem);
    }
    else {
        src_map = tma_make_map((void*)src, d, n_local, s_global, b, tile_n, tile_s, elem);
        for (int p = 0; p < ws; ++p)
            dst.m[p] = tma_make_map((void*)peers[p], d, n_global, s_local, b, tile_n, tile_s, elem);
    }
}

// Launch the TMA kernel for a given config (build maps + launch only). mode0/mode1 use different src/dst tensormap
// layouts.
void launch_a2a_tma(const void*                  src,
                    const std::vector<uint64_t>& peer_ptrs,
                    const Ulysses4DDims&         dims,
                    int                          mode,
                    int                          elem_size,
                    const A2AConfig&             cfg,
                    cudaStream_t                 stream)
{
    const int      ws = (int)peer_ptrs.size();
    const int      d = dims.d, b = dims.b, rank = dims.rank;
    const int      s_local = dims.s_local, n_local = dims.n_local;
    const int      tile_n     = std::min({cfg.tile_n, n_local, 256});
    const int      tile_s     = std::min({cfg.tile_s, s_local, 256});
    constexpr int  stages     = 4;  // fixed (measured DiT optimum; keeps tma_wait_group at {3,0})
    const uint32_t tile_bytes = (uint32_t)tile_s * tile_n * d * elem_size;
    const uint32_t tb_al      = (tile_bytes + 127u) & ~127u;
    const int      smem       = (int)(tb_al * (uint32_t)stages) + 128 + 8 * stages;

    static bool attr_set = false;
    if (!attr_set) {
        ULYSSES_CUDA_CHECK(
            cudaFuncSetAttribute(a2a_tma_kernel<stages>, cudaFuncAttributeMaxDynamicSharedMemorySize, 200 * 1024));
        attr_set = true;
    }

    const int     n_stiles = (s_local + tile_s - 1) / tile_s;
    const int     n_ntiles = (n_local + tile_n - 1) / tile_n;
    const int     total    = ws * b * n_stiles * n_ntiles;
    constexpr int bdiv     = 4;  // fixed (measured DiT optimum): more tiles per block -> enables pipelining
    const int     blocks   = std::max(std::min(total, 65535) / bdiv, 1);

    CUtensorMap src_map;
    TmaMaps     dst;
    build_tma_maps(mode, dims, src, peer_ptrs, elem_size, tile_n, tile_s, src_map, dst);
    a2a_tma_kernel<stages>
        <<<blocks, 1, smem, stream>>>(src_map, dst, ws, mode, rank, s_local, n_local, b, tile_s, tile_n, tile_bytes);
}

// Candidate configs: cover small N (small tile_n, shallow pipeline) to large N (large tile_n, deep pipeline).
// In mode0 a non-divisor tile_n (e.g. n_local-1) often beats a whole block -- it forms multiple n-tiles for
// higher TMA concurrency (a single whole-block tile is actually slower). Seed includes default_config; rest
// are empirically tuned.
//
// CORRECTNESS GUARD: the dst dim offset by this rank (mode0: s at rank*s_local; mode1: n at rank*n_local) is
// tiled over the rank's chunk, but a TMA store only clips at the dst's GLOBAL dim, not at this rank's
// sub-segment. So if that tile does not divide the chunk, the trailing tile overruns into the neighbor rank's
// region (corruption; and during autotune the cross-rank overrun writes race). Require it to divide: mode0
// needs s_local % tile_s == 0 (tile_s is 1 here, always ok), mode1 needs n_local % tile_n == 0. The other dim
// spans the rank's full chunk == the dst global dim, so its trailing tile is clipped correctly.
static std::vector<A2AConfig> tma_candidates(int mode, int n_local, int s_local)
{
    std::vector<A2AConfig> v;
    auto                   add = [&](int tn, int ts) {
        tn = std::max(1, std::min(tn, n_local));
        ts = std::max(1, std::min(ts, s_local));
        if (mode == 0 ? (s_local % ts != 0) : (n_local % tn != 0))
            return;  // would overrun the neighbor rank's region (see CORRECTNESS GUARD above)
        for (auto& c : v)
            if (c.tile_n == tn && c.tile_s == ts)
                return;
        A2AConfig c{};
        c.tile_n = tn;
        c.tile_s = ts;
        v.push_back(c);
    };
    // Converged set (stages/bdiv fixed at 4): vary tile_n {default, n_local-1, n_local/2, n_local, 8} and,
    // for mode1, also tile_s {1,2}. The overrun guard in add() drops non-dividing tile_n for mode1.
    const A2AConfig def = default_config(mode, n_local);  // seed: default config enters candidates first
    add(def.tile_n, def.tile_s);
    const int nl1 = std::max(1, n_local - 1), nlh = std::max(1, n_local / 2);
    const int ts_extra = (mode == 0) ? 1 : 2;  // mode1 also probes tile_s=2 (whole-s clips, always safe)
    add(nl1, 1);
    add(nlh, 1);
    add(n_local, 1);
    add(8, 1);
    add(nl1, ts_extra);
    add(nlh, ts_extra);
    return v;
}

// autotune: microbench all candidates (local timing via the shared microbench_us), return the fastest
// config (remote writes are real but overwritten by the subsequent final launch, so correctness is
// unaffected). No collective primitive of its own -- under SPMD all ranks miss the same (shape,mode)
// on the first call and run this concurrently (contention captured); the cache lives in the caller
// (UlyssesGroup::cfg_cache_). verbose: rank-0 debug print.
A2AConfig resolve_config_tma(const void*                  src,
                             const std::vector<uint64_t>& peer_ptrs,
                             const Ulysses4DDims&         dims,
                             int                          mode,
                             int                          elem_size,
                             bool                         verbose,
                             cudaStream_t                 stream,
                             const std::function<void()>& finish)
{
    const int  n_local = dims.n_local, s_local = dims.s_local, ws = (int)peer_ptrs.size();
    const auto cands = tma_candidates(mode, n_local, s_local);

    A2AConfig best   = cands[0];
    float     best_t = 1e30f;
    for (const auto& c : cands) {
        // Timed run is the real per-call op (launch + finish=quiet+barrier) so the ranking matches steady
        // state (a launch-only microbench mis-ranks tile_n: a smaller tile_n that loses on raw launch time
        // wins once the cross-rank barrier serializes per-iteration contention).
        float us = microbench_us(
            [&] {
                launch_a2a_tma(src, peer_ptrs, dims, mode, elem_size, c, stream);
                finish();
            },
            stream);
        if (verbose)
            std::cerr << "[tma-at] ws=" << ws << " mode=" << mode << " nl=" << n_local << " sl=" << s_local
                      << " | tn=" << c.tile_n << " ts=" << c.tile_s << " -> " << std::fixed << std::setprecision(1)
                      << us << " us/call" << std::endl;
        if (us < best_t) {
            best_t = us;
            best   = c;
        }
    }
    return best;
}

}  // namespace ulysses
