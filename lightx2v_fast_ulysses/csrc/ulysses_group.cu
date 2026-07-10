// Copied and adapted from https://github.com/triple-mu/fast-ulysses
#include "tma_ptx.cuh"
#include "ulysses_group.cuh"
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cstring>
#include <cuda_runtime.h>
#include <nvshmem.h>
#include <nvshmemx.h>
#include <torch/torch.h>

namespace ulysses {

static bool g_world_inited = false;
static int  g_live_groups  = 0;

// ---- Custom NVLink flag barrier ----
// Flag layout: each rank holds uint64 flags[ws]. On arrival, rank r writes epoch into every peer p's
// flags[r] (P2P, release/sys), then spins until its own flags[0..ws-1] all == epoch (acquire/sys).
// Monotonically increasing epoch means no reset and no ABA. Strict lockstep (SPMD collective) keeps
// epoch identical across ranks throughout.
struct BarPeers {
    uint64_t p[8];
};

__global__ void ulysses_barrier_kernel(uint64_t* local, BarPeers peers, int ws, int rank, uint64_t epoch)
{
    int t = threadIdx.x;
    if (t >= ws)
        return;
    uint64_t* remote = reinterpret_cast<uint64_t*>(peers.p[t]) + rank;  // peer t's flags[rank]
    st_release_sys_u64(remote, epoch);
    uint64_t  v;
    uint64_t* mine = local + t;  // own flags[t] (written by peer t)
    do {
        v = ld_acquire_sys_u64(mine);
    } while (v < epoch);
}

void UlyssesGroup::fast_barrier(cudaStream_t stream)
{
    if (world_size_ == 1)
        return;
    if (!bar_ready_) {
        const auto& buf = pool_->acquire({static_cast<int64_t>(world_size_)}, at::kLong, "__ulysses_sync__");
        bar_local_      = buf.sym_base;
        bar_peers_      = buf.peer_ptrs;
        ULYSSES_CUDA_CHECK(
            cudaMemsetAsync(bar_local_, 0, world_size_ * sizeof(uint64_t), stream));  // init 0; epoch starts at 1
        // One slow sync: ensure all ranks finish clearing before anyone writes a flag (otherwise the
        // clear could overwrite an already-written epoch).
        nvshmemx_barrier_on_stream(team_, stream);
        bar_ready_ = true;
    }
    ++bar_epoch_;
    BarPeers peers;
    for (int i = 0; i < world_size_; ++i)
        peers.p[i] = bar_peers_[i];
    ulysses_barrier_kernel<<<1, 32, 0, stream>>>(
        reinterpret_cast<uint64_t*>(bar_local_), peers, world_size_, my_rank_, bar_epoch_);
    ULYSSES_CUDA_CHECK(cudaGetLastError());  // catch a barrier-kernel launch failure
}

int64_t UlyssesGroup::uniqueid_nints()
{
    return static_cast<int64_t>((sizeof(nvshmemx_uniqueid_t) + 7) / 8);
}

std::vector<int64_t> UlyssesGroup::get_uniqueid()
{
    nvshmemx_uniqueid_t uid;
    std::memset(&uid, 0, sizeof(uid));
    TORCH_CHECK(nvshmemx_get_uniqueid(&uid) == 0, "nvshmemx_get_uniqueid failed");
    std::vector<int64_t> out(uniqueid_nints(), 0);
    std::memcpy(out.data(), &uid, sizeof(uid));
    return out;
}

void UlyssesGroup::init_world(std::vector<int64_t> uid_ints, int64_t rank, int64_t nranks)
{
    if (g_world_inited)
        return;
    TORCH_CHECK(static_cast<int64_t>(uid_ints.size()) >= uniqueid_nints(), "uid_ints too short");
    nvshmemx_uniqueid_t uid;
    std::memcpy(&uid, uid_ints.data(), sizeof(uid));
    // Use INITIALIZER, not memset(0): it stamps the version field of attr/args/uid_args.
    // nvshmemx_set_attr_uniqueid_args does not write version, and hostlib_init_attr dispatches the V2
    // path based on attr.args.version, so the version must be stamped first (inline nvshmemx_init_attr
    // auto-stamps when version is invalid; here we explicitly substitute that step).
    nvshmemx_init_attr_t attr = NVSHMEMX_INIT_ATTR_INITIALIZER;
    TORCH_CHECK(
        nvshmemx_set_attr_uniqueid_args(static_cast<int>(rank), static_cast<int>(nranks), &uid, &attr) == 0,
        "nvshmemx_set_attr_uniqueid_args failed");
    // DEVIATION (see task-5-report): use the host-lib direct entry nvshmemx_hostlib_init_attr instead of
    // inline nvshmemx_init_attr. The inline version calls nvshmemi_init_thread, a symbol that lives only
    // in static libnvshmem_device.a; linking it clashes with the NVSHMEM version node of torch's bundled
    // libtorch_nvshmem.so (undefined symbol nvshmem_selected_device_transport). hostlib_init_attr is the
    // equivalent entry exported directly by the host shared library (NVSHMEM's own python UID path uses it).
    TORCH_CHECK(nvshmemx_hostlib_init_attr(NVSHMEMX_INIT_WITH_UNIQUEID, &attr) == 0,
                "nvshmemx_hostlib_init_attr failed");
    g_world_inited = true;
}

UlyssesGroup::UlyssesGroup(std::vector<int64_t> peer_global_pes,
                           int64_t              my_rank,
                           int64_t              device_id,
                           int64_t              reserved_bytes):
    my_rank_(static_cast<int>(my_rank)),
    world_size_(static_cast<int>(peer_global_pes.size())),
    device_id_(static_cast<int>(device_id))
{
    TORCH_CHECK(g_world_inited, "init_world must be called before constructing UlyssesGroup");
    // Cache compute capability major (TMA cp.async.bulk.tensor is Hopper(sm90)+ only); whether a call
    // actually takes the TMA path is decided per call by resolve_config (explicit use_tma, or auto-measured).
    int major = 0;
    ULYSSES_CUDA_CHECK(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device_id_));
    sm_major_ = major;
    TORCH_CHECK(
        world_size_ >= 1 && world_size_ <= 8, "world_size must be in [1, 8] (single-node NVLink), got ", world_size_);
    ULYSSES_CUDA_CHECK(cudaSetDevice(device_id_));
    peer_global_pes_.reserve(world_size_);
    for (auto pe : peer_global_pes)
        peer_global_pes_.push_back(static_cast<int>(pe));

    // team: if the group covers the whole world contiguously -> TEAM_WORLD; else a contiguous stride-1
    // subgroup via split_strided.
    const int gpes          = nvshmem_n_pes();
    bool      is_full_world = (world_size_ == gpes);
    for (int i = 0; i < world_size_ && is_full_world; ++i)
        if (peer_global_pes_[i] != i)
            is_full_world = false;
    if (is_full_world) {
        team_      = NVSHMEM_TEAM_WORLD;
        owns_team_ = false;
    }
    else {
        int start = peer_global_pes_[0];
        for (int i = 1; i < world_size_; ++i)
            TORCH_CHECK(peer_global_pes_[i] == start + i, "phase-1 only supports a contiguous PE subgroup");
        nvshmem_team_config_t cfg;
        std::memset(&cfg, 0, sizeof(cfg));
        TORCH_CHECK(nvshmem_team_split_strided(NVSHMEM_TEAM_WORLD, start, 1, world_size_, &cfg, 0, &team_) == 0,
                    "nvshmem_team_split_strided failed");
        owns_team_ = true;
    }

    pool_ = std::make_unique<SymmetricHeapPool>(reserved_bytes, world_size_, peer_global_pes_);
    ++g_live_groups;
}

// See the resolve_config / hang-safety notes in ulysses_group.cuh. finish is the real per-call tail (quiet +
// fast_barrier); every micro-benchmark times launch+finish so its ranking matches steady state and the two
// paths' times are directly comparable.
UlyssesGroup::PathConfig UlyssesGroup::resolve_config(const Ulysses4DDims&         dims,
                                                      int                          mode,
                                                      int                          use_tma_i,
                                                      const void*                  src,
                                                      const std::vector<uint64_t>& peers,
                                                      int                          elem,
                                                      cudaStream_t                 stream)
{
    auto finish = [this, stream] {
        nvshmemx_quiet_on_stream(stream);
        fast_barrier(stream);
    };
    // Autotune the given path (cached per (ws,mode,tma,dims)).
    auto resolve_path = [&](bool tma) -> A2AConfig {
        const ConfigKey key = config_key(world_size_, mode, tma, dims);
        auto            it  = cfg_cache_.find(key);
        if (it != cfg_cache_.end())
            return it->second;
        A2AConfig cfg   = tma ? resolve_config_tma(src, peers, dims, mode, elem, false, stream, finish) :
                                resolve_config_nontma(src, peers, dims, mode, elem, stream, false, finish);
        cfg_cache_[key] = cfg;
        return cfg;
    };

    // Explicit path: force it.
    if (use_tma_i == 1)
        return {true, resolve_path(true)};
    if (use_tma_i == 0)
        return {false, resolve_path(false)};

    // Auto: sm<9 has no TMA; sm90+ picks the faster of the two paths (runtime replacement of the DiT table).
    if (sm_major_ < 9)
        return {false, resolve_path(false)};
    const std::tuple<int, int, int, int> akey{mode, dims.n_local, dims.s_local, dims.d};
    auto                                 ap = auto_path_cache_.find(akey);
    if (ap != auto_path_cache_.end())
        return {ap->second, resolve_path(ap->second)};
    // First auto call for this shape: tune both paths, time each real per-call, keep the faster.
    const A2AConfig cfg_n = resolve_path(false);
    const A2AConfig cfg_t = resolve_path(true);
    const float     t_n   = microbench_us(
        [&] {
            launch_a2a(src, peers, dims, mode, elem, cfg_n, stream);
            finish();
        },
        stream);
    const float t_t = microbench_us(
        [&] {
            launch_a2a_tma(src, peers, dims, mode, elem, cfg_t, stream);
            finish();
        },
        stream);
    const bool tma         = t_t <= t_n;
    auto_path_cache_[akey] = tma;
    return {tma, tma ? cfg_t : cfg_n};
}

void UlyssesGroup::destroy()
{
    if (destroyed_)
        return;
    if (pool_)
        pool_->destroy();
    if (owns_team_)
        nvshmem_team_destroy(team_);
    destroyed_ = true;
    if (--g_live_groups == 0 && g_world_inited) {
        // DEVIATION: nvshmem_finalize() is inline and calls nvshmemi_finalize() (again only in static
        // device.a). Use nvshmemx_hostlib_finalize() exported by the host shared library instead.
        nvshmemx_hostlib_finalize();
        g_world_inited = false;
    }
}

UlyssesGroup::~UlyssesGroup()
{
    // Collective cleanup must be explicit. Python synchronizes ranks before calling destroy().
}

}  // namespace ulysses
