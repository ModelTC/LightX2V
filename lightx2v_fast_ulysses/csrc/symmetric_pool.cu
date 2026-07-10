// Copied and adapted from https://github.com/triple-mu/fast-ulysses
#include "symmetric_pool.cuh"
#include <c10/cuda/CUDAGuard.h>
#include <nvshmem.h>
#include <nvshmemx.h>
#include <torch/torch.h>

namespace ulysses {

SymmetricHeapPool::SymmetricHeapPool(int64_t reserved_bytes, int world_size, std::vector<int> peer_global_pes):
    reserved_(reserved_bytes),
    world_size_(world_size),
    peer_global_pes_(std::move(peer_global_pes))
{
}

const SymmetricHeapPool::Buffer&
SymmetricHeapPool::acquire(const std::vector<int64_t>& shape, c10::ScalarType dtype, const std::string& tag)
{
    TORCH_CHECK(!destroyed_, "SymmetricHeapPool::acquire called after destroy()");
    Key  key{tag, shape, dtype};
    auto it = registry_.find(key);
    if (it != registry_.end())
        return it->second;  // reuse

    int64_t numel = 1;
    for (auto s : shape)
        numel *= s;
    const int64_t elem   = c10::elementSize(dtype);
    int64_t       nbytes = numel * elem;
    nbytes               = (nbytes + 15) / 16 * 16;  // uint4 alignment

    // Uniform: every rank computes an identical nbytes (out_shape is rank-independent), so the
    // collective (uniform-size) nvshmem_align below needs no max-reduce.
    const int64_t alloc_bytes = nbytes;
    TORCH_CHECK(used_ + alloc_bytes <= reserved_,
                "SymmetricHeapPool OOM: need ",
                alloc_bytes,
                " B, used ",
                used_,
                " / reserved ",
                reserved_,
                " B. Increase initial_pool_bytes.");

    void* p = nvshmem_align(256, alloc_bytes);  // collective alloc (uniform size); address never moves
    TORCH_CHECK(p != nullptr, "nvshmem_align failed for ", alloc_bytes, " B");
    segments_.push_back(p);
    used_ += alloc_bytes;

    Buffer buf;
    buf.sym_base = p;
    buf.nbytes   = alloc_bytes;
    buf.peer_ptrs.resize(world_size_);
    for (int i = 0; i < world_size_; ++i)
        buf.peer_ptrs[i] = reinterpret_cast<uint64_t>(nvshmem_ptr(p, peer_global_pes_[i]));
    // Single-node P2P: all peer pointers must be non-null.
    for (int i = 0; i < world_size_; ++i)
        TORCH_CHECK(buf.peer_ptrs[i] != 0,
                    "nvshmem_ptr returned NULL for peer ",
                    i,
                    " (non-P2P-reachable; phase-1 requires single-node NVLink).");

    auto opts = at::TensorOptions().dtype(dtype).device(at::kCUDA, at::cuda::current_device());
    buf.view  = at::from_blob(
        p, shape, [](void*) {}, opts);  // no-op deleter

    auto res = registry_.emplace(std::move(key), std::move(buf));
    return res.first->second;
}

void SymmetricHeapPool::destroy()
{
    if (destroyed_)
        return;
    registry_.clear();  // drop from_blob views (does not free underlying memory)
    for (void* p : segments_)
        nvshmem_free(p);
    segments_.clear();
    destroyed_ = true;
}

}  // namespace ulysses
