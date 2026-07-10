// Copied and adapted from https://github.com/triple-mu/fast-ulysses
#pragma once
#include <ATen/ATen.h>
#include <cstdint>
#include <map>
#include <nvshmem.h>
#include <string>
#include <tuple>
#include <vector>

namespace ulysses {

class SymmetricHeapPool {
public:
    // reserved_bytes: per-group cap (must be <= NVSHMEM_SYMMETRIC_SIZE reserved at init).
    SymmetricHeapPool(int64_t reserved_bytes, int world_size, std::vector<int> peer_global_pes);

    struct Buffer {
        void*                 sym_base;
        int64_t               nbytes;
        std::vector<uint64_t> peer_ptrs;  // nvshmem_ptr(sym_base, peer_global_pe)
        at::Tensor            view;       // from_blob with no-op deleter (pool owns lifetime)
    };

    // Reuse on (tag,shape,dtype) hit; otherwise collectively allocate a new segment and register it.
    const Buffer& acquire(const std::vector<int64_t>& shape, c10::ScalarType dtype, const std::string& tag);

    // Terminal collective op: before calling, release all from_blob views returned by acquire() and
    // ensure no A2A/collective is in flight, since this nvshmem_free's the segments those views alias.
    void destroy();  // nvshmem_free all segments + clear registry

private:
    using Key = std::tuple<std::string, std::vector<int64_t>, c10::ScalarType>;
    int64_t               reserved_, used_ = 0;
    int                   world_size_;
    std::vector<int>      peer_global_pes_;
    std::vector<void*>    segments_;
    std::map<Key, Buffer> registry_;
    bool                  destroyed_ = false;
};

}  // namespace ulysses
