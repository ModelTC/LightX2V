// Copied and adapted from https://github.com/triple-mu/fast-ulysses
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <nvshmem.h>
#include <nvshmemx.h>
#include <torch/extension.h>
#include <torch/library.h>

#include "a2a_config.cuh"
#include "ulysses_common.cuh"
#include "ulysses_group.cuh"

namespace ulysses {

int64_t nvshmem_uniqueid_nbytes()
{
    return static_cast<int64_t>(sizeof(nvshmemx_uniqueid_t));
}

at::Tensor all_to_all_single_4d(const c10::intrusive_ptr<UlyssesGroup>& group,
                                at::Tensor                              input,
                                int64_t                                 mode,
                                std::string                             tag,
                                c10::optional<bool>                     use_tma)
{
    TORCH_CHECK(input.is_cuda() && input.dim() == 4, "input must be a 4D CUDA tensor");
    TORCH_CHECK(input.scalar_type() == at::kHalf || input.scalar_type() == at::kBFloat16,
                "dtype must be float16 or bfloat16");
    input        = input.contiguous();
    const int ws = static_cast<int>(group->world_size());
    TORCH_CHECK(ws >= 1 && ws <= 8, "world_size must be in [1, 8] (single-node NVLink), got ", ws);
    const int me   = static_cast<int>(group->rank());
    const int b    = static_cast<int>(input.size(0));
    const int x1   = static_cast<int>(input.size(1));
    const int x2   = static_cast<int>(input.size(2));
    const int d    = static_cast<int>(input.size(3));
    const int elem = static_cast<int>(input.element_size());
    TORCH_CHECK((static_cast<int64_t>(d) * elem) % 16 == 0, "d*elem_size must be 16B-aligned");
    TORCH_CHECK(mode == 0 || mode == 1, "mode must be 0 or 1");

    const at::cuda::CUDAGuard guard(input.device());
    cudaStream_t              stream = at::cuda::getCurrentCUDAStream();

    Ulysses4DDims dims;
    dims.b    = b;
    dims.d    = d;
    dims.rank = me;
    std::vector<int64_t> out_shape;
    if (mode == 0) {
        TORCH_CHECK(x2 % ws == 0, "n_global must be divisible by world_size");
        dims.s_local  = x1;
        dims.n_global = x2;
        dims.s_global = x1 * ws;
        dims.n_local  = x2 / ws;
        out_shape     = {b, dims.s_global, dims.n_local, d};
    }
    else {
        TORCH_CHECK(x1 % ws == 0, "s_global must be divisible by world_size");
        dims.s_global = x1;
        dims.n_local  = x2;
        dims.s_local  = x1 / ws;
        dims.n_global = x2 * ws;
        out_shape     = {b, dims.s_local, dims.n_global, d};
    }

    const auto& buf       = group->pool().acquire(out_shape, input.scalar_type(), tag);
    const int   use_tma_i = use_tma.has_value() ? (*use_tma ? 1 : 0) : -1;
    if (use_tma_i > 0)
        TORCH_CHECK(group->sm_major() >= 9, "use_tma=True requires sm90+ (TMA unavailable on this GPU)");
    const auto pc =
        group->resolve_config(dims, static_cast<int>(mode), use_tma_i, input.data_ptr(), buf.peer_ptrs, elem, stream);
    if (pc.tma)
        launch_a2a_tma(input.data_ptr(), buf.peer_ptrs, dims, static_cast<int>(mode), elem, pc.cfg, stream);
    else
        launch_a2a(input.data_ptr(), buf.peer_ptrs, dims, static_cast<int>(mode), elem, pc.cfg, stream);
    ULYSSES_CUDA_CHECK(cudaGetLastError());
    nvshmemx_quiet_on_stream(stream);
    group->fast_barrier(stream);
    return buf.view;
}

}  // namespace ulysses

TORCH_LIBRARY(fast_ulysses, m)
{
    m.def("nvshmem_uniqueid_nbytes() -> int");
    m.impl("nvshmem_uniqueid_nbytes", &ulysses::nvshmem_uniqueid_nbytes);

    m.class_<ulysses::UlyssesGroup>("UlyssesGroup")
        .def(torch::init<std::vector<int64_t>, int64_t, int64_t, int64_t>())
        .def("rank", &ulysses::UlyssesGroup::rank)
        .def("world_size", &ulysses::UlyssesGroup::world_size)
        .def("destroy", &ulysses::UlyssesGroup::destroy)
        .def_static("uniqueid_nints", &ulysses::UlyssesGroup::uniqueid_nints)
        .def_static("get_uniqueid", &ulysses::UlyssesGroup::get_uniqueid)
        .def_static("init_world", &ulysses::UlyssesGroup::init_world);

    m.def("all_to_all_single_4d(__torch__.torch.classes.fast_ulysses.UlyssesGroup group, "
          "Tensor input, int mode, str tag, bool? use_tma=None) -> Tensor");
    m.impl("all_to_all_single_4d", c10::DispatchKey::CompositeExplicitAutograd, &ulysses::all_to_all_single_4d);
}

PYBIND11_MODULE(_C, m) {}
