"""Reusable WeightModule blocks backed by two disk-prefetched MPS buffers."""

import torch

from lightx2v.common.modules.weight_module import WeightModule, WeightModuleList
from lightx2v.common.offload.mps_manager import MpsSharedWeightAsyncStreamManager, host_view
from lightx2v.common.ops.utils import resolve_block_name
from lightx2v.utils.envs import GET_DTYPE


def iter_weight_modules(module):
    if hasattr(module, "base_attrs"):
        yield module
    for child in getattr(module, "_modules", {}).values():
        if child is not None:
            yield from iter_weight_modules(child)


class MpsStreamingBlockWeights(WeightModule):
    """Iterate logical layers while retaining only two physical weight blocks.

    Each yielded block is valid until iteration advances. The consumer must
    enqueue its computation on the default MPS stream before requesting the
    next block. This keeps the ordinary model forward loops unchanged.
    """

    def __init__(self, checkpoint, block_factory, num_layers):
        super().__init__()
        self.checkpoint = checkpoint
        self.block_factory = block_factory
        self.num_layers = num_layers
        if num_layers < 1:
            raise ValueError("MPS disk streaming requires at least one layer")
        if not hasattr(torch.mps, "_host_alias_storage"):
            raise RuntimeError("MPS disk streaming requires torch.mps._host_alias_storage (PyTorch >= 2.13)")
        self.add_module("buffers", WeightModuleList([]))
        # Validate every layer before allocating buffers or starting a worker.
        template = block_factory()
        for module in iter_weight_modules(template):
            for name, _, _ in module.base_attrs:
                dtype, shape, _, _ = checkpoint.tensor_metadata(name)
                if dtype != GET_DTYPE():
                    raise ValueError(f"MPS disk streaming requires matching checkpoint/inference dtypes: {name}")
                for index in range(num_layers):
                    layer_dtype, layer_shape, _, _ = checkpoint.tensor_metadata(resolve_block_name(name, index))
                    if (layer_dtype, layer_shape) != (dtype, shape):
                        raise ValueError(f"MPS disk streaming requires uniform layer shapes and dtypes: {name}, layer {index}")
        self.manager = MpsSharedWeightAsyncStreamManager()

    @property
    def blocks(self):
        return self

    def __len__(self):
        return self.num_layers

    def _ensure_buffers(self):
        if self.buffers:
            return
        buffers = WeightModuleList([self.block_factory() for _ in range(2)])
        for block in buffers:
            tensors = {}
            for module in iter_weight_modules(block):
                for name, _, _ in module.base_attrs:
                    dtype, shape, _, _ = self.checkpoint.tensor_metadata(name)
                    tensors[name] = torch.empty(shape, dtype=dtype, device="mps")
            # Ordinary loaders retain their usual transpose/dtype semantics.
            block.load(tensors)
            block.shared_host_tensors = {}
            for module in iter_weight_modules(block):
                for name, attr, transpose in module.base_attrs:
                    tensor = getattr(module, attr)
                    source = tensor.t() if transpose else tensor
                    if source.dtype != tensors[name].dtype or source.data_ptr() != tensors[name].data_ptr():
                        raise ValueError(f"MPS streaming operator must retain checkpoint storage: {name}")
                    block.shared_host_tensors[name] = host_view(source)
        self.add_module("buffers", buffers)
        self.manager.init_cuda_buffer(buffers)

    def load_block_into(self, block, block_index):
        destinations = {resolve_block_name(name, block_index): tensor for name, tensor in block.shared_host_tensors.items()}
        self.checkpoint.load_tensors_into(destinations)

    def __iter__(self):
        self._ensure_buffers()
        manager = self.manager
        completed = False
        try:
            if manager.need_init_first_buffer:
                manager.init_first_buffer(self)
            for index in range(self.num_layers):
                manager.prefetch_weights((index + 1) % self.num_layers, self)
                yield manager.cuda_buffers[0]
                manager.swap_blocks()
            completed = True
        finally:
            if not completed:
                # Also drain the worker when a consumer raises or exits early.
                self.release()

    def release(self):
        self.manager.close()
        for block in self.buffers:
            block.shared_host_tensors.clear()
            for module in iter_weight_modules(block):
                for _, attr, _ in module.base_attrs:
                    setattr(module, attr, None)
        self.add_module("buffers", WeightModuleList([]))
        torch.mps.empty_cache()
