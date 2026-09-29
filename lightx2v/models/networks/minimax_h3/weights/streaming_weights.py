"""Two shared MPS weight buffers for H3 disk prefetch."""

import gc

import torch

from lightx2v.common.modules.weight_module import WeightModule, WeightModuleList
from lightx2v.common.offload.mps_manager import host_view
from lightx2v.common.ops.utils import resolve_block_name
from lightx2v.models.networks.minimax_h3.checkpoint import MiniMaxH3ShardCheckpoint

from .transformer_weights import MiniMaxH3TransformerBlockWeights


def _iter_base_attrs(module):
    for child in _iter_weight_modules(module):
        yield from child.base_attrs


def _iter_weight_modules(module):
    if hasattr(module, "base_attrs"):
        yield module
    for child in getattr(module, "_modules", {}).values():
        if child is not None:
            yield from _iter_weight_modules(child)


class MiniMaxH3StreamingTransformerWeights(WeightModule):
    """Own two reusable MPS blocks for disk streaming."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.num_layers = int(config.get("num_layers", 50))
        checkpoint_dir = config.get("dit_original_ckpt")
        if checkpoint_dir is None:
            raise ValueError("MiniMax-H3 dit_disk_streaming requires config['dit_original_ckpt'] to point to the diffusers transformer checkpoint directory.")
        self.checkpoint = MiniMaxH3ShardCheckpoint(checkpoint_dir)
        expected_block_indices = tuple(range(self.num_layers))
        if self.checkpoint.block_indices != expected_block_indices:
            raise ValueError(f"MiniMax-H3 dit_disk_streaming checkpoint block indices mismatch: expected {expected_block_indices}, found {self.checkpoint.block_indices}")

        self.add_module("blocks", WeightModuleList([]))
        self.add_module("offload_block_cuda_buffers", WeightModuleList([]))
        self.offload_phase_cuda_buffers = None
        self._ensure_streaming_buffers()

    def __len__(self):
        return self.num_layers

    def _ensure_streaming_buffers(self):
        if self.offload_block_cuda_buffers:
            return
        buffers = WeightModuleList([MiniMaxH3TransformerBlockWeights(0, self.config, create_cuda_buffer=True) for _ in range(2)])
        for block in buffers:
            tensors = {}
            for name, _, _ in _iter_base_attrs(block):
                dtype, shape, _, _ = self.checkpoint.tensor_metadata(name)
                tensors[name] = torch.empty(shape, dtype=dtype, device="mps")
            # Reuse the ordinary weight loaders, including their transpose and
            # dtype rules. These source tensors are already on the device.
            block.load(tensors)
            for module in _iter_weight_modules(block):
                for name, attr, _ in module.base_attrs:
                    buffer = getattr(module, f"{attr}_cuda_buffer")
                    if buffer.dtype != tensors[name].dtype:
                        raise ValueError(f"Shared weight loading requires matching file/inference dtypes: {name}")
                    setattr(module, attr, buffer)
            block.attn._build_fused_qkv()
            block.shared_host_tensors = {}
            for module in _iter_weight_modules(block):
                for name, attr, transpose in module.base_attrs:
                    buffer = getattr(module, attr)
                    # Disk data keeps its original row-major layout; compute
                    # retains the existing transposed view of the same storage.
                    block.shared_host_tensors[name] = host_view(buffer.t() if transpose else buffer)
        self.add_module("offload_block_cuda_buffers", buffers)

    def load_block_into(self, block, block_index):
        """CPU-only prefetch into an idle offload block, ordered by its manager."""
        destinations = {resolve_block_name(name, block_index): tensor for name, tensor in block.shared_host_tensors.items()}
        self.checkpoint.load_tensors_into(destinations)

    def release_disk_streaming_buffer(self):
        if not self.offload_block_cuda_buffers:
            return
        # The model closes the prefetch worker and clears compiled block references first.
        torch.mps.synchronize()
        for block in self.offload_block_cuda_buffers:
            block.shared_host_tensors.clear()
            for module in _iter_weight_modules(block):
                for _, attr, _ in module.base_attrs:
                    setattr(module, attr, None)
                    setattr(module, f"{attr}_cuda_buffer", None)
            # Fused QKV retains views outside the registered child modules.
            block.attn._build_fused_qkv()

        self.add_module("offload_block_cuda_buffers", WeightModuleList([]))
        gc.collect()
        # Metal completion may need the GIL to release safetensors storage.
        torch.mps.synchronize()
        torch.mps.empty_cache()
