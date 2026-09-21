"""Reusable block weights for H3 disk streaming and shared MPS prefetch."""

import gc

import torch

from lightx2v.common.modules.weight_module import WeightModule, WeightModuleList
from lightx2v.common.ops.utils import resolve_block_name
from lightx2v.models.networks.minimax_h3.checkpoint import MiniMaxH3ShardCheckpoint
from lightx2v_platform.base.global_var import AI_DEVICE

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


def _empty_device_cache():
    if not isinstance(AI_DEVICE, str):
        return
    device_module = getattr(torch, AI_DEVICE, None)
    if device_module is not None and hasattr(device_module, "empty_cache"):
        if AI_DEVICE == "mps":
            # Drain copies before empty_cache waits while holding the GIL.
            # Metal completion may need the GIL to release safetensors storage.
            device_module.synchronize()
        device_module.empty_cache()


class MiniMaxH3StreamingTransformerWeights(WeightModule):
    """Own transient device blocks for disk streaming."""

    disk_streaming = True

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.num_layers = int(config.get("num_layers", 50))
        self.shared_buffer = bool(config.get("dit_mps_shared_buffer", False))
        checkpoint_dir = config.get("dit_original_ckpt")
        if checkpoint_dir is None:
            raise ValueError("MiniMax-H3 dit_disk_streaming requires config['dit_original_ckpt'] to point to the diffusers transformer checkpoint directory.")
        self.checkpoint = MiniMaxH3ShardCheckpoint(checkpoint_dir)
        expected_block_indices = tuple(range(self.num_layers))
        if self.checkpoint.block_indices != expected_block_indices:
            raise ValueError(f"MiniMax-H3 dit_disk_streaming checkpoint block indices mismatch: expected {expected_block_indices}, found {self.checkpoint.block_indices}")

        self.add_module("blocks", WeightModuleList([]))
        self.streaming_block = None
        self._ensure_streaming_block()

    def __len__(self):
        return self.num_layers

    def _init_shared_streaming_buffers(self):
        from lightx2v.common.offload.mps_manager import host_view

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
        self.offload_block_cuda_buffers = buffers
        self.offload_phase_cuda_buffers = None
        self.add_module("offload_block_cuda_buffers", buffers)
        self.streaming_block = buffers[0]

    def load_block_into(self, block, block_index):
        """CPU-only prefetch into an idle offload block, ordered by its manager."""
        destinations = {resolve_block_name(name, block_index): tensor for name, tensor in block.shared_host_tensors.items()}
        self.checkpoint.load_tensors_into(destinations)

    def load_streaming_block(self, block_index):
        block_index = int(block_index)
        if block_index not in self.checkpoint.block_indices:
            raise IndexError(f"MiniMax-H3 checkpoint does not contain transformer block {block_index}.")

        self._ensure_streaming_block()
        tensor_names = self.checkpoint.tensor_names_for_block(block_index)
        tensors = self.checkpoint.load_tensors(tensor_names, device="cpu")
        try:
            self.streaming_block.load_state_dict(self._prepare_streaming_state_dict(tensors, block_index), block_index)
        finally:
            del tensors
        return self.streaming_block

    def _ensure_streaming_block(self):
        if self.streaming_block is not None:
            return
        if self.shared_buffer:
            self._init_shared_streaming_buffers()
            return
        self.streaming_block = MiniMaxH3TransformerBlockWeights(0, self.config, create_cuda_buffer=True)
        self.add_module("streaming_block", self.streaming_block)
        block0_tensors = self.checkpoint.load_tensors(self.checkpoint.tensor_names_for_block(0), device="cpu")
        try:
            self.streaming_block.load(block0_tensors)
            self.streaming_block.load_state_dict(self._prepare_streaming_state_dict(block0_tensors, 0), 0)
        finally:
            del block0_tensors
            gc.collect()
            _empty_device_cache()

    def release_disk_streaming_buffer(self):
        if self.streaming_block is None:
            return
        # The model closes the prefetch worker and clears compiled block references first.
        getattr(torch, AI_DEVICE).synchronize()
        buffers = self.offload_block_cuda_buffers if self.shared_buffer else [self.streaming_block]
        for block in buffers:
            if self.shared_buffer:
                block.shared_host_tensors.clear()
            for module in _iter_weight_modules(block):
                for _, attr, _ in module.base_attrs:
                    setattr(module, attr, None)
                    setattr(module, f"{attr}_cuda_buffer", None)
            # Fused QKV retains views outside the registered child modules.
            block.attn._build_fused_qkv()

        self.add_module("streaming_block", None)
        if self.shared_buffer:
            self.add_module("offload_block_cuda_buffers", WeightModuleList([]))
        gc.collect()
        _empty_device_cache()

    def _prepare_streaming_state_dict(self, tensors, block_index):
        state_dict = dict(tensors)
        for name, _, transpose in _iter_base_attrs(self.streaming_block):
            if transpose:
                actual_name = resolve_block_name(name, block_index)
                if actual_name in state_dict:
                    state_dict[actual_name] = state_dict[actual_name].t()
        return state_dict
