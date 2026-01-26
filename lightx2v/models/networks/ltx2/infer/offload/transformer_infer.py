"""
Transformer inference module with CPU offloading support for LTX2.

This module implements block-level CPU offloading to reduce GPU memory usage.
"""

import torch

from lightx2v.common.offload.manager import WeightAsyncStreamManager
from lightx2v.models.networks.ltx2.infer.module_io import LTX2PreInferModuleOutput
from lightx2v.models.networks.ltx2.infer.transformer_infer import LTX2TransformerInfer
from lightx2v_platform.base.global_var import AI_DEVICE

torch_device_module = getattr(torch, AI_DEVICE)


class LTX2OffloadTransformerInfer(LTX2TransformerInfer):
    """
    LTX2 Transformer inference with CPU offloading support.

    Supports block-level offloading to reduce GPU memory usage by:
    - Keeping only one block in GPU memory at a time
    - Prefetching next block while current block is computing
    - Using async streams for overlap of data transfer and computation
    """

    def __init__(self, config):
        """
        Initialize transformer inference with offloading.

        Args:
            config: Model configuration dictionary with offloading settings:
                - cpu_offload: Enable CPU offloading
                - offload_granularity: "block" or "model" (only "block" supported here)
                - lazy_load: Enable lazy loading from disk
        """
        super().__init__(config)

        if self.config.get("cpu_offload", False):
            offload_granularity = self.config.get("offload_granularity", "block")

            if offload_granularity == "block":
                # Use block-level offloading
                self.infer_func = self.infer_with_blocks_offload
                self.offload_manager = WeightAsyncStreamManager(offload_granularity="block")
            elif offload_granularity == "model":
                # No offloading, keep full model in GPU
                self.infer_func = self.infer_without_offload
            else:
                raise ValueError(f"Unsupported offload_granularity: {offload_granularity}")

            # Initialize lazy loading if enabled
            self.lazy_load = self.config.get("lazy_load", False)
            if self.lazy_load and offload_granularity == "block":
                num_workers = self.config.get("num_disk_workers", 4)
                self.offload_manager.init_lazy_load(num_workers=num_workers)
        else:
            # No offloading
            self.infer_func = self.infer_without_offload

    def infer_without_offload(self, weights, pre_infer_out: LTX2PreInferModuleOutput):
        """
        Standard inference without offloading (full model in GPU).

        Args:
            weights: LTX2TransformerWeights instance
            pre_infer_out: LTX2PreInferModuleOutput from pre-inference

        Returns:
            Tuple of (video_x, audio_x, video_timestep, audio_timestep)
        """
        return super().infer(weights, pre_infer_out)

    def infer_with_blocks_offload(self, weights, pre_infer_out: LTX2PreInferModuleOutput):
        """
        Inference with block-level CPU offloading.

        This method:
        1. Keeps only one block in GPU at a time
        2. Prefetches the next block while computing current block
        3. Uses async CUDA streams for overlapped data transfer and computation

        Args:
            weights: LTX2TransformerWeights instance (blocks stored in CPU)
            pre_infer_out: LTX2PreInferModuleOutput from pre-inference

        Returns:
            Tuple of (video_x, audio_x, video_timestep, audio_timestep)
        """
        vx = pre_infer_out.video_args.x
        ax = pre_infer_out.audio_args.x

        blocks = weights.blocks

        # Process all transformer blocks with offloading
        for block_idx in range(len(blocks)):
            # Initialize first buffer on first iteration
            if self.offload_manager.need_init_first_buffer:
                self.offload_manager.init_first_buffer(blocks)

            # Prefetch next block to GPU (async, in background stream)
            # Use modulo to handle wrap-around for warmup
            next_block_idx = (block_idx + 1) % len(blocks)
            self.offload_manager.prefetch_weights(next_block_idx, blocks)

            # Compute current block (using compute stream)
            with torch_device_module.stream(self.offload_manager.compute_stream):
                # Use the block currently in cuda_buffers[0]
                current_block = self.offload_manager.cuda_buffers[0]
                vx, ax = self.infer_block(current_block, vx, ax, pre_infer_out)

            # Swap buffers: cuda_buffers[1] (prefetched) -> cuda_buffers[0] (current)
            self.offload_manager.swap_blocks()

        # Clean up if needed
        if self.clean_cuda_cache:
            del (
                pre_infer_out.video_args.context,
                pre_infer_out.audio_args.context,
            )
            torch_device_module.empty_cache()

        return vx, ax, pre_infer_out.video_args.embedded_timestep, pre_infer_out.audio_args.embedded_timestep

    def infer(self, weights, pre_infer_out: LTX2PreInferModuleOutput):
        """
        Main inference entry point.

        Delegates to the appropriate inference function based on offloading configuration.

        Args:
            weights: LTX2TransformerWeights instance
            pre_infer_out: LTX2PreInferModuleOutput from pre-inference

        Returns:
            Tuple of (video_x, audio_x, video_timestep, audio_timestep)
        """
        return self.infer_func(weights, pre_infer_out)
