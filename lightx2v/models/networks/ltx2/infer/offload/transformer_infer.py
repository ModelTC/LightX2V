"""
Transformer inference module with CPU offloading support for LTX2.

This module implements block-level CPU offloading to reduce GPU memory usage.
"""

import torch

from lightx2v.common.offload.config import get_offload_granularity
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
                - offload_granularity: "block" or "model"
                - lazy_load: Enable lazy loading from disk
        """
        super().__init__(config)

        if self.config.get("cpu_offload", False):
            offload_granularity = get_offload_granularity(self.config)

            if offload_granularity == "block":
                # Use block-level offloading
                self.infer_func = self.infer_with_blocks_offload
            elif offload_granularity == "model":
                # Model movement is handled by LTX2Model.
                self.infer_func = self.infer_without_offload
            else:
                raise ValueError(f"Unsupported offload_granularity: {offload_granularity}")

            self.lazy_load = self.config.get("lazy_load", False)
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

    def get_compile_block_key(self, _block_idx, block):
        return id(block)

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
        self.reset_infer_states()

        vx = pre_infer_out.video_args.x
        ax = pre_infer_out.audio_args.x
        if self.use_compile:
            self.v_attn_cu_seqlens_qkv = self._create_cu_seqlens(vx.shape[0])
            self.a_attn_cu_seqlens_qkv = self._create_cu_seqlens(ax.shape[0])

        blocks = weights.blocks

        if self.lazy_load:
            vx, ax = self.infer_with_lazy_blocks_offload(blocks, vx, ax, pre_infer_out)
        else:

            def run_ltx_block(block_idx, block):
                nonlocal vx, ax
                vx, ax = self.run_block(
                    block_idx,
                    block,
                    vx,
                    ax,
                    pre_infer_out,
                    block_idx in self._mm_skip_video_self_blocks,
                    block_idx in self._mm_skip_audio_self_blocks,
                    self._mm_skip_a2v,
                    self._mm_skip_v2a,
                )
                return vx, ax

            self.run_blocks_with_offload(blocks, run_ltx_block)

        # Clean up if needed
        if self.clean_cuda_cache:
            del (
                pre_infer_out.video_args.context,
                pre_infer_out.audio_args.context,
            )
            torch_device_module.empty_cache()

        return vx, ax, pre_infer_out.video_args.embedded_timestep, pre_infer_out.audio_args.embedded_timestep

    def infer_with_lazy_blocks_offload(self, blocks, vx, ax, pre_infer_out):
        manager = self.get_block_offload_manager(blocks)
        for block_idx in range(len(blocks)):
            next_block_idx = (block_idx + 1) % len(blocks)
            manager.start_prefetch_block(next_block_idx)
            if manager.need_init_first_buffer:
                manager.init_first_buffer(blocks)
            manager.swap_cpu_buffers()
            manager.prefetch_weights(next_block_idx, blocks)
            with torch_device_module.stream(manager.compute_stream):
                vx, ax = self.run_block(
                    block_idx,
                    manager.cuda_buffers[0],
                    vx,
                    ax,
                    pre_infer_out,
                    block_idx in self._mm_skip_video_self_blocks,
                    block_idx in self._mm_skip_audio_self_blocks,
                    self._mm_skip_a2v,
                    self._mm_skip_v2a,
                )
            manager.swap_blocks()
        return vx, ax

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
