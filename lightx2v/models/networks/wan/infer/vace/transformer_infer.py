import torch
import torch.distributed as dist
import torch.nn.functional as F

from lightx2v.models.networks.wan.infer.offload.transformer_infer import WanOffloadTransformerInfer
from lightx2v.utils.envs import *


class WanVaceTransformerInfer(WanOffloadTransformerInfer):
    def __init__(self, config):
        super().__init__(config)
        self.vace_blocks_num = len(self.config["vace_layers"])
        self.vace_blocks_mapping = {orig_idx: seq_idx for seq_idx, orig_idx in enumerate(self.config["vace_layers"])}

    def infer(self, weights, pre_infer_out):
        pre_infer_out.c = self.vace_pre_process(weights.vace_patch_embedding, pre_infer_out.vace_context)

        # In seq parallel mode, chunk c to match pre_infer_out.x
        if self.config.get("seq_parallel", False):
            pre_infer_out.c = self._chunk_c_for_seq_parallel(pre_infer_out.c, pre_infer_out.x)

        self.infer_vace_blocks(weights.vace_blocks, pre_infer_out)
        x = self.infer_main_blocks(weights.blocks, pre_infer_out)
        return self.infer_non_blocks(weights, x, pre_infer_out.embed)

    def _chunk_c_for_seq_parallel(self, c, x):
        """Chunk c along sequence dimension to match x in seq parallel mode."""
        world_size = dist.get_world_size(self.seq_p_group)
        cur_rank = dist.get_rank(self.seq_p_group)

        # x is already chunked in _seq_parallel_pre_process
        # c should have the same sequence length as original x (before chunking)
        # We need to infer the original length from the chunked x
        # However, we can't know the exact original length due to padding
        # So we assume c's length matches the original x length (which may include padding)

        # Get padding_multiple from config (same as used in _seq_parallel_pre_process)
        padding_multiple = self.config.get("padding_multiple", 1)
        multiple = world_size * padding_multiple

        # Calculate padding for c to match the same chunking pattern as x
        padding_size = (multiple - (c.shape[0] % multiple)) % multiple
        if padding_size > 0:
            c = F.pad(c, (0, 0, 0, padding_size))

        # Chunk c the same way as x
        c_chunks = torch.chunk(c, world_size, dim=0)
        return c_chunks[cur_rank]

    def vace_pre_process(self, patch_embedding, vace_context):
        c = patch_embedding.apply(vace_context.unsqueeze(0).to(self.sensitive_layer_dtype))
        c = c.flatten(2).transpose(1, 2).contiguous().squeeze(0)
        return c

    def infer_vace_blocks(self, vace_blocks, pre_infer_out):
        pre_infer_out.adapter_args["hints"] = []
        self.infer_state = "vace"
        if hasattr(self, "offload_manager"):
            self.offload_manager.init_cuda_buffer(self.vace_offload_block_cuda_buffers, self.vace_offload_phase_cuda_buffers)
        self.infer_func(vace_blocks, pre_infer_out.c, pre_infer_out)
        self.infer_state = "base"
        if hasattr(self, "offload_manager"):
            self.offload_manager.init_cuda_buffer(self.offload_block_cuda_buffers, self.offload_phase_cuda_buffers)

    def post_process(self, x, y, c_gate_msa, pre_infer_out):
        x = super().post_process(x, y, c_gate_msa, pre_infer_out)
        if self.infer_state == "base" and self.block_idx in self.vace_blocks_mapping:
            hint_idx = self.vace_blocks_mapping[self.block_idx]
            x = x + pre_infer_out.adapter_args["hints"][hint_idx] * pre_infer_out.adapter_args.get("context_scale", 1.0)
        return x
