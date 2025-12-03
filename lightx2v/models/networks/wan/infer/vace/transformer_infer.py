from lightx2v.models.networks.wan.infer.offload.transformer_infer import WanOffloadTransformerInfer
from lightx2v.utils.envs import *


class WanVaceTransformerInfer(WanOffloadTransformerInfer):
    def __init__(self, config):
        super().__init__(config)
        self.vace_blocks_num = len(self.config["vace_layers"])
        self.vace_blocks_mapping = {orig_idx: seq_idx for seq_idx, orig_idx in enumerate(self.config["vace_layers"])}

    def infer(self, weights, pre_infer_out):
        self.get_scheduler_values()
        pre_infer_out.c = self.vace_pre_process(weights.vace_patch_embedding, pre_infer_out.vace_context)
        self.infer_vace_blocks(weights.vace_blocks, weights.blocks, pre_infer_out)
        x = self.infer_main_blocks(weights.blocks, pre_infer_out)
        return self.infer_non_blocks(weights, x, pre_infer_out.embed)

    def vace_pre_process(self, patch_embedding, vace_context):
        c = patch_embedding.apply(vace_context.unsqueeze(0).to(self.sensitive_layer_dtype))
        c = c.flatten(2).transpose(1, 2).contiguous().squeeze(0)
        return c

    def infer_vace_blocks(self, vace_blocks, blocks, pre_infer_out):
        pre_infer_out.adapter_args["hints"] = []
        self.infer_state = "vace"
        if hasattr(self, "offload_manager"):
            self.offload_manager.init_cuda_buffer(self.vace_offload_block_cuda_buffers, self.vace_offload_phase_cuda_buffers)
            self.offload_manager.init_first_buffer(vace_blocks)
        self.infer_func(vace_blocks, pre_infer_out.c, pre_infer_out)
        self.infer_state = "base"
        if hasattr(self, "offload_manager"):
            self.offload_manager.init_cuda_buffer(self.offload_block_cuda_buffers, self.offload_phase_cuda_buffers)
            self.offload_manager.init_first_buffer(blocks)

    def post_process(self, x, y, c_gate_msa, pre_infer_out):
        x = super().post_process(x, y, c_gate_msa, pre_infer_out)
        if self.infer_state == "base" and self.block_idx in self.vace_blocks_mapping:
            hint_idx = self.vace_blocks_mapping[self.block_idx]
            x = x + pre_infer_out.adapter_args["hints"][hint_idx] * pre_infer_out.adapter_args.get("context_scale", 1.0)
        return x
