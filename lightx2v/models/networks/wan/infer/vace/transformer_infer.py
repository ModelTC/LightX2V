from lightx2v.models.networks.wan.infer.offload.transformer_infer import WanOffloadTransformerInfer
from lightx2v.utils.envs import *


class WanVaceTransformerInfer(WanOffloadTransformerInfer):
    def __init__(self, config):
        super().__init__(config)
        self.vace_block_nums = len(self.config.vace_layers)
        self.vace_blocks_mapping = {orig_idx: seq_idx for seq_idx, orig_idx in enumerate(self.config.vace_layers)}

    def infer(self, weights, pre_infer_out):
        pre_infer_out.hints = self.infer_vace(weights, pre_infer_out)
        x = self.infer_main_blocks(weights, pre_infer_out)
        return self.infer_non_blocks(weights, x, pre_infer_out.embed)

    def infer_vace(self, weights, pre_infer_out):
        c = weights.vace_patch_embedding.apply(pre_infer_out.vace_context.unsqueeze(0).to(self.sensitive_layer_dtype))
        c = c.flatten(2).transpose(1, 2).contiguous().squeeze(0)

        self.infer_state = "vace"
        hints = []

        for i in range(self.vace_block_nums):
            c, c_skip = self.infer_vace_block(weights.vace_blocks[i], c, pre_infer_out.x, pre_infer_out)
            hints.append(c_skip)

        self.infer_state = "base"
        return hints

    def infer_vace_block(self, weights, c, x, pre_infer_out):
        if hasattr(weights, "before_proj"):
            c = weights.before_proj.apply(c) + x

        c = self.infer_block(weights, c, pre_infer_out)
        c_skip = weights.after_proj.apply(c)

        return c, c_skip

    def post_process(self, x, y, c_gate_msa, pre_infer_out):
        x = super().post_process(x, y, c_gate_msa, pre_infer_out)
        if self.infer_state == "base" and self.block_idx in self.vace_blocks_mapping:
            hint_idx = self.vace_blocks_mapping[self.block_idx]
            x = x + pre_infer_out.hints[hint_idx] * pre_infer_out.context_scale

        return x
