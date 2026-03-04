# lightx2v/models/networks/mova/transformer_infer.py

import torch
from lightx2v.models.networks.wan.infer.transformer_infer import WanTransformerInfer


class MOVATransformerInfer(WanTransformerInfer):
    """
    为 MOVA 定制的推理引擎，支持逐块前向。
    参数签名与现有调用一致：forward_block(x, context, t_mod, freqs, block_weights)
    """
    def __init__(self, config):
        super().__init__(config)
        # 如果 apply_rope_func 仍为 None，设置一个默认函数
        if self.apply_rope_func is None:
            from lightx2v.models.networks.wan.infer.utils import apply_wan_rope_with_torch
            self.apply_rope_func = apply_wan_rope_with_torch
    def forward_block(self, block_weights, x, pre_infer_out, block_idx=None):
        original_shape = x.shape
        if x.ndim == 3:
            x = x.view(-1, x.shape[-1])

        # 从自注意力模块获取目标 dtype
        target_dtype = block_weights.compute_phases[0].self_attn_q.weight.dtype
        x = x.to(target_dtype)
        pre_infer_out.embed0 = pre_infer_out.embed0.to(target_dtype)
        pre_infer_out.context = pre_infer_out.context.to(target_dtype)
        pre_infer_out.cos_sin = pre_infer_out.cos_sin.to(target_dtype)

        self.reset_infer_states()
        self.cos_sin = pre_infer_out.cos_sin
        if block_idx is not None:
            self.block_idx = block_idx

        x = self.infer_block(block_weights, x, pre_infer_out)

        if len(original_shape) == 3:
            x = x.view(*original_shape)

        return x