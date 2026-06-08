import math

import torch
import torch.nn.functional as F

from lightx2v.models.networks.wan.infer.transformer_infer import WanTransformerInfer
from lightx2v.utils.envs import GET_DTYPE


def _token_modulation(x: torch.Tensor) -> torch.Tensor:
    return x.reshape(-1, x.shape[-1])


class LingbotVATransformerInfer(WanTransformerInfer):
    def __init__(self, config):
        super().__init__(config)
        self.kv_cache_manager = None

    def _split_modulation(self, modulation, timestep_proj):
        table = modulation.tensor.to(timestep_proj.device, timestep_proj.dtype)
        while table.dim() > timestep_proj.dim():
            table = table.squeeze(0)
        values = table + timestep_proj
        return [_token_modulation(item) for item in values.chunk(6, dim=1)]

    @staticmethod
    def _apply_rotary(x: torch.Tensor, rotary_emb: torch.Tensor) -> torch.Tensor:
        x_dtype = x.dtype
        seq_len, num_heads, head_dim = x.shape
        freqs = rotary_emb.to(x.device)
        x_complex = torch.view_as_complex(x.to(torch.float64).reshape(seq_len, num_heads, -1, 2))
        out = torch.view_as_real(x_complex * freqs).flatten(2)
        return out.to(x_dtype)

    def infer_self_attn_with_kvcache(self, phase, x, shift_msa, scale_msa, rotary_emb, update_cache, cache_name):
        query_len = x.shape[0]
        norm1_out = phase.norm1.apply(x).to(x.dtype)
        norm1_out = (norm1_out.float() * (1.0 + scale_msa.float()) + shift_msa.float()).to(x.dtype)

        q = phase.self_attn_norm_q.apply(phase.self_attn_q.apply(norm1_out)).view(query_len, self.num_heads, self.head_dim)
        k = phase.self_attn_norm_k.apply(phase.self_attn_k.apply(norm1_out)).view(query_len, self.num_heads, self.head_dim)
        v = phase.self_attn_v.apply(norm1_out).view(query_len, self.num_heads, self.head_dim)
        q = self._apply_rotary(q, rotary_emb)
        k = self._apply_rotary(k, rotary_emb)

        cache = self.kv_cache_manager.get_self_attn_kv_cache(cache_name) if self.kv_cache_manager is not None else None
        slots = None
        if cache is not None:
            slots = cache.store_kv(k, v, layer_id=self.block_idx, is_pred=(update_cache == 1))
            k = cache.k_cache(self.block_idx)
            v = cache.v_cache(self.block_idx)

        kv_len = k.shape[0]
        attn_out = phase.self_attn_1.apply(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=torch.tensor([0, query_len], device=q.device, dtype=torch.int32),
            cu_seqlens_kv=torch.tensor([0, kv_len], device=k.device, dtype=torch.int32),
            max_seqlen_q=query_len,
            max_seqlen_kv=kv_len,
        )

        if cache is not None and update_cache == 0 and slots is not None:
            cache.restore(self.block_idx, slots)

        return phase.self_attn_o.apply(attn_out)

    def infer_ffn(self, phase, x, attn_out, c_shift_msa, c_scale_msa):
        x.add_(attn_out)
        norm2_out = phase.norm2.apply(x).to(x.dtype)
        c_shift_msa = _token_modulation(c_shift_msa)
        c_scale_msa = _token_modulation(c_scale_msa)
        norm2_out = (norm2_out.float() * (1.0 + c_scale_msa.float()) + c_shift_msa.float()).to(x.dtype)
        y = phase.ffn_0.apply(norm2_out)
        y = F.gelu(y, approximate="tanh")
        return phase.ffn_2.apply(y)

    def infer_block(self, block, x, pre_infer_out, update_cache=0, cache_name="pos"):
        shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = self._split_modulation(
            block.compute_phases[0].modulation,
            pre_infer_out.timestep_proj,
        )
        y_out = self.infer_self_attn_with_kvcache(
            block.compute_phases[0],
            x,
            shift_msa,
            scale_msa,
            pre_infer_out.rotary_emb,
            update_cache=update_cache,
            cache_name=cache_name,
        )
        x, attn_out = self.infer_cross_attn(block.compute_phases[1], x, pre_infer_out.context, y_out, gate_msa)
        y = self.infer_ffn(block.compute_phases[2], x, attn_out, c_shift_msa, c_scale_msa)
        return self.post_process(x, y, c_gate_msa, pre_infer_out)

    def infer_main_blocks(self, blocks, pre_infer_out, update_cache=0, cache_name="pos"):
        x = pre_infer_out.x
        for block_idx in range(len(blocks)):
            self.block_idx = block_idx
            x = self.infer_block(blocks[block_idx], x, pre_infer_out, update_cache=update_cache, cache_name=cache_name)
        return x

    def infer_non_blocks(self, weights, x, pre_infer_out, action_mode=False):
        table = weights.head_modulation.tensor.to(x.device, x.dtype)
        while table.dim() > 2:
            table = table.squeeze(0)
        values = table + pre_infer_out.temb[:, None, :]
        shift, scale = [_token_modulation(item) for item in values.chunk(2, dim=1)]
        x = weights.norm.apply(x).to(x.dtype)
        x = (x.float() * (1.0 + scale.float()) + shift.float()).to(x.dtype)

        if action_mode:
            return weights.action_head.apply(x).unsqueeze(0)

        x = weights.head.apply(x)
        return x.reshape(1, x.shape[0], math.prod(self.config.get("patch_size", (1, 2, 2))), -1).reshape(1, -1, self.config["out_dim"])

    @torch.no_grad()
    def infer(self, weights, pre_infer_out, action_mode=False, update_cache=0, cache_name="pos"):
        self.reset_infer_states()
        x = self.infer_main_blocks(weights.blocks, pre_infer_out, update_cache=update_cache, cache_name=cache_name)
        return self.infer_non_blocks(weights, x, pre_infer_out, action_mode=action_mode).to(GET_DTYPE())
