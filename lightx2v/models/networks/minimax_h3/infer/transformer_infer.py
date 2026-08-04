import torch
import torch.nn.functional as F

from lightx2v.common.transformer_infer.transformer_infer import BaseTransformerInfer
from lightx2v.utils.envs import GET_DTYPE


def _apply_rotary_emb(hidden_states: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    rotary_dim = cos.shape[-1]
    rotary = hidden_states[..., :rotary_dim]
    passthrough = hidden_states[..., rotary_dim:]
    cos = cos.to(hidden_states.dtype)[:, None, :]
    sin = sin.to(hidden_states.dtype)[:, None, :]
    x1, x2 = rotary.chunk(2, dim=-1)
    rotated = torch.cat((-x2, x1), dim=-1)
    rotary = rotary * cos + rotated * sin
    return torch.cat((rotary, passthrough), dim=-1).contiguous()


class MiniMaxH3TransformerInfer(BaseTransformerInfer):
    def __init__(self, config):
        self.config = config
        self.hidden_size = int(config.get("hidden_size", 5376))
        self.num_heads = int(config.get("num_attention_heads", 56))
        self.head_dim = int(config.get("attention_head_dim", 128))
        self.init_compile(config)

    def _attention(self, weights, hidden_states, rotary_emb):
        q = weights.to_q.apply(hidden_states).unflatten(-1, (self.num_heads, self.head_dim))
        k = weights.to_k.apply(hidden_states).unflatten(-1, (self.num_heads, self.head_dim))
        v = weights.to_v.apply(hidden_states).unflatten(-1, (self.num_heads, self.head_dim))
        q = _apply_rotary_emb(weights.norm_q.apply(q), *rotary_emb)
        k = _apply_rotary_emb(weights.norm_k.apply(k), *rotary_emb)
        seq_len = q.shape[0]
        cu_seqlens = torch.tensor((0, seq_len), dtype=torch.int32, device=q.device)
        out = weights.calculate.apply(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_kv=cu_seqlens,
            max_seqlen_q=seq_len,
            max_seqlen_kv=seq_len,
            causal=False,
        )
        return weights.to_out.apply(out.to(GET_DTYPE()))

    @staticmethod
    def _ff(weights, hidden_states):
        value, gate = weights.in_proj.apply(hidden_states).chunk(2, dim=-1)
        return weights.out_proj.apply(value * F.silu(gate))

    def infer_block(self, weights, hidden_states, pre_infer_out):
        # Activation is evaluated in fp32, then cast immediately before the
        # checkpoint's bf16 AdaLN projection.
        modulation = weights.adaln.apply(F.silu(pre_infer_out.temb).to(GET_DTYPE()))
        modulation = modulation.view(-1, 6 * self.hidden_size)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = modulation.chunk(6, dim=-1)
        indices = pre_infer_out.adaln_indices

        residual = hidden_states
        normed = weights.norm1.apply(hidden_states)
        normed = normed * (1.0 + scale_msa.index_select(0, indices))
        normed = normed + shift_msa.index_select(0, indices)
        hidden_states = residual + gate_msa.index_select(0, indices) * self._attention(weights.attn, normed, pre_infer_out.rotary_emb)

        residual = hidden_states
        normed = weights.norm2.apply(hidden_states)
        normed = normed * (1.0 + scale_mlp.index_select(0, indices))
        normed = normed + shift_mlp.index_select(0, indices)
        hidden_states = residual + gate_mlp.index_select(0, indices) * self._ff(weights.ff, normed)
        return hidden_states

    def infer(self, block_weights, pre_infer_out):
        hidden_states = pre_infer_out.hidden_states
        for block_index, block in enumerate(block_weights.blocks):
            hidden_states = self.run_block(block_index, block, hidden_states, pre_infer_out)
        return hidden_states
