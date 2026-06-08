import torch
import torch.nn.functional as F


class ErnieImageTransformerInfer:
    def __init__(self, config):
        self.config = config
        self.num_heads = config["num_attention_heads"]
        self.head_dim = config["hidden_size"] // config["num_attention_heads"]

    def set_scheduler(self, scheduler):
        self.scheduler = scheduler

    @staticmethod
    def _apply_rotary_emb(x_in: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
        rot_dim = freqs_cis.shape[-1]
        x, x_pass = x_in[..., :rot_dim], x_in[..., rot_dim:]
        cos_ = torch.cos(freqs_cis).to(x.dtype)
        sin_ = torch.sin(freqs_cis).to(x.dtype)
        x1, x2 = x.chunk(2, dim=-1)
        x_rotated = torch.cat((-x2, x1), dim=-1)
        return torch.cat((x * cos_ + x_rotated * sin_, x_pass), dim=-1)

    def _attention(self, weights, hidden_states, rotary_pos_emb):
        query = weights.to_q.apply(hidden_states).unflatten(-1, (self.num_heads, self.head_dim))
        key = weights.to_k.apply(hidden_states).unflatten(-1, (self.num_heads, self.head_dim))
        value = weights.to_v.apply(hidden_states).unflatten(-1, (self.num_heads, self.head_dim))

        query = weights.norm_q.apply(query)
        key = weights.norm_k.apply(key)
        query = self._apply_rotary_emb(query, rotary_pos_emb)
        key = self._apply_rotary_emb(key, rotary_pos_emb)

        hidden_states = weights.attn.apply(
            query,
            key,
            value,
            max_seqlen_q=query.shape[0],
            max_seqlen_kv=key.shape[0],
            causal=False,
        )
        return weights.to_out.apply(hidden_states)

    @staticmethod
    def _mlp(weights, hidden_states):
        return weights.linear_fc2.apply(weights.up_proj.apply(hidden_states) * F.gelu(weights.gate_proj.apply(hidden_states)))

    def _block(self, weights, hidden_states, rotary_pos_emb, temb):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = temb

        residual = hidden_states
        hidden_states = weights.adaLN_sa_ln.apply(hidden_states)
        hidden_states = (hidden_states.float() * (1 + scale_msa.float()) + shift_msa.float()).to(residual.dtype)
        attn_out = self._attention(weights, hidden_states, rotary_pos_emb)
        hidden_states = residual + (gate_msa.float() * attn_out.float()).to(residual.dtype)

        residual = hidden_states
        hidden_states = weights.adaLN_mlp_ln.apply(hidden_states)
        hidden_states = (hidden_states.float() * (1 + scale_mlp.float()) + shift_mlp.float()).to(residual.dtype)
        return residual + (gate_mlp.float() * self._mlp(weights, hidden_states).float()).to(residual.dtype)

    def infer(self, block_weights, pre_infer_out):
        hidden_states = pre_infer_out.hidden_states
        for weights in block_weights.blocks:
            hidden_states = self._block(
                weights,
                hidden_states,
                pre_infer_out.rotary_pos_emb,
                pre_infer_out.temb,
            )
        return hidden_states
