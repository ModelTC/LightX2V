import torch
import torch.nn.functional as F

from lightx2v.common.transformer_infer.transformer_infer import BaseTransformerInfer
from lightx2v.models.networks.hunyuan3d.infer.module_io import Hunyuan3DPreInferOutput
from lightx2v.models.networks.hunyuan3d.infer.moe_infer import infer_moe_block


class Hunyuan3DTransformerInfer(BaseTransformerInfer):
    """Transformer inference for Hunyuan3D shape DiT blocks."""

    def __init__(self, config):
        self.config = config
        self.depth = config["depth"]
        self.num_heads = config["num_heads"]
        self.head_dim = config["hidden_size"] // self.num_heads
        self.scheduler = None

    def set_scheduler(self, scheduler):
        self.scheduler = scheduler

    @staticmethod
    def _flatten_norm(norm_weight, hidden_states):
        batch_size, seq_len, hidden_dim = hidden_states.shape
        flat = hidden_states.reshape(-1, hidden_dim)
        normed = norm_weight.apply(flat)
        return normed.reshape(batch_size, seq_len, hidden_dim)

    @staticmethod
    def _reshape_self_qkv(q, k, v, num_heads, head_dim, batch_size, seq_len):
        qkv = torch.cat((q, k, v), dim=-1)
        split_size = qkv.shape[-1] // num_heads // 3
        qkv = qkv.reshape(batch_size * seq_len, num_heads, split_size * 3)
        q, k, v = torch.split(qkv, split_size, dim=-1)
        return (
            q.reshape(batch_size, seq_len, num_heads, head_dim),
            k.reshape(batch_size, seq_len, num_heads, head_dim),
            v.reshape(batch_size, seq_len, num_heads, head_dim),
        )

    @staticmethod
    def _reshape_cross_kv(k, v, num_heads, head_dim, batch_size, seq_len):
        kv = torch.cat((k, v), dim=-1)
        split_size = kv.shape[-1] // num_heads // 2
        kv = kv.reshape(batch_size * seq_len, num_heads, split_size * 2)
        k, v = torch.split(kv, split_size, dim=-1)
        return (
            k.reshape(batch_size, seq_len, num_heads, head_dim),
            v.reshape(batch_size, seq_len, num_heads, head_dim),
        )

    def _project_qkv(self, hidden_states, to_q, to_k, to_v, norm_q, norm_k):
        batch_size, seq_len, hidden_dim = hidden_states.shape
        flat = hidden_states.reshape(-1, hidden_dim)
        q = to_q.apply(flat).reshape(batch_size, seq_len, hidden_dim)
        k = to_k.apply(flat).reshape(batch_size, seq_len, hidden_dim)
        v = to_v.apply(flat).reshape(batch_size, seq_len, hidden_dim)
        query, key, value = self._reshape_self_qkv(q, k, v, self.num_heads, self.head_dim, batch_size, seq_len)
        if norm_q is not None:
            query = norm_q.apply(query.reshape(-1, self.head_dim)).reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        if norm_k is not None:
            key = norm_k.apply(key.reshape(-1, self.head_dim)).reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        return query, key, value

    def _run_attention(self, query, key, value, calculate, merge_batch=False):
        batch_size = query.shape[0]
        if batch_size == 1:
            q = query[0]
            k = key[0]
            v = value[0]
            seqlen_q = q.shape[0]
            seqlen_k = k.shape[0]
            cu_seqlens_q = torch.tensor([0, q.shape[0]], dtype=torch.int32, device=q.device)
            cu_seqlens_k = torch.tensor([0, k.shape[0]], dtype=torch.int32, device=k.device)
            attn_output = calculate.apply(
                q=q,
                k=k,
                v=v,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_kv=cu_seqlens_k,
                max_seqlen_q=seqlen_q,
                max_seqlen_kv=seqlen_k,
                model_cls="hunyuan3d",
            )
            return attn_output.unsqueeze(0)

        if merge_batch:
            q = query.reshape(-1, self.num_heads, self.head_dim)
            k = key.reshape(-1, self.num_heads, self.head_dim)
            v = value.reshape(-1, self.num_heads, self.head_dim)
            seqlen_q = q.shape[0] // batch_size
            seqlen_k = k.shape[0] // batch_size
            cu_seqlens_q = torch.tensor([0, q.shape[0]], dtype=torch.int32, device=q.device)
            cu_seqlens_k = torch.tensor([0, k.shape[0]], dtype=torch.int32, device=k.device)
            attn_output = calculate.apply(
                q=q,
                k=k,
                v=v,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_kv=cu_seqlens_k,
                max_seqlen_q=seqlen_q,
                max_seqlen_kv=seqlen_k,
                model_cls="hunyuan3d",
            )
            return attn_output.reshape(batch_size, seqlen_q, -1)

        seqlen_q = query.shape[1]
        seqlen_k = key.shape[1]
        q = query.reshape(-1, self.num_heads, self.head_dim)
        k = key.reshape(-1, self.num_heads, self.head_dim)
        v = value.reshape(-1, self.num_heads, self.head_dim)
        cu_seqlens_q = torch.arange(0, batch_size + 1, dtype=torch.int32, device=q.device) * seqlen_q
        cu_seqlens_k = torch.arange(0, batch_size + 1, dtype=torch.int32, device=k.device) * seqlen_k
        attn_output = calculate.apply(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_kv=cu_seqlens_k,
            max_seqlen_q=seqlen_q,
            max_seqlen_kv=seqlen_k,
            model_cls="hunyuan3d",
        )
        return attn_output.reshape(batch_size, seqlen_q, -1)

    def _infer_self_attention(self, block_weights, hidden_states):
        norm_hidden = self._flatten_norm(block_weights.norm1, hidden_states)
        query, key, value = self._project_qkv(
            norm_hidden,
            block_weights.attn1.to_q,
            block_weights.attn1.to_k,
            block_weights.attn1.to_v,
            block_weights.attn1.norm_q,
            block_weights.attn1.norm_k,
        )
        attn_output = self._run_attention(query, key, value, block_weights.attn1.calculate, merge_batch=False)
        batch_size, seq_len, _ = hidden_states.shape
        flat = attn_output.reshape(-1, attn_output.shape[-1])
        return block_weights.attn1.out_proj.apply(flat).reshape(batch_size, seq_len, -1)

    def _infer_cross_attention(self, block_weights, hidden_states, cond):
        norm_hidden = self._flatten_norm(block_weights.norm2, hidden_states)
        batch_size, seq_len, _ = norm_hidden.shape
        cond_len = cond.shape[1]

        q_flat = block_weights.attn2.to_q.apply(norm_hidden.reshape(-1, norm_hidden.shape[-1]))
        k_flat = block_weights.attn2.to_k.apply(cond.reshape(-1, cond.shape[-1]))
        v_flat = block_weights.attn2.to_v.apply(cond.reshape(-1, cond.shape[-1]))

        query = q_flat.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        k = k_flat.reshape(batch_size, cond_len, self.num_heads * self.head_dim)
        v = v_flat.reshape(batch_size, cond_len, self.num_heads * self.head_dim)
        key, value = self._reshape_cross_kv(k, v, self.num_heads, self.head_dim, batch_size, cond_len)

        if block_weights.attn2.norm_q is not None:
            query = block_weights.attn2.norm_q.apply(query.reshape(-1, self.head_dim)).reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        if block_weights.attn2.norm_k is not None:
            key = block_weights.attn2.norm_k.apply(key.reshape(-1, self.head_dim)).reshape(batch_size, cond_len, self.num_heads, self.head_dim)

        attn_output = self._run_attention(query, key, value, block_weights.attn2.calculate, merge_batch=False)
        flat = attn_output.reshape(-1, attn_output.shape[-1])
        return block_weights.attn2.out_proj.apply(flat).reshape(batch_size, seq_len, -1)

    def _infer_mlp(self, block_weights, hidden_states):
        batch_size, seq_len, hidden_dim = hidden_states.shape
        flat = block_weights.mlp.fc1.apply(hidden_states.reshape(-1, hidden_dim))
        flat = F.gelu(flat)
        flat = block_weights.mlp.fc2.apply(flat)
        return flat.reshape(batch_size, seq_len, hidden_dim)

    def infer_block(self, block_weights, hidden_states, cond, skip_value=None):
        if block_weights.skip_linear is not None:
            cat = torch.cat([skip_value, hidden_states], dim=-1)
            batch_size, seq_len, hidden_dim = hidden_states.shape
            flat = block_weights.skip_linear.apply(cat.reshape(-1, cat.shape[-1]))
            hidden_states = block_weights.skip_norm.apply(flat).reshape(batch_size, seq_len, hidden_dim)
        hidden_states = hidden_states + self._infer_self_attention(block_weights, hidden_states)
        hidden_states = hidden_states + self._infer_cross_attention(block_weights, hidden_states, cond)
        norm_hidden = self._flatten_norm(block_weights.norm3, hidden_states)
        if block_weights.moe is not None:
            hidden_states = hidden_states + infer_moe_block(block_weights.moe, norm_hidden)
        else:
            hidden_states = hidden_states + self._infer_mlp(block_weights, norm_hidden)
        return hidden_states

    def infer(self, block_weights, pre_infer_out: Hunyuan3DPreInferOutput):
        hidden_states = pre_infer_out.hidden_states
        cond = pre_infer_out.cond

        skip_value_list = []
        for layer, block in enumerate(block_weights.blocks):
            skip_value = None if layer <= self.depth // 2 else skip_value_list.pop()
            hidden_states = self.infer_block(block, hidden_states, cond, skip_value=skip_value)
            if layer < self.depth // 2:
                skip_value_list.append(hidden_states)
        return hidden_states
