import torch
import torch.distributed as dist
import torch.nn.functional as F

from lightx2v.common.transformer_infer.transformer_infer import BaseTransformerInfer
from lightx2v.models.networks.cosmos3.infer.module_io import Cosmos3TransformerInferModuleOutput
from lightx2v.models.networks.cosmos3.infer.utils import build_rotary_embeddings


class Cosmos3TransformerInfer(BaseTransformerInfer):
    def __init__(self, config):
        self.config = config
        self.hidden_size = config["hidden_size"]
        self.head_dim = config["head_dim"]
        self.num_attention_heads = config["num_attention_heads"]
        self.num_key_value_heads = config["num_key_value_heads"]
        self.num_key_value_groups = self.num_attention_heads // self.num_key_value_heads
        self.rope_theta = config.get("rope_theta", 5000000)
        rope_scaling = config.get("rope_scaling", None)
        self.rope_axes_dim = tuple(rope_scaling.get("mrope_section", [24, 20, 20]) if rope_scaling else [24, 20, 20])
        if self.config.get("seq_parallel", False):
            self.seq_p_group = self.config.get("device_mesh").get_group(mesh_dim="seq_p")
        else:
            self.seq_p_group = None
        self.seq_p_gen_len = None

    def _repeat_kv_for_gqa(self, key, value):
        if self.num_key_value_heads == self.num_attention_heads:
            return key, value
        key = key.repeat_interleave(self.num_key_value_groups, dim=1)
        value = value.repeat_interleave(self.num_key_value_groups, dim=1)
        return key, value

    def _all_gather_gen_tokens(self, tensor):
        if self.seq_p_group is None:
            return tensor
        world_size = dist.get_world_size(self.seq_p_group)
        tensor = tensor.contiguous()
        gathered = [torch.empty_like(tensor) for _ in range(world_size)]
        dist.all_gather(gathered, tensor, group=self.seq_p_group)
        gathered_tensor = torch.cat(gathered, dim=0)
        if self.seq_p_gen_len is not None:
            gathered_tensor = gathered_tensor[: self.seq_p_gen_len]
        return gathered_tensor

    @staticmethod
    def _infer_attn(attn_weight, q, k, v, causal=False):
        cu_seqlens_q = torch.tensor([0, q.shape[0]], dtype=torch.int32, device="cpu")
        cu_seqlens_kv = torch.tensor([0, k.shape[0]], dtype=torch.int32, device="cpu")
        return attn_weight.apply(
            q=q,
            k=k,
            v=v,
            causal=causal,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_kv=cu_seqlens_kv,
            max_seqlen_q=q.shape[0],
            max_seqlen_kv=k.shape[0],
        )

    def _infer_attention(self, block_weights, und_seq, gen_seq, rotary_emb):
        q_und = block_weights.to_q.apply(und_seq).view(-1, self.num_attention_heads, self.head_dim)
        k_und = block_weights.to_k.apply(und_seq).view(-1, self.num_key_value_heads, self.head_dim)
        v_und = block_weights.to_v.apply(und_seq).view(-1, self.num_key_value_heads, self.head_dim)
        q_gen = block_weights.add_q_proj.apply(gen_seq).view(-1, self.num_attention_heads, self.head_dim)
        k_gen = block_weights.add_k_proj.apply(gen_seq).view(-1, self.num_key_value_heads, self.head_dim)
        v_gen = block_weights.add_v_proj.apply(gen_seq).view(-1, self.num_key_value_heads, self.head_dim)

        q_und = block_weights.norm_q.apply(q_und)
        k_und = block_weights.norm_k.apply(k_und)
        q_gen = block_weights.norm_added_q.apply(q_gen)
        k_gen = block_weights.norm_added_k.apply(k_gen)

        cos_und, sin_und, cos_gen, sin_gen = rotary_emb
        q_und, k_und = block_weights.rope.apply(q_und, k_und, (cos_und, sin_und))
        q_gen, k_gen = block_weights.rope.apply(q_gen, k_gen, (cos_gen, sin_gen))

        k_und_full, v_und_full = self._repeat_kv_for_gqa(k_und, v_und)
        causal_out = self._infer_attn(block_weights.causal_self_attn, q_und, k_und_full, v_und_full, causal=True)

        full_k_gen = self._all_gather_gen_tokens(k_gen)
        full_v_gen = self._all_gather_gen_tokens(v_gen)
        all_k = torch.cat([k_und, full_k_gen], dim=0)
        all_v = torch.cat([v_und, full_v_gen], dim=0)
        all_k, all_v = self._repeat_kv_for_gqa(all_k, all_v)
        full_out = self._infer_attn(block_weights.self_attn, q_gen, all_k, all_v, causal=False)

        return block_weights.to_out.apply(causal_out), block_weights.to_add_out.apply(full_out)

    @staticmethod
    def _infer_mlp(weights, hidden_states):
        return weights.down_proj.apply(F.silu(weights.gate_proj.apply(hidden_states)) * weights.up_proj.apply(hidden_states))

    def _infer_block(self, layer_weights, und_seq, gen_seq, rotary_emb):
        und_norm = layer_weights.input_layernorm.apply(und_seq)
        gen_norm = layer_weights.input_layernorm_moe_gen.apply(gen_seq)
        und_attn_out, gen_attn_out = self._infer_attention(layer_weights.self_attn, und_norm, gen_norm, rotary_emb)
        residual_und = und_seq + und_attn_out
        residual_gen = gen_seq + gen_attn_out
        mlp_out_und = self._infer_mlp(layer_weights.mlp, layer_weights.post_attention_layernorm.apply(residual_und))
        mlp_out_gen = self._infer_mlp(layer_weights.mlp_moe_gen, layer_weights.post_attention_layernorm_moe_gen.apply(residual_gen))
        return residual_und + mlp_out_und, residual_gen + mlp_out_gen

    def infer_layers(self, layers, und_seq, gen_seq, rotary_emb):
        for layer_weights in layers:
            und_seq, gen_seq = self._infer_block(layer_weights, und_seq, gen_seq, rotary_emb)
        return und_seq, gen_seq

    def infer(self, block_weights, pre_infer_out):
        hidden_states = pre_infer_out.hidden_states
        self.seq_p_gen_len = getattr(pre_infer_out, "seq_p_gen_len", None)
        cos, sin = build_rotary_embeddings(
            position_ids=pre_infer_out.position_ids,
            head_dim=self.head_dim,
            rope_theta=self.rope_theta,
            rope_axes_dim=self.rope_axes_dim,
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )
        und_len = pre_infer_out.und_len
        rotary_emb = (cos[:und_len], sin[:und_len], cos[und_len:], sin[und_len:])
        und_seq = hidden_states[:und_len]
        gen_seq = hidden_states[und_len:]
        und_seq, gen_seq = self.infer_layers(block_weights.layers, und_seq, gen_seq, rotary_emb)
        return Cosmos3TransformerInferModuleOutput(und_seq=und_seq, gen_seq=gen_seq)
