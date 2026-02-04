from copy import deepcopy
from typing import Optional

import torch
import torch.nn.functional as F

try:
    import flash_attn  # noqa: F401
    from flash_attn.flash_attn_interface import flash_attn_varlen_func
except ImportError:
    flash_attn_varlen_func = None

from lightx2v.common.transformer_infer.transformer_infer import BaseTransformerInfer
from lightx2v.models.networks.bagel.model_io import NaiveCache
from lightx2v.utils.envs import *
from lightx2v_platform.base.global_var import AI_DEVICE


# Copied from transformers.models.llama.modeling_llama.rotate_half
def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


# Copied from transformers.models.llama.modeling_llama.apply_rotary_pos_emb
def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class BagelTransformerInfer(BaseTransformerInfer):
    def __init__(self, config, llm_config):
        self.config = config
        self.llm_config = llm_config
        self.num_layers = llm_config["num_hidden_layers"]
        self.use_moe = "Mo" in llm_config["layer_module"]
        self.hidden_size = llm_config["hidden_size"]
        self.num_heads = llm_config["num_attention_heads"]
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = llm_config["num_key_value_heads"]
        self.init_kv_cache()

    def set_scheduler(self, scheduler):
        self.scheduler = scheduler

    def init_gen_context(self):
        gen_context = {
            "kv_lens": [0],
            "ropes": [0],
            "past_key_values": NaiveCache(self.num_layers),
        }
        return gen_context

    def init_kv_cache(self):
        self.gen_context = self.init_gen_context()
        self.cfg_text_context = deepcopy(self.gen_context)
        self.cfg_img_context = deepcopy(self.gen_context)

    def self_attn(
        self,
        weights,
        packed_query_sequence,
        query_lens,
        packed_query_position_embeddings,
        packed_query_indexes,
        past_key_values,
        key_values_lens,
        packed_key_value_indexes,
        update_past_key_values,
        is_causal,
        mode,
        packed_vae_token_indexes,
        packed_text_indexes,
        layer_idx,
    ):
        if mode == "und":
            packed_query_states = weights.q_proj.apply(packed_query_sequence).view(-1, self.num_heads, self.head_dim)
            packed_key_states = weights.k_proj.apply(packed_query_sequence).view(-1, self.num_key_value_heads, self.head_dim)
            packed_value_states = weights.v_proj.apply(packed_query_sequence).view(-1, self.num_key_value_heads, self.head_dim)
            packed_query_states = weights.q_norm.apply(packed_query_states)
            packed_key_states = weights.k_norm.apply(packed_key_states)
        elif mode == "gen":
            packed_text_indexes = packed_text_indexes.to(AI_DEVICE)
            packed_vae_token_indexes = packed_vae_token_indexes.to(AI_DEVICE)

            packed_query_sequence = packed_query_sequence.to(torch.bfloat16)
            packed_query_states = packed_query_sequence.new_zeros((packed_query_sequence.shape[0], self.num_heads * self.head_dim))
            packed_key_states = packed_query_sequence.new_zeros((packed_query_sequence.shape[0], self.num_key_value_heads * self.head_dim))
            packed_value_states = packed_query_sequence.new_zeros((packed_query_sequence.shape[0], self.num_key_value_heads * self.head_dim))

            packed_text_query_sequence = packed_query_sequence[packed_text_indexes]
            packed_vae_query_sequence = packed_query_sequence[packed_vae_token_indexes]

            packed_query_states[packed_text_indexes] = weights.q_proj.apply(packed_text_query_sequence)
            packed_query_states[packed_vae_token_indexes] = weights.q_proj_moe_gen.apply(packed_vae_query_sequence)

            packed_key_states[packed_text_indexes] = weights.k_proj.apply(packed_text_query_sequence)
            packed_key_states[packed_vae_token_indexes] = weights.k_proj_moe_gen.apply(packed_vae_query_sequence)

            packed_value_states[packed_text_indexes] = weights.v_proj.apply(packed_text_query_sequence)
            packed_value_states[packed_vae_token_indexes] = weights.v_proj_moe_gen.apply(packed_vae_query_sequence)

            packed_query_states = packed_query_states.view(-1, self.num_heads, self.head_dim)
            packed_key_states = packed_key_states.view(-1, self.num_key_value_heads, self.head_dim)
            packed_value_states = packed_value_states.view(-1, self.num_key_value_heads, self.head_dim)

            packed_query_states = packed_query_states.to(torch.float32)
            packed_query_states[packed_text_indexes] = weights.q_norm.apply(packed_query_states[packed_text_indexes], moe_gen=True)
            packed_query_states[packed_vae_token_indexes] = weights.q_norm_moe_gen.apply(packed_query_states[packed_vae_token_indexes], moe_gen=True)

            packed_key_states = packed_key_states.to(torch.float32)
            packed_key_states[packed_text_indexes] = weights.k_norm.apply(packed_key_states[packed_text_indexes], moe_gen=True)
            packed_key_states[packed_vae_token_indexes] = weights.k_norm_moe_gen.apply(packed_key_states[packed_vae_token_indexes], moe_gen=True)

        packed_cos, packed_sin = packed_query_position_embeddings
        packed_query_states, packed_key_states = apply_rotary_pos_emb(packed_query_states, packed_key_states, packed_cos, packed_sin, unsqueeze_dim=1)

        packed_query_states = packed_query_states.to(torch.bfloat16)
        packed_key_states = packed_key_states.to(torch.bfloat16)
        packed_value_states = packed_value_states.to(torch.bfloat16)

        if past_key_values is not None and past_key_values.key_cache[layer_idx] is not None:
            past_key_states = past_key_values.key_cache[layer_idx]
            past_value_states = past_key_values.value_cache[layer_idx]

            seqlens = sum(query_lens) + sum(key_values_lens)
            merged_key_states = past_key_states.new_zeros(size=[seqlens, self.num_key_value_heads, self.head_dim])
            merged_value_states = past_key_states.new_zeros(size=[seqlens, self.num_key_value_heads, self.head_dim])
            merged_key_states[packed_query_indexes] = packed_key_states
            merged_key_states[packed_key_value_indexes] = past_key_states
            merged_value_states[packed_query_indexes] = packed_value_states
            merged_value_states[packed_key_value_indexes] = past_value_states
            key_values_lens = key_values_lens + query_lens
        else:
            merged_key_states = packed_key_states
            merged_value_states = packed_value_states
            key_values_lens = query_lens

        cu_seqlens_q = torch.nn.functional.pad(torch.cumsum(query_lens, dim=0), (1, 0)).to(AI_DEVICE)
        cu_seqlens_k = torch.nn.functional.pad(torch.cumsum(key_values_lens, dim=0), (1, 0)).to(AI_DEVICE)

        packed_attn_output = flash_attn_varlen_func(
            q=packed_query_states,
            k=merged_key_states,
            v=merged_value_states,
            cu_seqlens_q=cu_seqlens_q.to(torch.int32),
            cu_seqlens_k=cu_seqlens_k.to(torch.int32),
            max_seqlen_q=max(query_lens).item(),
            max_seqlen_k=max(key_values_lens).item(),
            causal=is_causal,
        )
        packed_attn_output = packed_attn_output.reshape(-1, self.hidden_size)

        if mode == "und":
            packed_attn_output = weights.o_proj.apply(packed_attn_output)
        elif mode == "gen":
            packed_attn_output[packed_text_indexes] = weights.o_proj.apply(packed_attn_output[packed_text_indexes])
            packed_attn_output[packed_vae_token_indexes] = weights.o_proj_moe_gen.apply(packed_attn_output[packed_vae_token_indexes])

        if update_past_key_values:
            past_key_values.key_cache[layer_idx] = merged_key_states
            past_key_values.value_cache[layer_idx] = merged_value_states

        return packed_attn_output, past_key_values

    def mlp(self, weights, hidden_state):
        gate_proj = weights.gate_proj.apply(hidden_state)
        up_proj = weights.up_proj.apply(hidden_state)
        h0 = F.silu(gate_proj) * up_proj
        h1 = weights.down_proj.apply(h0)
        return h1

    def mlp_moe_gen(self, weights, hidden_state):
        gate_proj = weights.gate_proj.apply(hidden_state)
        up_proj = weights.up_proj.apply(hidden_state)
        h0 = F.silu(gate_proj) * up_proj
        h1 = weights.down_proj.apply(h0)
        return h1

    def decoder_layer(
        self,
        block_weight,
        layer_idx,
        packed_query_sequence: torch.Tensor,
        query_lens: torch.Tensor,
        packed_query_position_embeddings: torch.Tensor,
        packed_query_indexes: torch.Tensor,
        past_key_values: Optional[NaiveCache] = None,
        key_values_lens: Optional[torch.Tensor] = None,
        packed_key_value_indexes: Optional[torch.Tensor] = None,
        update_past_key_values=True,
        is_causal=True,
        mode="und",
        packed_vae_token_indexes=None,
        packed_text_indexes=None,
    ):
        enable_taylorseer = getattr(self, "enable_taylorseer", False)

        if not enable_taylorseer or (enable_taylorseer and self.current["type"] == "full"):
            residual = packed_query_sequence
            if mode == "und":
                packed_query_sequence = block_weight.input_layernorm.apply(packed_query_sequence)

            elif mode == "gen":
                packed_query_sequence_ = torch.zeros_like(packed_query_sequence)
                packed_query_sequence_[packed_text_indexes] = block_weight.input_layernorm.apply(packed_query_sequence[packed_text_indexes])
                packed_query_sequence_[packed_vae_token_indexes] = block_weight.input_layernorm_moe_gen.apply(packed_query_sequence[packed_vae_token_indexes])
                packed_query_sequence = packed_query_sequence_

            # Self Attention
            packed_query_sequence, past_key_values = self.self_attn(
                weights=block_weight.self_attn,
                packed_query_sequence=packed_query_sequence,
                query_lens=query_lens,
                packed_query_position_embeddings=packed_query_position_embeddings,
                packed_query_indexes=packed_query_indexes,
                past_key_values=past_key_values,
                key_values_lens=key_values_lens,
                packed_key_value_indexes=packed_key_value_indexes,
                update_past_key_values=update_past_key_values,
                is_causal=is_causal,
                mode=mode,
                packed_vae_token_indexes=packed_vae_token_indexes,
                packed_text_indexes=packed_text_indexes,
                layer_idx=layer_idx,
            )

            packed_query_sequence = residual + packed_query_sequence

            # Fully Connected
            residual = packed_query_sequence
            if mode == "und":
                packed_query_sequence = block_weight.post_attention_layernorm.apply(packed_query_sequence)
                packed_query_sequence = self.mlp(block_weight.mlp, packed_query_sequence)
            elif mode == "gen":
                packed_text_query_sequence = packed_query_sequence[packed_text_indexes]
                packed_vae_query_sequence = packed_query_sequence[packed_vae_token_indexes]
                packed_text_query_sequence = block_weight.post_attention_layernorm.apply(packed_text_query_sequence).to(torch.bfloat16)
                packed_vae_query_sequence = block_weight.post_attention_layernorm_moe_gen.apply(packed_vae_query_sequence).to(torch.bfloat16)

                packed_query_sequence_ = torch.zeros_like(packed_query_sequence).to(torch.bfloat16)
                packed_query_sequence_[packed_text_indexes] = self.mlp(block_weight.mlp, packed_text_query_sequence)
                packed_query_sequence_[packed_vae_token_indexes] = self.mlp_moe_gen(block_weight.mlp_moe_gen, packed_vae_query_sequence)
                packed_query_sequence = packed_query_sequence_

            packed_query_sequence = residual + packed_query_sequence
        return packed_query_sequence, past_key_values

    def infer(
        self,
        block_weights,
        packed_query_sequence: torch.Tensor,
        query_lens: torch.Tensor,
        packed_query_indexes: torch.Tensor,
        past_key_values: Optional[NaiveCache] = None,
        key_values_lens: Optional[torch.Tensor] = None,
        packed_key_value_indexes: Optional[torch.Tensor] = None,
        update_past_key_values=True,
        is_causal=True,
        mode="und",
        packed_vae_token_indexes=None,
        packed_text_indexes=None,
        packed_query_position_embeddings=None,
        enable_taylorseer=False,
    ):
        for layer_idx, block_weight in enumerate(block_weights):
            if enable_taylorseer:
                assert NotImplementedError
            packed_query_sequence, past_key_values = self.decoder_layer(
                block_weight=block_weight,
                packed_query_sequence=packed_query_sequence,
                query_lens=query_lens,
                packed_query_position_embeddings=packed_query_position_embeddings,
                packed_query_indexes=packed_query_indexes,
                past_key_values=past_key_values,
                key_values_lens=key_values_lens,
                packed_key_value_indexes=packed_key_value_indexes,
                update_past_key_values=update_past_key_values,
                is_causal=is_causal,
                layer_idx=layer_idx,
                mode=mode,
                packed_text_indexes=packed_text_indexes,
                packed_vae_token_indexes=packed_vae_token_indexes,
            )

        return packed_query_sequence, past_key_values
