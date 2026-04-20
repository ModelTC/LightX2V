import torch
import torch.nn.functional as F

# from flashinfer.activation import silu_and_mul as flashinfer_silu_and_mul
try:
    from flashinfer.fused_moe import cutlass_fused_moe as flashinfer_cutlass_fused_moe
except ImportError:
    flashinfer_cutlass_fused_moe = None


from lightx2v.common.transformer_infer.transformer_infer import BaseTransformerInfer
from lightx2v.models.networks.neopp.infer.kv_cache_manager import KVCacheManager
from lightx2v.utils.profiler import *


class NeoppTransformerInfer(BaseTransformerInfer):
    def __init__(self, config):
        self.config = config
        llm_config = config["llm_config"]
        self.num_layers = llm_config["num_hidden_layers"]
        self.hidden_size = llm_config["hidden_size"]
        self.num_heads = llm_config["num_attention_heads"]
        self.num_kv_heads = llm_config["num_key_value_heads"]
        self.head_dim = llm_config["head_dim"]
        self.num_key_value_groups = self.num_heads // self.num_kv_heads
        self.scaling = self.head_dim**-0.5
        self.use_triton_qknorm_rope = config.get("use_triton_qknorm_rope", True)
        self.version = config.get("version", "moe")
        if self.version == "moe":
            self.num_experts_per_tok = llm_config["num_experts_per_tok"]
            self.norm_topk_prob = llm_config.get("norm_topk_prob", True)
            self._mlp_forward = self._sparse_moe
        else:
            self._mlp_forward = self._dense_mlp
        if self.config["seq_parallel"]:
            self.seq_p_group = self.config.get("device_mesh").get_group(mesh_dim="seq_p")
        else:
            self.seq_p_group = None
        self.cross_attn_type = self.config["attn_type"]
        self.kv_cache = KVCacheManager()

    @torch.no_grad()
    def infer(self, weights, pre_infer_out, inputs):
        infer_pass = self.scheduler.infer_pass

        past_key_values = inputs[f"past_key_values_{infer_pass}"]  # [layers, 2, past_seq, num_kv_heads, head_dim]
        cos_sin = inputs[f"cos_sin_{infer_pass}"]

        hidden_states = pre_infer_out.image_embeds.squeeze(0)  # [seq, hidden]

        self._kvcache_len = past_key_values.shape[2]
        seq_len_q = hidden_states.shape[0]
        seq_len_k = self._kvcache_len + seq_len_q
        _cache_key = infer_pass
        if not hasattr(self, "_seqlen_cache"):
            self._seqlen_cache = {}
        if self._seqlen_cache.get(_cache_key, {}).get("seqlens") != (seq_len_q, seq_len_k):
            self._seqlen_cache[_cache_key] = {
                "seqlens": (seq_len_q, seq_len_k),
                "cu_q": torch.tensor([0, seq_len_q], dtype=torch.int32),
                "cu_k": torch.tensor([0, seq_len_k], dtype=torch.int32),
            }
        self._cu_seqlens_q = self._seqlen_cache[_cache_key]["cu_q"]
        self._cu_seqlens_k = self._seqlen_cache[_cache_key]["cu_k"]
        self._max_seqlen_q = seq_len_q
        self._max_seqlen_k = seq_len_k

        self.kv_cache.prepare(past_key_values, seq_len_q, self.num_layers, self.scheduler.infer_condition)

        # with ProfilingContext4DebugL1("Decoder Blocks"):
        for layer_idx, block_weight in enumerate(weights.blocks):
            hidden_states = self._decoder_layer(block_weight, layer_idx, hidden_states, cos_sin)

        hidden_states = weights.norm_mot_gen.apply(hidden_states)
        hidden_states = self._fm_head(weights.fm_head, hidden_states)
        return hidden_states.unsqueeze(0)

    # @ProfilingContext4DebugL1("Decoder Layer")
    def _decoder_layer(self, block_weight, layer_idx, hidden_states, cos_sin):
        residual = hidden_states
        hidden_states = block_weight.input_layernorm_mot_gen.apply(hidden_states)

        hidden_states = self._self_attn(block_weight.self_attn, layer_idx, hidden_states, cos_sin)
        hidden_states = residual + hidden_states

        residual = hidden_states
        gen_hidden = block_weight.post_attention_layernorm_mot_gen.apply(hidden_states)
        gen_hidden = self._mlp_forward(block_weight.mlp_mot_gen, gen_hidden)
        hidden_states = residual + gen_hidden

        return hidden_states

    # @ProfilingContext4DebugL1("Self Attn")
    def _self_attn(self, attn_w, layer_idx, hidden_states, cos_sin):
        query_states = attn_w.q_proj_mot_gen.apply(hidden_states)
        query_states = query_states.view(-1, self.num_heads, self.head_dim)  # [seq, num_heads, head_dim]

        key_states = attn_w.k_proj_mot_gen.apply(hidden_states)
        key_states = key_states.view(-1, self.num_kv_heads, self.head_dim)  # [seq, num_kv_heads, head_dim]

        if self.use_triton_qknorm_rope:
            # Triton fused path: dual-RMSNorm + 3D Neox-RoPE in one kernel launch (in-place).
            attn_w.qk_norm.apply(query_states, key_states, cos_sin)
        else:
            # Pure torch path: expanded dual-RMSNorm + 3D Neox-RoPE.
            query_states, key_states = self._qk_norm_rope_torch(attn_w, query_states, key_states, cos_sin, hidden_states.dtype)

        value_states = attn_w.v_proj_mot_gen.apply(hidden_states)
        value_states = value_states.view(-1, self.num_kv_heads, self.head_dim)  # [seq, num_kv_heads, head_dim]

        key_states, value_states = self.kv_cache.update(layer_idx, key_states, value_states)

        attn_output = self._compute_attn(attn_w, query_states, key_states, value_states)

        attn_output = attn_w.o_proj_mot_gen.apply(attn_output)
        return attn_output

    def _qk_norm_rope_torch(self, attn_w, query_states, key_states, cos_sin, out_dtype):
        """Pure-torch dual-RMSNorm + 3D Neox-RoPE for Q and K.

        Equivalent to the triton fused_qk_norm_3drope kernel.
        head_dim layout (input and output): [t_h1 | t_h2 | h_h1 | h_h2 | w_h1 | w_h2]
          where half=head_dim//2, quarter=half//2, eighth=quarter//2.

        cos_t : [1, seq, half]    (cat(freqs_t, freqs_t); unique part is [:quarter])
        cos_h : [1, seq, quarter] (cat(freqs_h, freqs_h); unique part is [:eighth])
        cos_w : [1, seq, quarter] (cat(freqs_w, freqs_w); unique part is [:eighth])
        """
        cos_t, sin_t, cos_h, sin_h, cos_w, sin_w = cos_sin
        half = self.head_dim // 2
        quarter = self.head_dim // 4
        eighth = self.head_dim // 8

        # Extract the unique (non-repeated) half of each cos/sin; unsqueeze for head broadcast.
        c_t = cos_t.squeeze(0)[:, :quarter].unsqueeze(1).float()  # [seq, 1, quarter]
        s_t = sin_t.squeeze(0)[:, :quarter].unsqueeze(1).float()
        c_h = cos_h.squeeze(0)[:, :eighth].unsqueeze(1).float()  # [seq, 1, eighth]
        s_h = sin_h.squeeze(0)[:, :eighth].unsqueeze(1).float()
        c_w = cos_w.squeeze(0)[:, :eighth].unsqueeze(1).float()  # [seq, 1, eighth]
        s_w = sin_w.squeeze(0)[:, :eighth].unsqueeze(1).float()

        def _norm_rope(x, norm_t, norm_hw):
            x_t = norm_t.apply(x[..., :half])  # RMSNorm t-segment  → [seq, heads, half]
            x_hw = norm_hw.apply(x[..., half:])  # RMSNorm hw-segment → [seq, heads, half]
            x_t_h1, x_t_h2 = x_t[..., :quarter].float(), x_t[..., quarter:].float()
            x_h = x_hw[..., :quarter]
            x_w = x_hw[..., quarter:]
            x_h_h1, x_h_h2 = x_h[..., :eighth].float(), x_h[..., eighth:].float()
            x_w_h1, x_w_h2 = x_w[..., :eighth].float(), x_w[..., eighth:].float()
            return torch.cat(
                [
                    x_t_h1 * c_t - x_t_h2 * s_t,
                    x_t_h2 * c_t + x_t_h1 * s_t,
                    x_h_h1 * c_h - x_h_h2 * s_h,
                    x_h_h2 * c_h + x_h_h1 * s_h,
                    x_w_h1 * c_w - x_w_h2 * s_w,
                    x_w_h2 * c_w + x_w_h1 * s_w,
                ],
                dim=-1,
            ).to(out_dtype)

        query_states = _norm_rope(query_states, attn_w.q_norm_mot_gen, attn_w.q_norm_hw_mot_gen)
        key_states = _norm_rope(key_states, attn_w.k_norm_mot_gen, attn_w.k_norm_hw_mot_gen)
        return query_states, key_states

    # @ProfilingContext4DebugL1("Compute Attn")
    def _compute_attn(self, attn_w, query_states, key_states, value_states):
        if self.config["seq_parallel"]:
            attn_output = attn_w.cross_attn_parallel.apply(
                q=query_states,
                k=key_states,
                v=value_states,
                slice_qkv_len=self._kvcache_len,
                cu_seqlens_qkv=self._cu_seqlens_k,
                attention_module=attn_w.cross_attn,
                attention_type=self.cross_attn_type,
                seq_p_group=self.seq_p_group,
                img_first=False,
                q_only_img=True,
            )
        else:
            attn_output = attn_w.cross_attn.apply(
                q=query_states,
                k=key_states,
                v=value_states,
                cu_seqlens_q=self._cu_seqlens_q,
                cu_seqlens_kv=self._cu_seqlens_k,
                max_seqlen_q=self._max_seqlen_q,
                max_seqlen_kv=self._max_seqlen_k,
            )
        return attn_output

    # @ProfilingContext4DebugL1("Sparse MoE")
    def _sparse_moe(self, moe_w, hidden_states):
        router_logits = moe_w.gate.apply(hidden_states)
        if self.norm_topk_prob:
            _, selected_experts = torch.topk(router_logits, self.num_experts_per_tok, dim=-1, sorted=False)
            routing_weights = F.softmax(router_logits.gather(1, selected_experts).float(), dim=-1)
        else:
            routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
            routing_weights, selected_experts = torch.topk(routing_weights, self.num_experts_per_tok, dim=-1)

        output = flashinfer_cutlass_fused_moe(
            hidden_states if hidden_states.is_contiguous() else hidden_states.contiguous(),
            selected_experts.to(torch.int32),
            routing_weights,
            moe_w._fi_fc1_weight,
            moe_w._fi_fc2_weight,
            hidden_states.dtype,
            quant_scales=None,
        )[0]

        return output

    # @ProfilingContext4DebugL1("FM Head")
    def _fm_head(self, fm_head_w, hidden_states):
        hidden_states = fm_head_w.fm_head_0.apply(hidden_states)
        hidden_states = F.gelu(hidden_states)
        hidden_states = fm_head_w.fm_head_2.apply(hidden_states)
        return hidden_states

    def _dense_mlp(self, mlp_w, hidden_states):
        up_states = mlp_w.up_proj.apply(hidden_states)
        gate_states = mlp_w.gate_proj.apply(hidden_states)
        intermediate_states = F.silu(gate_states) * up_states
        return mlp_w.down_proj.apply(intermediate_states)

    # def _dense_mlp(self, mlp_w, hidden_states):
    #     gate_up_states = torch.mm(hidden_states, mlp_w._fi_gate_up_weight)
    #     intermediate_states = flashinfer_silu_and_mul(gate_up_states)
    #     return mlp_w.down_proj.apply(intermediate_states)
