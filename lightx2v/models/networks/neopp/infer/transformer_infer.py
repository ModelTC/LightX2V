import torch
import torch.nn.functional as F
from loguru import logger

# from flashinfer.activation import silu_and_mul as flashinfer_silu_and_mul
try:
    from flashinfer.fused_moe import cutlass_fused_moe as flashinfer_cutlass_fused_moe
except ImportError:
    flashinfer_cutlass_fused_moe = None

try:
    from magi_compiler import magi_compile, magi_register_custom_op
except ImportError:
    magi_compile = None
    magi_register_custom_op = None

from lightx2v.common.magi_custom_op_mode import configure_dynamo_for_magi_compile
from lightx2v.common.transformer_infer.transformer_infer import BaseTransformerInfer
from lightx2v.models.networks.neopp.infer.kv_cache_manager import KVCacheManager
from lightx2v.utils.profiler import *

# Register neopp::kv_update as a PyTorch custom op via torch.library.
# We use torch.library (define + impl) instead of magi_register_custom_op
# because the latter internally calls torch.library.custom_op, which has
# a known extra dispatch overhead when mutates_args is set (PyTorch #139500).
# torch.library.define + impl does not suffer from this overhead.
_LIB = torch.library.Library("neopp", "FRAGMENT")
_LIB.define("kv_update(Tensor kv_buf, int layer_idx, Tensor key_states, Tensor value_states) -> (Tensor, Tensor)")


@torch.library.impl(_LIB, "kv_update", "CUDA")
def _kv_update_impl(kv_buf, layer_idx, key_states, value_states):
    past_seq = kv_buf.shape[2] - key_states.shape[0]
    end = past_seq + key_states.shape[0]
    kv_buf[layer_idx, 0, past_seq:end] = key_states
    kv_buf[layer_idx, 1, past_seq:end] = value_states
    return kv_buf[layer_idx, 0, :end], kv_buf[layer_idx, 1, :end]


@torch.library.register_fake("neopp::kv_update")
def _kv_update_meta(kv_buf, layer_idx, key_states, value_states):
    return (
        torch.empty_like(kv_buf[layer_idx, 0]),
        torch.empty_like(kv_buf[layer_idx, 1]),
    )


# Tell MagiCompiler to split the FX graph at neopp::kv_update so that
# Inductor does not try to fuse the slice-scatter with surrounding ops.
try:
    from magi_compiler.config import get_compile_config

    _SPLITTING_OP_NAME = "neopp::kv_update"
    if _SPLITTING_OP_NAME not in get_compile_config().splitting_ops:
        get_compile_config().splitting_ops.append(_SPLITTING_OP_NAME)
except ImportError:
    pass


class NeoppTransformerInfer(BaseTransformerInfer, torch.nn.Module):
    __constants__ = ["num_layers", "num_heads", "num_kv_heads", "head_dim"]

    def __init__(self, config):
        torch.nn.Module.__init__(self)
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
        self.kv_cache = KVCacheManager()

        # MagiCompiler enable/disable switch
        self.use_magi_compile = config.get("use_magi_compile", False)
        if self.use_magi_compile and magi_compile is None:
            logger.warning("use_magi_compile=True but magi_compiler is not available, using eager mode")
            self.use_magi_compile = False
        if self.use_magi_compile:
            configure_dynamo_for_magi_compile()
            logger.info("Using Magi Compile (per-layer decoder, split at kv_update)")

    @torch.no_grad()
    def infer(self, weights, pre_infer_out, inputs):
        pass_key = "cond" if self.scheduler.infer_condition else "uncond"
        past_key_values = inputs[f"past_key_values_{pass_key}"]  # [layers, 2, past_seq, num_kv_heads, head_dim]
        cos_sin = inputs[f"cos_sin_{pass_key}"]
        hidden_states = pre_infer_out.image_embeds.squeeze(0)  # [seq, hidden]

        hidden_states = self.infer_without_offload(weights.blocks, hidden_states, cos_sin, past_key_values)

        hidden_states = weights.norm_mot_gen.apply(hidden_states)
        hidden_states = self._fm_head(weights.fm_head, hidden_states)
        return hidden_states.unsqueeze(0)

    def infer_without_offload(self, blocks, hidden_states, cos_sin, past_key_values):
        seq_len_q = hidden_states.shape[0]
        kvcache_len = past_key_values.shape[2]
        seq_len_k = kvcache_len + seq_len_q

        # Allocate the KV buffer fresh each step so Dynamo sees it as a local
        # tensor inside the compiled region.
        self.kv_cache.clear()
        self.kv_cache.prepare(past_key_values, seq_len_q)

        kv_buf = self.kv_cache._kv_buf
        cos_t, sin_t, cos_h, sin_h, cos_w, sin_w = cos_sin
        for layer_idx, block_weight in enumerate(blocks):
            if self.use_magi_compile:
                hidden_states = self._decoder_layer_magi(
                    block_weight,
                    layer_idx,
                    hidden_states,
                    cos_t,
                    sin_t,
                    cos_h,
                    sin_h,
                    cos_w,
                    sin_w,
                    kv_buf,
                )
            else:
                hidden_states = self._decoder_layer(block_weight, layer_idx, hidden_states, cos_sin, kv_buf)
        return hidden_states

    if magi_compile is not None:

        def _magi_config_patch(c):
            return c.model_copy(
                update={
                    "disable_cache": True,
                }
            )

        @magi_compile(
            dynamic_arg_dims={
                "hidden_states": 0,
                "kv_buf": 2,
                "cos_t": 1,
                "sin_t": 1,
                "cos_h": 1,
                "sin_h": 1,
                "cos_w": 1,
                "sin_w": 1,
            },
            config_patch=_magi_config_patch,
        )
        def _decoder_layer_magi(
            self,
            block_weight,
            layer_idx,
            hidden_states,
            cos_t,
            sin_t,
            cos_h,
            sin_h,
            cos_w,
            sin_w,
            kv_buf,
        ):
            cos_sin = (cos_t, sin_t, cos_h, sin_h, cos_w, sin_w)
            return self._decoder_layer(block_weight, layer_idx, hidden_states, cos_sin, kv_buf)

    # @ProfilingContext4DebugL1("Decoder Layer")
    def _decoder_layer(self, block_weight, layer_idx, hidden_states, cos_sin, kv_buf=None):
        residual = hidden_states
        hidden_states = block_weight.input_layernorm_mot_gen.apply(hidden_states)

        hidden_states = self._self_attn(block_weight.self_attn, layer_idx, hidden_states, cos_sin, kv_buf)
        hidden_states = residual + hidden_states

        residual = hidden_states
        gen_hidden = block_weight.post_attention_layernorm_mot_gen.apply(hidden_states)
        gen_hidden = self._mlp_forward(block_weight.mlp_mot_gen, gen_hidden)
        hidden_states = residual + gen_hidden

        return hidden_states

    # @ProfilingContext4DebugL1("Self Attn")
    def _self_attn(self, attn_w, layer_idx, hidden_states, cos_sin, kv_buf=None):
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

        if kv_buf is None:
            kv_buf = self.kv_cache._kv_buf

        # Custom op: forces MagiCompiler to split the FX graph at this op,
        # isolating the slice-scatter from the surrounding compiled regions.
        key_states, value_states = torch.ops.neopp.kv_update(kv_buf, layer_idx, key_states, value_states)

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
        # All sequence-related scalars are derived from tensor shapes so that
        # Dynamo can trace them as symbolic ints instead of reading them from
        # self._xxx attributes (which would create graph breaks).
        seq_len_q = query_states.shape[0]
        seq_len_k = key_states.shape[0]

        if self.config["seq_parallel"]:
            kvcache_len = seq_len_k - seq_len_q
            # Pass cu_seqlens_qkv as list[int] so that Dynamo sees a plain Python
            # int when UlyssesAttnWeight.apply does cu_seqlens_qkv[1].  A Tensor
            # index in fake-tensor mode returns a 0-d Tensor which Dynamo treats
            # as data-dependent and refuses to trace through reshape.
            attn_output = attn_w.cross_attn_parallel.apply(
                q=query_states,
                k=key_states,
                v=value_states,
                slice_qkv_len=kvcache_len,
                cu_seqlens_qkv=[0, seq_len_k],
                attention_module=attn_w.cross_attn,
                seq_p_group=self.seq_p_group,
                img_first=False,
                q_only_img=True,
            )
        else:
            attn_output = attn_w.cross_attn.apply(
                q=query_states,
                k=key_states,
                v=value_states,
                cu_seqlens_q=torch.tensor([0, seq_len_q], dtype=torch.int32),
                cu_seqlens_kv=torch.tensor([0, seq_len_k], dtype=torch.int32),
                max_seqlen_q=seq_len_q,
                max_seqlen_kv=seq_len_k,
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
