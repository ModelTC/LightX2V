import torch
import torch.nn.functional as F

from lightx2v.common.ops.norm.rms_norm_weight import apply_qk_rms_norm
from lightx2v.utils.registry_factory import ROPE_REGISTER


class QwenImage21TransformerInfer:
    def __init__(self, config):
        self.config = config
        self.heads = config["num_attention_heads"]
        self.use_fused_qk_rms_norm = config.get("fused_qk_rms_norm", True)
        self.use_fused_block_ops = config.get("fused_block_ops", True) and config.get("modulate_type", "triton") == "triton"
        self.fused_block_min_tokens = config.get("fused_block_min_tokens", 1024)
        if self.use_fused_block_ops:
            from . import triton_ops

            self.block_ops = triton_ops
        self.rope = ROPE_REGISTER[config.get("rope_type", "torch_complex_rope")](layout="interleaved", compute_dtype=torch.float32)
        self.rope.set_config(config)
        self.seq_parallel = config.get("seq_parallel", False)
        if self.seq_parallel:
            self.seq_p_group = config["device_mesh"].get_group(mesh_dim="seq_p")
            parallel = config["parallel"]
            self.seq_p_prepost_backend = parallel.get("seq_p_prepost_backend", "torch")
            self.seq_p_a2a_backend = parallel.get("seq_p_a2a_backend", "torch")
            self.seq_p_quant_scheme = parallel.get("seq_p_quant_scheme")
            self.seq_p_tensor_fusion = parallel.get("seq_p_tensor_fusion", False)
            self.seq_p_head_parallel = parallel.get("seq_p_head_parallel", False)
        else:
            self.seq_p_group = None

    def set_scheduler(self, scheduler):
        self.scheduler = scheduler

    def _modulation(self, state):
        # One timestep per stream; the same modulation is shared by all blocks.
        scale1, gate1, scale2, gate2 = state.modulation.chunk(4, -1)
        return 1 + scale1, gate1.tanh(), 1 + scale2, gate2.tanh()

    def _qkv(self, block, x, scale, state):
        h = block.norm1.apply(x) * scale
        q = block.q.apply(h).reshape(-1, self.heads, self.config["attention_head_dim"])
        k = block.k.apply(h).reshape_as(q)
        v = block.v.apply(h).reshape_as(q)
        q, k = apply_qk_rms_norm(q, k, block.norm_q, block.norm_k, use_triton=self.use_fused_qk_rms_norm)
        q, k = self.rope.apply(q, k, state.rotary, positions=state.rotary_positions)
        return q, k, v

    def _finish_block(self, block, x, attention, modulation):
        _, gate1, scale2, gate2 = modulation
        # Short sequences are launch-bound; eager Torch is faster there.
        if self.use_fused_block_ops and x.shape[0] >= self.fused_block_min_tokens and not torch.compiler.is_compiling():
            x, h = self.block_ops.fused_residual_norm_scale(x, block.out.apply(attention), gate1, scale2, block.norm2.eps)
            hidden = self.block_ops.fused_silu_mul(block.gate.apply(h), block.up.apply(h))
            return self.block_ops.fused_residual_add(x, block.down.apply(hidden), gate2)

        x = x + gate1 * block.out.apply(attention)
        h = block.norm2.apply(x) * scale2
        x = x + gate2 * block.down.apply(F.silu(block.gate.apply(h)) * block.up.apply(h))
        return x.clamp(-65504, 65504) if x.dtype == torch.float16 else x

    def prefill(self, weights, state, cache):
        """Run the condition prefix once and store every layer's K/V."""
        x = state.hidden_states
        modulation = self._modulation(state)
        for index, block in enumerate(weights.blocks):
            q, k, v = self._qkv(block, x, modulation[0], state)
            cache.store_kv(k, v, index)
            attention = torch.empty_like(x)
            for begin, end, is_text in state.layout.segments:
                mask = None
                if is_text:
                    mask = torch.arange(end, device=x.device)[None] <= torch.arange(begin, end, device=x.device)[:, None]
                op = block.prefix_attention if is_text else block.attention
                attention[begin:end] = op.apply(q[begin:end], k[:end], v[:end], attn_mask=mask)
            x = self._finish_block(block, x, attention, modulation)

    def infer(self, weights, state, cache):
        """Denoise only target tokens, attending to the prefilled condition K/V."""
        if not cache.is_ready():
            raise RuntimeError("Condition KV must be prefilled before denoising")
        x = state.hidden_states
        modulation = self._modulation(state)
        for index, block in enumerate(weights.blocks):
            q, k, v = self._qkv(block, x, modulation[0], state)
            cached_k = cache.k_cache(index)
            cached_v = cache.v_cache(index)
            if self.seq_parallel:
                attention, _ = block.calculate_parallel.apply(
                    q=q,
                    k=k,
                    v=v,
                    aux_q=None,
                    aux_k=cached_k,
                    aux_v=cached_v,
                    attention_module=block.attention,
                    seq_p_group=self.seq_p_group,
                    prepost_backend=self.seq_p_prepost_backend,
                    a2a_backend=self.seq_p_a2a_backend,
                    quant_scheme=self.seq_p_quant_scheme,
                    tensor_fusion=self.seq_p_tensor_fusion,
                    head_parallel=self.seq_p_head_parallel,
                    aux_first=True,
                    attention_kwargs={},
                )
            else:
                k = torch.cat((cached_k, k))
                v = torch.cat((cached_v, v))
                attention = block.attention.apply(q, k, v)
            x = self._finish_block(block, x, attention, modulation)
        return x
