import torch.nn.functional as F

from lightx2v.common.ops.attn.utils.all2all import all2all_head2seq, all2all_seq2head
from lightx2v.common.ops.norm.rms_norm_weight import RMSWeightTPTorch
from lightx2v.models.networks.wan.infer.autoregressive.causal_transformer_infer import WanCausalTransformerInfer

from .rope import MatrixGame4AttentionState, SinkRoPE, apply_dynamic_causal_rope, apply_dynamic_causal_rope_single


class WanMatrixGame4TransformerInfer(WanCausalTransformerInfer):
    def __init__(self, config):
        ar = config["ar_config"]
        if ar.get("step_kv_cache", False) or ar.get("kv_cache_scheme", "rolling") != "rolling":
            raise ValueError("Matrix Game 4 requires a rolling KV cache shared by denoising steps")
        if ar.get("sink_size", 1) != 1 or ar.get("num_frame_per_chunk", 1) != 1:
            raise ValueError("Matrix Game 4 requires one sink latent frame and one latent frame per chunk")
        if config.get("sink_rope_mode", "fixed") not in ("absolute", "fixed"):
            raise ValueError("Matrix Game 4 sink_rope_mode must be absolute or fixed")
        super().__init__(config)
        self.sink_rope = SinkRoPE(config)

    def finish_self_attn_cache(self, pool, layer_id, update):
        pool.finish_attention(layer_id, update, sink_tokens=update.sink_tokens)

    def get_kv_cache_update(self, token_count, is_ref_prefill=False, *, global_token_count=None):
        if global_token_count is None:
            global_token_count = self.num_frame_per_chunk * self.kv_cache_manager.frame_seq_length
        return super().get_kv_cache_update(token_count, is_ref_prefill, global_token_count=global_token_count)

    def prepare_self_attn_cache(self, pool, layer_id, update):
        ring = self.self_attn_backend.prepare(pool, layer_id, update)
        return MatrixGame4AttentionState(ring, self.sink_rope.prepare(layer_id, ring, update.sink_tokens))

    def clear_request_state(self):
        super().clear_request_state()
        self.sink_rope.reset(keep_storage=True)

    def reset_cuda_graph(self):
        super().reset_cuda_graph()
        self.sink_rope.reset()

    def apply_qk_norm(self, norm, x):
        # Reference prefill used native TP reductions before block capture.
        # Preserve that rounding while denoising keeps its compiled arithmetic.
        if getattr(self.scheduler, "is_reference_prefill", False) and isinstance(norm, RMSWeightTPTorch):
            return norm.apply(x, native_reduction=True)
        return norm.apply(x)

    def apply_self_attn_norm(self, phase, x, shift, scale):
        normalized = self.apply_layer_norm(phase.norm1, x)
        frames = shift.shape[0]
        normalized = normalized.unflatten(0, (frames, x.shape[0] // frames))
        return (normalized * (1 + scale) + shift).flatten(0, 1)

    def infer_self_attn_with_kvcache(
        self,
        phase,
        grid_sizes,
        x,
        seq_lens,
        freqs,
        shift_msa,
        scale_msa,
        is_ref_prefill=False,
        self_attn_cache=None,
        start_frame=None,
    ):
        normalized = self.apply_self_attn_norm(phase, x, shift_msa, scale_msa)
        query, raw_key, value = self.project_self_attn_qkv(phase, normalized)
        if self.seq_parallel:
            query, raw_key, value = (all2all_seq2head(tensor, group=self.seq_p_group) for tensor in (query, raw_key, value))
            padded_tokens = query.shape[0]
            tokens = self.num_frame_per_chunk * self.kv_cache_manager.frame_seq_length
            query, raw_key, value = query[:tokens], raw_key[:tokens], value[:tokens]
        self.sink_rope.update(raw_key, self_attn_cache.sink, self_attn_cache.ring.key, grid_sizes, freqs, start_frame)
        query, key = apply_dynamic_causal_rope(query, raw_key, grid_sizes, freqs, start_frame)
        output = self.self_attn_backend.apply(query, key, value, self_attn_cache.ring, phase.self_attn_1)
        if self.seq_parallel:
            output = output.view(tokens, -1, self.head_dim)
            output = F.pad(output, (0, 0, 0, 0, 0, padded_tokens - tokens))
            output = all2all_head2seq(output, group=self.seq_p_group).flatten(1)
        return phase.self_attn_o.apply(output)

    def infer_kv_block(self, block, x, pre_infer_out, start_frame, self_attn_cache):
        phase = block.compute_phases[0]
        shift, scale, _, _, _, _ = self.pre_process(phase.modulation, pre_infer_out.embed0)
        normalized = self.apply_self_attn_norm(phase, x, shift, scale)
        raw_key, value = self.project_self_attn_kv(phase, normalized)
        if self.seq_parallel:
            tokens = self.num_frame_per_chunk * self.kv_cache_manager.frame_seq_length
            raw_key, value = (all2all_seq2head(tensor, group=self.seq_p_group)[:tokens] for tensor in (raw_key, value))
        self.sink_rope.update(raw_key, self_attn_cache.sink, self_attn_cache.ring.key, pre_infer_out.grid_sizes.tensor, pre_infer_out.freqs, start_frame)
        return apply_dynamic_causal_rope_single(raw_key, pre_infer_out.grid_sizes.tensor, pre_infer_out.freqs, start_frame), value

    def infer_cache_commit(self, x, embed0, context, grid_sizes, start_frame, *conditional_inputs, block_idx, block, pre, cache, conditional_names):
        prepared = self.make_block_pre(pre, x, embed0, context, grid_sizes, dict(zip(conditional_names, conditional_inputs)))
        key, value = self.run_kv_block(block_idx, block, x, prepared, start_frame, cache)
        self.self_attn_backend.update(key, value, cache.ring)
        return x
