from dataclasses import replace
from functools import partial

import torch
import torch.distributed as dist
import torch.nn.functional as F

from lightx2v.common.kvcache.layout import KVCacheUpdate
from lightx2v.common.ops.attn.utils.all2all import all2all_head2seq, all2all_seq2head_qkv
from lightx2v.common.transformer_infer.autoregressive_cuda_graph import _tensor_layout
from lightx2v.common.transformer_infer.transformer_infer import BaseTransformerInfer
from lightx2v.models.networks.minimax_h3.infer.autoregressive.causal.attention import H3PhysicalRoPERingAttention, H3SageRoPERingAttention
from lightx2v.models.networks.minimax_h3.infer.autoregressive.causal.communication import exchange_output_fp8, exchange_qkv_fp8
from lightx2v.models.networks.minimax_h3.infer.autoregressive.transformer_infer import H3AutoregressiveTransformerInferMixin
from lightx2v.models.networks.minimax_h3.infer.offload.transformer_infer import MiniMaxH3OffloadTransformerInfer
from lightx2v.models.networks.minimax_h3.infer.transformer_infer import MiniMaxH3TransformerInfer


class MiniMaxH3CausalTransformerInfer(H3AutoregressiveTransformerInferMixin, MiniMaxH3TransformerInfer):
    def __init__(self, config):
        super().__init__(config)
        self.fp8_communication = self.seq_p_group is not None and self.seq_p_quant_scheme is not None
        if self.fp8_communication and (self.seq_p_quant_scheme != "fp8" or self.seq_p_prepost_backend != "triton" or not self.seq_p_tensor_fusion):
            raise ValueError("Causal H3 quantized communication requires FP8, Triton pre/post and seq_p_tensor_fusion=true")
        self.fused_kv_rope = config.get("ar_config", {}).get("fused_kv_rope", False)
        self.transposed_v = self.fused_kv_rope and config.get("attn_type") == "sage_attn2_sm120_cached"
        if self.fused_kv_rope and (config.get("ar_config", {}).get("kv_cache_scheme") != "fp8" or config.get("rope_type") != "torch_real_rope" or config.get("rope_compute_dtype", "fp32") != "fp32"):
            raise ValueError("fused_kv_rope requires FP8 KV and torch_real_rope with FP32 compute")
        self.fused_qkv_cache = config.get("ar_config", {}).get("fused_qkv_cache", False)
        if self.fused_qkv_cache and (not self.fused_kv_rope or self.transposed_v):
            raise ValueError("fused_qkv_cache requires fused_kv_rope and a non-transposed V attention backend")
        self.fp8_kv = config.get("ar_config", {}).get("kv_cache_scheme") == "fp8"
        if self.fp8_kv and self.cuda_graph_stack is not None:
            raise ValueError("Causal H3 block CUDA Graph requires rolling KV; FP8 KV uses compiled tensor kernels")
        rotary = dict(rotary_dim=int(config.get("rope_freq_dim", 16)) * 6, rope_layout="split_half", rope_fma=False)
        if self.fp8_kv:
            self.self_attn_backend = None
        elif self.use_sage_attention:
            candidates = (config.get("ar_config") or {}).get("sage_segment_candidates")
            self.self_attn_backend = H3SageRoPERingAttention(**rotary, segment_candidates=candidates)
        else:
            self.self_attn_backend = H3PhysicalRoPERingAttention(**rotary)

    def reset_cuda_graph(self):
        super().reset_cuda_graph()
        if self.self_attn_backend is not None:
            self.self_attn_backend.reset(keep_storage=True)

    def prepare_attention_cache(self, pre):
        """Advance CPU ring metadata and stage index maps before block execution."""
        cache = self.scheduler.cache
        layer = self.block_idx
        length = pre.valid_sequence_length
        start = self.scheduler.cache_start
        end = start + length
        update = KVCacheUpdate(start, end, 0, end, self.scheduler.plan.sink_rows, self.scheduler.evicted_rows)
        previous_ends = cache.get_seq_length(layer), cache.get_valid_length(layer)
        prepared = self.self_attn_backend.prepare(cache, layer, update, freqs=pre.key_rotary_emb)
        if self.scheduler.overwrite_condition:
            cache.set_ends(layer, *previous_ends)
        cache.upload_write_metadata()
        return prepared, update

    def infer_prepared_block(self, hidden, temb, indices, query_cos, query_sin, key_cos, key_sin, modulation, *, block_idx, block, pre, cache):
        prepared = replace(pre, hidden_states=hidden, temb=temb, adaln_indices=indices, rotary_emb=(query_cos, query_sin), key_rotary_emb=(key_cos, key_sin), attention_cache=cache)
        return BaseTransformerInfer.run_block(self, block_idx, block, hidden, prepared, modulation)

    @staticmethod
    def _graph_variant(block_idx, step, update, overwrite_condition, cache, inputs):
        # At a full window, different audio-row counts can share write_end and
        # physical spans. Keep their graph inputs in separate variants.
        return (block_idx, step, update.write_end, overwrite_condition, getattr(cache, "graph_key", None), tuple(_tensor_layout(tensor) for tensor in inputs))

    def run_block(self, block_idx, block, hidden_states, pre):
        self.block_idx = block_idx
        if self.fp8_kv:
            return MiniMaxH3TransformerInfer.run_block(self, block_idx, block, hidden_states, pre)
        cache, update = self.prepare_attention_cache(pre)
        modulation = self._get_cached_adaln(block_idx) if self.use_adaln_cache else self._compute_adaln_table(block, pre)
        inputs = (hidden_states, pre.temb, pre.adaln_indices, *pre.rotary_emb, *pre.key_rotary_emb, modulation)
        # Persistent AdaLN mode does not consume temb, but graph inputs are tensors.
        if pre.temb is None:
            inputs = (hidden_states, modulation.new_empty(0), *inputs[2:])
        runner = partial(self.infer_prepared_block, block_idx=block_idx, block=block, pre=pre, cache=cache)
        if self.cuda_graph_stack is None or not hidden_states.is_cuda:
            output = runner(*inputs)
        else:
            weights = self.get_graph_weight_bindings((block,))
            output = self.run_captured_block(
                (id(block), self.use_compile),
                runner,
                inputs,
                bindings=(*cache.bindings, *weights),
                mutated_buffers=self.scheduler.cache.write_views(block_idx, update.write_start, update.write_end),
                variant=self._graph_variant(block_idx, self.scheduler.cache.current_step, update, self.scheduler.overwrite_condition, cache, inputs),
            )
        self.scheduler.cache.finish_attention(block_idx, update)
        return output

    @torch.compiler.disable
    def update_kv_cache(self, q, k, v, query_freqs, key_freqs):
        # Ring metadata and step/layer selection belong outside the compiled
        # tensor graph. This also keeps block-offload graphs independent of
        # the logical layer currently occupying a staging buffer.
        cache = self.scheduler.cache
        if self.scheduler.evicted_rows:
            cache.roll_window(self.block_idx, self.scheduler.plan.sink_rows, self.scheduler.evicted_rows)
        start = self.scheduler.cache_start
        end = start + k.shape[0]
        if self.fused_qkv_cache:
            output = cache.store_read_rope(q, k, v, start, self.block_idx, query_freqs, key_freqs)
        else:
            cache.store_kv(k, v, start, end, self.block_idx)
        # A prompt refresh attends only to the condition prefix. Its write
        # must leave the history length and ring metadata intact for the next chunk.
        if not self.scheduler.overwrite_condition:
            cache.set_ends(self.block_idx, end, end)
        if self.fused_qkv_cache:
            return output
        if self.fused_kv_rope:
            k = cache.read_rope(self.block_idx, end, key_freqs)
            v = cache.read_transposed(self.block_idx, end) if self.transposed_v else cache.v_cache(self.block_idx, 0, end)
            return q, k, v
        return q, cache.k_cache(self.block_idx, 0, end), cache.v_cache(self.block_idx, 0, end)

    def _attention(self, weights, hidden_states, pre_infer_out):
        # Cache normalized keys before RoPE: older chunks move to bounded
        # temporal slots while the reference and first media chunk stay fixed.
        q, k, v = self._prepare_qkv(weights, hidden_states, None)
        if self.seq_p_group is not None:
            if self.fp8_communication:
                # Fused QKV projections can return strided views.
                q, k, v = exchange_qkv_fp8(q.contiguous(), k.contiguous(), v.contiguous(), self.seq_p_group.group_name, dist.get_world_size(self.seq_p_group))
            else:
                q, k, v = all2all_seq2head_qkv(q, k, v, self.seq_p_group)
        padded_length = q.shape[0]
        length = pre_infer_out.valid_sequence_length
        q, k, v = q[:length], k[:length], v[:length]
        if self.fp8_kv:
            q, k, v = self.update_kv_cache(q, k, v, pre_infer_out.rotary_emb, pre_infer_out.key_rotary_emb)
            if not self.fused_qkv_cache:
                q = weights.rope.apply_single(q, pre_infer_out.rotary_emb, rotary_dim=pre_infer_out.rotary_emb[0].shape[-1])
            if not self.fused_kv_rope:
                k = weights.rope.apply_single(k, pre_infer_out.key_rotary_emb, rotary_dim=pre_infer_out.key_rotary_emb[0].shape[-1])
            if self.transposed_v:
                output = weights.calculate.apply_pretransposed(q, k, v)
            else:
                output = weights.calculate.apply(q=q, k=k, v=v, causal=False)
        else:
            output = self.self_attn_backend.apply_rope_ring(
                q,
                k,
                v,
                pre_infer_out.attention_cache,
                attention=weights.calculate,
                rope=weights.rope,
                query_freqs=pre_infer_out.rotary_emb,
            )
        output = output.reshape(length, q.shape[1], q.shape[2])
        if self.seq_p_group is not None:
            output = F.pad(output, (0, 0, 0, 0, 0, padded_length - length))
            if self.fp8_communication:
                output = exchange_output_fp8(output.contiguous(), self.seq_p_group.group_name, dist.get_world_size(self.seq_p_group))
            else:
                output = all2all_head2seq(output, self.seq_p_group)
        return weights.to_out.apply(output.flatten(1).to(self.infer_dtype))


class MiniMaxH3CausalOffloadTransformerInfer(MiniMaxH3CausalTransformerInfer, MiniMaxH3OffloadTransformerInfer):
    pass
