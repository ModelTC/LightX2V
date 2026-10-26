from dataclasses import replace
from functools import partial

import torch.nn.functional as F

from lightx2v.common.kvcache.layout import KVCacheUpdate
from lightx2v.common.ops.attn.utils.all2all import all2all_head2seq, all2all_seq2head_qkv
from lightx2v.common.transformer_infer.autoregressive_cuda_graph import _tensor_layout
from lightx2v.common.transformer_infer.transformer_infer import BaseTransformerInfer
from lightx2v.models.networks.minimax_h3.infer.autoregressive.causal.attention import H3PhysicalRoPERingAttention, H3SageRoPERingAttention
from lightx2v.models.networks.minimax_h3.infer.autoregressive.transformer_infer import H3AutoregressiveTransformerInferMixin
from lightx2v.models.networks.minimax_h3.infer.offload.transformer_infer import MiniMaxH3OffloadTransformerInfer
from lightx2v.models.networks.minimax_h3.infer.transformer_infer import MiniMaxH3TransformerInfer


class MiniMaxH3CausalTransformerInfer(H3AutoregressiveTransformerInferMixin, MiniMaxH3TransformerInfer):
    def __init__(self, config):
        super().__init__(config)
        rotary = dict(rotary_dim=int(config.get("rope_freq_dim", 16)) * 6, rope_layout="split_half", rope_fma=False)
        if self.use_sage_attention:
            self.self_attn_backend = H3SageRoPERingAttention(**rotary)
        else:
            self.self_attn_backend = H3PhysicalRoPERingAttention(**rotary)

    def reset_cuda_graph(self):
        super().reset_cuda_graph()
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

    def _attention(self, weights, hidden_states, pre_infer_out):
        q, k, v = self._prepare_qkv(weights, hidden_states, None)
        if self.seq_p_group is not None:
            q, k, v = all2all_seq2head_qkv(q, k, v, self.seq_p_group)
        padded_length = q.shape[0]
        length = pre_infer_out.valid_sequence_length
        q, k, v = q[:length], k[:length], v[:length]
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
            output = all2all_head2seq(output, self.seq_p_group)
        return weights.to_out.apply(output.flatten(1).to(self.infer_dtype))


class MiniMaxH3CausalOffloadTransformerInfer(MiniMaxH3CausalTransformerInfer, MiniMaxH3OffloadTransformerInfer):
    pass
