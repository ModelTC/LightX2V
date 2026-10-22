from dataclasses import replace
from functools import partial

import torch
import torch.nn.functional as F

from lightx2v.common.kvcache.layout import KVCacheUpdate
from lightx2v.common.ops.attn.utils.all2all import all2all_head2seq, all2all_seq2head
from lightx2v.common.transformer_infer.transformer_infer import BaseTransformerInfer
from lightx2v.models.networks.autoregressive_h3.transformer_infer import H3AutoregressiveTransformerInferMixin
from lightx2v.models.networks.minimax_h3.infer.offload.transformer_infer import MiniMaxH3OffloadTransformerInfer
from lightx2v.models.networks.minimax_h3.infer.transformer_infer import MiniMaxH3TransformerInfer


class MiniMaxH3CausalTransformerInfer(H3AutoregressiveTransformerInferMixin, MiniMaxH3TransformerInfer):
    def __init__(self, config):
        super().__init__(config)
        self.attention_workspaces = {}

    def reset_cuda_graph(self):
        super().reset_cuda_graph()
        self.attention_workspaces.clear()

    def prepare_attention_cache(self, pre):
        """Advance CPU ring metadata and stage index maps before block execution."""
        cache = self.scheduler.cache
        layer = self.block_idx
        length = pre.valid_sequence_length
        start = self.scheduler.cache_start
        end = start + length
        cache.begin_layer(layer)
        if self.scheduler.evicted_rows:
            cache.roll_window(layer, self.scheduler.plan.sink_rows, self.scheduler.evicted_rows)
        layout = cache.ring_layout(layer)
        key, value = cache._key_buffer(layer), cache._value_buffer(layer)
        identity = key.device, key.dtype, tuple(key.shape)
        workspace = self.attention_workspaces.get(identity)
        if workspace is None or workspace[0].shape != key.shape or workspace[0].device != key.device:
            workspace = (torch.empty_like(key), torch.empty_like(value), torch.empty(key.shape[0], dtype=torch.long, device=key.device), torch.empty(key.shape[0], dtype=torch.long, device=key.device))
            self.attention_workspaces[identity] = workspace
        logical_key, logical_value, write_indices, read_indices = workspace
        write_positions = [index for span in layout.spans(start, end) for index in range(span.physical_start, span.physical_start + span.length)]
        read_positions = [index for span in layout.spans(0, end) for index in range(span.physical_start, span.physical_start + span.length)]
        write_indices[:length].copy_(torch.tensor(write_positions, dtype=torch.long), non_blocking=True)
        read_indices[:end].copy_(torch.tensor(read_positions, dtype=torch.long), non_blocking=True)
        layout.record_write(end)
        cache._attention_layouts[cache._metadata_index(layer)] = layout
        if not self.scheduler.overwrite_condition:
            cache.set_ends(layer, end, end)
        update = KVCacheUpdate(start, end, 0, end, self.scheduler.plan.sink_rows, 0)
        # Separate storage descriptors keep mutation offsets local to each layer.
        storage = torch.from_dlpack(key), torch.from_dlpack(value)
        return (*storage, write_indices[:length], read_indices[:end], logical_key[:end], logical_value[:end]), update

    def infer_prepared_block(self, hidden, temb, indices, query_cos, query_sin, key_cos, key_sin, modulation, *, block_idx, block, pre, cache):
        prepared = replace(pre, hidden_states=hidden, temb=temb, adaln_indices=indices, rotary_emb=(query_cos, query_sin), key_rotary_emb=(key_cos, key_sin), attention_cache=cache)
        return BaseTransformerInfer.run_block(self, block_idx, block, hidden, prepared, modulation)

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
            weights = self.graph_weight_bindings(block, hidden_states.device)
            output = self.run_captured_block(
                (id(block), self.use_compile),
                runner,
                inputs,
                bindings=(*cache, *weights),
                mutated_buffers=self.scheduler.cache.write_views(block_idx, update.write_start, update.write_end),
                variant=(block_idx, self.scheduler.cache.current_step, update.write_end, self.scheduler.overwrite_condition),
            )
        self.scheduler.cache.finish_attention(block_idx, update)
        return output

    def _attention(self, weights, hidden_states, pre_infer_out):
        q, k, v = self._prepare_qkv(weights, hidden_states, None)
        if self.seq_p_group is not None:
            q, k, v = (all2all_seq2head(x, self.seq_p_group) for x in (q, k, v))
        padded_length = q.shape[0]
        length = pre_infer_out.valid_sequence_length
        q, k, v = q[:length], k[:length], v[:length]
        key, value, write_indices, read_indices, logical_key, logical_value = pre_infer_out.attention_cache
        key.index_copy_(0, write_indices, k)
        value.index_copy_(0, write_indices, v)
        torch.index_select(key, 0, read_indices, out=logical_key)
        torch.index_select(value, 0, read_indices, out=logical_value)
        q = weights.rope.apply_single(q, pre_infer_out.rotary_emb, rotary_dim=pre_infer_out.rotary_emb[0].shape[-1])
        k = weights.rope.apply_single(logical_key, pre_infer_out.key_rotary_emb, rotary_dim=pre_infer_out.key_rotary_emb[0].shape[-1])
        output = weights.calculate.apply(q=q, k=k, v=logical_value, causal=False)
        output = output.reshape(length, q.shape[1], q.shape[2])
        if self.seq_p_group is not None:
            output = F.pad(output, (0, 0, 0, 0, 0, padded_length - length))
            output = all2all_head2seq(output, self.seq_p_group)
        return weights.to_out.apply(output.flatten(1).to(self.infer_dtype))


class MiniMaxH3CausalOffloadTransformerInfer(MiniMaxH3CausalTransformerInfer, MiniMaxH3OffloadTransformerInfer):
    pass
