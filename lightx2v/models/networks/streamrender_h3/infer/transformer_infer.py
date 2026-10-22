"""Native H3 block attention for StreamRender's fixed prefix policy."""

from dataclasses import replace
from functools import partial

import torch.nn.functional as F

from lightx2v.common.ops.attn.utils.all2all import all2all_head2seq, all2all_seq2head
from lightx2v.common.transformer_infer.transformer_infer import BaseTransformerInfer
from lightx2v.models.networks.autoregressive_h3.transformer_infer import H3AutoregressiveTransformerInferMixin
from lightx2v.models.networks.minimax_h3.infer.offload.transformer_infer import MiniMaxH3OffloadTransformerInfer
from lightx2v.models.networks.minimax_h3.infer.transformer_infer import MiniMaxH3TransformerInfer
from lightx2v.models.networks.streamrender_h3.infer.prefix_cache import streamrender_prefix_attention


class StreamRenderH3TransformerInfer(H3AutoregressiveTransformerInferMixin, MiniMaxH3TransformerInfer):
    def infer_prepared_block(self, hidden, temb, indices, cos, sin, modulation, *, cache_buffers, block_idx, block, pre, compile_block):
        prepared = replace(pre, hidden_states=hidden, temb=temb, adaln_indices=indices, rotary_emb=(cos, sin))
        if cache_buffers:
            prepared = replace(prepared, prepared_key=cache_buffers[0], prepared_value=cache_buffers[1])
        if compile_block:
            return BaseTransformerInfer.run_block(self, block_idx, block, hidden, prepared, modulation)
        return self.infer_block(block, hidden, prepared, modulation)

    def run_block(self, block_idx, block, hidden_states, pre):
        self.block_idx = block_idx
        bank = pre.prefix_bank
        if bank is not None:
            bank.prepare_layer(block_idx)
        modulation = self._get_cached_adaln(block_idx) if self.use_adaln_cache else self._compute_adaln_table(block, pre)
        cache_buffers = bank.graph_bindings(block_idx) if bank is not None and not pre.capture_prefix else ()
        inputs = (hidden_states, pre.temb, pre.adaln_indices, *pre.rotary_emb, modulation)
        compile_block = not pre.capture_prefix
        runner = partial(
            self.infer_prepared_block,
            cache_buffers=cache_buffers,
            block_idx=block_idx,
            block=block,
            pre=pre,
            compile_block=compile_block,
        )
        if self.cuda_graph_stack is None or pre.capture_prefix or not hidden_states.is_cuda:
            output = runner(*inputs)
        else:
            weights = self.graph_weight_bindings(block, hidden_states.device)
            bindings = (*bank.graph_bindings(block_idx), *weights) if bank is not None else weights
            writes = bank.write_views(block_idx, pre.valid_sequence_length) if bank is not None else ()
            output = self.run_captured_block(
                (id(block), self.use_compile),
                runner,
                inputs,
                bindings=bindings,
                mutated_buffers=writes,
                variant=(block_idx, id(bank), pre.prefix_length, pre.valid_sequence_length, pre.temb.shape[0]),
            )
        if bank is not None:
            bank.finish_layer(block_idx)
        return output

    def _attention(self, weights, hidden_states, pre_infer_out):
        q, k, v = self._prepare_qkv(weights, hidden_states, None)
        if self.seq_p_group is not None:
            q, k, v = (all2all_seq2head(tensor, self.seq_p_group) for tensor in (q, k, v))
        padded_length = q.shape[0]
        valid_length = getattr(pre_infer_out, "valid_sequence_length", padded_length)
        q, k, v = q[:valid_length], k[:valid_length], v[:valid_length]
        q, k = weights.rope.apply(q, k, pre_infer_out.rotary_emb, rotary_dim=pre_infer_out.rotary_emb[0].shape[-1])

        output = streamrender_prefix_attention(
            weights.calculate,
            q,
            k,
            v,
            bank=pre_infer_out.prefix_bank,
            layer_id=self.block_idx if pre_infer_out.capture_prefix else 0,
            prefix_length=pre_infer_out.prefix_length,
            capture_prefix=pre_infer_out.capture_prefix,
            prepared_key=pre_infer_out.prepared_key,
            prepared_value=pre_infer_out.prepared_value,
        )
        if self.seq_p_group is not None:
            output = output.unflatten(-1, (q.shape[1], q.shape[2]))
            output = F.pad(output, (0, 0, 0, 0, 0, padded_length - valid_length))
            output = all2all_head2seq(output, self.seq_p_group).flatten(1)
        return weights.to_out.apply(output.to(self.infer_dtype))


class StreamRenderH3OffloadTransformerInfer(StreamRenderH3TransformerInfer, MiniMaxH3OffloadTransformerInfer):
    """Stream real H3 block weights while retaining the prefix-KV lifecycle."""

    pass
