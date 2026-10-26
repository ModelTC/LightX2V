"""Stage3 H3 transformer using LightX2V weight and clean-history attention."""

from dataclasses import replace
from functools import partial

import torch
import torch.distributed as dist

from lightx2v.common.ops.attn.utils.all2all import all2all_head2seq, all2all_seq2head_qkv
from lightx2v.common.transformer_infer.transformer_infer import BaseTransformerInfer
from lightx2v.models.networks.minimax_h3.infer.autoregressive.transformer_infer import H3AutoregressiveTransformerInferMixin
from lightx2v.models.networks.minimax_h3.infer.offload.transformer_infer import MiniMaxH3OffloadTransformerInfer
from lightx2v.models.networks.minimax_h3.infer.transformer_infer import MiniMaxH3TransformerInfer


class TaoMateH3TransformerInfer(H3AutoregressiveTransformerInferMixin, MiniMaxH3TransformerInfer):
    """Reuse the H3 block math while replacing only self-attention."""

    def set_clean_kv_cache(self, clean_cache):
        from lightx2v.models.networks.minimax_h3.infer.autoregressive.taomate.attention import TaoMateH3Attention

        self.self_attn_backend = TaoMateH3Attention(clean_cache, sage=self.use_sage_attention)

    def _run_prepared_block(
        self,
        hidden,
        adaln_indices,
        rope_cos,
        rope_sin,
        modulation,
        full_rope_cos=None,
        full_rope_sin=None,
        *,
        block_idx,
        block,
        pre,
        attention_state,
    ):
        sp_state = pre.sequence_parallel_state
        if sp_state is not None:
            sp_state = replace(sp_state, rotary_emb=(full_rope_cos, full_rope_sin))
        prepared = replace(
            pre,
            hidden_states=hidden,
            adaln_indices=adaln_indices,
            rotary_emb=(rope_cos, rope_sin),
            sequence_parallel_state=sp_state,
            attention_state=attention_state,
        )
        return BaseTransformerInfer.run_block(self, block_idx, block, hidden, prepared, modulation)

    def run_block(self, block_idx, block, hidden_states, pre):
        self.block_idx = block_idx
        cache = self.self_attn_backend.cache
        self._history_length = cache.history_tokens
        self._condition_count = int(self.scheduler.layout_cpu.text_indices.numel())
        seq_parallel_size = dist.get_world_size(self.seq_p_group) if self.seq_p_group is not None else 1
        aux_length = pre.sequence_parallel_state.aux_length if pre.sequence_parallel_state is not None else 0
        token_count = aux_length + (hidden_states.shape[0] - aux_length) * seq_parallel_size
        attention_state = self.self_attn_backend.prepare_history(
            block_idx,
            token_count=token_count,
            heads=self.num_heads // seq_parallel_size,
            head_dim=self.head_dim,
            dtype=hidden_states.dtype,
            device=hidden_states.device,
            condition_count=self._condition_count,
            history_length=self._history_length,
            media_end=int(self.scheduler.cu_seqlens_host[1]),
        )
        modulation = self._get_cached_adaln(block_idx) if self.use_adaln_cache else self._compute_adaln_table(block, pre)
        if self.scheduler.clean_commit:
            try:
                return self.infer_block(block, hidden_states, replace(pre, attention_state=attention_state), modulation)
            finally:
                cache.pool.end_layer(block_idx)
        runner = partial(self._run_prepared_block, block_idx=block_idx, block=block, pre=pre, attention_state=attention_state)
        inputs = (hidden_states, pre.adaln_indices, *pre.rotary_emb, modulation)
        sp_state = pre.sequence_parallel_state
        if sp_state is not None:
            inputs += sp_state.rotary_emb
        if self.cuda_graph_stack is None or not hidden_states.is_cuda:
            try:
                return runner(*inputs)
            finally:
                cache.pool.end_layer(block_idx)

        variant = (
            block_idx,
            self._history_length,
            self._condition_count,
            self.scheduler.cu_seqlens_host[1],
            modulation.shape[0],
            sp_state.aux_length if sp_state is not None else 0,
        )
        weights = self.get_graph_weight_bindings((block,))
        output = self.run_captured_block(
            (id(block), self.use_compile),
            runner,
            inputs,
            bindings=(*attention_state.bindings, *weights),
            variant=variant,
        )
        cache.pool.end_layer(block_idx)
        return output

    def _exchange_attention_qkv(self, query, key, value, aux_length, head_slice):
        inputs = query, key, value
        exchanged = all2all_seq2head_qkv(*(tensor[aux_length:] for tensor in inputs), self.seq_p_group)
        return tuple(torch.cat((tensor[:aux_length, head_slice], main), dim=0) for tensor, main in zip(inputs, exchanged)) if aux_length else exchanged

    def _restore_attention_sequence(self, output, aux_length):
        main = all2all_head2seq(output[aux_length:].contiguous(), self.seq_p_group)
        if not aux_length:
            return main
        local_aux = output[:aux_length].contiguous()
        world_size = dist.get_world_size(self.seq_p_group)
        aux_by_rank = [torch.empty_like(local_aux) for _ in range(world_size)]
        dist.all_gather(aux_by_rank, local_aux, group=self.seq_p_group)
        return torch.cat((torch.cat(aux_by_rank, dim=1), main), dim=0)

    def _attention(self, weights, hidden_states, pre_infer_out):
        q, k, v = self._prepare_qkv(weights, hidden_states, None)
        sp_state = pre_infer_out.sequence_parallel_state
        if sp_state is not None:
            aux_length = sp_state.aux_length
            world_size = dist.get_world_size(self.seq_p_group)
            rank = dist.get_rank(self.seq_p_group)
            heads_per_rank = q.shape[1] // world_size
            head_slice = slice(rank * heads_per_rank, (rank + 1) * heads_per_rank)

            q, k, v = self._exchange_attention_qkv(q, k, v, aux_length, head_slice)
        rotary_emb = sp_state.rotary_emb if sp_state is not None else pre_infer_out.rotary_emb
        q, k = weights.rope.apply(q, k, rotary_emb, rotary_dim=rotary_emb[0].shape[-1])
        scheduler = self.scheduler
        output = self.self_attn_backend.apply(
            self.block_idx if scheduler.clean_commit else 0,
            weights.calculate,
            q,
            k,
            v,
            scheduler.token_tags,
            scheduler.commit_mask,
            scheduler.cu_seqlens_host,
            condition_count=self._condition_count,
            history_length=self._history_length,
            clean_commit=scheduler.clean_commit,
            softmax_scale=self.head_dim**-0.5,
            prepared=pre_infer_out.attention_state,
        )
        if sp_state is not None:
            output = self._restore_attention_sequence(output, aux_length)
        return weights.to_out.apply(output.flatten(1).to(self.infer_dtype))


class TaoMateH3OffloadTransformerInfer(TaoMateH3TransformerInfer, MiniMaxH3OffloadTransformerInfer):
    pass


__all__ = ["TaoMateH3TransformerInfer", "TaoMateH3OffloadTransformerInfer"]
