"""Stage3 H3 transformer using LightX2V weight and clean-history attention."""

from dataclasses import replace
from functools import partial

import torch
import torch.distributed as dist

from lightx2v.common.ops.attn.utils.all2all import all2all_head2seq, all2all_seq2head
from lightx2v.common.transformer_infer.transformer_infer import BaseTransformerInfer
from lightx2v.models.networks.autoregressive_h3.transformer_infer import H3AutoregressiveTransformerInferMixin
from lightx2v.models.networks.minimax_h3.infer.offload.transformer_infer import MiniMaxH3OffloadTransformerInfer
from lightx2v.models.networks.minimax_h3.infer.transformer_infer import MiniMaxH3TransformerInfer


class TaoMateH3TransformerInfer(H3AutoregressiveTransformerInferMixin, MiniMaxH3TransformerInfer):
    """Reuse the H3 block math while replacing only self-attention."""

    def set_clean_kv_cache(self, clean_cache):
        from lightx2v.models.networks.taomate_h3.infer.attention import TaoMateH3Attention

        self.clean_attention = TaoMateH3Attention(clean_cache)

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
        )
        return BaseTransformerInfer.run_block(self, block_idx, block, hidden, prepared, modulation)

    def run_block(self, block_idx, block, hidden_states, pre):
        self.block_idx = block_idx
        cache = self.clean_attention.cache
        self._history_length = cache.history_tokens
        self._condition_count = int(self.scheduler.layout_cpu.text_indices.numel())
        seq_parallel_size = dist.get_world_size(self.seq_p_group) if self.seq_p_group is not None else 1
        aux_length = pre.sequence_parallel_state.aux_length if pre.sequence_parallel_state is not None else 0
        token_count = aux_length + (hidden_states.shape[0] - aux_length) * seq_parallel_size
        key_workspace, value_workspace = self.clean_attention.prepare_history(
            block_idx,
            token_count=token_count,
            heads=self.num_heads // seq_parallel_size,
            head_dim=self.head_dim,
            dtype=hidden_states.dtype,
            device=hidden_states.device,
            condition_count=self._condition_count,
            history_length=self._history_length,
        )
        modulation = self._get_cached_adaln(block_idx) if self.use_adaln_cache else self._compute_adaln_table(block, pre)
        if self.scheduler.clean_commit:
            return self.infer_block(block, hidden_states, pre, modulation)
        runner = partial(self._run_prepared_block, block_idx=block_idx, block=block, pre=pre)
        inputs = (hidden_states, pre.adaln_indices, *pre.rotary_emb, modulation)
        sp_state = pre.sequence_parallel_state
        if sp_state is not None:
            inputs += sp_state.rotary_emb
        if self.cuda_graph_stack is None or not hidden_states.is_cuda:
            return runner(*inputs)

        variant = (
            block_idx,
            self._history_length,
            self._condition_count,
            self.scheduler.cu_seqlens_host[1],
            modulation.shape[0],
            sp_state.aux_length if sp_state is not None else 0,
        )
        weights = self.graph_weight_bindings(block, hidden_states.device)
        return self.run_captured_block(
            (id(block), self.use_compile),
            runner,
            inputs,
            bindings=(key_workspace, value_workspace, *weights),
            variant=variant,
        )

    def _exchange_attention_sequence(self, tensor, aux_length, head_slice):
        aux = tensor[:aux_length, head_slice]
        main = all2all_seq2head(tensor[aux_length:].contiguous(), self.seq_p_group)
        return torch.cat((aux, main), dim=0) if aux_length else main

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

            q, k, v = (self._exchange_attention_sequence(tensor, aux_length, head_slice) for tensor in (q, k, v))
        rotary_emb = sp_state.rotary_emb if sp_state is not None else pre_infer_out.rotary_emb
        q, k = weights.rope.apply(q, k, rotary_emb, rotary_dim=rotary_emb[0].shape[-1])
        scheduler = self.scheduler
        output = self.clean_attention.apply(
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
        )
        if sp_state is not None:
            output = self._restore_attention_sequence(output, aux_length)
        return weights.to_out.apply(output.flatten(1).to(self.infer_dtype))


class TaoMateH3OffloadTransformerInfer(TaoMateH3TransformerInfer, MiniMaxH3OffloadTransformerInfer):
    pass


__all__ = ["TaoMateH3TransformerInfer", "TaoMateH3OffloadTransformerInfer"]
