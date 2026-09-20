import torch
import torch.nn.functional as F

from lightx2v.common.ops.attn.utils.all2all import all2all_head2seq, all2all_seq2head
from lightx2v.models.networks.minimax_h3.infer.offload.transformer_infer import MiniMaxH3OffloadTransformerInfer
from lightx2v.models.networks.minimax_h3.infer.transformer_infer import MiniMaxH3TransformerInfer


class MiniMaxH3CausalTransformerInfer(MiniMaxH3TransformerInfer):
    @torch.compiler.disable
    def update_kv_cache(self, k, v):
        # Ring metadata and step/layer selection belong outside the compiled
        # tensor graph. This also keeps block-offload graphs independent of
        # the logical layer currently occupying a staging buffer.
        cache = self.scheduler.cache
        if self.scheduler.evicted_rows:
            cache.roll_window(self.block_idx, self.scheduler.plan.sink_rows, self.scheduler.evicted_rows)
        start = self.scheduler.cache_start
        end = start + k.shape[0]
        cache.store_kv(k, v, start, end, self.block_idx)
        cache.set_ends(self.block_idx, end, end)
        return cache.k_cache(self.block_idx, 0, end), cache.v_cache(self.block_idx, 0, end)

    def _attention(self, weights, hidden_states, pre_infer_out):
        # Cache normalized keys before RoPE: older chunks move to bounded
        # temporal slots while the reference and first media chunk stay fixed.
        q, k, v = self._prepare_qkv(weights, hidden_states, None)
        if self.seq_p_group is not None:
            q, k, v = (all2all_seq2head(x, self.seq_p_group) for x in (q, k, v))
        padded_length = q.shape[0]
        length = pre_infer_out.valid_sequence_length
        q, k, v = q[:length], k[:length], v[:length]
        k, v = self.update_kv_cache(k, v)
        q = weights.rope.apply_single(q, pre_infer_out.rotary_emb, rotary_dim=pre_infer_out.rotary_emb[0].shape[-1])
        k = weights.rope.apply_single(k, pre_infer_out.key_rotary_emb, rotary_dim=pre_infer_out.key_rotary_emb[0].shape[-1])
        output = weights.calculate.apply(q=q, k=k, v=v, causal=False)
        output = output.reshape(length, q.shape[1], q.shape[2])
        if self.seq_p_group is not None:
            output = F.pad(output, (0, 0, 0, 0, 0, padded_length - length))
            output = all2all_head2seq(output, self.seq_p_group)
        return weights.to_out.apply(output.flatten(1).to(self.infer_dtype))


class MiniMaxH3CausalOffloadTransformerInfer(MiniMaxH3CausalTransformerInfer, MiniMaxH3OffloadTransformerInfer):
    pass
