"""H3-World directed action attention over the native packed H3 sequence."""

from functools import cache

import torch

from lightx2v.models.networks.minimax_h3.infer.offload import MiniMaxH3OffloadTransformerInfer
from lightx2v.models.networks.minimax_h3.infer.transformer_infer import MiniMaxH3TransformerInfer


@cache
def _h3_world_attention():
    from torch.nn.attention.flex_attention import flex_attention

    return torch.compile(flex_attention, dynamic=False)


class MiniMaxH3WorldTransformerInfer(MiniMaxH3TransformerInfer):
    def _attention(self, weights, hidden_states, pre_infer_out):
        if pre_infer_out.sequence_parallel_state is not None:
            raise NotImplementedError("H3-World ia2av requires the full packed sequence; sequence parallel is not supported")
        action_mask = self.scheduler.layout.action_block_mask
        if action_mask is None:
            raise ValueError("H3-World ia2av requires an action attention mask")
        q, k, v = self._prepare_qkv(weights, hidden_states, pre_infer_out.rotary_emb)
        # FlexAttention requires a batch axis only at its kernel boundary.
        output = _h3_world_attention()(
            q.transpose(0, 1).unsqueeze(0),
            k.transpose(0, 1).unsqueeze(0),
            v.transpose(0, 1).unsqueeze(0),
            block_mask=action_mask,
        )
        output = output.squeeze(0).transpose(0, 1).reshape(q.shape[0], -1)
        return weights.to_out.apply(output.to(self.infer_dtype))


class MiniMaxH3WorldOffloadTransformerInfer(MiniMaxH3WorldTransformerInfer, MiniMaxH3OffloadTransformerInfer):
    """Reuse native model/block offload while retaining directed attention."""
