"""PipeFusion transformer infer for MiniMax-H3.

Runs only the current stage's block subset (split at weight-construction time)
and applies stale-KV caching in async mode: each patch refreshes the text rows
and its own slot of a full-length KV buffer while other slots keep the previous
step's values.
"""

import torch

from lightx2v.common.distributed import get_pipeline_runtime_state

from ..transformer_infer import MiniMaxH3TransformerInfer


class MiniMaxH3PipeFusionTransformerInfer(MiniMaxH3TransformerInfer):
    """Transformer infer with PipeFusion block splitting and stale-KV cache."""

    def __init__(self, config):
        super().__init__(config)
        self.pipeline_state = get_pipeline_runtime_state()
        self._full_k_bufs = {}
        self._full_v_bufs = {}

    # PipeFusion forward

    def infer(self, block_weights, pre_infer_out):
        """Run this stage's blocks only.

        Blocks are addressed by their global index (block.block_index) so the
        AdaLN cache and stale-KV buffers stay consistent across stages.
        """
        if self.use_adaln_cache:
            self._prepare_adaln_cache(pre_infer_out)
        self._num_txt_tokens = int(pre_infer_out.text_indices.numel())
        hidden_states = pre_infer_out.hidden_states
        for block in block_weights.blocks:
            self.block_idx = block.block_index
            hidden_states = self.run_block(block.block_index, block, hidden_states, pre_infer_out)
        return hidden_states

    # Stale-KV attention (overrides base class)

    def _attention(self, weights, hidden_states, pre_infer_out):
        q, k, v = self._prepare_qkv(weights, hidden_states, pre_infer_out.rotary_emb)
        k, v = self._maybe_apply_stale_kv(self.block_idx, k, v, self._num_txt_tokens)
        seq_len = q.shape[0]
        kv_len = k.shape[0]
        cu_seqlens_q = torch.tensor((0, seq_len), dtype=torch.int32, device=q.device)
        cu_seqlens_kv = torch.tensor((0, kv_len), dtype=torch.int32, device=q.device)
        out = weights.calculate.apply(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_kv=cu_seqlens_kv,
            max_seqlen_q=seq_len,
            max_seqlen_kv=kv_len,
            causal=False,
            scheduler=self.scheduler,
            block_idx=self.block_idx,
        )
        return weights.to_out.apply(out.to(self.infer_dtype))

    def _maybe_apply_stale_kv(self, block_idx, key, value, num_txt_tokens):
        """Per-patch-slot KV cache for PipeFusion.

        Sync mode stores the full K/V so async mode can reuse it; async mode
        refreshes the text prefix plus the current patch's slot, keeping the
        other patches' slots at the previous step's (stale) values.
        """
        state = self.pipeline_state
        if state.num_pipeline_patch <= 1 or num_txt_tokens <= 0:
            return key, value
        patch_token_nums = state.pp_patches_token_num
        full_len = num_txt_tokens + sum(patch_token_nums)
        buf_k, buf_v = self._get_full_bufs(block_idx, full_len, key, value)

        if not state.patch_mode:
            buf_k.copy_(key)
            buf_v.copy_(value)
            return key, value

        cur_slot = state.pipeline_patch_idx
        buf_k[:num_txt_tokens].copy_(key[:num_txt_tokens])
        buf_v[:num_txt_tokens].copy_(value[:num_txt_tokens])
        offset = num_txt_tokens + sum(patch_token_nums[:cur_slot])
        n = patch_token_nums[cur_slot]
        buf_k[offset : offset + n].copy_(key[num_txt_tokens:])
        buf_v[offset : offset + n].copy_(value[num_txt_tokens:])
        return buf_k, buf_v

    def _get_full_bufs(self, block_idx, full_len, key, value):
        """Return (k_buf, v_buf) pre-allocated to the full sequence length."""
        cached = self._full_k_bufs.get(block_idx)
        if cached is None or cached.shape[0] != full_len or cached.dtype != key.dtype:
            self._full_k_bufs[block_idx] = torch.empty(full_len, *key.shape[1:], dtype=key.dtype, device=key.device)
            self._full_v_bufs[block_idx] = torch.empty(full_len, *value.shape[1:], dtype=value.dtype, device=value.device)
        return self._full_k_bufs[block_idx], self._full_v_bufs[block_idx]

    def clear_kv_cache(self):
        """Drop the stale-KV buffers.

        The cache persists across timesteps by design (that is what "stale"
        means); clear it only between requests, never between async timesteps.
        """
        self._full_k_bufs.clear()
        self._full_v_bufs.clear()
