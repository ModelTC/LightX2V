"""LightX2V-native TaoMate-H3 clean-history attention.

Q/K must already have the released model's normalization and RoPE applied.
Condition queries see current condition KV. Media queries see condition KV,
retained clean AV KV, then current media KV, in exactly that order.
"""

import torch


class TaoMateH3Attention:
    def __init__(self, cache):
        self.cache = cache
        self._workspaces = {}

    def _workspace(self, token_count, heads, head_dim, dtype, device):
        capacity = self.cache.pool.get_max_cache_shape(0) + token_count
        shape = (capacity, heads, head_dim)
        layout = (shape, dtype, torch.device(device))
        if layout not in self._workspaces:
            key = torch.empty(shape, device=device, dtype=dtype)
            self._workspaces[layout] = key, torch.empty_like(key)
        return self._workspaces[layout]

    def prepare_history(self, layer, *, token_count, heads, head_dim, dtype, device, condition_count, history_length):
        """Copy visible clean AV rows before compile or CUDA Graph capture."""
        key_workspace, value_workspace = self._workspace(token_count, heads, head_dim, dtype, device)
        if history_length:
            start = condition_count
            stop = start + history_length
            self.cache.copy_history(
                layer,
                key_workspace[start:stop],
                value_workspace[start:stop],
                history_length,
            )
        return key_workspace, value_workspace

    @staticmethod
    def _attend(calculate, query, key, value, softmax_scale):
        output = calculate.apply(q=query, k=key, v=value, causal=False, softmax_scale=softmax_scale)
        return output.reshape_as(query)

    def apply(
        self,
        layer,
        calculate,
        query,
        key,
        value,
        token_tags,
        commit_mask,
        cu_seqlens_host,
        *,
        condition_count,
        history_length,
        clean_commit=False,
        softmax_scale=None,
    ):
        """Attend the fixed `[condition, audio, video, padding]` layout.

        The phase scheduler supplies lengths before capture, so no device-side
        mask inspection or dynamic index construction occurs in the hot path.
        """
        media_end = int(cu_seqlens_host[1])
        output = torch.empty_like(query)
        condition_q = query[:condition_count]
        condition_k = key[:condition_count]
        condition_v = value[:condition_count]
        media_q = query[condition_count:media_end]
        media_k = key[condition_count:media_end]
        media_v = value[condition_count:media_end]

        output[:condition_count].copy_(self._attend(calculate, condition_q, condition_k, condition_v, softmax_scale))
        key_workspace, value_workspace = self._workspace(query.shape[0], query.shape[1], query.shape[2], query.dtype, query.device)
        key_workspace[:condition_count].copy_(condition_k)
        value_workspace[:condition_count].copy_(condition_v)
        media_start = condition_count + history_length
        media_stop = media_start + media_k.shape[0]
        key_workspace[media_start:media_stop].copy_(media_k)
        value_workspace[media_start:media_stop].copy_(media_v)
        output[condition_count:media_end].copy_(self._attend(calculate, media_q, key_workspace[:media_stop], value_workspace[:media_stop], softmax_scale))

        if media_end < query.shape[0]:
            output[media_end:].copy_(self._attend(calculate, query[media_end:], key[media_end:], value[media_end:], softmax_scale))
        if clean_commit:
            self.cache.stage(layer, key, value, token_tags, commit_mask)
        return output
