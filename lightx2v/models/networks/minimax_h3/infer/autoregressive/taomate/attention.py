"""TaoMate clean-history attention over physical KV and quantized segments."""

from typing import NamedTuple

import torch

from lightx2v.common.kvcache.attn.flash import FlashAttentionWorkspace, apply_flash_attention
from lightx2v.common.kvcache.attn.sage import SageRingAttention, sage_kv_attention_spans, update_sage_kv_cache
from lightx2v.common.kvcache.layout import KVCacheUpdate


class TaoMateAttentionState(NamedTuple):
    history: tuple
    current: tuple
    metadata: torch.Tensor | None
    sage: bool
    workspace: tuple = ()

    @property
    def bindings(self):
        if self.sage:
            return (
                *tuple(tensor for state, _ in self.history for tensor in state.bindings),
                *tuple(tensor for state in self.current for tensor in (() if state is None else state.bindings)),
                self.metadata,
            )
        return (*tuple(tensor for key, value in self.history for tensor in (key, value)), *self.workspace)


class TaoMateH3Attention:
    def __init__(self, cache, *, sage=False, segment_tokens=None, segment_candidates=None):
        self.cache = cache
        self.sage = sage
        self.segment_tokens = segment_tokens
        self.quantizer = SageRingAttention(profile="taomate", segment_candidates=segment_candidates) if sage else None
        self.flash_workspace = FlashAttentionWorkspace()
        self._history_generations = {}
        self._metadata = {}
        self._current = {}
        self._allow_offload_history_reuse = True

    def reset(self):
        self._history_generations.clear()
        if self.quantizer is not None:
            self.quantizer.reset(keep_storage=True)
        self.flash_workspace.reset(keep_storage=True)

    def prepare_history(self, layer, *, token_count, heads, head_dim, dtype, device, condition_count, history_length, media_end):
        """Stage raw history once; graphs bind physical spans or quantized banks."""
        pool = self.cache.pool
        history = ()
        offload = getattr(pool, "_offload", None)
        reuse_offload = self.sage and self._allow_offload_history_reuse and offload is not None
        if offload is not None and self.sage:
            # A later layer's eligibility is only known here. Eager raw
            # prefetch would perform the transfer before this decision.
            pool._defer_raw_prefetch = reuse_offload
        if history_length:
            update = KVCacheUpdate(history_length, history_length, 0, history_length, pool.ring_layout(layer).sink_tokens, 0)
            if self.sage:
                prototype = pool._key_buffer(layer) if reuse_offload else None
                if prototype is None:
                    raw = pool.prepare_pending_attention(layer, update)
                    pool.upload_write_metadata()
                    prototype = raw.key
                segment_tokens = self.quantizer.select_segment_tokens(
                    prototype,
                    write_tokens=history_length,
                    sink_tokens=pool.ring_layout(layer).sink_tokens,
                    visible_tokens=history_length,
                    query_tokens=media_end - condition_count,
                    default_segment_tokens=self.segment_tokens,
                    logical_start=0,
                )
                # Each segment layout stays bound to its captured graph. A
                # later phase may select another layout, then revisit this
                # one on the next request without allocating new graph inputs.
                quantized = self.quantizer.get_quantized_cache(("history", layer, segment_tokens), prototype, segment_tokens)
                generation = (
                    id(pool),
                    len(self.cache.commits),
                    tuple(pool.ring_layout(layer).spans(0, history_length)),
                    segment_tokens,
                    getattr(offload, "storage_generation", None),
                    tuple(tensor.data_ptr() for tensor in quantized.bindings),
                )
                cache_hit = self._history_generations.get(layer) == generation
                if reuse_offload and cache_hit:
                    offload.begin_layer_without_history(layer)
                    # Only the quantized bank is valid. A subsequent raw
                    # reader (including sink promotion) must load CPU history,
                    # even for a one-layer pool whose layer ID did not change.
                    offload.loaded_layer = -1
                elif not cache_hit:
                    if reuse_offload:
                        raw = pool.prepare_pending_attention(layer, update)
                        pool.upload_write_metadata()
                    quantized.valid.zero_()
                    quantized.key_valid.zero_()
                    update_sage_kv_cache(raw.key, raw.value, raw.slot_metadata, quantized, 0, history_length, logical_start=0)
                    self._history_generations[layer] = generation
                history = ((quantized, history_length),)
            else:
                raw = pool.prepare_pending_attention(layer, update)
                pool.upload_write_metadata()
                history = tuple(
                    (raw.key[span.physical_start : span.physical_start + span.length], raw.value[span.physical_start : span.physical_start + span.length])
                    for span in pool.ring_layout(layer).spans(0, history_length)
                )
        if not self.sage:
            workspace = ()
            if bool(condition_count) + len(history) + bool(media_end - condition_count) > 1:
                prototype = torch.empty((1, heads, head_dim), dtype=dtype, device=device)
                layout = token_count, condition_count, media_end, history_length, heads, head_dim
                workspace = self.flash_workspace.prepare(layout, prototype, media_end + history_length)
            return TaoMateAttentionState(history, (), None, False, workspace)
        metadata_key = torch.device(device)
        if metadata_key not in self._metadata:
            self._metadata[metadata_key] = torch.zeros(3, dtype=torch.int32, device=device)
        current = []
        for label, length in (("condition", condition_count), ("media", media_end - condition_count), ("padding", token_count - media_end)):
            if length:
                cache_key = (label, token_count, condition_count, media_end, heads, head_dim, dtype, torch.device(device))
                if cache_key not in self._current:
                    prototype = torch.empty((1, heads, head_dim), dtype=dtype, device=device).expand(length, -1, -1)
                    segment_tokens = self.quantizer.select_segment_tokens(prototype, write_tokens=length, default_segment_tokens=self.segment_tokens)
                    self._current[cache_key] = self.quantizer.get_quantized_cache(cache_key, prototype, segment_tokens)
                current.append(self._current[cache_key])
            else:
                current.append(None)
        return TaoMateAttentionState(history, tuple(current), self._metadata[metadata_key], True)

    @staticmethod
    def _attend(calculate, query, spans, softmax_scale, workspace=()):
        spans = tuple((key, value) for key, value in spans if key.shape[0])
        return apply_flash_attention(query, spans, calculate, workspace=workspace, softmax_scale=softmax_scale).reshape_as(query)

    def apply(self, layer, calculate, query, key, value, token_tags, commit_mask, cu_seqlens_host, *, condition_count, history_length, prepared, clean_commit=False, softmax_scale=None):
        """Condition sees condition; media sees condition, clean history, current."""
        media_end = int(cu_seqlens_host[1])
        output = torch.empty_like(query)
        ranges = ((0, condition_count), (condition_count, media_end), (media_end, query.shape[0]))
        if prepared.sage:
            for state, (start, stop) in zip(prepared.current, ranges):
                if state is not None:
                    update_sage_kv_cache(key[start:stop], value[start:stop], prepared.metadata, state, stop - start, stop - start)
            condition, media, padding = prepared.current
            if condition_count:
                output[:condition_count].copy_(sage_kv_attention_spans(query[:condition_count], ((condition, condition_count),)).reshape_as(query[:condition_count]))
            banks = (((condition, condition_count),) if condition_count else ()) + prepared.history + ((media, media_end - condition_count),)
            output[condition_count:media_end].copy_(sage_kv_attention_spans(query[condition_count:media_end], banks).reshape_as(query[condition_count:media_end]))
            if media_end < query.shape[0]:
                output[media_end:].copy_(sage_kv_attention_spans(query[media_end:], ((padding, query.shape[0] - media_end),)).reshape_as(query[media_end:]))
        else:
            if condition_count:
                output[:condition_count].copy_(self._attend(calculate, query[:condition_count], ((key[:condition_count], value[:condition_count]),), softmax_scale))
            spans = ((key[:condition_count], value[:condition_count]), *prepared.history, (key[condition_count:media_end], value[condition_count:media_end]))
            output[condition_count:media_end].copy_(self._attend(calculate, query[condition_count:media_end], spans, softmax_scale, prepared.workspace))
            if media_end < query.shape[0]:
                output[media_end:].copy_(self._attend(calculate, query[media_end:], ((key[media_end:], value[media_end:]),), softmax_scale))
        if clean_commit:
            self.cache.stage(layer, key, value, token_tags, commit_mask)
        return output
