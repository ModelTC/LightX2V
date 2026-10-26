"""Clean AV history for a native TaoMate-H3 attention implementation.

The shared rolling pool owns physical KV storage and optional CPU offload.
The first chunk remains complete for the next chunk; after that, its video
rows become the sink and the two newest complete AV chunks occupy the ring.
Keys arrive after Q/K normalization and RoPE, as in the released model.
"""

from dataclasses import dataclass

import torch

from lightx2v.common.kvcache.layout import KVCacheUpdate
from lightx2v.models.networks.minimax_h3.packing import AUDIO_TAG, VIDEO_TAG


@dataclass(frozen=True)
class CleanAVChunk:
    tokens: int
    video_tokens: int
    audio_tokens: int


class TaoMateCleanKVCache:
    def __init__(self, pool, num_layers):
        self.pool = pool
        self.num_layers = num_layers
        self.commits = []
        self.staged = {}
        self.staged_chunk = None
        self.staged_indices = None
        self.active = False
        self.pool._ensure_kv_buffer()

    @property
    def history_tokens(self):
        return self.pool.get_valid_length(0)

    def begin_clean_commit(self, chunk_index):
        if self.active or chunk_index != len(self.commits):
            raise RuntimeError("clean KV commit must begin at the next chunk boundary")
        self.active = True
        self.staged_chunk = None
        self.staged_indices = None

    def stage(self, layer, key, value, token_tags, commit_mask):
        if not self.active:
            raise RuntimeError("begin_clean_commit must precede KV staging")
        if layer in self.staged:
            raise RuntimeError(f"layer {layer} already staged clean KV")
        if key.shape != value.shape or key.ndim != 3 or key.shape[0] != commit_mask.numel() or key.shape[0] != token_tags.numel():
            raise ValueError("clean KV and commit mask have inconsistent shapes")
        if self.staged_indices is None:
            indices = torch.nonzero(commit_mask.to(torch.bool), as_tuple=False).flatten()
            tags = token_tags.index_select(0, indices)
            video = int((tags == VIDEO_TAG).sum().item())
            audio = int((tags == AUDIO_TAG).sum().item())
            chunk = CleanAVChunk(indices.numel(), video, audio)
            if not video or not audio or chunk.tokens != video + audio:
                raise ValueError("clean H3 chunk must contain audio and video rows")
            if not torch.all(tags[:audio] == AUDIO_TAG).item() or not torch.all(tags[audio:] == VIDEO_TAG).item():
                raise ValueError("clean H3 media order must be audio, then video")
            self.staged_indices = indices
            self.staged_chunk = chunk
        else:
            indices = self.staged_indices
        self.staged[layer] = key.index_select(0, indices).detach(), value.index_select(0, indices).detach()

    def commit(self):
        if not self.active or len(self.staged) != self.num_layers or self.staged_chunk is None:
            raise RuntimeError("clean KV commit requires every transformer layer")
        chunk = self.staged_chunk
        old_end = self.history_tokens
        first_sink_promotion = len(self.commits) == 2
        if len(self.commits) < 2:
            sink, evicted = 0, 0
        else:
            sink = self.commits[0].video_tokens
            evicted = self.commits[0].audio_tokens if first_sink_promotion else self.commits[-2].tokens
        start = old_end - evicted
        end = start + chunk.tokens
        if end > self.pool.get_max_cache_shape(0):
            raise ValueError("clean AV history exceeds the rolling KV capacity")
        update = KVCacheUpdate(start, end, 0, sum(item.tokens for item in self.commits) + chunk.tokens, sink, evicted)
        for layer in range(self.num_layers):
            if first_sink_promotion:
                self._promote_first_video_sink(layer)
            key, value = self.staged[layer]
            layer_update = KVCacheUpdate(start, end, 0, update.global_end, sink, 0 if first_sink_promotion else evicted)
            self.pool.update_kv(key, value, layer, layer_update)
            self.pool.end_layer(layer)
        self.commits.append(chunk)
        self.staged.clear()
        self.staged_chunk = None
        self.staged_indices = None
        self.active = False

    def _promote_first_video_sink(self, layer):
        """Age the first audio-first commit into a video-only fixed sink.

        The source keeps first-chunk video rows, then the next complete AV
        chunk. A ring sink must occupy physical slots zero onward. Promote
        only those video rows once; the recent AV rows stay in place.
        """
        first = self.commits[0]
        self.pool.begin_layer(layer)
        video_start = first.audio_tokens
        video_end = video_start + first.video_tokens
        key = self.pool.k_cache(layer, video_start, video_end).clone()
        value = self.pool.v_cache(layer, video_start, video_end).clone()
        layout = self.pool.ring_layout(layer)
        layout.active = True
        layout.sink_tokens = first.video_tokens
        layout.recent_head = first.audio_tokens
        layout.recent_length = self.commits[1].tokens
        local_end = first.video_tokens + self.commits[1].tokens
        self.pool.set_ends(layer, self.pool.get_seq_length(layer), local_end)
        self.pool.store_kv(key, value, 0, first.video_tokens, layer)

    def copy_history(self, layer, key_out, value_out, length):
        if length:
            self.pool.begin_layer(layer)
            self.pool.k_cache(layer, 0, length, out=key_out[:length])
            self.pool.v_cache(layer, 0, length, out=value_out[:length])
            self.pool.end_layer(layer)
        return length

    def rollback(self):
        self.staged.clear()
        self.staged_chunk = None
        self.staged_indices = None
        self.active = False

    def clear(self):
        self.pool.sync_all()
        self.pool.reset()
        self.commits.clear()
        self.rollback()
