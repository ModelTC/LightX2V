"""Frame-slot caches for RelayVSR's unrotated Wan and Flash keys/values."""

from .base import BaseKVCachePool
from .manager import register_self_attn_kv_cache


class FrameRingKVCachePool(BaseKVCachePool):
    """Resident physical frame slots, overwritten in place as frames advance."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._init_kv_buffer()

    def update(self, k, v, layer_id, frame_index):
        span = k.shape[0]
        if self.cache_size % span:
            raise ValueError("Frame cache capacity must be a multiple of the frame token count")
        frames = self.cache_size // span
        start = (frame_index % frames) * span
        self._k_buffer[layer_id, start : start + span].copy_(k)
        self._v_buffer[layer_id, start : start + span].copy_(v)
        visible = min(frame_index + 1, frames) * span
        return self.k_cache(layer_id, 0, visible), self.v_cache(layer_id, 0, visible)


register_self_attn_kv_cache("frame_ring", FrameRingKVCachePool)
