"""Krea's refilled context occupies local frames zero through two."""

import torch

from lightx2v.common.kvcache.layout import KVCacheUpdate
from lightx2v.models.networks.wan.infer.self_forcing.transformer_infer import WanSFTransformerInfer


class WanRealtimeVideoTransformerInfer(WanSFTransformerInfer):
    def frame_position(self, pre, device):
        return torch.full((), self.scheduler.local_frame_start, dtype=torch.int64, device=device)

    def get_kv_cache_update(self, token_count, is_ref_prefill=False, *, global_token_count=None):
        frame_tokens = self.kv_cache_manager.frame_seq_length
        world_size = torch.distributed.get_world_size(self.seq_p_group) if self.seq_parallel else 1
        tokens = global_token_count if global_token_count is not None else getattr(self, "_cache_token_count", token_count * world_size)
        start = self.scheduler.local_frame_start * frame_tokens
        return KVCacheUpdate(write_start=start, write_end=start + tokens, attention_start=0, global_end=start + tokens, sink_tokens=0, evicted_tokens=0)

    def reset_context_cache(self):
        self.kv_cache_manager.self_attn_kv_cache.reset()
        self.self_attn_backend.reset(keep_storage=True)
