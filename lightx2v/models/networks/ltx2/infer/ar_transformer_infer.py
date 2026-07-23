from __future__ import annotations

from dataclasses import dataclass

import torch

from lightx2v.common.kvcache.rolling import RollingKVCachePool
from lightx2v.models.networks.ltx2.infer.transformer_infer import LTX2TransformerInfer


@dataclass(frozen=True)
class _CacheSpec:
    cache_size: int
    attention_window_size: int
    sink_tokens: int
    num_heads: int
    head_dim: int


class LTX2ARTransformerInfer(LTX2TransformerInfer):
    """LTX2 self-forcing inference with rolling video/audio KV caches."""

    def __init__(self, config):
        super().__init__(config)
        self._ar_caches = {}
        self._ar_cache_specs = {}
        self._ar_cache_dtype = None
        self._ar_cache_device = None
        self._ar_branch = "positive"
        self._ar_block_idx = None
        self._ar_video_start = 0
        self._ar_audio_start = 0

    def configure_ar_cache(
        self,
        *,
        video_total_tokens: int,
        audio_total_tokens: int,
        video_chunk_tokens: int,
        audio_chunk_tokens: int,
        video_tokens_per_frame: int,
        audio_tokens_per_frame: int,
        dtype: torch.dtype,
        device: torch.device,
    ):
        ar_config = self.config.get("ar_config", {})
        if ar_config.get("kv_offload", False):
            raise NotImplementedError("ltx2_ar does not support KV-cache offload yet.")
        if ar_config.get("kv_quant"):
            raise NotImplementedError("ltx2_ar does not support quantized KV caches yet.")
        if ar_config.get("kv_cache_scheme", "fp") != "fp":
            raise NotImplementedError("ltx2_ar currently supports ar_config.kv_cache_scheme='fp' only.")

        local_attn_size = int(ar_config.get("local_attn_size", -1))
        sink_size = int(ar_config.get("sink_size", 0))
        if local_attn_size != -1 and local_attn_size <= 0:
            raise ValueError("ar_config.local_attn_size must be -1 or a positive number of latent frames.")
        if sink_size < 0:
            raise ValueError("ar_config.sink_size must be non-negative.")

        tp_size = max(1, int(self.tp_size))
        if self.v_num_heads % tp_size or self.a_num_heads % tp_size:
            raise ValueError("LTX2 attention heads must be divisible by tensor-parallel size.")

        self._ar_cache_specs = {
            "video": self._build_cache_spec(
                total_tokens=video_total_tokens,
                chunk_tokens=video_chunk_tokens,
                tokens_per_frame=video_tokens_per_frame,
                local_attn_size=local_attn_size,
                sink_size=sink_size,
                num_heads=self.v_num_heads // tp_size,
                head_dim=self.v_head_dim,
            ),
            "audio": self._build_cache_spec(
                total_tokens=audio_total_tokens,
                chunk_tokens=audio_chunk_tokens,
                tokens_per_frame=audio_tokens_per_frame,
                local_attn_size=local_attn_size,
                sink_size=sink_size,
                num_heads=self.a_num_heads // tp_size,
                head_dim=self.a_head_dim,
            ),
        }
        self._ar_cache_dtype = dtype
        self._ar_cache_device = torch.device(device)
        self.clear_ar_cache()

    @staticmethod
    def _build_cache_spec(
        *,
        total_tokens: int,
        chunk_tokens: int,
        tokens_per_frame: int,
        local_attn_size: int,
        sink_size: int,
        num_heads: int,
        head_dim: int,
    ) -> _CacheSpec:
        total_tokens = int(total_tokens)
        chunk_tokens = int(chunk_tokens)
        tokens_per_frame = int(tokens_per_frame)
        if min(total_tokens, chunk_tokens, tokens_per_frame) <= 0:
            raise ValueError("LTX2 AR cache sizes must be positive.")

        sink_tokens = sink_size * tokens_per_frame
        if local_attn_size == -1:
            cache_size = total_tokens
            attention_window_size = total_tokens
            sink_tokens = 0
        else:
            attention_window_size = max(chunk_tokens, local_attn_size * tokens_per_frame)
            cache_size = max(attention_window_size, sink_tokens + chunk_tokens)

        return _CacheSpec(
            cache_size=cache_size,
            attention_window_size=attention_window_size,
            sink_tokens=sink_tokens,
            num_heads=int(num_heads),
            head_dim=int(head_dim),
        )

    def clear_ar_cache(self):
        self._ar_caches = {}

    def set_ar_branch(self, branch: str):
        self._ar_branch = str(branch)

    def set_ar_chunk(self, *, video_start: int, audio_start: int):
        self._ar_video_start = int(video_start)
        self._ar_audio_start = int(audio_start)

    def infer_block(self, block_idx, block, vx, ax, pre_infer_out):
        self._ar_block_idx = int(block_idx)
        return super().infer_block(block_idx, block, vx, ax, pre_infer_out)

    def _cache_for(self, modality: str) -> tuple[RollingKVCachePool, _CacheSpec]:
        if not self._ar_cache_specs:
            raise RuntimeError("LTX2 AR cache is not configured.")
        key = (self._ar_branch, modality)
        cache = self._ar_caches.get(key)
        spec = self._ar_cache_specs[modality]
        if cache is None:
            cache = RollingKVCachePool(
                num_layers=self.blocks_num,
                cache_size=spec.cache_size,
                num_heads=spec.num_heads,
                head_dim=spec.head_dim,
                dtype=self._ar_cache_dtype,
                device=self._ar_cache_device,
            )
            cache._init_kv_buffer()
            self._ar_caches[key] = cache
        return cache, spec

    def _prepare_self_attention_kv(self, attn_phase, k: torch.Tensor, v: torch.Tensor, *, is_audio: bool):
        if self._ar_block_idx is None:
            raise RuntimeError("LTX2 AR cache was used outside a transformer block.")

        modality = "audio" if is_audio else "video"
        current_start = self._ar_audio_start if is_audio else self._ar_video_start
        cache, spec = self._cache_for(modality)
        layer_id = self._ar_block_idx
        num_new_tokens = int(k.shape[0])
        current_end = current_start + num_new_tokens
        global_end = cache.get_global_end(layer_id)
        local_end = cache.get_local_end(layer_id)

        first_pass_for_chunk = current_start == global_end
        overwrite_current_chunk = current_end == global_end
        if not first_pass_for_chunk and not overwrite_current_chunk:
            raise RuntimeError(f"LTX2 AR {modality} cache is not contiguous for branch={self._ar_branch}: global_end={global_end}, current=[{current_start}, {current_end}).")

        need_roll = first_pass_for_chunk and num_new_tokens + local_end > spec.cache_size
        if need_roll:
            num_evicted = num_new_tokens + local_end - spec.cache_size
            local_end_after_roll = local_end - num_evicted
            cache.roll_window(layer_id, spec.sink_tokens, num_evicted)
        else:
            local_end_after_roll = local_end

        local_end_index = local_end_after_roll + current_end - global_end
        local_start_index = local_end_index - num_new_tokens
        attention_start = max(0, local_end_index - spec.attention_window_size)
        cache.store_kv(k, v, local_start_index, local_end_index, layer_id)
        cache.set_ends(layer_id, current_end, local_end_index)

        return (
            cache.k_cache(layer_id, attention_start, local_end_index),
            cache.v_cache(layer_id, attention_start, local_end_index),
        )
