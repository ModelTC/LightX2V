"""CUDA Graph execution for HunyuanImage3 autoregressive decoding."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist
from loguru import logger

from lightx2v.models.networks.hunyuan_image3.infer.kv_cache import HunyuanImage3StaticKVCache


def _tensor_signature(tensor: torch.Tensor):
    return (
        tuple(tensor.shape),
        tuple(tensor.stride()),
        tensor.dtype,
        tensor.device.type,
        tensor.device.index,
    )


@dataclass(frozen=True)
class HunyuanImage3ARCudaGraphKey:
    hidden_states: tuple
    position_ids: tuple
    rope_cos: tuple
    rope_sin: tuple
    cache_capacity: int


@dataclass
class _HunyuanImage3ARCudaGraphEntry:
    key: HunyuanImage3ARCudaGraphKey
    graph: torch.cuda.CUDAGraph
    pre_infer_out: Any
    hidden_states: torch.Tensor
    position_ids: torch.Tensor
    rope_cos: torch.Tensor
    rope_sin: torch.Tensor
    logits: torch.Tensor | None


class HunyuanImage3ARCudaGraphController:
    """Own persistent buffers and capture the full q_len=1 AR forward."""

    def __init__(self, config, model, device):
        self.config = config
        self.model = model
        self.device = torch.device(device)

        self.enabled = config.get("enable_ar_cuda_graph", False)
        self.kv_bucket_size = int(config.get("ar_cuda_graph_kv_bucket_size", 128))
        self.capture_warmups = int(config.get("ar_cuda_graph_capture_warmups", 2))
        self.page_size = int(config.get("ar_kv_page_size", 16))
        self.max_num_splits = int(config.get("ar_flash_attn_max_num_splits", 32))

        if self.kv_bucket_size < 1:
            raise ValueError(f"ar_cuda_graph_kv_bucket_size must be positive, got {self.kv_bucket_size}.")
        if self.page_size < 1:
            raise ValueError(f"ar_kv_page_size must be positive, got {self.page_size}.")
        if self.max_num_splits < 1:
            raise ValueError(f"ar_flash_attn_max_num_splits must be positive, got {self.max_num_splits}.")

        self._entries: dict[HunyuanImage3ARCudaGraphKey, _HunyuanImage3ARCudaGraphEntry] = {}
        self._kv_cache: HunyuanImage3StaticKVCache | None = None
        self._pool = None
        self._capture_stream = None
        self._closed = False

    @staticmethod
    def _round_up(value, multiple):
        value = int(value)
        multiple = int(multiple)
        return ((value + multiple - 1) // multiple) * multiple

    def clear(self):
        if self._closed:
            return
        if self.device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize(self.device)
        self._entries.clear()
        self._pool = None
        self._capture_stream = None

    def close(self):
        if self._closed:
            return
        self.clear()
        self._kv_cache = None
        self.model = None
        self._closed = True

    def acquire_kv_cache(self, *, num_layers, max_cache_len):
        capacity = self._round_up(max_cache_len, self.kv_bucket_size)
        cache = self._kv_cache
        if cache is None or cache.num_layers != int(num_layers) or cache.max_cache_len < capacity:
            if cache is not None:
                logger.warning(
                    "Growing HunyuanImage3 AR graph KV cache from {} to {}; cached graphs will be rebuilt.",
                    cache.max_cache_len,
                    capacity,
                )
                self.clear()
            cache = HunyuanImage3StaticKVCache(
                num_layers=num_layers,
                max_cache_len=capacity,
                dynamic=True,
                paged=True,
                page_size=self.page_size,
            )
            self._kv_cache = cache
        cache.begin_request()
        return cache

    def is_target_decode(self, pre_infer_out):
        return self.enabled and pre_infer_out.hidden_states.shape[1] == 1

    def _validate_runtime(self, pre_infer_out, valid_kv_len):
        if self._closed:
            raise RuntimeError("HunyuanImage3 AR CUDA Graph controller is closed.")
        hidden_states = pre_infer_out.hidden_states
        cache = pre_infer_out.past_key_values
        if hidden_states.shape[:2] != (1, 1):
            raise RuntimeError(f"AR CUDA Graph expects [1, 1, H], got {tuple(hidden_states.shape)}.")
        if cache is not self._kv_cache:
            raise RuntimeError("HunyuanImage3 AR CUDA Graph requires its persistent paged KV cache.")
        unsupported = {
            name
            for name in (
                "attention_mask",
                "image_mask",
                "timesteps",
                "token_hw",
                "full_attn_slices",
                "sequence_parallel_state",
                "attention_segment_specs",
            )
            if getattr(pre_infer_out, name, None) is not None
        }
        if unsupported:
            raise RuntimeError(f"AR CUDA Graph does not capture fields: {sorted(unsupported)}.")
        if int(valid_kv_len) > cache.max_cache_len:
            raise RuntimeError(f"Valid KV length {valid_kv_len} exceeds graph cache capacity {cache.max_cache_len}.")
        if any(layer.key is None or layer.value is None for layer in cache.layers):
            raise RuntimeError("The eager AR prefill must allocate every paged KV layer before graph capture.")

    def _make_key(self, pre_infer_out):
        cos, sin = pre_infer_out.custom_pos_emb
        cache = pre_infer_out.past_key_values
        return HunyuanImage3ARCudaGraphKey(
            hidden_states=_tensor_signature(pre_infer_out.hidden_states),
            position_ids=_tensor_signature(pre_infer_out.position_ids),
            rope_cos=_tensor_signature(cos),
            rope_sin=_tensor_signature(sin),
            cache_capacity=int(cache.max_cache_len),
        )

    def _active_tp_group(self):
        return self.config["parallel_context"].active_tp_group

    def _active_tp_size(self):
        return self.config["parallel_context"].active_tp_size

    def _local_query_heads(self):
        return self.model.transformer_infer.global_num_heads // self._active_tp_size()

    def _copy_runtime_inputs(self, entry, pre_infer_out, valid_kv_len):
        entry.hidden_states.copy_(pre_infer_out.hidden_states)
        entry.position_ids.copy_(pre_infer_out.position_ids)
        cos, sin = pre_infer_out.custom_pos_emb
        entry.rope_cos.copy_(cos)
        entry.rope_sin.copy_(sin)
        pre_infer_out.past_key_values.prepare_paged_decode_scheduler(
            valid_length=int(valid_kv_len),
            num_query_heads=self._local_query_heads(),
            max_num_splits=self.max_num_splits,
        )

    def _tp_barrier(self):
        group = self._active_tp_group()
        if dist.get_world_size(group) <= 1:
            return
        dist.barrier(group=group, device_ids=[self.device.index])

    def _build_entry(self, key, pre_infer_out):
        hidden_states = torch.empty_like(pre_infer_out.hidden_states)
        position_ids = torch.empty_like(pre_infer_out.position_ids)
        runtime_cos, runtime_sin = pre_infer_out.custom_pos_emb
        rope_cos = torch.empty_like(runtime_cos)
        rope_sin = torch.empty_like(runtime_sin)
        static_pre_infer_out = type(pre_infer_out)(
            hidden_states=hidden_states,
            attention_mask=None,
            position_ids=position_ids,
            custom_pos_emb=(rope_cos, rope_sin),
            past_key_values=pre_infer_out.past_key_values,
            use_cache=True,
            image_mask=None,
            timesteps=None,
            token_hw=None,
            first_step=pre_infer_out.first_step,
            full_attn_slices=None,
            sequence_parallel_state=None,
            attention_segment_specs=None,
        )
        return _HunyuanImage3ARCudaGraphEntry(
            key=key,
            graph=torch.cuda.CUDAGraph(),
            pre_infer_out=static_pre_infer_out,
            hidden_states=hidden_states,
            position_ids=position_ids,
            rope_cos=rope_cos,
            rope_sin=rope_sin,
            logits=None,
        )

    def _capture(self, key, pre_infer_out, valid_kv_len):
        entry = self._build_entry(key, pre_infer_out)
        self._copy_runtime_inputs(entry, pre_infer_out, valid_kv_len)
        if self._pool is None:
            self._pool = torch.cuda.graph_pool_handle()
        if self._capture_stream is None:
            self._capture_stream = torch.cuda.Stream(device=self.device)

        current_stream = torch.cuda.current_stream(self.device)
        self._capture_stream.wait_stream(current_stream)
        self._tp_barrier()

        with torch.cuda.stream(self._capture_stream):
            for _ in range(self.capture_warmups):
                self.model.infer_ar_prepared(entry.pre_infer_out)["logits"][:, -1, :]
        self._capture_stream.synchronize()
        self._tp_barrier()

        with self.config["parallel_context"].custom_all_reduce_capture():
            with torch.cuda.graph(entry.graph, pool=self._pool, stream=self._capture_stream):
                entry.logits = self.model.infer_ar_prepared(entry.pre_infer_out)["logits"][:, -1, :]
            current_stream.wait_stream(self._capture_stream)
        self._tp_barrier()
        self._entries[key] = entry
        logger.info(
            "Captured HunyuanImage3 AR CUDA Graph: cache_capacity={} cached_graphs={} rank={}.",
            key.cache_capacity,
            len(self._entries),
            dist.get_rank(),
        )
        return entry

    def prepare_replay(self, pre_infer_out, *, valid_kv_len):
        self._validate_runtime(pre_infer_out, valid_kv_len)
        key = self._make_key(pre_infer_out)
        if key not in self._entries:
            self._capture(key, pre_infer_out, valid_kv_len)

        entry = self._entries[key]
        self._copy_runtime_inputs(entry, pre_infer_out, valid_kv_len)
        return entry

    def run(self, pre_infer_out, *, valid_kv_len):
        entry = self.prepare_replay(pre_infer_out, valid_kv_len=valid_kv_len)
        entry.graph.replay()
        return entry.logits


__all__ = ["HunyuanImage3ARCudaGraphController", "HunyuanImage3ARCudaGraphKey"]
