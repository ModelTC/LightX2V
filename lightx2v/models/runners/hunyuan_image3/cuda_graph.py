"""CUDA Graph execution for HunyuanImage3 autoregressive decoding."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist
from loguru import logger

from lightx2v.models.networks.hunyuan_image3.infer.kv_cache import HunyuanImage3StaticKVCache


def _config_bool(value, *, name):
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    raise ValueError(f"{name} must be a boolean, got {value!r}.")


def _tensor_signature(tensor: torch.Tensor | None):
    if tensor is None:
        return None
    return (
        tuple(tensor.shape),
        tuple(tensor.stride()),
        tensor.dtype,
        tensor.device.type,
        tensor.device.index,
    )


@dataclass(frozen=True)
class HunyuanImage3ARCudaGraphKey:
    q_len: int
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
        if self.device.type == "cuda" and self.device.index is None and torch.cuda.is_available():
            self.device = torch.device("cuda", torch.cuda.current_device())

        self.enabled = _config_bool(config.get("enable_ar_cuda_graph", False), name="enable_ar_cuda_graph")
        self.required = _config_bool(config.get("ar_cuda_graph_required", False), name="ar_cuda_graph_required")
        self.mode = str(config.get("ar_cuda_graph_mode", "full_decode")).strip().lower()
        self.kv_bucket_size = int(config.get("ar_cuda_graph_kv_bucket_size", 128))
        self.capture_q_lens = tuple(int(value) for value in config.get("ar_cuda_graph_capture_q_lens", [1]))
        self.capture_warmups = int(config.get("ar_cuda_graph_capture_warmups", 2))
        self.decode_attn_impl = str(config.get("ar_decode_attn_impl", "disabled") or "disabled").strip().lower()
        self.page_size = int(config.get("ar_kv_page_size", 16))
        self.max_num_splits = int(config.get("ar_flash_attn_max_num_splits", 32))

        if self.required and not self.enabled:
            raise ValueError("ar_cuda_graph_required=true requires enable_ar_cuda_graph=true.")
        if self.enabled:
            if self.mode != "full_decode":
                raise ValueError(f"Unsupported HunyuanImage3 ar_cuda_graph_mode={self.mode!r}; expected 'full_decode'.")
            if self.capture_q_lens != (1,):
                raise ValueError("HunyuanImage3 full-decode CUDA Graph supports only ar_cuda_graph_capture_q_lens=[1].")
            if self.decode_attn_impl != "flash_attn3_paged":
                raise ValueError("HunyuanImage3 AR CUDA Graph requires ar_decode_attn_impl='flash_attn3_paged'.")
            if self.device.type != "cuda":
                raise ValueError(f"HunyuanImage3 AR CUDA Graph requires a CUDA device, got {self.device}.")
        if self.kv_bucket_size < 1:
            raise ValueError(f"ar_cuda_graph_kv_bucket_size must be positive, got {self.kv_bucket_size}.")
        if self.capture_warmups < 1:
            raise ValueError(f"ar_cuda_graph_capture_warmups must be positive, got {self.capture_warmups}.")
        if self.page_size < 1:
            raise ValueError(f"ar_kv_page_size must be positive, got {self.page_size}.")
        if self.max_num_splits < 1:
            raise ValueError(f"ar_flash_attn_max_num_splits must be positive, got {self.max_num_splits}.")

        self._entries: dict[HunyuanImage3ARCudaGraphKey, _HunyuanImage3ARCudaGraphEntry] = {}
        self._disabled_keys: set[HunyuanImage3ARCudaGraphKey] = set()
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
        self._disabled_keys.clear()
        self._pool = None
        self._capture_stream = None

    def close(self):
        if self._closed:
            return
        self.clear()
        self._kv_cache = None
        self.model = None
        context = self.config.get("parallel_context")
        close_custom = getattr(context, "close_custom_all_reduce", None)
        if callable(close_custom):
            close_custom()
        if self.device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize(self.device)
        self._closed = True

    def acquire_kv_cache(self, *, num_layers, max_cache_len):
        if not self.enabled:
            raise RuntimeError("Cannot acquire a graph KV cache while AR CUDA Graph is disabled.")
        configured_capacity = int(self.config.get("ar_cuda_graph_max_cache_len", 0) or 0)
        requested_capacity = max(int(max_cache_len), configured_capacity)
        capacity = self._round_up(requested_capacity, self.kv_bucket_size)
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
        return cache

    def is_target_decode(self, pre_infer_out):
        hidden_states = getattr(pre_infer_out, "hidden_states", None)
        return self.enabled and torch.is_tensor(hidden_states) and hidden_states.ndim == 3 and int(hidden_states.shape[1]) in self.capture_q_lens

    def _validate_runtime(self, pre_infer_out, valid_kv_len):
        if self._closed:
            raise RuntimeError("HunyuanImage3 AR CUDA Graph controller is closed.")
        if not self.enabled:
            raise RuntimeError("HunyuanImage3 AR CUDA Graph controller was called while disabled.")
        if not torch.cuda.is_available():
            raise RuntimeError("HunyuanImage3 AR CUDA Graph requires CUDA.")

        hidden_states = getattr(pre_infer_out, "hidden_states", None)
        position_ids = getattr(pre_infer_out, "position_ids", None)
        custom_pos_emb = getattr(pre_infer_out, "custom_pos_emb", None)
        cache = getattr(pre_infer_out, "past_key_values", None)
        if not torch.is_tensor(hidden_states) or hidden_states.ndim != 3 or int(hidden_states.shape[1]) != 1:
            raise RuntimeError(f"HunyuanImage3 AR CUDA Graph requires hidden_states [B, 1, H], got {getattr(hidden_states, 'shape', None)}.")
        if int(hidden_states.shape[0]) != 1:
            raise RuntimeError(f"HunyuanImage3 AR CUDA Graph requires batch size 1, got {hidden_states.shape[0]}.")
        if not torch.is_tensor(position_ids) or tuple(position_ids.shape) != tuple(hidden_states.shape[:2]):
            raise RuntimeError("HunyuanImage3 AR CUDA Graph requires position_ids matching the token axes.")
        if not isinstance(custom_pos_emb, tuple) or len(custom_pos_emb) != 2 or not all(torch.is_tensor(value) for value in custom_pos_emb):
            raise RuntimeError("HunyuanImage3 AR CUDA Graph requires a static (cos, sin) rotary embedding tuple.")
        if not isinstance(cache, HunyuanImage3StaticKVCache) or cache is not self._kv_cache or not cache.paged:
            raise RuntimeError("HunyuanImage3 AR CUDA Graph requires its persistent paged KV cache.")
        if not bool(getattr(pre_infer_out, "use_cache", False)):
            raise RuntimeError("HunyuanImage3 AR CUDA Graph requires use_cache=true.")
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
            raise RuntimeError(f"HunyuanImage3 q_len=1 graph inputs contain unsupported fields: {sorted(unsupported)}.")
        if int(valid_kv_len) > cache.max_cache_len:
            raise RuntimeError(f"Valid KV length {valid_kv_len} exceeds graph cache capacity {cache.max_cache_len}.")
        if any(layer.key is None or layer.value is None for layer in cache.layers):
            raise RuntimeError("The eager AR prefill must allocate every paged KV layer before graph capture.")

    def _make_key(self, pre_infer_out):
        cos, sin = pre_infer_out.custom_pos_emb
        cache = pre_infer_out.past_key_values
        return HunyuanImage3ARCudaGraphKey(
            q_len=int(pre_infer_out.hidden_states.shape[1]),
            hidden_states=_tensor_signature(pre_infer_out.hidden_states),
            position_ids=_tensor_signature(pre_infer_out.position_ids),
            rope_cos=_tensor_signature(cos),
            rope_sin=_tensor_signature(sin),
            cache_capacity=int(cache.max_cache_len),
        )

    def _active_tp_group(self):
        context = self.config.get("parallel_context")
        if context is not None:
            return getattr(context, "active_tp_group", getattr(context, "tp_group", None))
        return getattr(self.model, "tp_group", None)

    def _active_tp_size(self):
        context = self.config.get("parallel_context")
        if context is not None:
            return int(getattr(context, "active_tp_size", getattr(context, "tp_size", 1)))
        group = self._active_tp_group()
        if group is not None and dist.is_available() and dist.is_initialized():
            return int(dist.get_world_size(group))
        return 1

    def _local_query_heads(self):
        transformer = getattr(self.model, "transformer_infer", None)
        global_heads = getattr(transformer, "global_num_heads", None)
        if global_heads is None:
            global_heads = self.config.get("num_attention_heads", self.config.get("num_heads"))
        if global_heads is None:
            raise RuntimeError("Could not resolve the HunyuanImage3 query-head count.")
        tp_size = self._active_tp_size()
        if int(global_heads) % tp_size:
            raise RuntimeError(f"Attention heads {global_heads} are not divisible by active TP size {tp_size}.")
        return int(global_heads) // tp_size

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
        if group is None or not dist.is_available() or not dist.is_initialized() or dist.get_world_size(group) <= 1:
            return
        if "nccl" in str(dist.get_backend(group)).lower():
            dist.barrier(group=group, device_ids=[self.device.index])
        else:
            dist.barrier(group=group)

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
        context = self.config.get("parallel_context")
        custom_capture = getattr(context, "custom_all_reduce_capture", None)

        with torch.cuda.stream(self._capture_stream):
            for _ in range(self.capture_warmups):
                self.model.infer_ar_prepared(entry.pre_infer_out)["logits"][:, -1, :]
        self._capture_stream.synchronize()
        self._tp_barrier()

        capture_context = custom_capture() if callable(custom_capture) else nullcontext()
        with capture_context:
            with torch.cuda.graph(entry.graph, pool=self._pool, stream=self._capture_stream):
                entry.logits = self.model.infer_ar_prepared(entry.pre_infer_out)["logits"][:, -1, :]
            current_stream.wait_stream(self._capture_stream)
        self._tp_barrier()
        self._entries[key] = entry
        logger.info(
            "Captured HunyuanImage3 AR CUDA Graph: q_len={} cache_capacity={} cached_graphs={} rank={}.",
            key.q_len,
            key.cache_capacity,
            len(self._entries),
            dist.get_rank() if dist.is_available() and dist.is_initialized() else 0,
        )
        return entry

    def _prepare_paged_eager(self, pre_infer_out, valid_kv_len):
        pre_infer_out.past_key_values.prepare_paged_decode_scheduler(
            valid_length=int(valid_kv_len),
            num_query_heads=self._local_query_heads(),
            max_num_splits=self.max_num_splits,
        )

    def prepare_replay(self, pre_infer_out, *, valid_kv_len):
        self._validate_runtime(pre_infer_out, valid_kv_len)
        key = self._make_key(pre_infer_out)
        if key in self._disabled_keys:
            self._prepare_paged_eager(pre_infer_out, valid_kv_len)
            return None

        if key not in self._entries:
            try:
                self._capture(key, pre_infer_out, valid_kv_len)
            except Exception as error:
                self._disabled_keys.add(key)
                if self.required:
                    raise RuntimeError(f"Required HunyuanImage3 AR CUDA Graph capture failed for cache capacity {key.cache_capacity}.") from error
                logger.exception(
                    "HunyuanImage3 AR graph capture failed for cache capacity {}; using eager paged FA3.",
                    key.cache_capacity,
                )
                self._prepare_paged_eager(pre_infer_out, valid_kv_len)
                return None

        entry = self._entries[key]
        self._copy_runtime_inputs(entry, pre_infer_out, valid_kv_len)
        return entry

    def run(self, pre_infer_out, *, valid_kv_len):
        entry = self.prepare_replay(pre_infer_out, valid_kv_len=valid_kv_len)
        if entry is None:
            return self.model.infer_ar_prepared(pre_infer_out)["logits"][:, -1, :]
        entry.graph.replay()
        if entry.logits is None:
            raise RuntimeError("HunyuanImage3 AR CUDA Graph replay completed without a captured logits tensor.")
        return entry.logits


__all__ = ["HunyuanImage3ARCudaGraphController", "HunyuanImage3ARCudaGraphKey"]
