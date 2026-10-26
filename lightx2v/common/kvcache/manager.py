import torch
import torch.distributed as dist
from loguru import logger

from lightx2v.utils.envs import GET_DTYPE

from .base import BaseKVCachePool
from .fifo import FIFOKVCachePool
from .rolling import HybridStepRollingKVCachePool, RollingKVCachePool, StepRollingKVCachePool
from .rolling_offload import OffloadedHybridStepRollingKVCachePool, OffloadedRollingKVCachePool, OffloadedStepRollingKVCachePool
from .static import StaticKVCachePool


def parse_hybrid_attn_pattern(pattern: str) -> tuple[int, ...]:
    windows = []
    for part in pattern.split("-"):
        if not part.startswith("W") or not part[1:].isdigit() or int(part[1:]) <= 0:
            raise ValueError(f"Invalid hybrid attention window {part!r}; expected a positive window such as W2.")
        windows.append(int(part[1:]))
    return tuple(windows)


def build_self_attn_kv_cache(
    config,
    ar_config,
    kv_size,
    dtype,
    device,
    *,
    num_heads: int | None = None,
    layer_cache_sizes: tuple[int, ...] | None = None,
    key_only: bool = False,
    quantized_only: bool = False,
    value_write_alignment: tuple[int, ...] = (),
):
    """Select storage policy without changing model or manager configuration."""
    kwargs = dict(
        num_layers=config["num_layers"],
        cache_size=kv_size,
        num_heads=config["num_heads"] if num_heads is None else num_heads,
        head_dim=config["dim"] // config["num_heads"],
        dtype=dtype,
        device=device,
    )
    scheme = ar_config.get("kv_cache_scheme", "rolling")
    step_cache = ar_config.get("step_kv_cache", False)
    kv_offload = ar_config.get("kv_offload", False)
    if layer_cache_sizes is not None:
        if scheme != "rolling" or not step_cache:
            raise ValueError("Hybrid attention requires the rolling cache with step_kv_cache=true.")
        cache_cls = OffloadedHybridStepRollingKVCachePool if kv_offload else HybridStepRollingKVCachePool
        kwargs.pop("cache_size")
        kwargs["layer_cache_sizes"] = layer_cache_sizes
    elif scheme == "rolling":
        if kv_offload:
            cache_cls = OffloadedStepRollingKVCachePool if step_cache else OffloadedRollingKVCachePool
        else:
            cache_cls = StepRollingKVCachePool if step_cache else RollingKVCachePool
    elif scheme == "fifo" and not step_cache:
        cache_cls = FIFOKVCachePool
    elif scheme == "static" and not step_cache:
        cache_cls = StaticKVCachePool
    else:
        raise ValueError(f"Unsupported KV cache policy: scheme={scheme!r}, step_kv_cache={step_cache}.")
    if step_cache:
        kwargs["num_steps"] = config.get("infer_steps", ar_config.get("cache_step", 1))
    cache = cache_cls(**kwargs)
    if quantized_only:
        if scheme != "rolling" or kv_offload:
            raise ValueError("Encoded-only KV storage requires a resident rolling cache")
        cache.configure_quantized_only(write_alignment=value_write_alignment)
    elif key_only:
        if scheme != "rolling":
            raise ValueError("Quantized V storage requires the rolling cache")
        cache.configure_key_only(write_alignment=value_write_alignment)
    return cache


class KVCacheManager:
    def __init__(self, config, device=torch.device("cuda"), sp_group=None):
        self.config = config
        self.ar_config = config.get("ar_config", {})
        self.dtype = GET_DTYPE()
        self.device = device
        self.sp_group = sp_group
        self.sp_head_sharded_kv = False
        self.cache_num_heads = self.tensor_num_heads
        self.self_attn_kv_cache = None
        self.cross_attn_kv_cache = None
        self.self_attn_kv_caches = {}
        self.cross_attn_kv_caches = {}
        self._buffer_signature = None
        self.key_only = config.get("self_attn_1_type") == "sage_attn2" and config.get("model_cls") in {"wan21_edenlive_ar", "wan22_edenlive_ar", "seko_talk_ar"}
        self.quantized_only = False

    @property
    def tensor_num_heads(self) -> int:
        if self.config.get("tensor_parallel", False):
            group = self.config["device_mesh"].get_group(mesh_dim="tensor_p")
            return self.config["num_heads"] // dist.get_world_size(group)
        return self.config["num_heads"]

    @property
    def current_step(self) -> int:
        return getattr(self.self_attn_kv_cache, "current_step", 0)

    @current_step.setter
    def current_step(self, value: int) -> None:
        if hasattr(self.self_attn_kv_cache, "current_step"):
            self.self_attn_kv_cache.current_step = value

    def create_self_attn_kv_cache(
        self,
        cache_name,
        kv_size: int,
        *,
        kv_cache_scheme: str | None = None,
        step_kv_cache: bool | None = None,
        dtype: torch.dtype | None = None,
        num_heads: int | None = None,
    ):
        ar_config = dict(self.ar_config)
        if kv_cache_scheme is not None:
            ar_config["kv_cache_scheme"] = kv_cache_scheme
        if step_kv_cache is not None:
            ar_config["step_kv_cache"] = step_kv_cache
        cache = build_self_attn_kv_cache(
            self.config,
            ar_config,
            kv_size,
            self.dtype if dtype is None else dtype,
            self.device,
            num_heads=self.cache_num_heads if num_heads is None else num_heads,
            key_only=self.key_only,
            quantized_only=self.quantized_only,
        )
        cache.sp_head_sharded = self.sp_head_sharded_kv
        self.self_attn_kv_caches[cache_name] = cache
        self.self_attn_kv_cache = cache
        self._buffer_signature = None
        return cache

    def get_self_attn_kv_cache(self, cache_name=None):
        if cache_name is None:
            return self.self_attn_kv_cache
        return self.self_attn_kv_caches.get(cache_name)

    def get_cross_attn_kv_cache(self, infer_condition=True, kind="text"):
        cache_name = infer_condition if kind == "text" else (infer_condition, kind)
        pool = self.cross_attn_kv_caches.get(cache_name)
        if pool is None:
            pool = BaseKVCachePool(
                num_layers=self.config["num_layers"],
                cache_size=self.config["text_len"] if kind == "text" else 257,
                num_heads=self.tensor_num_heads,
                head_dim=self.config["dim"] // self.config["num_heads"],
                dtype=self.dtype,
                device=self.device,
            )
            self.cross_attn_kv_caches[cache_name] = pool
            if infer_condition and kind == "text":
                self.cross_attn_kv_cache = pool
        return pool

    def _set_cache_layout(self, latent_shape, ref_num_frames):
        _, latent_frames, latent_height, latent_width = latent_shape
        _, patch_height, patch_width = self.config.get("patch_size", (1, 2, 2))
        chunk_frames = self.ar_config.get("num_frame_per_chunk", 3)
        if chunk_frames <= 0:
            raise ValueError("ar_config.num_frame_per_chunk must be positive.")
        self.frame_seq_length = (latent_height // patch_height) * (latent_width // patch_width)
        self.num_output_frames = latent_frames - latent_frames % chunk_frames
        self.ref_num_frames = self.ar_config.get("ref_num_frames", 0) if ref_num_frames is None else ref_num_frames
        self.ref_tokens_global = self.ref_num_frames * self.frame_seq_length
        # SP caches hold every token and shard the attention heads.
        world_size = dist.get_world_size(self.sp_group) if self.sp_group is not None else 1
        if self.tensor_num_heads % world_size:
            raise ValueError(f"TP-local num_heads={self.tensor_num_heads} must be divisible by SP world_size={world_size}.")
        self.sp_head_sharded_kv = world_size > 1
        self.cache_num_heads = self.tensor_num_heads // world_size
        self.ref_tokens = self.ref_tokens_global
        self.local_attn_size = self.ar_config.get("local_attn_size", -1)
        self.sink_size = self.ar_config.get("sink_size", 0)
        if self.sink_size < 0:
            raise ValueError("ar_config.sink_size must be non-negative.")
        self.hybrid_attn_pattern = self.ar_config.get("hybrid_attn")
        self.hybrid_attn_windows = None
        self.layer_kv_sizes = None
        self.layer_visible_frames = None
        self.max_attention_size = self.ar_config.get("max_attention_size")
        visible_frames = self.num_output_frames if self.local_attn_size == -1 else self.local_attn_size
        self.kv_size = (visible_frames + self.ref_num_frames) * self.frame_seq_length
        if self.hybrid_attn_pattern is not None:
            self._set_hybrid_layout(chunk_frames)
        if self.max_attention_size is None:
            self.max_attention_size = self.kv_size

    def _set_hybrid_layout(self, chunk_frames):
        if self.local_attn_size != -1 or self.max_attention_size is not None:
            raise ValueError("hybrid_attn cannot be combined with local_attn_size or max_attention_size.")
        if not self.ar_config.get("step_kv_cache", False):
            raise ValueError("Hybrid attention requires a step KV cache.")
        if self.sink_size < self.ref_num_frames:
            raise ValueError("Hybrid attention sink_size must cover all reference frames.")
        windows = parse_hybrid_attn_pattern(self.hybrid_attn_pattern)
        self.hybrid_attn_windows = tuple(windows[layer % len(windows)] for layer in range(self.config["num_layers"]))
        self.layer_visible_frames = tuple(self.sink_size + window * chunk_frames for window in self.hybrid_attn_windows)
        self.layer_kv_sizes = tuple(frames * self.frame_seq_length for frames in self.layer_visible_frames)
        self.kv_size = max(self.layer_kv_sizes)
        self.max_attention_size = self.kv_size

    def _cache_signature(self):
        return (
            self.kv_size,
            self.cache_num_heads,
            self.config["num_heads"],
            self.config["num_layers"],
            self.config["dim"] // self.config["num_heads"],
            self.dtype,
            str(self.device),
            self.config["text_len"],
            self.frame_seq_length,
            self.ar_config.get("kv_cache_scheme", "rolling"),
            self.ar_config.get("step_kv_cache", False),
            self.config.get("infer_steps", self.ar_config.get("cache_step", 1)),
            self.ar_config.get("kv_offload", False),
            self.layer_kv_sizes,
            self.key_only,
            self.quantized_only,
            self.ar_config.get("num_frame_per_chunk", 3),
            self.ref_tokens,
            self.sink_size,
            self.max_attention_size,
            self.layer_visible_frames,
        )

    def initialize(self, latent_shape, ref_num_frames: int | None = None, *, before_allocate=None):
        """Reset matching buffers or allocate pools for the request layout."""
        self._set_cache_layout(latent_shape, ref_num_frames)
        signature = self._cache_signature()
        if self._buffer_signature == signature:
            self.reset()
            logger.info("[KVCacheManager] reuse KV buffers: capacity={}, frame_tokens={}", self.kv_size, self.frame_seq_length)
            return
        self._buffer_signature = None
        if before_allocate is not None:
            before_allocate()
        self.self_attn_kv_caches.clear()
        self.cross_attn_kv_caches.clear()
        self.cross_attn_kv_cache = None
        self.self_attn_kv_cache = build_self_attn_kv_cache(
            self.config,
            self.ar_config,
            self.kv_size,
            self.dtype,
            self.device,
            num_heads=self.cache_num_heads,
            layer_cache_sizes=self.layer_kv_sizes,
            key_only=self.key_only,
            quantized_only=self.quantized_only,
            value_write_alignment=(
                self.ar_config.get("num_frame_per_chunk", 3) * self.frame_seq_length,
                self.ref_tokens,
                *((self.max_attention_size,) if self.quantized_only else ()),
            ),
        )
        self.self_attn_kv_cache.sp_head_sharded = self.sp_head_sharded_kv
        self.cross_attn_kv_cache = self.get_cross_attn_kv_cache()
        self.self_attn_kv_cache._init_kv_buffer()
        self._buffer_signature = signature
        if self.layer_kv_sizes is not None:
            self._log_hybrid_allocation()
        logger.info(
            "[KVCacheManager] init: frame_tokens={}, output_frames={}, capacity={}, local_attn_size={}, sink_size={}, heads={}, policy={}, kv_offload={}",
            self.frame_seq_length,
            self.num_output_frames,
            self.kv_size,
            self.local_attn_size,
            self.sink_size,
            self.cache_num_heads,
            self.ar_config.get("kv_cache_scheme", "rolling"),
            self.ar_config.get("kv_offload", False),
        )

    def _log_hybrid_allocation(self):
        actual_bytes = self.self_attn_kv_cache.allocated_bytes
        head_dim = self.config["dim"] // self.config["num_heads"]
        element_size = torch.empty((), dtype=self.dtype).element_size()
        dense_bytes = self.self_attn_kv_cache.num_steps * self.config["num_layers"] * self.kv_size * self.cache_num_heads * head_dim * element_size * 2
        logger.info(
            "[KVCacheManager] hybrid windows={} visible_frames={} buckets={} self_kv={:.2f} GiB saved={:.1%}",
            self.hybrid_attn_windows,
            self.layer_visible_frames,
            self.self_attn_kv_cache.bucket_counts,
            actual_bytes / 1024**3,
            1 - actual_bytes / dense_bytes,
        )

    def kv_size_for_layer(self, layer_id: int) -> int:
        return self.kv_size if self.layer_kv_sizes is None else self.layer_kv_sizes[layer_id]

    def max_attention_size_for_layer(self, layer_id: int) -> int:
        return self.max_attention_size if self.layer_kv_sizes is None else self.layer_kv_sizes[layer_id]

    def uses_sliding_window(self, layer_id: int) -> bool:
        return self.hybrid_attn_windows is not None or self.local_attn_size != -1

    def reset(self) -> None:
        """Clear request state while retaining buffers bound to compiled graphs."""
        pools = (self.self_attn_kv_cache, *self.self_attn_kv_caches.values(), *self.cross_attn_kv_caches.values())
        for pool in dict.fromkeys(pools):
            if pool is not None:
                pool.reset()
