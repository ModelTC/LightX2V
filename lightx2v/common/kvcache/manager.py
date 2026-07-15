import torch
import torch.distributed as dist
from loguru import logger

from lightx2v.utils.envs import GET_DTYPE

from .base import BaseKVCachePool
from .fifo import FIFOKVCachePool
from .quant import KIVIQuantRollingKVCachePool, StepKiviQuantRollingKVCachePool
from .rolling import RollingKVCachePool, SpatialRollingKVCachePool, StepRollingKVCachePool
from .utils import *

SELF_ATTN_KV_CACHE_REGISTRY = {}


def register_self_attn_kv_cache(scheme: str, cache_cls, *, step: bool = False, kwargs_builder=None) -> None:
    """Register a self-attention KV cache pool implementation."""
    SELF_ATTN_KV_CACHE_REGISTRY[(scheme, bool(step))] = (cache_cls, kwargs_builder or (lambda _config, _ar_config, _kv_quant: {}))


def _kv_cache_common_kwargs(config, kv_size, dtype, device, *, num_heads: int | None = None):
    return dict(
        num_layers=config["num_layers"],
        cache_size=kv_size,
        num_heads=config["num_heads"] if num_heads is None else int(num_heads),
        head_dim=config["dim"] // config["num_heads"],
        dtype=dtype,
        device=device,
    )


def _fp_kwargs(_config, ar_config, _kv_quant):
    return {"kv_offload": ar_config.get("kv_offload", False)}


def _step_fp_kwargs(config, ar_config, _kv_quant):
    return {
        "num_steps": config.get("infer_steps", ar_config.get("cache_step", 1)),
        "kv_offload": ar_config.get("kv_offload", False),
    }


def _kivi_kwargs(_config, ar_config, kv_quant):
    return {
        "k_cache_type": kv_quant.get("k_cache_type", "int4"),
        "v_cache_type": kv_quant.get("v_cache_type", "int4"),
        "group_size": kv_quant.get("group_size", 64),
        "kv_offload": ar_config.get("kv_offload", False),
    }


def _step_kivi_kwargs(config, ar_config, kv_quant):
    kwargs = _kivi_kwargs(config, ar_config, kv_quant)
    kwargs["num_steps"] = config.get("infer_steps")
    return kwargs


def _fifo_kwargs(_config, _ar_config, _kv_quant):
    return {}


def _get_self_attn_kv_cache_entry(scheme: str, step: bool):
    entry = SELF_ATTN_KV_CACHE_REGISTRY.get((scheme, bool(step)))
    if entry is None:
        raise NotImplementedError(f"self-attention KV cache scheme={scheme!r}, step_kv_cache={step} is not registered.")
    return entry


register_self_attn_kv_cache("fp", RollingKVCachePool, kwargs_builder=_fp_kwargs)
register_self_attn_kv_cache("fp", StepRollingKVCachePool, step=True, kwargs_builder=_step_fp_kwargs)
register_self_attn_kv_cache("kivi", KIVIQuantRollingKVCachePool, kwargs_builder=_kivi_kwargs)
register_self_attn_kv_cache("kivi", StepKiviQuantRollingKVCachePool, step=True, kwargs_builder=_step_kivi_kwargs)
register_self_attn_kv_cache("fifo", FIFOKVCachePool, kwargs_builder=_fifo_kwargs)


def build_self_attn_kv_cache(config, ar_config, kv_size, dtype, device, *, frame_seq_length: int | None = None, num_heads: int | None = None):
    kv_quant = ar_config.get("kv_quant")
    common = _kv_cache_common_kwargs(config, kv_size, dtype, device, num_heads=num_heads)

    if not kv_quant:
        scheme = ar_config.get("kv_cache_scheme", "fp")
        step = ar_config.get("step_kv_cache", False)
    else:
        quant_scheme = kv_quant.get("quant_scheme", "kivi")
        if quant_scheme != "kivi":
            raise NotImplementedError(f"Only quant_scheme='kivi' is supported, got {quant_scheme!r}.")
        if kv_quant.get("calibrate", False):
            raise NotImplementedError("KV calibration caches were removed; only KIVI inference cache is supported.")
        else:
            scheme = quant_scheme
            step = ar_config.get("step_kv_cache", False)

    cache_cls, kwargs_builder = _get_self_attn_kv_cache_entry(scheme, step)
    return cache_cls(**common, **kwargs_builder(config, ar_config, kv_quant or {}))


class KVCacheManager:
    def __init__(
        self,
        config={},
        device=torch.device("cuda"),
        sp_group=None,
    ):
        self.config = config
        self.ar_config = self.config.get("ar_config", {})
        self.dtype = GET_DTYPE()
        self.device = device
        self.sp_group = sp_group
        self.self_attn_kv_cache = None
        self.self_attn_kv_caches = {}

    @property
    def current_step(self) -> int:
        return getattr(self.self_attn_kv_cache, "current_step", 0)

    @current_step.setter
    def current_step(self, value: int) -> None:
        pool = self.self_attn_kv_cache
        if hasattr(pool, "current_step"):
            pool.current_step = value

    def _create_self_attn_kv_cache(self):
        cache = build_self_attn_kv_cache(
            self.config,
            self.ar_config,
            self.kv_size,
            self.dtype,
            self.device,
            frame_seq_length=getattr(self, "frame_seq_length", None),
            num_heads=getattr(self, "cache_num_heads", None),
        )
        cache.sp_head_sharded = bool(getattr(self, "sp_head_sharded_kv", False))
        return cache

    def create_self_attn_kv_cache(
        self,
        cache_name,
        kv_size: int,
        *,
        kv_cache_scheme: str | None = None,
        step_kv_cache: bool | None = None,
        dtype: torch.dtype | None = None,
    ):
        missing = object()
        old_kv_size = getattr(self, "kv_size", missing)
        old_dtype = self.dtype
        old_scheme = self.ar_config.get("kv_cache_scheme")
        old_step = self.ar_config.get("step_kv_cache")
        try:
            self.kv_size = int(kv_size)
            self.dtype = self.dtype if dtype is None else dtype
            if kv_cache_scheme is not None:
                self.ar_config["kv_cache_scheme"] = kv_cache_scheme
            if step_kv_cache is not None:
                self.ar_config["step_kv_cache"] = step_kv_cache
            cache = self._create_self_attn_kv_cache()
        finally:
            if old_kv_size is missing:
                delattr(self, "kv_size")
            else:
                self.kv_size = old_kv_size
            self.dtype = old_dtype
            if old_scheme is None:
                self.ar_config.pop("kv_cache_scheme", None)
            else:
                self.ar_config["kv_cache_scheme"] = old_scheme
            if old_step is None:
                self.ar_config.pop("step_kv_cache", None)
            else:
                self.ar_config["step_kv_cache"] = old_step
        self.self_attn_kv_caches[cache_name] = cache
        self.self_attn_kv_cache = cache
        return cache

    def get_self_attn_kv_cache(self, cache_name=None):
        if cache_name is None:
            return self.self_attn_kv_cache
        return self.self_attn_kv_caches.get(cache_name)

    def _create_cross_attn_kv_cache(self):
        return BaseKVCachePool(
            num_layers=self.config["num_layers"],
            cache_size=self.config["text_len"],
            num_heads=self.config["num_heads"],
            head_dim=self.config["dim"] // self.config["num_heads"],
            dtype=self.dtype,
            device=self.device,
        )

    def _compute_frame_seq_length(self, latent_shape, ref_num_frames: int | None = None):
        lat_f = latent_shape[1]
        lat_h = latent_shape[2]
        lat_w = latent_shape[3]
        patch_size = self.config.get("patch_size", (1, 2, 2))
        frame_seq_length = (lat_h // patch_size[1]) * (lat_w // patch_size[2])
        num_output_frames = lat_f - (lat_f % self.ar_config.get("num_frame_per_chunk", 3))
        self.ref_num_frames = int(ref_num_frames if ref_num_frames is not None else self.ar_config.get("ref_num_frames", 0))
        self.ref_tokens_global = self.ref_num_frames * frame_seq_length
        return frame_seq_length, num_output_frames

    def _create_kv_caches(self, latent_shape, ref_num_frames: int | None = None):
        """Create (or recreate) cache pools with resolution-dependent sizes."""

        self.frame_seq_length, self.num_output_frames = self._compute_frame_seq_length(latent_shape, ref_num_frames=ref_num_frames)
        ws = dist.get_world_size(self.sp_group) if self.sp_group is not None else 1
        self.sp_head_sharded_kv = bool(ws > 1)
        if self.sp_head_sharded_kv and self.config["num_heads"] % ws != 0:
            raise ValueError(f"num_heads={self.config['num_heads']} must be divisible by SP world_size={ws} for head-sharded KV cache")
        self.cache_num_heads = self.config["num_heads"] // ws if self.sp_head_sharded_kv else self.config["num_heads"]

        self.kv_size = self.frame_seq_length * (self.num_output_frames + self.ref_num_frames)
        self.ref_tokens = self.ref_tokens_global if self.sp_head_sharded_kv else self.ref_tokens_global // ws
        self.local_attn_size = self.ar_config.get("local_attn_size", -1)
        self.sink_size = self.ar_config.get("sink_size", 0)
        self.max_attention_size = self.ar_config.get("max_attention_size", None)

        if self.local_attn_size != -1:
            self.kv_size = (self.local_attn_size + self.ref_num_frames) * self.frame_seq_length
            if not self.sp_head_sharded_kv:
                self.kv_size = self.kv_size // ws
        else:
            if not self.sp_head_sharded_kv:
                self.kv_size = self.kv_size // ws

        if self.max_attention_size is not None and not self.sp_head_sharded_kv:
            self.max_attention_size = self.max_attention_size // ws

        self.max_attention_size = self.kv_size if self.max_attention_size is None else self.max_attention_size

        head_dim = self.config["dim"] // self.config["num_heads"]
        buffer_sig = (
            int(self.kv_size),
            int(self.cache_num_heads),
            int(self.config["num_layers"]),
            int(head_dim),
            str(self.dtype),
            int(self.frame_seq_length),
            bool(self.ar_config.get("kv_quant")),
            bool(self.ar_config.get("kv_offload")),
            self.config.get("infer_steps"),
        )
        if getattr(self, "self_attn_kv_cache", None) is not None and getattr(self, "_buffer_sig", None) == buffer_sig:
            self._reset_pools()
            logger.info(
                "[KVCacheManager] reuse cached KV buffers (signature matched), skip realloc: kv_cache_size={}, num_output_frames={}",
                self.kv_size,
                self.num_output_frames,
            )
            return

        self.self_attn_kv_cache = self._create_self_attn_kv_cache()
        self.cross_attn_kv_cache = self._create_cross_attn_kv_cache()
        self.self_attn_kv_cache._init_kv_buffer()
        self.cross_attn_kv_cache._init_kv_buffer()
        self._create_matrix_action_kv_caches()
        self._buffer_sig = buffer_sig

        logger.info(
            "[KVCacheManager] init: frame_seq_length={}, num_output_frames={}, kv_cache_size={}, max_attention_size={}, ws={}, local_attn_size={}, sink_size={}, kv_quant={}, kv_offload={}, sp_head_sharded_kv={}",
            self.frame_seq_length,
            self.num_output_frames,
            self.kv_size,
            self.max_attention_size,
            ws,
            self.local_attn_size,
            self.sink_size,
            bool(self.ar_config.get("kv_quant")),
            bool(self.ar_config.get("kv_offload")),
            self.sp_head_sharded_kv,
        )

    def _reset_pools(self) -> None:
        """Reset metadata/state of existing pools while keeping allocated buffers resident."""
        for attr in ("self_attn_kv_cache", "cross_attn_kv_cache", "action_keyboard_kv_cache", "action_mouse_kv_cache"):
            pool = getattr(self, attr, None)
            if pool is not None:
                pool.reset()

    def _create_matrix_action_kv_caches(self) -> None:
        """Matrix Game action K/V: keyboard ``RollingKVCachePool``, mouse ``SpatialRollingKVCachePool``."""
        ac = self.config.get("action_config")
        self.action_keyboard_kv_cache = None
        self.action_mouse_kv_cache = None
        if not ac:
            return

        heads = int(ac["heads_num"])
        k_hidden = int(ac.get("keyboard_hidden_dim", 1024))
        head_dim = k_hidden // heads
        la = self.ar_config.get("local_attn_size", -1)
        cache_sz = int(la) if la != -1 else 15
        num_layers = int(self.config["num_layers"])

        if ac.get("enable_keyboard", False):
            self.action_keyboard_kv_cache = RollingKVCachePool(
                num_layers=num_layers,
                cache_size=cache_sz,
                num_heads=heads,
                head_dim=head_dim,
                dtype=self.dtype,
                device=self.device,
                kv_offload=False,
            )
            self.action_keyboard_kv_cache._init_kv_buffer()

        if ac.get("enable_mouse", False):
            self.action_mouse_kv_cache = SpatialRollingKVCachePool(
                spatial_len=self.frame_seq_length,
                num_layers=num_layers,
                cache_size=cache_sz,
                num_heads=heads,
                head_dim=head_dim,
                dtype=self.dtype,
                device=self.device,
                kv_offload=False,
            )
            self.action_mouse_kv_cache._init_kv_buffer()
