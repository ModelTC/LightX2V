import json
import os

import torch
import torch.distributed as dist
from loguru import logger

from lightx2v.utils.envs import GET_DTYPE

from .base import BaseKVCachePool
from .calib import CalibRollingKVCachePool, StepCalibRollingKVCachePool
from .quant import (
    KIVIQuantRollingKVCachePool,
    LongLiveQuantRollingKVCachePool,
    SageQuantRollingKVCachePool,
    StepKiviQuantRollingKVCachePool,
    StepLongLiveQuantRollingKVCachePool,
    StepTurboQuantRollingKVCachePool,
    TurboQuantRollingKVCachePool,
)
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


def _sage_kwargs(_config, ar_config, kv_quant):
    return {
        "k_cache_type": kv_quant.get("k_cache_type", "int8"),
        "v_cache_type": kv_quant.get("v_cache_type", "fp8"),
        "calib_path": kv_quant.get("calib_path", None),
        "kv_offload": ar_config.get("kv_offload", False),
    }


def _turboquant_kwargs(_config, ar_config, kv_quant):
    return {
        "key_bits": kv_quant.get("key_bits", 3),
        "value_bits": kv_quant.get("value_bits", 2),
        "seed": kv_quant.get("turboquant_seed", kv_quant.get("seed", 42)),
        "per_layer_compressors": kv_quant.get("per_layer_compressors", True),
        "kv_offload": ar_config.get("kv_offload", False),
        "codebook_dir": kv_quant.get("codebook_dir"),
        "codebook_cache_dir": kv_quant.get("codebook_cache_dir"),
        "export_missing_codebooks": kv_quant.get("export_missing_codebooks", False),
        "value_group_size": kv_quant.get("value_group_size", 32),
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


def _step_turboquant_kwargs(config, ar_config, kv_quant):
    kwargs = _turboquant_kwargs(config, ar_config, kv_quant)
    kwargs["num_steps"] = config.get("infer_steps", ar_config.get("cache_step", 1))
    return kwargs


def _longlive_fp4_kwargs(_config, ar_config, kv_quant, *, frame_seq_length: int | None = None):
    block_token_size = kv_quant.get("block_token_size")
    if block_token_size is None and frame_seq_length is not None:
        block_token_size = frame_seq_length * ar_config.get("num_frame_per_chunk", 1)
    return {
        "block_token_size": block_token_size,
        "scale_rule": kv_quant.get("scale_rule", "mse"),
        "backend": kv_quant.get("backend", "pytorch"),
        "kv_offload": ar_config.get("kv_offload", False),
    }


def _step_longlive_fp4_kwargs(config, ar_config, kv_quant, *, frame_seq_length: int | None = None):
    kwargs = _longlive_fp4_kwargs(config, ar_config, kv_quant, frame_seq_length=frame_seq_length)
    kwargs["num_steps"] = config.get("infer_steps", ar_config.get("cache_step", 1))
    return kwargs


def _calib_kwargs(config, _ar_config, kv_quant):
    kwargs = {"num_steps": config.get("infer_steps", 1)}
    if kv_quant.get("quant_scheme") == "turboquant":
        kwargs.update(
            turboquant_calibrate=True,
            key_bits=kv_quant.get("key_bits", 3),
            turboquant_seed=kv_quant.get("turboquant_seed", kv_quant.get("seed", 42)),
            per_layer_compressors=kv_quant.get("per_layer_compressors", True),
        )
    return kwargs


def _get_self_attn_kv_cache_entry(scheme: str, step: bool):
    entry = SELF_ATTN_KV_CACHE_REGISTRY.get((scheme, bool(step)))
    if entry is None:
        raise NotImplementedError(f"self-attention KV cache scheme={scheme!r}, step_kv_cache={step} is not registered.")
    return entry


register_self_attn_kv_cache("fp", RollingKVCachePool, kwargs_builder=_fp_kwargs)
register_self_attn_kv_cache("fp", StepRollingKVCachePool, step=True, kwargs_builder=_step_fp_kwargs)
register_self_attn_kv_cache("calib", CalibRollingKVCachePool, kwargs_builder=_calib_kwargs)
register_self_attn_kv_cache("calib", StepCalibRollingKVCachePool, step=True, kwargs_builder=_calib_kwargs)
register_self_attn_kv_cache("sage", SageQuantRollingKVCachePool, kwargs_builder=_sage_kwargs)
register_self_attn_kv_cache("turboquant", TurboQuantRollingKVCachePool, kwargs_builder=_turboquant_kwargs)
register_self_attn_kv_cache("turboquant", StepTurboQuantRollingKVCachePool, step=True, kwargs_builder=_step_turboquant_kwargs)
register_self_attn_kv_cache("kivi", KIVIQuantRollingKVCachePool, kwargs_builder=_kivi_kwargs)
register_self_attn_kv_cache("kivi", StepKiviQuantRollingKVCachePool, step=True, kwargs_builder=_step_kivi_kwargs)
register_self_attn_kv_cache("longlive_fp4", LongLiveQuantRollingKVCachePool, kwargs_builder=_longlive_fp4_kwargs)
register_self_attn_kv_cache(
    "longlive_fp4",
    StepLongLiveQuantRollingKVCachePool,
    step=True,
    kwargs_builder=_step_longlive_fp4_kwargs,
)


def build_self_attn_kv_cache(config, ar_config, kv_size, dtype, device, *, frame_seq_length: int | None = None, num_heads: int | None = None):
    kv_quant = ar_config.get("kv_quant")
    common = _kv_cache_common_kwargs(config, kv_size, dtype, device, num_heads=num_heads)

    if not kv_quant:
        scheme = "fp"
        step = ar_config.get("step_kv_cache", False)
    else:
        quant_scheme = kv_quant.get("quant_scheme", "sage")
        registered_schemes = {registered_scheme for registered_scheme, _step in SELF_ATTN_KV_CACHE_REGISTRY if registered_scheme not in {"fp", "calib"}}
        if config.get("parallel"):
            assert quant_scheme in {"kivi", "longlive_fp4"}, f"Invalid quant_scheme: {quant_scheme} for parallel inference"
        assert quant_scheme in registered_schemes, f"Invalid quant_scheme: {quant_scheme}"
        if kv_quant.get("calibrate", False):
            scheme = "calib"
            step = ar_config.get("step_kv_cache", False)
        else:
            scheme = quant_scheme
            step = ar_config.get("step_kv_cache", False)
            if step and scheme == "sage":
                raise NotImplementedError("step_kv_cache does not support quant_scheme='sage'. Use step_kv_cache with quant_scheme='kivi', or disable step_kv_cache for sage.")

    cache_cls, kwargs_builder = _get_self_attn_kv_cache_entry(scheme, step)
    extra = {}
    if scheme == "longlive_fp4":
        extra["frame_seq_length"] = frame_seq_length
    return cache_cls(**common, **kwargs_builder(config, ar_config, kv_quant or {}, **extra))


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

        self.self_attn_kv_cache = self._create_self_attn_kv_cache()
        self.cross_attn_kv_cache = self._create_cross_attn_kv_cache()
        self.self_attn_kv_cache._init_kv_buffer()
        self.cross_attn_kv_cache._init_kv_buffer()
        self._create_matrix_action_kv_caches()

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
            kv_offload = bool(self.ar_config.get("kv_offload", False))
            self.action_mouse_kv_cache = SpatialRollingKVCachePool(
                spatial_len=self.frame_seq_length,
                num_layers=num_layers,
                cache_size=cache_sz,
                num_heads=heads,
                head_dim=head_dim,
                dtype=self.dtype,
                device=self.device,
                kv_offload=kv_offload,
            )
            self.action_mouse_kv_cache._init_kv_buffer()

    def save_calibration(self) -> None:
        """Auto-save calibration if running in calibrate mode with calib_path."""
        kv_quant = self.ar_config.get("kv_quant")
        if not kv_quant or not isinstance(kv_quant, dict):
            return
        if not kv_quant.get("calibrate", False):
            return
        output_path = kv_quant.get("calib_path", "calib_kv.pt")
        pool = self.self_attn_kv_cache
        if not isinstance(pool, CalibRollingKVCachePool):
            return
        calib = pool.export_calibration()
        hk = calib.pop("_turboquant_hist_k", None)

        rank = 0
        world_size = 1
        pg = None
        if dist.is_available() and dist.is_initialized():
            if self.sp_group is not None:
                rank = dist.get_rank(self.sp_group)
                world_size = dist.get_world_size(self.sp_group)
                pg = self.sp_group
            else:
                rank = dist.get_rank()
                world_size = dist.get_world_size()

        if hk is not None:
            hk_acc = hk.to(device=self.device, dtype=torch.int64)
            if world_size > 1:
                dist.all_reduce(hk_acc, op=dist.ReduceOp.SUM, group=pg)
            if rank == 0:
                out_dir = kv_quant.get("codebook_dir")
                if not out_dir:
                    out_dir = os.path.dirname(os.path.abspath(output_path)) or "."
                os.makedirs(out_dir, exist_ok=True)
                head_dim = self.config["dim"] // self.config["num_heads"]
                books = build_turboquant_codebooks_from_calib_histograms(
                    hk_acc.cpu(),
                    head_dim=head_dim,
                    key_bits=kv_quant.get("key_bits", 3),
                )
                for fname, cb_dict in books.items():
                    fpath = os.path.join(out_dir, fname)
                    with open(fpath, "w", encoding="utf-8") as f:
                        json.dump(cb_dict, f, indent=2)
                    logger.info("[KVCacheManager] TurboQuant empirical codebook written {!r}", fpath)

        if not calib:
            return

        save_path = output_path
        if world_size > 1:
            save_path = ranked_calib_path(output_path, rank)
        torch.save(calib, save_path)
        logger.info(
            "[KVCacheManager] calibration saved to {} (rank {}/{}) — km {}, v_scale {}, k_block_scale {}",
            save_path,
            rank,
            world_size,
            list(calib["km"].shape),
            list(calib["v_scale"].shape),
            list(calib["k_block_scale"].shape),
        )
