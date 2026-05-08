import json
import os

import torch
import torch.distributed as dist
from loguru import logger

from lightx2v.utils.envs import GET_DTYPE

from .base import BaseKVCachePool
from .calib import CalibRollingKVCachePool
from .quant import (
    KIVIQuantRollingKVCachePool,
    SageQuantRollingKVCachePool,
    TurboQuantRollingKVCachePool,
)
from .rolling import RollingKVCachePool
from .utils import *


def build_self_attn_kv_cache(config, ar_config, kv_size, dtype, device):
    kv_offload = ar_config.get("kv_offload", False)
    kv_quant = ar_config.get("kv_quant")

    common = dict(
        num_layers=config["num_layers"],
        cache_size=kv_size,
        num_heads=config["num_heads"],
        head_dim=config["dim"] // config["num_heads"],
        dtype=dtype,
        device=device,
    )
    if not kv_quant:
        return RollingKVCachePool(**common, kv_offload=kv_offload)
    else:
        quant_scheme = kv_quant.get("quant_scheme", "sage")
        if config.get("parallel"):
            assert quant_scheme == "kivi", f"Invalid quant_scheme: {quant_scheme} for parallel inference"
        assert quant_scheme in ["sage", "turboquant", "kivi"], f"Invalid quant_scheme: {quant_scheme}"

        calibrate = kv_quant.get("calibrate", False)
        calib_path = kv_quant.get("calib_path", None)
        if calibrate:
            tq_extra = {}
            if kv_quant.get("quant_scheme") == "turboquant":
                tq_extra = dict(
                    turboquant_calibrate=True,
                    key_bits=kv_quant.get("key_bits", 3),
                    turboquant_seed=kv_quant.get("turboquant_seed", kv_quant.get("seed", 42)),
                    per_layer_compressors=kv_quant.get("per_layer_compressors", True),
                )
            return CalibRollingKVCachePool(
                **common,
                num_steps=config.get("infer_steps", 1),
                **tq_extra,
            )

        if quant_scheme == "sage":
            return SageQuantRollingKVCachePool(
                **common,
                k_cache_type=kv_quant.get("k_cache_type", "int8"),
                v_cache_type=kv_quant.get("v_cache_type", "fp8"),
                calib_path=calib_path,
                kv_offload=kv_offload,
            )
        elif quant_scheme == "turboquant":
            return TurboQuantRollingKVCachePool(
                **common,
                key_bits=kv_quant.get("key_bits", 3),
                value_bits=kv_quant.get("value_bits", 2),
                seed=kv_quant.get("turboquant_seed", kv_quant.get("seed", 42)),
                per_layer_compressors=kv_quant.get("per_layer_compressors", True),
                kv_offload=kv_offload,
                codebook_dir=kv_quant.get("codebook_dir"),
                codebook_cache_dir=kv_quant.get("codebook_cache_dir"),
                export_missing_codebooks=kv_quant.get("export_missing_codebooks", False),
                value_group_size=kv_quant.get("value_group_size", 32),
            )
        elif quant_scheme == "kivi":
            return KIVIQuantRollingKVCachePool(
                **common,
                k_cache_type=kv_quant.get("k_cache_type", "int4"),
                v_cache_type=kv_quant.get("v_cache_type", "int4"),
                group_size=kv_quant.get("group_size", 64),
                kv_offload=kv_offload,
            )


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
        return build_self_attn_kv_cache(
            self.config,
            self.ar_config,
            self.kv_size,
            self.dtype,
            self.device,
        )

    def _create_cross_attn_kv_cache(self):
        return BaseKVCachePool(
            num_layers=self.config["num_layers"],
            cache_size=self.config["text_len"],
            num_heads=self.config["num_heads"],
            head_dim=self.config["dim"] // self.config["num_heads"],
            dtype=self.dtype,
            device=self.device,
        )

    def _compute_frame_seq_length(self, latent_shape):
        lat_f = latent_shape[1]
        lat_h = latent_shape[2]
        lat_w = latent_shape[3]
        patch_size = self.config.get("patch_size", (1, 2, 2))
        frame_seq_length = (lat_h // patch_size[1]) * (lat_w // patch_size[2])
        num_output_frames = lat_f - (lat_f % self.ar_config.get("num_frame_per_chunk", 3))
        return frame_seq_length, num_output_frames

    def _create_kv_caches(self, latent_shape):
        """Create (or recreate) cache pools with resolution-dependent sizes."""

        self.frame_seq_length, self.num_output_frames = self._compute_frame_seq_length(latent_shape)
        ws = dist.get_world_size(self.sp_group) if self.sp_group is not None else 1
        self.kv_size = self.frame_seq_length * self.num_output_frames
        self.local_attn_size = self.ar_config.get("local_attn_size", -1)
        self.sink_size = self.ar_config.get("sink_size", 0)
        self.max_attention_size = self.ar_config.get("max_attention_size", None)

        if self.local_attn_size != -1:
            self.kv_size = self.local_attn_size * self.frame_seq_length // ws
        else:
            self.kv_size = self.kv_size // ws

        if self.max_attention_size is not None:
            self.max_attention_size = self.max_attention_size // ws
        else:
            self.max_attention_size = self.kv_size

        self.self_attn_kv_cache = self._create_self_attn_kv_cache()
        self.cross_attn_kv_cache = self._create_cross_attn_kv_cache()
        self.self_attn_kv_cache._init_kv_buffer()
        self.cross_attn_kv_cache._init_kv_buffer()

        logger.info(
            "[KVCacheManager] init: frame_seq_length={}, num_output_frames={}, kv_cache_size={}, max_attention_size={}, ws={}, local_attn_size={}, sink_size={}, kv_quant={}, kv_offload={}",
            self.frame_seq_length,
            self.num_output_frames,
            self.kv_size,
            self.max_attention_size,
            ws,
            self.local_attn_size,
            self.sink_size,
            bool(self.ar_config.get("kv_quant")),
            bool(self.ar_config.get("kv_offload")),
        )

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
