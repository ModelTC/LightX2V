import torch
import torch.distributed as dist
from loguru import logger

from lightx2v.utils.envs import GET_DTYPE

from .base import BaseKVCachePool
from .quant import CalibRollingKVCachePool, SageQuantRollingKVCachePool, KIVIQuantRollingKVCachePool
from .rolling import RollingKVCachePool


def _ranked_calib_path(path: str, rank: int) -> str:
    if not path:
        return path
    dot = path.rfind(".")
    if dot <= 0:
        return f"{path}.rank{rank}"
    return f"{path[:dot]}.rank{rank}{path[dot:]}"


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
        assert quant_scheme in ["sage", "kivi"], f"Invalid quant_scheme: {quant_scheme}"

        calibrate = kv_quant.get("calibrate", False)
        calib_path = kv_quant.get("calib_path", None)
        if calibrate:
            return CalibRollingKVCachePool(**common, num_steps=config.get("infer_steps", 1))

        if quant_scheme == "sage":
            return SageQuantRollingKVCachePool(
                **common,
                k_cache_type=kv_quant.get("k_cache_type", "int8"),
                v_cache_type=kv_quant.get("v_cache_type", "fp8"),
                calib_path=calib_path,
                kv_offload=kv_offload,
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
        save_path = output_path
        rank = 0
        world_size = 1
        if dist.is_available() and dist.is_initialized():
            if self.sp_group is not None:
                rank = dist.get_rank(self.sp_group)
                world_size = dist.get_world_size(self.sp_group)
            else:
                rank = dist.get_rank()
                world_size = dist.get_world_size()
            if world_size > 1:
                save_path = _ranked_calib_path(output_path, rank)
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
