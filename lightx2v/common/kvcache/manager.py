import torch
import torch.distributed as dist
from loguru import logger

from lightx2v.utils.envs import GET_DTYPE

from .base import BaseKVCachePool
from .quant import CalibRollingKVCachePool, QuantRollingKVCachePool
from .rolling import RollingKVCachePool


def _self_attn_pool_from_config(config, ar_config, kv_size, dtype, device):
    kv_offload = ar_config.get("kv_offload", False)
    sq = ar_config.get("kv_quant")

    if not sq:
        if kv_offload:
            from .offload import OffloadRollingKVCachePool

            return OffloadRollingKVCachePool(
                num_layers=config["num_layers"],
                cache_size=kv_size,
                num_heads=config["num_heads"],
                head_dim=config["dim"] // config["num_heads"],
                dtype=dtype,
                device=device,
            )
        return RollingKVCachePool(
            num_layers=config["num_layers"],
            cache_size=kv_size,
            num_heads=config["num_heads"],
            head_dim=config["dim"] // config["num_heads"],
            dtype=dtype,
            device=device,
        )
    else:
        common = dict(
            num_layers=config["num_layers"],
            cache_size=kv_size,
            num_heads=config["num_heads"],
            head_dim=config["dim"] // config["num_heads"],
            dtype=dtype,
            device=device,
            smooth_k=sq.get("smooth_k", True),
        )

        calibrate = sq.get("calibrate", False)
        calib_path = sq.get("calib_path", None)
        if not calibrate:
            if kv_offload:
                from .offload import OffloadQuantRollingKVCachePool

                return OffloadQuantRollingKVCachePool(**common, calib_path=calib_path)
            return QuantRollingKVCachePool(**common, calib_path=calib_path)
        else:
            num_steps = config.get("infer_steps", 1)
            return CalibRollingKVCachePool(**common, num_steps=num_steps)


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
        return _self_attn_pool_from_config(
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

    def maybe_save_calibration(self) -> None:
        """Auto-save calibration if running in calibrate mode with calib_path."""
        sq = self.ar_config.get("kv_quant")
        if not sq or not isinstance(sq, dict):
            return
        if not sq.get("calibrate", False):
            return
        output_path = sq.get("calib_path", "calib_kv.pt")
        pool = self.self_attn_kv_cache
        if not isinstance(pool, CalibRollingKVCachePool):
            return
        calib = pool.export_calibration()
        torch.save(calib, output_path)
        logger.info(
            "[KVCacheManager] calibration saved to {} — km {}, v_scale {}, k_block_scale {}",
            output_path,
            list(calib["km"].shape),
            list(calib["v_scale"].shape),
            list(calib["k_block_scale"].shape),
        )
