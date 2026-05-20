import os

import torch
import torch.distributed as dist
from loguru import logger

from .kernel import *
from .offload import KVOffloadPlugin
from .rolling import RollingKVCachePool
from .utils import *


class SageQuantRollingKVCachePool(RollingKVCachePool):
    _BLKK = 128
    _SCALES_PER_BLK = 4  # (BLKK // WARPK) * 4, WARPK=128
    _PERM_16_VAL = [0, 1, 8, 9, 2, 3, 10, 11, 4, 5, 12, 13, 6, 7, 14, 15]
    _INV_PERM_16_VAL = [0, 1, 4, 5, 8, 9, 12, 13, 2, 3, 6, 7, 10, 11, 14, 15]

    def __init__(
        self,
        num_layers: int,
        cache_size: int,
        num_heads: int,
        head_dim: int,
        dtype: torch.dtype,
        device: torch.device,
        k_cache_type: str = "int8",
        v_cache_type: str = "fp8",
        calib_path: str = None,
        kv_offload: bool = False,
    ) -> None:
        assert k_cache_type in ["int8"], f"Invalid k_cache_type: {k_cache_type}"
        assert v_cache_type in ["fp8", "fp16"], f"Invalid v_cache_type: {v_cache_type}"
        self._k_cache_type = k_cache_type
        self._v_cache_type = v_cache_type
        self._calib_path = calib_path
        self.current_step: int = 0
        self._PERM_16 = torch.tensor(self._PERM_16_VAL, dtype=torch.long, device=device)
        self._INV_PERM_16 = torch.tensor(self._INV_PERM_16_VAL, dtype=torch.long, device=device)
        self._load_calib()
        super().__init__(num_layers, cache_size, num_heads, head_dim, dtype, device, kv_offload=kv_offload)

    def _init_kv_buffer(self) -> None:
        if self._kv_offload:
            self._init_kv_buffer_offload()
            return
        L = self._num_layers
        N = self._cache_size
        H = self._num_heads
        D = self._head_dim
        self._k_buffer = torch.zeros(L, N, H, D, dtype=torch.int8, device=self._device)
        self._v_buffer = torch.zeros(L, N, H, D, dtype=self._v_cache_type == "fp8" and torch.float8_e4m3fn or torch.float16, device=self._device)

        self._global_end = torch.zeros(L, dtype=torch.long, device=self._device)
        self._local_end = torch.zeros(L, dtype=torch.long, device=self._device)

    def _init_kv_buffer_offload(self) -> None:
        L = self._num_layers
        N = self._cache_size
        H = self._num_heads
        D = self._head_dim
        self._k_cpu = torch.zeros(L, N, H, D, dtype=torch.int8, device="cpu").pin_memory()
        self._v_cpu = torch.zeros(L, N, H, D, dtype=self._v_cache_type == "fp8" and torch.float8_e4m3fn or torch.float16, device="cpu").pin_memory()
        self._k_gpu_buf = torch.zeros(2, N, H, D, dtype=torch.int8, device=self._device)
        self._v_gpu_buf = torch.zeros(2, N, H, D, dtype=self._v_cache_type == "fp8" and torch.float8_e4m3fn or torch.float16, device=self._device)
        self._global_end = torch.zeros(L, dtype=torch.long, device=self._device)
        self._local_end = torch.zeros(L, dtype=torch.long, device=self._device)

        def _async_load(layer_id: int, buf: int) -> None:
            self._k_gpu_buf[buf].copy_(self._k_cpu[layer_id], non_blocking=True)
            self._v_gpu_buf[buf].view(torch.float8_e4m3fn).copy_(self._v_cpu[layer_id], non_blocking=True)

        def _async_store(layer_id: int, buf: int, start: int, end: int) -> None:
            self._k_cpu[layer_id, start:end].copy_(
                self._k_gpu_buf[buf, start:end],
                non_blocking=True,
            )
            v_gpu_slice_u8 = self._v_gpu_buf[buf, start:end].view(torch.float8_e4m3fn)
            self._v_cpu[layer_id, start:end].copy_(v_gpu_slice_u8, non_blocking=True)

        self._offload = KVOffloadPlugin(self._device, _async_load, _async_store)
        gpu_mb = (self._k_gpu_buf.nbytes + self._v_gpu_buf.nbytes) / (1024 * 1024)
        cpu_mb = (self._k_cpu.nbytes + self._v_cpu.nbytes) / (1024 * 1024)
        logger.info(
            "[SageQuantRollingKVCachePool+offload] GPU fixed buffer: {:.1f} MB, CPU pinned: {:.1f} MB (saved {:.1f} MB GPU)",
            gpu_mb,
            cpu_mb,
            cpu_mb - gpu_mb,
        )
        return

    def _load_calib(self, device=torch.device("cuda")) -> None:
        load_path = self._calib_path
        if dist.is_available() and dist.is_initialized():
            rank = dist.get_rank()
            rank_path = ranked_calib_path(self._calib_path, rank)
            if os.path.exists(rank_path):
                load_path = rank_path
        calib = torch.load(load_path, map_location=device, weights_only=True)
        self._calib_km = calib["km"].to(device=device, dtype=torch.float32)
        self._calib_v_scale = calib["v_scale"].to(device=device, dtype=torch.float32)
        if "k_block_scale" not in calib:
            raise RuntimeError(f"Calibration file {load_path!r} is missing 'k_block_scale'. Re-run calibration with CalibRollingKVCachePool.")
        self._calib_k_block_scale = calib["k_block_scale"].to(
            device=device,
            dtype=torch.float32,
        )
        if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
            if load_path == self._calib_path:
                logger.warning(
                    "Sage KV calibration: loaded shared file {!r} while world_size={}. "
                    "k_block_scale is indexed by *local* rolling-buffer block id; with "
                    "seq_parallel each rank uses a shorter buffer and a different sequence "
                    "shard than a single-GPU run, so a single-GPU calib file is usually *not* "
                    "applicable. Re-run calibrate with the same world_size / seq_p as "
                    "inference (saves per-rank calib_kv.rankR.pt) or use unquantized KV to compare.",
                    self._calib_path,
                    dist.get_world_size(),
                )

    def _quant_key(
        self,
        k_smoothed: torch.Tensor,
        preset_scale: torch.Tensor,
        start_idx: int,
        BLKK: int = 128,
    ) -> torch.Tensor:
        chunk_len, H, D = k_smoothed.shape
        num_blk = preset_scale.size(0)

        k_int8 = torch.empty_like(k_smoothed, dtype=torch.int8)
        preset_scale_c = preset_scale.contiguous()
        grid = (num_blk * 4, H, 1)
        quant_key_per_thread_int8_static_scale_kernel[grid](
            k_smoothed,
            k_int8,
            preset_scale_c,
            chunk_len,
            start_idx,
            0,
            k_smoothed.stride(1),
            k_smoothed.stride(0),
            0,
            k_int8.stride(1),
            k_int8.stride(0),
            preset_scale_c.stride(0),
            preset_scale_c.stride(1),
            C=D,
            BLK=BLKK,
        )
        return k_int8

    def _lookup_km(self, layer_id: int) -> torch.Tensor | None:
        """Return km of shape [1, 1, H, D] for the current (step, layer),
        or None if K smoothing is disabled.

        Supported calibration file shapes (newest → legacy):
          [S, L, 1, H, D]  – per (step, layer)            ← preferred
          [   L, 1, H, D]  – per (layer)                  ← legacy
        """
        km_cal = self._calib_km
        if km_cal.dim() == 5:
            return km_cal[self.current_step, layer_id].unsqueeze(0)
        return km_cal[layer_id].unsqueeze(0)

    def _lookup_v_scale(self, layer_id: int) -> torch.Tensor:
        """Return v_scale of shape [H, D] for the current (step, layer).

        Supported calibration file shapes (newest → legacy):
          [S, L, H, D]  – per (step, layer)               ← preferred
          [   L, H, D]  – per (layer)                     ← legacy
        """
        vs_cal = self._calib_v_scale
        if vs_cal.dim() == 4:
            return vs_cal[self.current_step, layer_id]
        return vs_cal[layer_id]

    def _lookup_k_block_scale(
        self,
        layer_id: int,
        blk_start: int,
        num_blk: int,
    ) -> torch.Tensor:
        """Return ``[num_blk, H, scales_per_blk]`` slice of the calibrated
        k-block scale at the given absolute buffer block range.

        Calibration file shape: ``[S, L, max_blks, H, scales_per_blk]``.
        """
        return self._calib_k_block_scale[
            self.current_step,
            layer_id,
            blk_start : blk_start + num_blk,
        ]

    def store_kv(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        start_idx: int,
        end_idx: int,
        layer_id: int,
    ) -> None:
        km = self._lookup_km(layer_id)
        if km is not None:
            km_lowp = km.to(k.dtype).squeeze(0)
            k_smoothed = k - km_lowp
        else:
            k_smoothed = k

        blk_start = start_idx // self._BLKK
        last_blk = (end_idx - 1) // self._BLKK
        num_blk = last_blk - blk_start + 1

        preset_scale = self._lookup_k_block_scale(layer_id, blk_start, num_blk)
        k_int8 = self._quant_key(k_smoothed, preset_scale, start_idx, self._BLKK)
        v_scale = self._lookup_v_scale(layer_id)
        v_fp8 = quant_value_per_channel_fp8_static_scale_kernel(v, v_scale, fp8_max=448.0)

        if not self._kv_offload:
            self._k_buffer[layer_id, start_idx:end_idx] = k_int8
            self._v_buffer[layer_id, start_idx:end_idx] = v_fp8
            return
        buf = self._offload.cur_buf
        self._k_gpu_buf[buf, start_idx:end_idx] = k_int8
        self._v_gpu_buf[buf, start_idx:end_idx] = v_fp8
        self._mark_offload_dirty(start_idx, end_idx)

    def _gather_per_token_k_scale(
        self,
        layer_id: int,
        start_pos: int,
        num_tokens: int,
    ) -> torch.Tensor:
        positions = torch.arange(
            start_pos,
            start_pos + num_tokens,
            device=self._device,
        )
        blk_idx = positions // self._BLKK
        thread = (positions % self._BLKK // 2) % 4
        return self._calib_k_block_scale[
            self.current_step,
            layer_id,
            blk_idx,
            :,
            thread,
        ]

    def _transpose_permute_v(self, v: torch.Tensor) -> torch.Tensor:
        kv_len, H, D = v.shape
        padded_len = (kv_len + 127) // 128 * 128

        if padded_len > kv_len:
            v_t = v.new_zeros(D, H, padded_len)
            v_t[:, :, :kv_len].copy_(v.permute(2, 1, 0))
        else:
            v_t = v.permute(2, 1, 0).contiguous()

        v_t = v_t.view(D, H, -1, 16)[:, :, :, self._PERM_16].contiguous()
        v_t = v_t.view(1, D, H, padded_len)
        return v_t

    def _roll_window_on_k_v(self, kb: torch.Tensor, vb: torch.Tensor, layer_id: int, sink_tokens: int, num_evicted: int) -> None:
        num_kept = int(self._local_end[layer_id].item()) - num_evicted - sink_tokens
        src_start = sink_tokens + num_evicted
        src_end = src_start + num_kept
        dst_start = sink_tokens
        dst_end = dst_start + num_kept
        if num_kept > 0:
            x = kb[src_start:src_end].contiguous()  # [num_kept, H, D]
            out = kb[dst_start:dst_end]
            src_scale = self._gather_per_token_k_scale(layer_id, src_start, num_kept)
            dst_scale = self._gather_per_token_k_scale(layer_id, dst_start, num_kept)
            k_int8_roll_rescale_triton(x, out, src_scale, dst_scale, scale_eps=1e-5)
        vb[dst_start:dst_end].copy_(vb[src_start:src_end].clone())

    def roll_window(self, layer_id: int, sink_tokens: int, num_evicted: int) -> None:
        if not self._kv_offload:
            self._roll_window_on_k_v(
                self._k_buffer[layer_id],
                self._v_buffer[layer_id],
                layer_id,
                sink_tokens,
                num_evicted,
            )
            return
        num_kept = int(self._local_end[layer_id].item()) - num_evicted - sink_tokens
        dst_s = sink_tokens
        self._roll_window_on_k_v(
            self._k_gpu_buf[self._offload.cur_buf],
            self._v_gpu_buf[self._offload.cur_buf],
            layer_id,
            sink_tokens,
            num_evicted,
        )
        self._mark_offload_dirty(dst_s, dst_s + num_kept)

    def k_cache(
        self,
        layer_id: int,
        attn_start: int,
        local_end: int,
    ):
        BLK = self._BLKK
        aligned_start = (attn_start // BLK) * BLK
        buf = self._offload.cur_buf if self._kv_offload else None
        kb = self._k_gpu_buf[buf] if self._kv_offload else self._k_buffer[layer_id]
        k_int8 = kb[aligned_start:local_end].unsqueeze(0).contiguous()
        blk_s = aligned_start // BLK
        blk_e = (local_end + BLK - 1) // BLK
        k_scale = self._calib_k_block_scale[self.current_step, layer_id, blk_s:blk_e].permute(1, 0, 2).reshape(1, self._num_heads, -1).contiguous()
        return k_int8, k_scale

    def v_cache(
        self,
        layer_id: int,
        attn_start: int,
        local_end: int,
    ):
        BLK = self._BLKK
        aligned_start = (attn_start // BLK) * BLK
        buf = self._offload.cur_buf if self._kv_offload else None
        vb = self._v_gpu_buf[buf] if self._kv_offload else self._v_buffer[layer_id]
        v_fp8 = vb[aligned_start:local_end]
        v_fp8 = self._transpose_permute_v(v_fp8)
        v_scale = self._lookup_v_scale(layer_id).unsqueeze(0).contiguous()
        return v_fp8, v_scale


class TurboQuantRollingKVCachePool(RollingKVCachePool):
    """Rolling KV cache using TurboQuant-style quantization.

    Aligned with ``/turboquant``: keys use TurboQuantProd ((key_bits - 1) MSE + QJL
    residual), and values use group min–max quantization.

    Pre-compute codebooks with :func:`export_turboquant_codebook_json` or set ``codebook_cache_dir`` and
    ``export_missing_codebooks`` to generate missing JSON on first run (needs scipy).

    ``k_cache`` / ``v_cache`` return **dequantized** tensors in ``self._dtype``.
    """

    def __init__(
        self,
        num_layers: int,
        cache_size: int,
        num_heads: int,
        head_dim: int,
        dtype: torch.dtype,
        device: torch.device,
        key_bits: int = 3,
        value_bits: int = 2,
        seed: int = 42,
        per_layer_compressors: bool = True,
        kv_offload: bool = False,
        *,
        codebook_dir: str | None = None,
        codebook_cache_dir: str | None = None,
        export_missing_codebooks: bool = False,
        value_group_size: int = 32,
    ) -> None:
        self._key_bits = int(key_bits)
        self._value_bits = int(value_bits)
        self._seed_base = int(seed)
        self._per_layer_compressors = bool(per_layer_compressors)
        self._n_layers = int(num_layers)
        dev_str = str(device)

        self._value_group_size = int(value_group_size)

        if self._key_bits < 2:
            raise ValueError("TurboQuantProd requires key_bits >= 2 (inner MSE uses key_bits - 1).")
        if head_dim % self._value_group_size != 0:
            raise ValueError(f"head_dim {head_dim} must divide value_group_size {self._value_group_size} for group value quant.")

        device_t = torch.device(dev_str)
        inf_dtype = torch.float32
        nk_bits = self._key_bits - 1
        cb_key = tq_fw_load_codebook_record(
            head_dim,
            nk_bits,
            codebook_dir,
            codebook_cache_dir,
            export_missing_codebooks,
        )

        self._inf_nk = tq_fw_packed_width(head_dim, nk_bits)
        self._inf_nqjl = (head_dim + 7) // 8

        def _make_k_mod(seed_k: int) -> torch.nn.Module:
            return TurboQuantProdInference(head_dim, self._key_bits, device_t, seed_k, cb_key, dtype=inf_dtype)

        if self._per_layer_compressors:
            self._k_inference_modules = [_make_k_mod(self._seed_base + lid * 7) for lid in range(self._n_layers)]
        else:
            _km = _make_k_mod(self._seed_base)
            self._k_inference_modules = [_km for _ in range(self._n_layers)]

        self._inf_v_width = tq_value_group_packed_width(head_dim, self._value_bits)
        self._inf_v_n_groups = head_dim // self._value_group_size

        super().__init__(num_layers, cache_size, num_heads, head_dim, dtype, device, kv_offload=kv_offload)

    def _k_mod_inf(self, layer_id: int) -> torch.nn.Module:
        assert self._k_inference_modules is not None
        return self._k_inference_modules[layer_id]

    def _init_kv_buffer(self) -> None:
        if self._kv_offload:
            self._init_kv_buffer_offload()
            return

        L = self._num_layers
        N = self._cache_size
        H = self._num_heads

        self._k_packed = torch.zeros(L, N, H, self._inf_nk, dtype=torch.uint8, device=self._device)
        self._k_norms = torch.zeros(L, N, H, dtype=torch.float16, device=self._device)
        self._k_qjl_packed = torch.zeros(L, N, H, self._inf_nqjl, dtype=torch.uint8, device=self._device)
        self._k_res_norms = torch.zeros(L, N, H, dtype=torch.float16, device=self._device)
        ng = self._inf_v_n_groups
        self._v_group_data = torch.zeros(L, N, H, self._inf_v_width, dtype=torch.uint8, device=self._device)
        self._v_group_scales = torch.zeros(L, N, H, ng, dtype=torch.float16, device=self._device)
        self._v_group_zeros = torch.zeros(L, N, H, ng, dtype=torch.float16, device=self._device)

        self._global_end = torch.zeros(L, dtype=torch.long, device=self._device)
        self._local_end = torch.zeros(L, dtype=torch.long, device=self._device)

    def _init_kv_buffer_offload(self) -> None:
        L = self._num_layers
        N = self._cache_size
        H = self._num_heads
        nk = self._inf_nk
        nqjl = self._inf_nqjl
        vw = self._inf_v_width
        ng = self._inf_v_n_groups
        d = self._device

        self._k_packed_cpu = torch.zeros(L, N, H, nk, dtype=torch.uint8, device="cpu").pin_memory()
        self._k_norms_cpu = torch.zeros(L, N, H, dtype=torch.float16, device="cpu").pin_memory()
        self._k_qjl_packed_cpu = torch.zeros(L, N, H, nqjl, dtype=torch.uint8, device="cpu").pin_memory()
        self._k_res_norms_cpu = torch.zeros(L, N, H, dtype=torch.float16, device="cpu").pin_memory()
        self._v_group_data_cpu = torch.zeros(L, N, H, vw, dtype=torch.uint8, device="cpu").pin_memory()
        self._v_group_scales_cpu = torch.zeros(L, N, H, ng, dtype=torch.float16, device="cpu").pin_memory()
        self._v_group_zeros_cpu = torch.zeros(L, N, H, ng, dtype=torch.float16, device="cpu").pin_memory()

        self._k_packed_gpu = torch.zeros(2, N, H, nk, dtype=torch.uint8, device=d)
        self._k_norms_gpu = torch.zeros(2, N, H, dtype=torch.float16, device=d)
        self._k_qjl_packed_gpu = torch.zeros(2, N, H, nqjl, dtype=torch.uint8, device=d)
        self._k_res_norms_gpu = torch.zeros(2, N, H, dtype=torch.float16, device=d)
        self._v_group_data_gpu = torch.zeros(2, N, H, vw, dtype=torch.uint8, device=d)
        self._v_group_scales_gpu = torch.zeros(2, N, H, ng, dtype=torch.float16, device=d)
        self._v_group_zeros_gpu = torch.zeros(2, N, H, ng, dtype=torch.float16, device=d)

        self._global_end = torch.zeros(L, dtype=torch.long, device=d)
        self._local_end = torch.zeros(L, dtype=torch.long, device=d)

        def _async_load(layer_id: int, buf: int) -> None:
            self._k_packed_gpu[buf].copy_(self._k_packed_cpu[layer_id], non_blocking=True)
            self._k_norms_gpu[buf].copy_(self._k_norms_cpu[layer_id], non_blocking=True)
            self._k_qjl_packed_gpu[buf].copy_(self._k_qjl_packed_cpu[layer_id], non_blocking=True)
            self._k_res_norms_gpu[buf].copy_(self._k_res_norms_cpu[layer_id], non_blocking=True)
            self._v_group_data_gpu[buf].copy_(self._v_group_data_cpu[layer_id], non_blocking=True)
            self._v_group_scales_gpu[buf].copy_(self._v_group_scales_cpu[layer_id], non_blocking=True)
            self._v_group_zeros_gpu[buf].copy_(self._v_group_zeros_cpu[layer_id], non_blocking=True)

        def _async_store(layer_id: int, buf: int, start: int, end: int) -> None:
            self._k_packed_cpu[layer_id, start:end].copy_(self._k_packed_gpu[buf, start:end], non_blocking=True)
            self._k_norms_cpu[layer_id, start:end].copy_(self._k_norms_gpu[buf, start:end], non_blocking=True)
            self._k_qjl_packed_cpu[layer_id, start:end].copy_(self._k_qjl_packed_gpu[buf, start:end], non_blocking=True)
            self._k_res_norms_cpu[layer_id, start:end].copy_(self._k_res_norms_gpu[buf, start:end], non_blocking=True)
            self._v_group_data_cpu[layer_id, start:end].copy_(self._v_group_data_gpu[buf, start:end], non_blocking=True)
            self._v_group_scales_cpu[layer_id, start:end].copy_(self._v_group_scales_gpu[buf, start:end], non_blocking=True)
            self._v_group_zeros_cpu[layer_id, start:end].copy_(self._v_group_zeros_gpu[buf, start:end], non_blocking=True)

        self._offload = KVOffloadPlugin(self._device, _async_load, _async_store)
        gpu_mb = (
            self._k_packed_gpu.nbytes
            + self._k_norms_gpu.nbytes
            + self._k_qjl_packed_gpu.nbytes
            + self._k_res_norms_gpu.nbytes
            + self._v_group_data_gpu.nbytes
            + self._v_group_scales_gpu.nbytes
            + self._v_group_zeros_gpu.nbytes
        ) / (1024 * 1024)
        cpu_mb = (
            self._k_packed_cpu.nbytes
            + self._k_norms_cpu.nbytes
            + self._k_qjl_packed_cpu.nbytes
            + self._k_res_norms_cpu.nbytes
            + self._v_group_data_cpu.nbytes
            + self._v_group_scales_cpu.nbytes
            + self._v_group_zeros_cpu.nbytes
        ) / (1024 * 1024)
        logger.info(
            "[TurboQuantRollingKVCachePool+offload] GPU fixed buffer: {:.1f} MB, CPU pinned: {:.1f} MB (saved {:.1f} MB GPU)",
            gpu_mb,
            cpu_mb,
            cpu_mb - gpu_mb,
        )

    def _tq_k_packed(self, layer_id: int) -> torch.Tensor:
        if self._kv_offload:
            return self._k_packed_gpu[self._offload.cur_buf]
        return self._k_packed[layer_id]

    def _tq_k_norms(self, layer_id: int) -> torch.Tensor:
        if self._kv_offload:
            return self._k_norms_gpu[self._offload.cur_buf]
        return self._k_norms[layer_id]

    def _tq_k_qjl_packed(self, layer_id: int) -> torch.Tensor:
        if self._kv_offload:
            return self._k_qjl_packed_gpu[self._offload.cur_buf]
        return self._k_qjl_packed[layer_id]

    def _tq_k_res_norms(self, layer_id: int) -> torch.Tensor:
        if self._kv_offload:
            return self._k_res_norms_gpu[self._offload.cur_buf]
        return self._k_res_norms[layer_id]

    def _tq_v_group_data(self, layer_id: int) -> torch.Tensor:
        if self._kv_offload:
            return self._v_group_data_gpu[self._offload.cur_buf]
        return self._v_group_data[layer_id]

    def _tq_v_group_scales(self, layer_id: int) -> torch.Tensor:
        if self._kv_offload:
            return self._v_group_scales_gpu[self._offload.cur_buf]
        return self._v_group_scales[layer_id]

    def _tq_v_group_zeros(self, layer_id: int) -> torch.Tensor:
        if self._kv_offload:
            return self._v_group_zeros_gpu[self._offload.cur_buf]
        return self._v_group_zeros[layer_id]

    @staticmethod
    def _sh_extra_to_bhs(extra_sh: torch.Tensor) -> torch.Tensor:
        """(S, H, G) -> (1, H, S, G)."""
        return extra_sh.unsqueeze(0).permute(0, 2, 1, 3).contiguous()

    def store_kv(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        start_idx: int,
        end_idx: int,
        layer_id: int,
    ) -> None:
        chunk_len = int(end_idx - start_idx)
        if chunk_len <= 0:
            return
        if k.size(0) != chunk_len or v.size(0) != chunk_len:
            raise ValueError(f"TurboQuantRollingKVCachePool.store_kv: chunk_len={chunk_len}, k={k.size(0)}, v={v.size(0)}.")

        k_bhsd = k.unsqueeze(0).transpose(1, 2).contiguous()  # [1, H, S, D]
        v_bhsd = v.unsqueeze(0).transpose(1, 2).contiguous()

        with torch.no_grad():
            ck = self._k_mod_inf(layer_id).compress_bhsd(k_bhsd)

        self._tq_k_packed(layer_id)[start_idx:end_idx].copy_(ck["mse_idx_bytes"][0].transpose(0, 1).contiguous())
        self._tq_k_norms(layer_id)[start_idx:end_idx].copy_(ck["vec_norms"][0].transpose(0, 1).contiguous())
        self._tq_k_qjl_packed(layer_id)[start_idx:end_idx].copy_(ck["qjl_bytes"][0].transpose(0, 1).contiguous())
        self._tq_k_res_norms(layer_id)[start_idx:end_idx].copy_(ck["residual_norms"][0].transpose(0, 1).contiguous())

        if ck["shape"][-1] != self._head_dim:
            raise RuntimeError("TurboQuant inference key compress shape mismatch.")

        with torch.no_grad():
            cv = tq_group_quantize_values(v_bhsd, self._value_bits, self._value_group_size)
        self._tq_v_group_data(layer_id)[start_idx:end_idx].copy_(cv["data"][0].transpose(0, 1).contiguous())
        self._tq_v_group_scales(layer_id)[start_idx:end_idx].copy_(cv["scales"][0].transpose(0, 1).contiguous())
        self._tq_v_group_zeros(layer_id)[start_idx:end_idx].copy_(cv["zeros"][0].transpose(0, 1).contiguous())

        if self._kv_offload:
            self._mark_offload_dirty(start_idx, end_idx)

    def k_cache(self, layer_id: int, attn_start: int, local_end: int) -> torch.Tensor:
        kv_len = local_end - attn_start
        if kv_len <= 0:
            return torch.empty(0, self._num_heads, self._head_dim, device=self._device, dtype=self._dtype)

        packed = self._tq_k_packed(layer_id)[attn_start:local_end]
        norms = self._tq_k_norms(layer_id)[attn_start:local_end]
        idx_bytes = packed.unsqueeze(0).permute(0, 2, 1, 3).contiguous()
        norms_bhs = norms.unsqueeze(0).transpose(1, 2).contiguous()
        B, H, S, D = 1, self._num_heads, kv_len, self._head_dim

        qjl_bhs = self._sh_extra_to_bhs(self._tq_k_qjl_packed(layer_id)[attn_start:local_end])
        res_bhs = self._tq_k_res_norms(layer_id)[attn_start:local_end].unsqueeze(0).transpose(1, 2).contiguous()
        comp = {
            "mse_idx_bytes": idx_bytes,
            "qjl_bytes": qjl_bhs,
            "residual_norms": res_bhs,
            "vec_norms": norms_bhs,
            "shape": (B, H, S, D),
            "mse_bits": self._key_bits - 1,
        }
        with torch.no_grad():
            out_bhsd = self._k_mod_inf(layer_id).decompress_bhsd(comp)
        return out_bhsd[0].transpose(0, 1).to(dtype=self._dtype)

    def v_cache(self, layer_id: int, attn_start: int, local_end: int) -> torch.Tensor:
        kv_len = local_end - attn_start
        if kv_len <= 0:
            return torch.empty(0, self._num_heads, self._head_dim, device=self._device, dtype=self._dtype)

        data = self._tq_v_group_data(layer_id)[attn_start:local_end]
        scales = self._tq_v_group_scales(layer_id)[attn_start:local_end]
        zeros = self._tq_v_group_zeros(layer_id)[attn_start:local_end]
        comp = {
            "data": data.unsqueeze(0).permute(0, 2, 1, 3).contiguous(),
            "scales": scales.unsqueeze(0).transpose(1, 2).contiguous(),
            "zeros": zeros.unsqueeze(0).transpose(1, 2).contiguous(),
            "bits": self._value_bits,
            "group_size": self._value_group_size,
            "shape": (1, self._num_heads, kv_len, self._head_dim),
        }
        with torch.no_grad():
            out_bhsd = tq_group_dequantize_values(comp)
        return out_bhsd[0].transpose(0, 1).to(dtype=self._dtype)

    def roll_window(
        self,
        layer_id: int,
        sink_tokens: int,
        num_evicted: int,
    ) -> None:
        num_kept = int(self._local_end[layer_id].item()) - num_evicted - sink_tokens
        if num_kept <= 0:
            return
        src_start = sink_tokens + num_evicted
        src_end = src_start + num_kept
        dst_start = sink_tokens
        dst_end = dst_start + num_kept

        self._tq_k_packed(layer_id)[dst_start:dst_end].copy_(self._tq_k_packed(layer_id)[src_start:src_end].clone())
        self._tq_k_norms(layer_id)[dst_start:dst_end].copy_(self._tq_k_norms(layer_id)[src_start:src_end].clone())
        self._tq_k_qjl_packed(layer_id)[dst_start:dst_end].copy_(self._tq_k_qjl_packed(layer_id)[src_start:src_end].clone())
        self._tq_k_res_norms(layer_id)[dst_start:dst_end].copy_(self._tq_k_res_norms(layer_id)[src_start:src_end].clone())
        self._tq_v_group_data(layer_id)[dst_start:dst_end].copy_(self._tq_v_group_data(layer_id)[src_start:src_end].clone())
        self._tq_v_group_scales(layer_id)[dst_start:dst_end].copy_(self._tq_v_group_scales(layer_id)[src_start:src_end].clone())
        self._tq_v_group_zeros(layer_id)[dst_start:dst_end].copy_(self._tq_v_group_zeros(layer_id)[src_start:src_end].clone())
        if self._kv_offload:
            self._mark_offload_dirty(dst_start, dst_end)

    def reset(self) -> None:
        if self._kv_offload:
            self._k_packed_cpu.zero_()
            self._k_norms_cpu.zero_()
            self._k_qjl_packed_cpu.zero_()
            self._k_res_norms_cpu.zero_()
            self._v_group_data_cpu.zero_()
            self._v_group_scales_cpu.zero_()
            self._v_group_zeros_cpu.zero_()
            self._k_packed_gpu.zero_()
            self._k_norms_gpu.zero_()
            self._k_qjl_packed_gpu.zero_()
            self._k_res_norms_gpu.zero_()
            self._v_group_data_gpu.zero_()
            self._v_group_scales_gpu.zero_()
            self._v_group_zeros_gpu.zero_()
            self._global_end.zero_()
            self._local_end.zero_()
            self._offload.reset_state()
            return

        self._k_packed.zero_()
        self._k_norms.zero_()
        self._k_qjl_packed.zero_()
        self._k_res_norms.zero_()
        self._v_group_data.zero_()
        self._v_group_scales.zero_()
        self._v_group_zeros.zero_()
        self._global_end.zero_()
        self._local_end.zero_()


class KIVIQuantRollingKVCachePool(RollingKVCachePool):
    def __init__(
        self,
        num_layers: int,
        cache_size: int,
        num_heads: int,
        head_dim: int,
        dtype: torch.dtype,
        device: torch.device,
        k_cache_type: str = "int4",
        v_cache_type: str = "int4",
        group_size: int = 64,
        kv_offload: bool = False,
    ) -> None:
        assert k_cache_type in ["int2", "int4", "int8"], f"Invalid k_cache_type: {k_cache_type}"
        assert v_cache_type in ["int2", "int4", "int8"], f"Invalid v_cache_type: {v_cache_type}"
        assert k_cache_type == v_cache_type, "k_cache_type and v_cache_type must be the same"
        self._bits = int(k_cache_type[-1])
        self._group_size = group_size
        self._feats = 32 // self._bits
        self._align = lcm(self._feats, group_size)
        n_alloc = cdiv(int(cache_size), self._align) * self._align
        self.current_step: int = 0
        self._N_alloc = n_alloc
        self._kivi_io_dtype = torch.float16
        super().__init__(num_layers, n_alloc, num_heads, head_dim, dtype, device, kv_offload=kv_offload)

    @staticmethod
    def _nhd_to_bhdt(nhd: torch.Tensor) -> torch.Tensor:
        return nhd.permute(1, 2, 0).contiguous().unsqueeze(0)

    @staticmethod
    def _slice_token_range(nhd: torch.Tensor, t0: int, t1: int) -> torch.Tensor:
        return nhd[t0:t1, :, :].contiguous()

    def _quant_nhd(
        self,
        nhd: torch.Tensor,
    ) -> tuple[tuple[torch.Tensor, torch.Tensor, torch.Tensor], int, int]:
        T = nhd.size(0)
        if T == 0:
            raise ValueError("empty K/V chunk in KIVI store")
        T_pad = cdiv(T, self._align) * self._align
        if nhd.size(0) < T_pad:
            pad = nhd.new_zeros((T_pad - nhd.size(0),) + nhd.shape[1:])
            nhd = torch.cat((nhd, pad), dim=0)
        elif nhd.size(0) > T_pad:
            nhd = nhd[:T_pad]
        t4 = self._nhd_to_bhdt(nhd.to(self._kivi_io_dtype))
        trip = triton_quantize_and_pack_along_last_dim(t4, self._group_size, self._bits)
        return (trip[0], trip[1], trip[2]), T, T_pad

    @staticmethod
    def _dequant_bhdn(
        code4: torch.Tensor,
        sc: torch.Tensor,
        mn: torch.Tensor,
        group_size: int,
        bits: int,
        as_dtype: torch.dtype,
    ) -> torch.Tensor:
        # code4 [1, H, D, n_packs], sc/mn [1, H, D, n_groups]
        # Match kernel.test_vcache: last dim of scale/mn must be 1 to broadcast
        # over the (num_groups, group_size) view inside unpack_and_dequant_cache.
        out = unpack_and_dequant_cache(code4, sc.unsqueeze(-1), mn.unsqueeze(-1), group_size, bits)
        return out.to(as_dtype).squeeze(0)  # [H, D, T]

    def _init_kv_buffer(self) -> None:
        if self._kv_offload:
            self._init_kv_buffer_offload()
            return
        L = self._num_layers
        N = self._N_alloc
        H, D = self._num_heads, self._head_dim
        fe, G = self._feats, self._group_size
        n_packs = N // fe
        n_groups = N // G
        self._kivi_n_packs = n_packs
        self._kivi_n_groups = n_groups
        d = self._device

        self._k_code = torch.zeros(L, H, D, n_packs, dtype=torch.int32, device=d)
        self._v_code = torch.zeros(L, H, D, n_packs, dtype=torch.int32, device=d)
        self._k_scale = torch.zeros(L, H, D, n_groups, dtype=torch.float32, device=d)
        self._k_mn = torch.zeros(L, H, D, n_groups, dtype=torch.float32, device=d)
        self._v_scale = torch.zeros(L, H, D, n_groups, dtype=torch.float32, device=d)
        self._v_mn = torch.zeros(L, H, D, n_groups, dtype=torch.float32, device=d)
        self._global_end = torch.zeros(L, dtype=torch.long, device=d)
        self._local_end = torch.zeros(L, dtype=torch.long, device=d)

    def _init_kv_buffer_offload(self) -> None:
        L = self._num_layers
        N = self._N_alloc
        H, D = self._num_heads, self._head_dim
        fe, G = self._feats, self._group_size
        n_packs = N // fe
        n_groups = N // G
        self._kivi_n_packs = n_packs
        self._kivi_n_groups = n_groups
        d = self._device

        self._k_code_cpu = torch.zeros(L, H, D, n_packs, dtype=torch.int32, device="cpu").pin_memory()
        self._v_code_cpu = torch.zeros(L, H, D, n_packs, dtype=torch.int32, device="cpu").pin_memory()
        self._k_scale_cpu = torch.zeros(L, H, D, n_groups, dtype=torch.float32, device="cpu").pin_memory()
        self._k_mn_cpu = torch.zeros(L, H, D, n_groups, dtype=torch.float32, device="cpu").pin_memory()
        self._v_scale_cpu = torch.zeros(L, H, D, n_groups, dtype=torch.float32, device="cpu").pin_memory()
        self._v_mn_cpu = torch.zeros(L, H, D, n_groups, dtype=torch.float32, device="cpu").pin_memory()
        self._k_code_gpu = torch.zeros(2, H, D, n_packs, dtype=torch.int32, device=d)
        self._v_code_gpu = torch.zeros(2, H, D, n_packs, dtype=torch.int32, device=d)
        self._k_scale_gpu = torch.zeros(2, H, D, n_groups, dtype=torch.float32, device=d)
        self._k_mn_gpu = torch.zeros(2, H, D, n_groups, dtype=torch.float32, device=d)
        self._v_scale_gpu = torch.zeros(2, H, D, n_groups, dtype=torch.float32, device=d)
        self._v_mn_gpu = torch.zeros(2, H, D, n_groups, dtype=torch.float32, device=d)
        self._global_end = torch.zeros(L, dtype=torch.long, device=d)
        self._local_end = torch.zeros(L, dtype=torch.long, device=d)

        def _async_load(lid: int, buf: int) -> None:
            self._k_code_gpu[buf].copy_(self._k_code_cpu[lid], non_blocking=True)
            self._v_code_gpu[buf].copy_(self._v_code_cpu[lid], non_blocking=True)
            self._k_scale_gpu[buf].copy_(self._k_scale_cpu[lid], non_blocking=True)
            self._k_mn_gpu[buf].copy_(self._k_mn_cpu[lid], non_blocking=True)
            self._v_scale_gpu[buf].copy_(self._v_scale_cpu[lid], non_blocking=True)
            self._v_mn_gpu[buf].copy_(self._v_mn_cpu[lid], non_blocking=True)

        def _async_store(lid: int, buf: int, t0: int, t1: int) -> None:
            fe, G = self._feats, self._group_size
            p0, p1 = t0 // fe, cdiv(t1, fe)
            p1 = min(p1, self._kivi_n_packs)
            g0, g1 = t0 // G, cdiv(t1, G)
            g1 = min(g1, self._kivi_n_groups)
            if p0 < p1:
                self._k_code_cpu[lid, :, :, p0:p1].copy_(self._k_code_gpu[buf, :, :, p0:p1], non_blocking=True)
                self._v_code_cpu[lid, :, :, p0:p1].copy_(self._v_code_gpu[buf, :, :, p0:p1], non_blocking=True)
            if g0 < g1:
                self._k_scale_cpu[lid, :, :, g0:g1].copy_(self._k_scale_gpu[buf, :, :, g0:g1], non_blocking=True)
                self._k_mn_cpu[lid, :, :, g0:g1].copy_(self._k_mn_gpu[buf, :, :, g0:g1], non_blocking=True)
                self._v_scale_cpu[lid, :, :, g0:g1].copy_(self._v_scale_gpu[buf, :, :, g0:g1], non_blocking=True)
                self._v_mn_cpu[lid, :, :, g0:g1].copy_(self._v_mn_gpu[buf, :, :, g0:g1], non_blocking=True)

        self._offload = KVOffloadPlugin(self._device, _async_load, _async_store)
        gpu_mb = (self._k_code_gpu.nbytes + self._v_code_gpu.nbytes + self._k_scale_gpu.nbytes + self._k_mn_gpu.nbytes + self._v_scale_gpu.nbytes + self._v_mn_gpu.nbytes) / (1024 * 1024)
        cpu_mb = (self._k_code_cpu.nbytes + self._v_code_cpu.nbytes + self._k_scale_cpu.nbytes + self._k_mn_cpu.nbytes + self._v_scale_cpu.nbytes + self._v_mn_cpu.nbytes) / (1024 * 1024)
        logger.info(
            "[KIVIQuantRollingKVCachePool+offload] GPU fixed buffer: {:.1f} MB, CPU pinned: {:.1f} MB (saved {:.1f} MB GPU)",
            gpu_mb,
            cpu_mb,
            cpu_mb - gpu_mb,
        )

    def _kivi_k_code(self, _layer_id: int) -> torch.Tensor:
        if self._kv_offload:
            return self._k_code_gpu[self._offload.cur_buf]
        return self._k_code[_layer_id]

    def _kivi_v_code(self, _layer_id: int) -> torch.Tensor:
        if self._kv_offload:
            return self._v_code_gpu[self._offload.cur_buf]
        return self._v_code[_layer_id]

    def _kivi_k_scale(self, _layer_id: int) -> torch.Tensor:
        if self._kv_offload:
            return self._k_scale_gpu[self._offload.cur_buf]
        return self._k_scale[_layer_id]

    def _kivi_k_mn(self, _layer_id: int) -> torch.Tensor:
        if self._kv_offload:
            return self._k_mn_gpu[self._offload.cur_buf]
        return self._k_mn[_layer_id]

    def _kivi_v_scale(self, _layer_id: int) -> torch.Tensor:
        if self._kv_offload:
            return self._v_scale_gpu[self._offload.cur_buf]
        return self._v_scale[_layer_id]

    def _kivi_v_mn(self, _layer_id: int) -> torch.Tensor:
        if self._kv_offload:
            return self._v_mn_gpu[self._offload.cur_buf]
        return self._v_mn[_layer_id]

    def _write_segment(
        self,
        code: torch.Tensor,
        sc: torch.Tensor,
        mn: torch.Tensor,
        layer: int,
        t_start: int,
    ) -> None:
        """Write quant outputs for a chunk placed at **token** ``t_start`` (0-based)."""
        H, D = self._num_heads, self._head_dim
        # code [1, H, D, n_pl], n_pl = T_pad / fe
        b, h, d, np_l = code.shape
        assert b == 1 and h == H and d == D
        fe, G = self._feats, self._group_size
        t_pad = code.shape[3] * fe
        g_cnt = t_pad // G
        t0, t1 = t_start, t_start + t_pad
        p0, p1 = t0 // fe, t0 // fe + code.shape[3]
        g0, g1 = t0 // G, t0 // G + g_cnt
        if t1 > self._N_alloc:
            raise RuntimeError("KIVI store overflow (increase max_attention or alignment)")
        if p0 + code.shape[3] > self._kivi_n_packs:
            raise RuntimeError("KIVI pack range overflow")
        csl = code[0]
        self._kivi_k_code(layer)[:, :, p0:p1] = csl
        self._kivi_k_scale(layer)[:, :, g0:g1] = sc[0, :, :, :g_cnt]
        self._kivi_k_mn(layer)[:, :, g0:g1] = mn[0, :, :, :g_cnt]

    def _write_v_segment(
        self,
        code: torch.Tensor,
        sc: torch.Tensor,
        mn: torch.Tensor,
        layer: int,
        t_start: int,
    ) -> None:
        H, D = self._num_heads, self._head_dim
        b, h, d, _np = code.shape
        assert b == 1 and h == H and d == D
        fe, G = self._feats, self._group_size
        t_pad = code.shape[3] * fe
        g_cnt = t_pad // G
        t0, t1 = t_start, t_start + t_pad
        p0, p1 = t0 // fe, t0 // fe + code.shape[3]
        g0, g1 = t0 // G, t0 // G + g_cnt
        if t1 > self._N_alloc:
            raise RuntimeError("KIVI store overflow")
        csl = code[0]
        self._kivi_v_code(layer)[:, :, p0:p1] = csl
        self._kivi_v_scale(layer)[:, :, g0:g1] = sc[0, :, :, :g_cnt]
        self._kivi_v_mn(layer)[:, :, g0:g1] = mn[0, :, :, :g_cnt]

    def store_kv(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        start_idx: int,
        end_idx: int,
        layer_id: int,
    ) -> None:
        m = self._align
        Ls = end_idx - start_idx
        if Ls == 0:
            return
        s0 = (start_idx // m) * m
        lid = layer_id
        d = self._kivi_io_dtype
        parts_k = []
        parts_v = []
        if s0 < start_idx:
            pk = self._dequant_nhd(
                self._kivi_k_code(lid),
                self._kivi_k_scale(lid).to(d),
                self._kivi_k_mn(lid).to(d),
                s0,
                start_idx,
            )
            pv = self._dequant_nhd(
                self._kivi_v_code(lid),
                self._kivi_v_scale(lid).to(d),
                self._kivi_v_mn(lid).to(d),
                s0,
                start_idx,
            )
            need = start_idx - s0
            if pk.size(0) < need:
                z = k.new_zeros(need - pk.size(0), *k.shape[1:], dtype=pk.dtype, device=pk.device)
                pk = torch.cat((pk, z), dim=0)
            if pv.size(0) < need:
                z2 = v.new_zeros(need - pv.size(0), *v.shape[1:], dtype=pv.dtype, device=pv.device)
                pv = torch.cat((pv, z2), dim=0)
            parts_k.append(pk)
            parts_v.append(pv)
        parts_k.append(self._slice_token_range(k, 0, Ls))
        parts_v.append(self._slice_token_range(v, 0, Ls))
        k_cat = torch.cat(parts_k, dim=0)
        v_cat = torch.cat(parts_v, dim=0)
        (k_code, k_sc, k_mn), _, t_pad_k = self._quant_nhd(k_cat)
        (v_code, v_sc, v_mn), _, t_pad_v = self._quant_nhd(v_cat)
        if t_pad_k != t_pad_v:
            raise RuntimeError("KIVI store: K/V padded length mismatch")
        self._write_segment(k_code, k_sc, k_mn, layer_id, s0)
        self._write_v_segment(v_code, v_sc, v_mn, layer_id, s0)
        if self._kv_offload:
            self._mark_offload_dirty(s0, s0 + t_pad_k)

    def _dequant_nhd(
        self,
        code: torch.Tensor,
        sc: torch.Tensor,
        mn: torch.Tensor,
        attn_start: int,
        local_end: int,
    ) -> torch.Tensor:
        H, D = self._num_heads, self._head_dim
        m = self._align
        t0 = (attn_start // m) * m
        t1 = min(cdiv(max(local_end, 0), m) * m, self._N_alloc)
        if t1 <= t0 or local_end <= attn_start:
            return torch.empty(0, H, D, device=self._device, dtype=self._dtype)
        fe, G = self._feats, self._group_size
        p0, p1 = t0 // fe, t1 // fe
        g0, g1 = t0 // G, t1 // G
        c4 = code[:, :, p0:p1].unsqueeze(0)
        out = self._dequant_bhdn(c4, sc[:, :, g0:g1].unsqueeze(0), mn[:, :, g0:g1].unsqueeze(0), self._group_size, self._bits, self._dtype)
        nhd = out.permute(2, 0, 1)
        o0 = max(attn_start, t0) - t0
        o1 = o0 + (local_end - max(attn_start, t0))
        return nhd[o0:o1].contiguous()

    def k_cache(self, layer_id: int, attn_start: int, local_end: int) -> torch.Tensor:
        d = self._kivi_io_dtype
        o = self._dequant_nhd(
            self._kivi_k_code(layer_id),
            self._kivi_k_scale(layer_id).to(d),
            self._kivi_k_mn(layer_id).to(d),
            attn_start,
            local_end,
        )
        if self._dtype in (torch.bfloat16, torch.float32) and o.dtype != self._dtype:
            return o.to(self._dtype)
        return o

    def v_cache(self, layer_id: int, attn_start: int, local_end: int) -> torch.Tensor:
        d = self._kivi_io_dtype
        o = self._dequant_nhd(
            self._kivi_v_code(layer_id),
            self._kivi_v_scale(layer_id).to(d),
            self._kivi_v_mn(layer_id).to(d),
            attn_start,
            local_end,
        )
        if self._dtype in (torch.bfloat16, torch.float32) and o.dtype != self._dtype:
            return o.to(self._dtype)
        return o

    def roll_window(
        self,
        layer_id: int,
        sink_tokens: int,
        num_evicted: int,
    ) -> None:
        num_kept = int(self._local_end[layer_id].item()) - num_evicted - sink_tokens
        src_start = sink_tokens + num_evicted
        dst_start = sink_tokens
        if num_kept <= 0:
            return
        fe, G = self._feats, self._group_size
        t0, t1 = int(src_start), int(src_start + num_kept)
        d0, d1 = int(dst_start), int(dst_start + num_kept)
        p0, p1 = t0 // fe, cdiv(t1, fe)
        p2, p3 = d0 // fe, cdiv(d1, fe)
        w = p1 - p0
        if w != p3 - p2 or p0 + w > self._kivi_n_packs or p2 + w > self._kivi_n_packs:
            raise RuntimeError("KIVI roll: pack range mismatch (internal alignment).")
        g0, g1 = t0 // G, cdiv(t1, G)
        h0, h1 = d0 // G, cdiv(d1, G)
        w_g = g1 - g0
        if w_g != h1 - h0 or g0 + w_g > self._kivi_n_groups or h0 + w_g > self._kivi_n_groups:
            raise RuntimeError("KIVI roll: group range mismatch (internal alignment).")
        lid = layer_id
        kc, vc = self._kivi_k_code(lid), self._kivi_v_code(lid)
        kc[:, :, p2 : p2 + w] = kc[:, :, p0 : p0 + w].clone()
        vc[:, :, p2 : p2 + w] = vc[:, :, p0 : p0 + w].clone()
        for tbuf in (
            self._kivi_k_scale(lid),
            self._kivi_k_mn(lid),
            self._kivi_v_scale(lid),
            self._kivi_v_mn(lid),
        ):
            tbuf[:, :, h0 : h0 + w_g] = tbuf[:, :, g0 : g0 + w_g].clone()
        if self._kv_offload:
            self._mark_offload_dirty(dst_start, dst_start + num_kept)

    def reset(self) -> None:
        if self._kv_offload:
            self._k_code_cpu.zero_()
            self._v_code_cpu.zero_()
            self._k_scale_cpu.zero_()
            self._k_mn_cpu.zero_()
            self._v_scale_cpu.zero_()
            self._v_mn_cpu.zero_()
            self._k_code_gpu.zero_()
            self._v_code_gpu.zero_()
            self._k_scale_gpu.zero_()
            self._k_mn_gpu.zero_()
            self._v_scale_gpu.zero_()
            self._v_mn_gpu.zero_()
            self._global_end.zero_()
            self._local_end.zero_()
            self._offload.reset_state()
            return
        self._k_code.zero_()
        self._v_code.zero_()
        self._k_scale.zero_()
        self._k_mn.zero_()
        self._v_scale.zero_()
        self._v_mn.zero_()
        self._global_end.zero_()
        self._local_end.zero_()
