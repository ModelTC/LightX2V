import torch
import math
import torch.distributed as dist
import os

try:
    from sageattention.triton.quant_per_thread import quant_key_per_thread_int8_kernel
except ImportError:
    quant_key_per_thread_int8_kernel = None

from .kernel import *
from .offload import KVOffloadPlugin
from .rolling import RollingKVCachePool
from loguru import logger
from lightx2v.common.ops.attn.utils.all2all import all2all_seq2head

_FP8_MAX = 448.0


def _ranked_calib_path(path: str, rank: int) -> str:
    if not path:
        return path
    dot = path.rfind(".")
    if dot <= 0:
        return f"{path}.rank{rank}"
    return f"{path[:dot]}.rank{rank}{path[dot:]}"

def _cdiv(n: int, m: int) -> int:
    return (n + m - 1) // m


def _lcm(a: int, b: int) -> int:
    if a == 0 or b == 0:
        return max(a, b) or 1
    return a * b // math.gcd(a, b)


class CalibRollingKVCachePool(RollingKVCachePool):
    """Normal bf16 rolling cache that additionally captures the (km,
    v_channel_max, k_block_scale) that sage_attn computes internally —
    keyed by ``(step, layer)`` and **shared across all chunks**.

    Capture semantics
    -----------------
    Each chunk's call to ``capture_attn`` runs sage's K-quant kernel on
    the full attention window currently in the buffer and overwrites the
    entries at ``[step, layer]`` — but only as long as the window keeps
    growing.  Once rolling kicks in (window stops growing), captures are
    skipped: the rolled-state buffer no longer matches what early-chunk
    inference will see at those positions, so freezing the pre-roll
    snapshot gives consistent calibration.

    The scales are stored at *buffer-absolute* block positions so that
    the quant cache can index them directly when storing later chunks.

    After inference, ``export_calibration()`` returns:
        ``km``            shape [S, L,         1,        H, D]    fp32
        ``v_scale``       shape [S, L,                   H, D]    fp32
        ``k_block_scale`` shape [S, L, max_blks, H, scales_per_blk] fp32

    Set ``current_step`` before each denoising step so captures land in
    the right slot.

    Implementation notes
    --------------------
    - The K slice fed to the calibration kernel starts at ``aligned_start
      = (attn_start // 128) * 128`` so the per-block scales line up with
      the buffer's natural 128-token blocks. The same alignment is used
      by ``QuantRollingKVCachePool.k_cache`` / ``v_cache`` with
      ``attn_start`` and ``local_end``.
    - km is captured in bf16 (matching sage's ``k.mean(...)`` dtype),
      then cast to fp32 for storage. This avoids extra mantissa bits
      that would otherwise diverge from sage at the bf16 ``k - km``
      subtraction step.
    """

    _BLKK = 128
    _SCALES_PER_BLK = 4  # WARPK=128 ⇒ 4 thread groups per block per head

    def __init__(
        self,
        num_layers: int,
        cache_size: int,
        num_heads: int,
        head_dim: int,
        dtype: torch.dtype,
        device: torch.device,
        num_steps: int = 1,
    ) -> None:
        self._num_steps = num_steps
        self.current_step: int = 0
        super().__init__(num_layers, cache_size, num_heads, head_dim, dtype, device)

    def _init_kv_buffer(self) -> None:
        super()._init_kv_buffer()
        S = self._num_steps
        L, H, D = self._num_layers, self._num_heads, self._head_dim
        BLK = self._BLKK
        max_blks = (self._cache_size + BLK - 1) // BLK
        self._km = torch.zeros(S, L, 1, H, D, dtype=torch.float32, device=self._device)
        self._v_channel_max = torch.zeros(S, L, H, D, dtype=torch.float32, device=self._device)
        self._k_block_scale_calib = torch.zeros(
            S,
            L,
            max_blks,
            H,
            self._SCALES_PER_BLK,
            dtype=torch.float32,
            device=self._device,
        )
        self._capture_flag = torch.zeros(S, L, dtype=torch.bool, device=self._device)
        self._captured_window_size = torch.zeros(S, L, dtype=torch.long, device="cpu")

    def _quant_key(self, k: torch.Tensor, km: torch.Tensor | None = None, BLKK: int = 128, WARPK: int = 128):
        """Run sage's per_thread int8 K-quantisation kernel on ``k``.

        Returns ``(k_int8, k_scale)`` where ``k`` is ``[B, kv_len, H, D]`` (NHD).
        The km subtraction (if any) is done in ``k.dtype`` to match sage's
        behaviour exactly — sage does ``k - km`` in bf16, NOT fp32.

        This is the source-of-truth quantisation used both at calibration time
        (to capture the per-block scale we'll later replay) and as a reference
        for the preset-scale quantisation path.
        """
        if km is not None:
            km_lowp = km.to(k.dtype) if km.dtype != k.dtype else km
            k = k - km_lowp

        k_int8 = torch.empty(k.shape, dtype=torch.int8, device=k.device)
        b, kv_len, h_kv, head_dim = k.shape

        stride_bz_k, stride_h_k, stride_seq_k = k.stride(0), k.stride(2), k.stride(1)
        stride_bz_ko, stride_h_ko, stride_seq_ko = (
            k_int8.stride(0),
            k_int8.stride(2),
            k_int8.stride(1),
        )

        num_blk = (kv_len + BLKK - 1) // BLKK
        scales_per_blk = (BLKK // WARPK) * 4
        k_scale = torch.empty(
            (b, h_kv, num_blk * scales_per_blk),
            device=k.device,
            dtype=torch.float32,
        )

        grid = (num_blk * scales_per_blk, h_kv, b)
        quant_key_per_thread_int8_kernel[grid](
            k,
            k_int8,
            k_scale,
            kv_len,
            stride_bz_k,
            stride_h_k,
            stride_seq_k,
            stride_bz_ko,
            stride_h_ko,
            stride_seq_ko,
            k_scale.stride(0),
            k_scale.stride(1),
            C=head_dim,
            BLK=WARPK,
        )
        return k_int8, k_scale

    def capture_attn(
        self,
        layer_id: int,
        attn_start: int,
        local_end: int,
    ) -> None:
        """Capture (km, v_channel_max, k_block_scale) from the buffer's
        current state — exactly what sage_attn would see at this call.

        Parameters
        ----------
        attn_start : start position of the attention window in the buffer
                     (may not be 128-aligned).
        local_end  : end position (exclusive) — the buffer's current valid
                     length for this layer.

        The captured K slice is aligned down to the nearest 128 boundary
        so per-block scales map cleanly to buffer block indices.
        """
        BLK = self._BLKK
        aligned_start = (attn_start // BLK) * BLK
        step, layer = self.current_step, layer_id

        k_full = self._k_buffer[layer_id, aligned_start:local_end]  # [kv_len_a, H, D] bf16
        v_full = self._v_buffer[layer_id, aligned_start:local_end]  # [kv_len_a, H, D] bf16
        kv_len_a = k_full.size(0)
        if kv_len_a == 0:
            return

        prev_window = int(self._captured_window_size[step, layer].item())
        if 0 < prev_window >= kv_len_a:
            return
        self._captured_window_size[step, layer] = kv_len_a

        # ---- km (bf16 mean to match sage) ----
        km_lowp = k_full.mean(dim=0, keepdim=True)  # bf16 [1, H, D]
        self._km[step, layer] = km_lowp.to(torch.float32)

        # ---- k_block_scale via sage's quant kernel on (k - km) ----
        k_batch = k_full.unsqueeze(0).contiguous()  # [1, kv_len_a, H, D]
        _, k_scale_raw = self._quant_key(k_batch, km_lowp)  # [1, H, num_blk*4]
        num_blk_local = (kv_len_a + BLK - 1) // BLK
        k_scale_local = k_scale_raw[0].reshape(self._num_heads, num_blk_local, self._SCALES_PER_BLK).permute(1, 0, 2)  # [num_blk_local, H, 4]
        blk_offset = aligned_start // BLK
        self._k_block_scale_calib[step, layer, blk_offset : blk_offset + num_blk_local] = k_scale_local
        self._v_channel_max[step, layer] = v_full.float().abs().amax(dim=0)  # [H, D]
        self._capture_flag[step, layer] = True

    def export_calibration(self) -> dict[str, torch.Tensor]:
        v_scale = self._v_channel_max.clamp(min=1e-5) / _FP8_MAX
        return {
            "km": self._km.clone(),
            "v_scale": v_scale,
            "k_block_scale": self._k_block_scale_calib.clone(),
        }

    def reset(self) -> None:
        super().reset()
        self._km.zero_()
        self._v_channel_max.zero_()
        self._k_block_scale_calib.zero_()
        self._capture_flag.zero_()
        self._captured_window_size.zero_()


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
            rank_path = _ranked_calib_path(self._calib_path, rank)
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
        v_fp8 = quant_value_per_channel_fp8_static_scale_kernel(v, v_scale, fp8_max=_FP8_MAX)

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

    def _sp_quant_kv_to_head_shard(
        self,
        k_cache,
        v_cache,
        shard_heads: int,
        seq_p_group,
        *,
        attn_start: int | None = None,
        local_end: int | None = None,
    ):
        if not (isinstance(k_cache, tuple) and isinstance(v_cache, tuple)):
            raise TypeError("SageQuant SP path expects tuple k_cache and v_cache.")
        if len(k_cache) != 2 or len(v_cache) != 2:
            raise ValueError("Unsupported SageQuant KV tuple format in SP path.")
        if attn_start is None or local_end is None:
            raise ValueError("SageQuant SP path requires attn_start and local_end (k_scale is buffer-aligned; see k_cache).")

        cur_rank = dist.get_rank(seq_p_group)
        hs = slice(cur_rank * shard_heads, (cur_rank + 1) * shard_heads)

        k_int8, k_scale = k_cache
        v_data, v_scale = v_cache

        # Must match k_cache: slice starts at aligned 128, not at attn_start.
        BLK = self._BLKK
        aligned_start = (attn_start // BLK) * BLK
        blk_s = aligned_start // BLK

        k_nhd = self._to_nhd(k_int8, "k_int8")
        full_k_nhd = all2all_seq2head(k_nhd, group=seq_p_group)
        full_k_int8 = full_k_nhd.unsqueeze(0).contiguous()
        full_kv_len = int(full_k_nhd.size(0))

        # Rebuild k_scale to match full-seq blocks:
        # local block-scale -> per-token (buffer-absolute) -> all2all(seq->head) -> full block-scale.
        k_scale_hs = self._to_heads_scale(k_scale, "k_scale")  # [H, local_num_blk*4]
        local_kv_len = int(k_nhd.size(0))
        local_tok_scale = self._expand_k_scale_to_tokens(
            k_scale_hs, local_kv_len, aligned_start=aligned_start, buffer_blk_s=blk_s
        )  # [local_kv_len, H]
        full_tok_scale = all2all_seq2head(local_tok_scale.unsqueeze(-1), group=seq_p_group).squeeze(-1)  # [full_kv_len, shard_heads]
        k_scale_shard = self._compress_token_scale_to_block4(full_tok_scale).unsqueeze(0).contiguous()  # [1, shard_heads, ceil(full_kv_len/128)*4]

        v_nhd = self._sage_v_layout_to_nhd(v_data, local_kv_len)
        full_v_nhd = all2all_seq2head(v_nhd, group=seq_p_group)
        full_v_data = self._nhd_to_sage_v_layout(full_v_nhd)

        v_scale_hd = self._to_heads_dim_scale(v_scale, "v_scale")
        v_scale_shard = v_scale_hd[hs, :].unsqueeze(0).contiguous()

        return (full_k_int8, k_scale_shard), (full_v_data, v_scale_shard), full_kv_len

    @staticmethod
    def _to_nhd(x, name: str):
        if x.dim() == 4 and x.size(0) == 1:
            return x[0].contiguous()
        if x.dim() == 3:
            return x.contiguous()
        raise ValueError(f"Unsupported {name} shape {tuple(x.shape)} for SP quant KV.")

    @staticmethod
    def _to_heads_scale(x, name: str):
        if x.dim() == 3 and x.size(0) == 1:
            return x[0].contiguous()
        if x.dim() == 2:
            return x.contiguous()
        raise ValueError(f"Unsupported {name} shape {tuple(x.shape)} for SP quant KV.")

    @staticmethod
    def _to_heads_dim_scale(x, name: str):
        if x.dim() == 3 and x.size(0) == 1:
            return x[0].contiguous()
        if x.dim() == 2:
            return x.contiguous()
        raise ValueError(f"Unsupported {name} shape {tuple(x.shape)} for SP quant KV.")

    def _sage_v_layout_to_nhd(self, v_data: torch.Tensor, kv_len: int) -> torch.Tensor:
        if v_data.dim() == 4 and v_data.size(0) == 1:
            v_data = v_data[0]
        if v_data.dim() != 3:
            raise ValueError(f"Unsupported v_data shape {tuple(v_data.shape)} for SP quant KV.")
        d, h, padded_len = v_data.shape
        v_unperm = v_data.view(d, h, -1, 16)[:, :, :, self._INV_PERM_16].contiguous().view(d, h, padded_len)
        v_unperm = v_unperm[:, :, :kv_len]
        return v_unperm.permute(2, 1, 0).contiguous()

    def _nhd_to_sage_v_layout(self, v_nhd: torch.Tensor) -> torch.Tensor:
        kv_len, h, d = v_nhd.shape
        padded_len = (kv_len + 127) // 128 * 128
        v_dhp = v_nhd.permute(2, 1, 0).contiguous()
        if padded_len > kv_len:
            padded = v_dhp.new_zeros(d, h, padded_len)
            padded[:, :, :kv_len].copy_(v_dhp)
            v_dhp = padded
        return v_dhp.view(d, h, -1, 16)[:, :, :, self._PERM_16].contiguous().view(1, d, h, padded_len)

    @staticmethod
    def _expand_k_scale_to_tokens(
        k_scale_hs: torch.Tensor,
        kv_len: int,
        *,
        aligned_start: int,
        buffer_blk_s: int,
    ) -> torch.Tensor:
        """[H, (slice_num_blk*4)] -> [kv_len, H], one scale per (buffer token index).

        ``k_int8`` is ``buffer[aligned_start:aligned_start+kv_len]``; k_scale is
        ``k_block_scale[..., buffer_blk_s:buffer_blk_e]`` in the same order as
        :meth:`k_cache` — **not** 0..kv_len-1 block indices. Use global buffer
        indices ``g = aligned_start + t``.
        """
        if k_scale_hs.dim() != 2:
            raise ValueError(f"Expected k_scale_hs 2D, got {tuple(k_scale_hs.shape)}")
        h, total = k_scale_hs.shape
        if total % 4 != 0:
            raise ValueError(f"Expected k_scale last dim multiple of 4, got {total}")
        num_blk_slice = total // 4
        scales = k_scale_hs.view(h, num_blk_slice, 4)
        g = torch.arange(aligned_start, aligned_start + kv_len, device=k_scale_hs.device, dtype=torch.long)
        blk_global = g // 128
        rel_blk = blk_global - int(buffer_blk_s)
        if (rel_blk < 0).any() or (rel_blk >= num_blk_slice).any():
            raise RuntimeError(
                f"k_scale slice mismatch: buffer_blk_s={buffer_blk_s}, rel_blk in [{int(rel_blk.min())},{int(rel_blk.max())}], "
                f"num_blk_slice={num_blk_slice}, aligned_start={aligned_start}, kv_len={kv_len}."
            )
        thr = ((g % 128) // 2) % 4
        return scales[:, rel_blk, thr].transpose(0, 1).contiguous()

    @staticmethod
    def _compress_token_scale_to_block4(tok_scale: torch.Tensor) -> torch.Tensor:
        """[kv_len, shard_heads] -> [shard_heads, ceil(kv_len/128)*4]."""
        if tok_scale.dim() != 2:
            raise ValueError(f"Expected tok_scale 2D, got {tuple(tok_scale.shape)}")
        kv_len, shard_heads = tok_scale.shape
        num_blk = (kv_len + 127) // 128
        out = tok_scale.new_zeros(shard_heads, num_blk, 4)
        pos = torch.arange(kv_len, device=tok_scale.device, dtype=torch.long)
        blk = pos // 128
        thr = ((pos % 128) // 2) % 4
        out[:, blk, thr] = tok_scale.transpose(0, 1)
        return out.view(shard_heads, num_blk * 4)


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
        self._align = _lcm(self._feats, group_size)
        n_alloc = _cdiv(int(cache_size), self._align) * self._align
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
        T_pad = _cdiv(T, self._align) * self._align
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
        out = unpack_and_dequant_cache(
            code4, sc.unsqueeze(-1), mn.unsqueeze(-1), group_size, bits
        )
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
            p0, p1 = t0 // fe, _cdiv(t1, fe)
            p1 = min(p1, self._kivi_n_packs)
            g0, g1 = t0 // G, _cdiv(t1, G)
            g1 = min(g1, self._kivi_n_groups)
            if p0 < p1:
                self._k_code_cpu[lid, :, :, p0:p1].copy_(
                    self._k_code_gpu[buf, :, :, p0:p1], non_blocking=True
                )
                self._v_code_cpu[lid, :, :, p0:p1].copy_(
                    self._v_code_gpu[buf, :, :, p0:p1], non_blocking=True
                )
            if g0 < g1:
                self._k_scale_cpu[lid, :, :, g0:g1].copy_(
                    self._k_scale_gpu[buf, :, :, g0:g1], non_blocking=True
                )
                self._k_mn_cpu[lid, :, :, g0:g1].copy_(
                    self._k_mn_gpu[buf, :, :, g0:g1], non_blocking=True
                )
                self._v_scale_cpu[lid, :, :, g0:g1].copy_(
                    self._v_scale_gpu[buf, :, :, g0:g1], non_blocking=True
                )
                self._v_mn_cpu[lid, :, :, g0:g1].copy_(
                    self._v_mn_gpu[buf, :, :, g0:g1], non_blocking=True
                )

        self._offload = KVOffloadPlugin(self._device, _async_load, _async_store)
        gpu_mb = (
            self._k_code_gpu.nbytes
            + self._v_code_gpu.nbytes
            + self._k_scale_gpu.nbytes
            + self._k_mn_gpu.nbytes
            + self._v_scale_gpu.nbytes
            + self._v_mn_gpu.nbytes
        ) / (1024 * 1024)
        cpu_mb = (
            self._k_code_cpu.nbytes
            + self._v_code_cpu.nbytes
            + self._k_scale_cpu.nbytes
            + self._k_mn_cpu.nbytes
            + self._v_scale_cpu.nbytes
            + self._v_mn_cpu.nbytes
        ) / (1024 * 1024)
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
        t1 = min(_cdiv(max(local_end, 0), m) * m, self._N_alloc)
        if t1 <= t0 or local_end <= attn_start:
            return torch.empty(0, H, D, device=self._device, dtype=self._dtype)
        fe, G = self._feats, self._group_size
        p0, p1 = t0 // fe, t1 // fe
        g0, g1 = t0 // G, t1 // G
        c4 = code[:, :, p0:p1].unsqueeze(0)
        out = self._dequant_bhdn(
            c4, sc[:, :, g0:g1].unsqueeze(0), mn[:, :, g0:g1].unsqueeze(0), self._group_size, self._bits, self._dtype
        )
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
        p0, p1 = t0 // fe, _cdiv(t1, fe)
        p2, p3 = d0 // fe, _cdiv(d1, fe)
        w = p1 - p0
        if w != p3 - p2 or p0 + w > self._kivi_n_packs or p2 + w > self._kivi_n_packs:
            raise RuntimeError("KIVI roll: pack range mismatch (internal alignment).")
        g0, g1 = t0 // G, _cdiv(t1, G)
        h0, h1 = d0 // G, _cdiv(d1, G)
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