import torch

try:
    from sageattention.triton.quant_per_thread import quant_key_per_thread_int8_kernel
except ImportError:
    quant_key_per_thread_int8_kernel = None

from .kernel import (
    k_int8_roll_rescale_triton,
    quant_key_per_thread_int8_static_scale_kernel,
    quant_value_per_channel_fp8_static_scale_kernel,
)
from .rolling import RollingKVCachePool

_FP8_MAX = 448.0


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
        *,
        smooth_k: bool = True,
        num_steps: int = 1,
    ) -> None:
        self._smooth_k = smooth_k
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
        if self._smooth_k:
            km_lowp = k_full.mean(dim=0, keepdim=True)  # bf16 [1, H, D]
            self._km[step, layer] = km_lowp.to(torch.float32)
        else:
            km_lowp = None

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


class QuantRollingKVCachePool(RollingKVCachePool):
    _BLKK = 128
    _SCALES_PER_BLK = 4  # (BLKK // WARPK) * 4, WARPK=128
    _PERM_16 = torch.tensor([0, 1, 8, 9, 2, 3, 10, 11, 4, 5, 12, 13, 6, 7, 14, 15], dtype=torch.long, device="cuda")

    def __init__(
        self,
        num_layers: int,
        cache_size: int,
        num_heads: int,
        head_dim: int,
        dtype: torch.dtype,
        device: torch.device,
        *,
        smooth_k: bool = True,
        calib_path: str,
    ) -> None:
        self._smooth_k_sage = smooth_k
        self._calib_path = calib_path
        self.current_step: int = 0
        super().__init__(num_layers, cache_size, num_heads, head_dim, dtype, device)

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
        if not self._smooth_k_sage:
            return None
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

    def _load_calib(self) -> None:
        calib = torch.load(self._calib_path, map_location=self._device, weights_only=True)
        self._calib_km = calib["km"].to(device=self._device, dtype=torch.float32)
        self._calib_v_scale = calib["v_scale"].to(device=self._device, dtype=torch.float32)
        if "k_block_scale" not in calib:
            raise RuntimeError(f"Calibration file {self._calib_path!r} is missing 'k_block_scale'. Re-run calibration with CalibRollingKVCachePool.")
        self._calib_k_block_scale = calib["k_block_scale"].to(
            device=self._device,
            dtype=torch.float32,
        )

    def _init_kv_buffer(self) -> None:
        L = self._num_layers
        N = self._cache_size
        H = self._num_heads
        D = self._head_dim
        self._load_calib()
        self._k_buffer = torch.zeros(L, N, H, D, dtype=torch.int8, device=self._device)
        self._v_buffer = torch.zeros(L, N, H, D, dtype=torch.float8_e4m3fn, device=self._device)

        self._global_end = torch.zeros(L, dtype=torch.long, device=self._device)
        self._local_end = torch.zeros(L, dtype=torch.long, device=self._device)

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

        self._k_buffer[layer_id, start_idx:end_idx] = k_int8

        v_scale = self._lookup_v_scale(layer_id)
        v_fp8 = quant_value_per_channel_fp8_static_scale_kernel(v, v_scale, fp8_max=_FP8_MAX)
        self._v_buffer[layer_id, start_idx:end_idx] = v_fp8

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
        self._roll_window_on_k_v(
            self._k_buffer[layer_id],
            self._v_buffer[layer_id],
            layer_id,
            sink_tokens,
            num_evicted,
        )

    def k_cache(
        self,
        layer_id: int,
        attn_start: int,
        local_end: int,
    ):
        BLK = self._BLKK
        aligned_start = (attn_start // BLK) * BLK
        k_int8 = self._k_buffer[layer_id, aligned_start:local_end].unsqueeze(0).contiguous()
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
        v_fp8 = self._v_buffer[layer_id, aligned_start:local_end]
        v_fp8 = self._transpose_permute_v(v_fp8)
        v_scale = self._lookup_v_scale(layer_id).unsqueeze(0).contiguous()
        return v_fp8, v_scale
