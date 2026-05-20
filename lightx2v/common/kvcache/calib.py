import torch

try:
    from sageattention.triton.quant_per_thread import quant_key_per_thread_int8_kernel
except ImportError:
    quant_key_per_thread_int8_kernel = None

from .rolling import RollingKVCachePool
from .utils import tq_fw_generate_rotation_matrix


class CalibRollingKVCachePool(RollingKVCachePool):
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
        *,
        turboquant_calibrate: bool = False,
        key_bits: int = 3,
        turboquant_seed: int = 42,
        per_layer_compressors: bool = True,
    ) -> None:
        self._num_steps = num_steps
        self.current_step: int = 0
        self._turboquant_calibrate = bool(turboquant_calibrate)
        self._tq_key_bits = int(key_bits)
        self._turboquant_seed = int(turboquant_seed)
        self._tq_per_layer = bool(per_layer_compressors)
        if self._turboquant_calibrate and self._tq_key_bits < 2:
            raise ValueError("TurboQuantProd calibration requires key_bits >= 2")
        super().__init__(num_layers, cache_size, num_heads, head_dim, dtype, device)

    def _init_kv_buffer(self) -> None:
        super()._init_kv_buffer()
        S = self._num_steps
        L, H, D = self._num_layers, self._num_heads, self._head_dim
        self._captured_window_size = torch.zeros(S, L, dtype=torch.long, device="cpu")

        if self._turboquant_calibrate:
            self._tq_hist_k = torch.zeros(4096, dtype=torch.int64, device=self._device)
            return

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
        """Capture calibration data from the current buffer state.

        For TurboQuant calibration, only the rotated-unit-K histogram needed
        for empirical codebook export is collected. Otherwise this captures
        (km, v_channel_max, k_block_scale) from exactly what sage_attn would see.

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
        kv_len_a = k_full.size(0)
        if kv_len_a == 0:
            return

        prev_window = int(self._captured_window_size[step, layer].item())
        if 0 < prev_window >= kv_len_a:
            return
        self._captured_window_size[step, layer] = kv_len_a

        if self._turboquant_calibrate:
            self._capture_turboquant_marginals(layer_id, k_full)
            return

        v_full = self._v_buffer[layer_id, aligned_start:local_end]  # [kv_len_a, H, D] bf16

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

    def _capture_turboquant_marginals(
        self,
        layer_id: int,
        k_full: torch.Tensor,
    ) -> None:
        """Histogram rotated coordinate marginals (same convention as TurboQuant inference)."""
        D = self._head_dim
        dev = self._device
        nb = self._tq_hist_k.numel()

        seed_k = self._turboquant_seed + layer_id * 7 if self._tq_per_layer else self._turboquant_seed
        Pi_k = tq_fw_generate_rotation_matrix(D, dev, torch.float32, seed=seed_k)
        x = k_full.float()
        norms = x.norm(dim=-1, keepdim=True).clamp(min=1e-10)
        x_unit = x / norms
        y = torch.matmul(x_unit, Pi_k.T).clamp(-1.0 + 1e-7, 1.0 - 1e-7)
        idx = ((y + 1.0) * (0.5 * (nb - 1))).long().clamp(0, nb - 1)
        ones = torch.ones(idx.numel(), dtype=torch.int64, device=dev)
        self._tq_hist_k.scatter_add_(0, idx.reshape(-1), ones)

    def export_calibration(self) -> dict[str, torch.Tensor]:
        if self._turboquant_calibrate:
            return {"_turboquant_hist_k": self._tq_hist_k.clone()}

        v_scale = self._v_channel_max.clamp(min=1e-5) / 448.0
        out: dict[str, torch.Tensor] = {
            "km": self._km.clone(),
            "v_scale": v_scale,
            "k_block_scale": self._k_block_scale_calib.clone(),
        }
        return out

    def reset(self) -> None:
        super().reset()
        self._captured_window_size.zero_()
        if self._turboquant_calibrate:
            self._tq_hist_k.zero_()
            return

        self._km.zero_()
        self._v_channel_max.zero_()
        self._k_block_scale_calib.zero_()
        self._capture_flag.zero_()
