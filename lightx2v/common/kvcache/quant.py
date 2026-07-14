import torch
from loguru import logger

from .kernel import *
from .rolling import RollingKVCachePool
from .utils import *

# =============================================================================


class KIVIQuantRollingKVCachePool(RollingKVCachePool):
    """KIVI quantized rolling KV cache with recent-ring buffer and kv offload.

    Logical layout exposed to callers is unchanged:
        [0:local_end) == [fixed sink][recent logical window]

    Physical layout after the first roll:
        [fixed sink region][recent ring region]

    Non-offload:
        compressed KIVI payloads live on GPU.

    Offload:
        compressed KIVI payloads live in pinned CPU physical-ring layout, and a
        single compressed GPU staging layer holds the layer currently being
        computed.
    """

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
        self._group_size = int(group_size)
        self._feats = 32 // self._bits
        self._align = lcm(self._feats, self._group_size)
        self._N_alloc = cdiv(int(cache_size), self._align) * self._align
        self._kivi_io_dtype = torch.float16
        self.current_step: int = 0

        super().__init__(num_layers, self._N_alloc, num_heads, head_dim, dtype, device, kv_offload=kv_offload)

    # ------------------------------------------------------------------
    # Shape helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _nhd_to_bhdt(nhd: torch.Tensor) -> torch.Tensor:
        return nhd.permute(1, 2, 0).contiguous().unsqueeze(0)

    # ------------------------------------------------------------------
    # Quant / dequant
    # ------------------------------------------------------------------
    def _quant_nhd(
        self,
        nhd: torch.Tensor,
    ) -> tuple[tuple[torch.Tensor, torch.Tensor, torch.Tensor], int, int]:
        T = int(nhd.size(0))
        if T == 0:
            raise ValueError("empty K/V chunk in KIVI store")

        T_pad = cdiv(T, self._align) * self._align
        if T < T_pad:
            pad = nhd.new_zeros((T_pad - T,) + nhd.shape[1:])
            nhd = torch.cat((nhd, pad), dim=0)
        elif T > T_pad:
            nhd = nhd[:T_pad]

        t4 = self._nhd_to_bhdt(nhd.to(self._kivi_io_dtype))
        code, scale, mn = triton_quantize_and_pack_along_last_dim(
            t4,
            self._group_size,
            self._bits,
        )
        return (code, scale, mn), T, T_pad

    @staticmethod
    def _dequant_bhdn(
        code4: torch.Tensor,
        sc: torch.Tensor,
        mn: torch.Tensor,
        group_size: int,
        bits: int,
        as_dtype: torch.dtype,
    ) -> torch.Tensor:
        out = unpack_and_dequant_cache_triton(
            code4,
            sc,
            mn,
            group_size,
            bits,
            dtype=as_dtype,
        )
        return out.squeeze(0)  # [H, D, T]

    def _dequant_nhd(
        self,
        code: torch.Tensor,
        sc: torch.Tensor,
        mn: torch.Tensor,
        attn_start: int,
        local_end: int,
    ) -> torch.Tensor:
        H, D = self._num_heads, self._head_dim
        if local_end <= attn_start:
            return torch.empty(0, H, D, device=self._device, dtype=self._dtype)

        m = self._align
        t0 = (int(attn_start) // m) * m
        t1 = min(cdiv(max(int(local_end), 0), m) * m, self._N_alloc)
        if t1 <= t0:
            return torch.empty(0, H, D, device=self._device, dtype=self._dtype)

        fe, G = self._feats, self._group_size
        p0, p1 = t0 // fe, t1 // fe
        g0, g1 = t0 // G, t1 // G

        c4 = code[:, :, p0:p1].unsqueeze(0)
        out = self._dequant_bhdn(
            c4,
            sc[:, :, g0:g1].unsqueeze(0),
            mn[:, :, g0:g1].unsqueeze(0),
            self._group_size,
            self._bits,
            self._dtype,
        )
        nhd = out.permute(2, 0, 1)
        o0 = max(int(attn_start), t0) - t0
        o1 = o0 + (int(local_end) - max(int(attn_start), t0))
        return nhd[o0:o1].contiguous()

    # ------------------------------------------------------------------
    # Ring metadata. Step subclass overrides _ring_index.
    # ------------------------------------------------------------------
    def _ring_shape(self):
        return (self._num_layers,)

    def _ring_index(self, layer_id: int):
        return int(layer_id)

    def _init_ring_metadata(self) -> None:
        # These values are only consumed by Python control flow.  Keeping them
        # on CUDA makes every ``.item()`` a device synchronization, which is
        # very expensive in the offload path.
        shape = self._ring_shape()
        self._ring_active = torch.zeros(shape, dtype=torch.bool, device="cpu")
        self._ring_sink_len = torch.zeros(shape, dtype=torch.long, device="cpu")
        self._ring_recent_head = torch.zeros(shape, dtype=torch.long, device="cpu")
        self._ring_recent_len = torch.zeros(shape, dtype=torch.long, device="cpu")

    def _ring_is_active(self, layer_id: int) -> bool:
        return bool(self._ring_active[self._ring_index(layer_id)].item())

    def _set_ring_active(self, layer_id: int, value: bool) -> None:
        self._ring_active[self._ring_index(layer_id)] = bool(value)

    def _get_sink_len(self, layer_id: int) -> int:
        return int(self._ring_sink_len[self._ring_index(layer_id)].item())

    def _set_sink_len(self, layer_id: int, value: int) -> None:
        self._ring_sink_len[self._ring_index(layer_id)] = int(value)

    def _get_recent_head(self, layer_id: int) -> int:
        return int(self._ring_recent_head[self._ring_index(layer_id)].item())

    def _set_recent_head(self, layer_id: int, value: int) -> None:
        self._ring_recent_head[self._ring_index(layer_id)] = int(value)

    def _get_recent_len(self, layer_id: int) -> int:
        return int(self._ring_recent_len[self._ring_index(layer_id)].item())

    def _set_recent_len(self, layer_id: int, value: int) -> None:
        self._ring_recent_len[self._ring_index(layer_id)] = int(value)

    def _recent_capacity(self, layer_id: int) -> int:
        cap = self._N_alloc - self._get_sink_len(layer_id)
        if cap <= 0:
            raise RuntimeError(f"invalid KIVI recent capacity: N_alloc={self._N_alloc}, sink={self._get_sink_len(layer_id)}")
        return cap

    # ------------------------------------------------------------------
    # GPU compressed buffer accessors. Step subclass overrides.
    # ------------------------------------------------------------------
    def _kivi_k_code(self, layer_id: int) -> torch.Tensor:
        if self._kv_offload:
            return self._k_code_gpu
        return self._k_code[layer_id]

    def _kivi_v_code(self, layer_id: int) -> torch.Tensor:
        if self._kv_offload:
            return self._v_code_gpu
        return self._v_code[layer_id]

    def _kivi_k_scale(self, layer_id: int) -> torch.Tensor:
        if self._kv_offload:
            return self._k_scale_gpu
        return self._k_scale[layer_id]

    def _kivi_k_mn(self, layer_id: int) -> torch.Tensor:
        if self._kv_offload:
            return self._k_mn_gpu
        return self._k_mn[layer_id]

    def _kivi_v_scale(self, layer_id: int) -> torch.Tensor:
        if self._kv_offload:
            return self._v_scale_gpu
        return self._v_scale[layer_id]

    def _kivi_v_mn(self, layer_id: int) -> torch.Tensor:
        if self._kv_offload:
            return self._v_mn_gpu
        return self._v_mn[layer_id]

    # ------------------------------------------------------------------
    # CPU compressed buffer accessors for offload. Step subclass overrides.
    # ------------------------------------------------------------------
    def _cpu_k_code(self, layer_id: int) -> torch.Tensor:
        return self._k_code_cpu[layer_id]

    def _cpu_v_code(self, layer_id: int) -> torch.Tensor:
        return self._v_code_cpu[layer_id]

    def _cpu_k_scale(self, layer_id: int) -> torch.Tensor:
        return self._k_scale_cpu[layer_id]

    def _cpu_k_mn(self, layer_id: int) -> torch.Tensor:
        return self._k_mn_cpu[layer_id]

    def _cpu_v_scale(self, layer_id: int) -> torch.Tensor:
        return self._v_scale_cpu[layer_id]

    def _cpu_v_mn(self, layer_id: int) -> torch.Tensor:
        return self._v_mn_cpu[layer_id]

    # ------------------------------------------------------------------
    # Init / reset
    # ------------------------------------------------------------------
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
        # End pointers are Python-side bookkeeping; keep them on CPU to avoid
        # CUDA synchronizations from ``.item()`` in get_* methods.
        self._global_end = torch.zeros(L, dtype=torch.long, device="cpu")
        self._local_end = torch.zeros(L, dtype=torch.long, device="cpu")
        self._init_ring_metadata()

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

        # CPU keeps the authoritative compressed physical ring layout.
        self._k_code_cpu = torch.zeros(L, H, D, n_packs, dtype=torch.int32, device="cpu").pin_memory()
        self._v_code_cpu = torch.zeros(L, H, D, n_packs, dtype=torch.int32, device="cpu").pin_memory()
        self._k_scale_cpu = torch.zeros(L, H, D, n_groups, dtype=torch.float32, device="cpu").pin_memory()
        self._k_mn_cpu = torch.zeros(L, H, D, n_groups, dtype=torch.float32, device="cpu").pin_memory()
        self._v_scale_cpu = torch.zeros(L, H, D, n_groups, dtype=torch.float32, device="cpu").pin_memory()
        self._v_mn_cpu = torch.zeros(L, H, D, n_groups, dtype=torch.float32, device="cpu").pin_memory()

        self._k_code_gpu = torch.zeros(H, D, n_packs, dtype=torch.int32, device=d)
        self._v_code_gpu = torch.zeros(H, D, n_packs, dtype=torch.int32, device=d)
        self._k_scale_gpu = torch.zeros(H, D, n_groups, dtype=torch.float32, device=d)
        self._k_mn_gpu = torch.zeros(H, D, n_groups, dtype=torch.float32, device=d)
        self._v_scale_gpu = torch.zeros(H, D, n_groups, dtype=torch.float32, device=d)
        self._v_mn_gpu = torch.zeros(H, D, n_groups, dtype=torch.float32, device=d)

        self._global_end = torch.zeros(L, dtype=torch.long, device="cpu")
        self._local_end = torch.zeros(L, dtype=torch.long, device="cpu")
        self._init_ring_metadata()

        self._init_offload_state((L,))
        gpu_mb = (self._k_code_gpu.nbytes + self._v_code_gpu.nbytes + self._k_scale_gpu.nbytes + self._k_mn_gpu.nbytes + self._v_scale_gpu.nbytes + self._v_mn_gpu.nbytes) / (1024 * 1024)
        cpu_mb = (self._k_code_cpu.nbytes + self._v_code_cpu.nbytes + self._k_scale_cpu.nbytes + self._k_mn_cpu.nbytes + self._v_scale_cpu.nbytes + self._v_mn_cpu.nbytes) / (1024 * 1024)
        logger.info(
            "[KIVIQuantRollingKVCachePool+ring+offload] GPU compressed staging layer: {:.1f} MB, CPU pinned: {:.1f} MB",
            gpu_mb,
            cpu_mb,
        )

    def reset(self) -> None:
        if self._kv_offload:
            self.sync_all()
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
            self._init_ring_metadata()
            self._reset_offload_state()
            return

        self._k_code.zero_()
        self._v_code.zero_()
        self._k_scale.zero_()
        self._k_mn.zero_()
        self._v_scale.zero_()
        self._v_mn.zero_()
        self._global_end.zero_()
        self._local_end.zero_()
        self._init_ring_metadata()

    # ------------------------------------------------------------------
    # Compressed physical range copy helpers for offload.
    # ------------------------------------------------------------------
    def _copy_cpu_to_gpu_physical(self, layer_id: int, start: int, end: int) -> None:
        if end <= start:
            return
        fe, G = self._feats, self._group_size
        p0, p1 = start // fe, end // fe
        g0, g1 = start // G, end // G
        self._k_code_gpu[:, :, p0:p1].copy_(self._cpu_k_code(layer_id)[:, :, p0:p1], non_blocking=True)
        self._v_code_gpu[:, :, p0:p1].copy_(self._cpu_v_code(layer_id)[:, :, p0:p1], non_blocking=True)
        self._k_scale_gpu[:, :, g0:g1].copy_(self._cpu_k_scale(layer_id)[:, :, g0:g1], non_blocking=True)
        self._k_mn_gpu[:, :, g0:g1].copy_(self._cpu_k_mn(layer_id)[:, :, g0:g1], non_blocking=True)
        self._v_scale_gpu[:, :, g0:g1].copy_(self._cpu_v_scale(layer_id)[:, :, g0:g1], non_blocking=True)
        self._v_mn_gpu[:, :, g0:g1].copy_(self._cpu_v_mn(layer_id)[:, :, g0:g1], non_blocking=True)

    def _copy_layer_to_gpu(self, layer_id: int) -> None:
        self._copy_cpu_to_gpu_physical(layer_id, 0, self._N_alloc)

    # ------------------------------------------------------------------
    # Physical KIVI store. It preserves prefix/suffix inside touched aligned blocks.
    # ------------------------------------------------------------------
    def _write_k_segment(self, code: torch.Tensor, sc: torch.Tensor, mn: torch.Tensor, layer_id: int, physical_start: int) -> None:
        H, D = self._num_heads, self._head_dim
        b, h, d, n_pack = code.shape
        assert b == 1 and h == H and d == D
        fe, G = self._feats, self._group_size
        t_pad = n_pack * fe
        g_cnt = t_pad // G
        p0 = physical_start // fe
        p1 = p0 + n_pack
        g0 = physical_start // G
        g1 = g0 + g_cnt
        if physical_start + t_pad > self._N_alloc:
            raise RuntimeError("KIVI store overflow")
        self._kivi_k_code(layer_id)[:, :, p0:p1] = code[0]
        self._kivi_k_scale(layer_id)[:, :, g0:g1] = sc[0, :, :, :g_cnt]
        self._kivi_k_mn(layer_id)[:, :, g0:g1] = mn[0, :, :, :g_cnt]
        if self._kv_offload:
            self._cpu_k_code(layer_id)[:, :, p0:p1].copy_(code[0], non_blocking=True)
            self._cpu_k_scale(layer_id)[:, :, g0:g1].copy_(sc[0, :, :, :g_cnt], non_blocking=True)
            self._cpu_k_mn(layer_id)[:, :, g0:g1].copy_(mn[0, :, :, :g_cnt], non_blocking=True)

    def _write_v_segment(self, code: torch.Tensor, sc: torch.Tensor, mn: torch.Tensor, layer_id: int, physical_start: int) -> None:
        H, D = self._num_heads, self._head_dim
        b, h, d, n_pack = code.shape
        assert b == 1 and h == H and d == D
        fe, G = self._feats, self._group_size
        t_pad = n_pack * fe
        g_cnt = t_pad // G
        p0 = physical_start // fe
        p1 = p0 + n_pack
        g0 = physical_start // G
        g1 = g0 + g_cnt
        if physical_start + t_pad > self._N_alloc:
            raise RuntimeError("KIVI store overflow")
        self._kivi_v_code(layer_id)[:, :, p0:p1] = code[0]
        self._kivi_v_scale(layer_id)[:, :, g0:g1] = sc[0, :, :, :g_cnt]
        self._kivi_v_mn(layer_id)[:, :, g0:g1] = mn[0, :, :, :g_cnt]
        if self._kv_offload:
            self._cpu_v_code(layer_id)[:, :, p0:p1].copy_(code[0], non_blocking=True)
            self._cpu_v_scale(layer_id)[:, :, g0:g1].copy_(sc[0, :, :, :g_cnt], non_blocking=True)
            self._cpu_v_mn(layer_id)[:, :, g0:g1].copy_(mn[0, :, :, :g_cnt], non_blocking=True)

    def _physical_store_kv(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        physical_start: int,
        physical_end: int,
        layer_id: int,
    ) -> None:
        length = int(physical_end) - int(physical_start)
        if length <= 0:
            return
        if k.size(0) != length or v.size(0) != length:
            raise ValueError(f"KIVI physical store length mismatch: length={length}, k={k.size(0)}, v={v.size(0)}")

        A = self._align
        s0 = (int(physical_start) // A) * A
        e1 = min(cdiv(int(physical_end), A) * A, self._N_alloc)

        self._check_layer_loaded(layer_id)

        parts_k: list[torch.Tensor] = []
        parts_v: list[torch.Tensor] = []

        if s0 < physical_start:
            parts_k.append(self._dequant_nhd(self._kivi_k_code(layer_id), self._kivi_k_scale(layer_id), self._kivi_k_mn(layer_id), s0, physical_start))
            parts_v.append(self._dequant_nhd(self._kivi_v_code(layer_id), self._kivi_v_scale(layer_id), self._kivi_v_mn(layer_id), s0, physical_start))

        parts_k.append(k.contiguous())
        parts_v.append(v.contiguous())

        if physical_end < e1:
            parts_k.append(self._dequant_nhd(self._kivi_k_code(layer_id), self._kivi_k_scale(layer_id), self._kivi_k_mn(layer_id), physical_end, e1))
            parts_v.append(self._dequant_nhd(self._kivi_v_code(layer_id), self._kivi_v_scale(layer_id), self._kivi_v_mn(layer_id), physical_end, e1))

        k_cat = torch.cat(parts_k, dim=0)
        v_cat = torch.cat(parts_v, dim=0)
        expected = e1 - s0
        if k_cat.size(0) != expected or v_cat.size(0) != expected:
            raise RuntimeError(f"KIVI physical store preserved block length mismatch: expected={expected}, got k={k_cat.size(0)}, v={v_cat.size(0)}")

        (k_code, k_sc, k_mn), _, t_pad_k = self._quant_nhd(k_cat)
        (v_code, v_sc, v_mn), _, t_pad_v = self._quant_nhd(v_cat)
        if t_pad_k != expected or t_pad_v != expected:
            raise RuntimeError(f"KIVI physical store padding mismatch: expected={expected}, k_pad={t_pad_k}, v_pad={t_pad_v}")

        self._write_k_segment(k_code, k_sc, k_mn, layer_id, s0)
        self._write_v_segment(v_code, v_sc, v_mn, layer_id, s0)

    # ------------------------------------------------------------------
    # Ring mapping
    # ------------------------------------------------------------------
    def _recent_logical_to_physical_chunks(self, layer_id: int, recent_offset: int, length: int) -> list[tuple[int, int]]:
        if length <= 0:
            return []
        sink_len = self._get_sink_len(layer_id)
        cap = self._recent_capacity(layer_id)
        head = self._get_recent_head(layer_id)
        pos = (head + int(recent_offset)) % cap
        first = min(int(length), cap - pos)
        chunks = [(sink_len + pos, sink_len + pos + first)]
        remain = int(length) - first
        if remain > 0:
            chunks.append((sink_len, sink_len + remain))
        return chunks

    def _cache_chunks_for_logical_range(
        self,
        layer_id: int,
        attn_start: int,
        local_end: int,
        *,
        strict: bool = True,
    ) -> list[tuple[int, int]]:
        attn_start = int(attn_start)
        local_end = int(local_end)
        if local_end <= attn_start:
            return []

        if not self._ring_is_active(layer_id):
            if strict and local_end > self.get_local_end(layer_id):
                raise RuntimeError(f"KIVI non-ring read exceeds local_end: read_end={local_end}, local_end={self.get_local_end(layer_id)}")
            return [(attn_start, local_end)]

        sink_len = self._get_sink_len(layer_id)
        recent_len = self._get_recent_len(layer_id)
        logical_end = sink_len + recent_len
        if strict and local_end > logical_end:
            raise RuntimeError(f"KIVI ring read exceeds logical cache end: read_end={local_end}, logical_end={logical_end}, sink={sink_len}, recent_len={recent_len}")

        # For non-strict prefetch, clamp to existing logical tokens.
        if not strict:
            local_end = min(local_end, logical_end)
            if local_end <= attn_start:
                return []

        chunks: list[tuple[int, int]] = []
        s0 = attn_start
        s1 = min(local_end, sink_len)
        if s1 > s0:
            chunks.append((s0, s1))

        r0 = max(attn_start, sink_len)
        r1 = local_end
        if r1 > r0:
            chunks.extend(self._recent_logical_to_physical_chunks(layer_id, recent_offset=r0 - sink_len, length=r1 - r0))
        return chunks

    def _dequant_physical_chunks(self, layer_id: int, chunks: list[tuple[int, int]], which: str) -> torch.Tensor:
        H, D = self._num_heads, self._head_dim
        if not chunks:
            return torch.empty(0, H, D, device=self._device, dtype=self._dtype)

        if which == "k":
            code, scale, mn = self._kivi_k_code(layer_id), self._kivi_k_scale(layer_id), self._kivi_k_mn(layer_id)
        elif which == "v":
            code, scale, mn = self._kivi_v_code(layer_id), self._kivi_v_scale(layer_id), self._kivi_v_mn(layer_id)
        else:
            raise ValueError(f"invalid KIVI chunk kind: {which}")
        return kivi_dequant_ring_nhd_triton(code, scale, mn, chunks, self._group_size, self._bits, dtype=self._dtype)

    def _update_recent_len_after_store(self, layer_id: int, end_idx: int) -> None:
        if not self._ring_is_active(layer_id):
            return
        sink_len = self._get_sink_len(layer_id)
        if end_idx > sink_len:
            new_len = max(self._get_recent_len(layer_id), int(end_idx) - sink_len)
            cap = self._recent_capacity(layer_id)
            if new_len > cap:
                raise RuntimeError(f"KIVI recent ring overflow: len={new_len}, cap={cap}")
            self._set_recent_len(layer_id, new_len)

    # ------------------------------------------------------------------
    # Public store/read API
    # ------------------------------------------------------------------
    def store_kv(self, k: torch.Tensor, v: torch.Tensor, start_idx: int, end_idx: int, layer_id: int) -> None:
        start_idx = int(start_idx)
        end_idx = int(end_idx)
        length = end_idx - start_idx
        if length <= 0:
            return
        if k.size(0) != length or v.size(0) != length:
            raise ValueError(f"KIVI store length mismatch: length={length}, k={k.size(0)}, v={v.size(0)}")

        if not self._ring_is_active(layer_id):
            self._physical_store_kv(k, v, start_idx, end_idx, layer_id)
            if self._kv_offload:
                self._record_cpu_update(layer_id)
            return

        sink_len = self._get_sink_len(layer_id)
        remaining_k = k
        remaining_v = v
        logical_start = start_idx
        remaining = length

        if logical_start < sink_len:
            first = min(remaining, sink_len - logical_start)
            self._physical_store_kv(remaining_k[:first], remaining_v[:first], logical_start, logical_start + first, layer_id)
            logical_start += first
            remaining_k = remaining_k[first:]
            remaining_v = remaining_v[first:]
            remaining -= first
            if remaining == 0:
                self._update_recent_len_after_store(layer_id, end_idx)
                if self._kv_offload:
                    self._record_cpu_update(layer_id)
                return

        recent_offset = logical_start - sink_len
        chunks = self._recent_logical_to_physical_chunks(layer_id, recent_offset, remaining)
        off = 0
        for p0, p1 in chunks:
            n = p1 - p0
            self._physical_store_kv(remaining_k[off : off + n], remaining_v[off : off + n], p0, p1, layer_id)
            off += n

        self._update_recent_len_after_store(layer_id, end_idx)
        if self._kv_offload:
            self._record_cpu_update(layer_id)

    def k_cache(self, layer_id: int, attn_start: int, local_end: int) -> torch.Tensor:
        attn_start, local_end = int(attn_start), int(local_end)
        self._check_layer_loaded(layer_id)
        chunks = self._cache_chunks_for_logical_range(layer_id, attn_start, local_end, strict=True)
        out = self._dequant_physical_chunks(layer_id, chunks, "k")
        if self._dtype in (torch.bfloat16, torch.float32) and out.dtype != self._dtype:
            return out.to(self._dtype)
        return out

    def v_cache(self, layer_id: int, attn_start: int, local_end: int) -> torch.Tensor:
        attn_start, local_end = int(attn_start), int(local_end)
        self._check_layer_loaded(layer_id)
        chunks = self._cache_chunks_for_logical_range(layer_id, attn_start, local_end, strict=True)
        out = self._dequant_physical_chunks(layer_id, chunks, "v")
        if self._dtype in (torch.bfloat16, torch.float32) and out.dtype != self._dtype:
            return out.to(self._dtype)
        return out

    def roll_window(self, layer_id: int, sink_tokens: int, num_evicted: int) -> None:
        old_local_end = self.get_local_end(layer_id)
        sink_tokens = int(sink_tokens)
        num_evicted = int(num_evicted)
        if num_evicted <= 0:
            return
        num_kept = old_local_end - num_evicted - sink_tokens
        if num_kept <= 0:
            self._set_ring_active(layer_id, True)
            self._set_sink_len(layer_id, sink_tokens)
            self._set_recent_head(layer_id, 0)
            self._set_recent_len(layer_id, 0)
            return

        if sink_tokens < 0 or sink_tokens >= self._N_alloc:
            raise RuntimeError(f"invalid sink_tokens={sink_tokens} for N_alloc={self._N_alloc}")

        if not self._ring_is_active(layer_id):
            self._set_ring_active(layer_id, True)
            self._set_sink_len(layer_id, sink_tokens)
            cap = self._recent_capacity(layer_id)
            if num_kept > cap:
                raise RuntimeError(f"KIVI ring kept tokens {num_kept} exceed recent capacity {cap}")
            self._set_recent_head(layer_id, num_evicted % cap)
            self._set_recent_len(layer_id, num_kept)
            return

        if sink_tokens != self._get_sink_len(layer_id):
            raise RuntimeError(f"KIVI ring sink size changed after activation: old={self._get_sink_len(layer_id)}, new={sink_tokens}")
        cap = self._recent_capacity(layer_id)
        recent_len = self._get_recent_len(layer_id)
        if num_evicted > recent_len:
            raise RuntimeError(f"KIVI ring evict exceeds recent length: evict={num_evicted}, recent_len={recent_len}")
        self._set_recent_head(layer_id, (self._get_recent_head(layer_id) + num_evicted) % cap)
        self._set_recent_len(layer_id, recent_len - num_evicted)


class StepKiviQuantRollingKVCachePool(KIVIQuantRollingKVCachePool):
    """Step-isolated KIVI ring-buffer KV cache with kv offload."""

    def __init__(
        self,
        num_steps: int,
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
        self.num_steps = int(num_steps)
        self._current_step = 0
        super().__init__(
            num_layers=num_layers,
            cache_size=cache_size,
            num_heads=num_heads,
            head_dim=head_dim,
            dtype=dtype,
            device=device,
            k_cache_type=k_cache_type,
            v_cache_type=v_cache_type,
            group_size=group_size,
            kv_offload=kv_offload,
        )

    @property
    def current_step(self) -> int:
        return self._current_step

    @current_step.setter
    def current_step(self, value: int) -> None:
        value = int(value)
        if value == self._current_step:
            return
        if self._kv_offload and hasattr(self, "_prefetch_stream"):
            self.sync_all()
            self._current_step = value
            self._reset_offload_state()
            return
        self._current_step = value

    def _step(self) -> int:
        return int(self._current_step)

    def _ring_shape(self):
        return (self.num_steps, self._num_layers)

    def _ring_index(self, layer_id: int):
        return (self._step(), int(layer_id))

    def _init_kv_buffer(self) -> None:
        if self._kv_offload:
            self._init_kv_buffer_offload()
            return

        T, L = self.num_steps, self._num_layers
        N = self._N_alloc
        H, D = self._num_heads, self._head_dim
        fe, G = self._feats, self._group_size
        n_packs = N // fe
        n_groups = N // G
        self._kivi_n_packs = n_packs
        self._kivi_n_groups = n_groups
        d = self._device

        self._k_code = torch.zeros(T, L, H, D, n_packs, dtype=torch.int32, device=d)
        self._v_code = torch.zeros(T, L, H, D, n_packs, dtype=torch.int32, device=d)
        self._k_scale = torch.zeros(T, L, H, D, n_groups, dtype=torch.float32, device=d)
        self._k_mn = torch.zeros(T, L, H, D, n_groups, dtype=torch.float32, device=d)
        self._v_scale = torch.zeros(T, L, H, D, n_groups, dtype=torch.float32, device=d)
        self._v_mn = torch.zeros(T, L, H, D, n_groups, dtype=torch.float32, device=d)
        self._global_end = torch.zeros(T, L, dtype=torch.long, device="cpu")
        self._local_end = torch.zeros(T, L, dtype=torch.long, device="cpu")
        self._init_ring_metadata()

    def _init_kv_buffer_offload(self) -> None:
        T, L = self.num_steps, self._num_layers
        N = self._N_alloc
        H, D = self._num_heads, self._head_dim
        fe, G = self._feats, self._group_size
        n_packs = N // fe
        n_groups = N // G
        self._kivi_n_packs = n_packs
        self._kivi_n_groups = n_groups
        d = self._device

        self._k_code_cpu = torch.zeros(T, L, H, D, n_packs, dtype=torch.int32, device="cpu").pin_memory()
        self._v_code_cpu = torch.zeros(T, L, H, D, n_packs, dtype=torch.int32, device="cpu").pin_memory()
        self._k_scale_cpu = torch.zeros(T, L, H, D, n_groups, dtype=torch.float32, device="cpu").pin_memory()
        self._k_mn_cpu = torch.zeros(T, L, H, D, n_groups, dtype=torch.float32, device="cpu").pin_memory()
        self._v_scale_cpu = torch.zeros(T, L, H, D, n_groups, dtype=torch.float32, device="cpu").pin_memory()
        self._v_mn_cpu = torch.zeros(T, L, H, D, n_groups, dtype=torch.float32, device="cpu").pin_memory()

        self._k_code_gpu = torch.zeros(H, D, n_packs, dtype=torch.int32, device=d)
        self._v_code_gpu = torch.zeros(H, D, n_packs, dtype=torch.int32, device=d)
        self._k_scale_gpu = torch.zeros(H, D, n_groups, dtype=torch.float32, device=d)
        self._k_mn_gpu = torch.zeros(H, D, n_groups, dtype=torch.float32, device=d)
        self._v_scale_gpu = torch.zeros(H, D, n_groups, dtype=torch.float32, device=d)
        self._v_mn_gpu = torch.zeros(H, D, n_groups, dtype=torch.float32, device=d)

        self._global_end = torch.zeros(T, L, dtype=torch.long, device="cpu")
        self._local_end = torch.zeros(T, L, dtype=torch.long, device="cpu")
        self._init_ring_metadata()
        self._init_offload_state((T, L))

        gpu_mb = (self._k_code_gpu.nbytes + self._v_code_gpu.nbytes + self._k_scale_gpu.nbytes + self._k_mn_gpu.nbytes + self._v_scale_gpu.nbytes + self._v_mn_gpu.nbytes) / (1024 * 1024)
        cpu_mb = (self._k_code_cpu.nbytes + self._v_code_cpu.nbytes + self._k_scale_cpu.nbytes + self._k_mn_cpu.nbytes + self._v_scale_cpu.nbytes + self._v_mn_cpu.nbytes) / (1024 * 1024)
        logger.info(
            "[StepKiviQuantRollingKVCachePool+ring+offload] steps={}, GPU compressed staging layer: {:.1f} MB, CPU pinned: {:.1f} MB",
            self.num_steps,
            gpu_mb,
            cpu_mb,
        )

    def _kivi_k_code(self, layer_id: int) -> torch.Tensor:
        if self._kv_offload:
            return self._k_code_gpu
        return self._k_code[self._step(), layer_id]

    def _kivi_v_code(self, layer_id: int) -> torch.Tensor:
        if self._kv_offload:
            return self._v_code_gpu
        return self._v_code[self._step(), layer_id]

    def _kivi_k_scale(self, layer_id: int) -> torch.Tensor:
        if self._kv_offload:
            return self._k_scale_gpu
        return self._k_scale[self._step(), layer_id]

    def _kivi_k_mn(self, layer_id: int) -> torch.Tensor:
        if self._kv_offload:
            return self._k_mn_gpu
        return self._k_mn[self._step(), layer_id]

    def _kivi_v_scale(self, layer_id: int) -> torch.Tensor:
        if self._kv_offload:
            return self._v_scale_gpu
        return self._v_scale[self._step(), layer_id]

    def _kivi_v_mn(self, layer_id: int) -> torch.Tensor:
        if self._kv_offload:
            return self._v_mn_gpu
        return self._v_mn[self._step(), layer_id]

    def _cpu_k_code(self, layer_id: int) -> torch.Tensor:
        return self._k_code_cpu[self._step(), layer_id]

    def _cpu_v_code(self, layer_id: int) -> torch.Tensor:
        return self._v_code_cpu[self._step(), layer_id]

    def _cpu_k_scale(self, layer_id: int) -> torch.Tensor:
        return self._k_scale_cpu[self._step(), layer_id]

    def _cpu_k_mn(self, layer_id: int) -> torch.Tensor:
        return self._k_mn_cpu[self._step(), layer_id]

    def _cpu_v_scale(self, layer_id: int) -> torch.Tensor:
        return self._v_scale_cpu[self._step(), layer_id]

    def _cpu_v_mn(self, layer_id: int) -> torch.Tensor:
        return self._v_mn_cpu[self._step(), layer_id]

    def _offload_index(self, layer_id: int) -> tuple[int, ...]:
        return (self._step(), int(layer_id))

    def get_global_end(self, layer_id: int) -> int:
        return int(self._global_end[self._step(), layer_id].item())

    def get_local_end(self, layer_id: int) -> int:
        return int(self._local_end[self._step(), layer_id].item())

    def set_ends(self, layer_id: int, global_end: int, local_end: int) -> None:
        self._global_end[self._step(), layer_id] = int(global_end)
        self._local_end[self._step(), layer_id] = int(local_end)
