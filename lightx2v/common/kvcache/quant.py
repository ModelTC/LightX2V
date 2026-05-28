import os

import torch
import torch.distributed as dist
from loguru import logger

from .kernel import *
from .rolling import RollingKVCachePool
from .utils import *

try:
    from fouroversix import quantize_to_fp4
    from fouroversix.quantize.quantized_tensor import QuantizedTensor
except ImportError:
    QuantizedTensor = None
    quantize_to_fp4 = None


# =============================================================================
# Generic token-ring helper for quantized rolling caches.
#
# Logical layout exposed to callers:
#   [sink logical tokens][recent logical tokens]
#
# Physical layout in cache tensors:
#   [sink fixed region][recent ring region]
#
# `roll_window()` is O(1): it updates head/len metadata and never moves the
# kept window.  `k_cache()` / `v_cache()` materialize a contiguous logical range
# by reading one or more physical chunks and concatenating them.
# =============================================================================


class _QuantTokenRingMixin:
    def _ring_index(self, layer_id: int):
        return int(layer_id)

    def _init_ring_metadata(self, *shape: int) -> None:
        self._ring_active = torch.zeros(*shape, dtype=torch.bool, device="cpu")
        self._ring_sink_len = torch.zeros(*shape, dtype=torch.long, device="cpu")
        self._ring_recent_head = torch.zeros(*shape, dtype=torch.long, device="cpu")
        self._ring_recent_len = torch.zeros(*shape, dtype=torch.long, device="cpu")

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

    def _zero_tensors(self, names: list[str]) -> None:
        for name in names:
            getattr(self, name).zero_()

    def _reset_ring(self) -> None:
        self._zero_tensors(["_ring_active", "_ring_sink_len", "_ring_recent_head", "_ring_recent_len"])

    def _reset_ends(self) -> None:
        self._zero_tensors(["_global_end", "_local_end"])

    def _recent_capacity(self, layer_id: int) -> int:
        return int(self._cache_size) - self._get_sink_len(layer_id)

    def _recent_offset_to_physical_chunks(
        self,
        layer_id: int,
        recent_offset: int,
        length: int,
    ) -> list[tuple[int, int]]:
        if length <= 0:
            return []
        sink_len = self._get_sink_len(layer_id)
        cap = self._recent_capacity(layer_id)
        if cap <= 0:
            raise RuntimeError("ring cache has no recent capacity")
        head = self._get_recent_head(layer_id)
        pos = (head + int(recent_offset)) % cap
        first = min(int(length), cap - pos)
        out = [(sink_len + pos, sink_len + pos + first)]
        remain = int(length) - first
        if remain > 0:
            out.append((sink_len, sink_len + remain))
        return out

    def _logical_to_physical_chunks(
        self,
        layer_id: int,
        start: int,
        end: int,
    ) -> list[tuple[int, int]]:
        start = int(start)
        end = int(end)
        if end <= start:
            return []
        if not self._ring_is_active(layer_id):
            return [(start, end)]

        sink_len = self._get_sink_len(layer_id)
        recent_len = self._get_recent_len(layer_id)
        logical_end = sink_len + recent_len
        if end > logical_end:
            raise RuntimeError(f"ring read exceeds logical end: read=[{start},{end}), logical_end={logical_end}, sink={sink_len}, recent_len={recent_len}")

        chunks: list[tuple[int, int]] = []
        # Fixed sink: logical == physical.
        s0, s1 = start, min(end, sink_len)
        if s1 > s0:
            chunks.append((s0, s1))

        # Recent ring.
        r0, r1 = max(start, sink_len), end
        if r1 > r0:
            chunks.extend(
                self._recent_offset_to_physical_chunks(
                    layer_id,
                    recent_offset=r0 - sink_len,
                    length=r1 - r0,
                )
            )
        return chunks

    def _logical_store_chunks(
        self,
        layer_id: int,
        start: int,
        end: int,
    ) -> list[tuple[int, int]]:
        # Store range uses the same mapping as read range.  If it extends the
        # recent length, update recent metadata after physical stores.
        return self._logical_to_physical_chunks_for_store(layer_id, start, end)

    def _logical_to_physical_chunks_for_store(
        self,
        layer_id: int,
        start: int,
        end: int,
    ) -> list[tuple[int, int]]:
        start = int(start)
        end = int(end)
        if end <= start:
            return []
        if not self._ring_is_active(layer_id):
            return [(start, end)]

        sink_len = self._get_sink_len(layer_id)
        chunks: list[tuple[int, int]] = []

        if start < sink_len:
            s1 = min(end, sink_len)
            chunks.append((start, s1))
            if s1 == end:
                return chunks
            start = sink_len

        recent_offset = start - sink_len
        length = end - start
        chunks.extend(self._recent_offset_to_physical_chunks(layer_id, recent_offset, length))
        self._set_recent_len(layer_id, max(self._get_recent_len(layer_id), recent_offset + length))
        return chunks

    def roll_window(self, layer_id: int, sink_tokens: int, num_evicted: int) -> None:
        old_local_end = self.get_local_end(layer_id)
        sink_tokens = int(sink_tokens)
        num_evicted = int(num_evicted)
        num_kept = old_local_end - num_evicted - sink_tokens
        if num_kept <= 0:
            self._set_ring_active(layer_id, True)
            self._set_sink_len(layer_id, sink_tokens)
            self._set_recent_head(layer_id, 0)
            self._set_recent_len(layer_id, 0)
            return

        if not self._ring_is_active(layer_id):
            self._set_ring_active(layer_id, True)
            self._set_sink_len(layer_id, sink_tokens)
            cap = self._recent_capacity(layer_id)
            if num_kept > cap:
                raise RuntimeError(f"ring kept tokens {num_kept} exceed recent capacity {cap}")
            # Before first roll physical layout is contiguous:
            # [sink][evicted][kept].  Recent ring starts after sink, so the
            # kept range physical offset inside recent ring is num_evicted.
            self._set_recent_head(layer_id, num_evicted % cap)
            self._set_recent_len(layer_id, num_kept)
            return

        if sink_tokens != self._get_sink_len(layer_id):
            raise RuntimeError(f"ring sink size changed: old={self._get_sink_len(layer_id)}, new={sink_tokens}")
        cap = self._recent_capacity(layer_id)
        recent_len = self._get_recent_len(layer_id)
        if num_evicted > recent_len:
            raise RuntimeError(f"ring evict exceeds recent length: evict={num_evicted}, recent_len={recent_len}")
        self._set_recent_head(layer_id, (self._get_recent_head(layer_id) + num_evicted) % cap)
        self._set_recent_len(layer_id, recent_len - num_evicted)


# =============================================================================
# SageQuant
# =============================================================================


class SageQuantRollingKVCachePool(_QuantTokenRingMixin, RollingKVCachePool):
    _BLKK = 128
    _SCALES_PER_BLK = 4
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
        assert k_cache_type in ["int8"]
        assert v_cache_type in ["fp8", "fp16"]
        self._k_cache_type = k_cache_type
        self._v_cache_type = v_cache_type
        self._calib_path = calib_path
        self.current_step: int = 0
        self._PERM_16 = torch.tensor(self._PERM_16_VAL, dtype=torch.long, device=device)
        self._INV_PERM_16 = torch.tensor(self._INV_PERM_16_VAL, dtype=torch.long, device=device)
        self._load_calib(device=device)
        super().__init__(num_layers, cache_size, num_heads, head_dim, dtype, device, kv_offload=kv_offload)

    def _init_kv_buffer(self) -> None:
        if self._kv_offload:
            self._init_kv_buffer_offload()
            return
        L, N, H, D = self._num_layers, self._cache_size, self._num_heads, self._head_dim
        self._k_buffer = torch.zeros(L, N, H, D, dtype=torch.int8, device=self._device)
        v_dtype = torch.float8_e4m3fn if self._v_cache_type == "fp8" else torch.float16
        self._v_buffer = torch.zeros(L, N, H, D, dtype=v_dtype, device=self._device)
        self._global_end = torch.zeros(L, dtype=torch.long, device="cpu")
        self._local_end = torch.zeros(L, dtype=torch.long, device="cpu")
        self._init_ring_metadata(L)

    def _init_kv_buffer_offload(self) -> None:
        L, N, H, D = self._num_layers, self._cache_size, self._num_heads, self._head_dim
        v_dtype = torch.float8_e4m3fn if self._v_cache_type == "fp8" else torch.float16
        self._k_cpu = torch.zeros(L, N, H, D, dtype=torch.int8, device="cpu").pin_memory()
        self._v_cpu = torch.zeros(L, N, H, D, dtype=v_dtype, device="cpu").pin_memory()
        self._k_gpu_buf = torch.zeros(N, H, D, dtype=torch.int8, device=self._device)
        self._v_gpu_buf = torch.zeros(N, H, D, dtype=v_dtype, device=self._device)
        self._global_end = torch.zeros(L, dtype=torch.long, device="cpu")
        self._local_end = torch.zeros(L, dtype=torch.long, device="cpu")
        self._init_ring_metadata(L)
        self._init_offload_state((L,))

    def _load_calib(self, device=torch.device("cuda")) -> None:
        load_path = self._calib_path
        if dist.is_available() and dist.is_initialized() and self._calib_path is not None:
            rank = dist.get_rank()
            rank_path = ranked_calib_path(self._calib_path, rank)
            if os.path.exists(rank_path):
                load_path = rank_path
        calib = torch.load(load_path, map_location=device, weights_only=True)
        self._calib_km = calib["km"].to(device=device, dtype=torch.float32)
        self._calib_v_scale = calib["v_scale"].to(device=device, dtype=torch.float32)
        self._calib_k_block_scale = calib["k_block_scale"].to(device=device, dtype=torch.float32)

    def _sage_k_storage(self, layer_id: int) -> torch.Tensor:
        if self._kv_offload:
            return self._k_cpu[layer_id]
        return self._k_buffer[layer_id]

    def _sage_v_storage(self, layer_id: int) -> torch.Tensor:
        if self._kv_offload:
            return self._v_cpu[layer_id]
        return self._v_buffer[layer_id]

    def _lookup_km(self, layer_id: int) -> torch.Tensor | None:
        km_cal = self._calib_km
        if km_cal.dim() == 5:
            return km_cal[self.current_step, layer_id].unsqueeze(0)
        return km_cal[layer_id].unsqueeze(0)

    def _lookup_v_scale(self, layer_id: int) -> torch.Tensor:
        vs_cal = self._calib_v_scale
        if vs_cal.dim() == 4:
            return vs_cal[self.current_step, layer_id]
        return vs_cal[layer_id]

    def _lookup_k_block_scale(self, layer_id: int, blk_start: int, num_blk: int) -> torch.Tensor:
        return self._calib_k_block_scale[self.current_step, layer_id, blk_start : blk_start + num_blk]

    def _quant_key(self, k_smoothed: torch.Tensor, preset_scale: torch.Tensor, start_idx: int, BLKK: int = 128) -> torch.Tensor:
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

    def _physical_store_kv(self, k: torch.Tensor, v: torch.Tensor, p0: int, p1: int, layer_id: int) -> None:
        km = self._lookup_km(layer_id)
        if km is not None:
            k_smoothed = k - km.to(k.dtype).squeeze(0)
        else:
            k_smoothed = k
        blk_start = p0 // self._BLKK
        last_blk = (p1 - 1) // self._BLKK
        num_blk = last_blk - blk_start + 1
        preset_scale = self._lookup_k_block_scale(layer_id, blk_start, num_blk)
        k_int8 = self._quant_key(k_smoothed, preset_scale, p0, self._BLKK)
        v_scale = self._lookup_v_scale(layer_id)
        v_fp8 = quant_value_per_channel_fp8_static_scale_kernel(v, v_scale, fp8_max=448.0)

        if self._kv_offload:
            self._check_layer_loaded(layer_id)
            self._k_gpu_buf[p0:p1].copy_(k_int8)
            self._v_gpu_buf[p0:p1].copy_(v_fp8)
            self._k_cpu[layer_id, p0:p1].copy_(k_int8, non_blocking=True)
            self._v_cpu[layer_id, p0:p1].copy_(v_fp8, non_blocking=True)
        else:
            self._k_buffer[layer_id, p0:p1] = k_int8
            self._v_buffer[layer_id, p0:p1] = v_fp8

    def store_kv(self, k: torch.Tensor, v: torch.Tensor, start_idx: int, end_idx: int, layer_id: int) -> None:
        length = int(end_idx) - int(start_idx)
        if length <= 0:
            return
        chunks = self._logical_to_physical_chunks_for_store(layer_id, start_idx, end_idx)
        off = 0
        for p0, p1 in chunks:
            n = p1 - p0
            self._physical_store_kv(k[off : off + n].contiguous(), v[off : off + n].contiguous(), p0, p1, layer_id)
            off += n
        if self._kv_offload:
            self._record_cpu_update(layer_id)

    def _copy_layer_to_gpu(self, layer_id: int) -> None:
        self._k_gpu_buf.copy_(self._k_cpu[layer_id], non_blocking=True)
        self._v_gpu_buf.copy_(self._v_cpu[layer_id], non_blocking=True)

    def _k_scale_for_physical_chunks(self, layer_id: int, chunks: list[tuple[int, int]]) -> torch.Tensor:
        scales = []
        for p0, p1 in chunks:
            a0 = (p0 // self._BLKK) * self._BLKK
            blk_s = a0 // self._BLKK
            blk_e = (p1 + self._BLKK - 1) // self._BLKK
            scales.append(self._calib_k_block_scale[self.current_step, layer_id, blk_s:blk_e])
        if len(scales) == 1:
            sc = scales[0]
        else:
            sc = torch.cat(scales, dim=0)
        return sc.permute(1, 0, 2).reshape(1, self._num_heads, -1).contiguous()

    def k_cache(self, layer_id: int, attn_start: int, local_end: int):
        chunks = self._logical_to_physical_chunks(layer_id, attn_start, local_end)
        if self._kv_offload:
            self._check_layer_loaded(layer_id)
            parts = [self._k_gpu_buf[p0:p1] for p0, p1 in chunks]
            k_int8 = (parts[0] if len(parts) == 1 else torch.cat(parts, dim=0)).unsqueeze(0).contiguous()
            k_scale = self._k_scale_for_physical_chunks(layer_id, chunks)
            return k_int8, k_scale

        parts = [self._sage_k_storage(layer_id)[p0:p1] for p0, p1 in chunks]
        k_int8 = (parts[0] if len(parts) == 1 else torch.cat(parts, dim=0)).unsqueeze(0).contiguous()
        k_scale = self._k_scale_for_physical_chunks(layer_id, chunks)
        return k_int8, k_scale

    def _transpose_permute_v(self, v: torch.Tensor) -> torch.Tensor:
        kv_len, H, D = v.shape
        padded_len = (kv_len + 127) // 128 * 128
        if padded_len > kv_len:
            v_t = v.new_zeros(D, H, padded_len)
            v_t[:, :, :kv_len].copy_(v.permute(2, 1, 0))
        else:
            v_t = v.permute(2, 1, 0).contiguous()
        v_t = v_t.view(D, H, -1, 16)[:, :, :, self._PERM_16].contiguous()
        return v_t.view(1, D, H, padded_len)

    def v_cache(self, layer_id: int, attn_start: int, local_end: int):
        chunks = self._logical_to_physical_chunks(layer_id, attn_start, local_end)
        if self._kv_offload:
            self._check_layer_loaded(layer_id)
            parts = [self._v_gpu_buf[p0:p1] for p0, p1 in chunks]
            v = parts[0] if len(parts) == 1 else torch.cat(parts, dim=0)
        else:
            parts = [self._sage_v_storage(layer_id)[p0:p1] for p0, p1 in chunks]
            v = parts[0] if len(parts) == 1 else torch.cat(parts, dim=0)
        return self._transpose_permute_v(v), self._lookup_v_scale(layer_id).unsqueeze(0).contiguous()

    def reset(self) -> None:
        if self._kv_offload:
            self.sync_all()
            self._zero_tensors(["_k_cpu", "_v_cpu", "_k_gpu_buf", "_v_gpu_buf"])
            self._reset_offload_state()
        else:
            self._zero_tensors(["_k_buffer", "_v_buffer"])
        self._reset_ends()
        self._reset_ring()


# =============================================================================
# TurboQuant
# =============================================================================


class TurboQuantRollingKVCachePool(_QuantTokenRingMixin, RollingKVCachePool):
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
        self._value_group_size = int(value_group_size)
        if self._key_bits < 2:
            raise ValueError("TurboQuantProd requires key_bits >= 2")
        if head_dim % self._value_group_size != 0:
            raise ValueError(f"head_dim {head_dim} must divide value_group_size {self._value_group_size}")

        device_t = torch.device(str(device))
        inf_dtype = torch.float32
        nk_bits = self._key_bits - 1
        cb_key = tq_fw_load_codebook_record(head_dim, nk_bits, codebook_dir, codebook_cache_dir, export_missing_codebooks)
        self._inf_nk = tq_fw_packed_width(head_dim, nk_bits)
        self._inf_nqjl = (head_dim + 7) // 8

        def _make_k_mod(seed_k: int) -> torch.nn.Module:
            return TurboQuantProdInference(head_dim, self._key_bits, device_t, seed_k, cb_key, dtype=inf_dtype)

        if self._per_layer_compressors:
            self._k_inference_modules = [_make_k_mod(self._seed_base + lid * 7) for lid in range(self._n_layers)]
        else:
            km = _make_k_mod(self._seed_base)
            self._k_inference_modules = [km for _ in range(self._n_layers)]

        self._inf_v_width = tq_value_group_packed_width(head_dim, self._value_bits)
        self._inf_v_n_groups = head_dim // self._value_group_size
        super().__init__(num_layers, cache_size, num_heads, head_dim, dtype, device, kv_offload=kv_offload)

    def _k_mod_inf(self, layer_id: int) -> torch.nn.Module:
        return self._k_inference_modules[layer_id]

    def _init_kv_buffer(self) -> None:
        if self._kv_offload:
            self._init_kv_buffer_offload()
            return
        L, N, H = self._num_layers, self._cache_size, self._num_heads
        ng = self._inf_v_n_groups
        d = self._device
        self._k_packed = torch.zeros(L, N, H, self._inf_nk, dtype=torch.uint8, device=d)
        self._k_norms = torch.zeros(L, N, H, dtype=torch.float16, device=d)
        self._k_qjl_packed = torch.zeros(L, N, H, self._inf_nqjl, dtype=torch.uint8, device=d)
        self._k_res_norms = torch.zeros(L, N, H, dtype=torch.float16, device=d)
        self._v_group_data = torch.zeros(L, N, H, self._inf_v_width, dtype=torch.uint8, device=d)
        self._v_group_scales = torch.zeros(L, N, H, ng, dtype=torch.float16, device=d)
        self._v_group_zeros = torch.zeros(L, N, H, ng, dtype=torch.float16, device=d)
        self._global_end = torch.zeros(L, dtype=torch.long, device="cpu")
        self._local_end = torch.zeros(L, dtype=torch.long, device="cpu")
        self._init_ring_metadata(L)

    def _init_kv_buffer_offload(self) -> None:
        L, N, H = self._num_layers, self._cache_size, self._num_heads
        ng = self._inf_v_n_groups
        self._k_packed_cpu = torch.zeros(L, N, H, self._inf_nk, dtype=torch.uint8, device="cpu").pin_memory()
        self._k_norms_cpu = torch.zeros(L, N, H, dtype=torch.float16, device="cpu").pin_memory()
        self._k_qjl_packed_cpu = torch.zeros(L, N, H, self._inf_nqjl, dtype=torch.uint8, device="cpu").pin_memory()
        self._k_res_norms_cpu = torch.zeros(L, N, H, dtype=torch.float16, device="cpu").pin_memory()
        self._v_group_data_cpu = torch.zeros(L, N, H, self._inf_v_width, dtype=torch.uint8, device="cpu").pin_memory()
        self._v_group_scales_cpu = torch.zeros(L, N, H, ng, dtype=torch.float16, device="cpu").pin_memory()
        self._v_group_zeros_cpu = torch.zeros(L, N, H, ng, dtype=torch.float16, device="cpu").pin_memory()
        d = self._device
        self._k_packed_gpu = torch.zeros(N, H, self._inf_nk, dtype=torch.uint8, device=d)
        self._k_norms_gpu = torch.zeros(N, H, dtype=torch.float16, device=d)
        self._k_qjl_packed_gpu = torch.zeros(N, H, self._inf_nqjl, dtype=torch.uint8, device=d)
        self._k_res_norms_gpu = torch.zeros(N, H, dtype=torch.float16, device=d)
        self._v_group_data_gpu = torch.zeros(N, H, self._inf_v_width, dtype=torch.uint8, device=d)
        self._v_group_scales_gpu = torch.zeros(N, H, ng, dtype=torch.float16, device=d)
        self._v_group_zeros_gpu = torch.zeros(N, H, ng, dtype=torch.float16, device=d)
        self._global_end = torch.zeros(L, dtype=torch.long, device="cpu")
        self._local_end = torch.zeros(L, dtype=torch.long, device="cpu")
        self._init_ring_metadata(L)
        self._init_offload_state((L,))

    def _store_arrays(self, layer_id: int, p0: int, p1: int, tensors: dict[str, torch.Tensor]) -> None:
        if self._kv_offload:
            self._check_layer_loaded(layer_id)
            self._k_packed_gpu[p0:p1].copy_(tensors["mse"])
            self._k_norms_gpu[p0:p1].copy_(tensors["norms"])
            self._k_qjl_packed_gpu[p0:p1].copy_(tensors["qjl"])
            self._k_res_norms_gpu[p0:p1].copy_(tensors["res"])
            self._v_group_data_gpu[p0:p1].copy_(tensors["v_data"])
            self._v_group_scales_gpu[p0:p1].copy_(tensors["v_scales"])
            self._v_group_zeros_gpu[p0:p1].copy_(tensors["v_zeros"])
            self._k_packed_cpu[layer_id, p0:p1].copy_(tensors["mse"], non_blocking=True)
            self._k_norms_cpu[layer_id, p0:p1].copy_(tensors["norms"], non_blocking=True)
            self._k_qjl_packed_cpu[layer_id, p0:p1].copy_(tensors["qjl"], non_blocking=True)
            self._k_res_norms_cpu[layer_id, p0:p1].copy_(tensors["res"], non_blocking=True)
            self._v_group_data_cpu[layer_id, p0:p1].copy_(tensors["v_data"], non_blocking=True)
            self._v_group_scales_cpu[layer_id, p0:p1].copy_(tensors["v_scales"], non_blocking=True)
            self._v_group_zeros_cpu[layer_id, p0:p1].copy_(tensors["v_zeros"], non_blocking=True)
        else:
            self._k_packed[layer_id, p0:p1].copy_(tensors["mse"])
            self._k_norms[layer_id, p0:p1].copy_(tensors["norms"])
            self._k_qjl_packed[layer_id, p0:p1].copy_(tensors["qjl"])
            self._k_res_norms[layer_id, p0:p1].copy_(tensors["res"])
            self._v_group_data[layer_id, p0:p1].copy_(tensors["v_data"])
            self._v_group_scales[layer_id, p0:p1].copy_(tensors["v_scales"])
            self._v_group_zeros[layer_id, p0:p1].copy_(tensors["v_zeros"])

    def _physical_store_kv(self, k: torch.Tensor, v: torch.Tensor, p0: int, p1: int, layer_id: int) -> None:
        chunk_len = p1 - p0
        k_bhsd = k.unsqueeze(0).transpose(1, 2).contiguous()
        v_bhsd = v.unsqueeze(0).transpose(1, 2).contiguous()
        with torch.no_grad():
            ck = self._k_mod_inf(layer_id).compress_bhsd(k_bhsd)
            cv = tq_group_quantize_values(v_bhsd, self._value_bits, self._value_group_size)
        tensors = {
            "mse": ck["mse_idx_bytes"][0].transpose(0, 1).contiguous(),
            "norms": ck["vec_norms"][0].transpose(0, 1).contiguous(),
            "qjl": ck["qjl_bytes"][0].transpose(0, 1).contiguous(),
            "res": ck["residual_norms"][0].transpose(0, 1).contiguous(),
            "v_data": cv["data"][0].transpose(0, 1).contiguous(),
            "v_scales": cv["scales"][0].transpose(0, 1).contiguous(),
            "v_zeros": cv["zeros"][0].transpose(0, 1).contiguous(),
        }
        self._store_arrays(layer_id, p0, p1, tensors)

    def store_kv(self, k: torch.Tensor, v: torch.Tensor, start_idx: int, end_idx: int, layer_id: int) -> None:
        chunks = self._logical_to_physical_chunks_for_store(layer_id, start_idx, end_idx)
        off = 0
        for p0, p1 in chunks:
            n = p1 - p0
            self._physical_store_kv(k[off : off + n], v[off : off + n], p0, p1, layer_id)
            off += n
        if self._kv_offload:
            self._record_cpu_update(layer_id)

    def _copy_layer_to_gpu(self, layer_id: int) -> None:
        self._k_packed_gpu.copy_(self._k_packed_cpu[layer_id], non_blocking=True)
        self._k_norms_gpu.copy_(self._k_norms_cpu[layer_id], non_blocking=True)
        self._k_qjl_packed_gpu.copy_(self._k_qjl_packed_cpu[layer_id], non_blocking=True)
        self._k_res_norms_gpu.copy_(self._k_res_norms_cpu[layer_id], non_blocking=True)
        self._v_group_data_gpu.copy_(self._v_group_data_cpu[layer_id], non_blocking=True)
        self._v_group_scales_gpu.copy_(self._v_group_scales_cpu[layer_id], non_blocking=True)
        self._v_group_zeros_gpu.copy_(self._v_group_zeros_cpu[layer_id], non_blocking=True)

    @staticmethod
    def _sh_extra_to_bhs(extra_sh: torch.Tensor) -> torch.Tensor:
        return extra_sh.unsqueeze(0).permute(0, 2, 1, 3).contiguous()

    def _decompress_k_from_arrays(self, layer_id: int, arrays: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]) -> torch.Tensor:
        packed, norms, qjl, res = arrays
        kv_len = packed.size(0)
        idx_bytes = packed.unsqueeze(0).permute(0, 2, 1, 3).contiguous()
        norms_bhs = norms.unsqueeze(0).transpose(1, 2).contiguous()
        qjl_bhs = self._sh_extra_to_bhs(qjl)
        res_bhs = res.unsqueeze(0).transpose(1, 2).contiguous()
        comp = {
            "mse_idx_bytes": idx_bytes,
            "qjl_bytes": qjl_bhs,
            "residual_norms": res_bhs,
            "vec_norms": norms_bhs,
            "shape": (1, self._num_heads, kv_len, self._head_dim),
            "mse_bits": self._key_bits - 1,
        }
        with torch.no_grad():
            out = self._k_mod_inf(layer_id).decompress_bhsd(comp)
        return out[0].transpose(0, 1).to(dtype=self._dtype)

    def k_cache(self, layer_id: int, attn_start: int, local_end: int) -> torch.Tensor:
        chunks = self._logical_to_physical_chunks(layer_id, attn_start, local_end)
        if self._kv_offload:
            self._check_layer_loaded(layer_id)
            parts = []
            for p0, p1 in chunks:
                parts.append(
                    (
                        self._k_packed_gpu[p0:p1],
                        self._k_norms_gpu[p0:p1],
                        self._k_qjl_packed_gpu[p0:p1],
                        self._k_res_norms_gpu[p0:p1],
                    )
                )
            arrays = tuple(torch.cat([part[i] for part in parts], dim=0) for i in range(4)) if len(parts) > 1 else parts[0]
            return self._decompress_k_from_arrays(layer_id, arrays)

        parts = []
        for p0, p1 in chunks:
            arrays = (
                self._k_packed[layer_id, p0:p1],
                self._k_norms[layer_id, p0:p1],
                self._k_qjl_packed[layer_id, p0:p1],
                self._k_res_norms[layer_id, p0:p1],
            )
            parts.append(self._decompress_k_from_arrays(layer_id, arrays))
        return parts[0] if len(parts) == 1 else torch.cat(parts, dim=0).contiguous()

    def _decompress_v_from_arrays(self, arrays: tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> torch.Tensor:
        data, scales, zeros = arrays
        kv_len = data.size(0)
        comp = {
            "data": data.unsqueeze(0).permute(0, 2, 1, 3).contiguous(),
            "scales": scales.unsqueeze(0).transpose(1, 2).contiguous(),
            "zeros": zeros.unsqueeze(0).transpose(1, 2).contiguous(),
            "bits": self._value_bits,
            "group_size": self._value_group_size,
            "shape": (1, self._num_heads, kv_len, self._head_dim),
        }
        with torch.no_grad():
            out = tq_group_dequantize_values(comp)
        return out[0].transpose(0, 1).to(dtype=self._dtype)

    def v_cache(self, layer_id: int, attn_start: int, local_end: int) -> torch.Tensor:
        chunks = self._logical_to_physical_chunks(layer_id, attn_start, local_end)
        if self._kv_offload:
            self._check_layer_loaded(layer_id)
            parts = []
            for p0, p1 in chunks:
                parts.append(
                    (
                        self._v_group_data_gpu[p0:p1],
                        self._v_group_scales_gpu[p0:p1],
                        self._v_group_zeros_gpu[p0:p1],
                    )
                )
            arrays = tuple(torch.cat([part[i] for part in parts], dim=0) for i in range(3)) if len(parts) > 1 else parts[0]
            return self._decompress_v_from_arrays(arrays)

        parts = []
        for p0, p1 in chunks:
            arrays = (
                self._v_group_data[layer_id, p0:p1],
                self._v_group_scales[layer_id, p0:p1],
                self._v_group_zeros[layer_id, p0:p1],
            )
            parts.append(self._decompress_v_from_arrays(arrays))
        return parts[0] if len(parts) == 1 else torch.cat(parts, dim=0).contiguous()

    def reset(self) -> None:
        if self._kv_offload:
            self.sync_all()
            self._zero_tensors(
                [
                    "_k_packed_cpu",
                    "_k_norms_cpu",
                    "_k_qjl_packed_cpu",
                    "_k_res_norms_cpu",
                    "_v_group_data_cpu",
                    "_v_group_scales_cpu",
                    "_v_group_zeros_cpu",
                    "_k_packed_gpu",
                    "_k_norms_gpu",
                    "_k_qjl_packed_gpu",
                    "_k_res_norms_gpu",
                    "_v_group_data_gpu",
                    "_v_group_scales_gpu",
                    "_v_group_zeros_gpu",
                ]
            )
            self._reset_offload_state()
        else:
            self._zero_tensors(
                [
                    "_k_packed",
                    "_k_norms",
                    "_k_qjl_packed",
                    "_k_res_norms",
                    "_v_group_data",
                    "_v_group_scales",
                    "_v_group_zeros",
                ]
            )
        self._reset_ends()
        self._reset_ring()


class StepTurboQuantRollingKVCachePool(TurboQuantRollingKVCachePool):
    def __init__(self, num_steps: int, *args, **kwargs) -> None:
        self.num_steps = int(num_steps)
        self._current_step = 0
        super().__init__(*args, **kwargs)

    @property
    def current_step(self) -> int:
        return self._current_step

    @current_step.setter
    def current_step(self, value: int) -> None:
        value = int(value)
        if value == self._current_step:
            return
        if getattr(self, "_kv_offload", False) and hasattr(self, "_prefetch_stream"):
            self.sync_all()
            self._reset_offload_state()
        self._current_step = value

    def _step(self) -> int:
        return int(self._current_step)

    def _ring_index(self, layer_id: int):
        return (self._step(), int(layer_id))

    def _init_kv_buffer(self) -> None:
        if self._kv_offload:
            # Use parent non-step allocation, then add step dimension manually.
            T, L, N, H = self.num_steps, self._num_layers, self._cache_size, self._num_heads
            ng = self._inf_v_n_groups
            self._k_packed_cpu = torch.zeros(T, L, N, H, self._inf_nk, dtype=torch.uint8, device="cpu").pin_memory()
            self._k_norms_cpu = torch.zeros(T, L, N, H, dtype=torch.float16, device="cpu").pin_memory()
            self._k_qjl_packed_cpu = torch.zeros(T, L, N, H, self._inf_nqjl, dtype=torch.uint8, device="cpu").pin_memory()
            self._k_res_norms_cpu = torch.zeros(T, L, N, H, dtype=torch.float16, device="cpu").pin_memory()
            self._v_group_data_cpu = torch.zeros(T, L, N, H, self._inf_v_width, dtype=torch.uint8, device="cpu").pin_memory()
            self._v_group_scales_cpu = torch.zeros(T, L, N, H, ng, dtype=torch.float16, device="cpu").pin_memory()
            self._v_group_zeros_cpu = torch.zeros(T, L, N, H, ng, dtype=torch.float16, device="cpu").pin_memory()
            d = self._device
            self._k_packed_gpu = torch.zeros(N, H, self._inf_nk, dtype=torch.uint8, device=d)
            self._k_norms_gpu = torch.zeros(N, H, dtype=torch.float16, device=d)
            self._k_qjl_packed_gpu = torch.zeros(N, H, self._inf_nqjl, dtype=torch.uint8, device=d)
            self._k_res_norms_gpu = torch.zeros(N, H, dtype=torch.float16, device=d)
            self._v_group_data_gpu = torch.zeros(N, H, self._inf_v_width, dtype=torch.uint8, device=d)
            self._v_group_scales_gpu = torch.zeros(N, H, ng, dtype=torch.float16, device=d)
            self._v_group_zeros_gpu = torch.zeros(N, H, ng, dtype=torch.float16, device=d)
            self._global_end = torch.zeros(T, L, dtype=torch.long, device="cpu")
            self._local_end = torch.zeros(T, L, dtype=torch.long, device="cpu")
            self._init_ring_metadata(T, L)
            self._init_offload_state((T, L))
            return

        T, L, N, H = self.num_steps, self._num_layers, self._cache_size, self._num_heads
        ng = self._inf_v_n_groups
        d = self._device
        self._k_packed = torch.zeros(T, L, N, H, self._inf_nk, dtype=torch.uint8, device=d)
        self._k_norms = torch.zeros(T, L, N, H, dtype=torch.float16, device=d)
        self._k_qjl_packed = torch.zeros(T, L, N, H, self._inf_nqjl, dtype=torch.uint8, device=d)
        self._k_res_norms = torch.zeros(T, L, N, H, dtype=torch.float16, device=d)
        self._v_group_data = torch.zeros(T, L, N, H, self._inf_v_width, dtype=torch.uint8, device=d)
        self._v_group_scales = torch.zeros(T, L, N, H, ng, dtype=torch.float16, device=d)
        self._v_group_zeros = torch.zeros(T, L, N, H, ng, dtype=torch.float16, device=d)
        self._global_end = torch.zeros(T, L, dtype=torch.long, device="cpu")
        self._local_end = torch.zeros(T, L, dtype=torch.long, device="cpu")
        self._init_ring_metadata(T, L)

    # Step-aware storage helpers.
    def _store_arrays(self, layer_id: int, p0: int, p1: int, tensors: dict[str, torch.Tensor]) -> None:
        s = self._step()
        if self._kv_offload:
            self._check_layer_loaded(layer_id)
            self._k_packed_gpu[p0:p1].copy_(tensors["mse"])
            self._k_norms_gpu[p0:p1].copy_(tensors["norms"])
            self._k_qjl_packed_gpu[p0:p1].copy_(tensors["qjl"])
            self._k_res_norms_gpu[p0:p1].copy_(tensors["res"])
            self._v_group_data_gpu[p0:p1].copy_(tensors["v_data"])
            self._v_group_scales_gpu[p0:p1].copy_(tensors["v_scales"])
            self._v_group_zeros_gpu[p0:p1].copy_(tensors["v_zeros"])
            self._k_packed_cpu[s, layer_id, p0:p1].copy_(tensors["mse"], non_blocking=True)
            self._k_norms_cpu[s, layer_id, p0:p1].copy_(tensors["norms"], non_blocking=True)
            self._k_qjl_packed_cpu[s, layer_id, p0:p1].copy_(tensors["qjl"], non_blocking=True)
            self._k_res_norms_cpu[s, layer_id, p0:p1].copy_(tensors["res"], non_blocking=True)
            self._v_group_data_cpu[s, layer_id, p0:p1].copy_(tensors["v_data"], non_blocking=True)
            self._v_group_scales_cpu[s, layer_id, p0:p1].copy_(tensors["v_scales"], non_blocking=True)
            self._v_group_zeros_cpu[s, layer_id, p0:p1].copy_(tensors["v_zeros"], non_blocking=True)
        else:
            self._k_packed[s, layer_id, p0:p1].copy_(tensors["mse"])
            self._k_norms[s, layer_id, p0:p1].copy_(tensors["norms"])
            self._k_qjl_packed[s, layer_id, p0:p1].copy_(tensors["qjl"])
            self._k_res_norms[s, layer_id, p0:p1].copy_(tensors["res"])
            self._v_group_data[s, layer_id, p0:p1].copy_(tensors["v_data"])
            self._v_group_scales[s, layer_id, p0:p1].copy_(tensors["v_scales"])
            self._v_group_zeros[s, layer_id, p0:p1].copy_(tensors["v_zeros"])

    def _offload_index(self, layer_id: int) -> tuple[int, ...]:
        return (self._step(), int(layer_id))

    def _copy_layer_to_gpu(self, layer_id: int) -> None:
        s = self._step()
        self._k_packed_gpu.copy_(self._k_packed_cpu[s, layer_id], non_blocking=True)
        self._k_norms_gpu.copy_(self._k_norms_cpu[s, layer_id], non_blocking=True)
        self._k_qjl_packed_gpu.copy_(self._k_qjl_packed_cpu[s, layer_id], non_blocking=True)
        self._k_res_norms_gpu.copy_(self._k_res_norms_cpu[s, layer_id], non_blocking=True)
        self._v_group_data_gpu.copy_(self._v_group_data_cpu[s, layer_id], non_blocking=True)
        self._v_group_scales_gpu.copy_(self._v_group_scales_cpu[s, layer_id], non_blocking=True)
        self._v_group_zeros_gpu.copy_(self._v_group_zeros_cpu[s, layer_id], non_blocking=True)

    def get_global_end(self, layer_id: int) -> int:
        return int(self._global_end[self._step(), layer_id].item())

    def get_local_end(self, layer_id: int) -> int:
        return int(self._local_end[self._step(), layer_id].item())

    def set_ends(self, layer_id: int, global_end: int, local_end: int) -> None:
        s = self._step()
        self._global_end[s, layer_id] = int(global_end)
        self._local_end[s, layer_id] = int(local_end)


# =============================================================================
# LongLive FP4
# =============================================================================


class LongLiveQuantRollingKVCachePool(_QuantTokenRingMixin, RollingKVCachePool):
    def __init__(
        self,
        num_layers: int,
        cache_size: int,
        num_heads: int,
        head_dim: int,
        dtype: torch.dtype,
        device: torch.device,
        *,
        block_token_size: int | None = None,
        scale_rule: str = "mse",
        backend: str = "pytorch",
        kv_offload: bool = False,
    ) -> None:
        self._block_token_size = int(block_token_size or cache_size)
        if self._block_token_size <= 0:
            raise ValueError(f"block_token_size must be positive, got {block_token_size}")
        self._quant_config = build_fp4_quant_config(scale_rule=scale_rule, backend=backend)
        self._dequant_backend = normalize_dequant_backend(backend)
        n_alloc = cdiv(int(cache_size), self._block_token_size) * self._block_token_size
        self._max_blocks = n_alloc // self._block_token_size
        super().__init__(num_layers, n_alloc, num_heads, head_dim, dtype, device, kv_offload=kv_offload)

    def _make_zero_block_qt(self) -> QuantizedTensor:
        h, d, blk = self._num_heads, self._head_dim, self._block_token_size
        zero = torch.zeros(blk * h, d, dtype=self._dtype, device=self._device)
        return quantize_to_fp4(zero, self._quant_config)

    @staticmethod
    def _clone_qt_to(qt: QuantizedTensor, device: torch.device | str, *, pin_memory: bool = False) -> QuantizedTensor:
        def clone_tensor(t: torch.Tensor | None) -> torch.Tensor | None:
            if t is None:
                return None
            out = t.detach().to(device=device).clone()
            return out.pin_memory() if pin_memory else out

        return QuantizedTensor(
            values=clone_tensor(qt.values),
            scale_factors=clone_tensor(qt.scale_factors),
            amax=clone_tensor(qt.amax),
            dtype=qt.dtype,
            original_shape=qt.original_shape,
            scale_rule=qt.scale_rule,
            padded_shape=qt.padded_shape,
        )

    @staticmethod
    def _copy_qt(dst: QuantizedTensor, src: QuantizedTensor, *, non_blocking: bool = True) -> None:
        dst.values.copy_(src.values, non_blocking=non_blocking)
        dst.scale_factors.copy_(src.scale_factors, non_blocking=non_blocking)
        if dst.amax is not None and src.amax is not None:
            dst.amax.copy_(src.amax, non_blocking=non_blocking)

    def _make_qt_blocks(
        self,
        zero_qt: QuantizedTensor,
        shape: tuple[int, ...],
        device: torch.device | str,
        *,
        pin_memory: bool = False,
    ):
        if len(shape) == 1:
            return [self._clone_qt_to(zero_qt, device, pin_memory=pin_memory) for _ in range(shape[0])]
        return [self._make_qt_blocks(zero_qt, shape[1:], device, pin_memory=pin_memory) for _ in range(shape[0])]

    def _reset_qt_blocks(self, blocks, zero_qt: QuantizedTensor) -> None:
        if not blocks:
            return
        if hasattr(blocks[0], "values"):
            for block in blocks:
                self._copy_qt(block, zero_qt, non_blocking=False)
            return
        for sub_blocks in blocks:
            self._reset_qt_blocks(sub_blocks, zero_qt)

    def _init_kv_buffer(self) -> None:
        if self._kv_offload:
            self._init_kv_buffer_offload()
            return
        zero_qt = self._make_zero_block_qt()
        self._k_blocks = [[clone_quantized_tensor(zero_qt) for _ in range(self._max_blocks)] for _ in range(self._num_layers)]
        self._v_blocks = [[clone_quantized_tensor(zero_qt) for _ in range(self._max_blocks)] for _ in range(self._num_layers)]
        self._global_end = torch.zeros(self._num_layers, dtype=torch.long, device="cpu")
        self._local_end = torch.zeros(self._num_layers, dtype=torch.long, device="cpu")
        self._init_ring_metadata(self._num_layers)

    def _init_kv_buffer_offload(self) -> None:
        zero_qt = self._make_zero_block_qt()
        self._k_blocks_cpu = self._make_qt_blocks(zero_qt, (self._num_layers, self._max_blocks), "cpu", pin_memory=True)
        self._v_blocks_cpu = self._make_qt_blocks(zero_qt, (self._num_layers, self._max_blocks), "cpu", pin_memory=True)
        self._k_blocks_gpu = self._make_qt_blocks(zero_qt, (self._max_blocks,), self._device)
        self._v_blocks_gpu = self._make_qt_blocks(zero_qt, (self._max_blocks,), self._device)
        self._global_end = torch.zeros(self._num_layers, dtype=torch.long, device="cpu")
        self._local_end = torch.zeros(self._num_layers, dtype=torch.long, device="cpu")
        self._init_ring_metadata(self._num_layers)
        self._init_offload_state((self._num_layers,))

    def _layer_k_blocks(self, layer_id: int):
        if self._kv_offload:
            return self._k_blocks_gpu
        return self._k_blocks[layer_id]

    def _layer_v_blocks(self, layer_id: int):
        if self._kv_offload:
            return self._v_blocks_gpu
        return self._v_blocks[layer_id]

    def _cpu_layer_k_blocks(self, layer_id: int):
        return self._k_blocks_cpu[layer_id]

    def _cpu_layer_v_blocks(self, layer_id: int):
        return self._v_blocks_cpu[layer_id]

    def _copy_layer_to_gpu(self, layer_id: int) -> None:
        for dst, src in zip(self._k_blocks_gpu, self._cpu_layer_k_blocks(layer_id), strict=True):
            self._copy_qt(dst, src)
        for dst, src in zip(self._v_blocks_gpu, self._cpu_layer_v_blocks(layer_id), strict=True):
            self._copy_qt(dst, src)

    def _quantize_block(self, k_nhd: torch.Tensor, v_nhd: torch.Tensor):
        blk, h, d = self._block_token_size, self._num_heads, self._head_dim
        if k_nhd.shape[0] != blk:
            raise ValueError(f"K block token count {k_nhd.shape[0]} != block_token_size {blk}")
        k2d = k_smooth(k_nhd).reshape(blk * h, d).contiguous()
        v2d = v_nhd.reshape(blk * h, d).contiguous()
        return quantize_to_fp4(k2d, self._quant_config), quantize_to_fp4(v2d, self._quant_config)

    def _dequant_token_range(self, blocks: list[QuantizedTensor], start: int, end: int) -> torch.Tensor:
        return dequantize_token_range(
            blocks,
            start,
            end,
            cache_size=self._cache_size,
            num_heads=self._num_heads,
            head_dim=self._head_dim,
            block_token_size=self._block_token_size,
            dtype=self._dtype,
            device=self._device,
            backend=self._dequant_backend,
        )

    def _pad_nhd_to_blocks(self, k_nhd: torch.Tensor, v_nhd: torch.Tensor):
        blk = self._block_token_size
        t_len = k_nhd.size(0)
        t_pad = cdiv(t_len, blk) * blk
        if t_len == t_pad:
            return k_nhd, v_nhd
        pad = t_pad - t_len
        return (
            torch.cat((k_nhd, k_nhd.new_zeros(pad, *k_nhd.shape[1:])), dim=0),
            torch.cat((v_nhd, v_nhd.new_zeros(pad, *v_nhd.shape[1:])), dim=0),
        )

    def _write_blocks_from_nhd(self, k_nhd: torch.Tensor, v_nhd: torch.Tensor, layer_id: int, physical_start: int) -> None:
        blk = self._block_token_size
        t_len = k_nhd.size(0)
        if t_len % blk != 0:
            raise RuntimeError(f"longlive_fp4 store length {t_len} is not a multiple of block_token_size {blk}")
        b0 = physical_start // blk
        if self._kv_offload:
            self._check_layer_loaded(layer_id)
        for i in range(t_len // blk):
            bi = b0 + i
            ts, te = i * blk, (i + 1) * blk
            k_qt, v_qt = self._quantize_block(k_nhd[ts:te], v_nhd[ts:te])
            self._copy_qt(self._layer_k_blocks(layer_id)[bi], k_qt)
            self._copy_qt(self._layer_v_blocks(layer_id)[bi], v_qt)
            if self._kv_offload:
                self._copy_qt(self._cpu_layer_k_blocks(layer_id)[bi], k_qt)
                self._copy_qt(self._cpu_layer_v_blocks(layer_id)[bi], v_qt)

    def _physical_store_kv(self, k: torch.Tensor, v: torch.Tensor, p0: int, p1: int, layer_id: int) -> None:
        blk = self._block_token_size
        s0 = (p0 // blk) * blk
        e1 = min(cdiv(p1, blk) * blk, self._cache_size)
        parts_k, parts_v = [], []
        if s0 < p0:
            parts_k.append(self._dequant_token_range(self._layer_k_blocks(layer_id), s0, p0))
            parts_v.append(self._dequant_token_range(self._layer_v_blocks(layer_id), s0, p0))
        parts_k.append(k)
        parts_v.append(v)
        if p1 < e1:
            parts_k.append(self._dequant_token_range(self._layer_k_blocks(layer_id), p1, e1))
            parts_v.append(self._dequant_token_range(self._layer_v_blocks(layer_id), p1, e1))
        k_cat, v_cat = self._pad_nhd_to_blocks(torch.cat(parts_k, dim=0), torch.cat(parts_v, dim=0))
        self._write_blocks_from_nhd(k_cat, v_cat, layer_id, s0)

    def store_kv(self, k: torch.Tensor, v: torch.Tensor, start_idx: int, end_idx: int, layer_id: int) -> None:
        chunks = self._logical_to_physical_chunks_for_store(layer_id, start_idx, end_idx)
        off = 0
        for p0, p1 in chunks:
            n = p1 - p0
            self._physical_store_kv(k[off : off + n], v[off : off + n], p0, p1, layer_id)
            off += n
        if self._kv_offload:
            self._record_cpu_update(layer_id)

    def _read_chunks(self, layer_id: int, start: int, end: int, kind: str) -> torch.Tensor:
        self._check_layer_loaded(layer_id)
        chunks = self._logical_to_physical_chunks(layer_id, start, end)
        blocks = self._layer_k_blocks(layer_id) if kind == "k" else self._layer_v_blocks(layer_id)
        outs = [self._dequant_token_range(blocks, p0, p1) for p0, p1 in chunks]
        if not outs:
            return torch.empty(0, self._num_heads, self._head_dim, dtype=self._dtype, device=self._device)
        return outs[0] if len(outs) == 1 else torch.cat(outs, dim=0).contiguous()

    def k_cache(self, layer_id: int, attn_start: int | None = None, local_end: int | None = None):
        if attn_start is None or local_end is None:
            raise ValueError("longlive_fp4 k_cache requires attn_start and local_end")
        return self._read_chunks(layer_id, attn_start, local_end, "k")

    def v_cache(self, layer_id: int, attn_start: int | None = None, local_end: int | None = None):
        if attn_start is None or local_end is None:
            raise ValueError("longlive_fp4 v_cache requires attn_start and local_end")
        return self._read_chunks(layer_id, attn_start, local_end, "v")

    def reset(self) -> None:
        zero_qt = self._make_zero_block_qt()
        if self._kv_offload:
            self.sync_all()
            self._reset_qt_blocks(self._k_blocks_cpu, zero_qt)
            self._reset_qt_blocks(self._v_blocks_cpu, zero_qt)
            self._reset_qt_blocks(self._k_blocks_gpu, zero_qt)
            self._reset_qt_blocks(self._v_blocks_gpu, zero_qt)
            self._reset_offload_state()
        else:
            self._reset_qt_blocks(self._k_blocks, zero_qt)
            self._reset_qt_blocks(self._v_blocks, zero_qt)
        self._reset_ends()
        self._reset_ring()


class StepLongLiveQuantRollingKVCachePool(LongLiveQuantRollingKVCachePool):
    def __init__(self, num_steps: int, *args, **kwargs) -> None:
        self.num_steps = int(num_steps)
        self._current_step = 0
        super().__init__(*args, **kwargs)

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

    def _ring_index(self, layer_id: int):
        return (self._step(), int(layer_id))

    def _init_kv_buffer(self) -> None:
        if self._kv_offload:
            self._init_kv_buffer_offload()
            return
        zero_qt = self._make_zero_block_qt()
        self._k_blocks = [[[clone_quantized_tensor(zero_qt) for _ in range(self._max_blocks)] for _ in range(self._num_layers)] for _ in range(self.num_steps)]
        self._v_blocks = [[[clone_quantized_tensor(zero_qt) for _ in range(self._max_blocks)] for _ in range(self._num_layers)] for _ in range(self.num_steps)]
        self._global_end = torch.zeros(self.num_steps, self._num_layers, dtype=torch.long, device="cpu")
        self._local_end = torch.zeros(self.num_steps, self._num_layers, dtype=torch.long, device="cpu")
        self._init_ring_metadata(self.num_steps, self._num_layers)

    def _init_kv_buffer_offload(self) -> None:
        zero_qt = self._make_zero_block_qt()
        self._k_blocks_cpu = self._make_qt_blocks(
            zero_qt,
            (self.num_steps, self._num_layers, self._max_blocks),
            "cpu",
            pin_memory=True,
        )
        self._v_blocks_cpu = self._make_qt_blocks(
            zero_qt,
            (self.num_steps, self._num_layers, self._max_blocks),
            "cpu",
            pin_memory=True,
        )
        self._k_blocks_gpu = self._make_qt_blocks(zero_qt, (self._max_blocks,), self._device)
        self._v_blocks_gpu = self._make_qt_blocks(zero_qt, (self._max_blocks,), self._device)
        self._global_end = torch.zeros(self.num_steps, self._num_layers, dtype=torch.long, device="cpu")
        self._local_end = torch.zeros(self.num_steps, self._num_layers, dtype=torch.long, device="cpu")
        self._init_ring_metadata(self.num_steps, self._num_layers)
        self._init_offload_state((self.num_steps, self._num_layers))

    def _layer_k_blocks(self, layer_id: int):
        if self._kv_offload:
            return self._k_blocks_gpu
        return self._k_blocks[self._step()][layer_id]

    def _layer_v_blocks(self, layer_id: int):
        if self._kv_offload:
            return self._v_blocks_gpu
        return self._v_blocks[self._step()][layer_id]

    def _cpu_layer_k_blocks(self, layer_id: int):
        return self._k_blocks_cpu[self._step()][layer_id]

    def _cpu_layer_v_blocks(self, layer_id: int):
        return self._v_blocks_cpu[self._step()][layer_id]

    def _offload_index(self, layer_id: int) -> tuple[int, ...]:
        return (self._step(), int(layer_id))

    def get_global_end(self, layer_id: int) -> int:
        return int(self._global_end[self._step(), layer_id].item())

    def get_local_end(self, layer_id: int) -> int:
        return int(self._local_end[self._step(), layer_id].item())

    def set_ends(self, layer_id: int, global_end: int, local_end: int) -> None:
        self._global_end[self._step(), layer_id] = int(global_end)
        self._local_end[self._step(), layer_id] = int(local_end)


# =============================================================================
# KIVI classes are intentionally kept in their dedicated ring implementation.
# For KIVI, use the previously generated kivi_quant_cache_ring_no_padding.py and
# add offload there, because KIVI's packed [H,D,T/pack] layout needs its own
# physical chunk materialization.
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

        outs = []
        for p0, p1 in chunks:
            if which == "k":
                outs.append(self._dequant_nhd(self._kivi_k_code(layer_id), self._kivi_k_scale(layer_id), self._kivi_k_mn(layer_id), p0, p1))
            elif which == "v":
                outs.append(self._dequant_nhd(self._kivi_v_code(layer_id), self._kivi_v_scale(layer_id), self._kivi_v_mn(layer_id), p0, p1))
            else:
                raise ValueError(f"invalid KIVI chunk kind: {which}")
        return outs[0] if len(outs) == 1 else torch.cat(outs, dim=0).contiguous()

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
