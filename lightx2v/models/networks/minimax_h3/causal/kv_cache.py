"""Step-isolated FP8 storage for causal H3's rolling KV cache."""

import torch
import triton
import triton.language as tl

from lightx2v.common.kvcache.rolling import StepRollingKVCachePool


@triton.jit
def _store_fp8_kv(K, V, KQ, VQ, KS, VS, INPUT_ROW, PHYSICAL_ROW, D: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    d = tl.arange(0, BLOCK)
    k = tl.load(K + (INPUT_ROW + row) * D + d, d < D, 0).to(tl.float32)
    v = tl.load(V + (INPUT_ROW + row) * D + d, d < D, 0).to(tl.float32)
    ks = tl.maximum(tl.max(tl.abs(k), 0), 1e-8) / 448.0
    vs = tl.maximum(tl.max(tl.abs(v), 0), 1e-8) / 448.0
    tl.store(KQ + (PHYSICAL_ROW + row) * D + d, tl.minimum(tl.maximum(k / ks, -448.0), 448.0), d < D)
    tl.store(VQ + (PHYSICAL_ROW + row) * D + d, tl.minimum(tl.maximum(v / vs, -448.0), 448.0), d < D)
    tl.store(KS + PHYSICAL_ROW + row, ks)
    tl.store(VS + PHYSICAL_ROW + row, vs)


@triton.jit
def _read_fp8_kv(Q, S, OUTPUT, START, N, SINK, HEAD, CAP, STRIDE_N, STRIDE_H, STRIDE_D, H: tl.constexpr, D: tl.constexpr, ACTIVE: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    token = i // (H * D) + START
    head = i // D % H
    physical = token
    if ACTIVE:
        physical = tl.where(token < SINK, token, SINK + (HEAD + token - SINK) % CAP)
    q = tl.load(Q + (physical * H + head) * D + i % D, i < N, 0.0).to(tl.float32)
    scale = tl.load(S + physical * H + head, i < N, 0)
    output_offset = i // (H * D) * STRIDE_N + head * STRIDE_H + i % D * STRIDE_D
    tl.store(OUTPUT + output_offset, q * scale, i < N)


@triton.jit
def _read_rope(Q, S, COS, SIN, OUTPUT, N, SINK, HEAD, CAP, H: tl.constexpr, D: tl.constexpr, R: tl.constexpr, ACTIVE: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    token = i // (H * D)
    head = i // D % H
    col = i % D
    physical = token
    if ACTIVE:
        physical = tl.where(token < SINK, token, SINK + (HEAD + token - SINK) % CAP)
    scale = tl.load(S + physical * H + head, i < N, 0)
    base = (physical * H + head) * D
    x = tl.load(Q + base + col, i < N, 0.0).to(tl.float32) * scale
    # Preserve the intermediate cache-read rounding before the FP32 rotation.
    x = x.to(OUTPUT.dtype.element_ty).to(tl.float32)
    partner = tl.where(col < R // 2, col + R // 2, col - R // 2)
    y = tl.load(Q + base + partner, (i < N) & (col < R), 0.0).to(tl.float32) * scale
    y = y.to(OUTPUT.dtype.element_ty).to(tl.float32)
    y = tl.where(col < R // 2, -y, y)
    c = tl.load(COS + token * R + col, (i < N) & (col < R), 0)
    s = tl.load(SIN + token * R + col, (i < N) & (col < R), 0)
    rotated = tl.inline_asm_elementwise(
        "{.reg .f32 a,b; mul.rn.f32 a,$1,$2; mul.rn.f32 b,$3,$4; add.rn.f32 $0,a,b;}", constraints="=f,f,f,f,f", args=[x, c, y, s], dtype=tl.float32, is_pure=True, pack=1
    )
    tl.store(OUTPUT + i, tl.where(col < R, rotated, x), i < N)


@triton.jit
def _read_transposed(Q, S, OUTPUT, LENGTH, SINK, HEAD, CAP, PADDED, H: tl.constexpr, D: tl.constexpr, ACTIVE: tl.constexpr):
    row = tl.program_id(0) * 32 + tl.arange(0, 32)
    head = tl.program_id(1)
    col = tl.program_id(2) * 32 + tl.arange(0, 32)
    physical = row
    if ACTIVE:
        physical = tl.where(row < SINK, row, SINK + (HEAD + row - SINK) % CAP)
    scale = tl.load(S + physical * H + head, row < LENGTH, 0)
    x = tl.load(Q + (physical[:, None] * H + head) * D + col[None, :], (row[:, None] < LENGTH) & (col[None, :] < D), 0.0).to(tl.float32)
    x = x * scale[:, None]
    permuted = (row & ~15) | (row & 1) | ((row & 8) >> 2) | ((row & 2) << 1) | ((row & 4) << 1)
    address = (col[None, :] * H + head) * PADDED + permuted[:, None]
    # Include zero padding: Sage V quantization reads a 64-token-aligned buffer.
    tl.store(OUTPUT + address, x, (row[:, None] < PADDED) & (col[None, :] < D))


class FP8StepRollingKVCachePool(StepRollingKVCachePool):
    """Store E4M3 K/V with separate FP32 scales per token/head; read in compute dtype.

    Inherited ring metadata keeps every denoising step and layer independent.
    Keys remain pre-RoPE, so rolling positions are applied after dequantization.
    """

    def _init_kv_buffer(self):
        shape = (self.num_steps, self._num_layers, self._cache_size, self._num_heads)
        self._k_buffer = torch.empty(*shape, self._head_dim, device=self._device, dtype=torch.float8_e4m3fn)
        self._v_buffer = torch.empty_like(self._k_buffer)
        self._k_scale = torch.empty(*shape, device=self._device, dtype=torch.float32)
        self._v_scale = torch.empty_like(self._k_scale)
        self._init_cache_metadata()

    def store_kv(self, k, v, start_idx, end_idx, layer_id):
        if not 0 <= start_idx <= end_idx <= self._cache_size:
            raise ValueError(f"Invalid FP8 KV write range [{start_idx}, {end_idx}) for capacity {self._cache_size}")
        if end_idx == start_idx:
            return
        expected = (end_idx - start_idx, self._num_heads, self._head_dim)
        if k.shape != expected or v.shape != expected:
            raise ValueError(f"Expected FP8 KV inputs with shape {expected}, got {k.shape} and {v.shape}")
        k, v = k.contiguous(), v.contiguous()
        index = self._metadata_index(layer_id)
        for logical, physical, length in self.ring_layout(layer_id).spans(start_idx, end_idx):
            _store_fp8_kv[(length * self._num_heads,)](
                k,
                v,
                self._k_buffer[index],
                self._v_buffer[index],
                self._k_scale[index],
                self._v_scale[index],
                (logical - start_idx) * self._num_heads,
                physical * self._num_heads,
                self._head_dim,
                triton.next_power_of_2(self._head_dim),
                num_warps=4,
            )
        self.ring_layout(layer_id).record_write(end_idx)

    def _read_logical(self, layer_id, attn_start, local_end, which, out=None):
        if not 0 <= attn_start <= local_end <= self._cache_size:
            raise ValueError(f"Invalid FP8 KV read range [{attn_start}, {local_end}) for capacity {self._cache_size}")
        shape = (local_end - attn_start, self._num_heads, self._head_dim)
        if out is None:
            out = torch.empty(shape, device=self._device, dtype=self._dtype)
        elif out.shape != shape or out.device != self._k_buffer.device:
            raise ValueError(f"FP8 KV output must have shape {shape} on {self._k_buffer.device}")
        if shape[0] == 0:
            return out
        layout = self.ring_layout(layer_id)
        active, sink, head = layout.active, layout.sink_tokens, layout.recent_head
        index = self._metadata_index(layer_id)
        quantized, scale = (self._k_buffer[index], self._k_scale[index]) if which == "k" else (self._v_buffer[index], self._v_scale[index])
        _read_fp8_kv[(triton.cdiv(out.numel(), 1024),)](
            quantized,
            scale,
            out,
            attn_start,
            out.numel(),
            sink,
            head,
            self._cache_size - sink,
            *out.stride(),
            self._num_heads,
            self._head_dim,
            active,
            1024,
        )
        return out

    def k_cache(self, layer_id, attn_start=0, local_end=None, out=None):
        end = self.get_valid_length(layer_id) if local_end is None else local_end
        return self._read_logical(layer_id, attn_start, end, "k", out)

    def v_cache(self, layer_id, attn_start=0, local_end=None, out=None):
        end = self.get_valid_length(layer_id) if local_end is None else local_end
        return self._read_logical(layer_id, attn_start, end, "v", out)

    def read_rope(self, layer, end, freqs):
        cos, sin = freqs
        d, h = self._head_dim, self._num_heads
        r = cos.shape[-1]
        if cos.shape != (end, r) or sin.shape != cos.shape or r % 2 or r > d:
            raise ValueError("Unsupported split-half frequency layout")
        if not cos.is_contiguous() or not sin.is_contiguous():
            raise ValueError("RoPE frequencies must be contiguous")
        out = torch.empty(end, h, d, device=self._device, dtype=self._dtype)
        if end:
            layout = self.ring_layout(layer)
            active, sink, head = layout.active, layout.sink_tokens, layout.recent_head
            index = self._metadata_index(layer)
            _read_rope[(triton.cdiv(out.numel(), 1024),)](
                self._k_buffer[index], self._k_scale[index], cos, sin, out, out.numel(), sink, head, self._cache_size - sink, h, d, r, active, 1024, enable_fp_fusion=False
            )
        return out

    def read_transposed(self, layer, end):
        d, h = self._head_dim, self._num_heads
        padded = triton.cdiv(end, 64) * 64
        out = torch.empty(1, d, h, padded, device=self._device, dtype=self._dtype)
        if end:
            layout = self.ring_layout(layer)
            active, sink, head = layout.active, layout.sink_tokens, layout.recent_head
            index = self._metadata_index(layer)
            _read_transposed[(triton.cdiv(padded, 32), h, triton.cdiv(d, 32))](self._v_buffer[index], self._v_scale[index], out, end, sink, head, self._cache_size - sink, padded, h, d, active)
        return out
