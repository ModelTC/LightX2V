"""Step-isolated FP8 storage for causal H3's rolling KV cache."""

import torch
import triton
import triton.language as tl

from lightx2v.common.kvcache.rolling import StepRollingKVCachePool


@triton.jit
def _physical_token(token, sink, head, capacity, active: tl.constexpr):
    if active:
        return tl.where(token < sink, token, sink + (head + token - sink) % capacity)
    return token


@triton.jit
def _rotate_fp32(x, y, cos, sin):
    # Preserve separate FP32 multiply/add rounding instead of contracting to FMA.
    return tl.inline_asm_elementwise(
        "{.reg .f32 a,b; mul.rn.f32 a,$1,$2; mul.rn.f32 b,$3,$4; add.rn.f32 $0,a,b;}", constraints="=f,f,f,f,f", args=[x, cos, y, sin], dtype=tl.float32, is_pure=True, pack=1
    )


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
    physical = _physical_token(token, SINK, HEAD, CAP, ACTIVE)
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
    physical = _physical_token(token, SINK, HEAD, CAP, ACTIVE)
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
    rotated = _rotate_fp32(x, y, c, s)
    tl.store(OUTPUT + i, tl.where(col < R, rotated, x), i < N)


@triton.jit
def _read_transposed(Q, S, OUTPUT, LENGTH, SINK, HEAD, CAP, PADDED, H: tl.constexpr, D: tl.constexpr, ACTIVE: tl.constexpr):
    row = tl.program_id(0) * 32 + tl.arange(0, 32)
    head = tl.program_id(1)
    col = tl.program_id(2) * 32 + tl.arange(0, 32)
    physical = _physical_token(row, SINK, HEAD, CAP, ACTIVE)
    scale = tl.load(S + physical * H + head, row < LENGTH, 0)
    x = tl.load(Q + (physical[:, None] * H + head) * D + col[None, :], (row[:, None] < LENGTH) & (col[None, :] < D), 0.0).to(tl.float32)
    x = x * scale[:, None]
    permuted = (row & ~15) | (row & 1) | ((row & 8) >> 2) | ((row & 2) << 1) | ((row & 4) << 1)
    address = (col[None, :] * H + head) * PADDED + permuted[:, None]
    # Include zero padding: Sage V quantization reads a 64-token-aligned buffer.
    tl.store(OUTPUT + address, x, (row[:, None] < PADDED) & (col[None, :] < D))


@triton.jit
def _store_read_rope(
    Q, K, V, KQ, VQ, KS, VS, Q_COS, Q_SIN, K_COS, K_SIN, QO, KO, VO, START, END, SINK, HEAD, CAP, H: tl.constexpr, D: tl.constexpr, R: tl.constexpr, ACTIVE: tl.constexpr, B: tl.constexpr
):
    pid = tl.program_id(0)
    history_blocks = tl.cdiv(START * H, B)
    col = tl.arange(0, D)[None, :]
    partner = tl.where(col < R // 2, col + R // 2, col - R // 2)
    # Separate current/history blocks to avoid reading newly written cache rows.
    if pid < history_blocks:
        row = pid * B + tl.arange(0, B)
        valid = row < START * H
        token, head = row // H, row % H
        physical = _physical_token(token, SINK, HEAD, CAP, ACTIVE)
        cache_row = physical * H + head
        ks = tl.load(KS + cache_row, valid, 0)
        vs = tl.load(VS + cache_row, valid, 0)
        k = tl.load(KQ + cache_row[:, None] * D + col, valid[:, None], 0.0).to(tl.float32)
        ky = tl.load(KQ + cache_row[:, None] * D + partner, valid[:, None] & (col < R), 0.0).to(tl.float32)
        v = tl.load(VQ + cache_row[:, None] * D + col, valid[:, None], 0.0).to(tl.float32)
        # Match the BF16/FP16 cache-read rounding before FP32 RoPE.
        k = (k * ks[:, None]).to(KO.dtype.element_ty).to(tl.float32)
        ky = (ky * ks[:, None]).to(KO.dtype.element_ty).to(tl.float32)
        v = v * vs[:, None]
    else:
        local_row = (pid - history_blocks) * B + tl.arange(0, B)
        row = START * H + local_row
        valid = row < END * H
        token, head = row // H, row % H
        physical = _physical_token(token, SINK, HEAD, CAP, ACTIVE)
        cache_row = physical * H + head
        k = tl.load(K + local_row[:, None] * D + col, valid[:, None], 0.0).to(tl.float32)
        ky = tl.load(K + local_row[:, None] * D + partner, valid[:, None] & (col < R), 0.0).to(tl.float32)
        v = tl.load(V + local_row[:, None] * D + col, valid[:, None], 0.0).to(tl.float32)
        ks = tl.maximum(tl.max(tl.abs(k), 1), 1e-8) / 448.0
        vs = tl.maximum(tl.max(tl.abs(v), 1), 1e-8) / 448.0
        kq = tl.minimum(tl.maximum(k / ks[:, None], -448.0), 448.0).to(KQ.dtype.element_ty)
        kyq = tl.minimum(tl.maximum(ky / ks[:, None], -448.0), 448.0).to(KQ.dtype.element_ty)
        vq = tl.minimum(tl.maximum(v / vs[:, None], -448.0), 448.0).to(VQ.dtype.element_ty)
        tl.store(KQ + cache_row[:, None] * D + col, kq, valid[:, None])
        tl.store(VQ + cache_row[:, None] * D + col, vq, valid[:, None])
        tl.store(KS + cache_row, ks, valid)
        tl.store(VS + cache_row, vs, valid)
        k = (kq.to(tl.float32) * ks[:, None]).to(KO.dtype.element_ty).to(tl.float32)
        ky = (kyq.to(tl.float32) * ks[:, None]).to(KO.dtype.element_ty).to(tl.float32)
        v = vq.to(tl.float32) * vs[:, None]
        q = tl.load(Q + local_row[:, None] * D + col, valid[:, None], 0.0).to(tl.float32)
        qy = tl.load(Q + local_row[:, None] * D + partner, valid[:, None] & (col < R), 0.0).to(tl.float32)
        qy = tl.where(col < R // 2, -qy, qy)
        q_cos = tl.load(Q_COS + (local_row // H)[:, None] * R + col, valid[:, None] & (col < R), 0)
        q_sin = tl.load(Q_SIN + (local_row // H)[:, None] * R + col, valid[:, None] & (col < R), 0)
        qr = _rotate_fp32(q, qy, q_cos, q_sin)
        tl.store(QO + local_row[:, None] * D + col, tl.where(col < R, qr, q), valid[:, None])
    ky = tl.where(col < R // 2, -ky, ky)
    k_cos = tl.load(K_COS + token[:, None] * R + col, valid[:, None] & (col < R), 0)
    k_sin = tl.load(K_SIN + token[:, None] * R + col, valid[:, None] & (col < R), 0)
    kr = _rotate_fp32(k, ky, k_cos, k_sin)
    tl.store(KO + row[:, None] * D + col, tl.where(col < R, kr, k), valid[:, None])
    tl.store(VO + row[:, None] * D + col, v, valid[:, None])


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

    def store_read_rope(self, q, k, v, start_idx, layer_id, query_freqs, key_freqs):
        """Write pre-RoPE FP8 KV and materialize Q/K RoPE plus V in one launch."""
        length, h, d = q.shape
        end = start_idx + length
        expected = (length, self._num_heads, self._head_dim)
        if q.shape != expected or k.shape != expected or v.shape != expected:
            raise ValueError(f"Expected fused cache Q/K/V with shape {expected}")
        if d != 128 or not 0 <= start_idx <= end <= self._cache_size:
            raise ValueError("Fused cache update requires head_dim=128 and an in-bounds write")
        q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
        r = query_freqs[0].shape[-1]
        if r % 2 or r > d:
            raise ValueError("Unsupported split-half frequency layout")
        for freqs, tokens in ((query_freqs, length), (key_freqs, end)):
            if any(tensor.shape != (tokens, r) or not tensor.is_contiguous() for tensor in freqs):
                raise ValueError("Fused cache RoPE requires contiguous full-width cosine/sine tables")
        qo = torch.empty_like(q)
        ko = torch.empty(end, h, d, device=self._device, dtype=self._dtype)
        vo = torch.empty_like(ko)
        if end:
            layout = self.ring_layout(layer_id)
            index = self._metadata_index(layer_id)
            block = 8
            _store_read_rope[(triton.cdiv(start_idx * h, block) + triton.cdiv(length * h, block),)](
                q,
                k,
                v,
                self._k_buffer[index],
                self._v_buffer[index],
                self._k_scale[index],
                self._v_scale[index],
                *query_freqs,
                *key_freqs,
                qo,
                ko,
                vo,
                start_idx,
                end,
                layout.sink_tokens,
                layout.recent_head,
                self._cache_size - layout.sink_tokens,
                h,
                d,
                r,
                layout.active,
                block,
                num_warps=4,
                enable_fp_fusion=False,
            )
            layout.record_write(end)
        return qo, ko, vo

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
