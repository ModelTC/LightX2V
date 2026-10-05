import torch
import triton
import triton.language as tl


@triton.jit
def _paired_rms(Q, K, WQ, WK, OQ, OK, NQ: tl.constexpr, D: tl.constexpr, QS: tl.constexpr, KS: tl.constexpr, EPS: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    is_q = row < NQ
    local_row = row if is_q else row - NQ
    x_ptr = Q if is_q else K
    w_ptr = WQ if is_q else WK
    y_ptr = OQ if is_q else OK
    stride = QS if is_q else KS
    col = tl.arange(0, BLOCK)
    x = tl.load(x_ptr + local_row * stride + col, col < D, 0).to(tl.float32)
    scale = tl.rsqrt(tl.sum(x * x, axis=0) / D + EPS)
    normalized = (x * scale).to(Q.dtype.element_ty).to(tl.float32)
    weight = tl.load(w_ptr + col, col < D, 0).to(tl.float32)
    tl.store(y_ptr + local_row * D + col, normalized * weight, col < D)


@triton.jit
def _rope_fp64(X, FREQ, Y, PAIRS: tl.constexpr, D: tl.constexpr, XS: tl.constexpr, HD: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = i < PAIRS
    row, col = i // (D // 2), (i % (D // 2)) * 2
    offset = row * HD + col % HD
    real = tl.load(FREQ + offset, valid, 0).to(tl.float64)
    imag = tl.load(FREQ + offset + 1, valid, 0).to(tl.float64)
    x0 = tl.load(X + row * XS + col, valid, 0).to(tl.float64)
    x1 = tl.load(X + row * XS + col + 1, valid, 0).to(tl.float64)
    # Preserve PyTorch's double -> float -> BF16/FP16 conversion.
    tl.store(Y + row * D + col, (x0 * real - x1 * imag).to(tl.float32), valid)
    tl.store(Y + row * D + col + 1, (x0 * imag + x1 * real).to(tl.float32), valid)


@triton.jit
def _modulation(X, S, G, Y, N: tl.constexpr, D: tl.constexpr, SROWS: tl.constexpr, GROWS: tl.constexpr, SS: tl.constexpr, GS: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = i < N
    dtype = X.dtype.element_ty
    x = tl.load(X + i, valid, 0).to(tl.float32)
    s = tl.load(S + (i // D % SROWS) * SS + i % D, valid, 0).to(tl.float32)
    g = tl.load(G + (i // D % GROWS) * GS + i % D, valid, 0).to(tl.float32)
    factor = (1.0 + g).to(dtype).to(tl.float32)
    y = (x * factor).to(dtype).to(tl.float32) + s
    tl.store(Y + i, y.to(dtype), valid)


@triton.jit
def _residual_gate(X, R, G, Y, N: tl.constexpr, D: tl.constexpr, GROWS: tl.constexpr, GS: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = i < N
    dtype = X.dtype.element_ty
    x = tl.load(X + i, valid, 0).to(tl.float32)
    residual = tl.load(R + i, valid, 0).to(tl.float32)
    g = tl.load(G + (i // D % GROWS) * GS + i % D, valid, 0).to(tl.float32)
    y = x + (residual * g).to(dtype).to(tl.float32)
    tl.store(Y + i, y.to(dtype), valid)


def paired_qk_rms_norm(q, k, norm_q, norm_k):
    oq, ok = (torch.empty(x.shape, device=x.device, dtype=x.dtype) for x in (q, k))
    d = q.shape[1]
    _paired_rms[(q.shape[0] + k.shape[0],)](
        q,
        k,
        norm_q.weight,
        norm_k.weight,
        oq,
        ok,
        q.shape[0],
        d,
        q.stride(0),
        k.stride(0),
        norm_q.eps,
        triton.next_power_of_2(d),
        enable_fp_fusion=False,
    )
    return oq, ok


def rope_fp64(x, freqs, head_dim):
    """Rotate interleaved [N,D] heads using contiguous complex128 [N,1,HD/2] frequencies."""
    y = torch.empty(x.shape, device=x.device, dtype=x.dtype)
    _rope_fp64[(triton.cdiv(x.numel() // 2, 256),)](
        x,
        torch.view_as_real(freqs),
        y,
        x.numel() // 2,
        x.shape[1],
        x.stride(0),
        head_dim,
        256,
        enable_fp_fusion=False,
    )
    return y


def modulate(x, shift, scale):
    y = torch.empty_like(x)
    _modulation[(triton.cdiv(x.numel(), 256),)](
        x,
        shift,
        scale,
        y,
        x.numel(),
        x.shape[1],
        shift.shape[0],
        scale.shape[0],
        shift.stride(0),
        scale.stride(0),
        256,
        enable_fp_fusion=False,
    )
    return y


def residual_gate(x, gate, residual):
    y = torch.empty_like(x)
    _residual_gate[(triton.cdiv(x.numel(), 256),)](
        x,
        residual,
        gate,
        y,
        x.numel(),
        x.shape[1],
        gate.shape[0],
        gate.stride(0),
        256,
        enable_fp_fusion=False,
    )
    return y
