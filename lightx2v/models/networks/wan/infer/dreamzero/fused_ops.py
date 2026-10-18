"""Segment-wise modulation kernels for DreamZero.

DreamZero's per-token timesteps take only a few distinct values (video tokens,
action tokens, state tokens), so the AdaLN modulation has only a few distinct
rows.  Instead of materialising ``(seq, 6, dim)`` modulation tensors, callers
keep a ``(num_segments, dim)`` table plus an int32 ``seg_idx`` per token and
these kernels gather the row inside the elementwise pass.

All kernels reproduce the rounding of the unfused PyTorch / Triton ops they
replace, so the results are bitwise identical to the original path.
"""

import torch
import triton
import triton.language as tl


def can_use_seg_kernels(x):
    return x.is_cuda and x.dim() == 2 and x.is_contiguous() and x.dtype in (torch.bfloat16, torch.float16)


def _check_seg_idx(seg_idx, x):
    # The kernels read seg_idx[row] for every row of x without bounds checks.
    assert seg_idx.shape == (x.shape[0],) and seg_idx.dtype == torch.int32 and seg_idx.device == x.device and seg_idx.is_contiguous()


@triton.jit
def _seg_modulate_kernel(
    X,
    SEG,
    SCALE,
    SHIFT,
    Y,
    L,
    C,
    stride_scale,
    stride_shift,
    BLOCK_L: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    pid_l = tl.program_id(0)
    pid_c = tl.program_id(1)
    rows = pid_l * BLOCK_L + tl.arange(0, BLOCK_L)
    cols = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    mask_l = rows < L
    mask = mask_l[:, None] & (cols[None, :] < C)
    seg = tl.load(SEG + rows, mask=mask_l, other=0).to(tl.int64)
    x = tl.load(X + rows[:, None].to(tl.int64) * C + cols[None, :], mask=mask, other=0)
    scale = tl.load(SCALE + seg[:, None] * stride_scale + cols[None, :], mask=mask, other=0)
    shift = tl.load(SHIFT + seg[:, None] * stride_shift + cols[None, :], mask=mask, other=0)
    # Same expression (and dtypes) as fuse_scale_shift_kernel_blc_opt.
    y = x * (1 + scale) + shift
    tl.store(Y + rows[:, None].to(tl.int64) * C + cols[None, :], y, mask=mask)


def seg_modulate(x, seg_idx, scale, shift, out=None):
    """``x * (1 + scale[seg_idx]) + shift[seg_idx]`` with (nseg, C) tables."""
    L, C = x.shape
    if out is None:
        out = torch.empty_like(x)
    assert scale.stride(-1) == 1 and shift.stride(-1) == 1
    _check_seg_idx(seg_idx, x)
    grid = (triton.cdiv(L, 128), triton.cdiv(C, 128))
    with torch.cuda.device(x.device):
        _seg_modulate_kernel[grid](x, seg_idx, scale, shift, out, L, C, scale.stride(0), shift.stride(0), BLOCK_L=128, BLOCK_C=128, num_warps=4, num_stages=2)
    return out


@triton.jit
def _seg_gate_residual_kernel(
    X,
    Yin,
    SEG,
    GATE,
    L,
    C,
    stride_gate,
    BLOCK_L: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    pid_l = tl.program_id(0)
    pid_c = tl.program_id(1)
    rows = pid_l * BLOCK_L + tl.arange(0, BLOCK_L)
    cols = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    mask_l = rows < L
    mask = mask_l[:, None] & (cols[None, :] < C)
    seg = tl.load(SEG + rows, mask=mask_l, other=0).to(tl.int64)
    offs = rows[:, None].to(tl.int64) * C + cols[None, :]
    x = tl.load(X + offs, mask=mask, other=0)
    y = tl.load(Yin + offs, mask=mask, other=0)
    gate = tl.load(GATE + seg[:, None] * stride_gate + cols[None, :], mask=mask, other=0)
    # torch: tmp = bf16(float(y) * float(gate)); x = bf16(float(x) + float(tmp))
    prod = (y.to(tl.float32) * gate.to(tl.float32)).to(x.dtype)
    out = (x.to(tl.float32) + prod.to(tl.float32)).to(x.dtype)
    tl.store(X + offs, out, mask=mask)


def seg_gate_residual_(x, y, seg_idx, gate):
    """In place ``x += y * gate[seg_idx]`` with BF16 rounding of the product."""
    L, C = x.shape
    assert y.shape == x.shape and y.is_contiguous() and gate.stride(-1) == 1
    _check_seg_idx(seg_idx, x)
    grid = (triton.cdiv(L, 64), triton.cdiv(C, 256))
    with torch.cuda.device(x.device):
        _seg_gate_residual_kernel[grid](x, y, seg_idx, gate, L, C, gate.stride(0), BLOCK_L=64, BLOCK_C=256, num_warps=8, enable_fp_fusion=False)
    return x


# ---------------------------------------------------------------------------
# LayerNorm (PyTorch-compatible) fused with residual update and modulation.
#
# Reproduces at::native vectorized_layer_norm_kernel (PyTorch 2.10, CUDA) for
# N = 5120: 128 threads (4 warps x 32 lanes), each thread runs an online
# Welford update over 4-element vectors at stride 128, then a shuffle-down
# tree inside each warp and a shared-memory tree across warps, combined with
# cuWelfordCombine.  Mul+add pairs are written as explicit FMAs where nvcc
# contracts them (fp fusion disabled for everything else).
# ---------------------------------------------------------------------------

_LN_N = 5120
_LN_THREADS = 128
_LN_VEC = 4
_LN_ITERS = _LN_N // (_LN_THREADS * _LN_VEC)  # 10


def can_use_torch_layer_norm_5120(x):
    if torch.__version__.split(".")[:2] != ["2", "10"] or torch.version.hip is not None:
        return False
    # PyTorch only takes the vectorized kernel for 16-byte aligned rows.
    return x.is_cuda and x.dtype == torch.bfloat16 and x.dim() == 2 and x.shape[1] == _LN_N and x.is_contiguous() and x.numel() < 2**31 and x.data_ptr() % 16 == 0


@triton.jit
def _welford_combine(own_m, own_s, oth_m, oth_s, CNT: tl.constexpr):
    # cuWelfordCombine(dataB=own, dataA=oth) with equal counts CNT on both sides.
    delta = own_m - oth_m
    coef = tl.math.div_rn(tl.full(own_m.shape, 1.0, tl.float32), tl.full(own_m.shape, 2.0 * CNT, tl.float32))
    n_a = CNT * coef
    n_b = CNT * coef
    mean = tl.fma(n_a, oth_m, n_b * own_m)
    sigma2 = tl.fma(delta * delta * CNT, n_b, oth_s + own_s)
    return mean, sigma2


@triton.jit
def _welford_halves(m, s, R: tl.constexpr, G: tl.constexpr, H: tl.constexpr, CNT: tl.constexpr):
    # m, s: [R, G, 2*H] -> combine element i (own) with element i + H (other).
    m2 = tl.permute(tl.reshape(m, (R, G, 2, H)), (0, 1, 3, 2))
    s2 = tl.permute(tl.reshape(s, (R, G, 2, H)), (0, 1, 3, 2))
    own_m, oth_m = tl.split(m2)
    own_s, oth_s = tl.split(s2)
    return _welford_combine(own_m, own_s, oth_m, oth_s, CNT)


@triton.jit
def _welford_update(m, s, v, K: tl.constexpr):
    # cuWelfordOnlineSum: count goes K -> K + 1 (identical in every thread).
    delta = v - m
    r = tl.math.div_rn(tl.full(m.shape, 1.0, tl.float32), tl.full(m.shape, K + 1.0, tl.float32))
    m_new = tl.fma(delta, r, m)
    s_new = tl.fma(delta, v - m_new, s)
    return m_new, s_new


@triton.jit
def _ln_residual_row_tile(X, RES, GATE, seg, rows, rmask, j, stride_gate, RES_MODE: tl.constexpr, R: tl.constexpr):
    # Load the [R, 128, 4] tile of iteration j (thread t reads vector j*128 + t)
    # and apply the residual update in BF16 exactly like the unfused torch ops.
    t = tl.arange(0, 128)
    e = tl.arange(0, 4)
    cols = (j * 128 + t)[None, :, None] * 4 + e[None, None, :]
    mask = rmask[:, None, None]
    offs = rows[:, None, None].to(tl.int64) * 5120 + cols
    x = tl.load(X + offs, mask=mask, other=0)
    if RES_MODE == 1:
        r = tl.load(RES + offs, mask=mask, other=0)
        x = (x.to(tl.float32) + r.to(tl.float32)).to(x.dtype)
    elif RES_MODE == 2:
        r = tl.load(RES + offs, mask=mask, other=0)
        g = tl.load(GATE + seg[:, None, None] * stride_gate + cols, mask=mask, other=0)
        prod = (r.to(tl.float32) * g.to(tl.float32)).to(x.dtype)
        x = (x.to(tl.float32) + prod.to(tl.float32)).to(x.dtype)
    return x, offs, cols, mask


@triton.jit
def _fused_ln_kernel(
    X,
    RES,
    GATE,
    stride_gate,
    SEG,
    W,
    B,
    SCALE,
    SHIFT,
    stride_scale,
    stride_shift,
    Y,
    L,
    eps,
    RES_MODE: tl.constexpr,  # 0: none, 1: x += res, 2: x += res * gate[seg]  (x updated in place)
    AFFINE: tl.constexpr,
    MODULATE: tl.constexpr,
    R: tl.constexpr,
):
    rows = tl.program_id(0) * R + tl.arange(0, R)
    rmask = rows < L
    seg = tl.zeros((R,), tl.int64)
    if RES_MODE == 2 or MODULATE:
        seg = tl.load(SEG + rows, mask=rmask, other=0).to(tl.int64)

    # Pass 1: per-thread online Welford over 10 iterations x 4 elements.
    m = tl.zeros((R, 128), tl.float32)
    s = tl.zeros((R, 128), tl.float32)
    for j in tl.static_range(10):
        x, offs, cols, mask = _ln_residual_row_tile(X, RES, GATE, seg, rows, rmask, j, stride_gate, RES_MODE, R)
        xf = tl.reshape(x.to(tl.float32), (R, 128, 2, 2))
        even, odd = tl.split(xf)
        x0, x2 = tl.split(even)
        x1, x3 = tl.split(odd)
        m, s = _welford_update(m, s, x0, j * 4 + 0)
        m, s = _welford_update(m, s, x1, j * 4 + 1)
        m, s = _welford_update(m, s, x2, j * 4 + 2)
        m, s = _welford_update(m, s, x3, j * 4 + 3)

    # Warp shuffle-down tree (offsets 16, 8, 4, 2, 1), then the warp tree (2, 1).
    m = tl.reshape(m, (R, 4, 32))
    s = tl.reshape(s, (R, 4, 32))
    m, s = _welford_halves(m, s, R, 4, 16, 40)
    m, s = _welford_halves(m, s, R, 4, 8, 80)
    m, s = _welford_halves(m, s, R, 4, 4, 160)
    m, s = _welford_halves(m, s, R, 4, 2, 320)
    m, s = _welford_halves(m, s, R, 4, 1, 640)
    m = tl.reshape(m, (R, 4))
    s = tl.reshape(s, (R, 4))
    m, s = _welford_halves(m[:, None, :], s[:, None, :], R, 1, 2, 1280)
    m, s = _welford_halves(m, s, R, 1, 1, 2560)
    mean = tl.reshape(m, (R,))
    var = tl.math.div_rn(tl.reshape(s, (R,)), tl.full((R,), 5120.0, tl.float32))
    rstd = tl.rsqrt(var + eps)

    # Pass 2: recompute the residual (rows are L2 resident), write x and y.
    for j in tl.static_range(10):
        x, offs, cols, mask = _ln_residual_row_tile(X, RES, GATE, seg, rows, rmask, j, stride_gate, RES_MODE, R)
        if RES_MODE != 0:
            # The updated residual is not read again until the next norm, after
            # several GEMMs; evict-first keeps y (the next GEMM's A operand)
            # L2 resident instead.  Cache hint only: results are unchanged.
            tl.store(X + offs, x, mask=mask, eviction_policy="evict_first")
        normed = rstd[:, None, None] * (x.to(tl.float32) - mean[:, None, None])
        if AFFINE:
            w = tl.load(W + cols).to(tl.float32)
            b = tl.load(B + cols).to(tl.float32)
            normed = tl.fma(w, normed, b)
        y = normed.to(x.dtype)
        if MODULATE:
            scale = tl.load(SCALE + seg[:, None, None] * stride_scale + cols, mask=mask, other=0)
            shift = tl.load(SHIFT + seg[:, None, None] * stride_shift + cols, mask=mask, other=0)
            y = _modulate_expr(y, scale, shift)
        tl.store(Y + offs, y, mask=mask)


@triton.jit
def _modulate_expr(x, scale, shift):
    # fuse_scale_shift_kernel_blc_opt compiles (BF16 operands, fp fusion on)
    # to add.bf16x2 + fma.rn.bf16x2; keep that rounding with fusion off.
    return tl.fma(x, 1 + scale, shift)


def fused_layer_norm(x, eps, weight=None, bias=None, residual=None, gate=None, seg_idx=None, scale=None, shift=None, rows_per_program=2):
    """``y = modulate(layer_norm(x'))`` with optional in-place residual update.

    x' = x + residual                (gate is None)
    x' = x + residual * gate[seg]    (gate given; BF16 product rounding kept)
    x is overwritten with x' when a residual is given.  Modulation (``scale``,
    ``shift`` as [nseg, C] tables) is applied after the BF16 LayerNorm output.
    Caller must check ``can_use_torch_layer_norm_5120``.
    """
    L = x.shape[0]
    y = torch.empty_like(x)
    res_mode = 0 if residual is None else (2 if gate is not None else 1)
    if residual is not None:
        assert residual.shape == x.shape and residual.dtype == x.dtype and residual.is_contiguous()
    affine = weight is not None
    modulate = scale is not None
    if res_mode == 2 or modulate:
        _check_seg_idx(seg_idx, x)
    dummy = x
    R = rows_per_program
    with torch.cuda.device(x.device):
        _fused_ln_kernel[(triton.cdiv(L, R),)](
            x,
            residual if residual is not None else dummy,
            gate if gate is not None else dummy,
            gate.stride(0) if gate is not None else 0,
            seg_idx if seg_idx is not None else dummy,
            weight if affine else dummy,
            bias if affine else dummy,
            scale if modulate else dummy,
            shift if modulate else dummy,
            scale.stride(0) if modulate else 0,
            shift.stride(0) if modulate else 0,
            y,
            L,
            eps,
            RES_MODE=res_mode,
            AFFINE=affine,
            MODULATE=modulate,
            R=R,
            num_warps=4,
            enable_fp_fusion=False,
        )
    return y
