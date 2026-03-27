import torch
import triton
import triton.language as tl


@triton.jit
def _fused_norm_3drope_kernel(
    x_ptr,       # [seq, num_heads, HEAD_DIM] — bfloat16, in-place
    w_t_ptr,     # [HALF]    — RMSNorm weight for t-segment
    w_hw_ptr,    # [HALF]    — RMSNorm weight for hw-segment
    cos_t_ptr,   # [seq, HALF]    — unique freqs in [:QUARTER] per token
    sin_t_ptr,   # [seq, HALF]
    cos_h_ptr,   # [seq, QUARTER] — unique freqs in [:EIGHTH] per token
    sin_h_ptr,   # [seq, QUARTER]
    cos_w_ptr,   # [seq, QUARTER]
    sin_w_ptr,   # [seq, QUARTER]
    num_heads,   # int  (runtime)
    eps,         # float (runtime)
    HEAD_DIM: tl.constexpr,
    HALF:     tl.constexpr,
    QUARTER:  tl.constexpr,
    EIGHTH:   tl.constexpr,
):

    pid = tl.program_id(0)
    s   = pid // num_heads   # token index
    h   = pid % num_heads    # head  index

    base    = (s * num_heads + h) * HEAD_DIM
    offs_q  = tl.arange(0, QUARTER)   # 0 .. QUARTER-1
    offs_e  = tl.arange(0, EIGHTH)    # 0 .. EIGHTH-1

    # ------------------------------------------------------------------ #
    # t-segment  [base : base+HALF]  split into two QUARTER-wide chunks   #
    # ------------------------------------------------------------------ #
    xt1 = tl.load(x_ptr + base             + offs_q).to(tl.float32)
    xt2 = tl.load(x_ptr + base + QUARTER   + offs_q).to(tl.float32)

    # RMSNorm over HALF elements (fp32 variance, same as fp32_variance_qwen)
    var_t     = (tl.sum(xt1 * xt1) + tl.sum(xt2 * xt2)) * (1.0 / HALF) + eps
    irms_t    = tl.rsqrt(var_t)

    wt1 = tl.load(w_t_ptr             + offs_q).to(tl.float32)
    wt2 = tl.load(w_t_ptr + QUARTER   + offs_q).to(tl.float32)
    xt1 = xt1 * irms_t * wt1
    xt2 = xt2 * irms_t * wt2

    # Neox-style RoPE on t-segment.
    # cos_t is stored as cat(freqs, freqs) => unique part is [:QUARTER].
    # rotation: [xt1, xt2] -> [xt1*c - xt2*s,  xt2*c + xt1*s]
    c_t = tl.load(cos_t_ptr + s * HALF + offs_q).to(tl.float32)
    s_t = tl.load(sin_t_ptr + s * HALF + offs_q).to(tl.float32)
    new_xt1 = xt1 * c_t - xt2 * s_t
    new_xt2 = xt2 * c_t + xt1 * s_t

    # ------------------------------------------------------------------ #
    # hw-segment [base+HALF : base+HEAD_DIM]                               #
    #   h-segment: [base+HALF           : base+HALF+QUARTER]               #
    #   w-segment: [base+HALF+QUARTER   : base+HEAD_DIM    ]               #
    # Each sub-segment is split into two EIGHTH-wide chunks for RoPE.      #
    # ------------------------------------------------------------------ #
    xh1 = tl.load(x_ptr + base + HALF                    + offs_e).to(tl.float32)
    xh2 = tl.load(x_ptr + base + HALF + EIGHTH           + offs_e).to(tl.float32)
    xw1 = tl.load(x_ptr + base + HALF + QUARTER          + offs_e).to(tl.float32)
    xw2 = tl.load(x_ptr + base + HALF + QUARTER + EIGHTH + offs_e).to(tl.float32)

    # RMSNorm over HALF elements (shared scale for the whole hw-segment)
    var_hw  = (tl.sum(xh1 * xh1) + tl.sum(xh2 * xh2) +
               tl.sum(xw1 * xw1) + tl.sum(xw2 * xw2)) * (1.0 / HALF) + eps
    irms_hw = tl.rsqrt(var_hw)

    wh1 = tl.load(w_hw_ptr                    + offs_e).to(tl.float32)
    wh2 = tl.load(w_hw_ptr + EIGHTH           + offs_e).to(tl.float32)
    ww1 = tl.load(w_hw_ptr + QUARTER          + offs_e).to(tl.float32)
    ww2 = tl.load(w_hw_ptr + QUARTER + EIGHTH + offs_e).to(tl.float32)
    xh1 = xh1 * irms_hw * wh1
    xh2 = xh2 * irms_hw * wh2
    xw1 = xw1 * irms_hw * ww1
    xw2 = xw2 * irms_hw * ww2

    # Neox-style RoPE on h-segment (unique part of cos_h is [:EIGHTH])
    c_h = tl.load(cos_h_ptr + s * QUARTER + offs_e).to(tl.float32)
    s_h = tl.load(sin_h_ptr + s * QUARTER + offs_e).to(tl.float32)
    new_xh1 = xh1 * c_h - xh2 * s_h
    new_xh2 = xh2 * c_h + xh1 * s_h

    # Neox-style RoPE on w-segment
    c_w = tl.load(cos_w_ptr + s * QUARTER + offs_e).to(tl.float32)
    s_w = tl.load(sin_w_ptr + s * QUARTER + offs_e).to(tl.float32)
    new_xw1 = xw1 * c_w - xw2 * s_w
    new_xw2 = xw2 * c_w + xw1 * s_w

    # ------------------------------------------------------------------ #
    # Store back in-place (bfloat16)                                       #
    # ------------------------------------------------------------------ #
    tl.store(x_ptr + base             + offs_q, new_xt1.to(tl.bfloat16))
    tl.store(x_ptr + base + QUARTER   + offs_q, new_xt2.to(tl.bfloat16))

    tl.store(x_ptr + base + HALF                    + offs_e, new_xh1.to(tl.bfloat16))
    tl.store(x_ptr + base + HALF + EIGHTH           + offs_e, new_xh2.to(tl.bfloat16))
    tl.store(x_ptr + base + HALF + QUARTER          + offs_e, new_xw1.to(tl.bfloat16))
    tl.store(x_ptr + base + HALF + QUARTER + EIGHTH + offs_e, new_xw2.to(tl.bfloat16))


def fused_norm_3drope(
    x: torch.Tensor,
    w_t: torch.Tensor,
    w_hw: torch.Tensor,
    cos_t: torch.Tensor,
    sin_t: torch.Tensor,
    cos_h: torch.Tensor,
    sin_h: torch.Tensor,
    cos_w: torch.Tensor,
    sin_w: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Fused in-place: dual RMSNorm + 3D Neox-RoPE for one Q or K tensor.

    Layout per head (head_dim = D):
      [0 : D/2]       t-segment  -> q_norm_t  -> RoPE(cos_t, sin_t)
      [D/2 : 3D/4]    h-segment  -> q_norm_hw -> RoPE(cos_h, sin_h)
      [3D/4 : D]      w-segment  -> q_norm_hw -> RoPE(cos_w, sin_w)

    Args:
        x     : [seq, num_heads, head_dim]  bfloat16, contiguous
        w_t   : [head_dim//2]  RMSNorm weight for t-segment
        w_hw  : [head_dim//2]  RMSNorm weight for hw-segment
        cos_t : [1, seq, head_dim//2]  (= cat(freqs, freqs), unique part [:head_dim//4])
        sin_t : [1, seq, head_dim//2]
        cos_h : [1, seq, head_dim//4]  (unique part [:head_dim//8])
        sin_h : [1, seq, head_dim//4]
        cos_w : [1, seq, head_dim//4]
        sin_w : [1, seq, head_dim//4]
        eps   : RMSNorm epsilon

    Returns:
        x modified in-place
    """
    assert x.dtype == torch.bfloat16, "fused_norm_3drope requires bfloat16 input"
    assert x.is_contiguous(), "fused_norm_3drope requires contiguous input"

    seq, num_heads, head_dim = x.shape

    # squeeze the leading batch-1 dim; .contiguous() is a no-op when already so
    cos_t_sq = cos_t.squeeze(0).contiguous()   # [seq, head_dim//2]
    sin_t_sq = sin_t.squeeze(0).contiguous()
    cos_h_sq = cos_h.squeeze(0).contiguous()   # [seq, head_dim//4]
    sin_h_sq = sin_h.squeeze(0).contiguous()
    cos_w_sq = cos_w.squeeze(0).contiguous()   # [seq, head_dim//4]
    sin_w_sq = sin_w.squeeze(0).contiguous()

    half    = head_dim // 2
    quarter = head_dim // 4
    eighth  = head_dim // 8

    grid = (seq * num_heads,)
    _fused_norm_3drope_kernel[grid](
        x,
        w_t, w_hw,
        cos_t_sq, sin_t_sq,
        cos_h_sq, sin_h_sq,
        cos_w_sq, sin_w_sq,
        num_heads=num_heads,
        eps=eps,
        HEAD_DIM=head_dim,
        HALF=half,
        QUARTER=quarter,
        EIGHTH=eighth,
    )
    return x
