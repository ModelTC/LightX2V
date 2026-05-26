import numpy as np
import torch
import triton
import triton.language as tl
from triton import next_power_of_2


@triton.jit
def quant_key_per_thread_int8_static_scale_kernel(
    Input,  # [chunk_len, H, D]   bf16/fp16
    Output,  # [chunk_len, H, D]   int8
    Scale,  # [num_blk, H, 4]     fp32 (preset)
    L,  # chunk_len
    StartIdx,  # absolute position in buffer where chunk starts
    stride_iz,
    stride_ih,
    stride_in,
    stride_oz,
    stride_oh,
    stride_on,
    stride_sb,
    stride_sh,  # stride per-block, per-head; per-thread stride is 1
    C: tl.constexpr,
    BLK: tl.constexpr,
):
    off_blk = tl.program_id(0) // 4
    off_tld = tl.program_id(0) % 4
    off_h = tl.program_id(1)
    off_b = tl.program_id(2)

    # Translate block-relative token offsets into chunk-local indices.
    # When StartIdx % BLK != 0, the first chunk block begins at a
    # negative chunk-local index — those positions are masked off.
    block_local_base = off_blk * BLK - (StartIdx % BLK)
    offs_in_blk = tl.arange(0, BLK // 8) * 8 + off_tld * 2
    offs_n0 = block_local_base + offs_in_blk
    offs_n1 = offs_n0 + 1
    offs_k = tl.arange(0, C)

    mask_n0 = (offs_n0 >= 0) & (offs_n0 < L)
    mask_n1 = (offs_n1 >= 0) & (offs_n1 < L)

    input_ptrs0 = Input + off_b * stride_iz + off_h * stride_ih + offs_n0[:, None] * stride_in + offs_k[None, :]
    input_ptrs1 = Input + off_b * stride_iz + off_h * stride_ih + offs_n1[:, None] * stride_in + offs_k[None, :]
    output_ptrs0 = Output + off_b * stride_oz + off_h * stride_oh + offs_n0[:, None] * stride_on + offs_k[None, :]
    output_ptrs1 = Output + off_b * stride_oz + off_h * stride_oh + offs_n1[:, None] * stride_on + offs_k[None, :]

    # Scale layout [num_blk, H, 4] — per-thread stride is 1.
    scale = tl.load(Scale + off_blk * stride_sb + off_h * stride_sh + off_tld)

    x0 = tl.load(input_ptrs0, mask=mask_n0[:, None]).to(tl.float32)
    x1 = tl.load(input_ptrs1, mask=mask_n1[:, None]).to(tl.float32)

    x0_int8 = x0 / scale
    x1_int8 = x1 / scale
    x0_int8 += 0.5 * tl.where(x0_int8 >= 0, 1, -1)
    x1_int8 += 0.5 * tl.where(x1_int8 >= 0, 1, -1)

    # Saturate before int8 cast — preset scale doesn't bound |x/scale|.
    x0_int8 = tl.minimum(tl.maximum(x0_int8, -127.0), 127.0).to(tl.int8)
    x1_int8 = tl.minimum(tl.maximum(x1_int8, -127.0), 127.0).to(tl.int8)

    tl.store(output_ptrs0, x0_int8, mask=mask_n0[:, None])
    tl.store(output_ptrs1, x1_int8, mask=mask_n1[:, None])


@triton.jit
def fp8_v_quantize_nhd_prescale_kernel(
    X,
    OUT,
    S,  # [H, D]  fp32  (per-channel v_scale = amax / 448, shared across L)
    n_tok: tl.int32,
    n_heads: tl.int32,
    D: tl.int32,
    BLOCK_D: tl.constexpr,
    FP8_MAX_VAL: tl.constexpr,
    SCALE_EPS: tl.constexpr,
):
    """Quantise V ``[L, H, D]`` contiguous to fp32 staging, ``y = x / S[h,d]``."""
    row = tl.program_id(0)
    h = row % n_heads
    t = row // n_heads
    d_off = tl.arange(0, BLOCK_D)
    m = d_off < D
    base_v = t * n_heads * D + h * D
    base_s = h * D
    x = tl.load(X + base_v + d_off, mask=m, other=0.0).to(tl.float32)
    s = tl.load(S + base_s + d_off, mask=m, other=0.0).to(tl.float32)
    s = tl.maximum(s, SCALE_EPS)
    y = x / s
    y = tl.clamp(y, -FP8_MAX_VAL, FP8_MAX_VAL)
    tl.store(OUT + base_v + d_off, y, mask=m)


# --------------------------------------------------------------------------- #
#  K int8 rescale on rolling: new_int8 ≈ round( old * src_scale / dst_scale )
#  (per-token, per-head ratio; D channels share the same ratio for that t,h)
# --------------------------------------------------------------------------- #


@triton.jit
def k_int8_roll_rescale_nhd_kernel(
    X,  # int8 [T, H, D]
    OUT,  # int8 [T, H, D]
    S_SRC,  # f32 [T, H] row-major
    S_DST,  # f32 [T, H]
    T: tl.int32,
    H: tl.int32,
    D: tl.int32,
    BLOCK_D: tl.constexpr,
    SCALE_EPS: tl.constexpr,
):
    """Re-quant int8 so dequant with ``dst`` position's scale recovers
    the value encoded with ``src`` position's scale (Sage k_block_scale).
    Rounding matches the repo's int8 path: add 0.5*sign, clamp, to int8.
    """
    row = tl.program_id(0)
    h = row % H
    t = row // H
    offs = tl.arange(0, BLOCK_D)
    m = offs < D
    base = t * H * D + h * D
    s_src = tl.load(S_SRC + t * H + h).to(tl.float32)
    s_dst = tl.load(S_DST + t * H + h).to(tl.float32)
    s_dst = tl.maximum(s_dst, SCALE_EPS)
    ratio = s_src / s_dst
    x = tl.load(X + base + offs, mask=m, other=0.0).to(tl.float32)
    y = x * ratio
    y = y + 0.5 * tl.where(y >= 0, 1, -1)
    y = tl.minimum(tl.maximum(y, -127.0), 127.0)
    y = y.to(tl.int8)
    tl.store(OUT + base + offs, y, mask=m)


def k_int8_roll_rescale_triton(
    x: torch.Tensor,
    out: torch.Tensor,
    src_scale: torch.Tensor,
    dst_scale: torch.Tensor,
    *,
    scale_eps: float = 1e-5,
) -> None:
    """In-place: ``out[t,h,d] = sat(round_half_away( x * src_s/dst_s ))``.

    Shapes: ``x``, ``out`` are ``[T, H, D]`` int8, ``src_scale``/``dst_scale`` are
    ``[T, H]`` fp32 (Sage K scale one value per token per head for the
    current thread group).
    """
    if x.shape != out.shape:
        raise ValueError(f"x and out must match, got {x.shape} vs {out.shape}")
    T, h_, d_ = x.shape
    if src_scale.shape != (T, h_) or dst_scale.shape != (T, h_):
        raise ValueError("src_scale and dst_scale must be [T, H]")
    if not out.is_contiguous():
        raise ValueError("out must be contiguous to write in-place to the K buffer")
    if not x.is_contiguous():
        x = x.contiguous()
    ss = src_scale.to(device=x.device, dtype=torch.float32, copy=False).contiguous()
    ds = dst_scale.to(device=x.device, dtype=torch.float32, copy=False).contiguous()
    block_d = next_power_of_2(d_)
    grid = (T * h_,)
    t_i, h_i, d_i = int(T), int(h_), int(d_)
    k_int8_roll_rescale_nhd_kernel[grid](
        x,
        out,
        ss,
        ds,
        t_i,
        h_i,
        d_i,
        block_d,
        SCALE_EPS=scale_eps,
        num_warps=4,
    )


def quant_value_per_channel_fp8_static_scale_kernel(
    v: torch.Tensor,
    v_scale: torch.Tensor,
    *,
    fp8_max: float = 448.0,
    scale_eps: float = 1e-5,
) -> torch.Tensor:
    """Sage-matched per-channel V quant: ``v`` ``[L,H,D]``, ``v_scale`` ``[H,D]`` (``amax/448``)."""
    v = v.contiguous()
    vs = v_scale.to(device=v.device, dtype=torch.float32, copy=False)
    n_tok, n_h, d = v.shape
    if vs.shape != (n_h, d):
        raise ValueError(f"v_scale {tuple(vs.shape)} must be [H,D]={(n_h, d)} for v {tuple(v.shape)}")
    vs = vs.contiguous()
    out = torch.empty_like(v, dtype=torch.float32, device=v.device)
    block_d = next_power_of_2(d)
    grid = (n_tok * n_h,)
    n_t = int(n_tok)
    n_h_ = int(n_h)
    d_ = int(d)
    fp8_v_quantize_nhd_prescale_kernel[grid](
        v,
        out,
        vs,
        n_t,
        n_h_,
        d_,
        block_d,
        FP8_MAX_VAL=fp8_max,
        SCALE_EPS=scale_eps,
        num_warps=8,
    )
    return out.to(torch.float8_e4m3fn)


@triton.jit
def _pack_along_last_dim(bits: tl.constexpr, intensor_ptr, code_ptr, N, num_feats: tl.constexpr, feat_per_int: tl.constexpr, BLOCK_SIZE_N: tl.constexpr):
    num_int_per_y_dim = num_feats // feat_per_int
    bid = tl.program_id(axis=0)
    yid = tl.program_id(axis=1)
    offs_N = bid * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    block_start = intensor_ptr + offs_N * num_feats + yid * feat_per_int  # offset of the first element at current tile
    packed = tl.zeros((BLOCK_SIZE_N,), dtype=tl.int32)
    for i in range(feat_per_int):
        ptr = block_start + i
        element = tl.load(ptr, mask=offs_N < N, other=0.0)
        element = element << (i * bits)
        # Combine the value using bitwise OR
        packed = packed | element
    tl.store(code_ptr + offs_N * num_int_per_y_dim + yid, packed, mask=offs_N < N)


@triton.jit
def _minmax_along_last_dim(x_ptr, mn_ptr, mx_ptr, total_elements: tl.constexpr, N: tl.constexpr, num_groups: tl.constexpr, group_size: tl.constexpr, BLOCK_SIZE_N: tl.constexpr):
    bid = tl.program_id(axis=0)
    offsets_b = bid * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offsets = offsets_b[:, None] * group_size + tl.arange(0, group_size)[None, :]
    mask = offsets < total_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    mx_val = tl.max(x, axis=1)
    mn_val = tl.min(x, axis=1)
    # tl.device_print('shape', mn_val[:, None].shape)
    tl.store(mn_ptr + offsets_b, mn_val, mask=offsets_b < N * num_groups)
    tl.store(mx_ptr + offsets_b, mx_val, mask=offsets_b < N * num_groups)


def triton_quantize_and_pack_along_last_dim(data: torch.Tensor, group_size: int, bit: int):
    assert len(data.shape) == 4
    shape = data.shape
    B, nh, D, T = shape
    # ================== Get Scale & Zeros ===============
    assert T % group_size == 0
    num_groups = T // group_size
    new_shape = (B * nh * D, num_groups, group_size)
    scale_mn_shape = B, nh, D, num_groups
    # Quantize
    data = data.reshape(new_shape)
    mx = torch.empty((B * nh * D, num_groups), device=data.device, dtype=data.dtype)
    mn = torch.empty((B * nh * D, num_groups), device=data.device, dtype=data.dtype)
    BLOCK_SIZE_N = 128

    def grid(meta):
        return (triton.cdiv(data.shape[0] * data.shape[1], BLOCK_SIZE_N),)

    with torch.cuda.device(data.device):
        _minmax_along_last_dim[grid](data, mn, mx, data.numel(), data.shape[0], num_groups, group_size, BLOCK_SIZE_N=BLOCK_SIZE_N, num_warps=8)
    # mn = torch.min(data, dim=-1, keepdim=True)[0].squeeze(-1)
    # mx = torch.max(data, dim=-1, keepdim=True)[0].squeeze(-1)
    scale = (mx - mn) / (2**bit - 1)
    data = data - mn.unsqueeze(-1)
    data.div_(scale.unsqueeze(-1))
    data = data.clamp_(0, 2**bit - 1).round_().to(torch.int32)
    data = data.view(-1, T)
    feat_per_int = 32 // bit
    packshape = (
        np.prod(shape[:-1]),
        shape[-1] // feat_per_int,
    )
    code = torch.zeros(*packshape, device=data.device, dtype=torch.int32)

    def grid(meta):
        return (
            triton.cdiv(data.shape[0], BLOCK_SIZE_N),
            data.shape[1] // feat_per_int,
        )

    with torch.cuda.device(data.device):
        _pack_along_last_dim[grid](bit, data, code, data.shape[0], data.shape[1], feat_per_int, BLOCK_SIZE_N=BLOCK_SIZE_N, num_warps=8)
    return code.view(B, nh, D, -1), scale.reshape(scale_mn_shape), mn.reshape(scale_mn_shape)


@triton.jit
def _unpack_dequant_lastdim_kernel(
    code_ptr,
    scale_ptr,
    mn_ptr,
    out_ptr,
    total_rows,
    T: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    # code strides: [B, H, D, P]
    c_s0: tl.constexpr,
    c_s1: tl.constexpr,
    c_s2: tl.constexpr,
    c_s3: tl.constexpr,
    # scale/mn strides: [B, H, D, G]
    s_s0: tl.constexpr,
    s_s1: tl.constexpr,
    s_s2: tl.constexpr,
    s_s3: tl.constexpr,
    # output strides: [B, H, D, T]
    o_s0: tl.constexpr,
    o_s1: tl.constexpr,
    o_s2: tl.constexpr,
    o_s3: tl.constexpr,
    BITS: tl.constexpr,
    FEAT_PER_INT: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_T: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_t = tl.program_id(1)

    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # [BM]
    toks = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)  # [BT]

    hd = H * D
    b = rows // hd
    rem = rows - b * hd
    h = rem // D
    d = rem - h * D

    pack_col = toks // FEAT_PER_INT
    shift = (toks - pack_col * FEAT_PER_INT) * BITS
    group_col = toks // GROUP_SIZE

    row_mask = rows < total_rows
    tok_mask = toks < T
    mask = row_mask[:, None] & tok_mask[None, :]

    code_offsets = b[:, None] * c_s0 + h[:, None] * c_s1 + d[:, None] * c_s2 + pack_col[None, :] * c_s3
    packed = tl.load(code_ptr + code_offsets, mask=mask, other=0).to(tl.uint32)

    qmask = (1 << BITS) - 1
    q = ((packed >> shift[None, :]) & qmask).to(tl.float32)

    scale_offsets = b[:, None] * s_s0 + h[:, None] * s_s1 + d[:, None] * s_s2 + group_col[None, :] * s_s3
    sc = tl.load(scale_ptr + scale_offsets, mask=mask, other=0.0).to(tl.float32)
    z = tl.load(mn_ptr + scale_offsets, mask=mask, other=0.0).to(tl.float32)

    val = q * sc + z

    out_offsets = b[:, None] * o_s0 + h[:, None] * o_s1 + d[:, None] * o_s2 + toks[None, :] * o_s3
    tl.store(out_ptr + out_offsets, val, mask=mask)


def unpack_and_dequant_cache_triton(
    code: torch.Tensor,
    scale: torch.Tensor,
    mn: torch.Tensor,
    group_size: int,
    bits: int,
    dtype: torch.dtype = torch.float16,
    block_m: int = 4,
    block_t: int = 128,
) -> torch.Tensor:
    """Fused replacement for unpack_and_dequant_cache(...).

    Args:
        code: int32 packed tensor with shape [B, H, D, n_packs].
        scale: tensor with shape [B, H, D, n_groups] or [B, H, D, n_groups, 1].
        mn: same shape as scale.
        group_size: quantization group size along the original T dimension.
        bits: one of 2, 4, 8.
        dtype: output dtype, usually fp16/bf16/fp32.

    Returns:
        Dequantized tensor with shape [B, H, D, n_packs * (32 // bits)].
    """
    assert bits in (2, 4, 8), f"bits must be 2/4/8, got {bits}"
    assert code.is_cuda and scale.is_cuda and mn.is_cuda
    assert code.dtype == torch.int32, f"code must be int32, got {code.dtype}"
    assert code.dim() == 4, f"code must be [B,H,D,P], got {tuple(code.shape)}"

    if scale.dim() == 5:
        assert scale.shape[-1] == 1 and mn.shape[-1] == 1
        scale = scale.squeeze(-1)
        mn = mn.squeeze(-1)
    assert scale.dim() == 4 and mn.dim() == 4
    assert scale.shape == mn.shape
    assert scale.shape[:3] == code.shape[:3]

    B, H, D, n_packs = code.shape
    feat_per_int = 32 // bits
    T = n_packs * feat_per_int
    assert T % group_size == 0
    assert scale.shape[-1] == T // group_size, f"scale groups mismatch: scale G={scale.shape[-1]}, expected {T // group_size}"

    out = torch.empty((B, H, D, T), device=code.device, dtype=dtype)
    total_rows = B * H * D
    grid = (triton.cdiv(total_rows, block_m), triton.cdiv(T, block_t))

    _unpack_dequant_lastdim_kernel[grid](
        code,
        scale,
        mn,
        out,
        total_rows,
        T,
        H,
        D,
        code.stride(0),
        code.stride(1),
        code.stride(2),
        code.stride(3),
        scale.stride(0),
        scale.stride(1),
        scale.stride(2),
        scale.stride(3),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        out.stride(3),
        BITS=bits,
        FEAT_PER_INT=feat_per_int,
        GROUP_SIZE=group_size,
        BLOCK_M=block_m,
        BLOCK_T=block_t,
        num_warps=4,
    )
    return out


@triton.jit
def fp4_dequantize_kernel(
    packed_ptr,
    scale_ptr,
    global_scale_ptr,
    output_ptr,
    N,
    BLOCK_SIZE: tl.constexpr,
    TILE_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    packed_start = pid * TILE_SIZE
    packed_offs = packed_start + tl.arange(0, TILE_SIZE)
    packed_row_idx = packed_offs // (N // 2)
    packed_col_idx = packed_offs % (N // 2)
    packed_mask = packed_col_idx < (N // 2)

    global_scale = tl.load(global_scale_ptr)
    packed_data = tl.load(packed_ptr + packed_offs, mask=packed_mask, other=0)

    x_f16x2_packed = tl.inline_asm_elementwise(
        asm="""
        {
            .reg .b8 byte0, byte1, byte2, byte3;
            mov.b32 {byte0, byte1, byte2, byte3}, $4;
            cvt.rn.f16x2.e2m1x2 $0, byte0;
            cvt.rn.f16x2.e2m1x2 $1, byte1;
            cvt.rn.f16x2.e2m1x2 $2, byte2;
            cvt.rn.f16x2.e2m1x2 $3, byte3;
        }
        """,
        constraints="=r,=r,=r,=r,r",
        args=[packed_data],
        dtype=tl.uint32,
        is_pure=True,
        pack=4,
    )
    val_low = (x_f16x2_packed & 0xFFFF).cast(tl.uint16).cast(tl.float16, bitcast=True).cast(tl.float32)
    val_high = (x_f16x2_packed >> 16).cast(tl.uint16).cast(tl.float16, bitcast=True).cast(tl.float32)

    out_col_low = packed_col_idx * 2
    out_col_high = packed_col_idx * 2 + 1
    out_offs_low = packed_row_idx * N + out_col_low
    out_offs_high = packed_row_idx * N + out_col_high

    block_col_low = out_col_low // BLOCK_SIZE
    block_col_high = out_col_high // BLOCK_SIZE
    scale_offs_low = packed_row_idx * (N // BLOCK_SIZE) + block_col_low
    scale_offs_high = packed_row_idx * (N // BLOCK_SIZE) + block_col_high

    scale_low = tl.load(scale_ptr + scale_offs_low, mask=packed_mask & (out_col_low < N), other=1.0)
    scale_high = tl.load(scale_ptr + scale_offs_high, mask=packed_mask & (out_col_high < N), other=1.0)

    result_low = val_low * scale_low.to(tl.float32) * global_scale
    result_high = val_high * scale_high.to(tl.float32) * global_scale

    out_mask_low = packed_mask & (out_col_low < N)
    out_mask_high = packed_mask & (out_col_high < N)

    tl.store(output_ptr + out_offs_low, result_low, mask=out_mask_low)
    tl.store(output_ptr + out_offs_high, result_high, mask=out_mask_high)


def fp4_dequantize(
    packed_tensor: torch.Tensor,
    scale_tensor: torch.Tensor,
    global_scale: torch.Tensor,
    block_size: int = 16,
    tile_size: int = 128,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    if dtype is None:
        dtype = torch.get_default_dtype()
    packed_n = packed_tensor.shape[-1]
    n = packed_n * 2
    output_shape = list(packed_tensor.shape)
    output_shape[-1] = n
    output = torch.empty(output_shape, dtype=dtype, device=packed_tensor.device)

    def grid(meta):
        return (triton.cdiv(packed_tensor.numel(), meta["TILE_SIZE"]),)

    fp4_dequantize_kernel[grid](
        packed_tensor,
        scale_tensor,
        global_scale,
        output,
        n,
        BLOCK_SIZE=block_size,
        TILE_SIZE=tile_size,
    )
    return output
