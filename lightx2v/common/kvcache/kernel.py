import torch
import triton
import triton.language as tl
import numpy as np
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
def _pack_along_last_dim(
	bits: tl.constexpr,
	intensor_ptr,
	code_ptr,
	N,
	num_feats: tl.constexpr,
	feat_per_int: tl.constexpr,
	BLOCK_SIZE_N: tl.constexpr
):
	num_int_per_y_dim = num_feats // feat_per_int
	bid = tl.program_id(axis=0)
	yid = tl.program_id(axis=1)
	offs_N = bid * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
	block_start = intensor_ptr + offs_N * num_feats + yid * feat_per_int # offset of the first element at current tile
	packed = tl.zeros((BLOCK_SIZE_N,), dtype=tl.int32)
	for i in range(feat_per_int):
		ptr = block_start + i
		element = tl.load(ptr, mask=offs_N<N, other=0.)
		element = element << (i * bits)
		# Combine the value using bitwise OR
		packed = packed | element
	tl.store(code_ptr + offs_N * num_int_per_y_dim + yid, packed, mask=offs_N < N)

@triton.jit
def _minmax_along_last_dim(
	x_ptr,
	mn_ptr, mx_ptr,
	total_elements: tl.constexpr, 
	N: tl.constexpr,
	num_groups: tl.constexpr, 
	group_size: tl.constexpr,
	BLOCK_SIZE_N: tl.constexpr
):
	bid = tl.program_id(axis=0)
	offsets_b = bid * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
	offsets = offsets_b[:, None] * group_size + tl.arange(0, group_size)[None, :]
	mask = offsets < total_elements
	x = tl.load(x_ptr + offsets, mask=mask)
	mx_val = tl.max(x, axis=1)
	mn_val = tl.min(x, axis=1)
	# tl.device_print('shape', mn_val[:, None].shape)
	tl.store(mn_ptr+offsets_b, mn_val, mask=offsets_b<N*num_groups)
	tl.store(mx_ptr+offsets_b, mx_val, mask=offsets_b<N*num_groups)


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
	grid = lambda meta: (triton.cdiv(data.shape[0]*data.shape[1], BLOCK_SIZE_N),)
	with torch.cuda.device(data.device):
		_minmax_along_last_dim[grid](data, mn, mx,
							 data.numel(), data.shape[0], num_groups, group_size,
							 BLOCK_SIZE_N=BLOCK_SIZE_N, num_warps=8) 
	# mn = torch.min(data, dim=-1, keepdim=True)[0].squeeze(-1)
	# mx = torch.max(data, dim=-1, keepdim=True)[0].squeeze(-1)
	scale = (mx - mn) / (2 ** bit - 1)
	data = data - mn.unsqueeze(-1)
	data.div_(scale.unsqueeze(-1))
	data = data.clamp_(0, 2 ** bit - 1).round_().to(torch.int32)
	data = data.view(-1, T)
	feat_per_int = 32 // bit
	packshape = (np.prod(shape[:-1]), shape[-1] // feat_per_int,)
	code = torch.zeros(*packshape, device=data.device, dtype=torch.int32)
	grid = lambda meta: (triton.cdiv(data.shape[0], BLOCK_SIZE_N), data.shape[1] // feat_per_int,)
	with torch.cuda.device(data.device):
		_pack_along_last_dim[grid](bit, data, code, data.shape[0], 
								data.shape[1], feat_per_int, 
								BLOCK_SIZE_N=BLOCK_SIZE_N, 
								num_warps=8)
	return code.view(B, nh, D, -1), scale.reshape(scale_mn_shape), mn.reshape(scale_mn_shape)

def unpack_tensor(v_code: torch.FloatTensor, 
				  bits: int, 
				  pack_dim: int):
	assert bits in [2,4,8]
	shape = v_code.shape
	feat_per_int = 32 // bits
	new_shape = shape[:pack_dim] + (shape[pack_dim] * feat_per_int,) + shape[pack_dim+1:]
	unpacked_v_code = torch.zeros(new_shape, dtype=torch.int8, device=v_code.device)
	i = torch.arange(new_shape[pack_dim], device=v_code.device) // feat_per_int
	j = torch.arange(new_shape[pack_dim], device=v_code.device) % feat_per_int
	num = 0xFF >> (8 - bits)
	packed_indices = [slice(None)] * len(new_shape)
	packed_indices[pack_dim] = i
	if pack_dim == 2:
		unpacked_v_code = ((v_code[packed_indices] >> (j * bits)[None, None, :, None]).to(torch.int16)) & num
	elif pack_dim == 3:
		unpacked_v_code = ((v_code[packed_indices] >> (j * bits)).to(torch.int16)) & num
	else:
		raise NotImplementedError
	return unpacked_v_code

def unpack_and_dequant_cache(v_code: torch.FloatTensor, 
							  scale: torch.FloatTensor, 
							  mn: torch.FloatTensor,
							  group_size: int, 
							  bits: int,
							  ):
	assert bits in [2, 4, 8]
	assert len(v_code.shape) == 4
	data = unpack_tensor(v_code, bits, pack_dim=3)
	shape = data.shape
	num_groups = shape[-1] // group_size
	data = data.view(shape[:-1] + (num_groups, group_size,))
	data = data.to(torch.float16)
	data = data * scale + mn 
	return data.view(shape)
