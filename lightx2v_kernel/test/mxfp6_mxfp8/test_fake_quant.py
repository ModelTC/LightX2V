import torch
from torchao.prototype.mx_formats.constants import DTYPE_FP6_E3M2
from torchao.prototype.mx_formats.mx_tensor import to_mx, pack_uint6


def quant2mxfp8(x: torch.Tensor):
    block_size = 32
    m, _ = x.shape
    scale, output = to_mx(x, torch.float8_e4m3fn, block_size=block_size)
    return scale.reshape(m, -1), output


def quant2mxfp6(x: torch.Tensor):
    block_size = 32
    m, _ = x.shape
    scale, output = to_mx(x, DTYPE_FP6_E3M2, block_size=block_size, pack_fp6=False)
    return scale.reshape(m, -1), output


def scale_pad_and_swizzle(scale: torch.Tensor):
    m, s = scale.shape

    # pad the m up to 128, s up to 4
    padded_m = (m + 127) // 128 * 128
    padded_s = (s + 3) // 4 * 4
    padded_scale = torch.empty(padded_m, padded_s, device=scale.device, dtype=scale.dtype)
    padded_scale[:m, :s] = scale

    # swizzle the padded scale
    swizzled_scale = padded_scale.reshape(padded_m // 128, 128, padded_s // 4, 4).reshape(padded_m // 128, 4, 32, padded_s // 4, 4).permute(0, 3, 2, 1, 4)

    return swizzled_scale.reshape(padded_m, padded_s)


###############################################################
# Packing kernel and func
###############################################################

import triton  # noqa: E402
import triton.language as tl  # noqa: E402


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE_IN": 2}, num_warps=1),
        triton.Config({"BLOCK_SIZE_IN": 4}, num_warps=1),
        triton.Config({"BLOCK_SIZE_IN": 8}, num_warps=1),
        triton.Config({"BLOCK_SIZE_IN": 16}, num_warps=1),
    ],
    key=["n_mx_blocks"],
)
@triton.jit
def triton_pack_uint6_kernel(
    input_ptr,
    output_ptr,
    n_mx_blocks,
    MX_BLOCK_SIZE: tl.constexpr,
    PACKED_MX_BLOCK_SIZE: tl.constexpr,
    BLOCK_SIZE_IN: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE_IN

    # input_ptr is shape [n_mx_blocks, MX_BLOCK_SIZE]
    # Load BLOCK_SIZE rows of input_ptr
    offsets_rows = block_start + tl.arange(0, BLOCK_SIZE_IN)
    offsets_cols = tl.arange(0, MX_BLOCK_SIZE // 4)
    offsets = offsets_rows[:, None] * MX_BLOCK_SIZE + (4 * offsets_cols[None, :])
    mask = (offsets_rows[:, None] < n_mx_blocks) & (offsets_cols[None, :] < MX_BLOCK_SIZE // 4)

    # x is shape [BLOCK_SIZE, MX_BLOCK_SIZE]
    x_0 = tl.load(input_ptr + offsets, mask=mask)
    x_1 = tl.load(input_ptr + offsets + 1, mask=mask)
    x_2 = tl.load(input_ptr + offsets + 2, mask=mask)
    x_3 = tl.load(input_ptr + offsets + 3, mask=mask)

    # 4个fp6 a b c d. a:[a5 a4 a3 a2 a1 a0], b..., c..., d...
    # 3个unint8 pack0 pack1 pack2
    # cutlass需要的：
    # packed0: [b1 b0][a5 a4 a3 a2 a1 a0]
    # packed1: [c3 c2 c1 c0][b5 b4 b3 b2]
    # packed2: [d5 d4 d3 d2 d1 d0][c5 c4]
    bits_packed0 = (x_1 << 6) | x_0
    bits_packed1 = (x_2 << 4) | (x_1 >> 2)
    bits_packed2 = (x_3 << 2) | (x_2 >> 4)

    # Store values in a uint8 tensor of length `3 * MX_BLOCK_SIZE / 4`
    offsets_out_4_a = offsets_rows[:, None] * PACKED_MX_BLOCK_SIZE + 3 * offsets_cols[None, :]
    offsets_out_4_b = offsets_rows[:, None] * PACKED_MX_BLOCK_SIZE + 3 * offsets_cols[None, :] + 1
    offsets_out_2 = offsets_rows[:, None] * PACKED_MX_BLOCK_SIZE + 3 * offsets_cols[None, :] + 2

    # Store into output tensor
    tl.store(
        output_ptr + offsets_out_4_a,
        bits_packed0,
        mask=mask,
    )

    tl.store(
        output_ptr + offsets_out_4_b,
        bits_packed1,
        mask=mask,
    )

    tl.store(
        output_ptr + offsets_out_2,
        bits_packed2,
        mask=mask,
    )


def pack_uint6(uint8_data: torch.Tensor) -> torch.Tensor:
    # ensure input data is contiguous before passing to kernel
    assert uint8_data.is_contiguous()

    # tensor should already be of shape [..., mx_block_size]
    mx_block_size = uint8_data.shape[-1]
    assert mx_block_size % 4 == 0

    # effective mx block size since we're packing 2 fp4 into 1 uint8
    packed_mx_block_size = 3 * mx_block_size // 4
    packed_shape = [uint8_data.shape[0], packed_mx_block_size]
    n_mx_blocks = uint8_data.numel() // mx_block_size

    grid = lambda meta: (triton.cdiv(n_mx_blocks, meta["BLOCK_SIZE_IN"]),)  # noqa: E731

    # contiguous uint8 container in which we can store the unpacked tensor
    packed_uint8_data = torch.empty(packed_shape, dtype=torch.uint8, device=uint8_data.device)

    triton_pack_uint6_kernel[grid](
        uint8_data,
        packed_uint8_data,
        n_mx_blocks,
        MX_BLOCK_SIZE=mx_block_size,
        PACKED_MX_BLOCK_SIZE=packed_mx_block_size,
    )

    return packed_uint8_data


M = [257, 512, 1024, 13325, 32130, 32760]  # , 75348
N = [1536, 5120, 8960]  # , 13824
K = [128, 256, 512, 1024, 2048, 4096]  # , 13824


for m in M:
    for n in N:
        for k in K:
            x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
            w = torch.randn(n, k, device="cuda", dtype=torch.bfloat16)
            # excute quant
            x_scale, x_quant = quant2mxfp8(x)
            w_scale, w_quant = quant2mxfp6(w)

            # pack fp6 for cutlass
            w_quant_packed = pack_uint6(w_quant.reshape(-1, 32))

            # pad and swizzle scale
            padded_and_swizzled_x_scale = scale_pad_and_swizzle(x_scale)
            padded_and_swizzled_w_scale = scale_pad_and_swizzle(w_scale)

            # ref mm result
            ref_mm = torch.nn.functional.linear(x, w).to(torch.bfloat16)

            # custom scaled mm
            from lightx2v_kernel.gemm import cutlass_scaled_mxfp6_mxfp8_mm

            alpha = torch.tensor(1.0, device="cuda", dtype=torch.float32)
            bias = None
            x_quant = x_quant.reshape(m, k).view(torch.uint8)
            w_quant_packed = w_quant_packed.reshape(n, 3 * k // 4)
            custom_mm = cutlass_scaled_mxfp6_mxfp8_mm(x_quant, w_quant_packed, padded_and_swizzled_x_scale, padded_and_swizzled_w_scale, alpha, bias)

            # cal snr
            from lightx2v_kernel.utils import error

            print(f"m: {m}, n: {n}, k: {k}, error: {error(ref_mm, custom_mm)}")

            # cal cos
            cos_sim = torch.nn.functional.cosine_similarity(ref_mm.flatten(), custom_mm.flatten(), dim=0)
            print(f"m: {m}, n: {n}, k: {k}, cos_sim: {cos_sim}")
