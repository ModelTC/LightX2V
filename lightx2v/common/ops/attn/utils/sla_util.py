import torch
import triton
import triton.language as tl


@triton.jit(do_not_specialize=("L",))
def compress_kernel(
    X,
    XM,
    L,
    D: tl.constexpr,
    BLOCK_L: tl.constexpr,
):
    idx_l = tl.program_id(0)
    idx_bh = tl.program_id(1)

    offs_l = idx_l * BLOCK_L + tl.arange(0, BLOCK_L)
    offs_d = tl.arange(0, D)

    x_offset = idx_bh * L * D
    xm_offset = idx_bh * ((L + BLOCK_L - 1) // BLOCK_L) * D
    # Triton leaves masked lanes undefined when ``other`` is omitted. The
    # lanes participate in the reduction below, so zero the tail padding.
    x = tl.load(
        X + x_offset + offs_l[:, None] * D + offs_d[None, :],
        mask=offs_l[:, None] < L,
        other=0.0,
    )

    nx = min(BLOCK_L, L - idx_l * BLOCK_L)
    x_mean = tl.sum(x, axis=0, dtype=tl.float32) / nx
    tl.store(XM + xm_offset + idx_l * D + offs_d, x_mean.to(XM.dtype.element_ty))


@triton.jit
def centered_compress_kernel(
    X,
    CENTER,
    XM,
    L: tl.constexpr,
    D: tl.constexpr,
    BLOCK_L: tl.constexpr,
):
    idx_l = tl.program_id(0)
    idx_bh = tl.program_id(1)

    offs_l = idx_l * BLOCK_L + tl.arange(0, BLOCK_L)
    offs_d = tl.arange(0, D)
    valid_l = offs_l[:, None] < L

    x_offset = idx_bh * L * D
    xm_offset = idx_bh * ((L + BLOCK_L - 1) // BLOCK_L) * D
    center_offset = idx_bh * D
    x = tl.load(X + x_offset + offs_l[:, None] * D + offs_d[None, :], mask=valid_l)
    center = tl.load(CENTER + center_offset + offs_d)
    centered_x = (x - center[None, :]).to(XM.dtype.element_ty)
    centered_x = tl.where(valid_l, centered_x, 0.0)

    nx = min(BLOCK_L, L - idx_l * BLOCK_L)
    x_mean = tl.sum(centered_x, axis=0, dtype=tl.float32) / nx
    tl.store(XM + xm_offset + idx_l * D + offs_d, x_mean.to(XM.dtype.element_ty))


def mean_pool(x, BLK):
    assert x.is_contiguous()

    B, H, L, D = x.shape
    L_BLOCKS = (L + BLK - 1) // BLK
    x_mean = torch.empty((B, H, L_BLOCKS, D), device=x.device, dtype=x.dtype)

    grid = (L_BLOCKS, B * H)
    compress_kernel[grid](x, x_mean, L, D, BLK)
    return x_mean


def centered_mean_pool(x, center, BLK):
    assert x.is_contiguous()
    assert center.is_contiguous()

    B, H, L, D = x.shape
    L_BLOCKS = (L + BLK - 1) // BLK
    x_mean = torch.empty((B, H, L_BLOCKS, D), device=x.device, dtype=x.dtype)

    grid = (L_BLOCKS, B * H)
    centered_compress_kernel[grid](x, center, x_mean, L, D, BLK)
    return x_mean


def _get_block_lut(pooled_qblocks, pooled_kblocks, topk_ratio):
    # GQA
    num_q_heads = pooled_qblocks.size(1)
    num_kv_heads = pooled_kblocks.size(1)
    if num_q_heads != num_kv_heads:
        assert num_q_heads % num_kv_heads == 0, f"Number of Q heads ({num_q_heads}) must be divisible by number of KV heads ({num_kv_heads})"
        repeat_factor = num_q_heads // num_kv_heads
        pooled_kblocks = pooled_kblocks.repeat_interleave(repeat_factor, dim=1)

    pooled_score = pooled_qblocks @ pooled_kblocks.transpose(-1, -2)

    K = pooled_score.shape[-1]
    # Match the training router: short sequences still retain one key block.
    topk = max(1, min(K, int(topk_ratio * K)))
    lut = torch.topk(pooled_score, topk, dim=-1, sorted=False).indices

    return lut, topk, K


def get_block_lut(q, k, topk_ratio, BLKQ=64, BLKK=64):
    pooled_qblocks = mean_pool(q, BLKQ)
    k_mean = torch.mean(k, dim=-2, keepdim=True)
    pooled_kblocks = centered_mean_pool(k, k_mean, BLKK)
    return _get_block_lut(pooled_qblocks, pooled_kblocks, topk_ratio)


def block_lut_to_ordinal_metadata(lut, num_k_blocks):
    assert lut.dim() == 4
    assert lut.shape[-1] <= num_k_blocks

    full_block_idx = torch.zeros((*lut.shape[:-1], num_k_blocks), dtype=torch.int32, device=lut.device)
    full_block_idx[..., : lut.shape[-1]] = torch.sort(lut, dim=-1).values.to(torch.int32)
    full_block_cnt = torch.full(lut.shape[:-1], lut.shape[-1], dtype=torch.int32, device=lut.device)
    return full_block_idx, full_block_cnt


def get_block_map(q, k, topk_ratio, BLKQ=64, BLKK=64):
    arg_k = k - torch.mean(k, dim=-2, keepdim=True)  # smooth-k technique in SageAttention
    pooled_qblocks = mean_pool(q, BLKQ)
    pooled_kblocks = mean_pool(arg_k, BLKK)
    lut, topk, num_k_blocks = _get_block_lut(pooled_qblocks, pooled_kblocks, topk_ratio)

    sparse_map = torch.zeros((*lut.shape[:-1], num_k_blocks), dtype=torch.int8, device=lut.device)
    sparse_map.scatter_(-1, lut, 1)
    return sparse_map, lut, topk


def get_cuda_arch(device_index):
    major, minor = torch.cuda.get_device_capability(device_index)
    return f"sm{major}{minor}"
