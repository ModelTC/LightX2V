import torch
import triton
import triton.language as tl


@triton.jit
def compress_kernel(
    X,
    XM,
    L: tl.constexpr,
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
def compress_nhd_kernel(
    X,
    XM,
    L: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    BLOCK_L: tl.constexpr,
):
    idx_l = tl.program_id(0)
    idx_h = tl.program_id(1)

    offs_l = idx_l * BLOCK_L + tl.arange(0, BLOCK_L)
    offs_d = tl.arange(0, D)
    x = tl.load(
        X + offs_l[:, None] * H * D + idx_h * D + offs_d[None, :],
        mask=offs_l[:, None] < L,
        other=0.0,
    )

    nx = min(BLOCK_L, L - idx_l * BLOCK_L)
    x_mean = tl.sum(x, axis=0, dtype=tl.float32) / nx
    l_blocks = (L + BLOCK_L - 1) // BLOCK_L
    tl.store(
        XM + idx_h * l_blocks * D + idx_l * D + offs_d,
        x_mean.to(XM.dtype.element_ty),
    )


@triton.jit
def compress_centered_nhd_kernel(
    X,
    X_MEAN,
    XM,
    L: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    BLOCK_L: tl.constexpr,
):
    idx_l = tl.program_id(0)
    idx_h = tl.program_id(1)

    offs_l = idx_l * BLOCK_L + tl.arange(0, BLOCK_L)
    offs_d = tl.arange(0, D)
    valid = offs_l[:, None] < L
    x = tl.load(
        X + offs_l[:, None] * H * D + idx_h * D + offs_d[None, :],
        mask=valid,
        other=0.0,
    )
    x_mean = tl.load(X_MEAN + idx_h * D + offs_d)
    centered = (x - x_mean[None, :]).to(X.dtype.element_ty)
    centered = tl.where(valid, centered, 0.0)

    nx = min(BLOCK_L, L - idx_l * BLOCK_L)
    block_mean = tl.sum(centered, axis=0, dtype=tl.float32) / nx
    l_blocks = (L + BLOCK_L - 1) // BLOCK_L
    tl.store(
        XM + idx_h * l_blocks * D + idx_l * D + offs_d,
        block_mean.to(XM.dtype.element_ty),
    )


def mean_pool(x, BLK):
    assert x.is_contiguous()

    B, H, L, D = x.shape
    L_BLOCKS = (L + BLK - 1) // BLK
    x_mean = torch.empty((B, H, L_BLOCKS, D), device=x.device, dtype=x.dtype)

    grid = (L_BLOCKS, B * H)
    compress_kernel[grid](x, x_mean, L, D, BLK)
    return x_mean


def mean_pool_nhd(x, BLK):
    assert x.ndim == 3 and x.is_contiguous()

    L, H, D = x.shape
    L_BLOCKS = (L + BLK - 1) // BLK
    x_mean = torch.empty((1, H, L_BLOCKS, D), device=x.device, dtype=x.dtype)

    compress_nhd_kernel[(L_BLOCKS, H)](x, x_mean, L, H, D, BLK)
    return x_mean


def mean_pool_centered_nhd(x, x_mean, BLK):
    assert x.ndim == 3 and x.is_contiguous()
    assert x_mean.shape == x.shape[1:]

    L, H, D = x.shape
    L_BLOCKS = (L + BLK - 1) // BLK
    block_mean = torch.empty((1, H, L_BLOCKS, D), device=x.device, dtype=x.dtype)

    compress_centered_nhd_kernel[(L_BLOCKS, H)](
        x,
        x_mean,
        block_mean,
        L,
        H,
        D,
        BLK,
    )
    return block_mean


def get_block_lut(q, k, topk_ratio, BLKQ=64, BLKK=64, topk=None):
    arg_k = k - torch.mean(k, dim=-2, keepdim=True)  # smooth-k technique in SageAttention
    pooled_qblocks = mean_pool(q, BLKQ)
    pooled_kblocks = mean_pool(arg_k, BLKK)

    # GQA
    num_q_heads = q.size(1)
    num_kv_heads = k.size(1)
    if num_q_heads != num_kv_heads:
        assert num_q_heads % num_kv_heads == 0, f"Number of Q heads ({num_q_heads}) must be divisible by number of KV heads ({num_kv_heads})"
        repeat_factor = num_q_heads // num_kv_heads
        pooled_kblocks = pooled_kblocks.repeat_interleave(repeat_factor, dim=1)

    if topk is not None and int(topk) > 16:
        pooled_score = pooled_qblocks.float() @ pooled_kblocks.float().transpose(-1, -2)
    else:
        pooled_score = pooled_qblocks @ pooled_kblocks.transpose(-1, -2)

    K = pooled_score.shape[-1]
    # Match the training router: short sequences still retain one key block.
    if topk is None:
        topk = int(topk_ratio * K)
    topk = max(1, min(K, int(topk)))
    lut = torch.topk(pooled_score, topk, dim=-1, sorted=False).indices

    return lut, topk


def _force_local_score_blocks(pooled_score, count):
    count = min(int(count), int(pooled_score.shape[-1]))
    if count <= 0:
        return pooled_score
    q_blocks = int(pooled_score.shape[-2])
    k_blocks = int(pooled_score.shape[-1])
    centers = (torch.arange(q_blocks, device=pooled_score.device) * 2 + 1) * k_blocks // (2 * q_blocks)
    offsets = torch.arange(count, device=pooled_score.device) - count // 2
    local_indices = (centers[:, None] + offsets[None, :]).clamp(0, k_blocks - 1)
    local_indices = local_indices.view(*((1,) * (pooled_score.ndim - 2)), q_blocks, count).expand(*pooled_score.shape[:-2], q_blocks, count)
    return pooled_score.scatter(
        -1,
        local_indices,
        torch.finfo(pooled_score.dtype).max,
    )


def _get_block_lut_nhd(
    q,
    k,
    topk_ratio,
    BLKQ=64,
    BLKK=64,
    topk=None,
    force_local_blocks=0,
):
    pooled_qblocks = mean_pool_nhd(q, BLKQ)
    k_mean = torch.mean(k, dim=0)
    pooled_kblocks = mean_pool_centered_nhd(k, k_mean, BLKK)

    num_q_heads = q.size(1)
    num_kv_heads = k.size(1)
    if num_q_heads != num_kv_heads:
        assert num_q_heads % num_kv_heads == 0, f"Number of Q heads ({num_q_heads}) must be divisible by number of KV heads ({num_kv_heads})"
        pooled_kblocks = pooled_kblocks.repeat_interleave(
            num_q_heads // num_kv_heads,
            dim=1,
        )

    if topk is not None and int(topk) > 16:
        pooled_score = pooled_qblocks.float() @ pooled_kblocks.float().transpose(-1, -2)
    else:
        pooled_score = pooled_qblocks @ pooled_kblocks.transpose(-1, -2)
    pooled_score = _force_local_score_blocks(pooled_score, force_local_blocks)
    k_blocks = pooled_score.shape[-1]
    if topk is None:
        topk = int(topk_ratio * k_blocks)
    topk = max(1, min(k_blocks, int(topk)))
    return torch.topk(pooled_score, topk, dim=-1, sorted=False).indices, topk


def get_block_lut_nhd_uncentered(
    q,
    k,
    topk_ratio,
    BLKQ=64,
    BLKK=64,
    topk=None,
    force_local_blocks=0,
):
    pooled_qblocks = mean_pool_nhd(q, BLKQ)
    pooled_kblocks = mean_pool_nhd(k, BLKK)

    num_q_heads = q.size(1)
    num_kv_heads = k.size(1)
    if num_q_heads != num_kv_heads:
        pooled_kblocks = pooled_kblocks.repeat_interleave(
            num_q_heads // num_kv_heads,
            dim=1,
        )

    if topk is not None and int(topk) > 16:
        pooled_score = pooled_qblocks.float() @ pooled_kblocks.float().transpose(-1, -2)
    else:
        pooled_score = pooled_qblocks @ pooled_kblocks.transpose(-1, -2)
    pooled_score = _force_local_score_blocks(pooled_score, force_local_blocks)
    k_blocks = pooled_score.shape[-1]
    if topk is None:
        topk = int(topk_ratio * k_blocks)
    topk = max(1, min(k_blocks, int(topk)))
    return torch.topk(pooled_score, topk, dim=-1, sorted=False).indices, topk


@torch.library.custom_op(
    "lightx2v::block_lut_nhd",
    mutates_args=(),
    device_types="cuda",
)
def block_lut_nhd(
    q: torch.Tensor,
    k: torch.Tensor,
    topk_ratio: float,
    block_q: int,
    block_k: int,
    topk: int,
) -> torch.Tensor:
    return _get_block_lut_nhd(q, k, topk_ratio, block_q, block_k, topk)[0]


@block_lut_nhd.register_fake
def _block_lut_nhd_fake(
    q: torch.Tensor,
    k: torch.Tensor,
    topk_ratio: float,
    block_q: int,
    block_k: int,
    topk: int,
) -> torch.Tensor:
    q_blocks = (q.shape[0] + block_q - 1) // block_q
    return torch.empty(
        (1, q.shape[1], q_blocks, topk),
        dtype=torch.int64,
        device=q.device,
    )


def get_block_lut_nhd(
    q,
    k,
    topk_ratio,
    BLKQ=64,
    BLKK=64,
    topk=None,
    force_local_blocks=0,
):
    if topk is not None and int(topk) > 16 and int(force_local_blocks) == 0:
        topk = int(topk)
        return block_lut_nhd(q, k, float(topk_ratio), BLKQ, BLKK, topk), topk
    return _get_block_lut_nhd(
        q,
        k,
        topk_ratio,
        BLKQ,
        BLKK,
        topk,
        force_local_blocks,
    )


def get_block_map(q, k, topk_ratio, BLKQ=64, BLKK=64, topk=None):
    lut, topk = get_block_lut(q, k, topk_ratio, BLKQ, BLKK, topk)
    sparse_map = torch.zeros(
        (*lut.shape[:-1], (k.shape[-2] + BLKK - 1) // BLKK),
        device=lut.device,
        dtype=torch.int8,
    )
    sparse_map.scatter_(-1, lut, 1)
    return sparse_map, lut, topk


def get_cuda_arch(device_index):
    major, minor = torch.cuda.get_device_capability(device_index)
    return f"sm{major}{minor}"
