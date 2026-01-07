from torch.nn import functional as F
from lightx2v.utils.registry_factory import ATTN_WEIGHT_REGISTER
import torch
from einops import rearrange
from flex_block_attn import flex_block_attn_func
from functools import lru_cache
from loguru import logger

from .template import AttnWeightTemplate
from flash_attn_interface import flash_attn_varlen_func as flash_attn_varlen_func_v3
    

@lru_cache(maxsize=4096)
def create_sta_3d_mask_optimize_torch(canvas_thw, tile_thw, kernel_thw):
    canvas_thw = tuple(map(int, canvas_thw.split('_')))
    seq_len = canvas_thw[0] * canvas_thw[1] * canvas_thw[2]
    tile_thw = tuple(map(int, tile_thw.split('_')))
    kernel_thw = tuple(map(int, kernel_thw.split('_')))

    kernel_t, kernel_h, kernel_w = kernel_thw

    block_size = tile_thw[0] * tile_thw[1] * tile_thw[2]
    block_num = int(seq_len / block_size)

    tile_thw_num=(canvas_thw[0] // tile_thw[0], canvas_thw[1] // tile_thw[1], canvas_thw[2] // tile_thw[2])
    
    i_indices = torch.arange(block_num, dtype=torch.long)
    j_indices = torch.arange(block_num, dtype=torch.long)
    
    i_grid, j_grid = torch.meshgrid(i_indices, j_indices, indexing='ij')
    
    q_t_tile = i_grid // (tile_thw_num[1] * tile_thw_num[2])
    q_h_tile = (i_grid % (tile_thw_num[1] * tile_thw_num[2])) // tile_thw_num[2]
    q_w_tile = i_grid % tile_thw_num[2]
    
    kv_t_tile = j_grid // (tile_thw_num[1] * tile_thw_num[2])
    kv_h_tile = (j_grid % (tile_thw_num[1] * tile_thw_num[2])) // tile_thw_num[2]
    kv_w_tile = j_grid % tile_thw_num[2]
    
    kernel_center_t = torch.clamp(q_t_tile, kernel_t // 2, (tile_thw_num[0] - 1) - kernel_t // 2)
    kernel_center_h = torch.clamp(q_h_tile, kernel_h // 2, (tile_thw_num[1] - 1) - kernel_h // 2)
    kernel_center_w = torch.clamp(q_w_tile, kernel_w // 2, (tile_thw_num[2] - 1) - kernel_w // 2)
    
    time_mask = torch.abs(kernel_center_t - kv_t_tile) <= kernel_t // 2
    hori_mask = torch.abs(kernel_center_h - kv_h_tile) <= kernel_h // 2
    vert_mask = torch.abs(kernel_center_w - kv_w_tile) <= kernel_w // 2
    
    block_mask = time_mask & hori_mask & vert_mask
    
    return block_mask


def padding_HND_for_tile(x, x_thw, tile_thw=(3, 8, 8)):
    hd, _, d = x.shape
    t, h, w = x_thw
    tile_t, tile_h, tile_w = tile_thw
    if t % tile_t != 0 or h % tile_h !=0 or w % tile_w != 0:
        x_padded = x.reshape(hd, t, h, w, d)

        pad_t = 0 if t % tile_t == 0 else tile_t - t % tile_t
        if pad_t > 0:
            x_padded = F.pad(x_padded, (0, 0, 0, 0, 0, 0, 0, pad_t))
        pad_h = 0 if h % tile_h == 0 else tile_h - h % tile_h
        if pad_h > 0:
            x_padded = F.pad(x_padded, (0, 0, 0, 0, 0, pad_h))
        pad_w = 0 if w % tile_w == 0 else tile_w - w % tile_w
        if pad_w > 0:
            x_padded = F.pad(x_padded, (0, 0, 0, pad_w))
        return x_padded, True
    return x, False

def padding_HND_for_tile_qkv(q, k, v, x_thw, tile_thw=(3, 8, 8)):
    hd, _, d = q.shape
    t, h, w = x_thw
    tile_t, tile_h, tile_w = tile_thw
    if t % tile_t != 0 or h % tile_h !=0 or w % tile_w != 0:
        q_padded = q.reshape(hd, t, h, w, d)
        k_padded = k.reshape(hd, t, h, w, d)
        v_padded = v.reshape(hd, t, h, w, d)

        pad_t = 0 if t % tile_t == 0 else tile_t - t % tile_t
        if pad_t > 0:
            q_padded = F.pad(q_padded, (0, 0, 0, 0, 0, 0, 0, pad_t))
            k_padded = F.pad(k_padded, (0, 0, 0, 0, 0, 0, 0, pad_t))
            v_padded = F.pad(v_padded, (0, 0, 0, 0, 0, 0, 0, pad_t))
        pad_h = 0 if h % tile_h == 0 else tile_h - h % tile_h
        if pad_h > 0:
            q_padded = F.pad(q_padded, (0, 0, 0, 0, 0, pad_h))
            k_padded = F.pad(k_padded, (0, 0, 0, 0, 0, pad_h))
            v_padded = F.pad(v_padded, (0, 0, 0, 0, 0, pad_h))
        pad_w = 0 if w % tile_w == 0 else tile_w - w % tile_w
        if pad_w > 0:
            q_padded = F.pad(q_padded, (0, 0, 0, pad_w))
            k_padded = F.pad(k_padded, (0, 0, 0, pad_w))
            v_padded = F.pad(v_padded, (0, 0, 0, pad_w))
        q_padded = q_padded.reshape(hd, -1, d)
        k_padded = k_padded.reshape(hd, -1, d)
        v_padded = v_padded.reshape(hd, -1, d)
        return q_padded, k_padded, v_padded, True, (pad_t, pad_h, pad_w)
    return q, k, v, False, (0, 0, 0)

def tile(x, canvas_thw, tile_thw):
    r"""Rearrange tensor into tiles for block-based attention.
    
    Args:
        x: Input tensor with shape (b, head, s, d) where s = t * h * w
        canvas_thw: Tuple of (t, h, w) representing temporal, height, width dimensions
        tile_thw: Tuple of (tile_t, tile_h, tile_w) representing tile dimensions
        sp_size: Spatial size parameter, defaults to 1
    
    Returns:
        Rearranged tensor organized by tiles
    """
    h, s, d = x.shape
    t, h, w = canvas_thw
    assert t * h * w == s, f"t:{t} * h:{h} * w:{w} == s:{s}"

    tile_t_dim, tile_h_dim, tile_w_dim = tile_thw
    n_t = int(t/tile_t_dim)
    n_h = int(h/tile_h_dim)
    n_w = int(w/tile_w_dim)
    return rearrange(x,
                     "h (n_t ts_t n_h ts_h n_w ts_w) d -> h (n_t n_h n_w ts_t ts_h ts_w) d",
                     n_t=n_t,
                     n_h=n_h,
                     n_w=n_w,
                     ts_t=tile_t_dim,
                     ts_h=tile_h_dim,
                     ts_w=tile_w_dim)

def untile(x, canvas_thw, tile_thw):
    r"""Reverse the tiling operation to restore original tensor layout.
    
    Args:
        x: Tiled tensor
        canvas_thw: Tuple of (t, h, w) representing temporal, height, width dimensions
        tile_thw: Tuple of (tile_t, tile_h, tile_w) representing tile dimensions
        sp_size: Spatial size parameter, defaults to 1
    
    Returns:
        Restored tensor with original layout
    """
    t, h, w = canvas_thw

    tile_t_dim, tile_h_dim, tile_w_dim = tile_thw
    n_t = int(t/tile_t_dim)
    n_h = int(h/tile_h_dim)
    n_w = int(w/tile_w_dim)

    return rearrange(x,
                  "h (n_t n_h n_w ts_t ts_h ts_w) d -> h (n_t ts_t n_h ts_h n_w ts_w) d",
                  n_t=n_t,
                  n_h=n_h,
                  n_w=n_w,
                  ts_t=tile_t_dim,
                  ts_h=tile_h_dim,
                  ts_w=tile_w_dim)

def importance_sampling(q, k, topk, threshold=0.0, lambda_=0.9, adaptive_pool=None):
    r"""Select top-k blocks based on importance scores considering both similarity and redundancy.
    
    Args:
        q: Query tensor with shape (B, H, S, D)
        k: Key tensor with shape (B, H, K, D)
        topk: Number of top blocks to select
        threshold: Threshold parameter (not implemented)
        lambda_: Weight factor balancing similarity and redundancy
        adaptive_pool: Adaptive pooling parameter (unused)
    
    Returns:
        top_block_indices: Indices of selected blocks with shape (B, H, S, topk)
    """
    if threshold > 0.0:
        raise NotImplementedError("importance_sampling with threshold not implemented")
    
    q = q / q.norm(dim=-1, keepdim=True)
    k = k / k.norm(dim=-1, keepdim=True)
    gate_similarity = torch.einsum("bhsd,bhkd->bhsk", q, k)
    gate_similarity = (gate_similarity + 1.0) / 2.0
    gate_unique = torch.einsum("bhsd,bhkd->bhsk", k, k)
    gate_unique = (gate_unique + 1.0) / 2.0
    
    B, H, K_num, D = k.shape
    diag_indices = torch.arange(K_num, device=k.device)
    gate_unique[:, :, diag_indices, diag_indices] = torch.nan
    
    mean_redundancy = torch.nanmean(gate_unique, dim=-2, keepdim=True)
    
    importance_scores = lambda_ * gate_similarity - (1 - lambda_) * mean_redundancy
    
    topk = min(topk, importance_scores.size(-1))
    _, top_block_indices = importance_scores.topk(k=topk, dim=-1, sorted=False)
    return top_block_indices


def create_moba_3d_mask(q, 
                        k, 
                        canvas_thw, 
                        topk, 
                        tile_thw,
                        kernel_thw,
                        text_block_num=0, 
                        add_text_mask=False, 
                        threshold=0.0,
                        lambda_=None,
                        mask_share_within_head=True,
                        q_block_avg_pool=True,
                        adaptive_pool=None,
                        sampling_type=None):
    r"""Create MOBA (Mixture of Block Attention) 3D mask for sparse attention.
    
    Args:
        q: Query tensor
        k: Key tensor
        canvas_thw: Canvas dimensions (t, h, w)
        topk: Number of top blocks to attend to
        tile_thw: Tile dimensions
        kernel_thw: Kernel dimensions
        text_block_num: Number of text blocks
        add_text_mask: Whether to add text mask
        threshold: Threshold for similarity sampling
        lambda_: Weight factor for importance sampling
        mask_share_within_head: Whether to share mask across heads
        q_block_avg_pool: Whether to apply average pooling to query blocks
        adaptive_pool: Adaptive pooling size
        sampling_type: Type of sampling ("similarity" or "importance")
    
    Returns:
        moba_3d_mask: 3D attention mask with shape (num_heads, block_num, block_num)
    """
    seq_len = q.size(2)
    block_size = tile_thw[0] * tile_thw[1] * tile_thw[2]
    block_num = int(seq_len / block_size)

    image_shape = canvas_thw
    block_shape = tile_thw
    batch_size, num_heads, seq_len, head_dim = k.shape
    num_blocks_t = (image_shape[0] + block_shape[0] - 1) // block_shape[0]
    num_blocks_h = (image_shape[1] + block_shape[1] - 1) // block_shape[1]
    num_blocks_w = (image_shape[2] + block_shape[2] - 1) // block_shape[2]

    def get_block_avg_feat(x: torch.Tensor, adaptive_pool=None, pooling_type="avg") -> torch.Tensor:
        x_block_means = x.view(batch_size, num_heads, num_blocks_t, num_blocks_h, num_blocks_w,
                                               block_shape[0], block_shape[1], block_shape[2], head_dim)
        if adaptive_pool is not None:
            x_block_means = x_block_means.view(-1, block_shape[0], block_shape[1], block_shape[2], head_dim)
            x_block_means = x_block_means.permute(0, 4, 1, 2, 3)
            if pooling_type == "avg":
                global_avg = torch.nn.functional.adaptive_avg_pool3d(x_block_means, adaptive_pool)
                global_avg = global_avg.permute(0, 2, 3, 4, 1)
                global_avg = global_avg.reshape(batch_size, num_heads, -1, head_dim * adaptive_pool[0] * adaptive_pool[1] * adaptive_pool[2])
                x_block_means = global_avg
            elif pooling_type == "max":
                max_pool = torch.nn.functional.adaptive_max_pool3d(x_block_means, adaptive_pool)
                max_pool = max_pool.permute(0, 2, 3, 4, 1)
                max_pool = max_pool.reshape(batch_size, num_heads, -1, head_dim * adaptive_pool[0] * adaptive_pool[1] * adaptive_pool[2])
                x_block_means = max_pool
            elif pooling_type == "mix":
                global_avg = torch.nn.functional.adaptive_avg_pool3d(x_block_means, (1, 1, 1))
                global_avg = global_avg / (global_avg.norm(dim=1, keepdim=True) + 1e-8)
                global_avg = global_avg.permute(0, 2, 3, 4, 1)
                global_avg = global_avg.reshape(batch_size, num_heads, -1, head_dim)
                
                max_pool = torch.nn.functional.adaptive_max_pool3d(x_block_means, adaptive_pool)
                max_pool = max_pool / (max_pool.norm(dim=1, keepdim=True) + 1e-8)
                max_pool = max_pool.permute(0, 2, 3, 4, 1)
                max_pool = max_pool.reshape(batch_size, num_heads, -1, head_dim * adaptive_pool[0] * adaptive_pool[1] * adaptive_pool[2])
                x_block_means = torch.cat([max_pool, global_avg], dim=-1)
            else:
                raise ValueError(f"pooling_type={pooling_type} is not Supported")
        else:
            x_block_means = x_block_means.mean(dim=(-2, -3, -4)).view(batch_size, num_heads, -1, head_dim)
        return x_block_means

    if (sampling_type == "similarity" and threshold > 0.0) and adaptive_pool is None:
        adaptive_pool = (2,2,2)
            
    k_block_means = get_block_avg_feat(k, adaptive_pool)
    
    if q_block_avg_pool:
        q = get_block_avg_feat(q, adaptive_pool)
    q = q.type(torch.float32)
    k_block_means = k_block_means.type(torch.float32)  # float logit for better gate logit perception

    if mask_share_within_head:
        q = q.mean(dim=1, keepdim=True)
        k_block_means = k_block_means.mean(dim=1, keepdim=True)
    
    if sampling_type == "similarity":
        top_block_indices = similarity_sampling(q, k_block_means, topk, threshold,
                                                 block_num=block_num,
                                                 adaptive_pool=adaptive_pool)
    elif sampling_type == "importance":
        top_block_indices = importance_sampling(q,
                                                k_block_means,
                                                topk, 
                                                threshold,
                                                lambda_=lambda_,
                                                adaptive_pool=adaptive_pool)
    else:
        raise NotImplementedError(f"sampling_type={sampling_type} is not Supported")
    
    q = q.type_as(k)
    # Keep head dimension to dynamically build mask
    assert top_block_indices.size(0) == 1, "top_block_indices batch size must be 1"
    top_block_indices = top_block_indices.squeeze(0)  # Remove batch dimension [h, s_block, topk]
    
    # Create 3D mask template (heads, block_num, block_num)
    gate_idx_mask = torch.zeros(
        (top_block_indices.size(0), block_num, block_num), 
        dtype=torch.bool,
        device=q.device
    )
    
    # Iterate over each attention head using dim=0
    for head_idx in range(top_block_indices.size(0)):
        # Build index matrix for each head
        head_mask = torch.zeros(
            (block_num, block_num), 
            dtype=torch.bool, 
            device=q.device
        )
        # Fill indices for current head
        head_mask.scatter_(
            dim=-1, 
            index=top_block_indices[head_idx], 
            value=True
        )
        gate_idx_mask[head_idx] = head_mask

    if text_block_num > 0:
        pad_block_num = block_num + text_block_num 
        moba_3d_mask = torch.full(
            (gate_idx_mask.size(0), pad_block_num, pad_block_num), 
            False, 
            dtype=torch.bool, 
            device=gate_idx_mask.device
        )
        moba_3d_mask[:, :block_num, :block_num] = gate_idx_mask
        if add_text_mask:
            moba_3d_mask[:, :, -text_block_num:] = True
            moba_3d_mask[:, -text_block_num:, :] = True
    else:
        moba_3d_mask = gate_idx_mask

    return moba_3d_mask



# hyflex is Tencent-Hunyuan/flex-block-attn kernel
# https://github.com/Tencent-Hunyuan/flex-block-attn
@ATTN_WEIGHT_REGISTER("ssta_attn_hyflex")
class SstaAttnWeight(AttnWeightTemplate):
    def __init__(self):
        self.config = {}

    def apply(
        self,
        q,
        k,
        v,
        cu_seqlens_q=None,
        cu_seqlens_kv=None,
        max_seqlen_q=None,
        max_seqlen_kv=None,
        model_cls=None,
    ):
        if len(q.shape) == 3:
            bs = 1
        elif len(q.shape) == 4:
            bs = q.shape[0]
            q = q.reshape(-1, q.shape[-2], q.shape[-1])
            k = k.reshape(-1, k.shape[-2], k.shape[-1])
            v = v.reshape(-1, v.shape[-2], v.shape[-1])
        s, hd, d = q.shape
        q = q.permute(1, 0, 2)  # NHD -> HND
        k = k.permute(1, 0, 2)  # NHD -> HND
        v = v.permute(1, 0, 2)  # NHD -> HND
        x_thw = (18, 48, 80)
        tile_thw = (6, 8, 8)
        kernel_thw = (3, 3, 3)
        block_size = tile_thw[0] * tile_thw[1] * tile_thw[2]
        q, k, v, is_padded, pad_size = padding_HND_for_tile_qkv(q, k, v, x_thw, tile_thw)
        # print(f"is_padded : {is_padded}")
        # print(f"pad_size : {pad_size}")
        pad_t, pad_h, pad_w = pad_size
        x_thw = (x_thw[0] + pad_t, x_thw[1] + pad_h, x_thw[2] + pad_w)
        tile_q = tile(q, x_thw, tile_thw).unsqueeze(0).contiguous()
        tile_k = tile(k, x_thw, tile_thw).unsqueeze(0).contiguous()
        tile_v = tile(v, x_thw, tile_thw).unsqueeze(0).contiguous()
        sta_3d_mask = create_sta_3d_mask_optimize_torch("_".join(map(str, x_thw)), "_".join(map(str, tile_thw)), "_".join(map(str, kernel_thw))).cuda()

        topk = 16
        text_block_num = 0
        threshold = 0.0
        ssta_lambda = 0.7
        mask_share_within_head = False
        adaptive_pool = None
        sampling_type = "importance"
        moba_3d_mask = create_moba_3d_mask(q.unsqueeze(0), k.unsqueeze(0), x_thw, topk, tile_thw, kernel_thw, text_block_num, 
                                       threshold=threshold,
                                       lambda_=ssta_lambda,
                                       mask_share_within_head=mask_share_within_head,
                                       adaptive_pool=adaptive_pool,
                                       sampling_type=sampling_type)

        ssta_3d_mask = torch.logical_or(sta_3d_mask.unsqueeze(0), moba_3d_mask)
        

        sparsity = 1 - ssta_3d_mask.sum().item() / ssta_3d_mask.numel()
        # logger.info(f"STA Attention sparsity: {sparsity}")
        out = flex_block_attn_func(tile_q, tile_k, tile_v, block_size, block_size, ssta_3d_mask.unsqueeze(0)).squeeze(0)


        # tile_q = tile_q.squeeze(0).permute(1, 0, 2).contiguous()
        # tile_k = tile_k.squeeze(0).permute(1, 0, 2).contiguous()
        # tile_v = tile_v.squeeze(0).permute(1, 0, 2).contiguous()
        # # print(f"tile_q shape: {tile_q.shape}")
        # # print(f"tile_k shape: {tile_k.shape}")
        # # print(f"tile_v shape: {tile_v.shape}")
        # cu_seqlens_q = torch.tensor([0, tile_q.shape[0]], dtype=torch.int32).to(q.device, non_blocking=True)
        # cu_seqlens_kv = cu_seqlens_q.clone()
        # max_seqlen_q = tile_q.shape[0]
        # max_seqlen_kv = tile_k.shape[0]
        # out = flash_attn_varlen_func_v3(
        #     tile_q,
        #     tile_k,
        #     tile_v,
        #     cu_seqlens_q,
        #     cu_seqlens_kv,
        #     max_seqlen_q,
        #     max_seqlen_kv,
        # )
        # print(f"out shape: {out.shape}")
        # exit()
        # out = out.permute(1, 0, 2).contiguous()
        out = untile(out, x_thw, tile_thw)
        if is_padded:
            t, h, w = x_thw
            unpad_out = out.reshape(hd, t, h, w, d)
            if pad_t > 0: 
                unpad_out = unpad_out[:, :-pad_t, :, :, :]
            if pad_h > 0:
                unpad_out = unpad_out[:, :, :-pad_h, :, :]
            if pad_w > 0:
                unpad_out = unpad_out[:, :, :, :-pad_w, :]
            out = unpad_out.reshape(hd, -1, d)
        out = out.permute(1, 0, 2).reshape(s, -1)
        return out
