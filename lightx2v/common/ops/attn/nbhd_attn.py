import torch
from loguru import logger

try:
    from magi_attention.functional import flex_flash_attn_func as magi_ffa_func
except ImportError:
    magi_ffa_func = None

from lightx2v.utils.registry_factory import ATTN_WEIGHT_REGISTER

from .template import AttnWeightTemplate


def generate_nbhd_mask(a, block_num, attnmap_frame_num, device="cpu"):
    """
    a : block num per frame
    block_num : block num per col/row
    attnmap_frame_num : total frame num
    """
    i_indices = torch.arange(block_num, device=device).unsqueeze(1)  # [block_num, 1]
    j_indices = torch.arange(block_num, device=device).unsqueeze(0)  # [1, block_num]

    # 1. attention sink frame: j <= a
    mask_sink = j_indices <= a

    # 2. self-attention within the frame
    n = i_indices // a
    mask_self = (j_indices >= n * a) & (j_indices < (n + 1) * a)

    # 3. cross-frame attention
    mask_cross = torch.zeros((block_num, block_num), dtype=torch.bool, device=device)
    for n in range(1, attnmap_frame_num):
        if n == 1:
            width = 1 / 2 * a
        elif n >= 2:
            width = 1 / 8 * a

        mask_1 = (i_indices - j_indices + (n * a + width) >= 0) & (i_indices - j_indices + (n * a - width) < 0)
        mask_2 = (i_indices - j_indices - (n * a - width) > 0) & (i_indices - j_indices - (n * a + width) <= 0)

        mask_cross = mask_cross | mask_1 | mask_2

    # 合并所有mask
    mask = mask_sink | mask_self | mask_cross
    return mask


def generate_qk_ranges(mask, block_size, seqlen):
    indices = torch.nonzero(mask, as_tuple=False)  # shape: [N, 2]

    i_indices = indices[:, 0]  # [N]
    j_indices = indices[:, 1]  # [N]

    q_start = i_indices * block_size  # [N]
    q_end = torch.clamp((i_indices + 1) * block_size, max=seqlen)  # [N]

    k_start = j_indices * block_size  # [N]
    k_end = torch.clamp((j_indices + 1) * block_size, max=seqlen)  # [N]

    q_ranges = torch.stack([q_start, q_end], dim=1)  # [N, 2]
    k_ranges = torch.stack([k_start, k_end], dim=1)  # [N, 2]

    return q_ranges, k_ranges


@ATTN_WEIGHT_REGISTER("nbhd_attn")
class NbhdAttnWeight(AttnWeightTemplate):
    block_size = 128
    seqlen = None
    attnmap_frame_num = None
    q_ranges = None
    k_ranges = None
    attn_type_map = None

    def __init__(self):
        self.config = {}

    @classmethod
    def prepare_mask(cls, seqlen):
        if seqlen == cls.seqlen:
            return
        block_num = (seqlen + cls.block_size - 1) // cls.block_size
        block_num_per_frame = (seqlen // cls.attnmap_frame_num + cls.block_size - 1) // cls.block_size
        mask = generate_nbhd_mask(block_num_per_frame, block_num, cls.attnmap_frame_num, device="cpu")
        q_ranges, k_ranges = generate_qk_ranges(mask, cls.block_size, seqlen)
        attn_type_map = torch.zeros(len(q_ranges), dtype=torch.int32, device="cuda")
        q_ranges = q_ranges.to(torch.int32).to("cuda")
        k_ranges = k_ranges.to(torch.int32).to("cuda")
        cls.seqlen = seqlen
        cls.q_ranges = q_ranges
        cls.k_ranges = k_ranges
        cls.attn_type_map = attn_type_map
        logger.info(f"NbhdAttnWeight Update: seqlen={seqlen}")
        sparsity = 1 - mask.sum().item() / mask.numel()
        logger.info(f"Attention sparsity: {sparsity}")

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
        """
        q: [seqlen, head_num, head_dim]
        k: [seqlen, head_num, head_dim]
        v: [seqlen, head_num, head_dim]
        """
        self.prepare_mask(seqlen=q.shape[0])
        out = magi_ffa_func(
            q,
            k,
            v,
            q_ranges=self.q_ranges,
            k_ranges=self.k_ranges,
            attn_type_map=self.attn_type_map,
            auto_range_merge=True,
        )[0]
        return out.reshape(out.shape[0], -1)
