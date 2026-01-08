import torch
from loguru import logger

try:
    from magi_attention.functional import flex_flash_attn_func as magi_ffa_func
except ImportError:
    magi_ffa_func = None

try:
    import flashinfer
except ImportError:
    flashinfer = None

from lightx2v.utils.registry_factory import ATTN_WEIGHT_REGISTER

from .template import AttnWeightTemplate


def generate_nbhd_mask(a, block_num, attnmap_frame_num, coefficient=[1.0, 0.5, 0.056], min_width=1.0, device="cpu"):
    """
    a : block num per frame
    block_num : block num per col/row
    attnmap_frame_num : total frame num
    """
    i_indices = torch.arange(block_num, device=device).unsqueeze(1)  # [block_num, 1]
    j_indices = torch.arange(block_num, device=device).unsqueeze(0)  # [1, block_num]

    assert len(coefficient) <= attnmap_frame_num, f"coefficient length {len(coefficient)} should <= attnmap_frame_num {attnmap_frame_num}"
    width_list = [max(min_width, coefficient[i] * a) for i in range(len(coefficient))] + [min_width] * (attnmap_frame_num - len(coefficient))
    logger.info(f"nbhd_attn width_list: {width_list}, len={len(width_list)}")

    # attention sink frame: j <= a
    mask_sink = j_indices <= a

    mask_sparse = torch.zeros((block_num, block_num), dtype=torch.bool, device=device)
    for interval in range(0, attnmap_frame_num):
        n = i_indices // a
        mask_sparse_base_1 = (j_indices >= (n + interval) * a) & (j_indices <= (n + interval + 1) * a)
        n = j_indices // a
        mask_sparse_base_2 = (i_indices >= (n + interval) * a) & (i_indices <= (n + interval + 1) * a)

        width = width_list[interval]

        mask_1 = mask_sparse_base_1 & (i_indices - j_indices + (interval * a + width) >= 0) & (i_indices - j_indices + (interval * a - width) <= 0)
        mask_2 = mask_sparse_base_2 & (i_indices - j_indices - (interval * a - width) >= 0) & (i_indices - j_indices - (interval * a + width) <= 0)

        mask_sparse = mask_sparse | mask_1 | mask_2

    mask = mask_sink | mask_sparse
    return mask


def generate_qk_ranges(mask, q_block_size, k_block_size, seqlen):
    # mask: [H, Q_block_num, K_block_num]
    h_indices, i_indices, j_indices = torch.nonzero(mask, as_tuple=True)

    base_offset = h_indices * seqlen

    q_start = base_offset + i_indices * q_block_size
    q_end = base_offset + torch.clamp((i_indices + 1) * q_block_size, max=seqlen)

    k_start = base_offset + j_indices * k_block_size
    k_end = base_offset + torch.clamp((j_indices + 1) * k_block_size, max=seqlen)

    q_ranges = torch.stack([q_start, q_end], dim=1)
    k_ranges = torch.stack([k_start, k_end], dim=1)

    return q_ranges, k_ranges


@ATTN_WEIGHT_REGISTER("nbhd_attn")
class NbhdAttnWeight(AttnWeightTemplate):
    block_size = 128
    seqlen = None
    attnmap_frame_num = None
    q_ranges = None
    k_ranges = None
    attn_type_map = None
    coefficient = [1.0, 0.5, 0.056]
    min_width = 1.0

    def __init__(self):
        self.config = {}

    @classmethod
    @torch.compiler.disable
    def prepare_mask(cls, seqlen, head_num):
        if seqlen == cls.seqlen:
            return
        block_num = (seqlen + cls.block_size - 1) // cls.block_size
        block_num_per_frame = seqlen / cls.attnmap_frame_num / cls.block_size
        mask = generate_nbhd_mask(block_num_per_frame, block_num, cls.attnmap_frame_num, coefficient=cls.coefficient, min_width=cls.min_width, device="cpu")
        repeat_mask = mask.unsqueeze(0).repeat(head_num, 1, 1)
        q_ranges, k_ranges = generate_qk_ranges(repeat_mask, cls.block_size, cls.block_size, seqlen)
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
        seqlen, head_num, head_dim = q.shape
        self.prepare_mask(seqlen=seqlen, head_num=head_num)

        q = q.permute(1, 0, 2).reshape(head_num * seqlen, 1, head_dim)
        k = k.permute(1, 0, 2).reshape(head_num * seqlen, 1, head_dim)
        v = v.permute(1, 0, 2).reshape(head_num * seqlen, 1, head_dim)

        out = magi_ffa_func(
            q,
            k,
            v,
            q_ranges=self.q_ranges,
            k_ranges=self.k_ranges,
            attn_type_map=self.attn_type_map,
            auto_range_merge=True,
        )[0]

        out = out.reshape(head_num, seqlen, head_dim).permute(1, 0, 2)

        return out.reshape(out.shape[0], -1)


@ATTN_WEIGHT_REGISTER("nbhd_attn_flashinfer")
class NbhdAttnWeightFlashInfer(AttnWeightTemplate):
    block_size = 128
    seqlen = None
    attnmap_frame_num = None
    coefficient = [1.0, 0.5, 0.056]
    min_width = 1.0
    sparse_wrapper = None

    def __init__(self):
        self.config = {}

    @classmethod
    @torch.compiler.disable
    def prepare_mask(cls, seqlen, head_num, head_dim):
        if seqlen == cls.seqlen:
            return
        block_num = (seqlen + cls.block_size - 1) // cls.block_size
        block_num_per_frame = seqlen / cls.attnmap_frame_num / cls.block_size
        mask = generate_nbhd_mask(block_num_per_frame, block_num, cls.attnmap_frame_num, coefficient=cls.coefficient, min_width=cls.min_width, device="cpu")
        mask = mask.unsqueeze(0).repeat(head_num, 1, 1)
        block_rowcol_size = torch.ones(block_num, dtype=torch.int32) * cls.block_size
        block_rowcol_size[-1] = seqlen - cls.block_size * (block_num - 1)
        block_rowcol_size = block_rowcol_size.unsqueeze(0).repeat(head_num, 1)
        float_workspace_buffer = torch.empty(1024 * 1024 * 1024, dtype=torch.uint8, device="cuda:0")
        cls.sparse_wrapper = flashinfer.sparse.VariableBlockSparseAttentionWrapper(float_workspace_buffer, backend="fa2")
        cls.sparse_wrapper.plan(
            block_mask_map=mask,
            block_row_sz=block_rowcol_size,
            block_col_sz=block_rowcol_size,
            num_qo_heads=head_num,
            num_kv_heads=head_num,
            head_dim=head_dim,
            q_data_type=torch.bfloat16,
        )
        cls.seqlen = seqlen
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
        self.prepare_mask(seqlen=q.shape[0], head_num=q.shape[1], head_dim=q.shape[2])
        q = q.transpose(0, 1)
        k = k.transpose(0, 1)
        v = v.transpose(0, 1)
        out = self.sparse_wrapper.run(q, k, v)
        out = out.transpose(0, 1)
        return out.reshape(out.shape[0], -1)
