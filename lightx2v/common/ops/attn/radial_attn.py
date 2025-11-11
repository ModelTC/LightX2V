import torch
from loguru import logger

try:
    from magi_attention.functional import flex_flash_attn_func as magi_ffa_func
except ImportError:
    magi_ffa_func = None

from lightx2v.utils.registry_factory import ATTN_WEIGHT_REGISTER

from .template import AttnWeightTemplate


def shrinkMaskStrict(mask, block_size=128):
    seqlen = mask.shape[0]
    block_num = seqlen // block_size
    mask = mask[: block_num * block_size, : block_num * block_size].view(block_num, block_size, block_num, block_size)
    col_densities = mask.sum(dim=1) / block_size
    # we want the minimum non-zero column density in the block
    non_zero_densities = col_densities > 0
    high_density_cols = col_densities > 1 / 3
    frac_high_density_cols = high_density_cols.sum(dim=-1) / (non_zero_densities.sum(dim=-1) + 1e-9)
    block_mask = frac_high_density_cols > 0.6
    block_mask[0:0] = True
    block_mask[-1:-1] = True
    return block_mask


def get_window_width(i, j, token_per_frame, sparse_type, num_frame, decay_factor=1, block_size=128, model_type=None):
    assert sparse_type in ["radial"]
    dist = abs(i - j)
    if model_type == "wan":
        if dist < 1:
            return token_per_frame
        if dist == 1:
            return token_per_frame // 2
    elif model_type == "hunyuan":
        if dist <= 1:
            return token_per_frame
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    group = dist.bit_length()
    decay_length = 2 ** token_per_frame.bit_length() / 2**group * decay_factor
    threshold = block_size
    if decay_length >= threshold:
        return decay_length
    else:
        return threshold


def get_diagonal_split_mask(i, j, token_per_frame, sparse_type, device):
    assert sparse_type in ["radial"]
    dist = abs(i - j)
    group = dist.bit_length()
    threshold = 128  # hardcoded threshold for now, which is equal to block-size
    decay_length = 2 ** token_per_frame.bit_length() / 2**group
    if decay_length >= threshold:
        return torch.ones((token_per_frame, token_per_frame), device=device, dtype=torch.bool)

    split_factor = int(threshold / decay_length)
    modular = dist % split_factor
    if modular == 0:
        return torch.ones((token_per_frame, token_per_frame), device=device, dtype=torch.bool)
    else:
        return torch.zeros((token_per_frame, token_per_frame), device=device, dtype=torch.bool)


def gen_log_mask_shrinked(device, s, video_token_num, num_frame, block_size=128, sparse_type="log", decay_factor=0.5, model_type=None):
    """
    A more memory friendly version, we generate the attention mask of each frame pair at a time,
    shrinks it, and stores it into the final result
    """
    final_log_mask = torch.zeros(((s + block_size - 1) // block_size, (s + block_size - 1) // block_size), device=device, dtype=torch.bool)
    token_per_frame = video_token_num // num_frame
    video_text_border = video_token_num // block_size

    col_indices = torch.arange(0, token_per_frame, device=device).view(1, -1)
    row_indices = torch.arange(0, token_per_frame, device=device).view(-1, 1)
    final_log_mask[video_text_border:] = True
    final_log_mask[:, video_text_border:] = True
    for i in range(num_frame):
        for j in range(num_frame):
            local_mask = torch.zeros((token_per_frame, token_per_frame), device=device, dtype=torch.bool)
            if j == 0 and model_type == "wan":  # this is attention sink
                local_mask = torch.ones((token_per_frame, token_per_frame), device=device, dtype=torch.bool)
            else:
                window_width = get_window_width(i, j, token_per_frame, sparse_type, num_frame, decay_factor=decay_factor, block_size=block_size, model_type=model_type)
                local_mask = torch.abs(col_indices - row_indices) <= window_width
                split_mask = get_diagonal_split_mask(i, j, token_per_frame, sparse_type, device)
                local_mask = torch.logical_and(local_mask, split_mask)

            remainder_row = (i * token_per_frame) % block_size
            remainder_col = (j * token_per_frame) % block_size
            # get the padded size
            all_length_row = remainder_row + ((token_per_frame - 1) // block_size + 1) * block_size
            all_length_col = remainder_col + ((token_per_frame - 1) // block_size + 1) * block_size
            padded_local_mask = torch.zeros((all_length_row, all_length_col), device=device, dtype=torch.bool)
            padded_local_mask[remainder_row : remainder_row + token_per_frame, remainder_col : remainder_col + token_per_frame] = local_mask
            # shrink the mask
            block_mask = shrinkMaskStrict(padded_local_mask, block_size=block_size)
            # set the block mask to the final log mask
            block_row_start = (i * token_per_frame) // block_size
            block_col_start = (j * token_per_frame) // block_size
            block_row_end = block_row_start + block_mask.shape[0]
            block_col_end = block_col_start + block_mask.shape[1]
            final_log_mask[block_row_start:block_row_end, block_col_start:block_col_end] = torch.logical_or(final_log_mask[block_row_start:block_row_end, block_col_start:block_col_end], block_mask)
    return final_log_mask


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


@ATTN_WEIGHT_REGISTER("radial_attn")
class RadialAttnWeight(AttnWeightTemplate):
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
        mask = gen_log_mask_shrinked(
            device="cuda", s=seqlen, video_token_num=seqlen, num_frame=cls.attnmap_frame_num, block_size=cls.block_size, sparse_type="radial", decay_factor=0.2, model_type="wan"
        )
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
