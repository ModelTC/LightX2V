import math

import torch
import torch.nn.functional as F
from loguru import logger

from lightx2v.utils.registry_factory import ATTN_WEIGHT_REGISTER

from .template import AttnWeightTemplate

try:
    from magi_attention.functional import flex_flash_attn_func as magi_ffa_func
except ImportError:
    magi_ffa_func = None


flash_attn_varlen_func = None
try:
    from flash_attn.flash_attn_interface import flash_attn_varlen_func as _func

    flash_attn_varlen_func = _func
except ImportError:
    logger.info("flash_attn_varlen_func not found, please install flash_attn2 first")


try:
    from flash_attn_interface import flash_attn_varlen_func as _func

    flash_attn_varlen_func = _func
except ImportError:
    logger.info("flash_attn_varlen_func_v3 not found, please install flash_attn3 first")


@ATTN_WEIGHT_REGISTER("draft_attn")
class DraftAttnWeight(AttnWeightTemplate):
    sparsity_ratio = 0.75
    reorg_idx_dict = {}
    restore_idx_dict = {}
    bucket_offsets_dict = {}

    def __init__(self):
        self.config = {}

    @staticmethod
    def build_grid_gather_index_and_bucket_fast(H, W, pool_h, pool_w, seqlen):
        Gh = (H + pool_h - 1) // pool_h
        Gw = (W + pool_w - 1) // pool_w

        # Single frame
        gather_single = []
        bucket_sizes_single = []

        for gh in range(Gh):
            h0 = gh * pool_h
            h1 = min(h0 + pool_h, H)
            block_h = h1 - h0

            for gw in range(Gw):
                w0 = gw * pool_w
                w1 = min(w0 + pool_w, W)
                block_w = w1 - w0

                # bucket size
                bucket_size = block_h * block_w
                bucket_sizes_single.append(bucket_size)

                # gather index
                for i in range(h0, h1):
                    row_base = i * W
                    for j in range(w0, w1):
                        gather_single.append(row_base + j)

        bucket_sizes = []
        bucket_offsets = [0]
        running = 0
        # bucket + offsets
        for sz in bucket_sizes_single:
            bucket_sizes.append(sz)
            running += sz
            bucket_offsets.append(running)

        frame_num = seqlen // (H * W)
        gather_index = []
        for f in range(frame_num):
            frame_base = f * H * W
            # index
            gather_index.extend(idx + frame_base for idx in gather_single)

        return gather_index, bucket_sizes, bucket_offsets

    @classmethod
    @torch.compiler.disable
    def prepare_reorg_idx_and_bucket_offset(cls, seqlen, frame_h, frame_w, pool_h, pool_w, device):
        if (seqlen, frame_h, frame_w) in cls.reorg_idx_dict:
            return
        reorg_idx, bucket_sizes, bucket_offsets = cls.build_grid_gather_index_and_bucket_fast(
            H=frame_h,
            W=frame_w,
            pool_h=pool_h,
            pool_w=pool_w,
            seqlen=seqlen,
        )
        reorg_idx = torch.tensor(reorg_idx, dtype=torch.long, device=device)
        restore_idx = torch.empty_like(reorg_idx)
        restore_idx[reorg_idx] = torch.arange(reorg_idx.numel(), device=device)
        cls.reorg_idx_dict[(seqlen, frame_h, frame_w)] = reorg_idx
        cls.restore_idx_dict[(seqlen, frame_h, frame_w)] = restore_idx
        cls.bucket_offsets_dict[(seqlen, frame_h, frame_w)] = torch.tensor(bucket_offsets, dtype=torch.int32, device=device)
        logger.info(f"DraftAttnWeight: reorg_idx len: {len(reorg_idx)}")
        logger.info(f"DraftAttnWeight: bucket_sizes: {bucket_sizes}")
        logger.info(f"DraftAttnWeight: bucket_offsets: {bucket_offsets}")
        logger.info(f"DraftAttnWeight: using sparsity ratio {cls.sparsity_ratio}")

    def sample_qk_attention_2d(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        frame_h: int,
        frame_w: int,
        pool_h: int,
        pool_w: int,
    ):
        L, H, D = q.shape
        frame_tokens = frame_h * frame_w
        assert L % frame_tokens == 0, "L must be multiple of frame_h*frame_w"
        num_frames = L // frame_tokens

        # 1) Slice out the video part and reshape to frames:
        #    [L, H, D] → [num_frames, frame_h, frame_w, H, D]
        q_vid = q.view(num_frames, frame_h, frame_w, H, D)
        k_vid = k.view(num_frames, frame_h, frame_w, H, D)

        # 2) Permute & merge (num_frames, H*D) into channel dim:
        #    → [num_frames, H*D, frame_h, frame_w]
        q_vid = q_vid.permute(0, 3, 4, 1, 2).reshape(num_frames, H * D, frame_h, frame_w)
        k_vid = k_vid.permute(0, 3, 4, 1, 2).reshape(num_frames, H * D, frame_h, frame_w)

        # 3) 2D avg‐pool each frame (ceil_mode ensures we cover the edges):
        #    → [num_frames, H*D, S_h, S_w]
        q_pooled = F.avg_pool2d(q_vid, kernel_size=(pool_h, pool_w), stride=(pool_h, pool_w), ceil_mode=True)
        k_pooled = F.avg_pool2d(k_vid, kernel_size=(pool_h, pool_w), stride=(pool_h, pool_w), ceil_mode=True)

        S_h, S_w = q_pooled.shape[-2:]
        S = num_frames * S_h * S_w

        # 4) Un‐merge channel back to [S, H, D]:
        #    → [num_frames, H, D, S_h, S_w] → [S, H, D]
        def unmerge(x):
            x = x.reshape(num_frames, H, D, S_h, S_w)
            return x.permute(0, 3, 4, 1, 2).reshape(S, H, D)

        sampled_q = unmerge(q_pooled)
        sampled_k = unmerge(k_pooled)

        # 5) Compute per‐head scaled dot‐prod attention:
        #    [S, H, D] → [H, S, D]
        q_heads = sampled_q.permute(1, 0, 2)
        k_heads = sampled_k.permute(1, 0, 2)

        # → [H, S, S]
        scores = torch.einsum("hld,hmd->hlm", q_heads, k_heads) / math.sqrt(D)
        attn_map = torch.softmax(scores, dim=-1)

        return attn_map

    def attention_percentile_mask_headwise(self, attn_map: torch.Tensor, r: float) -> torch.BoolTensor:
        """
        Build a mask per head so that each head keeps its top-r fraction of entries as True.

        Args:
        attn_map: Tensor of shape [H, S, S], attention scores (e.g. after softmax).
        r: float in (0,1), fraction of entries *per head* to keep True.

        Returns:
        mask: BoolTensor of shape [H, S, S], where for each head h,
                mask[h].float().mean() ≈ r.
        """
        H, S, _ = attn_map.shape
        flat = attn_map.reshape(H, -1)  # [H, S*S]
        n = flat.shape[1]
        k = int((1.0 - r) * n)

        if k == 0:
            return torch.ones_like(attn_map, dtype=torch.bool)
        if k >= n:
            return torch.zeros_like(attn_map, dtype=torch.bool)

        # Calculate threshold for each head independently
        thresholds = torch.kthvalue(flat, k, dim=1).values  # [H]
        mask = attn_map >= thresholds[:, None, None]  # broadcasting

        return mask

    def apply(
        self,
        q,
        k,
        v,
        cu_seqlens_q=None,
        cu_seqlens_kv=None,
        max_seqlen_q=None,
        max_seqlen_kv=None,
        frame_h=32,
        frame_w=48,
        block_idx=0,
    ):
        if block_idx < 1:
            out = flash_attn_varlen_func(
                q,
                k,
                v,
                cu_seqlens_q,
                cu_seqlens_kv,
                max_seqlen_q,
                max_seqlen_kv,
            )
            return out.reshape(out.shape[0], -1)

        seqlen, head_num, head_dim = q.shape
        frame_size = frame_h * frame_w
        num_frames = seqlen // frame_size

        pool_h, pool_w = (8, 16) if frame_h < frame_w else (16, 8)

        self.prepare_reorg_idx_and_bucket_offset(
            seqlen=seqlen,
            frame_h=frame_h,
            frame_w=frame_w,
            pool_h=pool_h,
            pool_w=pool_w,
            device=q.device,
        )

        attn = self.sample_qk_attention_2d(
            q,
            k,
            frame_h=frame_h,
            frame_w=frame_w,
            pool_h=pool_h,
            pool_w=pool_w,
        )

        mask = self.attention_percentile_mask_headwise(attn, 1 - self.sparsity_ratio)

        # sink mask
        mask_size_pre_frame = mask.shape[1] // num_frames
        mask[:, :, :mask_size_pre_frame] = True

        # diagonal mask
        for f in range(num_frames):
            start = f * mask_size_pre_frame
            end = (f + 1) * mask_size_pre_frame
            mask[:, start:end, start:end] = True

        h_indices, i_indices, j_indices = torch.nonzero(mask, as_tuple=True)  # [N, 3] -> [head, i, j]
        bucket_offsets = self.bucket_offsets_dict[(seqlen, frame_h, frame_w)]

        base_offset = h_indices * seqlen

        q_frame_base = (i_indices // mask_size_pre_frame) * frame_size
        q_bucket_idx = i_indices % mask_size_pre_frame
        q_start = base_offset + q_frame_base + bucket_offsets[q_bucket_idx]
        q_end = base_offset + q_frame_base + bucket_offsets[q_bucket_idx + 1]

        k_frame_base = (j_indices // mask_size_pre_frame) * frame_size
        k_bucket_idx = j_indices % mask_size_pre_frame
        k_start = base_offset + k_frame_base + bucket_offsets[k_bucket_idx]
        k_end = base_offset + k_frame_base + bucket_offsets[k_bucket_idx + 1]

        q_ranges = torch.stack([q_start, q_end], dim=1).to(dtype=torch.int32)
        k_ranges = torch.stack([k_start, k_end], dim=1).to(dtype=torch.int32)
        attn_type_map = torch.zeros(len(q_ranges), dtype=torch.int32, device=q.device)

        reorg_idx = self.reorg_idx_dict[(seqlen, frame_h, frame_w)]
        q = q[reorg_idx]
        k = k[reorg_idx]
        v = v[reorg_idx]

        q = q.permute(1, 0, 2).reshape(head_num * seqlen, 1, head_dim)
        k = k.permute(1, 0, 2).reshape(head_num * seqlen, 1, head_dim)
        v = v.permute(1, 0, 2).reshape(head_num * seqlen, 1, head_dim)

        out = magi_ffa_func(
            q,
            k,
            v,
            q_ranges=q_ranges,
            k_ranges=k_ranges,
            attn_type_map=attn_type_map,
            auto_range_merge=True,
        )[0]

        out = out.reshape(head_num, seqlen, head_dim).permute(1, 0, 2)

        restore_idx = self.restore_idx_dict[(seqlen, frame_h, frame_w)]
        out = out[restore_idx]

        return out.reshape(out.shape[0], -1)
