from abc import ABC, abstractmethod

import torch
import torch.nn.functional as F
from loguru import logger

from lightx2v.utils.registry_factory import SPARSE_MASK_GENERATOR_REGISTER

from .nbhd_attn import generate_nbhd_mask
from .svg_attn import diagonal_band_mask_from_sparsity, get_attention_mask, wan_hidden_states_placement, wan_sparse_head_placement
from .utils.sla_util import get_block_map


class GeneralMaskGenerator(ABC):
    def __init__(self, q_block_size=128, k_block_size=128, sparse_setting={}, attnmap_frame_num=None):
        self.sparse_setting = sparse_setting
        self.q_block_size = q_block_size
        self.k_block_size = k_block_size
        self.attnmap_frame_num = attnmap_frame_num

    @abstractmethod
    def __call__(self, q, k):
        pass

    def reorg(self, q, k, v):
        return q, k, v

    def restore(self, out):
        return out


@SPARSE_MASK_GENERATOR_REGISTER("sla_mask_generator")
class SlaMaskGenerator(GeneralMaskGenerator):
    def __init__(self, q_block_size=128, k_block_size=128, sparse_setting={}, attnmap_frame_num=None):
        super().__init__(q_block_size, k_block_size, sparse_setting, attnmap_frame_num)
        sparsity_ratio = self.sparse_setting.get("sla_sparsity_ratio", 0.8)
        self.topk_ratio = 1 - sparsity_ratio

    def __call__(self, q, k):
        # (L, H, D) -> (B, H, L, D)
        q = q.unsqueeze(0).transpose(1, 2).contiguous()
        k = k.unsqueeze(0).transpose(1, 2).contiguous()
        sparse_map, lut, topk = get_block_map(q, k, topk_ratio=self.topk_ratio, BLKQ=self.q_block_size, BLKK=self.k_block_size)
        # return: [H, Q_block_num, K_block_num]
        return sparse_map[0]


@SPARSE_MASK_GENERATOR_REGISTER("nbhd_mask_generator")
class NbhdMaskGenerator(GeneralMaskGenerator):
    seqlen = None
    mask = None

    def __init__(self, q_block_size=128, k_block_size=128, sparse_setting={}, attnmap_frame_num=None):
        super().__init__(q_block_size, k_block_size, sparse_setting, attnmap_frame_num)
        self.coefficient = self.sparse_setting.get("nbhd_coefficient", [1.0, 0.5, 0.056])
        self.min_width = self.sparse_setting.get("nbhd_min_width", 1.0)
        self.block_size = self.q_block_size

    def __call__(self, q, k):
        seqlen, head_num, head_dim = q.shape
        if seqlen == NbhdMaskGenerator.seqlen:
            return NbhdMaskGenerator.mask
        block_num = (seqlen + self.block_size - 1) // self.block_size
        block_num_per_frame = seqlen / self.attnmap_frame_num / self.block_size
        mask = generate_nbhd_mask(block_num_per_frame, block_num, self.attnmap_frame_num, coefficient=self.coefficient, min_width=self.min_width, device=q.device)
        mask = mask.unsqueeze(0).repeat(head_num, 1, 1)
        # return: [H, Q_block_num, K_block_num]
        NbhdMaskGenerator.seqlen = seqlen
        NbhdMaskGenerator.mask = mask
        return mask


@SPARSE_MASK_GENERATOR_REGISTER("svg_mask_generator")
class SvgMaskGenerator(GeneralMaskGenerator):
    seqlen = None
    attention_masks = None
    mask = None

    def __init__(self, q_block_size=128, k_block_size=128, sparse_setting={}, attnmap_frame_num=None):
        super().__init__(q_block_size, k_block_size, sparse_setting, attnmap_frame_num)
        self.sample_mse_max_row = self.sparse_setting.get("svg_sample_mse_max_row", 10000)
        self.num_sampled_rows = self.sparse_setting.get("svg_num_sampled_rows", 64)
        self.context_length = self.sparse_setting.get("svg_context_length", 0)
        self.sparsity = self.sparse_setting.get("svg_sparsity", 0.75)
        self.block_size = self.k_block_size
        self.best_model_idx = None
        self.head_num = None
        self.head_dim = None

    def prepare_mask(self, q):
        seqlen, head_num, head_dim = q.shape
        if seqlen == SvgMaskGenerator.seqlen:
            return
        logger.info(f"SvgMaskGenerator: Preparing mask for seqlen={seqlen}, head_num={head_num}, head_dim={head_dim}")
        frame_size = seqlen // self.attnmap_frame_num
        SvgMaskGenerator.attention_masks = [get_attention_mask(mask_name, self.sample_mse_max_row, self.context_length, self.attnmap_frame_num, frame_size) for mask_name in ["spatial", "temporal"]]
        block_num = (seqlen + self.block_size - 1) // self.block_size
        mask = diagonal_band_mask_from_sparsity(block_num, self.sparsity, device=q.device)
        SvgMaskGenerator.mask = mask.unsqueeze(0).repeat(head_num, 1, 1)
        SvgMaskGenerator.seqlen = seqlen

    def sample_mse(self, query, key, value):
        cfg, num_heads, seq_len, dim = query.size()
        num_sampled_rows = min(self.num_sampled_rows, seq_len)
        sampled_rows = torch.randint(low=0, high=self.sample_mse_max_row, size=(num_sampled_rows,))
        sampled_q = query[:, :, sampled_rows, :]
        sampled_qk_scores = torch.matmul(sampled_q, key.transpose(-2, -1)) / (dim**0.5)

        sampled_attn_weights = F.softmax(sampled_qk_scores, dim=-1)
        sampled_golden_hidden_states = torch.matmul(sampled_attn_weights, value)  # (1, seq_len, dim)

        sampled_mses = torch.zeros(len(self.attention_masks), cfg, num_heads, device=query.device, dtype=query.dtype)

        # Only have Tri-diagonal and Striped
        for mask_idx, attn_mask in enumerate(self.attention_masks):
            sampled_attention_mask = attn_mask[sampled_rows, :]
            sampled_attention_scores = sampled_qk_scores.masked_fill(sampled_attention_mask == 0, float("-inf"))
            sampled_attn_weights = F.softmax(sampled_attention_scores, dim=-1)
            sampled_hidden_states = torch.matmul(sampled_attn_weights, value)
            mse = torch.mean((sampled_hidden_states - sampled_golden_hidden_states) ** 2, dim=(2, 3))
            sampled_mses[mask_idx] = mse

        return sampled_mses

    def reorg(self, q, k, v):
        seqlen, head_num, head_dim = q.shape

        q = q.unsqueeze(0).transpose(1, 2)
        k = k.unsqueeze(0).transpose(1, 2)
        v = v.unsqueeze(0).transpose(1, 2)

        sampled_mses = self.sample_mse(q, k, v)
        self.best_mask_idx = torch.argmin(sampled_mses, dim=0)
        self.head_num = head_num
        self.head_dim = head_dim

        q_out, k_out, v_out = torch.zeros_like(q), torch.zeros_like(k), torch.zeros_like(v)

        wan_sparse_head_placement(q, k, v, q_out, k_out, v_out, self.best_mask_idx, self.context_length, self.attnmap_frame_num, seqlen // self.attnmap_frame_num)

        q_out = q_out.transpose(1, 2).squeeze(0)
        k_out = k_out.transpose(1, 2).squeeze(0)
        v_out = v_out.transpose(1, 2).squeeze(0)
        return q_out, k_out, v_out

    def restore(self, out):
        # out: (L, H*D)
        out = out.reshape(-1, self.head_num, self.head_dim)
        seqlen = out.shape[0]
        # (L, H, D) -> (B, H, L, D)
        out = out.unsqueeze(0).transpose(1, 2)

        restore_out = torch.zeros_like(out)
        wan_hidden_states_placement(out, restore_out, self.best_mask_idx, self.context_length, self.attnmap_frame_num, seqlen // self.attnmap_frame_num)

        restore_out = restore_out.transpose(1, 2).reshape(seqlen, -1)
        return restore_out

    def __call__(self, q, k):
        self.prepare_mask(q)
        # return: [H, Q_block_num, K_block_num]
        return self.mask
