from abc import ABC, abstractmethod

from lightx2v.utils.registry_factory import SPARSE_MASK_GENERATOR_REGISTER

from .nbhd_attn import generate_nbhd_mask
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
    def __init__(self, q_block_size=128, k_block_size=128, sparse_setting={}, attnmap_frame_num=None):
        super().__init__(q_block_size, k_block_size, sparse_setting, attnmap_frame_num)
        self.coefficient = self.sparse_setting.get("nbhd_coefficient", [1.0, 0.5, 0.056])
        self.min_width = self.sparse_setting.get("nbhd_min_width", 1.0)
        self.block_size = self.q_block_size
        self.seqlen = None
        self.mask = None

    def __call__(self, q, k):
        seqlen, head_num, head_dim = q.shape
        if seqlen == self.seqlen:
            return self.mask
        block_num = (seqlen + self.block_size - 1) // self.block_size
        block_num_per_frame = seqlen / self.attnmap_frame_num / self.block_size
        mask = generate_nbhd_mask(block_num_per_frame, block_num, self.attnmap_frame_num, coefficient=self.coefficient, min_width=self.min_width, device=q.device)
        mask = mask.unsqueeze(0).repeat(head_num, 1, 1)
        # return: [H, Q_block_num, K_block_num]
        self.seqlen = seqlen
        self.mask = mask
        return self.mask
