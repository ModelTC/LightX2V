import torch
from lightx2v.utils.registry_factory import SPARSE_OPERATOR_REGISTER

from .kernels.sla_kernel import _attention


try:
    from magi_attention.functional import flex_flash_attn_func as magi_ffa_func
except ImportError:
    magi_ffa_func = None



@SPARSE_OPERATOR_REGISTER("sla_triton_operator")
class SlaTritonOperator:

    def __init__(self):
        self.q_block_size = 128
        self.k_block_size = 128

    def __call__(
        self,
        q,
        k,
        v,
        mask,
        cu_seqlens_q=None,
        cu_seqlens_kv=None,
        max_seqlen_q=None,
        max_seqlen_kv=None,
        **kwargs,
    ):
        # (L, H, D) -> (B, H, L, D)
        q = q.unsqueeze(0).transpose(1, 2).contiguous()
        k = k.unsqueeze(0).transpose(1, 2).contiguous()
        v = v.unsqueeze(0).transpose(1, 2).contiguous()

        # (H, Q_block_num, K_block_num) -> (B, H, Q_block_num, K_block_num)
        mask = mask.unsqueeze(0).int()
        topk = int(mask.sum(dim=-1).max().item())
        lut = torch.topk(mask, topk, dim=-1, sorted=False).indices

        out = _attention.apply(q, k, v, mask, lut, topk, self.q_block_size, self.k_block_size)
        out = out.transpose(1, 2).reshape(max_seqlen_q, -1)
        return out


@SPARSE_OPERATOR_REGISTER("magi_operator")
class MagiOperator:

    def __init__(self):
        self.q_block_size = 128
        self.k_block_size = 128

    def generate_qk_ranges(self, mask, q_block_size, k_block_size, seqlen):
        # mask: [H, Q_block_num, K_block_num]
        h_indices, i_indices, j_indices = torch.nonzero(mask, as_tuple=True)

        base_offset = h_indices * seqlen

        q_start = base_offset + i_indices * q_block_size
        q_end = base_offset + torch.clamp((i_indices + 1) * q_block_size, max=seqlen)

        k_start = base_offset + j_indices * k_block_size
        k_end = base_offset + torch.clamp((j_indices + 1) * k_block_size, max=seqlen)

        q_ranges = torch.stack([q_start, q_end], dim=1).to(dtype=torch.int32)
        k_ranges = torch.stack([k_start, k_end], dim=1).to(dtype=torch.int32)

        return q_ranges, k_ranges

    def __call__(
        self,
        q,
        k,
        v,
        mask,
        cu_seqlens_q=None,
        cu_seqlens_kv=None,
        max_seqlen_q=None,
        max_seqlen_kv=None,
        **kwargs,
    ):
        seqlen, head_num, head_dim = q.shape
        q_ranges, k_ranges = self.generate_qk_ranges(mask, self.q_block_size, self.k_block_size, seqlen)
        attn_type_map = torch.zeros(len(q_ranges), dtype=torch.int32, device="cpu").to(q.device, non_blocking=True)

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

        return out.reshape(out.shape[0], -1)

