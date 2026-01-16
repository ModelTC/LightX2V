import torch

from lightx2v.utils.registry_factory import SPARSE_OPERATOR_REGISTER
from lightx2v_platform.base.global_var import AI_DEVICE

from .kernels.sla_kernel import _attention

try:
    from magi_attention.functional import flex_flash_attn_func as magi_ffa_func
except ImportError:
    magi_ffa_func = None

try:
    from flex_block_attn import flex_block_attn_func
except ImportError:
    flex_block_attn_func = None

try:
    import flashinfer
except ImportError:
    flashinfer = None


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

        # (B, H, Q_block_num, K_block_num)
        mask = mask.int()
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
        # (B, H, Q_block_num, K_block_num) -> (H, Q_block_num, K_block_num)
        mask = mask.squeeze(0)
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


@SPARSE_OPERATOR_REGISTER("flex_block_operator")
class FlexBlockOperator:
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
        q = q.unsqueeze(0).transpose(1, 2)
        k = k.unsqueeze(0).transpose(1, 2)
        v = v.unsqueeze(0).transpose(1, 2)

        pad_len = (self.q_block_size - q.shape[2] % self.q_block_size) % self.q_block_size
        if pad_len > 0:
            q = torch.nn.functional.pad(q, (0, 0, 0, pad_len))
            k = torch.nn.functional.pad(k, (0, 0, 0, pad_len))
            v = torch.nn.functional.pad(v, (0, 0, 0, pad_len))

        # (B, H, Q_block_num, K_block_num)
        mask = mask.bool()
        out = flex_block_attn_func(q, k, v, self.q_block_size, self.k_block_size, mask)

        if pad_len > 0:
            out = out[:, :, :-pad_len, :]

        out = out.transpose(1, 2)

        return out.reshape(max_seqlen_q, -1)



@SPARSE_OPERATOR_REGISTER("flashinfer_operator")
class FlashinferOperator:

    sparse_wrapper = None
    mask = None

    def __init__(self):
        self.q_block_size = 128
        self.k_block_size = 128
        if FlashinferOperator.sparse_wrapper is None:
            float_workspace_buffer = torch.empty(1024 * 1024 * 1024, dtype=torch.uint8, device=AI_DEVICE)
            FlashinferOperator.sparse_wrapper = flashinfer.sparse.VariableBlockSparseAttentionWrapper(float_workspace_buffer, backend="fa2")


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
        # (B, H, Q_block_num, K_block_num) -> (H, Q_block_num, K_block_num)
        mask = mask.squeeze(0)
        if FlashinferOperator.mask is None or not torch.equal(mask, FlashinferOperator.mask):
            _, q_block_num, k_block_num = mask.shape
            block_row_sz=torch.ones(q_block_num, dtype=torch.int32, device=q.device) * self.q_block_size
            block_row_sz[-1] = seqlen - self.q_block_size * (q_block_num - 1)
            block_row_sz = block_row_sz.unsqueeze(0).repeat(head_num, 1)
            block_col_sz=torch.ones(k_block_num, dtype=torch.int32, device=k.device) * self.k_block_size
            block_col_sz[-1] = seqlen - self.k_block_size * (k_block_num - 1)
            block_col_sz = block_col_sz.unsqueeze(0).repeat(head_num, 1)
            FlashinferOperator.sparse_wrapper.plan(
                block_mask_map=mask,
                block_row_sz=block_row_sz,
                block_col_sz=block_col_sz,
                num_qo_heads=head_num,
                num_kv_heads=head_num,
                head_dim=head_dim,
                q_data_type=q.dtype,
            )
            FlashinferOperator.mask = mask

        q = q.transpose(0, 1)
        k = k.transpose(0, 1)
        v = v.transpose(0, 1)
        out = FlashinferOperator.sparse_wrapper.run(q, k, v)
        out = out.transpose(0, 1)

        return out.reshape(max_seqlen_q, -1)

