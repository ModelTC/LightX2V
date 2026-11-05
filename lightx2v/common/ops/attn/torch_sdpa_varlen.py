import torch

from lightx2v.utils.registry_factory import ATTN_WEIGHT_REGISTER

from .torch_sdpa import TorchSDPAWeight


@ATTN_WEIGHT_REGISTER("torch_sdpa_varlen")
class TorchSDPAVarlenWeight(TorchSDPAWeight):
    cu_seqlens_on_device = False

    def apply(self, q, k, v, drop_rate=0, attn_mask=None, causal=False, cu_seqlens_q=None, cu_seqlens_kv=None, max_seqlen_q=None, max_seqlen_kv=None, **kwargs):
        if any(tensor.ndim != 3 for tensor in (q, k, v)):
            raise ValueError("Variable-length SDPA requires packed 3D q, k and v")
        if attn_mask is not None:
            raise ValueError("Variable-length SDPA does not support a global attn_mask")
        if k.shape[:2] != v.shape[:2]:
            raise ValueError("k and v must have matching token and head counts")

        lengths = []
        for name, cu, tokens in (("cu_seqlens_q", cu_seqlens_q, q.shape[0]), ("cu_seqlens_kv", cu_seqlens_kv, k.shape[0])):
            if cu is None or cu.device.type != "cpu" or cu.dtype not in (torch.int32, torch.int64) or cu.ndim != 1 or cu.numel() < 2:
                raise ValueError(f"{name} must be a CPU int32/int64 1D tensor with at least two entries")
            boundaries = cu.tolist()
            if boundaries[0] != 0 or boundaries[-1] != tokens:
                raise ValueError(f"{name} must span [0, {tokens}]")
            lengths.append([end - start for start, end in zip(boundaries, boundaries[1:])])

        apply_dense = super().apply
        outputs = [
            apply_dense(q_part, k_part, v_part, drop_rate=drop_rate, causal=causal, **kwargs)
            for q_part, k_part, v_part in zip(q.split(lengths[0]), k.split(lengths[1]), v.split(lengths[1]), strict=True)
            if q_part.shape[0]
        ]
        return torch.cat(outputs, dim=0) if outputs else q.new_empty((0, q.shape[1] * v.shape[2]))
