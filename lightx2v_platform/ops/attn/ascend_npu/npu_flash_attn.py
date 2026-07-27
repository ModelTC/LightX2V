import math

from lightx2v_platform.ops.attn.template import AttnWeightTemplate
from lightx2v_platform.registry_factory import PLATFORM_ATTN_WEIGHT_REGISTER

try:
    import torch_npu
except ImportError:
    torch_npu = None


@PLATFORM_ATTN_WEIGHT_REGISTER("npu_flash_attn")
class NpuFlashAttnWeight(AttnWeightTemplate):
    def __init__(self):
        self.config = {}
        assert torch_npu is not None, "torch_npu is not installed."

    @staticmethod
    def _sequence_metadata(cu_seqlens, batch_size, sequence_length, max_seqlen, name):
        if cu_seqlens is None:
            actual_seq_len = tuple(sequence_length * (batch_index + 1) for batch_index in range(batch_size))
            inferred_max_seqlen = sequence_length
        else:
            cumulative_lengths = cu_seqlens.cpu().tolist()
            actual_seq_len = tuple(int(length) for length in cumulative_lengths[1:])
            sequence_lengths = [int(end) - int(start) for start, end in zip(cumulative_lengths, cumulative_lengths[1:])]
            inferred_max_seqlen = max(sequence_lengths, default=0)

        if max_seqlen is None:
            max_seqlen = inferred_max_seqlen
        else:
            max_seqlen = int(max_seqlen)

        if max_seqlen < inferred_max_seqlen:
            raise ValueError(f"{name}={max_seqlen} is smaller than the inferred sequence length {inferred_max_seqlen}.")
        return actual_seq_len, max_seqlen

    def apply(self, q, k, v, cu_seqlens_q=None, cu_seqlens_kv=None, max_seqlen_q=None, max_seqlen_kv=None, **kwds):
        if q.ndim not in (3, 4):
            raise ValueError(f"npu_flash_attn expects 3D or 4D q/k/v tensors, but q is {q.ndim}D.")
        if k.ndim != q.ndim or v.ndim != q.ndim:
            raise ValueError(f"q, k, and v must have the same rank, but got {q.ndim}D, {k.ndim}D, and {v.ndim}D.")
        if k.shape[:-2] != v.shape[:-2]:
            raise ValueError(f"k and v must have matching batch and sequence dimensions, but got {k.shape[:-2]} and {v.shape[:-2]}.")

        if q.ndim == 3:
            bs = 1
            q_seqlen = q.shape[0]
            kv_seqlen = k.shape[0]
        else:
            bs = q.shape[0]
            if k.shape[0] != bs or v.shape[0] != bs:
                raise ValueError(f"q, k, and v must have the same batch size, but got {bs}, {k.shape[0]}, and {v.shape[0]}.")
            q_seqlen = q.shape[1]
            kv_seqlen = k.shape[1]
            q = q.reshape(-1, q.shape[-2], q.shape[-1])
            k = k.reshape(-1, k.shape[-2], k.shape[-1])
            v = v.reshape(-1, v.shape[-2], v.shape[-1])

        actual_seq_qlen, max_seqlen_q = self._sequence_metadata(cu_seqlens_q, bs, q_seqlen, max_seqlen_q, "max_seqlen_q")
        actual_seq_kvlen, _ = self._sequence_metadata(cu_seqlens_kv, bs, kv_seqlen, max_seqlen_kv, "max_seqlen_kv")
        q_token_count = bs * max_seqlen_q if cu_seqlens_q is None else actual_seq_qlen[-1]

        if kwds.get("softmax_scale", None) is None:
            softmax_scale = 1.0 / math.sqrt(q.shape[-1])
        else:
            softmax_scale = kwds.get("softmax_scale")
        if kwds.get("dropout", None) is None:
            keep_prob = 1.0
        else:
            keep_prob = 1.0 - kwds.get("dropout")
        head_num = q.shape[1]
        x = torch_npu.npu_fusion_attention(
            q,
            k,
            v,
            head_num,
            pse=None,
            atten_mask=None,
            scale=softmax_scale,
            keep_prob=keep_prob,
            input_layout="TND",
            actual_seq_qlen=actual_seq_qlen,
            actual_seq_kvlen=actual_seq_kvlen,
        )
        x = x[0]
        x = x.reshape(q_token_count, -1)
        return x
