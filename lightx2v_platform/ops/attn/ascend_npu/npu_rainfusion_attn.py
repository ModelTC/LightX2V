try:
    import mindiesd
    from mindiesd.layers.flash_attn.attention_forward import attention_forward

    _HAS_MINDIESD = True
except ImportError:
    mindiesd = None
    attention_forward = None
    _HAS_MINDIESD = False

try:
    import torch_npu
except ImportError:
    torch_npu = None

"""
RainFusion is a sparse attention acceleration algorithm.
RainFusion requires MindIE-SD to be installed. Installation guide: https://gitcode.com/Ascend/MindIE-SD/blob/master/docs/zh/installation.md
"""


class NpuRainfusionOperator:
    def __init__(self):
        assert torch_npu is not None, "torch_npu is not installed."
        assert _HAS_MINDIESD, "mindiesd is not installed. RainFusion requires MindIE-SD."

    def dense_attention(self, q, k, v):
        return attention_forward(q, k, v, opt_mode="manual", op_type="ascend_laser_attention", layout="BNSD")

    def sparse_attention(
        self,
        q,
        k,
        v,
        scale,
        head_num,
        select_idx,
        select_num_idx,
        block_shape,
        actual_seq_lengths,
        actual_seq_lengths_kv,
    ):
        q_bnsd = q.transpose(1, 2)
        k_bnsd = k.transpose(1, 2)
        v_bnsd = v.transpose(1, 2)
        out = mindiesd.layers.flash_attn.sparse_flash_attn_rf_v2.rain_fusion_attention(
            q_bnsd,
            k_bnsd,
            v_bnsd,
            scale=scale,
            head_num=head_num,
            input_layout="BNSD",
            select_idx=select_idx,
            select_num_idx=select_num_idx,
            blockshape=block_shape,
            actual_seq_lengths=actual_seq_lengths,
            actual_seq_lengths_kv=actual_seq_lengths_kv,
        )
        return out.transpose(1, 2).reshape(q.shape)
