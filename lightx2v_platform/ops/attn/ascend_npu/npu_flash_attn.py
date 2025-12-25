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

    def apply(self, q, k, v, cu_seqlens_q=None, cu_seqlens_kv=None, max_seqlen_q=None, max_seqlen_kv=None, model_cls=None, **kwds):
        if len(q.shape) == 3:
            bs = 1
        elif len(q.shape) == 4:
            bs = q.shape[0]
            q = q.reshape(-1, q.shape[-2], q.shape[-1])
            k = k.reshape(-1, k.shape[-2], k.shape[-1])
            v = v.reshape(-1, v.shape[-2], v.shape[-1])
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
            actual_seq_qlen=tuple(cu_seqlens_q[1:].cpu().numpy().tolist()),
            actual_seq_kvlen=tuple(cu_seqlens_kv[1:].cpu().numpy().tolist()),
        )
        x = x[0]
        x = x.reshape(bs * max_seqlen_q, -1)
        return x
