import torch
from loguru import logger

from lightx2v.utils.registry_factory import ATTN_WEIGHT_REGISTER

from .template import AttnWeightTemplate

if torch.cuda.get_device_capability(0) == (8, 9):
    try:
        from sageattention import sageattn_qk_int8_pv_fp16_triton as sageattn
    except ImportError:
        logger.info("sageattn not found, please install sageattention first")
        sageattn = None
else:
    try:
        from sageattention import sageattn
    except ImportError:
        logger.info("sageattn not found, please install sageattention first")
        sageattn = None


@ATTN_WEIGHT_REGISTER("sage_attn2")
class SageAttn2Weight(AttnWeightTemplate):
    def __init__(self):
        self.config = {}

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
        mask_map=None,
    ):
        q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
        if model_cls == "hunyuan":
            x1 = sageattn(
                q[: cu_seqlens_q[1]].unsqueeze(0),
                k[: cu_seqlens_kv[1]].unsqueeze(0),
                v[: cu_seqlens_kv[1]].unsqueeze(0),
                tensor_layout="NHD",
            )
            x2 = sageattn(
                q[cu_seqlens_q[1] :].unsqueeze(0),
                k[cu_seqlens_kv[1] :].unsqueeze(0),
                v[cu_seqlens_kv[1] :].unsqueeze(0),
                tensor_layout="NHD",
            )
            x = torch.cat((x1, x2), dim=1)
            x = x.view(max_seqlen_q, -1)
        elif model_cls in ["wan2.1", "wan2.1_distill", "wan2.1_causvid", "wan2.1_df", "wan2.1_audio"]:
            x = sageattn(
                q.unsqueeze(0),
                k.unsqueeze(0),
                v.unsqueeze(0),
                tensor_layout="NHD",
            )
            x = x.view(max_seqlen_q, -1)
        else:
            raise NotImplementedError(f"Model class '{model_cls}' is not implemented in this attention implementation")
        return x
