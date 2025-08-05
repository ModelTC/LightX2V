import torch.nn as nn
from loguru import logger

from lightx2v.utils.registry_factory import ATTN_WEIGHT_REGISTER

from .template import AttnWeightTemplate

try:
    from spas_sage_attn.autotune import SparseAttentionMeansim
except ImportError:
    logger.info("SparseAttentionMeansim not found, please install sparge first")
    SparseAttentionMeansim = None


@ATTN_WEIGHT_REGISTER("Sparge")
class SpargeAttnWeight(AttnWeightTemplate):
    def __init__(
        self,
        weight_name,
        verbose=False,
        l1=0.07,
        pv_l1=0.08,
        tune_pv=True,
        inner_attn_type="flash_attn3",
    ):
        self.verbose = (verbose,)
        self.l1 = (l1,)
        self.pv_l1 = (pv_l1,)
        self.tune_pv = (tune_pv,)
        self.inner_attn_type = inner_attn_type
        self.inner_cls = SparseAttentionMeansim(l1=l1, pv_l1=pv_l1, tune_pv=tune_pv)
        super().__init__(weight_name)

    def load(self, weight_dict):
        # match all key with prefix weight_name
        for key in weight_dict.keys():
            if key.startswith(self.weight_name):
                sub_name = key.split(".")[-1]
                setattr(
                    self.inner_cls,
                    sub_name,
                    nn.Parameter(weight_dict[key], requires_grad=False),
                )

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
    ):
        if len(q.shape) == 3:
            q = q.unsqueeze(0)
            k = k.unsqueeze(0)
            v = v.unsqueeze(0)

        x = self.inner_cls(q, k, v, tensor_layout="NHD")
        x = x.flatten(2)
        x = x.squeeze(0)

        return x
