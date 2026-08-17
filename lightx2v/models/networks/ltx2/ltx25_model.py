from lightx2v.models.networks.ltx2.infer.pre_infer import LTX25PreInfer
from lightx2v.models.networks.ltx2.model import LTX2Model
from lightx2v.models.networks.ltx2.weights.ltx25_weights import LTX25PreWeights


class LTX25Model(LTX2Model):
    """LTX-2.5 DiT using the shared LTX-2 block implementation."""

    pre_weight_class = LTX25PreWeights
    pre_infer_class = LTX25PreInfer
