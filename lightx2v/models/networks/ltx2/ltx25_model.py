from lightx2v.models.networks.ltx2.infer.offload.transformer_infer import LTX2OffloadTransformerInfer
from lightx2v.models.networks.ltx2.infer.post_infer import LTX2PostInfer
from lightx2v.models.networks.ltx2.infer.pre_infer import LTX25PreInfer
from lightx2v.models.networks.ltx2.infer.transformer_infer import LTX2TransformerInfer
from lightx2v.models.networks.ltx2.model import LTX2Model
from lightx2v.models.networks.ltx2.weights.pre_weights import LTX25PreWeights


class LTX25Model(LTX2Model):
    """LTX-2.5 DiT using the shared LTX-2 block implementation."""

    pre_weight_class = LTX25PreWeights

    def _init_infer_class(self):
        self.pre_infer_class = LTX25PreInfer
        self.post_infer_class = LTX2PostInfer
        self.transformer_infer_class = LTX2TransformerInfer if not self.cpu_offload else LTX2OffloadTransformerInfer
