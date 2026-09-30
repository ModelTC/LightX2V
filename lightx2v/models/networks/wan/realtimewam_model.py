from lightx2v.models.networks.wan.fastwam_model import FastWAMNativeModel
from lightx2v.models.networks.wan.infer.realtimewam import RealtimeWAMTransformerInfer
from lightx2v.models.networks.wan.weights.realtimewam import RealtimeWAMTransformerWeights


class RealtimeWAM(FastWAMNativeModel):
    model_type = "realtimewam"
    transformer_weight_class = RealtimeWAMTransformerWeights

    def _init_infer_class(self):
        super()._init_infer_class()
        self.transformer_infer_class = RealtimeWAMTransformerInfer

    def _init_weights(self, weight_dict=None):
        super()._init_weights(weight_dict)
        self.transformer_weights.pack_projections()
