from lightx2v.models.networks.hunyuan_video.weights.pre_weights import HunyuanVideo15PreWeights
from lightx2v.models.networks.worldplay.weights.action_weights import WorldPlayActionWeights


class WorldPlayPreWeights(HunyuanVideo15PreWeights):
    """
    Pre-processing weights for WorldPlay model.

    Extends HunyuanVideo15PreWeights with action conditioning weights.

    Note: byt5_in and vision_in weights are loaded by the external encoders
    (ByT5TextEncoder and SiglipVisionEncoder), not by the transformer.
    """

    def __init__(self, config):
        super().__init__(config)
        # Add action conditioning weights
        self.add_module("action_weights", WorldPlayActionWeights(config))
