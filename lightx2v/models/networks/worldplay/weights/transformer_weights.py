from lightx2v.models.networks.hunyuan_video.weights.transformer_weights import HunyuanVideo15TransformerWeights


class WorldPlayTransformerWeights(HunyuanVideo15TransformerWeights):
    """
    Transformer weights for WorldPlay model.

    Inherits from HunyuanVideo15TransformerWeights.
    The ProPE projection weights are loaded via WorldPlayPreWeights.action_weights.
    """

    def __init__(self, config):
        super().__init__(config)
