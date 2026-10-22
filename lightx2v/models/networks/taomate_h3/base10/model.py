"""Native FL2VA Base10 transformer sharing MiniMax-H3 weight containers."""

from lightx2v.models.networks.minimax_h3.model import MiniMaxH3Model

from .pre_infer import TaoMateBase10PreInfer
from .transformer_infer import TaoMateBase10OffloadTransformerInfer, TaoMateBase10TransformerInfer


class TaoMateBase10Model(MiniMaxH3Model):
    """Run the released base weights without the Stage3 LoRA adapter."""

    def __init__(self, model_path, config, device):
        native_config = dict(config)
        native_config.update(
            infer_steps=9,
            feature_caching="NoCaching",
            use_adaln_cache=False,
            allow_online_adaln_offload=True,
        )
        super().__init__(model_path, native_config, device)

    def _init_infer_class(self):
        super()._init_infer_class()
        self.pre_infer_class = TaoMateBase10PreInfer
        self.transformer_infer_class = TaoMateBase10OffloadTransformerInfer if self.cpu_offload else TaoMateBase10TransformerInfer
