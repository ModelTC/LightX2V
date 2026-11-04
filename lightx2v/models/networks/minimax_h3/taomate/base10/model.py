"""Native FL2VA Base10 transformer sharing MiniMax-H3 weight containers."""

from lightx2v.models.networks.minimax_h3.model import MiniMaxH3Model

from .pre_infer import TaoMateBase10PreInfer
from .transformer_infer import TaoMateBase10OffloadTransformerInfer, TaoMateBase10TransformerInfer


class TaoMateBase10Model(MiniMaxH3Model):
    """Run the released base weights without the Stage3 LoRA adapter."""

    def __init__(self, model_path, config, device, *, shared_base=None):
        self.shared_base_owner = shared_base
        native_config = dict(config)
        native_config.update(
            infer_steps=9,
            feature_caching="NoCaching",
            allow_online_adaln_offload=True,
        )
        if native_config.get("use_adaln_cache", False):
            native_config["adaln_cache_profile"] = "taomate_base10"
        super().__init__(model_path, native_config, device)

    def _init_weights(self, weight_dict=None):
        if self.shared_base_owner is None:
            return super()._init_weights(weight_dict)
        from .shared_weights import bind_base_tensors, validate_shared_base

        validate_shared_base(self.shared_base_owner, self.config)
        self.pre_weight = self.pre_weight_class(self.config)
        self.transformer_weights = self.transformer_weight_class(self.config)
        self.post_weight = self.post_weight_class(self.config)
        self.shared_base_tensor_count = sum(bind_base_tensors(getattr(self, name), getattr(self.shared_base_owner, name)) for name in ("pre_weight", "transformer_weights", "post_weight"))

    def _init_infer_class(self):
        super()._init_infer_class()
        self.pre_infer_class = TaoMateBase10PreInfer
        self.transformer_infer_class = TaoMateBase10OffloadTransformerInfer if self.cpu_offload else TaoMateBase10TransformerInfer
