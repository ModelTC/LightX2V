from lightx2v.common.ops.mm.mm_weight import MMWeightWfp8channelAfp8channelF16Accum, unwrap_tp_weight
from lightx2v.models.networks.wan.autoregressive_model import WanAutoregressiveModel
from lightx2v.models.networks.wan.infer.lingbot_fast.pre_infer import WanLingbotFastPreInfer
from lightx2v.models.networks.wan.infer.lingbot_fast.transformer_infer import WanLingbotFastTransformerInfer
from lightx2v.models.networks.wan.infer.post_infer import WanPostInfer
from lightx2v.models.networks.wan.lingbot_model import WanLingbotModel


class WanLingbotFastModel(WanAutoregressiveModel, WanLingbotModel):
    def clear_autoregressive_cache(self):
        pre = getattr(self, "pre_infer", None)
        if pre is not None:
            pre.clear_condition_cache()
        super().clear_autoregressive_cache()

    def _apply_weights(self, weight_dict=None):
        super()._apply_weights(weight_dict)
        if self.config.get("dit_quant_scheme") == "fp8-f16-accum":
            for _, leaf in self.pre_weight.named_weight_leaves():
                linear = unwrap_tp_weight(leaf)
                if type(linear) is MMWeightWfp8channelAfp8channelF16Accum:
                    linear.enable_fp8_f16_accum(self.fp8_activation_qmax)

    def _init_infer_class(self):
        self.pre_infer_class = WanLingbotFastPreInfer
        self.post_infer_class = WanPostInfer
        self.transformer_infer_class = WanLingbotFastTransformerInfer
