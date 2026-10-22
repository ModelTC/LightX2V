from lightx2v.models.networks.wan.autoregressive_model import WanAutoregressiveModel
from lightx2v.models.networks.wan.infer.lingbot_fast.pre_infer import WanLingbotFastPreInfer
from lightx2v.models.networks.wan.infer.lingbot_fast.transformer_infer import WanLingbotFastTransformerInfer
from lightx2v.models.networks.wan.infer.post_infer import WanPostInfer
from lightx2v.models.networks.wan.lingbot_model import WanLingbotModel


class WanLingbotFastModel(WanAutoregressiveModel, WanLingbotModel):
    def _init_infer_class(self):
        self.pre_infer_class = WanLingbotFastPreInfer
        self.post_infer_class = WanPostInfer
        self.transformer_infer_class = WanLingbotFastTransformerInfer
