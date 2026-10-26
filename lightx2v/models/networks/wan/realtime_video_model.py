"""Released Krea Realtime checkpoint with the shared Wan causal executor."""

from lightx2v.models.networks.wan.infer.post_infer import WanPostInfer
from lightx2v.models.networks.wan.infer.realtime_video.transformer_infer import WanRealtimeVideoTransformerInfer
from lightx2v.models.networks.wan.sf_model import WanSFModel
from lightx2v.utils.envs import GET_DTYPE


class WanRealtimeVideoPostInfer(WanPostInfer):
    def infer(self, x, pre_infer_out):
        # Krea's wrapper converts flow to x0 from the BF16 model output.
        return [value.to(GET_DTYPE()) for value in super().infer(x, pre_infer_out)]


class WanRealtimeVideoModel(WanSFModel):
    def _init_infer_class(self):
        super()._init_infer_class()
        self.transformer_infer_class = WanRealtimeVideoTransformerInfer
        self.post_infer_class = WanRealtimeVideoPostInfer
