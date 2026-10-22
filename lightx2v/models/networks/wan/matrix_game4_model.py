import torch

from lightx2v.models.networks.wan.autoregressive_model import WanAutoregressiveModel
from lightx2v.models.networks.wan.infer.matrix_game4.pre_infer import WanMatrixGame4PreInfer
from lightx2v.models.networks.wan.infer.matrix_game4.transformer_infer import WanMatrixGame4TransformerInfer
from lightx2v.models.networks.wan.infer.post_infer import WanPostInfer


class WanMatrixGame4Model(WanAutoregressiveModel):
    def _get_split_type(self, key):
        # Row projections use complete weights so their dense GEMM matches the resident model.
        split_type = super()._get_split_type(key)
        return None if split_type == "row" else split_type

    def _load_ckpt(self, unified_dtype, sensitive_layer):
        weights = super()._load_ckpt(unified_dtype, sensitive_layer)
        return {name.removeprefix("model."): value for name, value in weights.items()}

    def _load_quant_ckpt(self, unified_dtype, sensitive_layer):
        weights = super()._load_quant_ckpt(unified_dtype, sensitive_layer)
        return {name.removeprefix("model."): value for name, value in weights.items()}

    def _seq_parallel_pre_process(self, pre_infer_out):
        pre_infer_out = super()._seq_parallel_pre_process(pre_infer_out)
        pre_infer_out.x = torch.from_dlpack(pre_infer_out.x)
        return pre_infer_out

    def _init_infer_class(self):
        self.pre_infer_class = WanMatrixGame4PreInfer
        self.transformer_infer_class = WanMatrixGame4TransformerInfer
        self.post_infer_class = WanPostInfer
