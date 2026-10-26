import torch
import torch.distributed as dist
import torch.nn.functional as F

from lightx2v.models.networks.autoregressive_model import AutoregressiveModelMixin
from lightx2v.models.networks.wan.infer.edenlive.pre_infer import EdenLivePreInfer
from lightx2v.models.networks.wan.infer.edenlive.transformer_infer import EdenLiveARTransformerInfer, EdenLiveTransformerInfer
from lightx2v.models.networks.wan.infer.post_infer import WanPostInfer
from lightx2v.models.networks.wan.model import WanModel
from lightx2v.models.networks.wan.weights.edenlive.pre_weights import EdenLivePreWeights
from lightx2v.utils.envs import GET_DTYPE


class EdenLiveModel(WanModel):
    pre_weight_class = EdenLivePreWeights
    transformer_infer_class = EdenLiveTransformerInfer

    def _apply_weights(self, weight_dict=None):
        pre_infer = getattr(self, "pre_infer", None)
        if pre_infer is not None and hasattr(pre_infer, "clear_request_state"):
            pre_infer.clear_request_state()
        infer = getattr(self, "transformer_infer", None)
        if infer is not None and hasattr(infer, "reset_cuda_graph"):
            infer.reset_cuda_graph()
        super()._apply_weights(weight_dict)
        self.prepare_autoregressive_projection_groups()

    def _infer_cond_uncond(self, inputs, infer_condition=True):
        prediction = super()._infer_cond_uncond(inputs, infer_condition)
        return prediction.to(GET_DTYPE()) if prediction is not None else None

    def _init_infer_class(self):
        self.pre_infer_class = EdenLivePreInfer
        self.post_infer_class = WanPostInfer

    def _seq_parallel_pre_process(self, output):
        world = dist.get_world_size(self.seq_p_group)
        rank = dist.get_rank(self.seq_p_group)
        length = output.x.shape[0]
        padding = (-length) % world
        output.x = F.pad(output.x, (0, 0, 0, padding)).chunk(world)[rank]
        output.valid_token_len = length
        output.adapter_args["seq_start"] = rank * output.x.shape[0]
        if output.embed.shape[0] > 1:
            output.embed = F.pad(output.embed, (0, 0, 0, padding)).chunk(world)[rank]
            output.embed0 = F.pad(output.embed0, (0, 0, 0, 0, 0, padding)).chunk(world)[rank]
        return output


class EdenLiveARModel(AutoregressiveModelMixin, EdenLiveModel):
    transformer_infer_class = EdenLiveARTransformerInfer

    def clear_autoregressive_cache(self):
        pre = getattr(self, "pre_infer", None)
        if pre is not None:
            pre.clear_request_state()
        infer = getattr(self, "transformer_infer", None)
        if infer is not None:
            infer.reset_cuda_graph()
        super().clear_autoregressive_cache()

    @torch.no_grad()
    def infer(self, inputs):
        if self.cpu_offload and self.offload_granularity == "model" and self.scheduler.cache_update:
            self.to_cuda()
        super().infer(inputs)
