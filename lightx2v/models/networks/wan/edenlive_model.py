import torch.distributed as dist
import torch.nn.functional as F

from lightx2v.models.networks.wan.infer.edenlive.pre_infer import EdenLivePreInfer, Wan21EdenLivePreInfer
from lightx2v.models.networks.wan.infer.edenlive.transformer_infer import EdenLiveARTransformerInfer, EdenLiveTransformerInfer
from lightx2v.models.networks.wan.infer.post_infer import WanPostInfer
from lightx2v.models.networks.wan.model import WanModel
from lightx2v.models.networks.wan.weights.pre_weights import WanPreWeights
from lightx2v.utils.envs import GET_DTYPE
from lightx2v.utils.registry_factory import TENSOR_REGISTER


class EdenLivePreWeights(WanPreWeights):
    def __init__(self, config):
        super().__init__(config)
        if config["use_image_encoder"]:
            self.proj_0.eps = self.proj_4.eps = 1e-5
        if config["proxy_condition_rmsnorm"]:
            self.register_parameter("condition_norm", TENSOR_REGISTER["Default"]("patch_embedding.condition_norm.weight"))


class EdenLiveModel(WanModel):
    pre_weight_class = EdenLivePreWeights
    pre_infer_cls = EdenLivePreInfer
    transformer_infer_cls = EdenLiveTransformerInfer

    def _infer_cond_uncond(self, inputs, infer_condition=True):
        return super()._infer_cond_uncond(inputs, infer_condition).to(GET_DTYPE())

    def _init_infer_class(self):
        self.pre_infer_class = self.pre_infer_cls
        self.transformer_infer_class = self.transformer_infer_cls
        self.post_infer_class = WanPostInfer

    def _init_infer(self):
        super()._init_infer()
        if self.config["rope_type"] == "torch_real_rope":
            # Match upstream's BF16 rounding between the real-valued RoPE operations.
            for block in self.transformer_weights.blocks:
                rope = block.compute_phases[0].rope
                if self.config.get("rope_chunk", False):
                    rope = rope.inner
                rope.compute_dtype = GET_DTYPE()

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


class Wan21EdenLiveModel(EdenLiveModel):
    pre_infer_cls = Wan21EdenLivePreInfer


class EdenLiveARModel(EdenLiveModel):
    transformer_infer_cls = EdenLiveARTransformerInfer

    def infer(self, inputs):
        if self.cpu_offload and self.offload_granularity == "model" and self.scheduler.cache_update:
            self.to_cuda()
        super().infer(inputs)


class Wan21EdenLiveARModel(EdenLiveARModel):
    pre_infer_cls = Wan21EdenLivePreInfer
