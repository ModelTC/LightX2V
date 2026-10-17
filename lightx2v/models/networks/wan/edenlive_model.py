import torch
import torch.distributed as dist
import torch.nn.functional as F

from lightx2v.models.networks.wan.infer.edenlive.convrot_projection import ConvRotProjectionGroup
from lightx2v.models.networks.wan.infer.edenlive.packed_projection import FP8ProjectionGroup
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
        if infer is not None and hasattr(infer, "invalidate_cuda_graph"):
            infer.invalidate_cuda_graph()
        super()._apply_weights(weight_dict)
        pack = self.config.get("pack_qkv", False)
        share = pack or self.config.get("share_qkv_activation_quant", False)
        projection_group = {"fp8-sgl": FP8ProjectionGroup, "int8-convrot": ConvRotProjectionGroup}.get(self.config.get("dit_quant_scheme"))
        enabled = (
            share
            and projection_group is not None
            and not self.config.get("cpu_offload", False)
            and not self.config.get("lazy_load", False)
            and not self.config.get("seq_parallel", False)
            and not self.config.get("tensor_parallel", False)
        )
        for index, block in enumerate(self.transformer_weights.blocks):
            phase = block.compute_phases[0]
            modules = (phase.self_attn_q, phase.self_attn_k, phase.self_attn_v)
            phase.self_attn_projection_group = projection_group.from_modules(modules, packed=pack) if enabled else None
            phase.self_attn_cache_kv_projection_group = (
                projection_group.from_modules(modules[1:]) if enabled and self.config.get("skip_cache_update_suffix", False) and index == len(self.transformer_weights.blocks) - 1 else None
            )
            phase = block.compute_phases[1]
            modules = (phase.cross_attn_k, phase.cross_attn_v)
            # Text K/V share one quantization when the prompt cache is refreshed.
            phase.cross_kv_projection_group = projection_group.from_modules(modules) if enabled else None

    def _infer_cond_uncond(self, inputs, infer_condition=True):
        return super()._infer_cond_uncond(inputs, infer_condition).to(GET_DTYPE())

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


class EdenLiveARModel(EdenLiveModel):
    transformer_infer_class = EdenLiveARTransformerInfer

    @torch.no_grad()
    def infer(self, inputs):
        if (
            self.config.get("skip_cache_update_suffix", False)
            and self.scheduler.cache_update
            and not self.cpu_offload
            and not self.config.get("enable_cfg", False)
            and not self.config.get("seq_parallel", False)
            and not self.config.get("tensor_parallel", False)
            and not self.config.get("lazy_load", False)
            and not self.config.get("clean_cuda_cache", False)
        ):
            # A cache commit consumes only per-layer K/V. Its prediction is
            # discarded by the AR runner, so no unpatchify or noise assignment
            # is needed after the transformer's commit-only forward.
            self.scheduler.infer_condition = True
            pre_infer_out = self.pre_infer.infer(self.pre_weight, inputs)
            self.transformer_infer.infer(self.transformer_weights, pre_infer_out)
            return
        if self.cpu_offload and self.offload_granularity == "model" and self.scheduler.cache_update:
            self.to_cuda()
        super().infer(inputs)
