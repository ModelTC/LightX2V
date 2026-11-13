import torch
import torch.distributed as dist

from lightx2v.models.networks.wan.infer.tbdub.post_infer import TBDubPostInfer
from lightx2v.models.networks.wan.infer.tbdub.pre_infer import TBDubPreInfer
from lightx2v.models.networks.wan.infer.tbdub.transformer_infer import TBDubTransformerInfer
from lightx2v.models.networks.wan.model import WanModel
from lightx2v.models.networks.wan.weights.tbdub.pre_weights import TBDubPreWeights
from lightx2v.models.networks.wan.weights.tbdub.transformer_weights import TBDubTransformerWeights


class TBDubModel(WanModel):
    """Native Wan2.2 dense specialization for the released TBDub student."""

    pre_weight_class = TBDubPreWeights
    transformer_weight_class = TBDubTransformerWeights

    def __init__(self, model_path, config, device, model_type="wan2.2", **kwargs):
        if config.get("enable_cfg") or config.get("cfg_parallel"):
            raise ValueError("TBDub student uses one conditional branch; CFG and CFG parallelism are not applicable")
        if config.get("tensor_parallel"):
            raise NotImplementedError("TBDub audio projections have not been mapped for tensor parallelism; use sequence parallelism")
        if config.get("cpu_offload") and config.get("offload_granularity", "block") not in ("block", "model"):
            raise NotImplementedError("TBDub supports block/model offload, not phase offload")
        if config.get("feature_caching", "NoCaching") != "NoCaching":
            raise NotImplementedError("TBDub reference/audio conditioning has not been validated with feature caching")
        super().__init__(model_path, config, device, model_type=model_type, **kwargs)

    def _init_infer_class(self):
        self.pre_infer_class = TBDubPreInfer
        self.transformer_infer_class = TBDubTransformerInfer
        self.post_infer_class = TBDubPostInfer

    @torch.no_grad()
    def _seq_parallel_pre_process(self, pre_infer_out):
        world_size = dist.get_world_size(self.seq_p_group)
        rank = dist.get_rank(self.seq_p_group)
        if pre_infer_out.x.shape[0] % world_size:
            raise ValueError("TBDub sequence parallelism requires the joint reference/target token count to divide evenly across ranks")
        # Modulation is per token, including the fixed reference prefix.
        for name in ("x", "embed", "embed0"):
            value = getattr(pre_infer_out, name)
            setattr(pre_infer_out, name, value.chunk(world_size, dim=0)[rank])
        return pre_infer_out
