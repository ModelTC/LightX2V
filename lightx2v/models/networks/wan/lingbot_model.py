import torch
import torch.distributed as dist
import torch.nn.functional as F

from lightx2v.models.networks.wan.infer.lingbot.pre_infer import WanLingbotPreInfer
from lightx2v.models.networks.wan.infer.lingbot.transformer_infer import WanLingbotTransformerInfer
from lightx2v.models.networks.wan.infer.post_infer import WanPostInfer
from lightx2v.models.networks.wan.model import WanModel
from lightx2v.models.networks.wan.weights.lingbot.pre_weights import WanLingbotPreWeights
from lightx2v.models.networks.wan.weights.lingbot.transformer_weights import WanLingbotTransformerWeights


class WanLingbotModel(WanModel):
    pre_weight_class = WanLingbotPreWeights
    transformer_weight_class = WanLingbotTransformerWeights

    def _init_infer_class(self):
        self.pre_infer_class = WanLingbotPreInfer
        self.post_infer_class = WanPostInfer
        self.transformer_infer_class = WanLingbotTransformerInfer

    @torch.no_grad()
    def _seq_parallel_pre_process(self, pre_infer_out):
        pre_infer_out = super()._seq_parallel_pre_process(pre_infer_out)
        if "c2ws_plucker_emb" not in pre_infer_out.conditional_dict:
            return pre_infer_out

        world_size = dist.get_world_size(self.seq_p_group)
        cur_rank = dist.get_rank(self.seq_p_group)
        f, _, _ = pre_infer_out.grid_sizes.tuple
        multiple = world_size * f

        c2ws_plucker_emb = pre_infer_out.conditional_dict["c2ws_plucker_emb"]
        padding_size = (multiple - (c2ws_plucker_emb.shape[0] % multiple)) % multiple
        if padding_size > 0:
            c2ws_plucker_emb = F.pad(c2ws_plucker_emb, (0, 0, 0, padding_size))
        pre_infer_out.conditional_dict["c2ws_plucker_emb"] = torch.chunk(c2ws_plucker_emb, world_size, dim=0)[cur_rank]
        return pre_infer_out
