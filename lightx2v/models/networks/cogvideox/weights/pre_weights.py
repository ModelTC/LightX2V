from lightx2v.common.ops.mm.mm_weight import MMWeightTemplate
from lightx2v.common.ops.norm.layer_norm_weight import LNWeightTemplate
from lightx2v.utils.registry_factory import MM_WEIGHT_REGISTER


class CogvideoxPreWeights:
    def __init__(self, config):
        self.config = config

    def load_weights(self, weight_dict):
        self.time_embedding_linear_1 = MM_WEIGHT_REGISTER["Default"]("time_embedding.linear_1.weight", "time_embedding.linear_1.bias")
        self.time_embedding_linear_2 = MM_WEIGHT_REGISTER["Default"]("time_embedding.linear_2.weight", "time_embedding.linear_2.bias")
        self.patch_embed_proj = MM_WEIGHT_REGISTER["Default"]("patch_embed.proj.weight", "patch_embed.proj.bias")
        self.patch_embed_text_proj = MM_WEIGHT_REGISTER["Default"]("patch_embed.text_proj.weight", "patch_embed.text_proj.bias")

        self.weight_list = [self.time_embedding_linear_1, self.time_embedding_linear_2, self.patch_embed_proj, self.patch_embed_text_proj]

        for mm_weight in self.weight_list:
            mm_weight.set_config(self.config)
            mm_weight.load(weight_dict)

    def to_cpu(self):
        for mm_weight in self.weight_list:
            if isinstance(mm_weight, (MMWeightTemplate, LNWeightTemplate)):
                mm_weight.to_cpu()

    def to_cuda(self):
        for mm_weight in self.weight_list:
            if isinstance(mm_weight, (MMWeightTemplate, LNWeightTemplate)):
                mm_weight.to_cuda()
