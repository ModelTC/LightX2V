from lightx2v.common.ops.mm.mm_weight import MMWeightTemplate
from lightx2v.common.ops.norm.layer_norm_weight import LNWeightTemplate
from lightx2v.utils.registry_factory import LN_WEIGHT_REGISTER, MM_WEIGHT_REGISTER


class CogvideoxPostWeights:
    def __init__(self, config, mm_type="Default"):
        self.config = config
        self.mm_type = mm_type

    def load_weights(self, weight_dict):
        self.norm_out_linear = MM_WEIGHT_REGISTER[self.mm_type]("norm_out.linear.weight", "norm_out.linear.bias")
        self.proj_out = MM_WEIGHT_REGISTER[self.mm_type]("proj_out.weight", "proj_out.bias")
        self.norm_final = LN_WEIGHT_REGISTER[self.mm_type]("norm_final.weight", "norm_final.bias")
        self.norm_out_norm = LN_WEIGHT_REGISTER[self.mm_type]("norm_out.norm.weight", "norm_out.norm.bias", eps=1e-5)

        self.weight_list = [self.norm_out_linear, self.proj_out, self.norm_final, self.norm_out_norm]

        for mm_weight in self.weight_list:
            if isinstance(mm_weight, (MMWeightTemplate, LNWeightTemplate)):
                mm_weight.load(weight_dict)

    def to_cpu(self):
        for mm_weight in self.weight_list:
            if isinstance(mm_weight, (MMWeightTemplate, LNWeightTemplate)):
                mm_weight.to_cpu()

    def to_cuda(self):
        for mm_weight in self.weight_list:
            if isinstance(mm_weight, (MMWeightTemplate, LNWeightTemplate)):
                mm_weight.to_cuda()
