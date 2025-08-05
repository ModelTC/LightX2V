from lightx2v.common.ops.mm.mm_weight import MMWeightTemplate
from lightx2v.common.ops.norm.layer_norm_weight import LNWeightTemplate
from lightx2v.utils.registry_factory import LN_WEIGHT_REGISTER, MM_WEIGHT_REGISTER


class CogvideoxTransformerWeights:
    def __init__(self, config, task="t2v", mm_type="Default"):
        self.config = config
        self.task = task
        self.mm_type = mm_type
        self.init()

    def init(self):
        self.num_layers = self.config["num_layers"]

    def load_weights(self, weight_dict):
        self.blocks_weights = [CogVideoXBlock(i, self.task, self.mm_type) for i in range(self.num_layers)]
        for block in self.blocks_weights:
            block.load_weights(weight_dict)

    def to_cpu(self):
        for block in self.blocks_weights:
            block.to_cpu()

    def to_cuda(self):
        for block in self.blocks_weights:
            block.to_cuda()


class CogVideoXBlock:
    def __init__(self, block_index, task="t2v", mm_type="Default"):
        super().__init__()
        self.block_index = block_index
        self.mm_type = mm_type
        self.task = task

    def load_weights(self, weight_dict):
        self.attn1_to_k = MM_WEIGHT_REGISTER[self.mm_type](f"transformer_blocks.{self.block_index}.attn1.to_k.weight", f"transformer_blocks.{self.block_index}.attn1.to_k.bias")
        self.attn1_to_q = MM_WEIGHT_REGISTER[self.mm_type](f"transformer_blocks.{self.block_index}.attn1.to_q.weight", f"transformer_blocks.{self.block_index}.attn1.to_q.bias")
        self.attn1_to_v = MM_WEIGHT_REGISTER[self.mm_type](f"transformer_blocks.{self.block_index}.attn1.to_v.weight", f"transformer_blocks.{self.block_index}.attn1.to_v.bias")
        self.attn1_to_out = MM_WEIGHT_REGISTER[self.mm_type](f"transformer_blocks.{self.block_index}.attn1.to_out.0.weight", f"transformer_blocks.{self.block_index}.attn1.to_out.0.bias")
        self.ff_net_0_proj = MM_WEIGHT_REGISTER[self.mm_type](f"transformer_blocks.{self.block_index}.ff.net.0.proj.weight", f"transformer_blocks.{self.block_index}.ff.net.0.proj.bias")
        self.ff_net_2_proj = MM_WEIGHT_REGISTER[self.mm_type](f"transformer_blocks.{self.block_index}.ff.net.2.weight", f"transformer_blocks.{self.block_index}.ff.net.2.bias")
        self.norm1_linear = MM_WEIGHT_REGISTER[self.mm_type](f"transformer_blocks.{self.block_index}.norm1.linear.weight", f"transformer_blocks.{self.block_index}.norm1.linear.bias")
        self.norm2_linear = MM_WEIGHT_REGISTER[self.mm_type](f"transformer_blocks.{self.block_index}.norm2.linear.weight", f"transformer_blocks.{self.block_index}.norm2.linear.bias")
        self.attn1_norm_k = LN_WEIGHT_REGISTER[self.mm_type](f"transformer_blocks.{self.block_index}.attn1.norm_k.weight", f"transformer_blocks.{self.block_index}.attn1.norm_k.bias")
        self.attn1_norm_q = LN_WEIGHT_REGISTER[self.mm_type](f"transformer_blocks.{self.block_index}.attn1.norm_q.weight", f"transformer_blocks.{self.block_index}.attn1.norm_q.bias")
        self.norm1_norm = LN_WEIGHT_REGISTER[self.mm_type](f"transformer_blocks.{self.block_index}.norm1.norm.weight", f"transformer_blocks.{self.block_index}.norm1.norm.bias", eps=1e-05)
        self.norm2_norm = LN_WEIGHT_REGISTER[self.mm_type](f"transformer_blocks.{self.block_index}.norm2.norm.weight", f"transformer_blocks.{self.block_index}.norm2.norm.bias", eps=1e-05)

        self.weight_list = [
            self.attn1_to_k,
            self.attn1_to_q,
            self.attn1_to_v,
            self.attn1_to_out,
            self.ff_net_0_proj,
            self.ff_net_2_proj,
            self.norm1_linear,
            self.norm2_linear,
            self.attn1_norm_k,
            self.attn1_norm_q,
            self.norm1_norm,
            self.norm2_norm,
        ]
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
