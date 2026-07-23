import torch

from lightx2v.common.modules.weight_module import WeightModule, WeightModuleList
from lightx2v.utils.registry_factory import ATTN_WEIGHT_REGISTER, LN_WEIGHT_REGISTER, MM_WEIGHT_REGISTER, RMS_WEIGHT_REGISTER, ROPE_REGISTER, TENSOR_REGISTER


class FastWAMSelfAttentionWeights(WeightModule):
    def __init__(self, prefix, block_index, config):
        super().__init__()
        self.add_module(
            "rope",
            ROPE_REGISTER[config.get("fastwam_rope_type", "torch_complex_rope")](layout="interleaved", compute_dtype=torch.float64),
        )
        block = f"{prefix}.blocks.{block_index}"
        rms_type = config.get("rms_norm_type", "torch")
        eps = float(config.get("eps", 1e-6))

        self.add_module("modulation", TENSOR_REGISTER["Default"](f"{block}.modulation"))
        self.add_module("norm1", LN_WEIGHT_REGISTER[config.get("layer_norm_type", "torch")](eps=eps))
        self.add_module("q", MM_WEIGHT_REGISTER["Default"](f"{block}.self_attn.q.weight", f"{block}.self_attn.q.bias"))
        self.add_module("k", MM_WEIGHT_REGISTER["Default"](f"{block}.self_attn.k.weight", f"{block}.self_attn.k.bias"))
        self.add_module("v", MM_WEIGHT_REGISTER["Default"](f"{block}.self_attn.v.weight", f"{block}.self_attn.v.bias"))
        self.add_module("o", MM_WEIGHT_REGISTER["Default"](f"{block}.self_attn.o.weight", f"{block}.self_attn.o.bias"))
        self.add_module("norm_q", RMS_WEIGHT_REGISTER[rms_type](f"{block}.self_attn.norm_q.weight", eps=eps))
        self.add_module("norm_k", RMS_WEIGHT_REGISTER[rms_type](f"{block}.self_attn.norm_k.weight", eps=eps))
        self.add_module("attn", ATTN_WEIGHT_REGISTER["torch_sdpa"]())


class FastWAMCrossAttentionWeights(WeightModule):
    def __init__(self, prefix, block_index, config):
        super().__init__()
        block = f"{prefix}.blocks.{block_index}"
        rms_type = config.get("rms_norm_type", "torch")
        eps = float(config.get("eps", 1e-6))

        self.add_module("norm3", LN_WEIGHT_REGISTER[config.get("layer_norm_type", "torch")](f"{block}.norm3.weight", f"{block}.norm3.bias", eps=eps))
        self.add_module("q", MM_WEIGHT_REGISTER["Default"](f"{block}.cross_attn.q.weight", f"{block}.cross_attn.q.bias"))
        self.add_module("k", MM_WEIGHT_REGISTER["Default"](f"{block}.cross_attn.k.weight", f"{block}.cross_attn.k.bias"))
        self.add_module("v", MM_WEIGHT_REGISTER["Default"](f"{block}.cross_attn.v.weight", f"{block}.cross_attn.v.bias"))
        self.add_module("o", MM_WEIGHT_REGISTER["Default"](f"{block}.cross_attn.o.weight", f"{block}.cross_attn.o.bias"))
        self.add_module("norm_q", RMS_WEIGHT_REGISTER[rms_type](f"{block}.cross_attn.norm_q.weight", eps=eps))
        self.add_module("norm_k", RMS_WEIGHT_REGISTER[rms_type](f"{block}.cross_attn.norm_k.weight", eps=eps))
        self.add_module("attn", ATTN_WEIGHT_REGISTER["torch_sdpa"]())


class FastWAMFFNWeights(WeightModule):
    def __init__(self, prefix, block_index, config):
        super().__init__()
        block = f"{prefix}.blocks.{block_index}"
        eps = float(config.get("eps", 1e-6))

        self.add_module("norm2", LN_WEIGHT_REGISTER[config.get("layer_norm_type", "torch")](eps=eps))
        self.add_module("fc0", MM_WEIGHT_REGISTER["Default"](f"{block}.ffn.0.weight", f"{block}.ffn.0.bias"))
        self.add_module("fc2", MM_WEIGHT_REGISTER["Default"](f"{block}.ffn.2.weight", f"{block}.ffn.2.bias"))


class FastWAMBlockWeights(WeightModule):
    def __init__(self, prefix, block_index, config):
        super().__init__()
        self.add_module("self_attn", FastWAMSelfAttentionWeights(prefix, block_index, config))
        self.add_module("cross_attn", FastWAMCrossAttentionWeights(prefix, block_index, config))
        self.add_module("ffn", FastWAMFFNWeights(prefix, block_index, config))


class FastWAMExpertTransformerWeights(WeightModule):
    def __init__(self, prefix, config):
        super().__init__()
        self.blocks = WeightModuleList([FastWAMBlockWeights(prefix, i, config) for i in range(int(config["num_layers"]))])
        self.add_module("blocks", self.blocks)


class FastWAMTransformerWeights(WeightModule):
    def __init__(self, config, lazy_load_path=None, lora_path=None):
        del lazy_load_path, lora_path
        super().__init__()
        self.config = config
        self.add_module("video", FastWAMExpertTransformerWeights("mixtures.video", config))
        self.add_module("action", FastWAMExpertTransformerWeights("mixtures.action", config))
        self.add_module("video_head", MM_WEIGHT_REGISTER["Default"]("mixtures.video.head.head.weight", "mixtures.video.head.head.bias"))
        self.add_module("video_head_modulation", TENSOR_REGISTER["Default"]("mixtures.video.head.modulation"))
        self.add_module("action_head", MM_WEIGHT_REGISTER["Default"]("mixtures.action.head.weight", "mixtures.action.head.bias"))

    def non_block_weights_to_cuda(self):
        self.video_head.to_cuda()
        self.video_head_modulation.to_cuda()
        self.action_head.to_cuda()

    def non_block_weights_to_cpu(self):
        self.video_head.to_cpu()
        self.video_head_modulation.to_cpu()
        self.action_head.to_cpu()
