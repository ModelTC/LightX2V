from transformers.activations import ACT2FN

from lightx2v.common.modules.weight_module import WeightModule, WeightModuleList
from lightx2v.utils.registry_factory import ATTN_WEIGHT_REGISTER, MM_WEIGHT_REGISTER, RMS_WEIGHT_REGISTER
from lightx2v_platform.base.global_var import AI_DEVICE


class HidreamO1ImageTransformerWeights(WeightModule):
    def __init__(self, config, model_path=None, torch_dtype=None):
        super().__init__()
        self.config = config
        self.mm_type = config.get("mm_type", config.get("dit_quant_scheme", "Default"))
        self.rms_norm_type = config.get("rms_norm_type", "one-pass")
        self.attn_type = config["attn_type"]
        self.blocks_num = int(config.get("num_hidden_layers", config.get("num_layers", 0)))
        self.device = AI_DEVICE
        self.rotary_emb = None
        self.blocks = WeightModuleList(HidreamO1ImageDecoderBlockWeights(i, config, self.mm_type, self.rms_norm_type, self.attn_type) for i in range(self.blocks_num))
        self.add_module("blocks", self.blocks)
        self.add_module("norm", RMS_WEIGHT_REGISTER[self.rms_norm_type]("model.language_model.norm.weight", eps=config.get("rms_norm_eps", 1e-6)))

    def configure_model(self, text_config, rotary_emb, device):
        self.blocks_num = text_config.num_hidden_layers
        if len(self.blocks) != self.blocks_num:
            self.blocks = WeightModuleList(HidreamO1ImageDecoderBlockWeights(i, self.config, self.mm_type, self.rms_norm_type, self.attn_type) for i in range(self.blocks_num))
            self.add_module("blocks", self.blocks)
        self.rotary_emb = rotary_emb
        for block_weight in self.blocks:
            block_weight.configure(text_config)
        self.device = device

    def to_cpu(self, non_blocking=True):
        for module in self._modules.values():
            if module is not None and hasattr(module, "to_cpu"):
                module.to_cpu(non_blocking=non_blocking)
        if self.rotary_emb is not None:
            self.rotary_emb.to("cpu")
        self.device = "cpu"

    def to_cuda(self, non_blocking=True):
        for module in self._modules.values():
            if module is not None and hasattr(module, "to_cuda"):
                module.to_cuda(non_blocking=non_blocking)
        if self.rotary_emb is not None:
            self.rotary_emb.to(AI_DEVICE)
        self.device = AI_DEVICE


class HidreamO1ImageDecoderBlockWeights(WeightModule):
    def __init__(self, block_index, config, mm_type, rms_norm_type, attn_type):
        super().__init__()
        self.block_index = block_index
        self.config = config
        self.heads = None
        self.kv_heads = None
        self.head_dim = None
        self.act_fn = None
        prefix = f"model.language_model.layers.{block_index}"
        self.add_module("input_layernorm", RMS_WEIGHT_REGISTER[rms_norm_type](f"{prefix}.input_layernorm.weight", eps=config.get("rms_norm_eps", 1e-6)))
        self.add_module("post_attention_layernorm", RMS_WEIGHT_REGISTER[rms_norm_type](f"{prefix}.post_attention_layernorm.weight", eps=config.get("rms_norm_eps", 1e-6)))
        self.add_module("q_proj", MM_WEIGHT_REGISTER[mm_type](f"{prefix}.self_attn.q_proj.weight", f"{prefix}.self_attn.q_proj.bias"))
        self.add_module("k_proj", MM_WEIGHT_REGISTER[mm_type](f"{prefix}.self_attn.k_proj.weight", f"{prefix}.self_attn.k_proj.bias"))
        self.add_module("v_proj", MM_WEIGHT_REGISTER[mm_type](f"{prefix}.self_attn.v_proj.weight", f"{prefix}.self_attn.v_proj.bias"))
        self.add_module("o_proj", MM_WEIGHT_REGISTER[mm_type](f"{prefix}.self_attn.o_proj.weight", f"{prefix}.self_attn.o_proj.bias"))
        self.add_module("q_norm", RMS_WEIGHT_REGISTER[rms_norm_type](f"{prefix}.self_attn.q_norm.weight", eps=config.get("rms_norm_eps", 1e-6)))
        self.add_module("k_norm", RMS_WEIGHT_REGISTER[rms_norm_type](f"{prefix}.self_attn.k_norm.weight", eps=config.get("rms_norm_eps", 1e-6)))
        self.add_module("gate_proj", MM_WEIGHT_REGISTER[mm_type](f"{prefix}.mlp.gate_proj.weight", None))
        self.add_module("up_proj", MM_WEIGHT_REGISTER[mm_type](f"{prefix}.mlp.up_proj.weight", None))
        self.add_module("down_proj", MM_WEIGHT_REGISTER[mm_type](f"{prefix}.mlp.down_proj.weight", None))
        self.add_module("attn", ATTN_WEIGHT_REGISTER[attn_type]())

    def configure(self, text_config):
        self.head_dim = getattr(text_config, "head_dim", text_config.hidden_size // text_config.num_attention_heads)
        self.heads = text_config.num_attention_heads
        self.kv_heads = text_config.num_key_value_heads
        self.act_fn = ACT2FN[text_config.hidden_act]
