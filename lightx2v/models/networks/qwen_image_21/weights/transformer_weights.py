from lightx2v.common.modules.weight_module import WeightModule, WeightModuleList
from lightx2v.models.networks.qwen_image_21.fp8_f16_accum_policy import ACTIVATION_QMAX
from lightx2v.utils.registry_factory import ATTN_WEIGHT_REGISTER, LN_WEIGHT_REGISTER, MM_WEIGHT_REGISTER, RMS_WEIGHT_REGISTER


class QwenImage21BlockWeights(WeightModule):
    def __init__(self, index, config):
        super().__init__()
        prefix = f"transformer_blocks.{index}"
        mm_type = config["dit_quant_scheme"] if config.get("dit_quantized", False) else "Default"
        # Fuse q/k/v and gate/up into single fp8 GEMMs (fewer, larger GEMMs and
        # one shared activation quantization instead of one per linear).
        self.fused_qkv_mlp = bool(config.get("dit_fuse_qkv_mlp", False)) and mm_type == "fp8-rocm"
        if self.fused_qkv_mlp:
            from lightx2v.common.ops.mm.mm_weight import MMWeightFusedFp8Rocm

            self.add_module("qkv", MMWeightFusedFp8Rocm([f"{prefix}.attn.to_q.weight", f"{prefix}.attn.to_k.weight", f"{prefix}.attn.to_v.weight"], bias_name=None))
            self.add_module("gate_up", MMWeightFusedFp8Rocm([f"{prefix}.img_mlp.gate_layer.weight", f"{prefix}.img_mlp.proj.weight"], bias_name=None))
            linear_keys = {"out": "attn.to_out.0", "down": "img_mlp.out"}
        else:
            linear_keys = {
                "q": "attn.to_q",
                "k": "attn.to_k",
                "v": "attn.to_v",
                "out": "attn.to_out.0",
                "up": "img_mlp.proj",
                "gate": "img_mlp.gate_layer",
                "down": "img_mlp.out",
            }
        for name, key in linear_keys.items():
            linear = MM_WEIGHT_REGISTER[mm_type](f"{prefix}.{key}.weight", bias_name=None)
            if mm_type == "fp8-f16-accum":
                linear.enable_fp8_f16_accum(config.get("dit_fp8_activation_qmax", ACTIVATION_QMAX))
            self.add_module(name, linear)
        for name in ("q", "k"):
            self.add_module(f"norm_{name}", RMS_WEIGHT_REGISTER[config.get("rms_norm_type", "fp32_variance")](f"{prefix}.attn.norm_{name}.weight", eps=config["eps"]))
        for name in ("norm1", "norm2"):
            self.add_module(name, LN_WEIGHT_REGISTER[config.get("layer_norm_type", "torch")](eps=config["eps"]))
        attn_type = config.get("attn_type", "torch_sdpa")
        attention_cls = ATTN_WEIGHT_REGISTER[attn_type]
        if attn_type == "dynamic_sparse_attn":
            attention = attention_cls(config.get("dynamic_sparse_attn_setting", {}))
        else:
            attention = attention_cls()
        self.add_module("attention", attention)
        # Arbitrary triangular prefix masks use the common SDPA backend.
        self.add_module("prefix_attention", ATTN_WEIGHT_REGISTER["torch_sdpa"]())


class QwenImage21TransformerWeights(WeightModule):
    def __init__(self, config):
        super().__init__()
        self.add_module("blocks", WeightModuleList(QwenImage21BlockWeights(i, config) for i in range(config["num_layers"])))
