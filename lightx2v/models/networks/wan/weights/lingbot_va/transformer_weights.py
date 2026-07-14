import torch

from lightx2v.common.ops.rope import build_rope_weight
from lightx2v.models.networks.wan.weights.transformer_weights import WanTransformerWeights
from lightx2v.utils.registry_factory import MM_WEIGHT_REGISTER


class LingbotVATransformerWeights(WanTransformerWeights):
    """Wan block weights plus LingBot-VA action output head."""

    def __init__(self, config, lazy_load_path=None, lora_path=None):
        super().__init__(config, lazy_load_path=lazy_load_path, lora_path=lora_path)
        for phase in self.iter_self_attention_phases():
            phase.add_module(
                "lingbot_rope",
                build_rope_weight(
                    config,
                    config_key="lingbot_rope_type",
                    layout="interleaved",
                    default="torch_complex_rope",
                    compute_dtype=torch.float64,
                ),
            )
        mm_type = config.get("dit_quant_scheme", "Default")
        if mm_type != "Default":
            assert config.get("dit_quantized") is True
        self.add_module(
            "action_head",
            MM_WEIGHT_REGISTER[mm_type](
                "action_proj_out.weight",
                "action_proj_out.bias",
                lora_prefix="diffusion_model.action_proj_out",
                lora_path=lora_path,
            ),
        )

    def non_block_weights_to_cuda(self):
        super().non_block_weights_to_cuda()
        self.action_head.to_cuda()

    def non_block_weights_to_cpu(self):
        super().non_block_weights_to_cpu()
        self.action_head.to_cpu()
