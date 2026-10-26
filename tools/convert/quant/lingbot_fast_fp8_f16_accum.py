"""Reduced-range Lingbot Fast conversion, including its camera projections."""

if __package__:
    from .wan_fp8_f16_accum import WanFP8F16AccumQuantization
else:
    from wan_fp8_f16_accum import WanFP8F16AccumQuantization

TARGET_KEYS = ("blocks", "patch_embedding_wancamctrl", "c2ws_hidden_states_layer1", "c2ws_hidden_states_layer2")
BLOCK_LINEARS = (
    "self_attn.q",
    "self_attn.k",
    "self_attn.v",
    "self_attn.o",
    "cross_attn.q",
    "cross_attn.k",
    "cross_attn.v",
    "cross_attn.o",
    "ffn.0",
    "ffn.2",
    "cam_injector_layer1",
    "cam_injector_layer2",
    "cam_scale_layer",
    "cam_shift_layer",
)
EXPECTED_WEIGHT_NAMES = frozenset([f"blocks.{index}.{name}.weight" for index in range(40) for name in BLOCK_LINEARS] + [f"{name}.weight" for name in TARGET_KEYS[1:]])


class LingbotFastFP8F16AccumQuantization(WanFP8F16AccumQuantization):
    def __init__(self):
        super().__init__("wan_dit")
        self.converted_names = set()

    @property
    def metadata(self):
        return {**super().metadata, "model_type": "lingbot_fast", "quantized_linears": str(self.count)}

    def quantize_weight(self, name, weight, default_quantize):
        if name not in EXPECTED_WEIGHT_NAMES or name in self.converted_names:
            raise ValueError(f"Unexpected or duplicated Lingbot Fast linear: {name}")
        result = super().quantize_weight(name, weight, default_quantize)
        self.converted_names.add(name)
        return result

    def validate(self):
        missing = EXPECTED_WEIGHT_NAMES - self.converted_names
        if missing:
            raise ValueError(f"Lingbot Fast FP8 conversion is missing {len(missing)} linears, including: {sorted(missing)[:8]}")
        super().validate()
