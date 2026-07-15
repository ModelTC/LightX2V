from lightx2v.models.networks.lora_adapter import LoraAdapter
from lightx2v.models.networks.wan.infer.dancer import (
    WanDancerPostInfer,
    WanDancerPreInfer,
    WanDancerTransformerInfer,
)
from lightx2v.models.networks.wan.model import WanModel
from lightx2v.models.networks.wan.weights.dancer import WanDancerPreWeights, WanDancerTransformerWeights


class WanDancerLoraAdapter(LoraAdapter):
    """Apply Wan-I2V LoRA deltas to Dancer's equivalent extra branches too."""

    aliases = (
        ("diffusion_model.patch_embedding.", "diffusion_model.patch_embedding_global."),
        ("diffusion_model.head.", "diffusion_model.head_global."),
        ("diffusion_model.img_emb.", "diffusion_model.img_emb_refimage."),
    )

    def _load_lora_file(self, file_path):
        weights = super()._load_lora_file(file_path)
        for key, tensor in list(weights.items()):
            for source, target in self.aliases:
                if key.startswith(source):
                    weights[target + key[len(source) :]] = tensor
        return weights


class WanDancerModel(WanModel):
    pre_weight_class = WanDancerPreWeights
    transformer_weight_class = WanDancerTransformerWeights

    def __init__(self, model_path, config, device):
        self.preserved_keys = [
            "blocks.",
            "patch_embedding.",
            "patch_embedding_global.",
            "text_embedding.",
            "time_embedding.",
            "time_projection.",
            "img_emb.",
            "img_emb_refimage.",
            "music_projection.",
            "music_encoder.",
            "head.",
            "head_global.",
        ]
        for injector_id in range(len(config["music_inject_layers"])):
            self.preserved_keys.extend(
                [
                    f"music_injector.injector.{injector_id}.v.",
                    f"music_injector.injector.{injector_id}.o.",
                ]
            )
        super().__init__(model_path, config, device, model_type="wan_dancer")

    def _should_init_empty_model(self):
        if self.config.get("lora_configs") and not self.config.get("lora_dynamic_apply", False):
            return True
        return super()._should_init_empty_model()

    def apply_merged_lora(self, lora_configs):
        WanDancerLoraAdapter(self).apply_lora(lora_configs, model_type="wan_dancer")

    def _init_infer_class(self):
        if self.config.get("feature_caching", "NoCaching") != "NoCaching":
            raise NotImplementedError("Wan-Dancer parity mode requires feature_caching=NoCaching.")
        if self.config.get("cpu_offload", False) and self.config.get("offload_granularity", "block") not in {"block", "model"}:
            raise NotImplementedError("Wan-Dancer supports block/model offload.")
        self.pre_infer_class = WanDancerPreInfer
        self.post_infer_class = WanDancerPostInfer
        self.transformer_infer_class = WanDancerTransformerInfer
