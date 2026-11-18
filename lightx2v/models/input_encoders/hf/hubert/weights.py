"""HuBERT checkpoint mapping onto LightX2V common weight operators."""

from pathlib import Path

import torch
from safetensors.torch import load_file

from lightx2v.common.modules.weight_module import WeightModule
from lightx2v.common.ops.conv.conv1d import Conv1dWeight
from lightx2v.common.ops.mm.mm_weight import MMWeightForceFp32
from lightx2v.common.ops.norm.layer_norm_weight import LN_WEIGHT_REGISTER


class HubertBlockWeights(WeightModule):
    def __init__(self, index, eps):
        super().__init__()
        prefix = f"encoder.layers.{index}"
        for name in ("q_proj", "k_proj", "v_proj", "out_proj"):
            key = f"{prefix}.attention.{name}"
            self.add_module(name, MMWeightForceFp32(f"{key}.weight", f"{key}.bias"))
        for name in ("intermediate_dense", "output_dense"):
            key = f"{prefix}.feed_forward.{name}"
            self.add_module(name, MMWeightForceFp32(f"{key}.weight", f"{key}.bias"))
        for name in ("layer_norm", "final_layer_norm"):
            key = f"{prefix}.{name}"
            norm = LN_WEIGHT_REGISTER["torch"](f"{key}.weight", f"{key}.bias", eps=eps)
            # HuBERT stays FP32 even when the surrounding DiT runs BF16/FP16.
            norm.infer_dtype = norm.sensitive_layer_dtype = torch.float32
            self.add_module(name, norm)


class HubertWeights(WeightModule):
    def __init__(self, config, last_layer):
        super().__init__()
        self.config = config
        self.conv_layers = []
        self.blocks = []
        for index, stride in enumerate(config["conv_stride"]):
            prefix = f"feature_extractor.conv_layers.{index}"
            conv = Conv1dWeight(f"{prefix}.conv.weight", f"{prefix}.conv.bias" if config["conv_bias"] else None, stride=stride)
            norm = self._norm(f"{prefix}.layer_norm", eps=1e-5)
            self.add_module(f"conv_{index}", conv)
            self.add_module(f"conv_norm_{index}", norm)
            self.conv_layers.append((conv, norm))
        if config.get("feat_proj_layer_norm", True):
            self.add_module("projection_norm", self._norm("feature_projection.layer_norm", config["layer_norm_eps"]))
        else:
            self.projection_norm = None
        self.add_module("projection", MMWeightForceFp32("feature_projection.projection.weight", "feature_projection.projection.bias"))
        self.add_module(
            "position_conv",
            Conv1dWeight(
                "encoder.pos_conv_embed.conv.weight",
                "encoder.pos_conv_embed.conv.bias",
                padding=config["num_conv_pos_embeddings"] // 2,
                groups=config["num_conv_pos_embedding_groups"],
            ),
        )
        for index in range(last_layer):
            block = HubertBlockWeights(index, config["layer_norm_eps"])
            self.add_module(f"block_{index}", block)
            self.blocks.append(block)
        self.final_norm = None
        if last_layer == config["num_hidden_layers"]:
            self.add_module("final_norm", self._norm("encoder.layer_norm", config["layer_norm_eps"]))

    @staticmethod
    def _norm(prefix, eps):
        norm = LN_WEIGHT_REGISTER["torch"](f"{prefix}.weight", f"{prefix}.bias", eps=eps)
        norm.infer_dtype = norm.sensitive_layer_dtype = torch.float32
        return norm

    def load(self, state_dict, device="cpu"):
        # Convert the old/new PyTorch weight-norm layouts once at load time.
        state_dict = dict(state_dict)
        prefix = "encoder.pos_conv_embed.conv"
        if f"{prefix}.weight" not in state_dict:
            if f"{prefix}.weight_g" in state_dict:
                magnitude, direction = state_dict[f"{prefix}.weight_g"], state_dict[f"{prefix}.weight_v"]
            else:
                magnitude = state_dict[f"{prefix}.parametrizations.weight.original0"]
                direction = state_dict[f"{prefix}.parametrizations.weight.original1"]
            state_dict[f"{prefix}.weight"] = torch._weight_norm(direction.float(), magnitude.float(), 2)
        for _, operator in self.named_weight_leaves():
            weight = state_dict[operator.weight_name].detach().float()
            if isinstance(operator, MMWeightForceFp32):
                weight = weight.t()
            operator.weight = weight.to(device)
            operator.bias = state_dict[operator.bias_name].detach().to(device=device, dtype=torch.float32) if operator.bias_name else None
        return self

    def to_device(self, device):
        for _, operator in self.named_weight_leaves():
            operator.weight = operator.weight.to(device)
            if operator.bias is not None:
                operator.bias = operator.bias.to(device)
        return self


def load_hubert_weights(model_path, config, last_layer, device):
    path = Path(model_path)
    safetensors_path = path / "model.safetensors"
    if safetensors_path.is_file():
        state_dict = load_file(str(safetensors_path), device="cpu")
    else:
        state_dict = torch.load(path / "pytorch_model.bin", map_location="cpu", weights_only=True, mmap=True)
    return HubertWeights(config, last_layer).load(state_dict, device=device)
