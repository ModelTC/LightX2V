from dataclasses import dataclass

import torch

from lightx2v.common.modules.weight_module import WeightModule
from lightx2v.models.networks.wan.weights.pre_weights import WanPreWeights
from lightx2v.utils.registry_factory import TENSOR_REGISTER

from ._shared import SequentialLinearWeights, apply_time_embedding, build_mm_weight, load_prefixed_submodules, projector_layer_prefixes


@dataclass
class ActionExpertConfig:
    dim: int = 1024
    ffn_dim: int = 4096
    num_layers: int = 30
    state_dim: int = 14
    action_dim: int = 14
    chunk_size: int = 16
    video_feature_dim: int = 3072
    causal: bool = False
    num_registers: int = 4
    eps: float = 1e-6
    training_mode: str = "finetune"

    def __post_init__(self):
        assert self.chunk_size >= 2


@dataclass
class UndExpertConfig:
    dim: int = 512
    ffn_dim: int = 2048
    num_layers: int = 30
    vlm_input_dim: int = 2048
    vlm_projector_type: str = "mlp3x_silu"
    eps: float = 1e-5


@dataclass(frozen=True)
class MotusExpertConfigs:
    action: ActionExpertConfig
    und: UndExpertConfig


def build_action_expert_config(config):
    action_chunk_size = config["num_video_frames"] * config.get("video_action_freq_ratio", 2)
    training_mode = config.get("training_mode", "finetune")
    if training_mode == "pretrain":
        chunk_size = action_chunk_size
        num_registers = 0
    else:
        chunk_size = action_chunk_size + 1
        num_registers = 4
    return ActionExpertConfig(
        dim=config.get("action_expert_dim", 1024),
        ffn_dim=config.get("action_expert_dim", 1024) * config.get("action_expert_ffn_dim_multiplier", 4),
        num_layers=config.get("num_layers", 30),
        state_dim=config.get("action_state_dim", 14),
        action_dim=config.get("action_dim", 14),
        chunk_size=chunk_size,
        video_feature_dim=config["dim"],
        causal=False,
        num_registers=num_registers,
        eps=config.get("action_expert_norm_eps", 1e-6),
        training_mode=training_mode,
    )


def build_und_expert_config(config):
    return UndExpertConfig(
        dim=config.get("und_expert_hidden_size", 512),
        ffn_dim=config.get("und_expert_hidden_size", 512) * config.get("und_expert_ffn_dim_multiplier", 4),
        num_layers=config.get("num_layers", 30),
        vlm_input_dim=config.get("vlm_adapter_input_dim", 2048),
        vlm_projector_type=config.get("vlm_adapter_projector_type", "mlp3x_silu"),
        eps=config.get("und_expert_norm_eps", 1e-5),
    )


def build_motus_expert_configs(config):
    return MotusExpertConfigs(
        action=build_action_expert_config(config),
        und=build_und_expert_config(config),
    )


class MotusWanPreWeights(WanPreWeights):
    pass


class MotusActionPreWeights(WeightModule):
    def __init__(self, config, action_config):
        super().__init__()
        self.config = config
        self.action_config = action_config
        self.freq_dim = 256

        if action_config.training_mode == "pretrain":
            self.add_module(
                "action_encoder",
                SequentialLinearWeights(
                    projector_layer_prefixes("mlp3x_silu", "input_encoder.action_encoder"),
                    "silu",
                    config,
                ),
            )
        else:
            self.add_module(
                "state_encoder",
                SequentialLinearWeights(
                    projector_layer_prefixes("mlp3x_silu", "input_encoder.state_encoder"),
                    "silu",
                    config,
                ),
            )
            self.add_module(
                "action_encoder",
                SequentialLinearWeights(
                    projector_layer_prefixes("mlp3x_silu", "input_encoder.action_encoder"),
                    "silu",
                    config,
                ),
            )
        self.register_parameter("pos_embedding", TENSOR_REGISTER["Default"]("input_encoder.pos_embedding"))
        if action_config.num_registers > 0:
            self.register_parameter("registers", TENSOR_REGISTER["Default"]("registers"))
        else:
            self.registers = None

        self.add_module("time_embedding_0", build_mm_weight("time_embedding.0.weight", "time_embedding.0.bias", config))
        self.add_module("time_embedding_2", build_mm_weight("time_embedding.2.weight", "time_embedding.2.bias", config))
        self.add_module("time_projection_1", build_mm_weight("time_projection.1.weight", "time_projection.1.bias", config))

    def apply_input_encoder(self, state_tokens, action_tokens):
        if self.action_config.training_mode == "pretrain":
            encoded = self.action_encoder.apply(action_tokens)
        else:
            encoded = torch.cat([self.state_encoder.apply(state_tokens), self.action_encoder.apply(action_tokens)], dim=1)
        if self.registers is not None and hasattr(self.registers, "tensor"):
            registers = self.registers.tensor.expand(encoded.shape[0], -1, -1).to(dtype=encoded.dtype, device=encoded.device)
            encoded = torch.cat([encoded, registers], dim=1)
        pos_embedding = self.pos_embedding.tensor[:, : encoded.shape[1], :].to(dtype=encoded.dtype, device=encoded.device)
        return encoded + pos_embedding

    def get_time_embedding(self, timestep, seq_len):
        return apply_time_embedding(
            timestep,
            seq_len,
            self.freq_dim,
            self.action_config.dim,
            self.time_embedding_0,
            self.time_embedding_2,
            self.time_projection_1,
        )


class MotusUndPreWeights(WeightModule):
    def __init__(self, config, und_config):
        super().__init__()
        self.add_module(
            "vlm_adapter",
            SequentialLinearWeights(
                projector_layer_prefixes(und_config.vlm_projector_type, "vlm_adapter"),
                "silu",
                config,
            ),
        )


class MotusImageContextWeights(WeightModule):
    def __init__(self, config):
        super().__init__()
        self.available = True
        self._required_keys = {
            "image_context_adapter.0.weight",
            "image_context_adapter.0.bias",
            "image_context_adapter.2.weight",
            "image_context_adapter.2.bias",
        }
        self.add_module(
            "adapter",
            SequentialLinearWeights(
                ["image_context_adapter.0", "image_context_adapter.2"],
                "gelu_tanh",
                config,
            ),
        )

    def load(self, weight_dict):
        if not self._required_keys.issubset(weight_dict.keys()):
            self.available = False
            return
        self.available = True
        super().load(weight_dict)

    def apply(self, image_embeds):
        if not self.available:
            return None
        return self.adapter.apply(image_embeds)


class MotusPreWeights(WeightModule):
    weight_prefixes = {
        "video": "video_model.wan_model.",
        "action": "action_expert.",
        "und": "und_expert.",
        "image_context": None,
    }

    def __init__(self, config):
        super().__init__()
        expert_configs = build_motus_expert_configs(config)
        self.add_module("video", MotusWanPreWeights(config))
        self.add_module("action", MotusActionPreWeights(config, expert_configs.action))
        self.add_module("und", MotusUndPreWeights(config, expert_configs.und))
        self.add_module("image_context", MotusImageContextWeights(config))

    def load(self, weight_dict):
        load_prefixed_submodules(self, weight_dict, self.weight_prefixes)
