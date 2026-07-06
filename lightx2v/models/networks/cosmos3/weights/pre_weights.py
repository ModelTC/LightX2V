import torch

from lightx2v.common.modules.weight_module import WeightModule
from lightx2v.utils.registry_factory import EMBEDDING_WEIGHT_REGISTER, MM_WEIGHT_REGISTER, TENSOR_REGISTER


class Cosmos3DomainAwareLinearWeights(WeightModule):
    def __init__(self, prefix, input_size, output_size):
        super().__init__()
        self.input_size = int(input_size)
        self.output_size = int(output_size)
        self.register_parameter("fc", TENSOR_REGISTER["Default"](f"{prefix}.fc.weight"))
        self.register_parameter("bias", TENSOR_REGISTER["Default"](f"{prefix}.bias.weight"))

    def apply(self, x, domain_ids):
        domain_ids = domain_ids.to(device=x.device, dtype=torch.long).reshape(-1)
        if x.shape[0] != domain_ids.shape[0]:
            raise ValueError(f"Cosmos3 action domain ids must match token count: {domain_ids.shape[0]} vs {x.shape[0]}")
        weight = self.fc.tensor.to(device=x.device, dtype=x.dtype)[domain_ids].view(-1, self.input_size, self.output_size)
        bias = self.bias.tensor.to(device=x.device, dtype=x.dtype)[domain_ids].view(-1, self.output_size)
        return torch.bmm(x.unsqueeze(1), weight).squeeze(1) + bias


class Cosmos3PreWeights(WeightModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.add_module("embed_tokens", EMBEDDING_WEIGHT_REGISTER["Default"]("embed_tokens.weight"))
        self.add_module("proj_in", MM_WEIGHT_REGISTER["Default"]("proj_in.weight", "proj_in.bias"))
        self.add_module(
            "time_embedder_linear_1",
            MM_WEIGHT_REGISTER["Default"]("time_embedder.linear_1.weight", "time_embedder.linear_1.bias"),
        )
        self.add_module(
            "time_embedder_linear_2",
            MM_WEIGHT_REGISTER["Default"]("time_embedder.linear_2.weight", "time_embedder.linear_2.bias"),
        )
        if config.get("sound_gen", False):
            self.add_module("audio_proj_in", MM_WEIGHT_REGISTER["Default"]("audio_proj_in.weight", "audio_proj_in.bias"))
            self.register_parameter("audio_modality_embed", TENSOR_REGISTER["Default"]("audio_modality_embed"))
        if config.get("action_gen", False):
            self.add_module(
                "action_proj_in",
                Cosmos3DomainAwareLinearWeights(
                    "action_proj_in",
                    input_size=config.get("action_dim", config.get("max_action_dim", 64)),
                    output_size=config["hidden_size"],
                ),
            )
            self.register_parameter("action_modality_embed", TENSOR_REGISTER["Default"]("action_modality_embed"))
