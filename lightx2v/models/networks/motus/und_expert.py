import re
from dataclasses import dataclass

import torch
import torch.nn as nn

from .primitives import WanLayerNorm, WanRMSNorm


@dataclass
class UndExpertConfig:
    dim: int = 512
    ffn_dim: int = 2048
    num_layers: int = 30
    vlm_input_dim: int = 2048
    vlm_projector_type: str = "mlp3x_silu"
    eps: float = 1e-5


def build_condition_adapter(projector_type, in_features, out_features):
    if projector_type == "linear":
        return nn.Linear(in_features, out_features)
    mlp_silu_match = re.match(r"^mlp(\d+)x_silu$", projector_type)
    if mlp_silu_match:
        mlp_depth = int(mlp_silu_match.group(1))
        modules = [nn.Linear(in_features, out_features)]
        for _ in range(1, mlp_depth):
            modules.append(nn.SiLU())
            modules.append(nn.Linear(out_features, out_features))
        return nn.Sequential(*modules)
    raise ValueError(f"Unknown projector type: {projector_type}")


class UndExpertBlock(nn.Module):
    def __init__(self, config: UndExpertConfig, wan_config: dict):
        super().__init__()
        self.norm1 = WanLayerNorm(config.dim, eps=config.eps)
        self.norm2 = WanLayerNorm(config.dim, eps=config.eps)
        self.wan_num_heads = wan_config["num_heads"]
        self.wan_head_dim = wan_config["head_dim"]
        self.wan_dim = wan_config["dim"]
        self.wan_und_qkv = nn.Parameter(torch.randn(3, self.wan_num_heads, config.dim, self.wan_head_dim) / (config.dim * self.wan_head_dim) ** 0.5)
        self.wan_und_o = nn.Linear(self.wan_dim, config.dim, bias=False)
        self.wan_und_norm_q = WanRMSNorm(self.wan_dim, eps=config.eps)
        self.wan_und_norm_k = WanRMSNorm(self.wan_dim, eps=config.eps)
        self.ffn = nn.Sequential(nn.Linear(config.dim, config.ffn_dim), nn.GELU(approximate="tanh"), nn.Linear(config.ffn_dim, config.dim))


class UndExpert(nn.Module):
    def __init__(self, config: UndExpertConfig, wan_config: dict = None, vlm_config: dict = None):
        super().__init__()
        self.config = config
        self.freq_dim = 256
        self.vlm_adapter = build_condition_adapter(config.vlm_projector_type, config.vlm_input_dim, config.dim)
        block_cfg = wan_config or {"dim": 3072, "num_heads": 24, "head_dim": 128}
        self.blocks = nn.ModuleList([UndExpertBlock(config, block_cfg) for _ in range(config.num_layers)])
