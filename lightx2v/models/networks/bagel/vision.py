from dataclasses import dataclass

import torch
from torch import nn
from transformers.activations import ACT2FN

try:
    import flash_attn  # noqa: F401
    from flash_attn.flash_attn_interface import flash_attn_varlen_func
except ImportError:
    flash_attn_varlen_func = None


@dataclass
class BagelSiglipVisionConfig:
    hidden_size: int = 1152
    image_size: int = 980
    intermediate_size: int = 4304
    num_attention_heads: int = 16
    num_hidden_layers: int = 26
    patch_size: int = 14
    num_channels: int = 3
    hidden_act: str = "gelu_pytorch_tanh"
    layer_norm_eps: float = 1e-6
    attention_dropout: float = 0.0
    rope: bool = False

    @classmethod
    def from_dict(cls, config):
        values = dict(config)
        values.pop("model_type", None)
        allowed = cls.__dataclass_fields__
        return cls(**{key: value for key, value in values.items() if key in allowed})


class BagelSiglipVisionEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.patch_embedding = nn.Linear(config.num_channels * config.patch_size**2, config.hidden_size)
        self.position_embedding = nn.Embedding((config.image_size // config.patch_size) ** 2, config.hidden_size)

    def forward(self, packed_pixel_values, packed_flattened_position_ids):
        embeddings = self.patch_embedding(packed_pixel_values)
        embeddings = embeddings + self.position_embedding(packed_flattened_position_ids)
        return embeddings


class BagelSiglipAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        if flash_attn_varlen_func is None:
            raise ImportError("BAGEL I2I requires flash-attn (`flash_attn`) for the SigLIP vision encoder. Install a flash-attn build compatible with your CUDA/PyTorch environment.")
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(f"hidden_size must be divisible by num_attention_heads, got {self.embed_dim} and {self.num_heads}")

        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

    def forward(self, hidden_states, cu_seqlens, max_seqlen):
        total_q_len, _ = hidden_states.size()
        query_states = self.q_proj(hidden_states).view(total_q_len, self.num_heads, self.head_dim)
        key_states = self.k_proj(hidden_states).view(total_q_len, self.num_heads, self.head_dim)
        value_states = self.v_proj(hidden_states).view(total_q_len, self.num_heads, self.head_dim)

        attn_output = flash_attn_varlen_func(
            query_states.to(torch.bfloat16),
            key_states.to(torch.bfloat16),
            value_states.to(torch.bfloat16),
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_k=cu_seqlens,
            max_seqlen_q=max_seqlen,
            max_seqlen_k=max_seqlen,
            causal=False,
        )
        return self.out_proj(attn_output.reshape(total_q_len, -1))


class BagelSiglipMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.activation_fn = ACT2FN[config.hidden_act]
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

    def forward(self, hidden_states):
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states


class BagelSiglipEncoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self_attn = BagelSiglipAttention(config)
        self.layer_norm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.mlp = BagelSiglipMLP(config)
        self.layer_norm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states, cu_seqlens, max_seqlen):
        residual = hidden_states
        hidden_states = self.layer_norm1(hidden_states)
        hidden_states = self.self_attn(hidden_states, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        return residual + hidden_states


class BagelSiglipEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layers = nn.ModuleList([BagelSiglipEncoderLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(self, inputs_embeds, cu_seqlens, max_seqlen):
        hidden_states = inputs_embeds
        for encoder_layer in self.layers:
            hidden_states = encoder_layer(hidden_states, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen)
        return hidden_states


class BagelSiglipVisionTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embeddings = BagelSiglipVisionEmbeddings(config)
        self.encoder = BagelSiglipEncoder(config)
        self.post_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, packed_pixel_values, packed_flattened_position_ids, cu_seqlens, max_seqlen):
        hidden_states = self.embeddings(
            packed_pixel_values=packed_pixel_values,
            packed_flattened_position_ids=packed_flattened_position_ids,
        )
        hidden_states = self.encoder(hidden_states, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen)
        return self.post_layernorm(hidden_states)


class BagelSiglipVisionModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.vision_model = BagelSiglipVisionTransformer(config)

    def forward(self, packed_pixel_values, packed_flattened_position_ids, cu_seqlens, max_seqlen):
        device = next(self.parameters()).device
        dtype = next(self.parameters()).dtype
        return self.vision_model(
            packed_pixel_values=packed_pixel_values.to(device=device, dtype=dtype),
            packed_flattened_position_ids=packed_flattened_position_ids.to(device),
            cu_seqlens=cu_seqlens.to(device=device, dtype=torch.int32),
            max_seqlen=max_seqlen,
        )


def infer_bagel_vit_layer_count(weight_dict):
    prefix = "vit_model.vision_model.encoder.layers."
    layer_ids = set()
    for key in weight_dict:
        if key.startswith(prefix):
            layer_ids.add(int(key[len(prefix) :].split(".", 1)[0]))
    if not layer_ids:
        raise ValueError("BAGEL ema.safetensors is missing ViT encoder weights (`vit_model.vision_model.encoder.layers.*`) required for task='i2i'.")
    return max(layer_ids) + 1


def extract_bagel_vit_state_dict(weight_dict):
    state_dict = {key[len("vit_model.") :]: value for key, value in weight_dict.items() if key.startswith("vit_model.")}
    if not state_dict:
        raise ValueError("BAGEL ema.safetensors is missing ViT weights (`vit_model.*`) required for task='i2i'.")
    return state_dict


def build_bagel_vit_config(config, weight_dict=None):
    if "vit_config" not in config:
        raise ValueError("BAGEL I2I requires `vit_config` from the BAGEL model config.")
    vit_config = BagelSiglipVisionConfig.from_dict(config["vit_config"])
    vit_config.rope = False
    if weight_dict is not None:
        vit_config.num_hidden_layers = infer_bagel_vit_layer_count(weight_dict)
    elif vit_config.num_hidden_layers == 27:
        # BAGEL's published config.json reports 27 layers, while the official checkpoint
        # ships 26 SigLIP encoder blocks. Keep the fallback aligned with the checkpoint.
        vit_config.num_hidden_layers = 26
    return vit_config


def load_bagel_vit_model(config, weight_dict):
    vit_state_dict = extract_bagel_vit_state_dict(weight_dict)
    vit_config = build_bagel_vit_config(config, weight_dict=weight_dict)
    model = BagelSiglipVisionModel(vit_config)
    missing, unexpected = model.load_state_dict(vit_state_dict, strict=False, assign=True)
    if missing:
        raise ValueError(f"BAGEL ViT weights are incomplete for task='i2i'; missing key(s): {missing[:8]}")
    if unexpected:
        raise ValueError(f"BAGEL ViT weights do not match the local SigLIP encoder; unexpected key(s): {unexpected[:8]}")
    model.eval()
    model.requires_grad_(False)
    return model
