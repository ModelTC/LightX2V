"""Checkpoint selection and key normalization for DA3MONO-LARGE."""

from pathlib import Path

import torch


def resolve_checkpoint(model_path: Path) -> Path:
    if model_path.is_dir():
        candidates = [model_path / name for name in ("model.safetensors", "depth_anything_3_mono_large.safetensors")]
        found = [path for path in candidates if path.is_file()]
        if len(found) > 1:
            raise ValueError(f"Multiple DA3MONO-LARGE checkpoints found in {model_path}; pass the desired safetensors file explicitly")
        if not found:
            raise FileNotFoundError(f"DA3MONO-LARGE checkpoint not found in {model_path}; expected model.safetensors or depth_anything_3_mono_large.safetensors")
        return found[0]
    if not model_path.is_file() or model_path.suffix != ".safetensors":
        raise FileNotFoundError(f"DA3MONO-LARGE safetensors checkpoint not found: {model_path}")
    return model_path


def normalize_checkpoint_weights(weights: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    official = any(key.startswith("model.backbone.pretrained.") for key in weights)
    if not official:
        state = {key.removeprefix("model."): value for key, value in weights.items()}
        if len(state) != len(weights):
            raise ValueError("Duplicate DA3 checkpoint keys after removing the 'model.' prefix")
        return state

    state = {}

    def add(key: str, value: torch.Tensor) -> None:
        if key in state:
            raise ValueError(f"Duplicate DA3 checkpoint key after official weight conversion: {key}")
        state[key] = value

    for key, value in weights.items():
        if key.startswith("model.head."):
            add(key.removeprefix("model."), value)
            continue
        if not key.startswith("model.backbone.pretrained."):
            raise ValueError(f"Unexpected key in official DA3MONO-LARGE checkpoint: {key}")
        suffix = key.removeprefix("model.backbone.pretrained.")
        backbone_keys = {
            "cls_token": "embeddings.cls_token",
            "pos_embed": "embeddings.position_embeddings",
            "norm.weight": "layernorm.weight",
            "norm.bias": "layernorm.bias",
        }
        if suffix in backbone_keys:
            add("backbone." + backbone_keys[suffix], value)
        elif suffix.startswith("patch_embed.proj."):
            add("backbone.embeddings.patch_embeddings.projection." + suffix.removeprefix("patch_embed.proj."), value)
        elif suffix.startswith("blocks."):
            _, index, block_key = suffix.split(".", 2)
            if not index.isdecimal():
                raise ValueError(f"Invalid DA3 backbone block index in {key}")
            prefix = f"backbone.encoder.layer.{index}."
            if block_key in ("attn.qkv.weight", "attn.qkv.bias"):
                for name, part in zip(("query", "key", "value"), value.chunk(3, dim=0), strict=True):
                    add(prefix + f"attention.attention.{name}.{block_key.rsplit('.', 1)[-1]}", part)
            elif block_key.startswith("attn.proj."):
                add(prefix + "attention.output.dense." + block_key.removeprefix("attn.proj."), value)
            elif block_key == "ls1.gamma":
                add(prefix + "layer_scale1.lambda1", value)
            elif block_key == "ls2.gamma":
                add(prefix + "layer_scale2.lambda1", value)
            elif block_key.startswith(("mlp.", "norm1.", "norm2.")):
                add(prefix + block_key, value)
            else:
                raise ValueError(f"Unexpected key in official DA3MONO-LARGE checkpoint: {key}")
        else:
            raise ValueError(f"Unexpected key in official DA3MONO-LARGE checkpoint: {key}")
    return state
