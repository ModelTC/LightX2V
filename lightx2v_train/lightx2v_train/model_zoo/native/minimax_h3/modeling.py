"""Loader for the official trainable PyTorch MiniMax-H3 DiT.

The inference package in :mod:`lightx2v.models.networks.minimax_h3` stores
weights in immutable ``MMWeight`` objects and is intentionally not used here.
Training uses Diffusers' ordinary ``torch.nn.Module`` implementation so PEFT,
autograd, activation checkpointing, and FSDP2 all see real Parameters.
"""

import json
from pathlib import Path

import torch


def _transformer_class():
    try:
        from diffusers import MiniMaxH3Transformer3DModel
    except (ImportError, AttributeError) as exc:
        raise ImportError(
            "MiniMax-H3 training requires a Diffusers build containing MiniMaxH3Transformer3DModel. Use the model's local_diffusers environment or the corresponding upstream Diffusers revision."
        ) from exc
    return MiniMaxH3Transformer3DModel


def resolve_transformer_dir(model_path: str | Path) -> Path:
    """Accept either the converted model root or its transformer directory."""
    path = Path(model_path).expanduser().resolve()
    if (path / "transformer" / "config.json").is_file():
        path = path / "transformer"
    elif not (path / "config.json").is_file():
        raise FileNotFoundError(f"MiniMax-H3 Diffusers transformer config not found below {path}. Use the converted model root containing transformer/config.json.")
    with (path / "config.json").open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    if config.get("_class_name") != "MiniMaxH3Transformer3DModel" or "num_refiner_layers" not in config:
        raise ValueError(
            f"{path} is the original FL2VA transformer layout, not the trainable upstream Diffusers layout. "
            "Point pretrained_model_name_or_path at the converted model root (the directory containing "
            "modular_model_index.json and transformer/config.json)."
        )
    return path


def load_minimax_h3_transformer(
    model_path: str | Path,
    *,
    torch_dtype: torch.dtype | None = None,
    local_files_only: bool = True,
    attention_backend: str | None = None,
):
    """Load the official trainable H3 module without LightX2V MMWeight."""
    transformer_dir = resolve_transformer_dir(model_path)
    cls = _transformer_class()
    transformer = cls.from_pretrained(
        str(transformer_dir),
        torch_dtype=torch_dtype,
        local_files_only=local_files_only,
        low_cpu_mem_usage=True,
    )
    if attention_backend:
        if not hasattr(transformer, "set_attention_backend"):
            raise ValueError(f"Installed Diffusers cannot select attention backend {attention_backend!r} on MiniMax-H3.")
        transformer.set_attention_backend(attention_backend)
    return transformer
