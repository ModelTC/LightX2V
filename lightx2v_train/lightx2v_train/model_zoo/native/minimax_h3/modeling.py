"""Loader for the official trainable PyTorch MiniMax-H3 DiT.

The inference package in :mod:`lightx2v.models.networks.minimax_h3` stores
weights in immutable ``MMWeight`` objects and is intentionally not used here.
Training uses Diffusers' ordinary ``torch.nn.Module`` implementation so PEFT,
autograd, activation checkpointing, and FSDP2 all see real Parameters.
"""

import json
from pathlib import Path
from types import MethodType

import torch
from diffusers.models.modeling_utils import get_parameter_dtype


def _transformer_class():
    try:
        from diffusers import MiniMaxH3Transformer3DModel
    except (ImportError, AttributeError) as exc:
        raise ImportError(
            "MiniMax-H3 training requires a Diffusers build containing MiniMaxH3Transformer3DModel. Use the model's local_diffusers environment or the corresponding upstream Diffusers revision."
        ) from exc
    return MiniMaxH3Transformer3DModel


def resolve_transformer_dir(model_path: str | Path, component: str = "transformer") -> Path:
    """Accept either the converted model root or its transformer directory."""
    path = Path(model_path).expanduser().resolve()
    if (path / component / "config.json").is_file():
        path = path / component
    elif not (path / "config.json").is_file():
        raise FileNotFoundError(f"MiniMax-H3 Diffusers config not found at {path / component / 'config.json'}.")
    with (path / "config.json").open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    if config.get("_class_name") != "MiniMaxH3Transformer3DModel" or "num_refiner_layers" not in config:
        raise ValueError(
            f"{path} is the original FL2VA transformer layout, not the trainable upstream Diffusers layout. "
            f"Point pretrained_model_name_or_path at the converted model root containing {component}/config.json."
        )
    return path


def load_minimax_h3_transformer(
    model_path: str | Path,
    *,
    component: str = "transformer",
    torch_dtype: torch.dtype | None = None,
    local_files_only: bool = True,
    attention_backend: str | None = None,
    defer_weights: bool = False,
):
    """Load the official trainable H3 module without LightX2V MMWeight."""
    transformer_dir = resolve_transformer_dir(model_path, component)
    cls = _transformer_class()
    if defer_weights:
        config = cls.load_config(str(transformer_dir))
        previous_dtype = cls._set_default_torch_dtype(torch_dtype or torch.get_default_dtype())
        try:
            with torch.device("meta"):
                transformer = cls.from_config(config)
        finally:
            torch.set_default_dtype(previous_dtype)
        for name, module in transformer.named_modules():
            if name and any(part in name for part in transformer._keep_in_fp32_modules):
                module.float()
    else:
        transformer = cls.from_pretrained(
            str(transformer_dir),
            torch_dtype=torch_dtype,
            local_files_only=local_files_only,
            low_cpu_mem_usage=True,
        )
    if attention_backend:
        transformer.set_attention_backend(attention_backend)
    transformer._velocity_forward = transformer.forward
    transformer.forward = MethodType(_forward, transformer)
    return transformer


def _forward(self, *args, feature_block=None, target_video_indices=None, **kwargs):
    if feature_block is None:
        return self._velocity_forward(*args, **kwargs)
    return _video_features(self, *args, feature_block=feature_block, target_video_indices=target_video_indices, **kwargs)


def _video_features(
    model,
    hidden_states,
    audio_hidden_states,
    encoder_hidden_states,
    timestep,
    timestep_indices,
    token_tags,
    position_ids,
    video_indices,
    audio_indices,
    text_indices,
    return_dict,
    *,
    feature_block,
    target_video_indices,
):
    from diffusers.models.transformers.transformer_minimax_h3 import MINIMAX_H3_MODALITY_NUM

    if not 0 <= feature_block < len(model.transformer_blocks):
        raise ValueError(f"H3 GAN feature block {feature_block} is out of range.")
    video = model.proj_in(hidden_states.to(get_parameter_dtype(model.proj_in)))
    audio = model.audio_proj_in(audio_hidden_states.to(get_parameter_dtype(model.audio_proj_in)))
    text = model.context_embedder(encoder_hidden_states.to(get_parameter_dtype(model.context_embedder)))
    text = model.token_refiner(text)
    packed = text.new_zeros((text.shape[0], position_ids.shape[0], text.shape[-1]))
    packed = packed.index_copy(1, text_indices, text)
    packed = packed.index_copy(1, video_indices, video.to(text.dtype))
    packed = packed.index_copy(1, audio_indices, audio.to(text.dtype))
    rotary = model.rope(position_ids)
    temb = model.time_embedder(model.time_proj(timestep).to(get_parameter_dtype(model.time_embedder)))
    adaln_indices = timestep_indices * MINIMAX_H3_MODALITY_NUM + token_tags
    for block in model.transformer_blocks[: feature_block + 1]:
        if torch.is_grad_enabled() and model.gradient_checkpointing:
            packed = model._gradient_checkpointing_func(block, packed, temb, adaln_indices, rotary)
        else:
            packed = block(packed, temb, adaln_indices, rotary)
    # Return features from the FSDP root so it installs its backward hooks.
    return packed.index_select(1, target_video_indices)
