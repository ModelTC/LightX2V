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


def _transformer_class():
    try:
        from diffusers import MiniMaxH3Transformer3DModel
    except (ImportError, AttributeError) as exc:
        raise ImportError(
            "MiniMax-H3 training requires a Diffusers build containing MiniMaxH3Transformer3DModel. Use the model's local_diffusers environment or the corresponding upstream Diffusers revision."
        ) from exc
    return MiniMaxH3Transformer3DModel


def resolve_transformer_dir(
    model_path: str | Path,
    component_name: str = "transformer",
) -> Path:
    """Accept either the converted model root or a requested transformer directory."""
    if component_name not in {"transformer", "transformer_ref"}:
        raise ValueError(f"MiniMax-H3 transformer component must be 'transformer' or 'transformer_ref', got {component_name!r}.")
    path = Path(model_path).expanduser().resolve()
    if (path / component_name / "config.json").is_file():
        path = path / component_name
    elif not (path / "config.json").is_file():
        raise FileNotFoundError(f"MiniMax-H3 Diffusers transformer config not found below {path}. Use the converted model root containing {component_name}/config.json.")
    with (path / "config.json").open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    if config.get("_class_name") != "MiniMaxH3Transformer3DModel" or "num_refiner_layers" not in config:
        raise ValueError(
            f"{path} is the original FL2VA transformer layout, not the trainable upstream Diffusers layout. "
            "Point pretrained_model_name_or_path at the converted model root (the directory containing "
            f"modular_model_index.json and {component_name}/config.json)."
        )
    return path


def load_minimax_h3_transformer(
    model_path: str | Path,
    *,
    component_name: str = "transformer",
    torch_dtype: torch.dtype | None = None,
    local_files_only: bool = True,
    attention_backend: str | None = None,
):
    """Load the official trainable H3 module without LightX2V MMWeight."""
    transformer_dir = resolve_transformer_dir(
        model_path,
        component_name=component_name,
    )
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
    return _install_sequence_parallel(transformer)


def init_empty_minimax_h3_transformer(
    model_path: str | Path,
    *,
    component_name: str = "transformer",
    torch_dtype: torch.dtype | None = None,
    local_files_only: bool = True,
    attention_backend: str | None = None,
):
    """Construct H3 on ``meta`` for FSDP2-first checkpoint loading.

    The official checkpoint is mixed precision: most weights use bfloat16,
    while the input/output projections and timestep MLP remain float32.  We
    mirror Diffusers' ``from_pretrained(torch_dtype=...)`` construction here
    without materializing any parameter.  Checkpoint tensors are loaded only
    after FSDP2 has replaced the full parameters with per-rank DTensor shards.

    This function intentionally does *not* support an unsharded caller.  A
    meta model cannot run until ``stream_load_minimax_h3_transformer`` has been
    called after FSDP2 wrapping.
    """
    transformer_dir = resolve_transformer_dir(
        model_path,
        component_name=component_name,
    )
    cls = _transformer_class()
    config = cls.load_config(str(transformer_dir), local_files_only=local_files_only)
    requested_dtype = torch_dtype or torch.get_default_dtype()
    original_dtype = cls._set_default_torch_dtype(requested_dtype)
    try:
        with torch.device("meta"):
            transformer = cls.from_config(config)
    finally:
        torch.set_default_dtype(original_dtype)

    # Diffusers keeps these modules in fp32 when loading the mixed-precision
    # H3 checkpoint.  Setting their *meta* dtype now means FSDP creates shards
    # with exactly the checkpoint dtype before the streamed copies arrive.
    fp32_patterns = tuple(getattr(transformer, "_keep_in_fp32_modules", ()) or ())
    for module_name, module in transformer.named_modules():
        if module_name and any(pattern in module_name for pattern in fp32_patterns):
            module.to(dtype=torch.float32)

    transformer.register_to_config(_name_or_path=str(transformer_dir))
    transformer.eval()
    if attention_backend:
        if not hasattr(transformer, "set_attention_backend"):
            raise ValueError(f"Installed Diffusers cannot select attention backend {attention_backend!r} on MiniMax-H3.")
        transformer.set_attention_backend(attention_backend)
    return _install_sequence_parallel(transformer)


def _install_sequence_parallel(transformer):
    """Install H3 SP processors when the distributed runtime requests them."""
    # Kept lazy so ordinary single-process imports do not pull in distributed
    # attention code, and so the same finalization is shared by materialized
    # and meta-first loading.
    from .sequence_parallel import install_minimax_h3_sequence_parallel

    return install_minimax_h3_dmad_features(install_minimax_h3_sequence_parallel(transformer))


def _minimax_h3_dmad_feature_forward(
    self,
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
    attention_kwargs=None,
    return_dict=False,
    *,
    return_features_block,
    feature_video_indices=None,
    feature_audio_indices=None,
):
    """Return selected block features through the FSDP root's ordinary output.

    This is the dense upstream H3 forward up to its selected residual block.
    It deliberately does not run output AdaLN/projections, detach tensors, or
    use hooks/exceptions: the generator needs gradients through a frozen critic.
    """
    from diffusers.models.transformers.transformer_minimax_h3 import MINIMAX_H3_MODALITY_NUM

    from lightx2v_train.runtime.distributed import get_sequence_parallel_world_size

    del attention_kwargs  # Consumed by apply_lora_scale, as in upstream H3.
    if get_sequence_parallel_world_size() != 1:
        raise ValueError("H3 DMAD critic features require sequence_parallel.size=1.")
    if return_dict:
        raise ValueError("H3 DMAD critic features require return_dict=False.")
    block_index = int(return_features_block)
    if block_index != return_features_block or not 0 <= block_index < len(self.transformer_blocks):
        raise ValueError(f"Invalid H3 DMAD feature block {return_features_block!r}.")
    if position_ids.ndim != 2 or position_ids.shape[-1] != 3:
        raise ValueError("H3 position_ids must have shape [sequence, 3].")
    length = position_ids.shape[0]
    if token_tags.shape != (length,) or timestep_indices.shape != (length,):
        raise ValueError("H3 token_tags and timestep_indices must match the packed sequence.")

    rotary_emb = self.rope(position_ids)
    video_embeds = self.proj_in(hidden_states.to(self.proj_in.weight.dtype))
    audio_embeds = self.audio_proj_in(audio_hidden_states.to(self.audio_proj_in.weight.dtype))
    text_embeds = self.context_embedder(encoder_hidden_states.to(self.context_embedder.weight.dtype))
    text_embeds = self.token_refiner(text_embeds)
    packed = text_embeds.new_zeros((text_embeds.shape[0], length, text_embeds.shape[-1]))
    packed = packed.index_copy(1, text_indices, text_embeds)
    packed = packed.index_copy(1, video_indices, video_embeds.to(text_embeds.dtype))
    packed = packed.index_copy(1, audio_indices, audio_embeds.to(text_embeds.dtype))
    temb = self.time_proj(timestep)
    temb = self.time_embedder(temb.to(self.time_embedder.linear_1.weight.dtype))
    adaln_indices = timestep_indices * MINIMAX_H3_MODALITY_NUM + token_tags.clamp(min=0)
    is_pad = token_tags < 0
    attention_mask = is_pad[None, :] == is_pad[:, None] if bool(is_pad.any()) else None
    for block in self.transformer_blocks[: block_index + 1]:
        if torch.is_grad_enabled() and self.gradient_checkpointing:
            packed = self._gradient_checkpointing_func(block, packed, temb, adaln_indices, rotary_emb, attention_mask)
        else:
            packed = block(packed, temb, adaln_indices, rotary_emb, attention_mask)
    video_rows = video_indices if feature_video_indices is None else feature_video_indices
    audio_rows = audio_indices if feature_audio_indices is None else feature_audio_indices
    return packed.index_select(1, video_rows), packed.index_select(1, audio_rows)


def _minimax_h3_optional_dmad_forward(self, *args, return_features_block=None, **kwargs):
    if return_features_block is None:
        return self._lightx2v_h3_original_forward(*args, **kwargs)
    return self._lightx2v_h3_feature_forward(*args, return_features_block=return_features_block, **kwargs)


def install_minimax_h3_dmad_features(transformer):
    """Install an opt-in differentiable feature path before FSDP wrapping.

    Ordinary denoising (including SP) is delegated to the unmodified forward.
    The feature path currently supports only dense/SP1 execution.
    """
    if getattr(transformer, "_lightx2v_h3_dmad_features", False):
        return transformer
    from diffusers.utils import apply_lora_scale

    transformer._lightx2v_h3_original_forward = transformer.forward
    feature_forward = apply_lora_scale("attention_kwargs")(_minimax_h3_dmad_feature_forward)
    transformer._lightx2v_h3_feature_forward = MethodType(feature_forward, transformer)
    transformer.forward = MethodType(_minimax_h3_optional_dmad_forward, transformer)
    transformer._lightx2v_h3_dmad_features = True
    return transformer
