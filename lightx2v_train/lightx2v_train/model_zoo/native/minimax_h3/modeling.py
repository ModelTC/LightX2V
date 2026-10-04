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

    return install_minimax_h3_sequence_parallel(transformer)
