"""Memory-bounded pretrained loading for FSDP2-sharded MiniMax-H3.

MiniMax-H3's transformer checkpoint is about 62 GiB.  Loading a complete
fake/teacher transformer on CUDA before applying FSDP2 transiently stacks that
full copy on top of the already-sharded roles.  This module instead expects a
transformer constructed on ``meta`` and already wrapped by FSDP2.  It
materializes only local DTensor shards, then streams the fourteen safetensor
files through global rank 0.  PyTorch distributes every full tensor directly
into each rank's local DP shard; separate SP replicas receive identical data.
"""

from __future__ import annotations

import gc
import json
from collections.abc import Mapping
from pathlib import Path

import torch
import torch.distributed as dist
from loguru import logger
from safetensors import safe_open
from safetensors.torch import load_file
from torch.distributed.checkpoint.state_dict import StateDictOptions, set_model_state_dict

_SAFE_INDEX_NAMES = (
    "diffusion_pytorch_model.safetensors.index.json",
    "model.safetensors.index.json",
)
_SAFE_WEIGHT_NAMES = (
    "diffusion_pytorch_model.safetensors",
    "model.safetensors",
)


def _checkpoint_weight_map(transformer_dir: Path) -> tuple[dict[str, str], tuple[str, ...]]:
    for filename in _SAFE_INDEX_NAMES:
        index_path = transformer_dir / filename
        if not index_path.is_file():
            continue
        with index_path.open("r", encoding="utf-8") as handle:
            index = json.load(handle)
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError(f"Invalid safetensors weight_map in {index_path}.")
        normalized = {str(key): str(value) for key, value in weight_map.items()}
        shard_names = tuple(dict.fromkeys(normalized.values()))
        missing_files = [name for name in shard_names if not (transformer_dir / name).is_file()]
        if missing_files:
            raise FileNotFoundError(f"MiniMax-H3 checkpoint index {index_path} references missing shards: {missing_files}.")
        return normalized, shard_names

    for filename in _SAFE_WEIGHT_NAMES:
        weight_path = transformer_dir / filename
        if not weight_path.is_file():
            continue
        with safe_open(str(weight_path), framework="pt", device="cpu") as handle:
            return {str(key): filename for key in handle.keys()}, (filename,)

    raise FileNotFoundError(f"No sharded or single-file safetensors checkpoint found in {transformer_dir}.")


def _checkpoint_to_target_keys(
    target_state_keys: set[str],
    checkpoint_keys: set[str],
) -> dict[str, str]:
    """Map original Linear keys to PEFT's ``base_layer`` names when needed."""
    target_base_keys = {key for key in target_state_keys if ".lora_" not in key}
    checkpoint_to_target: dict[str, str] = {}
    for target_key in target_base_keys:
        checkpoint_key = target_key.replace(".base_layer.", ".")
        existing = checkpoint_to_target.setdefault(checkpoint_key, target_key)
        if existing != target_key:
            raise ValueError(f"MiniMax-H3 PEFT key mapping is ambiguous: {checkpoint_key!r} maps to both {existing!r} and {target_key!r}.")

    missing_targets = sorted(checkpoint_keys - checkpoint_to_target.keys())
    unexpected_targets = sorted(checkpoint_to_target.keys() - checkpoint_keys)
    if missing_targets or unexpected_targets:
        raise ValueError(f"MiniMax-H3 streamed checkpoint keys do not match the meta model: checkpoint_only={missing_targets[:10]}, model_only={unexpected_targets[:10]}.")
    return checkpoint_to_target


def _rebuild_nonpersistent_buffers(transformer: torch.nn.Module, device: torch.device) -> None:
    """Recreate H3's sole computed, nonpersistent RoPE buffer."""
    rope = transformer.rope
    rope_freq_dim = int(transformer.config.rope_freq_dim)
    rope_theta = float(transformer.config.rope_theta)
    inv_freq = 1.0 / (
        rope_theta
        ** (
            torch.arange(
                0,
                2 * rope_freq_dim,
                2,
                dtype=torch.float32,
                device=device,
            )
            / (2 * rope_freq_dim)
        )
    )
    rope.register_buffer("inv_freq", inv_freq, persistent=False)


def _lora_init_state(
    transformer: torch.nn.Module,
    *,
    seed: int,
    rank_zero: bool,
) -> dict[str, torch.Tensor]:
    """Build PEFT's Gaussian-A/zero-B initialization on global rank zero."""
    if not rank_zero:
        return {}
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    state: dict[str, torch.Tensor] = {}
    for name, parameter in transformer.named_parameters():
        shape = tuple(parameter.shape)
        if ".lora_A." in name:
            if len(shape) != 2 or shape[0] <= 0:
                raise ValueError(f"Unexpected MiniMax-H3 LoRA-A shape for {name}: {shape}.")
            value = torch.empty(shape, dtype=parameter.dtype, device="cpu")
            value.normal_(mean=0.0, std=1.0 / shape[0], generator=generator)
            state[name] = value
        elif ".lora_B." in name:
            state[name] = torch.zeros(shape, dtype=parameter.dtype, device="cpu")
    return state


def _load_partial_full_state(
    transformer: torch.nn.Module,
    state: Mapping[str, torch.Tensor],
) -> None:
    # All global ranks participate. Rank zero owns the full CPU tensors; the
    # others pass an empty mapping and receive one tensor at a time before
    # PyTorch slices it according to each model's local DP DeviceMesh.
    options = StateDictOptions(
        full_state_dict=True,
        broadcast_from_rank0=True,
        strict=False,
    )
    set_model_state_dict(transformer, dict(state), options=options)


@torch.no_grad()
def stream_load_minimax_h3_transformer(
    transformer: torch.nn.Module,
    transformer_dir: str | Path,
    *,
    device: torch.device,
    lora_seed: int = 0,
) -> None:
    """Materialize and populate an FSDP2-sharded, meta-built H3 transformer."""
    if not dist.is_available() or not dist.is_initialized() or dist.get_world_size() <= 1:
        raise RuntimeError("MiniMax-H3 streamed loading requires an initialized distributed process group.")

    transformer_dir = Path(transformer_dir).expanduser().resolve()
    weight_map, shard_names = _checkpoint_weight_map(transformer_dir)

    # FSDP2's Module.to_empty implementation preserves DTensor placements and
    # allocates only this rank's local shards, never a complete H3 copy.
    transformer.to_empty(device=device)
    _rebuild_nonpersistent_buffers(transformer, device)

    target_keys = set(transformer.state_dict().keys())
    key_map = _checkpoint_to_target_keys(target_keys, set(weight_map.keys()))
    rank_zero = dist.get_rank() == 0

    logger.info(
        "MiniMax-H3 FSDP2 streamed load: shards={} source={} rank0_only_disk_io=true",
        len(shard_names),
        transformer_dir,
    )
    for shard_index, shard_name in enumerate(shard_names, start=1):
        if rank_zero:
            raw_state = load_file(str(transformer_dir / shard_name), device="cpu")
            state = {key_map[key]: value for key, value in raw_state.items()}
        else:
            raw_state = None
            state = {}
        _load_partial_full_state(transformer, state)
        del state, raw_state
        gc.collect()
        if rank_zero:
            logger.info(
                "MiniMax-H3 FSDP2 streamed load shard {}/{}: {}",
                shard_index,
                len(shard_names),
                shard_name,
            )

    lora_state = _lora_init_state(transformer, seed=lora_seed, rank_zero=rank_zero)
    if lora_state or any(".lora_" in key for key in target_keys):
        _load_partial_full_state(transformer, lora_state)
    del lora_state
    gc.collect()

    meta_parameters = [name for name, parameter in transformer.named_parameters() if parameter.is_meta]
    meta_buffers = [name for name, buffer in transformer.named_buffers() if buffer.is_meta]
    if meta_parameters or meta_buffers:
        raise RuntimeError(f"MiniMax-H3 FSDP2 streamed loading left tensors on meta: parameters={meta_parameters[:10]}, buffers={meta_buffers[:10]}.")
