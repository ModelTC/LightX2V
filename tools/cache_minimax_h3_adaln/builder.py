import json
import os
import shutil
import tempfile
from pathlib import Path

import torch
import torch.nn.functional as F
from loguru import logger
from safetensors import safe_open
from safetensors.torch import save_file

from lightx2v.models.networks.minimax_h3.adaln_cache import (
    _block_key,
    _expected_table_shape,
    _norm_out_key,
    _timesteps_from_bits,
    _validate_cache,
    checkpoint_files,
    resolve_adaln_cache,
)
from lightx2v.models.networks.minimax_h3.adaln_profiles import cache_plan
from lightx2v.models.networks.minimax_h3.infer.pre_infer import timestep_embedding
from lightx2v_platform.base.global_var import AI_DEVICE


class _CheckpointTensors:
    """Read individual tensors without materializing the whole checkpoint."""

    def __init__(self, files: list[Path], layout):
        self.files = files
        self.tensors = torch.load(files[0], mmap=True, weights_only=True, map_location="cpu") if len(files) == 1 and files[0].suffix == ".pt" else None
        self.locations = self._find_locations() if self.tensors is None else {}
        self.layout = layout
        if layout == "streamrender_release":
            from lightx2v.models.networks.minimax_h3.streamrender.weights import native_weight_names

            self.native_to_source = {native: source for source in self.locations for native in native_weight_names(source)}
        elif layout == "minimax_h3_image":
            from lightx2v.models.networks.minimax_h3_image.checkpoint import convert_transformer_tensor

            # These projections only rename keys; native checkpoints already
            # use the requested names and need no conversion.
            self.native_to_source = {
                native: source
                for source in self.locations
                if source.startswith(("time_embedder.", "final_layer.adaln_proj.linear.")) or (source.startswith("blocks.") and ".adaln_proj.linear." in source)
                for native in convert_transformer_tensor(source, torch.empty(0), {})
            }

    def _find_locations(self) -> dict[str, Path]:
        # Read only the selected files' headers; a neighboring index may
        # describe a different checkpoint.
        locations = {}
        for path in self.files:
            with safe_open(path, framework="pt", device="cpu") as source:
                for name in source.keys():
                    if name in locations:
                        raise ValueError(f"Duplicate checkpoint tensor {name!r}: {locations[name]} and {path}")
                    locations[name] = path
        return locations

    def get(self, name: str, *, dtype=torch.bfloat16) -> torch.Tensor:
        if self.tensors is not None:
            tensor = self.tensors[name]
        else:
            source_name = self.native_to_source.get(name, name) if self.layout in ("streamrender_release", "minimax_h3_image") else name
            path = self.locations[source_name]
            with safe_open(path, framework="pt", device="cpu") as source:
                tensor = source.get_tensor(source_name)
            if self.layout == "streamrender_release":
                from lightx2v.models.networks.minimax_h3.streamrender.weights import remap_streamrender_weight

                tensor = remap_streamrender_weight(source_name, tensor)[name]
        return tensor.to(device=AI_DEVICE, dtype=dtype)


def _linear(
    input_tensor: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    # ADALN CACHE SYNC: Match the transpose, bias, output dtype, and operation
    # order of the online Default MMWeight.apply path. Do not change this
    # independently of MiniMax-H3's online modulation projections.
    output = torch.empty(
        (input_tensor.shape[0], weight.shape[0]),
        dtype=input_tensor.dtype,
        device=input_tensor.device,
    )
    return torch.addmm(bias, input_tensor, weight.t(), out=output)


@torch.inference_mode()
def build_cache_tables(spec: dict, checkpoint: _CheckpointTensors) -> dict[str, torch.Tensor]:
    """Compute block AdaLN and final-norm tables, returning CPU tensors."""
    # ADALN CACHE SYNC: Keep activation placement and casts aligned with
    # the three online infer modules named in this file's contract.
    time_weight_1 = checkpoint.get("time_embedder.linear_1.weight", dtype=torch.float32)
    time_bias_1 = checkpoint.get("time_embedder.linear_1.bias", dtype=torch.float32)
    time_weight_2 = checkpoint.get("time_embedder.linear_2.weight", dtype=torch.float32)
    time_bias_2 = checkpoint.get("time_embedder.linear_2.bias", dtype=torch.float32)

    adaln_inputs = {}
    for entry in spec["entries"]:
        timesteps = _timesteps_from_bits(entry["timestep_bits"], AI_DEVICE)
        embedded = timestep_embedding(timesteps, spec["freq_dim"])
        temb = _linear(embedded, time_weight_1, time_bias_1)
        temb = _linear(F.silu(temb), time_weight_2, time_bias_2)
        adaln_inputs[entry["name"]] = F.silu(temb).to(torch.bfloat16)

    del time_weight_1, time_bias_1, time_weight_2, time_bias_2

    norm_out_weight = checkpoint.get("norm_out.linear.weight")
    norm_out_bias = checkpoint.get("norm_out.linear.bias")
    cache_tables = {
        _norm_out_key(entry): _linear(
            adaln_inputs[entry["name"]],
            norm_out_weight,
            norm_out_bias,
        )
        .to("cpu")
        .contiguous()
        for entry in spec["entries"]
    }
    del norm_out_weight, norm_out_bias

    # Only one full, unsharded block projection is resident at a time.
    for block_index in range(spec["num_layers"]):
        prefix = f"transformer_blocks.{block_index}.adaln_proj.linear"
        weight = checkpoint.get(f"{prefix}.weight")
        bias = checkpoint.get(f"{prefix}.bias")
        for entry in spec["entries"]:
            # Keep each unique-timestep batch separate, exactly as
            # online step_pre/clean_commit prepares it (one or two
            # rows for Stage3), rather than pooling entries together.
            modulation = _linear(adaln_inputs[entry["name"]], weight, bias)
            table = modulation.view(*_expected_table_shape(spec, entry))
            cache_tables[_block_key(block_index, entry)] = table.to("cpu").contiguous()
        del weight, bias, modulation, table
        logger.info(
            "Built MiniMax-H3 AdaLN cache block {}/{}",
            block_index + 1,
            spec["num_layers"],
        )
    return cache_tables


def _cache_exists_error(cache_path: Path) -> FileExistsError:
    return FileExistsError(f"MiniMax-H3 AdaLN cache path already exists: {cache_path}. Use --overwrite to regenerate and replace the existing cache.")


def _remove_existing_cache(cache_path: Path, overwrite: bool) -> None:
    if not cache_path.exists():
        return
    if not overwrite:
        raise _cache_exists_error(cache_path)
    logger.warning("Removing existing AdaLN cache before regeneration with --overwrite: {}", cache_path)
    shutil.rmtree(cache_path)


def build_persistent_adaln_cache(config, *, overwrite: bool = False) -> Path:
    """Build the cache in this process for later inference processes to load."""
    cache_path, spec = resolve_adaln_cache(config)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    _remove_existing_cache(cache_path, overwrite)
    plan = cache_plan(config)
    files = checkpoint_files(config["dit_original_ckpt"])
    checkpoint = _CheckpointTensors(files, plan.checkpoint_layout)
    logger.info("Building MiniMax-H3 AdaLN cache on {}: {}", AI_DEVICE, cache_path)

    # Generate and validate in a temporary directory, then publish the cache.
    with tempfile.TemporaryDirectory(prefix=".building-", dir=cache_path.parent) as stage:
        stage_path = Path(stage)
        tables = build_cache_tables(spec, checkpoint)
        save_file(tables, stage_path / "adaln_cache.safetensors")
        with (stage_path / "manifest.json").open("w", encoding="utf-8") as handle:
            json.dump(spec, handle, ensure_ascii=False, indent=2, sort_keys=True)
        if not _validate_cache(stage_path, spec):
            raise RuntimeError(f"MiniMax-H3 AdaLN cache validation failed: {cache_path}")
        _remove_existing_cache(cache_path, overwrite)
        os.replace(stage_path, cache_path)

    logger.info("MiniMax-H3 AdaLN cache is ready: {}", cache_path)
    return cache_path
