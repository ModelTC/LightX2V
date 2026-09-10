"""Build a MiniMax-H3 AdaLN cache without loading the full model.

ADALN CACHE SYNC CONTRACT:
The calculations in ``_build_cache`` intentionally mirror the online BF16
path in ``infer/pre_infer.py``, ``infer/transformer_infer.py``, and
``infer/post_infer.py``. If timestep embedding, time-MLP activation/dtype,
AdaLN projection/reshape, final-norm modulation, or their checkpoint keys are
changed online, update this builder and the cache tests in the same change.
Regenerate the cache whenever its values can change.
"""

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
    _build_spec,
    _cache_path,
    _expected_table_shape,
    _norm_out_key,
    _timesteps_from_bits,
    _validate_cache,
)
from lightx2v.models.networks.minimax_h3.checkpoint import MiniMaxH3ShardCheckpoint
from lightx2v.models.networks.minimax_h3.infer.pre_infer import timestep_embedding
from lightx2v_platform.base.global_var import AI_DEVICE

torch_device_module = getattr(torch, AI_DEVICE)


def _checkpoint_files(config) -> list[Path]:
    checkpoint = Path(config["dit_original_ckpt"]).expanduser().resolve()
    files = sorted(checkpoint.glob("*.safetensors")) if checkpoint.is_dir() else [checkpoint]
    if not files or any(not path.is_file() for path in files):
        raise FileNotFoundError(f"MiniMax-H3 safetensors checkpoint not found: {checkpoint}")
    return files


class _CheckpointTensors:
    """Read individual tensors without materializing the whole checkpoint."""

    def __init__(self, files: list[Path], config=None):
        self.files = files
        self.adapter = None
        if (files[0].parent / "model.safetensors.index.json").is_file():
            checkpoint = MiniMaxH3ShardCheckpoint(files[0].parent, config=config)
            self.adapter = checkpoint.selected_reader
        self.locations = self._find_locations() if self.adapter is None else {}

    def _find_locations(self) -> dict[str, Path]:
        directory = self.files[0].parent
        index_files = sorted(directory.glob("*.safetensors.index.json"))
        if index_files:
            with index_files[0].open(encoding="utf-8") as handle:
                weight_map = json.load(handle)["weight_map"]
            return {name: directory / filename for name, filename in weight_map.items()}

        locations = {}
        for path in self.files:
            with safe_open(path, framework="pt", device="cpu") as source:
                locations.update({name: path for name in source.keys()})
        return locations

    def get(self, name: str) -> torch.Tensor:
        if self.adapter is not None:
            # Reuse the inference mapping and validation. The builder consumes
            # logical (out, in) weights; _linear owns the eventual transpose.
            try:
                spec = self.adapter.plan.targets[name][1]
            except KeyError as error:
                raise KeyError(f"MiniMax-H3 checkpoint tensor is missing: {name}") from error
            dtype = torch.float32 if spec.dtype == "F32" else torch.bfloat16
            tensor = torch.empty(spec.shape, dtype=dtype, device="cpu")
            self.adapter.write_targets({name: (tensor, False)})
            return tensor
        path = self.locations.get(name)
        if path is None:
            raise KeyError(f"MiniMax-H3 checkpoint tensor is missing: {name}")
        with safe_open(path, framework="pt", device="cpu") as source:
            return source.get_tensor(name)


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


def _empty_device_cache() -> None:
    if hasattr(torch_device_module, "empty_cache"):
        torch_device_module.empty_cache()


def _build_cache(spec: dict, cache_path: Path, checkpoint_files: list[Path], config=None) -> None:
    stage_path = Path(tempfile.mkdtemp(prefix=".building-", dir=cache_path.parent))
    try:
        checkpoint = _CheckpointTensors(checkpoint_files, config=config)
        with torch.inference_mode():
            # ADALN CACHE SYNC: Keep activation placement and casts aligned with
            # the three online infer modules named in this file's contract.
            time_weight_1 = checkpoint.get("time_embedder.linear_1.weight").to(AI_DEVICE)
            time_bias_1 = checkpoint.get("time_embedder.linear_1.bias").to(AI_DEVICE)
            time_weight_2 = checkpoint.get("time_embedder.linear_2.weight").to(AI_DEVICE)
            time_bias_2 = checkpoint.get("time_embedder.linear_2.bias").to(AI_DEVICE)

            adaln_inputs = {}
            for entry in spec["entries"]:
                timesteps = _timesteps_from_bits(entry["timestep_bits"], AI_DEVICE)
                embedded = timestep_embedding(timesteps, spec["freq_dim"])
                temb = _linear(embedded.float(), time_weight_1, time_bias_1)
                temb = _linear(F.silu(temb), time_weight_2, time_bias_2)
                adaln_inputs[entry["name"]] = F.silu(temb).to(torch.bfloat16)

            del time_weight_1, time_bias_1, time_weight_2, time_bias_2
            _empty_device_cache()

            norm_out_weight = checkpoint.get("norm_out.linear.weight").to(AI_DEVICE)
            norm_out_bias = checkpoint.get("norm_out.linear.bias").to(AI_DEVICE)
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
            _empty_device_cache()

            # Only one full, unsharded block projection is resident at a time.
            for block_index in range(spec["num_layers"]):
                prefix = f"transformer_blocks.{block_index}.adaln_proj.linear"
                weight = checkpoint.get(f"{prefix}.weight").to(AI_DEVICE)
                bias = checkpoint.get(f"{prefix}.bias").to(AI_DEVICE)
                for entry in spec["entries"]:
                    modulation = _linear(adaln_inputs[entry["name"]], weight, bias)
                    table = modulation.view(*_expected_table_shape(spec, entry))
                    cache_tables[_block_key(block_index, entry)] = table.to("cpu").contiguous()
                del weight, bias, modulation, table
                _empty_device_cache()
                logger.info(
                    "Built MiniMax-H3 AdaLN cache block {}/{}",
                    block_index + 1,
                    spec["num_layers"],
                )

            save_file(cache_tables, stage_path / "adaln_cache.safetensors")

        with (stage_path / "manifest.json").open("w", encoding="utf-8") as handle:
            json.dump(spec, handle, ensure_ascii=False, indent=2, sort_keys=True)
        os.replace(stage_path, cache_path)
    finally:
        if stage_path.exists():
            shutil.rmtree(stage_path)


def build_persistent_adaln_cache(config) -> Path:
    """Build the cache in this process for later inference processes to load."""
    spec = _build_spec(config)
    cache_path = _cache_path(config)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if cache_path.exists():
        raise FileExistsError(f"MiniMax-H3 AdaLN cache path already exists: {cache_path}")
    checkpoint_files = _checkpoint_files(config)
    logger.info("Building MiniMax-H3 AdaLN cache on {}: {}", AI_DEVICE, cache_path)
    _build_cache(spec, cache_path, checkpoint_files, config=config)

    if not _validate_cache(cache_path, spec):
        raise RuntimeError(f"MiniMax-H3 AdaLN cache validation failed: {cache_path}")
    logger.info("MiniMax-H3 AdaLN cache is ready: {}", cache_path)
    return cache_path
