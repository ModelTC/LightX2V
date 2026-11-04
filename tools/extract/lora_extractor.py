#!/usr/bin/env python3
"""Extract selected Linear weight deltas as SVD LoRAs using a JSON/YAML config."""

from __future__ import annotations

import argparse
import fnmatch
import json
import math
import os
import tempfile
from collections.abc import Mapping
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import torch
from safetensors.torch import save_file
from tqdm import tqdm

try:
    from .export_student import ModelWeights, torch_dtype
except ImportError:
    from export_student import ModelWeights, torch_dtype


@dataclass
class ExtractConfig:
    source_model: str
    target_model: str
    output: str
    layers: list[str]
    rank: int = 32
    alpha: float | None = None
    dtype: str = "bf16"
    device: str = "auto"
    key_prefix: str = "auto"

    def __post_init__(self):
        for name in ("source_model", "target_model", "output"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name):
                raise ValueError(f"{name} must be a non-empty path")
            setattr(self, name, os.path.expandvars(os.path.expanduser(getattr(self, name))))
        if not isinstance(self.layers, list) or not self.layers or any(not isinstance(layer, str) or not layer or layer.endswith(".weight") for layer in self.layers):
            raise ValueError("layers must be a non-empty list of module names/glob patterns without .weight")
        if type(self.rank) is not int or self.rank < 1:
            raise ValueError("rank must be a positive integer")
        self.alpha = float(self.rank if self.alpha is None else self.alpha)
        if not math.isfinite(self.alpha) or self.alpha <= 0:
            raise ValueError("alpha must be positive and finite")
        torch_dtype(self.dtype)
        if self.key_prefix not in ("auto", "transformer", "diffusion_model"):
            raise ValueError("key_prefix must be auto, transformer or diffusion_model")
        if not self.output.endswith(".safetensors"):
            raise ValueError("output must end in .safetensors")


def load_config(path, overrides=None):
    path = Path(path)
    if path.suffix.lower() == ".json":
        values = json.loads(path.read_text())
    else:
        import yaml

        values = yaml.safe_load(path.read_text())
    if not isinstance(values, dict):
        raise TypeError("Config must contain a mapping")
    unknown = set(values) - {field.name for field in fields(ExtractConfig)}
    if unknown:
        raise ValueError(f"Unknown config fields: {sorted(unknown)}")
    values.update({key: value for key, value in (overrides or {}).items() if value is not None})
    return ExtractConfig(**values)


def _wan_key(key):
    """Map Diffusers Wan tensor names to the native Wan names used by LightX2V."""
    if key == "scale_shift_table":
        return "head.modulation"
    replacements = (
        ("condition_embedder.text_embedder.linear_1.", "text_embedding.0."),
        ("condition_embedder.text_embedder.linear_2.", "text_embedding.2."),
        ("condition_embedder.time_embedder.linear_1.", "time_embedding.0."),
        ("condition_embedder.time_embedder.linear_2.", "time_embedding.2."),
        ("condition_embedder.time_proj.", "time_projection.1."),
        ("condition_embedder.image_embedder.norm1.", "img_emb.proj.0."),
        ("condition_embedder.image_embedder.ff.net.0.proj.", "img_emb.proj.1."),
        ("condition_embedder.image_embedder.ff.net.2.", "img_emb.proj.3."),
        ("condition_embedder.image_embedder.norm2.", "img_emb.proj.4."),
        ("proj_out.", "head.head."),
        (".attn1.", ".self_attn."),
        (".attn2.", ".cross_attn."),
        (".to_q.", ".q."),
        (".to_k.", ".k."),
        (".to_v.", ".v."),
        (".to_out.0.", ".o."),
        (".add_k_proj.", ".k_img."),
        (".add_v_proj.", ".v_img."),
        (".norm_added_k.", ".norm_k_img."),
        (".ffn.net.0.proj.", ".ffn.0."),
        (".ffn.net.2.", ".ffn.2."),
        (".scale_shift_table", ".modulation"),
        (".norm2.", ".norm3."),
    )
    for before, after in replacements:
        key = key.replace(before, after)
    return key


class ExtractionWeights(Mapping):
    """Normalize Wan layouts and expose fused Qwen MLPs as two Linear weights."""

    def __init__(self, source):
        self.source = source
        self.entries, self.shapes = {}, {}
        wan_diffusers = any(key.startswith("condition_embedder.time_embedder.") for key in source)
        for key in source:
            name = _wan_key(key) if wan_diffusers else key
            shape = source.shapes[key]
            if name.endswith(".img_mlp.gate_up.weight"):
                if len(shape) != 2 or shape[0] % 2:
                    raise ValueError(f"Invalid fused Qwen MLP shape: {name}: {shape}")
                half = shape[0] // 2
                for part, start in (("gate_layer", 0), ("proj", half)):
                    self._add(name.replace(".gate_up.", f".{part}."), (key, (start, half)), (half, shape[1]))
            else:
                self._add(name, (key, None), shape)

    def _add(self, name, entry, shape):
        if name in self.entries:
            raise ValueError(f"Ambiguous model weight after normalization: {name}")
        self.entries[name], self.shapes[name] = entry, shape

    def __len__(self):
        return len(self.entries)

    def __iter__(self):
        return iter(self.entries)

    def __contains__(self, key):
        return key in self.entries

    def __getitem__(self, key):
        original, section = self.entries[key]
        tensor = self.source[original]
        return tensor if section is None else tensor.narrow(0, *section)


def select_layers(source, target, config):
    selected = set()
    for pattern in config.layers:
        matches = {key for key in source if key.endswith(".weight") and fnmatch.fnmatchcase(key[:-7], pattern) and len(source.shapes[key]) == 2}
        if not matches:
            raise ValueError(f"Layer pattern matches no Linear weights: {pattern}")
        selected.update(matches)
    for key in sorted(selected):
        if key not in target:
            raise ValueError(f"Selected target weight missing: {key}")
        if source.shapes[key] != target.shapes[key]:
            raise ValueError(f"Selected weight shape mismatch: {key}: {source.shapes[key]} vs {target.shapes[key]}")
    return sorted(selected)


def decompose_to_lora(diff, module, rank, alpha, prefix, device):
    """Return factors satisfying (alpha / effective_rank) * up @ down = truncated SVD(delta)."""
    effective_rank = min(rank, *diff.shape)
    u, s, vh = torch.linalg.svd(diff.to(device=device, dtype=torch.float32), full_matrices=False)
    root = (s[:effective_rank] * (effective_rank / alpha)).sqrt()
    base = f"{prefix}.{module}"
    return {
        f"{base}.lora_up.weight": (u[:, :effective_rank] * root).cpu().contiguous(),
        f"{base}.lora_down.weight": (root[:, None] * vh[:effective_rank]).cpu().contiguous(),
        f"{base}.alpha": torch.tensor(alpha, dtype=torch.float32),
    }


@torch.no_grad()
def extract_lora(config):
    print("=" * 60)
    print("LoRA Extractor Started")
    print(json.dumps(asdict(config), indent=2))
    device = ("cuda" if torch.cuda.is_available() else "cpu") if config.device == "auto" else config.device
    device = torch.device(device)
    print(f"SVD device: {device}; compute dtype: float32; output dtype: {config.dtype}")
    source = ExtractionWeights(ModelWeights(config.source_model))
    target = ExtractionWeights(ModelWeights(config.target_model))
    keys = select_layers(source, target, config)
    prefix = config.key_prefix
    if prefix == "auto":
        prefix = "transformer" if any(key.startswith("transformer_blocks.") for key in source) else "diffusion_model"
    print(f"Selected {len(keys)} Linear layers; ignored {len(source) - len(keys)} source tensors; output prefix: {prefix}")
    if set(source) != set(target):
        print(f"Unselected schema differences: source-only={len(set(source) - set(target))}, target-only={len(set(target) - set(source))}")
    dtype = torch_dtype(config.dtype)
    weights, unchanged, clipped = {}, 0, 0
    pbar = tqdm(keys, desc="Extracting LoRA weights", unit="layer")
    for key in pbar:
        pbar.set_postfix_str(key)
        source_tensor, target_tensor = source[key], target[key]
        if not source_tensor.is_floating_point() or not target_tensor.is_floating_point():
            raise ValueError(f"Selected weight must be floating point: {key}")
        diff = target_tensor.float() - source_tensor.float()
        del source_tensor, target_tensor
        if not torch.isfinite(diff).all():
            raise ValueError(f"Non-finite weight delta: {key}")
        if not torch.count_nonzero(diff):
            unchanged += 1
            continue
        if config.rank > min(diff.shape):
            clipped += 1
            tqdm.write(f"Rank capped for {key}: {config.rank} -> {min(diff.shape)}")
        try:
            factors = decompose_to_lora(diff, key[:-7], config.rank, config.alpha, prefix, device)
        except RuntimeError as error:
            raise RuntimeError(f"SVD failed for {key} on {device}; no partial LoRA will be saved") from error
        for name, tensor in factors.items():
            # Alpha is a scaling scalar, kept in FP32 independently of factor dtype.
            tensor = tensor if name.endswith(".alpha") else tensor.to(dtype)
            if not torch.isfinite(tensor).all():
                raise ValueError(f"Output overflow in {name}; choose a wider dtype or a different alpha")
            weights[name] = tensor.contiguous()
        del diff, factors
    if not weights:
        raise ValueError("All selected layers are unchanged; no LoRA to save")
    output = Path(config.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    metadata = {"format": "pt", "extraction_config": json.dumps(asdict(config)), "linear_layers": str(len(keys) - unchanged)}
    print(f"Saving LoRA weights to: {output} ({config.dtype}, alpha scalars in FP32)", flush=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{output.name}-", dir=str(output.parent))
    os.close(fd)
    try:
        save_file(weights, temporary, metadata=metadata)
        os.replace(temporary, output)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    print(
        f"\nExtraction statistics:\n  Selected: {len(keys)}\n  LoRA decomposition: {len(keys) - unchanged}\n  Unchanged: {unchanged}\n  Rank capped: {clipped}\n  Total saved tensors: {len(weights)}"
    )
    print(f"LoRA extraction completed: {output}")
    return weights


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="JSON or YAML extraction config; relative paths use the working directory")
    parser.add_argument("--source-model", help="Override source_model")
    parser.add_argument("--target-model", help="Override target_model")
    parser.add_argument("--output", help="Override output")
    parser.add_argument("--rank", type=int)
    parser.add_argument("--alpha", type=float)
    parser.add_argument("--dtype", "--output-dtype", dest="dtype")
    parser.add_argument("--device", help="auto, cpu, cuda or cuda:N")
    parser.add_argument("--layers", nargs="+", help="Override module name/glob patterns")
    args = vars(parser.parse_args())
    path = args.pop("config")
    extract_lora(load_config(path, args))


if __name__ == "__main__":
    main()
