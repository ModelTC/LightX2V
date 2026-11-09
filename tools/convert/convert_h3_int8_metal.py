"""Convert merged H3 main-block projections to standalone INT8 Metal shards."""

import argparse
import gc
import json
import shutil
from collections import defaultdict
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
from safetensors import safe_open
from safetensors.torch import save_file

PROJECTION_SUFFIXES = (".ff.net.0.proj", ".ff.net.2", ".attn.to_q", ".attn.to_k", ".attn.to_v", ".attn.to_out.0")


def is_projection_weight(name):
    return name.startswith("transformer_blocks.") and name.endswith(tuple(suffix + ".weight" for suffix in PROJECTION_SUFFIXES))


def quantize_weight(tensor, device):
    """Compute per-output-channel scales and rounded INT8 weights in FP32."""
    if tensor.ndim != 2 or tensor.dtype != torch.bfloat16:
        raise ValueError("H3 INT8 conversion requires a two-dimensional BF16 weight")
    n, k = tensor.shape
    if not n or not k or n % 128 or k % 128:
        raise ValueError("INT8 Metal weight dimensions must be positive multiples of 128")
    weight = tensor.to(device=device, dtype=torch.float32)
    if not torch.isfinite(weight).all().item():
        raise ValueError("Nonfinite source weight")
    scales = weight.abs().amax(dim=1, keepdim=True).clamp_min(1e-12) / 127
    quantized = torch.round(weight / scales).clamp(-127, 127).to(torch.int8)
    return quantized.cpu().contiguous(), scales.cpu().contiguous()


def convert_shard(source, output, names, device):
    """Convert one indexed shard while preserving every unselected tensor."""
    weights = {}
    with safe_open(str(source), framework="pt", device="cpu") as checkpoint, torch.inference_mode():
        if set(checkpoint.keys()) != set(names):
            raise ValueError(f"Index/shard key mismatch: {source.name}")
        for name in names:
            tensor = checkpoint.get_tensor(name)
            if is_projection_weight(name):
                weights[name], weights[name + "_scale"] = quantize_weight(tensor, device)
            else:
                weights[name] = tensor
        output.parent.mkdir(parents=True, exist_ok=True)
        save_file(weights, str(output), metadata={"lightx2v_quant_scheme": "int8-metal", "lightx2v_quant_scope": "dit_proj"})
        payload_bytes = sum(tensor.numel() * tensor.element_size() for tensor in weights.values())
        weight_names = tuple(weights)
    del weights
    gc.collect()
    if device == "mps":
        torch.mps.synchronize()
        torch.mps.empty_cache()
    return weight_names, payload_bytes


def convert(source, output, device="cpu"):
    """Publish a complete INT8 checkpoint, preserving the source on failure.

    Input is an indexed merged BF16 H3 transformer. The destination must be a
    new independent directory. Shards and index become visible together after
    conversion succeeds, so an interrupted conversion can be retried.
    """
    source, output = Path(source).resolve(), Path(output).resolve()
    if source == output or source in output.parents:
        raise ValueError("Output must be independent of the source checkpoint directory")
    if output.exists():
        raise FileExistsError(f"Output checkpoint already exists: {output}")
    indices = list(source.glob("*.safetensors.index.json"))
    if len(indices) != 1:
        raise ValueError("Expected one source safetensors index")
    original = json.loads(indices[0].read_text())
    mapping = original["weight_map"]
    selected = {name for name in mapping if is_projection_weight(name)}
    if not selected:
        raise ValueError("No main-transformer projection weights found")
    if any(name + "_scale" in mapping for name in selected):
        raise ValueError("Input checkpoint is already quantized")
    shards = defaultdict(list)
    for name, shard in mapping.items():
        shards[shard].append(name)

    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=f".{output.name}-", dir=output.parent) as temporary:
        staging = Path(temporary)
        weight_map = {}
        total_size = 0
        for shard, names in sorted(shards.items()):
            converted_names, payload_bytes = convert_shard(source / shard, staging / shard, names, device)
            weight_map.update(dict.fromkeys(converted_names, shard))
            total_size += payload_bytes
            print(f"Converted {shard}", flush=True)
        if (source / "config.json").is_file():
            shutil.copy2(source / "config.json", staging / "config.json")
        metadata = {
            **original.get("metadata", {}),
            "total_size": total_size,
            "lightx2v_quant_scheme": "int8-metal",
            "lightx2v_quant_scope": "dit_proj",
            "quantized_linear_count": len(selected),
        }
        index = {"metadata": metadata, "weight_map": weight_map}
        (staging / indices[0].name).write_text(json.dumps(index, indent=2) + "\n")
        staging.rename(output)
    return metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    args = parser.parse_args()
    print(json.dumps(convert(args.source, args.output, args.device), indent=2))
