"""Convert merged H3 projection profiles to standalone per-channel INT8 shards.

Unselected tensors retain their exact dtype and bytes. The output is a separate
indexed checkpoint; the original merged model is never modified.
"""

import argparse
import gc
import hashlib
import json
import shutil
import time
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from lightx2v.models.networks.minimax_h3.int8_metal import QUANT_SCOPES, linear_component


def file_hash(path):
    result = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024**2), b""):
            result.update(block)
    return result.hexdigest()


def quantize_weight(tensor, device):
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


def convert(source, output, device="cpu", scope="dit_proj"):
    if scope not in QUANT_SCOPES:
        raise ValueError("Unknown H3 INT8 scope: " + str(scope))
    source, output = Path(source).resolve(), Path(output).resolve()
    if source == output or source in output.parents:
        raise ValueError("Output must be independent of the source checkpoint directory")
    indices = list(source.glob("*.safetensors.index.json"))
    if len(indices) != 1:
        raise ValueError("Expected one source safetensors index")
    original = json.loads(indices[0].read_text())
    mapping = original["weight_map"]
    selected = sorted(name for name in mapping if name.endswith(".weight") and linear_component(name[:-7]) in QUANT_SCOPES[scope])
    if not selected:
        raise ValueError("No main-transformer projection weights found")
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    manifest = {
        "scheme": "int8-metal",
        "scope": scope,
        "source": str(source),
        "source_index_sha256": file_hash(indices[0]),
        "device": device,
        "torch": torch.__version__,
        "quantized_tensors": selected,
        "shards": [],
    }
    new_mapping = dict(mapping)
    total_size = 0
    for shard in sorted(set(mapping.values())):
        source_path = source / shard
        source_hash = file_hash(source_path)
        weights = {}
        expected_keys = {key for key, value in mapping.items() if value == shard}
        with safe_open(str(source_path), framework="pt", device="cpu") as f, torch.inference_mode():
            if set(f.keys()) != expected_keys:
                raise ValueError("Index/shard key mismatch: " + shard)
            for key in f.keys():
                value = f.get_tensor(key)
                if key in selected:
                    quantized, scales = quantize_weight(value, device)
                    weights[key] = quantized
                    scale_name = key + "_scale"
                    if scale_name in mapping:
                        raise ValueError("Input is already quantized: " + key)
                    weights[scale_name] = scales
                    new_mapping[scale_name] = shard
                else:
                    weights[key] = value
            temporary = output / (shard + ".incomplete")
            save_file(weights, str(temporary), metadata={"lightx2v_quant_scheme": "int8-metal", "lightx2v_quant_scope": scope})
            destination = output / shard
            temporary.rename(destination)
            shard_bytes = sum(t.numel() * t.element_size() for t in weights.values())
            total_size += shard_bytes
            with safe_open(str(destination), framework="pt", device="cpu") as written:
                if set(written.keys()) != set(weights):
                    raise ValueError("Written keys mismatch")
                for key, tensor in weights.items():
                    header = written.get_slice(key)
                    if tuple(header.get_shape()) != tuple(tensor.shape):
                        raise ValueError("Written shape mismatch: " + key)
                    if key not in selected and not key.endswith(".weight_scale"):
                        if not torch.equal(written.get_tensor(key).reshape(-1).view(torch.uint8), f.get_tensor(key).reshape(-1).view(torch.uint8)):
                            raise ValueError("Unselected tensor changed: " + key)
            manifest["shards"].append({"file": shard, "source_sha256": source_hash, "output_sha256": file_hash(destination), "tensor_payload_bytes": shard_bytes})
        del weights
        gc.collect()
        if device == "mps":
            torch.mps.synchronize()
            torch.mps.empty_cache()
        print(json.dumps({"shard": shard, "done": len(manifest["shards"]), "payload_gib": shard_bytes / 1024**3, "elapsed_seconds": time.perf_counter() - started}), flush=True)
    for name in ("config.json",):
        if (source / name).is_file():
            shutil.copy2(source / name, output / name)
    metadata = dict(original.get("metadata", {}), total_size=total_size, lightx2v_quant_scheme="int8-metal", lightx2v_quant_scope=scope, quantized_linear_count=len(selected))
    (output / indices[0].name).write_text(json.dumps({"metadata": metadata, "weight_map": new_mapping}, indent=2) + "\n")
    manifest["elapsed_seconds"] = time.perf_counter() - started
    manifest["total_payload_bytes"] = total_size
    manifest["status"] = "COMPLETE"
    (output / "quantization_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    parser.add_argument("--scope", choices=tuple(QUANT_SCOPES), default="dit_proj")
    args = parser.parse_args()
    print(json.dumps(convert(args.source, args.output, args.device, args.scope), indent=2))
