"""Convert EdenLive ProxyDINO linear weights into a native INT8 ConvRot file.

The source's linear weights are rounded to BF16 before quantization, matching
ProxyDINO's effective BF16 linear weights. All other tensors keep their original
dtype and contents. The source checkpoint is never overwritten.
"""

import argparse
import hashlib
import json
import math
import os
import re
import tempfile
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

PREFIX = "model.module.encoder."
LINEAR_PATTERN = re.compile(r"^(?:backbone\.layer\.\d+\.(?:attention\.[qkvo]_proj|mlp\.(?:up_proj|down_proj))|patch_projection)\.weight$")
LAYER_PATTERN = re.compile(r"^backbone\.layer\.(\d+)\.")
LAYER_LINEAR_SUFFIXES = ("attention.q_proj", "attention.k_proj", "attention.v_proj", "attention.o_proj", "mlp.up_proj", "mlp.down_proj")


def is_proxy_linear_weight(name):
    return LINEAR_PATTERN.fullmatch(name.removeprefix(PREFIX)) is not None


def _source_digest(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _publish(temporary, destination, overwrite):
    if overwrite:
        os.replace(temporary, destination)
    else:
        # Both files are in the same directory/filesystem. link() publishes a
        # complete checkpoint atomically and fails if a concurrent writer won.
        os.link(temporary, destination)
        os.unlink(temporary)


def _atomic_checkpoint(weights, destination, metadata, overwrite):
    descriptor, name = tempfile.mkstemp(prefix=destination.name + ".", suffix=".tmp", dir=destination.parent)
    os.close(descriptor)
    temporary = Path(name)
    try:
        save_file(weights, str(temporary), metadata=metadata)
        _publish(temporary, destination, overwrite)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_json(value, destination, overwrite):
    descriptor, name = tempfile.mkstemp(prefix=destination.name + ".", suffix=".tmp", dir=destination.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w") as handle:
            json.dump(value, handle, indent=2)
            handle.write("\n")
        _publish(temporary, destination, overwrite)
    finally:
        temporary.unlink(missing_ok=True)


def _validate_linear_names(names):
    normalized = {name.removeprefix(PREFIX) for name in names}
    layers = {int(match.group(1)) for name in normalized if (match := LAYER_PATTERN.match(name))}
    if not layers or "patch_projection.weight" not in normalized:
        raise ValueError("Source must include ProxyDINO transformer layers and patch_projection.weight")
    required = {f"backbone.layer.{layer}.{suffix}.weight" for layer in layers for suffix in LAYER_LINEAR_SUFFIXES}
    missing = sorted(required - normalized)
    if missing:
        raise ValueError(f"Source is missing ProxyDINO linear weights: {missing[:8]}")


@torch.no_grad()
def convert_proxy_dino(source, output, device="cpu", group_size=256, overwrite=False, report_path=None):
    """Save an independent checkpoint compatible with ``int8-convrot`` modules."""
    from comfy_kitchen.tensor.int8 import TensorWiseINT8Layout

    source = Path(source).resolve()
    output = Path(output).resolve()
    report_path = Path(report_path).resolve() if report_path else output.with_suffix(".report.json")
    if group_size not in (16, 64, 256):
        raise ValueError("group_size must be 16, 64, or 256")
    if source == output or source == report_path or output == report_path:
        raise ValueError("Source, checkpoint output and report paths must be distinct")
    if not source.is_file():
        raise FileNotFoundError(source)
    for destination in (output, report_path):
        if destination.exists() and not overwrite:
            raise FileExistsError(f"Refusing to overwrite existing output: {destination}")
        destination.parent.mkdir(parents=True, exist_ok=True)
    device = torch.device(device)
    converted, errors = {}, []
    with safe_open(str(source), framework="pt", device="cpu") as handle:
        names = list(handle.keys())
        _validate_linear_names(names)
        source_metadata = handle.metadata() or {}
        linear_names = {name for name in names if is_proxy_linear_weight(name)}
        linear_biases = {name.removesuffix(".weight") + ".bias" for name in linear_names}
        for name in names:
            original = handle.get_tensor(name)
            if name not in linear_names:
                converted[name] = original.to(torch.bfloat16).contiguous() if name in linear_biases else original.contiguous()
                continue
            if original.ndim != 2 or not original.is_floating_point() or original.shape[1] % group_size:
                raise ValueError(f"Invalid ProxyDINO ConvRot linear {name}: shape={tuple(original.shape)}, dtype={original.dtype}")
            effective = original.to(device=device, dtype=torch.bfloat16).contiguous()
            if not torch.isfinite(effective).all():
                raise ValueError(f"Nonfinite values in ProxyDINO linear {name}")
            quantized, params = TensorWiseINT8Layout.quantize(
                effective,
                is_weight=True,
                per_channel=True,
                convrot=True,
                convrot_groupsize=group_size,
                stochastic_rounding=0,
            )
            scale = params.scale.to(torch.float32).contiguous()
            if quantized.dtype != torch.int8 or quantized.shape != effective.shape or scale.numel() != effective.shape[0] or not torch.isfinite(scale).all():
                raise ValueError(f"Invalid ConvRot quantizer output for {name}")
            reconstructed = TensorWiseINT8Layout.dequantize(quantized, params).float()
            reference = effective.float()
            difference = reconstructed - reference
            squared_error = float(difference.square().sum())
            squared_reference = float(reference.square().sum())
            error = {
                "name": name,
                "shape": list(original.shape),
                "source_dtype": str(original.dtype),
                "effective_dtype": str(effective.dtype),
                "elements": original.numel(),
                "squared_error": squared_error,
                "squared_reference": squared_reference,
                "rmse": math.sqrt(squared_error / original.numel()),
                "relative_rmse": math.sqrt(squared_error / squared_reference) if squared_reference else 0.0,
                "max_absolute_error": float(difference.abs().max()),
            }
            errors.append(error)
            stem = name.removesuffix(".weight")
            converted[name] = quantized.cpu().contiguous()
            converted[stem + ".weight_scale"] = scale.cpu().reshape(original.shape[0], 1).contiguous()
            converted[stem + ".convrot_groupsize"] = torch.tensor(group_size, dtype=torch.int32)
            print(f"Converted {name} {tuple(original.shape)} relative_rmse={error['relative_rmse']:.6f}", flush=True)
    elements = sum(error["elements"] for error in errors)
    squared_error = sum(error["squared_error"] for error in errors)
    squared_reference = sum(error["squared_reference"] for error in errors)
    source_sha256 = _source_digest(source)
    metadata = dict(source_metadata)
    metadata.update(
        format="lightx2v-edenlive-proxy-int8-convrot-v1",
        quant_scheme="int8-convrot",
        source=str(source),
        source_sha256=source_sha256,
        group_size=str(group_size),
        linear_count=str(len(errors)),
        effective_linear_dtype="torch.bfloat16",
        linear_bias_dtype="torch.bfloat16",
        other_tensors="original_dtype_and_contents",
        stochastic_rounding="0",
    )
    report = {
        "source": str(source),
        "source_sha256": source_sha256,
        "output": str(output),
        "scheme": "int8-convrot",
        "group_size": group_size,
        "device": str(device),
        "linear_count": len(errors),
        "source_tensor_count": len(names),
        "output_tensor_count": len(converted),
        "quantized_elements": elements,
        "rmse": math.sqrt(squared_error / elements),
        "relative_rmse": math.sqrt(squared_error / squared_reference) if squared_reference else 0.0,
        "max_absolute_error": max(error["max_absolute_error"] for error in errors),
        "linears": errors,
    }
    _atomic_checkpoint(converted, output, metadata, overwrite)
    _atomic_json(report, report_path, overwrite)
    print(f"Saved {output} ({len(errors)} INT8 ConvRot linears); report: {report_path}", flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="Original vision-encoder/projection.safetensors")
    parser.add_argument("--output", required=True, help="New standalone INT8 ConvRot safetensors path")
    parser.add_argument("--device", default="cpu", help="cpu or cuda:N")
    parser.add_argument("--group-size", type=int, choices=(16, 64, 256), default=256)
    parser.add_argument("--report", help="Optional JSON error report path")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing outputs; never replaces source")
    args = parser.parse_args()
    convert_proxy_dino(args.source, args.output, args.device, args.group_size, args.overwrite, args.report)


if __name__ == "__main__":
    main()
