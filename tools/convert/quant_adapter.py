import argparse
import gc
import os
import sys
from collections import Counter
from pathlib import Path

import safetensors
import torch
from safetensors.torch import save_file

NVFP4_BLOCK_SIZE = 16
NVFP4_SCALE_TILE_M = 128
NVFP4_SCALE_TILE_K = NVFP4_BLOCK_SIZE * 4
E2M1_VALUES = (
    0.0,
    0.5,
    1.0,
    1.5,
    2.0,
    3.0,
    4.0,
    6.0,
    -0.0,
    -0.5,
    -1.0,
    -1.5,
    -2.0,
    -3.0,
    -4.0,
    -6.0,
)


def _ceil_div(value, divisor):
    return (value + divisor - 1) // divisor


def _nvfp4_weight_keys(keys):
    key_set = set(keys)
    groups = {}
    for weight_key in keys:
        if not weight_key.endswith(".weight"):
            continue
        prefix = weight_key.removesuffix(".weight")
        companions = {
            "scale": f"{prefix}.weight_scale",
            "input_global_scale": f"{prefix}.input_global_scale",
            "alpha": f"{prefix}.alpha",
        }
        if all(key in key_set for key in companions.values()):
            groups[weight_key] = companions
    return groups


def _decode_e2m1(packed):
    if packed.dtype != torch.uint8 or packed.ndim != 2:
        raise ValueError(f"NVFP4 weight must be a 2D uint8 tensor, got shape={tuple(packed.shape)}, dtype={packed.dtype}.")

    lookup = torch.tensor(E2M1_VALUES, dtype=torch.float32, device=packed.device)
    low = lookup[(packed & 0x0F).to(torch.int64)]
    high = lookup[((packed >> 4) & 0x0F).to(torch.int64)]
    return torch.stack((low, high), dim=-1).reshape(packed.shape[0], packed.shape[1] * 2)


def _unswizzle_nvfp4_scales(swizzled, rows, columns, block_size=NVFP4_BLOCK_SIZE):
    if columns % block_size:
        raise ValueError(f"NVFP4 input dimension must be divisible by {block_size}, got {columns}.")

    row_tiles = _ceil_div(rows, NVFP4_SCALE_TILE_M)
    column_tiles = _ceil_div(columns, block_size * 4)
    expected_numel = row_tiles * column_tiles * 32 * 4 * 4
    if swizzled.numel() != expected_numel:
        raise ValueError(f"Unexpected NVFP4 scale layout: shape={tuple(swizzled.shape)}, numel={swizzled.numel()}, expected={expected_numel} for weight shape=({rows}, {columns}).")

    linear = swizzled.reshape(1, row_tiles, column_tiles, 32, 4, 4).permute(0, 1, 4, 3, 2, 5).reshape(row_tiles * NVFP4_SCALE_TILE_M, column_tiles * 4)
    return linear[:rows, : columns // block_size].to(torch.float32)


@torch.inference_mode()
def dequantize_nvfp4_weight(packed, swizzled_scale, input_global_scale, alpha):
    rows, packed_columns = packed.shape
    columns = packed_columns * 2

    input_global_scale = input_global_scale.to(torch.float32)
    alpha = alpha.to(torch.float32)
    if input_global_scale.numel() != 1 or alpha.numel() != 1:
        raise ValueError("NVFP4 input_global_scale and alpha must both be scalar tensors.")

    # The checkpoint stores alpha = 1 / (input_global_scale * weight_global_scale).
    weight_global_scale = torch.reciprocal(input_global_scale * alpha)
    if not torch.isfinite(weight_global_scale).item() or weight_global_scale.item() <= 0:
        raise ValueError(f"Invalid derived NVFP4 weight_global_scale={weight_global_scale.item()}.")

    values = _decode_e2m1(packed)
    scales = _unswizzle_nvfp4_scales(swizzled_scale, rows, columns)
    values = values.reshape(rows, columns // NVFP4_BLOCK_SIZE, NVFP4_BLOCK_SIZE)
    return (values * (scales / weight_global_scale).unsqueeze(-1)).reshape(rows, columns)


@torch.inference_mode()
def quantize_int8_per_output_channel(weight):
    if weight.dtype != torch.float32 or weight.ndim != 2:
        raise ValueError(f"INT8 source weight must be a 2D float32 tensor, got shape={tuple(weight.shape)}, dtype={weight.dtype}.")
    max_value = weight.abs().amax(dim=1, keepdim=True).clamp(min=1e-5)
    scale = max_value / 127.0
    quantized = torch.clamp(torch.round(weight / scale), -128, 127).to(torch.int8)
    return quantized.contiguous(), scale.to(torch.bfloat16).contiguous()


def _target_filename(source_name):
    if "NVFP4" in source_name:
        return source_name.replace("NVFP4", "INT8")
    if "nvfp4" in source_name:
        return source_name.replace("nvfp4", "int8")
    path = Path(source_name)
    return f"{path.stem}_int8{path.suffix}"


def _validate_int8_file(path, expected_quantized, expected_keys):
    dtype_counts = Counter()
    quantized_count = 0
    with safetensors.safe_open(path, framework="pt", device="cpu") as source:
        keys = list(source.keys())
        if len(keys) != expected_keys:
            raise ValueError(f"{path}: expected {expected_keys} tensors, found {len(keys)}.")
        if any(key.endswith((".alpha", ".input_global_scale")) for key in keys):
            raise ValueError(f"{path}: found stale NVFP4 alpha/input_global_scale tensors.")

        key_set = set(keys)
        for key in keys:
            tensor_slice = source.get_slice(key)
            dtype_counts[tensor_slice.get_dtype()] += 1
            if tensor_slice.get_dtype() != "I8":
                continue
            quantized_count += 1
            scale_key = f"{key.removesuffix('.weight')}.weight_scale"
            if not key.endswith(".weight") or scale_key not in key_set:
                raise ValueError(f"{path}: INT8 tensor {key} has no matching weight_scale.")
            scale_slice = source.get_slice(scale_key)
            weight_shape = tensor_slice.get_shape()
            if scale_slice.get_dtype() != "BF16" or scale_slice.get_shape() != [weight_shape[0], 1]:
                raise ValueError(f"{path}: invalid scale for {key}: dtype={scale_slice.get_dtype()}, shape={scale_slice.get_shape()}.")

    if quantized_count != expected_quantized:
        raise ValueError(f"{path}: expected {expected_quantized} INT8 weights, found {quantized_count}.")
    return dtype_counts


def convert_wan_nvfp4_file(source_path, output_path, non_quant_dtype=torch.float32, overwrite=False):
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {output_path}. Pass --overwrite to replace it.")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    temporary_path = output_path.with_name(f".{output_path.name}.tmp.{os.getpid()}")
    converted = {}
    try:
        with safetensors.safe_open(source_path, framework="pt", device="cpu") as source:
            keys = list(source.keys())
            groups = _nvfp4_weight_keys(keys)
            if not groups:
                raise ValueError(f"{source_path} does not contain LightX2V NVFP4 weight groups.")

            skipped_keys = set()
            for companions in groups.values():
                skipped_keys.update(companions.values())

            total = len(groups)
            converted_index = 0
            for key in keys:
                if key in skipped_keys:
                    continue
                if key not in groups:
                    converted[key] = source.get_tensor(key).to(non_quant_dtype).contiguous()
                    continue

                converted_index += 1
                companions = groups[key]
                dequantized = dequantize_nvfp4_weight(
                    source.get_tensor(key),
                    source.get_tensor(companions["scale"]),
                    source.get_tensor(companions["input_global_scale"]),
                    source.get_tensor(companions["alpha"]),
                )
                int8_weight, int8_scale = quantize_int8_per_output_channel(dequantized)
                converted[key] = int8_weight
                converted[companions["scale"]] = int8_scale
                del dequantized, int8_weight, int8_scale

                if converted_index == 1 or converted_index % 25 == 0 or converted_index == total:
                    print(f"[{source_path.name}] converted {converted_index}/{total} NVFP4 weights", flush=True)
                if converted_index % 25 == 0:
                    gc.collect()

            expected_keys = len(keys) - 2 * total
            if len(converted) != expected_keys:
                raise ValueError(f"Converted tensor count mismatch: expected {expected_keys}, got {len(converted)}.")

            metadata = {
                "source_file": source_path.name,
                "source_format": "nvfp4",
                "target_format": "int8-npu",
                "conversion": "nvfp4-dequant-then-int8-per-output-channel",
            }
            save_file(converted, str(temporary_path), metadata=metadata)

        if output_path.exists():
            output_path.unlink()
        os.replace(temporary_path, output_path)
        dtype_counts = _validate_int8_file(output_path, total, expected_keys)
        print(f"Saved {output_path} ({output_path.stat().st_size / 1e9:.2f} GB), dtypes={dict(dtype_counts)}", flush=True)
        return output_path
    finally:
        converted.clear()
        gc.collect()
        if temporary_path.exists():
            temporary_path.unlink()


def convert_wan_nvfp4_path(model_path, output_path, non_quant_dtype=torch.float32, overwrite=False):
    if model_path.is_file():
        target = output_path
        if output_path.exists() and output_path.is_dir():
            target = output_path / _target_filename(model_path.name)
        return [convert_wan_nvfp4_file(model_path, target, non_quant_dtype, overwrite)]

    if not model_path.is_dir():
        raise FileNotFoundError(f"Input path does not exist: {model_path}")
    if output_path.exists() and not output_path.is_dir():
        raise ValueError(f"Directory input requires a directory output, got file: {output_path}")

    sources = sorted(path for path in model_path.glob("*.safetensors") if "_comfy" not in path.stem)
    if not sources:
        raise FileNotFoundError(f"No non-ComfyUI safetensors files found under {model_path}.")

    comfy_count = sum(1 for path in model_path.glob("*_comfy.safetensors"))
    if comfy_count:
        print(f"Skipping {comfy_count} ComfyUI checkpoints because their packed weight layout differs.", flush=True)

    outputs = []
    for source_path in sources:
        target = output_path / _target_filename(source_path.name)
        outputs.append(convert_wan_nvfp4_file(source_path, target, non_quant_dtype, overwrite))
    return outputs


def convert_audio_adapter_fp8(model_path, output_path):
    # Keep the original SekoTalk adapter conversion available without importing
    # CUDA-specific quantization code in the Wan NVFP4 CPU conversion path.
    project_root = Path(__file__).parent.parent.parent
    if str(project_root) not in sys.path:
        sys.path.append(str(project_root))
    quant_path = str(Path(__file__).parent / "quant")
    if quant_path not in sys.path:
        sys.path.insert(0, quant_path)

    from lightx2v.utils.quant_utils import FloatQuantizer

    output_path.parent.mkdir(parents=True, exist_ok=True)
    state_dict = {}
    with safetensors.safe_open(model_path, framework="pt", device="cpu") as source:
        for key in source.keys():
            state_dict[key] = source.get_tensor(key)

    converted = {}
    for key, tensor in state_dict.items():
        if key.startswith("ca") and ".to" in key and "weight" in key:
            print(f"Converting {key} to FP8, dtype: {tensor.dtype}")
            weight = tensor.to(torch.float32).cuda()
            quantizer = FloatQuantizer("e4m3", True, "per_channel")
            weight, weight_scale, _ = quantizer.real_quant_tensor(weight)
            converted[key] = weight.to(torch.float8_e4m3fn).cpu()
            converted[f"{key}_scale"] = weight_scale.to(torch.float32).cpu()
        else:
            print(f"Converting {key} to BF16, dtype: {tensor.dtype}")
            converted[key] = tensor.to(torch.bfloat16)

    save_file(converted, str(output_path))
    print(f"Quantized adapter saved to: {output_path}")


def parse_args():
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent
    parser = argparse.ArgumentParser(description="Convert a SekoTalk adapter to FP8 or LightX2V Wan2.2 NVFP4 checkpoints to INT8-NPU.")
    parser.add_argument(
        "--model_path",
        type=Path,
        default=project_root / "models" / "SekoTalk-Distill" / "audio_adapter_model.safetensors",
        help="Input adapter file, Wan NVFP4 file, or Wan NVFP4 directory.",
    )
    parser.add_argument(
        "--output_path",
        type=Path,
        default=project_root / "models" / "SekoTalk-Distill-fp8" / "audio_adapter_model_fp8.safetensors",
        help="Output file or directory.",
    )
    parser.add_argument(
        "--mode",
        choices=("auto", "adapter-fp8", "wan-nvfp4-to-int8"),
        default="auto",
        help="Conversion mode. Auto selects Wan conversion for directory inputs and adapter conversion for file inputs.",
    )
    parser.add_argument(
        "--non_quant_dtype",
        choices=("float32", "bfloat16"),
        default="float32",
        help="Dtype for non-quantized tensors in Wan INT8 checkpoints. float32 matches existing LightX2V INT8-NPU files.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Replace existing output files.")
    return parser.parse_args()


def main():
    args = parse_args()
    mode = args.mode
    if mode == "auto":
        mode = "wan-nvfp4-to-int8" if args.model_path.is_dir() else "adapter-fp8"

    if mode == "adapter-fp8":
        convert_audio_adapter_fp8(args.model_path, args.output_path)
        return

    non_quant_dtype = torch.float32 if args.non_quant_dtype == "float32" else torch.bfloat16
    outputs = convert_wan_nvfp4_path(args.model_path, args.output_path, non_quant_dtype, args.overwrite)
    print(f"Converted {len(outputs)} Wan2.2 checkpoint(s).", flush=True)


if __name__ == "__main__":
    main()
