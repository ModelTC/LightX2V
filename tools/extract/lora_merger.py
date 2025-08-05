#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LoRA Merger Script
Merge a source model with LoRA weights to create a new model
"""

import argparse
import os
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from safetensors import safe_open
from safetensors import torch as st
from tqdm import tqdm


def _get_torch_dtype(dtype_str: str) -> torch.dtype:
    """
    Convert string to torch data type

    Args:
        dtype_str: Data type string

    Returns:
        Torch data type
    """
    dtype_mapping = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }

    if dtype_str not in dtype_mapping:
        raise ValueError(f"Unsupported data type: {dtype_str}")

    return dtype_mapping[dtype_str]


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Merge a source model with LoRA weights to create a new model", formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Source model parameters
    parser.add_argument("--source-model", type=str, required=True, help="Path to source model")
    parser.add_argument("--source-type", type=str, choices=["safetensors", "pytorch"], default="safetensors", help="Source model format type")

    # LoRA parameters
    parser.add_argument("--lora-model", type=str, required=True, help="Path to LoRA weights")
    parser.add_argument("--lora-type", type=str, choices=["safetensors", "pytorch"], default="safetensors", help="LoRA weights format type")

    # Output parameters
    parser.add_argument("--output", type=str, required=True, help="Path to output merged model")
    parser.add_argument("--output-format", type=str, choices=["safetensors", "pytorch"], default="safetensors", help="Output model format")

    # Merge parameters
    parser.add_argument("--alpha", type=float, default=1.0, help="LoRA merge strength (alpha value)")
    parser.add_argument("--output-dtype", type=str, choices=["float32", "fp32", "float16", "fp16", "bfloat16", "bf16"], default="bf16", help="Output weight data type")

    return parser.parse_args()


def load_model_weights(model_path: str, model_type: str) -> Dict[str, torch.Tensor]:
    """
    Load model weights (using fp32 precision)

    Args:
        model_path: Model file path or directory path
        model_type: Model type ("safetensors" or "pytorch")

    Returns:
        Model weights dictionary (fp32 precision)
    """
    print(f"Loading model: {model_path} (type: {model_type}, precision: fp32)")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model path does not exist: {model_path}")

    weights = {}

    if model_type == "safetensors":
        if os.path.isdir(model_path):
            # If it's a directory, load all .safetensors files in the directory
            safetensors_files = []
            for file in os.listdir(model_path):
                if file.endswith(".safetensors"):
                    safetensors_files.append(os.path.join(model_path, file))

            if not safetensors_files:
                raise ValueError(f"No .safetensors files found in directory: {model_path}")

            print(f"Found {len(safetensors_files)} safetensors files")

            # Load all files and merge weights
            for file_path in sorted(safetensors_files):
                print(f"  Loading file: {os.path.basename(file_path)}")
                with safe_open(file_path, framework="pt", device="cpu") as f:
                    for key in f.keys():
                        if key in weights:
                            print(f"Warning: weight key '{key}' is duplicated in multiple files, will be overwritten")
                        weights[key] = f.get_tensor(key)

        elif os.path.isfile(model_path):
            # If it's a single file
            if model_path.endswith(".safetensors"):
                with safe_open(model_path, framework="pt", device="cpu") as f:
                    for key in f.keys():
                        weights[key] = f.get_tensor(key)
            else:
                raise ValueError(f"safetensors type file should end with .safetensors: {model_path}")
        else:
            raise ValueError(f"Invalid path type: {model_path}")

    elif model_type == "pytorch":
        # Load pytorch format (.pt, .pth)
        if model_path.endswith((".pt", ".pth")):
            checkpoint = torch.load(model_path, map_location="cpu")

            # Handle possible nested structure
            if isinstance(checkpoint, dict):
                if "state_dict" in checkpoint:
                    weights = checkpoint["state_dict"]
                elif "model" in checkpoint:
                    weights = checkpoint["model"]
                else:
                    weights = checkpoint
            else:
                weights = checkpoint
        else:
            raise ValueError(f"pytorch type file should end with .pt or .pth: {model_path}")
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    # Convert all floating point weights to fp32 to ensure computational precision
    print("Converting weights to fp32 to ensure computational precision...")

    converted_weights = {}
    for key, tensor in weights.items():
        # Only convert floating point tensors, keep integer tensors unchanged
        if tensor.dtype.is_floating_point:
            converted_weights[key] = tensor.to(torch.float32)
        else:
            converted_weights[key] = tensor

    print(f"Successfully loaded model with {len(converted_weights)} weight tensors")
    return converted_weights


def save_model_weights(model_weights: Dict[str, torch.Tensor], output_path: str, output_format: str, output_dtype: str = "bf16"):
    """
    Save model weights

    Args:
        model_weights: Model weights dictionary
        output_path: Output path
        output_format: Output format
        output_dtype: Output data type
    """
    print(f"Saving merged model to: {output_path} (format: {output_format}, data type: {output_dtype})")

    # Ensure output directory exists
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # Convert data type
    target_dtype = _get_torch_dtype(output_dtype)
    print(f"Converting model weights to {output_dtype} type...")

    converted_weights = {}
    with tqdm(model_weights.items(), desc="Converting data type", unit="weights") as pbar:
        for key, tensor in pbar:
            # Only convert floating point tensors, keep integer tensors unchanged
            if tensor.dtype.is_floating_point:
                converted_weights[key] = tensor.to(target_dtype).contiguous()
            else:
                converted_weights[key] = tensor.contiguous()

    if output_format == "safetensors":
        # Save as safetensors format
        if not output_path.endswith(".safetensors"):
            output_path += ".safetensors"
        st.save_file(converted_weights, output_path)

    elif output_format == "pytorch":
        # Save as pytorch format
        if not output_path.endswith((".pt", ".pth")):
            output_path += ".pt"
        torch.save(converted_weights, output_path)
    else:
        raise ValueError(f"Unsupported output format: {output_format}")

    print(f"Merged model saved to: {output_path}")


def merge_lora_weights(source_weights: Dict[str, torch.Tensor], lora_weights: Dict[str, torch.Tensor], alpha: float = 1.0) -> Dict[str, torch.Tensor]:
    """
    Merge source model with LoRA weights

    Args:
        source_weights: Source model weights
        lora_weights: LoRA weights
        alpha: LoRA merge strength

    Returns:
        Merged model weights
    """
    print("Starting LoRA merge...")
    print(f"Merge parameters - alpha: {alpha}")
    print(f"Source model weight count: {len(source_weights)}")
    print(f"LoRA weight count: {len(lora_weights)}")

    merged_weights = source_weights.copy()
    processed_count = 0
    lora_merged_count = 0
    diff_merged_count = 0
    skipped_source_count = 0
    skipped_lora_count = 0
    skipped_source_keys = []
    skipped_lora_keys = []

    # Group LoRA weights by base key
    lora_pairs = {}
    diff_weights = {}

    for lora_key, lora_tensor in lora_weights.items():
        if lora_key.endswith(".lora_up.weight"):
            base_key = lora_key.replace(".lora_up.weight", "")
            if base_key not in lora_pairs:
                lora_pairs[base_key] = {}
            lora_pairs[base_key]["up"] = lora_tensor
        elif lora_key.endswith(".lora_down.weight"):
            base_key = lora_key.replace(".lora_down.weight", "")
            if base_key not in lora_pairs:
                lora_pairs[base_key] = {}
            lora_pairs[base_key]["down"] = lora_tensor
        elif lora_key.endswith((".diff", ".diff_b", ".diff_m")):
            diff_weights[lora_key] = lora_tensor

    print(f"Found {len(lora_pairs)} LoRA pairs and {len(diff_weights)} diff weights")

    # Process with progress bar
    all_items = list(lora_pairs.items()) + list(diff_weights.items())
    pbar = tqdm(all_items, desc="Merging LoRA weights", unit="weight")

    for item in pbar:
        if isinstance(item[1], dict):  # LoRA pair
            base_key, lora_pair = item
            if "up" in lora_pair and "down" in lora_pair:
                # Find corresponding source weight
                source_key = _find_source_key(base_key, source_weights)
                if source_key:
                    if source_weights[source_key].shape != (lora_pair["up"].shape[0], lora_pair["down"].shape[1]):
                        skipped_source_count += 1
                        skipped_source_keys.append(source_key)
                        continue
                    lora_up = lora_pair["up"]
                    lora_down = lora_pair["down"]

                    # Compute LoRA delta: alpha * (lora_up @ lora_down)
                    lora_delta = alpha * (lora_up @ lora_down)

                    # Apply to source weight
                    merged_weights[source_key] = source_weights[source_key] + lora_delta
                    lora_merged_count += 1
                    pbar.set_postfix_str(f"LoRA: {source_key.split('.')[-1]}")
                else:
                    skipped_source_count += 1
                    skipped_source_keys.append(base_key)
            else:
                print(f"Warning: Incomplete LoRA pair for: {base_key}")
                skipped_lora_count += 1
                skipped_lora_keys.append(base_key)
        else:  # Diff weight
            diff_key, diff_tensor = item
            # Find corresponding source weight
            source_key = _find_source_key_from_diff(diff_key, source_weights)
            if source_key:
                if source_weights[source_key].shape != diff_tensor.shape:
                    skipped_source_count += 1
                    skipped_source_keys.append(source_key)
                    continue
                # Apply diff: source + alpha * diff
                merged_weights[source_key] = source_weights[source_key] + alpha * diff_tensor
                diff_merged_count += 1
                pbar.set_postfix_str(f"Diff: {source_key.split('.')[-1]}")
            else:
                skipped_lora_count += 1
                skipped_lora_keys.append(diff_key)

        processed_count += 1

    pbar.close()

    print(f"\nMerge statistics:")
    print(f"  Processed weights: {processed_count}")
    print(f"  LoRA merged: {lora_merged_count}")
    print(f"  Diff merged: {diff_merged_count}")
    print(f"  Skipped source weights: {skipped_source_count}")
    if skipped_source_count > 0:
        print(f"  Skipped source keys:")
        for key in skipped_source_keys:
            print(f"    {key}")
    print(f"  Skipped LoRA weights: {skipped_lora_count}")
    if skipped_lora_count > 0:
        print(f"  Skipped LoRA keys:")
        for key in skipped_lora_keys:
            print(f"    {key}")
    print(f"  Total merged model weights: {len(merged_weights)}")
    print("LoRA merge completed")

    return merged_weights


def _find_source_key(lora_base_key: str, source_weights: Dict[str, torch.Tensor]) -> Optional[str]:
    """
    Find corresponding source weight key for LoRA base key

    Args:
        lora_base_key: LoRA base key (e.g., "diffusion_model.input_blocks.0.0.weight")
        source_weights: Source model weights

    Returns:
        Corresponding source key or None
    """
    # Remove diffusion_model prefix if present
    if lora_base_key.startswith("diffusion_model."):
        source_key = lora_base_key[16:] + ".weight"  # Remove "diffusion_model." and add ".weight"
    else:
        source_key = lora_base_key + ".weight"

    if source_key in source_weights:
        return source_key

    # Try without adding .weight (in case it's already included)
    if lora_base_key.startswith("diffusion_model."):
        source_key_alt = lora_base_key[16:]
    else:
        source_key_alt = lora_base_key

    if source_key_alt in source_weights:
        return source_key_alt

    return None


def _find_source_key_from_diff(diff_key: str, source_weights: Dict[str, torch.Tensor]) -> Optional[str]:
    """
    Find corresponding source weight key for diff key

    Args:
        diff_key: Diff key (e.g., "diffusion_model.input_blocks.0.diff")
        source_weights: Source model weights

    Returns:
        Corresponding source key or None
    """
    # Remove diffusion_model prefix and diff suffix
    if diff_key.startswith("diffusion_model."):
        base_key = diff_key[16:]  # Remove "diffusion_model."
    else:
        base_key = diff_key

    # Remove diff suffixes
    if base_key.endswith(".diff"):
        source_key = base_key[:-5] + ".weight"  # Remove ".diff" with ".weight"
    elif base_key.endswith(".diff_b"):
        source_key = base_key[:-7] + ".bias"  # Replace ".diff_b" with ".bias"
    elif base_key.endswith(".diff_m"):
        source_key = base_key[:-7] + ".modulation"  # Replace ".diff_m" with ".modulation"
    else:
        source_key = base_key

    if source_key in source_weights:
        return source_key

    return None


def main():
    """Main function"""
    args = parse_args()

    print("=" * 50)
    print("LoRA Merger Started")
    print("=" * 50)
    print(f"Source model: {args.source_model} ({args.source_type})")
    print(f"LoRA weights: {args.lora_model} ({args.lora_type})")
    print(f"Output path: {args.output} ({args.output_format})")
    print(f"Output data type: {args.output_dtype}")
    print(f"Merge parameters: alpha={args.alpha}")
    print("=" * 50)

    try:
        # Load source model and LoRA weights
        source_weights = load_model_weights(args.source_model, args.source_type)
        lora_weights = load_model_weights(args.lora_model, args.lora_type)

        # Merge LoRA weights with source model
        merged_weights = merge_lora_weights(source_weights, lora_weights, alpha=args.alpha)

        # Save merged model
        save_model_weights(merged_weights, args.output, args.output_format, args.output_dtype)

        print("=" * 50)
        print("LoRA merge completed!")
        print("=" * 50)

    except Exception as e:
        print(f"Error: {e}")
        raise


if __name__ == "__main__":
    main()
