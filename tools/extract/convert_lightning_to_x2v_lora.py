#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Lightning LoRA to Lightx2v LoRA Converter

This script converts Lightning LoRA format to Lightx2v LoRA format by:
1. Absorbing alpha/rank scaling into weights
2. Removing alpha parameters
3. Optionally converting to bf16

Usage:
    python convert_lightning_to_x2v_lora.py --input-lora input.safetensors --output-lora output.safetensors [--to-bf16]
"""

import argparse
import os
from collections import defaultdict

import torch
from safetensors import safe_open
from safetensors.torch import save_file


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Convert Lightning LoRA format to Lightx2v LoRA format", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input-lora", type=str, required=True, help="Path to input Lightning LoRA model")
    parser.add_argument("--output-lora", type=str, required=True, help="Path to output Lightx2v LoRA model")
    parser.add_argument("--to-bf16", action="store_true", help="Convert output weights to bf16 format")
    return parser.parse_args()


def load_lora_weights(lora_path: str) -> dict:
    """
    Load LoRA weights from file

    Args:
        lora_path: Path to LoRA model file

    Returns:
        Dictionary of weights
    """
    print(f"Loading LoRA model: {lora_path}")

    if not os.path.exists(lora_path):
        raise FileNotFoundError(f"Input LoRA file does not exist: {lora_path}")

    weights = {}

    if lora_path.endswith(".safetensors"):
        with safe_open(lora_path, framework="pt", device="cpu") as f:
            for key in f.keys():
                weights[key] = f.get_tensor(key)
    elif lora_path.endswith((".pt", ".pth")):
        checkpoint = torch.load(lora_path, map_location="cpu")
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
        raise ValueError(f"Unsupported file format: {lora_path}")

    print(f"Loaded {len(weights)} weight tensors")
    return weights


def get_lora_base_key(key: str) -> str:
    """
    Extract base key from LoRA weight key

    Examples:
        diffusion_model.blocks.0.cross_attn.o.alpha -> diffusion_model.blocks.0.cross_attn.o
        diffusion_model.blocks.0.cross_attn.o.lora_down.weight -> diffusion_model.blocks.0.cross_attn.o
        diffusion_model.blocks.0.cross_attn.o.lora_up.weight -> diffusion_model.blocks.0.cross_attn.o
    """
    if key.endswith(".alpha"):
        return key[:-6]  # Remove ".alpha"
    elif key.endswith(".lora_down.weight"):
        return key[:-17]  # Remove ".lora_down.weight"
    elif key.endswith(".lora_up.weight"):
        return key[:-15]  # Remove ".lora_up.weight"
    else:
        return None


def convert_lightning_to_x2v(weights: dict, to_bf16: bool = False) -> dict:
    """
    Convert Lightning LoRA weights to Lightx2v format

    The conversion process:
    1. Group weights by base key (alpha, lora_down, lora_up)
    2. For each group, compute: weight = weight * alpha / rank
    3. Remove alpha parameters from output

    Args:
        weights: Input LoRA weights dictionary
        to_bf16: Whether to convert to bf16 format

    Returns:
        Converted LoRA weights dictionary
    """
    print("Starting conversion...")

    # Group weights by base key
    groups = defaultdict(dict)
    other_weights = {}

    for key, tensor in weights.items():
        base_key = get_lora_base_key(key)
        if base_key is not None:
            if key.endswith(".alpha"):
                groups[base_key]["alpha"] = tensor
            elif key.endswith(".lora_down.weight"):
                groups[base_key]["lora_down"] = tensor
            elif key.endswith(".lora_up.weight"):
                groups[base_key]["lora_up"] = tensor
        else:
            # Keep other weights as-is
            other_weights[key] = tensor

    print(f"Found {len(groups)} LoRA groups")
    if other_weights:
        print(f"Found {len(other_weights)} other weights (will be kept as-is)")

    # Convert weights
    converted_weights = {}
    converted_count = 0
    skipped_count = 0

    for base_key, group in groups.items():
        has_alpha = "alpha" in group
        has_lora_down = "lora_down" in group
        has_lora_up = "lora_up" in group

        if not (has_lora_down and has_lora_up):
            print(f"Warning: Incomplete LoRA group for {base_key}, skipping...")
            skipped_count += 1
            continue

        lora_down = group["lora_down"]
        lora_up = group["lora_up"]

        # Get rank from lora_down shape (rank is the first dimension)
        rank = lora_down.shape[0]

        # Get alpha value (default to rank if not present)
        if has_alpha:
            alpha = group["alpha"]
            # Alpha might be a scalar tensor
            if alpha.numel() == 1:
                alpha_value = alpha.item()
            else:
                alpha_value = float(alpha)
        else:
            # If no alpha, assume alpha = rank (no scaling)
            alpha_value = float(rank)
            print(f"Warning: No alpha found for {base_key}, using default alpha={alpha_value}")

        # Compute scaling factor: alpha / rank
        scale = alpha_value / rank

        # Apply scaling to weights
        # We can apply sqrt(scale) to both lora_down and lora_up
        # Or apply full scale to one of them
        # Here we apply full scale to lora_up (more common convention)
        scaled_lora_down = lora_down.float()
        scaled_lora_up = lora_up.float() * scale

        # Determine output dtype
        if to_bf16:
            output_dtype = torch.bfloat16
        else:
            # Use original dtype
            output_dtype = lora_down.dtype

        # Convert to output dtype
        scaled_lora_down = scaled_lora_down.to(output_dtype)
        scaled_lora_up = scaled_lora_up.to(output_dtype)

        # Save converted weights (without alpha)
        converted_weights[f"{base_key}.lora_down.weight"] = scaled_lora_down
        converted_weights[f"{base_key}.lora_up.weight"] = scaled_lora_up

        converted_count += 1

    # Add other weights
    for key, tensor in other_weights.items():
        if to_bf16 and tensor.dtype.is_floating_point:
            converted_weights[key] = tensor.to(torch.bfloat16)
        else:
            converted_weights[key] = tensor

    print(f"\nConversion statistics:")
    print(f"  Converted LoRA groups: {converted_count}")
    print(f"  Skipped groups: {skipped_count}")
    print(f"  Other weights: {len(other_weights)}")
    print(f"  Total output weights: {len(converted_weights)}")

    return converted_weights


def save_lora_weights(weights: dict, output_path: str):
    """
    Save LoRA weights to file

    Args:
        weights: Weights dictionary
        output_path: Output file path
    """
    print(f"\nSaving converted LoRA model to: {output_path}")

    # Ensure output directory exists
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # Ensure tensors are contiguous
    contiguous_weights = {}
    for key, tensor in weights.items():
        contiguous_weights[key] = tensor.contiguous()

    # Save based on file extension
    if output_path.endswith(".safetensors"):
        save_file(contiguous_weights, output_path)
    elif output_path.endswith((".pt", ".pth")):
        torch.save(contiguous_weights, output_path)
    else:
        # Default to safetensors
        if not output_path.endswith(".safetensors"):
            output_path += ".safetensors"
        save_file(contiguous_weights, output_path)

    print(f"Successfully saved to: {output_path}")


def main():
    """Main function"""
    args = parse_args()

    print("=" * 60)
    print("Lightning LoRA to Lightx2v LoRA Converter")
    print("=" * 60)
    print(f"Input LoRA:  {args.input_lora}")
    print(f"Output LoRA: {args.output_lora}")
    print(f"Convert to bf16: {args.to_bf16}")
    print("=" * 60)

    try:
        # Load input LoRA weights
        weights = load_lora_weights(args.input_lora)

        # Convert weights
        converted_weights = convert_lightning_to_x2v(weights, to_bf16=args.to_bf16)

        # Save converted weights
        save_lora_weights(converted_weights, args.output_lora)

        print("=" * 60)
        print("Conversion completed successfully!")
        print("=" * 60)

    except Exception as e:
        print(f"Error: {e}")
        raise


if __name__ == "__main__":
    main()
