#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LoRA Extractor Script
Extract LoRA weights from the difference between two models
"""

import argparse
import os
from typing import Dict, Optional

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
    parser = argparse.ArgumentParser(description="Extract LoRA weights from the difference between source and target models", formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Source model parameters
    parser.add_argument("--source-model", type=str, required=True, help="Path to source model")
    parser.add_argument("--source-type", type=str, choices=["safetensors", "pytorch"], default="safetensors", help="Source model format type")

    # Target model parameters
    parser.add_argument("--target-model", type=str, required=True, help="Path to target model (fine-tuned model)")
    parser.add_argument("--target-type", type=str, choices=["safetensors", "pytorch"], default="safetensors", help="Target model format type")

    # Output parameters
    parser.add_argument("--output", type=str, required=True, help="Path to output LoRA model")
    parser.add_argument("--output-format", type=str, choices=["safetensors", "pytorch"], default="safetensors", help="Output LoRA model format")

    # LoRA related parameters
    parser.add_argument("--rank", type=int, default=32, help="LoRA rank value")

    parser.add_argument("--output-dtype", type=str, choices=["float32", "fp32", "float16", "fp16", "bfloat16", "bf16"], default="bf16", help="Output weight data type")
    parser.add_argument("--diff-only", action="store_true", help="Save all weights as direct diff without LoRA decomposition")

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


def save_lora_weights(lora_weights: Dict[str, torch.Tensor], output_path: str, output_format: str, output_dtype: str = "bf16"):
    """
    Save LoRA weights

    Args:
        lora_weights: LoRA weights dictionary
        output_path: Output path
        output_format: Output format
        output_dtype: Output data type
    """
    print(f"Saving LoRA weights to: {output_path} (format: {output_format}, data type: {output_dtype})")

    # Ensure output directory exists
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # Convert data type
    target_dtype = _get_torch_dtype(output_dtype)
    print(f"Converting LoRA weights to {output_dtype} type...")

    converted_weights = {}
    with tqdm(lora_weights.items(), desc="Converting data type", unit="weights") as pbar:
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

    print(f"LoRA weights saved to: {output_path}")


def _compute_weight_diff(source_tensor: torch.Tensor, target_tensor: torch.Tensor, key: str) -> Optional[torch.Tensor]:
    """
    Compute the difference between two weight tensors

    Args:
        source_tensor: Source weight tensor
        target_tensor: Target weight tensor
        key: Weight key name (for logging)

    Returns:
        Difference tensor, returns None if no change
    """
    # Check if tensor shapes match
    if source_tensor.shape != target_tensor.shape:
        return None

    # Check if tensor data types match
    if source_tensor.dtype != target_tensor.dtype:
        target_tensor = target_tensor.to(source_tensor.dtype)

    # Compute difference
    diff = target_tensor - source_tensor

    # Check if there are actual changes
    if torch.allclose(diff, torch.zeros_like(diff), atol=1e-8):
        # No change
        return None

    return diff


def _decompose_to_lora(diff: torch.Tensor, key: str, rank: int) -> Dict[str, torch.Tensor]:
    """
    Decompose weight difference into LoRA format

    Args:
        diff: Weight difference tensor
        key: Original weight key name
        rank: LoRA rank

    Returns:
        LoRA weights dictionary (containing lora_up and lora_down)
    """
    # Ensure it's a 2D tensor
    if len(diff.shape) != 2:
        raise ValueError(f"LoRA decomposition only supports 2D weights, but got {len(diff.shape)}D tensor: {key}")

    a, b = diff.shape

    # Check if rank is reasonable
    max_rank = min(a, b)
    if rank > max_rank:
        rank = max_rank

    # Choose compute device (prefer GPU, fallback to CPU)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    diff_device = diff.to(device)

    # SVD decomposition
    U, S, V = torch.linalg.svd(diff_device, full_matrices=False)

    # Take the first rank components
    U = U[:, :rank]  # (a, rank)
    S = S[:rank]  # (rank,)
    V = V[:rank, :]  # (rank, b)

    # Distribute square root of singular values to both matrices
    S_sqrt = S.sqrt()
    lora_up = U * S_sqrt.unsqueeze(0)  # (a, rank) * (1, rank) = (a, rank)
    lora_down = S_sqrt.unsqueeze(1) * V  # (rank, 1) * (rank, b) = (rank, b)

    # Move back to CPU and convert to original data type, ensure contiguous
    lora_up = lora_up.cpu().to(diff.dtype).contiguous()
    lora_down = lora_down.cpu().to(diff.dtype).contiguous()

    # Generate LoRA weight key names
    base_key = key.replace(".weight", "")
    lora_up_key = "diffusion_model." + f"{base_key}.lora_up.weight"
    lora_down_key = "diffusion_model." + f"{base_key}.lora_down.weight"

    # Return the decomposed weights
    lora_weights = {lora_up_key: lora_up, lora_down_key: lora_down}

    return lora_weights


def extract_lora_from_diff(source_weights: Dict[str, torch.Tensor], target_weights: Dict[str, torch.Tensor], rank: int = 16, diff_only: bool = False) -> Dict[str, torch.Tensor]:
    """
    Extract LoRA weights from model difference

    Args:
        source_weights: Source model weights
        target_weights: Target model weights
        rank: LoRA rank
        diff_only: If True, save all weights as direct diff without LoRA decomposition

    Returns:
        LoRA weights dictionary
    """
    print("Starting LoRA weight extraction...")
    if diff_only:
        print("Mode: Direct diff only (no LoRA decomposition)")
    else:
        print(f"Mode: Smart extraction - rank: {rank}")
    print(f"Source model weight count: {len(source_weights)}")
    print(f"Target model weight count: {len(target_weights)}")

    lora_weights = {}
    processed_count = 0
    diff_count = 0
    lora_count = 0
    similar_count = 0
    skipped_count = 0
    fail_count = 0

    # Find common keys between two models
    common_keys = set(source_weights.keys()) & set(target_weights.keys())
    source_only_keys = set(source_weights.keys()) - set(target_weights.keys())
    target_only_keys = set(target_weights.keys()) - set(source_weights.keys())

    if source_only_keys:
        print(f"Warning: Source model exclusive weight keys ({len(source_only_keys)} keys): {list(source_only_keys)[:5]}...")
    if target_only_keys:
        print(f"Warning: Target model exclusive weight keys ({len(target_only_keys)} keys): {list(target_only_keys)[:5]}...")

    print(f"Common weight keys count: {len(common_keys)}")

    # Process common keys, extract LoRA weights
    common_keys_sorted = sorted(common_keys)
    pbar = tqdm(common_keys_sorted, desc="Extracting LoRA weights", unit="layer")

    for key in pbar:
        source_tensor = source_weights[key]
        target_tensor = target_weights[key]

        # Update progress bar description
        short_key = key.split(".")[-2:] if "." in key else [key]
        pbar.set_postfix_str(f"Processing: {'.'.join(short_key)}")

        # Compute weight difference
        diff = _compute_weight_diff(source_tensor, target_tensor, key)

        if diff is None:
            # No change or shape mismatch
            if source_tensor.shape == target_tensor.shape:
                similar_count += 1
            else:
                skipped_count += 1
            continue

        # Calculate parameter count
        param_count = source_tensor.numel()
        is_1d = len(source_tensor.shape) == 1

        # Decide whether to save diff directly or perform LoRA decomposition
        if diff_only or is_1d or param_count < 1000000:
            # Save diff directly
            lora_key = _generate_lora_diff_key(key)
            if lora_key == "skip":
                skipped_count += 1
                continue
            lora_weights[lora_key] = diff
            diff_count += 1

        else:
            # Perform LoRA decomposition
            if len(diff.shape) == 2 and key.endswith(".weight"):
                try:
                    decomposed_weights = _decompose_to_lora(diff, key, rank)
                    lora_weights.update(decomposed_weights)
                    lora_count += 1
                except Exception as e:
                    print(f"Error: {e}")
                    fail_count += 1

            else:
                print(f"Error: {key} is not a 2D weight tensor")
                fail_count += 1

        processed_count += 1

    # Close progress bar
    pbar.close()

    print(f"\nExtraction statistics:")
    print(f"  Processed weights: {processed_count}")
    print(f"  Direct diff: {diff_count}")
    print(f"  LoRA decomposition: {lora_count}")
    print(f"  Skipped weights: {skipped_count}")
    print(f"  Similar weights: {similar_count}")
    print(f"  Failed weights: {fail_count}")
    print(f"  Total extracted LoRA weights: {len(lora_weights)}")
    print("LoRA weight extraction completed")

    return lora_weights


def _generate_lora_diff_key(original_key: str) -> str:
    """
    Generate LoRA weight key based on original weight key

    Args:
        original_key: Original weight key name

    Returns:
        LoRA weight key name
    """
    ret_key = "diffusion_model." + original_key
    if original_key.endswith(".weight"):
        return ret_key.replace(".weight", ".diff")
    elif original_key.endswith(".bias"):
        return ret_key.replace(".bias", ".diff_b")
    elif original_key.endswith(".modulation"):
        return ret_key.replace(".modulation", ".diff_m")
    else:
        # If no matching suffix, skip
        return "skip"


def main():
    """Main function"""
    args = parse_args()

    print("=" * 50)
    print("LoRA Extractor Started")
    print("=" * 50)
    print(f"Source model: {args.source_model} ({args.source_type})")
    print(f"Target model: {args.target_model} ({args.target_type})")
    print(f"Output path: {args.output} ({args.output_format})")
    print(f"Output data type: {args.output_dtype}")
    print(f"LoRA parameters: rank={args.rank}")
    print(f"Diff only mode: {args.diff_only}")
    print("=" * 50)

    try:
        # Load source and target models
        source_weights = load_model_weights(args.source_model, args.source_type)
        target_weights = load_model_weights(args.target_model, args.target_type)

        # Extract LoRA weights
        lora_weights = extract_lora_from_diff(source_weights, target_weights, rank=args.rank, diff_only=args.diff_only)

        # Save LoRA weights
        save_lora_weights(lora_weights, args.output, args.output_format, args.output_dtype)

        print("=" * 50)
        print("LoRA extraction completed!")
        print("=" * 50)

    except Exception as e:
        print(f"Error: {e}")
        raise


if __name__ == "__main__":
    main()
