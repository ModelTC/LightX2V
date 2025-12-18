#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convert ViGen-DiT Model Format (Diffusers) to Lightx2v (Standard Wan)

Usage:
    python convert_vigen_to_x2v_model.py --input-model model.pt --output-model model_converted.safetensors [--to-bf16]

ViGen-DiT Project Url: https://github.com/yl-1993/ViGen-DiT
"""

import argparse
import os
import re

import torch
from safetensors.torch import load_file, save_file

# Key replacement rules: (pattern, replacement)
# Applied in order, first match wins for regex patterns
KEY_REPLACEMENTS = [
    # Skip to_out.1 (bias placeholder)
    (r"\.to_out\.1\.", None),  # None means skip this key
    # Attention: attn1 -> self_attn, attn2 -> cross_attn
    ("attn1", "self_attn"),
    ("attn2", "cross_attn"),
    # Projection: to_q/k/v/out -> q/k/v/o
    (".to_q.", ".q."),
    (".to_k.", ".k."),
    (".to_v.", ".v."),
    (".to_out.0.", ".o."),
    (".to_out.", ".o."),
    # Image projection
    (".add_k_proj.", ".k_img."),
    (".add_v_proj.", ".v_img."),
    # FFN: ffn.net.0.proj -> ffn.0, ffn.net.2 -> ffn.2
    (".ffn.net.0.proj.", ".ffn.0."),
    (".ffn.net.2.", ".ffn.2."),
    # Modulation
    (".scale_shift_table", ".modulation"),
    # Norm: norm2 -> norm3
    (".norm2.", ".norm3."),
    # Text embedding: linear_1 -> 0, linear_2 -> 2
    ("condition_embedder.text_embedder.linear_1", "text_embedding.0"),
    ("condition_embedder.text_embedder.linear_2", "text_embedding.2"),
    # Time embedding: linear_1 -> 0, linear_2 -> 2
    ("condition_embedder.time_embedder.linear_1", "time_embedding.0"),
    ("condition_embedder.time_embedder.linear_2", "time_embedding.2"),
    # Time embedding r: linear_1 -> 0, linear_2 -> 2
    ("condition_embedder.time_embedder_r.linear_1", "time_embedding_r.0"),
    ("condition_embedder.time_embedder_r.linear_2", "time_embedding_r.2"),
    # Time projection
    ("condition_embedder.time_proj.", "time_projection.1."),
    # Head output
    ("proj_out.", "head.head."),
    ("scale_shift_table", "head.modulation"),
    # Remove .default suffix
    (".default", ""),
]


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Convert ViGen-DiT Model Format (Diffusers) to Lightx2v (Standard Wan)", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input-model", type=str, required=True, help="Path to input model")
    parser.add_argument("--output-model", type=str, required=True, help="Path to output model")
    parser.add_argument("--to-bf16", action="store_true", help="Convert to bf16 format")
    return parser.parse_args()


def load_model_weights(model_path: str) -> dict:
    """Load model weights from file"""
    print(f"Loading model: {model_path}")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Input file does not exist: {model_path}")

    if model_path.endswith(".safetensors"):
        state_dict = load_file(model_path)
    else:
        state_dict = torch.load(model_path, map_location="cpu", weights_only=True)

    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    elif "model" in state_dict:
        state_dict = state_dict["model"]

    print(f"Loaded {len(state_dict)} weight tensors")
    return state_dict


def convert_key(key: str) -> str | None:
    """
    Convert a single key from Diffusers format to Wan format.
    Returns None if the key should be skipped.
    """
    new_key = key

    for pattern, replacement in KEY_REPLACEMENTS:
        if replacement is None:
            # Skip pattern - return None if matches
            if re.search(pattern, new_key):
                return None
        elif pattern in new_key:
            new_key = new_key.replace(pattern, replacement)

    return new_key


def convert_vigen_to_x2v(state_dict: dict, to_bf16: bool = False) -> dict:
    """Convert ViGen-DiT model weights to Lightx2v format"""
    print("Starting conversion...")

    mapped_dict = {}
    converted_count = 0
    skipped_count = 0

    for k, v in state_dict.items():
        new_k = convert_key(k)

        if new_k is None:
            skipped_count += 1
            continue

        # Convert dtype if needed
        if to_bf16 and v.dtype.is_floating_point:
            mapped_dict[new_k] = v.to(torch.bfloat16).contiguous()
        else:
            mapped_dict[new_k] = v.contiguous()
        converted_count += 1

    print(f"\nConversion statistics:")
    print(f"  Converted: {converted_count}")
    print(f"  Skipped: {skipped_count}")
    print(f"  Total output weights: {len(mapped_dict)}")

    return mapped_dict


def save_model_weights(weights: dict, output_path: str):
    """Save model weights to safetensors format"""
    print(f"\nSaving to: {output_path}")

    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    if not output_path.endswith(".safetensors"):
        output_path += ".safetensors"

    save_file(weights, output_path)
    print(f"Successfully saved to: {output_path}")


def main():
    """Main function"""
    args = parse_args()

    print("=" * 60)
    print("ViGen-DiT Model to Lightx2v Model Converter")
    print("=" * 60)
    print(f"Input:  {args.input_model}")
    print(f"Output: {args.output_model}")
    print(f"Convert to bf16: {args.to_bf16}")
    print("=" * 60)

    try:
        state_dict = load_model_weights(args.input_model)
        converted_weights = convert_vigen_to_x2v(state_dict, to_bf16=args.to_bf16)
        save_model_weights(converted_weights, args.output_model)

        print("=" * 60)
        print("Conversion completed successfully!")
        print("=" * 60)

    except Exception as e:
        print(f"Error: {e}")
        raise


if __name__ == "__main__":
    main()
