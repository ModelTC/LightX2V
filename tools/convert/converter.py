import argparse
import gc
import glob
import json
import os
import re
import shutil
from collections import defaultdict

import torch
from loguru import logger
from qtorch.quant import float_quantize
from safetensors import safe_open
from safetensors import torch as st
from tqdm import tqdm


def get_key_mapping_rules(direction, model_type):
    if model_type == "wan_dit":
        unified_rules = [
            {
                "forward": (r"^head\.head$", "proj_out"),
                "backward": (r"^proj_out$", "head.head"),
            },
            {
                "forward": (r"^head\.modulation$", "scale_shift_table"),
                "backward": (r"^scale_shift_table$", "head.modulation"),
            },
            {
                "forward": (
                    r"^text_embedding\.0\.",
                    "condition_embedder.text_embedder.linear_1.",
                ),
                "backward": (
                    r"^condition_embedder.text_embedder.linear_1\.",
                    "text_embedding.0.",
                ),
            },
            {
                "forward": (
                    r"^text_embedding\.2\.",
                    "condition_embedder.text_embedder.linear_2.",
                ),
                "backward": (
                    r"^condition_embedder.text_embedder.linear_2\.",
                    "text_embedding.2.",
                ),
            },
            {
                "forward": (
                    r"^time_embedding\.0\.",
                    "condition_embedder.time_embedder.linear_1.",
                ),
                "backward": (
                    r"^condition_embedder.time_embedder.linear_1\.",
                    "time_embedding.0.",
                ),
            },
            {
                "forward": (
                    r"^time_embedding\.2\.",
                    "condition_embedder.time_embedder.linear_2.",
                ),
                "backward": (
                    r"^condition_embedder.time_embedder.linear_2\.",
                    "time_embedding.2.",
                ),
            },
            {
                "forward": (r"^time_projection\.1\.", "condition_embedder.time_proj."),
                "backward": (r"^condition_embedder.time_proj\.", "time_projection.1."),
            },
            {
                "forward": (r"blocks\.(\d+)\.self_attn\.q\.", r"blocks.\1.attn1.to_q."),
                "backward": (
                    r"blocks\.(\d+)\.attn1\.to_q\.",
                    r"blocks.\1.self_attn.q.",
                ),
            },
            {
                "forward": (r"blocks\.(\d+)\.self_attn\.k\.", r"blocks.\1.attn1.to_k."),
                "backward": (
                    r"blocks\.(\d+)\.attn1\.to_k\.",
                    r"blocks.\1.self_attn.k.",
                ),
            },
            {
                "forward": (r"blocks\.(\d+)\.self_attn\.v\.", r"blocks.\1.attn1.to_v."),
                "backward": (
                    r"blocks\.(\d+)\.attn1\.to_v\.",
                    r"blocks.\1.self_attn.v.",
                ),
            },
            {
                "forward": (
                    r"blocks\.(\d+)\.self_attn\.o\.",
                    r"blocks.\1.attn1.to_out.0.",
                ),
                "backward": (
                    r"blocks\.(\d+)\.attn1\.to_out\.0\.",
                    r"blocks.\1.self_attn.o.",
                ),
            },
            {
                "forward": (
                    r"blocks\.(\d+)\.cross_attn\.q\.",
                    r"blocks.\1.attn2.to_q.",
                ),
                "backward": (
                    r"blocks\.(\d+)\.attn2\.to_q\.",
                    r"blocks.\1.cross_attn.q.",
                ),
            },
            {
                "forward": (
                    r"blocks\.(\d+)\.cross_attn\.k\.",
                    r"blocks.\1.attn2.to_k.",
                ),
                "backward": (
                    r"blocks\.(\d+)\.attn2\.to_k\.",
                    r"blocks.\1.cross_attn.k.",
                ),
            },
            {
                "forward": (
                    r"blocks\.(\d+)\.cross_attn\.v\.",
                    r"blocks.\1.attn2.to_v.",
                ),
                "backward": (
                    r"blocks\.(\d+)\.attn2\.to_v\.",
                    r"blocks.\1.cross_attn.v.",
                ),
            },
            {
                "forward": (
                    r"blocks\.(\d+)\.cross_attn\.o\.",
                    r"blocks.\1.attn2.to_out.0.",
                ),
                "backward": (
                    r"blocks\.(\d+)\.attn2\.to_out\.0\.",
                    r"blocks.\1.cross_attn.o.",
                ),
            },
            {
                "forward": (r"blocks\.(\d+)\.norm3\.", r"blocks.\1.norm2."),
                "backward": (r"blocks\.(\d+)\.norm2\.", r"blocks.\1.norm3."),
            },
            {
                "forward": (r"blocks\.(\d+)\.ffn\.0\.", r"blocks.\1.ffn.net.0.proj."),
                "backward": (
                    r"blocks\.(\d+)\.ffn\.net\.0\.proj\.",
                    r"blocks.\1.ffn.0.",
                ),
            },
            {
                "forward": (r"blocks\.(\d+)\.ffn\.2\.", r"blocks.\1.ffn.net.2."),
                "backward": (r"blocks\.(\d+)\.ffn\.net\.2\.", r"blocks.\1.ffn.2."),
            },
            {
                "forward": (
                    r"blocks\.(\d+)\.modulation\.",
                    r"blocks.\1.scale_shift_table.",
                ),
                "backward": (
                    r"blocks\.(\d+)\.scale_shift_table(?=\.|$)",
                    r"blocks.\1.modulation",
                ),
            },
            {
                "forward": (
                    r"blocks\.(\d+)\.cross_attn\.k_img\.",
                    r"blocks.\1.attn2.add_k_proj.",
                ),
                "backward": (
                    r"blocks\.(\d+)\.attn2\.add_k_proj\.",
                    r"blocks.\1.cross_attn.k_img.",
                ),
            },
            {
                "forward": (
                    r"blocks\.(\d+)\.cross_attn\.v_img\.",
                    r"blocks.\1.attn2.add_v_proj.",
                ),
                "backward": (
                    r"blocks\.(\d+)\.attn2\.add_v_proj\.",
                    r"blocks.\1.cross_attn.v_img.",
                ),
            },
            {
                "forward": (
                    r"blocks\.(\d+)\.cross_attn\.norm_k_img\.weight",
                    r"blocks.\1.attn2.norm_added_k.weight",
                ),
                "backward": (
                    r"blocks\.(\d+)\.attn2\.norm_added_k\.weight",
                    r"blocks.\1.cross_attn.norm_k_img.weight",
                ),
            },
            {
                "forward": (
                    r"img_emb\.proj\.0\.",
                    r"condition_embedder.image_embedder.norm1.",
                ),
                "backward": (
                    r"condition_embedder\.image_embedder\.norm1\.",
                    r"img_emb.proj.0.",
                ),
            },
            {
                "forward": (
                    r"img_emb\.proj\.1\.",
                    r"condition_embedder.image_embedder.ff.net.0.proj.",
                ),
                "backward": (
                    r"condition_embedder\.image_embedder\.ff\.net\.0\.proj\.",
                    r"img_emb.proj.1.",
                ),
            },
            {
                "forward": (
                    r"img_emb\.proj\.3\.",
                    r"condition_embedder.image_embedder.ff.net.2.",
                ),
                "backward": (
                    r"condition_embedder\.image_embedder\.ff\.net\.2\.",
                    r"img_emb.proj.3.",
                ),
            },
            {
                "forward": (
                    r"img_emb\.proj\.4\.",
                    r"condition_embedder.image_embedder.norm2.",
                ),
                "backward": (
                    r"condition_embedder\.image_embedder\.norm2\.",
                    r"img_emb.proj.4.",
                ),
            },
            {
                "forward": (
                    r"blocks\.(\d+)\.self_attn\.norm_q\.weight",
                    r"blocks.\1.attn1.norm_q.weight",
                ),
                "backward": (
                    r"blocks\.(\d+)\.attn1\.norm_q\.weight",
                    r"blocks.\1.self_attn.norm_q.weight",
                ),
            },
            {
                "forward": (
                    r"blocks\.(\d+)\.self_attn\.norm_k\.weight",
                    r"blocks.\1.attn1.norm_k.weight",
                ),
                "backward": (
                    r"blocks\.(\d+)\.attn1\.norm_k\.weight",
                    r"blocks.\1.self_attn.norm_k.weight",
                ),
            },
            {
                "forward": (
                    r"blocks\.(\d+)\.cross_attn\.norm_q\.weight",
                    r"blocks.\1.attn2.norm_q.weight",
                ),
                "backward": (
                    r"blocks\.(\d+)\.attn2\.norm_q\.weight",
                    r"blocks.\1.cross_attn.norm_q.weight",
                ),
            },
            {
                "forward": (
                    r"blocks\.(\d+)\.cross_attn\.norm_k\.weight",
                    r"blocks.\1.attn2.norm_k.weight",
                ),
                "backward": (
                    r"blocks\.(\d+)\.attn2\.norm_k\.weight",
                    r"blocks.\1.cross_attn.norm_k.weight",
                ),
            },
            # head projection mapping
            {
                "forward": (r"^head\.head\.", "proj_out."),
                "backward": (r"^proj_out\.", "head.head."),
            },
        ]

        if direction == "forward":
            return [rule["forward"] for rule in unified_rules]
        elif direction == "backward":
            return [rule["backward"] for rule in unified_rules]
        else:
            raise ValueError(f"Invalid direction: {direction}")
    else:
        raise ValueError(f"Unsupported model type: {model_type}")


def quantize_tensor(w, w_bit=8, dtype=torch.int8):
    """
    Quantize a 2D tensor to specified bit width using symmetric min-max quantization

    Args:
        w: Input tensor to quantize (must be 2D)
        w_bit: Quantization bit width (default: 8)

    Returns:
        quantized: Quantized tensor (int8)
        scales: Scaling factors per row
    """
    if w.dim() != 2:
        raise ValueError(f"Only 2D tensors supported. Got {w.dim()}D tensor")
    if torch.isnan(w).any():
        raise ValueError("Tensor contains NaN values")
    if w_bit != 8:
        raise ValueError("Only support 8 bits")

    org_w_shape = w.shape
    # Calculate quantization parameters
    max_val = w.abs().amax(dim=1, keepdim=True).clamp(min=1e-5)

    if dtype == torch.float8_e4m3fn:
        finfo = torch.finfo(dtype)
        qmin, qmax = finfo.min, finfo.max
    elif dtype == torch.int8:
        qmin, qmax = -128, 127

    # Quantize tensor
    scales = max_val / qmax

    if dtype == torch.float8_e4m3fn:
        scaled_tensor = w / scales
        scaled_tensor = torch.clip(scaled_tensor, qmin, qmax)
        w_q = float_quantize(scaled_tensor.float(), 4, 3, rounding="nearest").to(dtype)
    else:
        w_q = torch.clamp(torch.round(w / scales), qmin, qmax).to(dtype)

    assert torch.isnan(scales).sum() == 0
    assert torch.isnan(w_q).sum() == 0

    scales = scales.view(org_w_shape[0], -1)
    w_q = w_q.reshape(org_w_shape)

    return w_q, scales


def quantize_model(
    weights,
    w_bit=8,
    target_keys=["attn", "ffn"],
    adapter_keys=None,
    key_idx=2,
    ignore_key=None,
    linear_dtype=torch.int8,
    non_linear_dtype=torch.float,
):
    """
    Quantize model weights in-place

    Args:
        weights: Model state dictionary
        w_bit: Quantization bit width
        target_keys: List of module names to quantize

    Returns:
        Modified state dictionary with quantized weights and scales
    """
    total_quantized = 0
    total_size = 0
    keys = list(weights.keys())

    with tqdm(keys, desc="Quantizing weights") as pbar:
        for key in pbar:
            pbar.set_postfix(current_key=key, refresh=False)

            if ignore_key is not None and any(ig_key in key for ig_key in ignore_key):
                del weights[key]
                continue

            tensor = weights[key]

            # Skip non-tensors and non-2D tensors
            if not isinstance(tensor, torch.Tensor) or tensor.dim() != 2:
                if tensor.dtype != non_linear_dtype:
                    weights[key] = tensor.to(non_linear_dtype)
                continue

            # Check if key matches target modules

            parts = key.split(".")

            if len(parts) < key_idx + 1 or parts[key_idx] not in target_keys:
                if adapter_keys is not None and not any(adapter_key in parts for adapter_key in adapter_keys):
                    if tensor.dtype != non_linear_dtype:
                        weights[key] = tensor.to(non_linear_dtype)
                    continue

            try:
                # Quantize tensor and store results
                w_q, scales = quantize_tensor(tensor, w_bit, linear_dtype)

                # Replace original tensor and store scales
                weights[key] = w_q
                weights[key + "_scale"] = scales

                total_quantized += 1
                total_size += tensor.numel() * tensor.element_size() / (1024**2)  # MB
                del w_q, scales

            except Exception as e:
                logger.error(f"Error quantizing {key}: {str(e)}")

            gc.collect()

        logger.info(f"Quantized {total_quantized} tensors, reduced size by {total_size:.2f} MB")
    return weights


def load_loras(lora_path, weight_dict, alpha):
    logger.info(f"Loading LoRA from: {lora_path}")
    with safe_open(lora_path, framework="pt") as f:
        lora_weights = {k: f.get_tensor(k) for k in f.keys()}

    lora_pairs = {}
    lora_diffs = {}
    prefix = "diffusion_model."

    def try_lora_pair(key, suffix_a, suffix_b, target_suffix):
        if key.endswith(suffix_a):
            base_name = key[len(prefix) :].replace(suffix_a, target_suffix)
            pair_key = key.replace(suffix_a, suffix_b)
            if pair_key in lora_weights:
                lora_pairs[base_name] = (key, pair_key)

    def try_lora_diff(key, suffix, target_suffix):
        if key.endswith(suffix):
            base_name = key[len(prefix) :].replace(suffix, target_suffix)
            lora_diffs[base_name] = key

    for key in lora_weights.keys():
        if not key.startswith(prefix):
            continue

        try_lora_pair(key, "lora_A.weight", "lora_B.weight", "weight")
        try_lora_pair(key, "lora_down.weight", "lora_up.weight", "weight")
        try_lora_diff(key, "diff", "weight")
        try_lora_diff(key, "diff_b", "bias")
        try_lora_diff(key, "diff_m", "modulation")

    applied_count = 0
    for name, param in weight_dict.items():
        if name in lora_pairs:
            name_lora_A, name_lora_B = lora_pairs[name]
            lora_A = lora_weights[name_lora_A].to(param.device, param.dtype)
            lora_B = lora_weights[name_lora_B].to(param.device, param.dtype)
            param += torch.matmul(lora_B, lora_A) * alpha
            applied_count += 1
        elif name in lora_diffs:
            name_diff = lora_diffs[name]
            lora_diff = lora_weights[name_diff].to(param.device, param.dtype)
            try:
                param += lora_diff * alpha
                applied_count += 1
            except Exception as e:
                continue
    logger.info(f"Applied {applied_count} LoRA weight adjustments")


def convert_weights(args):
    if os.path.isdir(args.source):
        src_files = glob.glob(os.path.join(args.source, "*.safetensors"), recursive=True)
    elif args.source.endswith((".pth", ".safetensors", "pt")):
        src_files = [args.source]
    else:
        raise ValueError("Invalid input path")

    merged_weights = {}
    logger.info(f"Processing source files: {src_files}")
    for file_path in tqdm(src_files, desc="Loading weights"):
        logger.info(f"Loading weights from: {file_path}")
        if file_path.endswith(".pt") or file_path.endswith(".pth"):
            weights = torch.load(file_path, map_location=args.device, weights_only=True)
            if args.model_type == "hunyuan_dit":
                weights = weights["module"]
        elif file_path.endswith(".safetensors"):
            with safe_open(file_path, framework="pt") as f:
                weights = {k: f.get_tensor(k) for k in f.keys()}

        duplicate_keys = set(weights.keys()) & set(merged_weights.keys())
        if duplicate_keys:
            raise ValueError(f"Duplicate keys found: {duplicate_keys} in file {file_path}")
        merged_weights.update(weights)

    if args.lora_path is not None:
        # Handle alpha list - if single alpha, replicate for all LoRAs
        if len(args.lora_alpha) == 1 and len(args.lora_path) > 1:
            args.lora_alpha = args.lora_alpha * len(args.lora_path)
        elif len(args.lora_alpha) != len(args.lora_path):
            raise ValueError(f"Number of lora_alpha ({len(args.lora_alpha)}) must match number of lora_path ({len(args.lora_path)}) or be 1")

        for path, alpha in zip(args.lora_path, args.lora_alpha):
            load_loras(path, merged_weights, alpha)

    if args.direction is not None:
        rules = get_key_mapping_rules(args.direction, args.model_type)
        converted_weights = {}
        logger.info("Converting keys...")
        for key in tqdm(merged_weights.keys(), desc="Converting keys"):
            new_key = key
            for pattern, replacement in rules:
                new_key = re.sub(pattern, replacement, new_key)
            converted_weights[new_key] = merged_weights[key]
    else:
        converted_weights = merged_weights

    if args.quantized:
        converted_weights = quantize_model(
            converted_weights,
            w_bit=args.bits,
            target_keys=args.target_keys,
            adapter_keys=args.adapter_keys,
            key_idx=args.key_idx,
            ignore_key=args.ignore_key,
            linear_dtype=args.linear_dtype,
            non_linear_dtype=args.non_linear_dtype,
        )

    os.makedirs(args.output, exist_ok=True)

    if args.output_ext == ".pth":
        torch.save(converted_weights, os.path.join(args.output, args.output_name + ".pth"))

    else:
        index = {"metadata": {"total_size": 0}, "weight_map": {}}

        if args.save_by_block:
            logger.info("Backward conversion: grouping weights by block")
            block_groups = defaultdict(dict)
            non_block_weights = {}
            block_pattern = re.compile(r"blocks\.(\d+)\.")

            for key, tensor in converted_weights.items():
                match = block_pattern.search(key)
                if match:
                    block_idx = match.group(1)
                    if args.model_type == "wan_animate_dit" and "face_adapter" in key:
                        block_idx = str(int(block_idx) * 5)
                    block_groups[block_idx][key] = tensor
                else:
                    non_block_weights[key] = tensor

            for block_idx, weights_dict in tqdm(block_groups.items(), desc="Saving block chunks"):
                output_filename = f"block_{block_idx}.safetensors"
                output_path = os.path.join(args.output, output_filename)
                st.save_file(weights_dict, output_path)
                for key in weights_dict:
                    index["weight_map"][key] = output_filename
                index["metadata"]["total_size"] += os.path.getsize(output_path)

            if non_block_weights:
                output_filename = f"non_block.safetensors"
                output_path = os.path.join(args.output, output_filename)
                st.save_file(non_block_weights, output_path)
                for key in non_block_weights:
                    index["weight_map"][key] = output_filename
                index["metadata"]["total_size"] += os.path.getsize(output_path)

        else:
            chunk_idx = 0
            current_chunk = {}
            for idx, (k, v) in tqdm(enumerate(converted_weights.items()), desc="Saving chunks"):
                current_chunk[k] = v
                if args.chunk_size > 0 and (idx + 1) % args.chunk_size == 0:
                    output_filename = f"{args.output_name}_part{chunk_idx}.safetensors"
                    output_path = os.path.join(args.output, output_filename)
                    logger.info(f"Saving chunk to: {output_path}")
                    st.save_file(current_chunk, output_path)
                    for key in current_chunk:
                        index["weight_map"][key] = output_filename
                    index["metadata"]["total_size"] += os.path.getsize(output_path)
                    current_chunk = {}
                    chunk_idx += 1

            if current_chunk:
                output_filename = f"{args.output_name}_part{chunk_idx}.safetensors"
                output_path = os.path.join(args.output, output_filename)
                logger.info(f"Saving final chunk to: {output_path}")
                st.save_file(current_chunk, output_path)
                for key in current_chunk:
                    index["weight_map"][key] = output_filename
                index["metadata"]["total_size"] += os.path.getsize(output_path)

        # Save index file
        index_path = os.path.join(args.output, "diffusion_pytorch_model.safetensors.index.json")
        with open(index_path, "w", encoding="utf-8") as f:
            json.dump(index, f, indent=2)
        logger.info(f"Index file written to: {index_path}")

    if os.path.isdir(args.source) and args.copy_no_weight_files:
        copy_non_weight_files(args.source, args.output)


def copy_non_weight_files(source_dir, target_dir):
    ignore_extensions = [".pth", ".pt", ".safetensors", ".index.json"]

    logger.info(f"Start copying non-weighted files and subdirectories...")

    for item in tqdm(os.listdir(source_dir), desc="copy non-weighted file"):
        source_item = os.path.join(source_dir, item)
        target_item = os.path.join(target_dir, item)

        try:
            if os.path.isdir(source_item):
                os.makedirs(target_item, exist_ok=True)
                copy_non_weight_files(source_item, target_item)
            elif os.path.isfile(source_item) and not any(source_item.endswith(ext) for ext in ignore_extensions):
                shutil.copy2(source_item, target_item)
                logger.debug(f"copy file: {source_item} -> {target_item}")
        except Exception as e:
            logger.error(f"copy {source_item} : {str(e)}")

    logger.info(f"Non-weight files and subdirectories copied")


def main():
    parser = argparse.ArgumentParser(description="Model weight format converter")
    parser.add_argument("-s", "--source", required=True, help="Input path (file or directory)")
    parser.add_argument("-o_e", "--output_ext", default=".safetensors", choices=[".pth", ".safetensors"])
    parser.add_argument("-o_n", "--output_name", type=str, default="converted", help="Output file name")
    parser.add_argument("-o", "--output", required=True, help="Output directory path")
    parser.add_argument(
        "-d",
        "--direction",
        choices=[None, "forward", "backward"],
        default=None,
        help="Conversion direction: forward = 'lightx2v' -> 'Diffusers', backward = reverse",
    )
    parser.add_argument(
        "-c",
        "--chunk-size",
        type=int,
        default=100,
        help="Chunk size for saving (only applies to forward), 0 = no chunking",
    )
    parser.add_argument(
        "-t",
        "--model_type",
        choices=["wan_dit", "hunyuan_dit", "wan_t5", "wan_clip", "wan_animate_dit", "qwen_image_dit"],
        default="wan_dit",
        help="Model type",
    )
    parser.add_argument("-b", "--save_by_block", action="store_true")

    # Quantization
    parser.add_argument("--quantized", action="store_true")
    parser.add_argument("--bits", type=int, default=8, choices=[8], help="Quantization bit width")
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to use for quantization (cpu/cuda)",
    )
    parser.add_argument(
        "--linear_dtype",
        type=str,
        choices=["torch.int8", "torch.float8_e4m3fn"],
        help="Data type for linear",
    )
    parser.add_argument(
        "--non_linear_dtype",
        type=str,
        default="torch.float32",
        choices=["torch.bfloat16", "torch.float16"],
        help="Data type for non-linear",
    )
    parser.add_argument("--lora_path", type=str, nargs="*", help="Path(s) to LoRA file(s). Can specify multiple paths separated by spaces.")
    parser.add_argument(
        "--lora_alpha",
        type=float,
        nargs="*",
        default=[1.0],
        help="Alpha for LoRA weight scaling",
    )
    parser.add_argument("--copy_no_weight_files", action="store_true")
    args = parser.parse_args()

    if args.quantized:
        args.linear_dtype = eval(args.linear_dtype)
        args.non_linear_dtype = eval(args.non_linear_dtype)

        model_type_keys_map = {
            "qwen_image_dit": {
                "key_idx": 2,
                "target_keys": ["attn", "img_mlp", "txt_mlp"],
                "ignore_key": None,
            },
            "wan_dit": {
                "key_idx": 2,
                "target_keys": ["self_attn", "cross_attn", "ffn"],
                "ignore_key": ["ca", "audio"],
            },
            "wan_animate_dit": {"key_idx": 2, "target_keys": ["self_attn", "cross_attn", "ffn"], "adapter_keys": ["linear1_kv", "linear1_q", "linear2"], "ignore_key": None},
            "hunyuan_dit": {
                "key_idx": 2,
                "target_keys": [
                    "img_mod",
                    "img_attn_qkv",
                    "img_attn_proj",
                    "img_mlp",
                    "txt_mod",
                    "txt_attn_qkv",
                    "txt_attn_proj",
                    "txt_mlp",
                    "linear1",
                    "linear2",
                    "modulation",
                ],
                "ignore_key": None,
            },
            "wan_t5": {"key_idx": 2, "target_keys": ["attn", "ffn"], "ignore_key": None},
            "wan_clip": {
                "key_idx": 3,
                "target_keys": ["attn", "mlp"],
                "ignore_key": "textual",
            },
        }

        args.target_keys = model_type_keys_map[args.model_type]["target_keys"]
        args.adapter_keys = model_type_keys_map[args.model_type]["adapter_keys"] if "adapter_keys" in model_type_keys_map[args.model_type] else None
        args.key_idx = model_type_keys_map[args.model_type]["key_idx"]
        args.ignore_key = model_type_keys_map[args.model_type]["ignore_key"]

    if os.path.isfile(args.output):
        raise ValueError("Output path must be a directory, not a file")

    logger.info("Starting model weight conversion...")
    convert_weights(args)
    logger.info(f"Conversion completed! Files saved to: {args.output}")


if __name__ == "__main__":
    main()
