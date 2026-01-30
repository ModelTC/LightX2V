"""
Model Merge and Multi-Precision Conversion Script

This script supports four conversion modes:
1. 'both' (default): Convert both R2V model and audio adapter (R2V + distill via LoRA)
2. 'r2v': Only convert R2V model (single model → FP32 → BF16/FP8, no distill merge)
3. 'audio': Only convert audio adapter
4. 'merged': Only merge R2V + distill (no precision conversion)

Pipeline:
- R2V (both/merged): R2V + distill via LoRA → merged.safetensors (FP32) → BF16/FP8
- R2V (r2v only): R2V model → merged.safetensors (FP32) → BF16/FP8
- Audio adapter: (optional: + LoRA) → audio_adapter.pt → BF16 → FP8

Usage Examples:
    # Convert both (default)
    python tools/convert/seko_talk_converter.py \
        --r2v_model /path/to/model.pt \
        --distill_model /path/to/model_ema.pt \
        --audio_adapter /path/to/audio_adapter.pt \
        --output_dir /data/output

    # Only convert R2V model (no distill merge)
    python tools/convert/seko_talk_converter.py \
        --mode r2v \
        --r2v_model /path/to/model.pt \
        --output_dir /data/output

    # Only convert audio adapter
    python tools/convert/seko_talk_converter.py \
        --mode audio \
        --audio_adapter /path/to/audio_adapter.pt \
        --output_dir /data/output

    # Convert audio adapter with LoRA merge
    python tools/convert/seko_talk_converter.py \
        --mode audio \
        --audio_adapter /path/to/audio_adapter.pt \
        --audio_lora /path/to/audio_lora.pt \
        --output_dir /data/output

    # Only merge R2V + distill (no precision conversion)
    python tools/convert/seko_talk_converter.py \
        --mode merged \
        --r2v_model /path/to/model.safetensors \
        --distill_model /path/to/model_ema.safetensors \
        --output_dir /data/output

    # Convert diffuser format to x2v format first, then merge
    python tools/convert/seko_talk_converter.py \
        --mode merged \
        --r2v_model /path/to/diffuser_model.pt \
        --distill_model /path/to/diffuser_model_ema.pt \
        --backward_convert \
        --output_dir /data/output

Output files (depending on mode):
    - merged.safetensors                  (FP32, R2V + distill merged)
    - merged_bf16.safetensors             (BF16)
    - merged_fp8.safetensors              (FP8)
    - audio_adapter_merged.safetensors    (FP32, audio + lora merged, optional)
    - audio_adapter_model.safetensors     (BF16)
    - audio_adapter_model_fp8.safetensors (FP8)
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import torch
from loguru import logger
from safetensors.torch import load_file, save_file
from tqdm import tqdm


def run_command(cmd: list, description: str):
    """Run a subprocess command and handle errors."""
    logger.info(f"\n{description}")
    logger.info("Command: " + " \\\n  ".join(cmd))

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        logger.error(f"{description} FAILED!")
        logger.error(f"STDOUT:\n{result.stdout}")
        logger.error(f"STDERR:\n{result.stderr}")
        raise RuntimeError(f"{description} failed")

    logger.info(f"✓ {description} completed!")
    return result


def load_checkpoint(ckpt_path: Path) -> dict:
    """Load checkpoint from .pt or .safetensors file."""
    logger.info(f"Loading: {ckpt_path.name}")

    if ckpt_path.suffix in [".pt", ".pth"]:
        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    elif ckpt_path.suffix == ".safetensors":
        checkpoint = load_file(str(ckpt_path))
    else:
        raise ValueError(f"Unsupported format: {ckpt_path.suffix}")

    logger.info(f"  Loaded {len(checkpoint)} keys")
    return checkpoint


def convert_to_bf16(state_dict: dict) -> dict:
    """Convert all tensors to bfloat16."""
    logger.info("Converting to BF16...")
    bf16_dict = {}
    for key, tensor in tqdm(state_dict.items(), desc="BF16 conversion"):
        bf16_dict[key] = tensor.to(torch.bfloat16)
    return bf16_dict


def convert_to_bf16_file(input_path: Path, output_path: Path, description: str = ""):
    """Convert checkpoint file to BF16 safetensors."""
    if description:
        logger.info("=" * 80)
        logger.info(f"{description}")
        logger.info("=" * 80)

    checkpoint = load_checkpoint(input_path) if input_path.suffix in [".pt", ".pth"] else load_file(str(input_path))
    bf16_dict = convert_to_bf16(checkpoint)
    save_file(bf16_dict, str(output_path))
    logger.info(f"  ✓ Created: {output_path}")


def checkpoint_to_safetensors(ckpt_path: Path, output_path: Path, description: str = ""):
    """Convert checkpoint to safetensors format."""
    if description:
        logger.info(f"\n{description}")
    checkpoint = load_checkpoint(ckpt_path)
    save_file(checkpoint, str(output_path))
    logger.info(f"  Saved: {output_path}")


def merge_via_lora(base_path: Path, lora_path: Path, output_dir: Path, output_name: str, lora_alpha: float, temp_dir: Path, description: str = "") -> Path:
    """通用LoRA合并函数：合并base + lora via converter.py"""
    if description:
        logger.info("=" * 80)
        logger.info(f"{description}")
        logger.info("=" * 80)

    temp_dir.mkdir(parents=True, exist_ok=True)

    base_safetensors = temp_dir / f"{base_path.stem}.safetensors"
    checkpoint_to_safetensors(base_path, base_safetensors, f"Converting base model to safetensors (FP32)...")

    lora_safetensors = temp_dir / f"{lora_path.stem}.safetensors"
    checkpoint_to_safetensors(lora_path, lora_safetensors, f"Converting LoRA to safetensors (FP32)...")

    logger.info("\nMerging via LoRA (converter.py)...")
    cmd = [
        "python",
        "tools/convert/converter.py",
        "-s",
        str(base_safetensors),
        "-o",
        str(output_dir),
        "-o_n",
        output_name,
        "--lora_path",
        str(lora_safetensors),
        "--lora_alpha",
        str(lora_alpha),
        "--single_file",
    ]
    run_command(cmd, "LoRA merge")

    merged_path = output_dir / f"{output_name}.safetensors"
    if not merged_path.exists():
        raise FileNotFoundError(f"Merged file not found: {merged_path}")

    logger.info(f"  ✓ Created: {merged_path} (FP32)")
    return merged_path


def backward_convert_model(input_path: Path, output_dir: Path, output_name: str) -> Path:
    """
    Convert diffuser format model to x2v format using converter.py -d backward.
    Returns path to converted safetensors file.
    """
    logger.info("=" * 80)
    logger.info(f"BACKWARD CONVERSION: {input_path.name} → x2v format")
    logger.info("=" * 80)

    cmd = [
        "python",
        "tools/convert/converter.py",
        "-d",
        "backward",
        "--single_file",
        "-s",
        str(input_path),
        "-o",
        str(output_dir),
        "-o_n",
        output_name,
    ]

    run_command(cmd, f"Backward conversion: {input_path.name}")

    converted_path = output_dir / f"{output_name}.safetensors"
    if not converted_path.exists():
        raise FileNotFoundError(f"Converted file not found: {converted_path}")

    logger.info(f"  ✓ Created: {converted_path}")
    return converted_path


def step1_merge_via_lora(r2v_model_path: Path, distill_model_path: Path, output_dir: Path, lora_alpha: float, temp_dir: Path) -> Path:
    """Merge R2V + distillation model via LoRA."""
    return merge_via_lora(r2v_model_path, distill_model_path, output_dir, "merged", lora_alpha, temp_dir, "STEP 1: Merge R2V + Distillation via LoRA (FP32)")


def step2_convert_merged_to_bf16(merged_path: Path, output_dir: Path):
    """Convert merged.safetensors (FP32) to BF16."""
    convert_to_bf16_file(merged_path, output_dir / "merged_bf16.safetensors", "STEP 2: Convert merged.safetensors (FP32) → BF16")


def step3_convert_merged_to_fp8(merged_path: Path, output_dir: Path, device: str = "cuda"):
    """Convert merged.safetensors (FP32) to FP8 using converter.py --quantized."""
    logger.info("=" * 80)
    logger.info("STEP 3: Convert merged.safetensors (FP32) → FP8")
    logger.info("=" * 80)

    cmd = [
        "python",
        "tools/convert/converter.py",
        "-s",
        str(merged_path),
        "-o",
        str(output_dir),
        "-o_n",
        "merged_fp8",
        "--linear_type",
        "fp8",
        "--quantized",
        "--device",
        device,
        "--single_file",
    ]
    run_command(cmd, "Merged FP8 conversion")
    logger.info(f"  ✓ Created: {output_dir / 'merged_fp8.safetensors'}")


def step_audio_merge_lora(audio_adapter_path: Path, audio_lora_path: Path, output_dir: Path, lora_alpha: float, temp_dir: Path) -> Path:
    """Merge audio adapter + LoRA using converter.py."""
    return merge_via_lora(audio_adapter_path, audio_lora_path, output_dir, "audio_adapter_merged", lora_alpha, temp_dir, "AUDIO STEP 1: Merge Audio Adapter + LoRA (FP32)")


def step4_convert_audio_adapter_to_bf16(audio_adapter_path: Path, output_dir: Path):
    """Convert audio adapter to BF16."""
    convert_to_bf16_file(audio_adapter_path, output_dir / "audio_adapter_model.safetensors", "AUDIO STEP 2: Convert audio adapter → BF16")


def step5_convert_audio_adapter_to_fp8(output_dir: Path):
    """
    Step 5: Convert audio adapter BF16 to FP8 using quant_adapter.py.
    """
    logger.info("=" * 80)
    logger.info("AUDIO STEP 3: Convert audio adapter → FP8")
    logger.info("=" * 80)

    input_path = output_dir / "audio_adapter_model.safetensors"
    output_path = output_dir / "audio_adapter_model_fp8.safetensors"

    cmd = ["python", "tools/convert/quant_adapter.py", "--model_path", str(input_path), "--output_path", str(output_path)]

    run_command(cmd, "Audio adapter FP8 conversion")

    logger.info(f"  ✓ Created: {output_path}")


def validate_args(args):
    """验证参数并返回路径对象"""
    if args.mode in ["both", "merged"]:
        if not args.r2v_model or not args.distill_model:
            raise ValueError("--r2v_model and --distill_model are required for 'both' and 'merged' modes")
    if args.mode == "r2v":
        if not args.r2v_model:
            raise ValueError("--r2v_model is required for 'r2v' mode")

    if args.mode in ["both", "audio"]:
        if not args.audio_adapter:
            raise ValueError("--audio_adapter is required for 'both' and 'audio' modes")

    output_dir = Path(args.output_dir)
    temp_dir = Path(args.temp_dir) if args.temp_dir else output_dir / "temp"

    paths = {
        "r2v": Path(args.r2v_model) if args.r2v_model else None,
        "distill": Path(args.distill_model) if args.distill_model else None,
        "audio": Path(args.audio_adapter) if args.audio_adapter else None,
        "audio_lora": Path(args.audio_lora) if args.audio_lora else None,
    }

    for name, path in paths.items():
        if path and not path.exists():
            raise FileNotFoundError(f"{name.replace('_', ' ').title()} not found: {path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir, temp_dir, paths


def log_pipeline_info(args, output_dir, paths):
    """输出pipeline信息"""
    logger.info("=" * 80)
    logger.info("MODEL CONVERSION PIPELINE")
    logger.info("=" * 80)
    logger.info(f"Mode:           {args.mode}")
    if paths["r2v"]:
        logger.info(f"R2V model:      {paths['r2v']}")
    if paths["distill"]:
        logger.info(f"Distill model:  {paths['distill']}")
    if paths["audio"]:
        logger.info(f"Audio adapter:  {paths['audio']}")
    if paths["audio_lora"]:
        logger.info(f"Audio LoRA:     {paths['audio_lora']}")
    logger.info(f"Output dir:     {output_dir}")
    logger.info(f"Backward convert: {args.backward_convert}")
    if args.mode in ["both", "r2v", "merged"]:
        logger.info(f"LoRA alpha:     {args.lora_alpha}")
    if paths["audio_lora"]:
        logger.info(f"Audio LoRA alpha: {args.audio_lora_alpha}")
    logger.info(f"Device:         {args.device}")
    logger.info("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Merge R2V+distill via LoRA and convert to multiple formats")

    parser.add_argument(
        "--mode",
        type=str,
        choices=["both", "r2v", "audio", "merged"],
        default="both",
        help="Conversion mode: 'both' (default), 'r2v' (only R2V model), 'audio' (only audio adapter), or 'merged' (only merge R2V+distill)",
    )
    parser.add_argument("--r2v_model", type=str, help="Path to R2V model (.pt) [required for 'both', 'r2v', 'merged' modes]")
    parser.add_argument("--distill_model", type=str, help="Path to distillation model (.pt) [required for 'both' and 'merged' modes]")
    parser.add_argument("--audio_adapter", type=str, help="Path to audio adapter (.pt) [required for 'both' and 'audio' modes]")
    parser.add_argument("--audio_lora", type=str, help="Path to audio LoRA (.pt/.safetensors) [optional, for merging with audio adapter]")
    parser.add_argument("--audio_lora_alpha", type=float, default=8.0, help="Alpha for audio LoRA merge (default: 8.0)")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--temp_dir", type=str, default=None, help="Temp directory (default: output_dir/temp)")
    parser.add_argument("--lora_alpha", type=float, default=8.0, help="Alpha for LoRA merge (default: 8.0)")
    parser.add_argument("--device", type=str, default="cuda", help="Device for FP8 quantization (default: cuda)")
    parser.add_argument("--backward_convert", action="store_true", help="Convert diffuser format to x2v format first (default: False)")
    parser.add_argument("--skip_merged_fp8", action="store_true", help="Skip merged FP8 conversion")
    parser.add_argument("--skip_audio_fp8", action="store_true", help="Skip audio adapter FP8 conversion")

    args = parser.parse_args()

    try:
        output_dir, temp_dir, paths = validate_args(args)
    except (ValueError, FileNotFoundError) as e:
        parser.error(str(e))

    log_pipeline_info(args, output_dir, paths)

    try:
        # Backward conversion if needed
        if args.backward_convert:
            if args.mode in ["both", "r2v", "merged"]:
                logger.info("\n>>> BACKWARD CONVERSION (Diffuser → x2v format)")
                if paths["r2v"]:
                    paths["r2v"] = backward_convert_model(paths["r2v"], output_dir, "model")
                if paths["distill"]:
                    paths["distill"] = backward_convert_model(paths["distill"], output_dir, "model_ema")
            if args.mode in ["both", "audio"] and paths["audio"]:
                logger.info("\n>>> BACKWARD CONVERSION: Audio Adapter")
                paths["audio"] = backward_convert_model(paths["audio"], output_dir, "audio_adapter")

        # Process R2V model
        if args.mode in ["both", "r2v", "merged"]:
            logger.info("\n>>> Processing R2V MODEL")
            if args.mode == "r2v":
                merged_path = output_dir / "merged.safetensors"
                if paths["r2v"].suffix in [".pt", ".pth"]:
                    checkpoint_to_safetensors(paths["r2v"], merged_path, "STEP 1: Convert R2V model to safetensors (FP32)")
                else:
                    shutil.copy(paths["r2v"], merged_path)
                    logger.info(f"  ✓ Copied R2V model to: {merged_path}")
            else:
                merged_path = step1_merge_via_lora(paths["r2v"], paths["distill"], output_dir, args.lora_alpha, temp_dir)

            if args.mode != "merged":
                step2_convert_merged_to_bf16(merged_path, output_dir)
                if not args.skip_merged_fp8:
                    step3_convert_merged_to_fp8(merged_path, output_dir, args.device)

        # Process audio adapter
        if args.mode in ["both", "audio"]:
            logger.info("\n>>> Processing AUDIO ADAPTER")
            audio_source = paths["audio"]
            if paths["audio_lora"]:
                audio_source = step_audio_merge_lora(paths["audio"], paths["audio_lora"], output_dir, args.audio_lora_alpha, temp_dir)

            step4_convert_audio_adapter_to_bf16(audio_source, output_dir)
            if not args.skip_audio_fp8:
                step5_convert_audio_adapter_to_fp8(output_dir)

    except Exception as e:
        logger.error(f"\n{'=' * 80}\nPIPELINE FAILED\n{'=' * 80}\nError: {e}")
        sys.exit(1)

    # Summary
    logger.info(f"\n{'=' * 80}\n✓ PIPELINE COMPLETED SUCCESSFULLY!\n{'=' * 80}")
    logger.info(f"\nMode: {args.mode}\nOutput directory: {output_dir}\nGenerated files:")

    if args.mode in ["both", "r2v", "merged"]:
        logger.info("  ✓ merged.safetensors                  (FP32, R2V+distill merged)")
        if args.mode != "merged":
            logger.info("  ✓ merged_bf16.safetensors             (BF16)")
            if not args.skip_merged_fp8:
                logger.info("  ✓ merged_fp8.safetensors              (FP8)")

    if args.mode in ["both", "audio"]:
        if paths["audio_lora"]:
            logger.info("  ✓ audio_adapter_merged.safetensors    (FP32, audio+lora merged)")
        logger.info("  ✓ audio_adapter_model.safetensors     (BF16)")
        if not args.skip_audio_fp8:
            logger.info("  ✓ audio_adapter_model_fp8.safetensors (FP8)")

    if args.mode in ["both", "r2v", "merged"]:
        logger.info(f"\nTemp files: {temp_dir}")


if __name__ == "__main__":
    main()
