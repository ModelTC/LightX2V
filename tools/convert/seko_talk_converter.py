"""
Model Merge and Multi-Precision Conversion Script

This script orchestrates the full conversion pipeline:
1. Merge R2V + distillation model via LoRA → merged.safetensors (FP32)
2. Convert merged.safetensors to BF16 and FP8 (using converter.py)
3. Convert audio_adapter to BF16 and FP8 (using quant_adapter.py)

Usage:
    python tools/convert/merge_models.py \
        --r2v_model /path/to/model.pt \
        --distill_model /path/to/model_ema.pt \
        --audio_adapter /path/to/audio_adapter.pt \
        --output_dir /data/output \
        --lora_alpha 8.0

Output files:
    - merged.safetensors                  (FP32, R2V + distill merged via LoRA)
    - merged_bf16.safetensors             (BF16, from merged.safetensors)
    - merged_fp8.safetensors              (FP8, from merged.safetensors via converter.py)
    - audio_adapter_model.safetensors     (BF16)
    - audio_adapter_model_fp8.safetensors (FP8, via quant_adapter.py)
"""

import argparse
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


def step1_merge_via_lora(r2v_model_path: Path, distill_model_path: Path, output_dir: Path, lora_alpha: float, temp_dir: Path) -> Path:
    """
    Step 1: Merge R2V + distillation model via LoRA using converter.py.
    Both models in FP32, output merged.safetensors (FP32).
    """
    logger.info("=" * 80)
    logger.info("STEP 1: Merge R2V + Distillation via LoRA (FP32)")
    logger.info("=" * 80)

    temp_dir.mkdir(parents=True, exist_ok=True)

    # Convert R2V to safetensors (keep FP32)
    logger.info("\n[1.1] Converting R2V model to safetensors (FP32)...")
    r2v_dict = load_checkpoint(r2v_model_path)
    r2v_safetensors = temp_dir / "model.safetensors"
    save_file(r2v_dict, str(r2v_safetensors))
    logger.info(f"  Saved: {r2v_safetensors}")

    # Convert distill to safetensors (keep FP32 for LoRA merge)
    logger.info("\n[1.2] Converting distillation model to safetensors (FP32)...")
    distill_dict = load_checkpoint(distill_model_path)
    distill_safetensors = temp_dir / "model_ema.safetensors"
    save_file(distill_dict, str(distill_safetensors))
    logger.info(f"  Saved: {distill_safetensors}")

    # Merge via LoRA using converter.py (FP32 + FP32 → FP32)
    logger.info("\n[1.3] Merging via LoRA (converter.py)...")
    cmd = [
        "python",
        "tools/convert/converter.py",
        "-s",
        str(r2v_safetensors),
        "-o",
        str(output_dir),
        "-o_n",
        "merged",
        "--lora_path",
        str(distill_safetensors),
        "--lora_alpha",
        str(lora_alpha),
        "--single_file",
    ]

    run_command(cmd, "LoRA merge")

    merged_path = output_dir / "merged.safetensors"
    if not merged_path.exists():
        raise FileNotFoundError(f"Merged file not found: {merged_path}")

    logger.info(f"  ✓ Created: {merged_path} (FP32)")
    return merged_path


def step2_convert_merged_to_bf16(merged_path: Path, output_dir: Path):
    """
    Step 2: Convert merged.safetensors (FP32) to BF16.
    """
    logger.info("=" * 80)
    logger.info("STEP 2: Convert merged.safetensors (FP32) → BF16")
    logger.info("=" * 80)

    merged_dict = load_file(str(merged_path))
    merged_bf16 = convert_to_bf16(merged_dict)

    bf16_path = output_dir / "merged_bf16.safetensors"
    save_file(merged_bf16, str(bf16_path))
    logger.info(f"  ✓ Created: {bf16_path}")


def step3_convert_merged_to_fp8(merged_path: Path, output_dir: Path, device: str = "cuda"):
    """
    Step 3: Convert merged.safetensors (FP32) to FP8 using converter.py --quantized.
    """
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
        "--linear_dtype",
        "torch.float8_e4m3fn",
        "--quantized",
        "--device",
        device,
        "--single_file",
    ]

    run_command(cmd, "Merged FP8 conversion")

    fp8_path = output_dir / "merged_fp8.safetensors"
    logger.info(f"  ✓ Created: {fp8_path}")


def step4_convert_audio_adapter_to_bf16(audio_adapter_path: Path, output_dir: Path):
    """
    Step 4: Convert audio adapter to BF16.
    """
    logger.info("=" * 80)
    logger.info("STEP 4: Convert audio adapter → BF16")
    logger.info("=" * 80)

    audio_dict = load_checkpoint(audio_adapter_path)
    audio_bf16 = convert_to_bf16(audio_dict)

    bf16_path = output_dir / "audio_adapter_model.safetensors"
    save_file(audio_bf16, str(bf16_path))
    logger.info(f"  ✓ Created: {bf16_path}")


def step5_convert_audio_adapter_to_fp8(output_dir: Path):
    """
    Step 5: Convert audio adapter BF16 to FP8 using quant_adapter.py.
    """
    logger.info("=" * 80)
    logger.info("STEP 5: Convert audio adapter → FP8")
    logger.info("=" * 80)

    input_path = output_dir / "audio_adapter_model.safetensors"
    output_path = output_dir / "audio_adapter_model_fp8.safetensors"

    cmd = ["python", "tools/convert/quant_adapter.py", "--model_path", str(input_path), "--output_path", str(output_path)]

    run_command(cmd, "Audio adapter FP8 conversion")

    logger.info(f"  ✓ Created: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Merge R2V+distill via LoRA and convert to multiple formats")

    # Inputs
    parser.add_argument("--r2v_model", type=str, required=True, help="Path to R2V model (.pt)")
    parser.add_argument("--distill_model", type=str, required=True, help="Path to distillation model (.pt)")
    parser.add_argument("--audio_adapter", type=str, required=True, help="Path to audio adapter (.pt)")

    # Outputs
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--temp_dir", type=str, default=None, help="Temp directory (default: output_dir/temp)")

    # Settings
    parser.add_argument("--lora_alpha", type=float, default=8.0, help="Alpha for LoRA merge (default: 8.0)")
    parser.add_argument("--device", type=str, default="cuda", help="Device for FP8 quantization (default: cuda)")

    # Options
    parser.add_argument("--skip_merged_fp8", action="store_true", help="Skip merged FP8 conversion")
    parser.add_argument("--skip_audio_fp8", action="store_true", help="Skip audio adapter FP8 conversion")

    args = parser.parse_args()

    # Setup paths
    r2v_path = Path(args.r2v_model)
    distill_path = Path(args.distill_model)
    audio_path = Path(args.audio_adapter)
    output_dir = Path(args.output_dir)
    temp_dir = Path(args.temp_dir) if args.temp_dir else output_dir / "temp"

    # Validate
    for path, name in [(r2v_path, "R2V"), (distill_path, "Distill"), (audio_path, "Audio")]:
        if not path.exists():
            raise FileNotFoundError(f"{name} not found: {path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 80)
    logger.info("MODEL MERGE AND CONVERSION PIPELINE")
    logger.info("=" * 80)
    logger.info(f"R2V model:      {r2v_path}")
    logger.info(f"Distill model:  {distill_path}")
    logger.info(f"Audio adapter:  {audio_path}")
    logger.info(f"Output dir:     {output_dir}")
    logger.info(f"LoRA alpha:     {args.lora_alpha}")
    logger.info(f"Device:         {args.device}")
    logger.info("=" * 80)

    # Execute pipeline
    try:
        # Step 1: Merge R2V (FP32) + Distill (FP32) via LoRA → merged.safetensors (FP32)
        merged_path = step1_merge_via_lora(r2v_path, distill_path, output_dir, args.lora_alpha, temp_dir)

        # Step 2: Convert merged (FP32) → BF16
        step2_convert_merged_to_bf16(merged_path, output_dir)

        # Step 3: Convert merged (FP32) → FP8 (via converter.py)
        if not args.skip_merged_fp8:
            step3_convert_merged_to_fp8(merged_path, output_dir, args.device)

        # Step 4: Convert audio adapter → BF16
        step4_convert_audio_adapter_to_bf16(audio_path, output_dir)

        # Step 5: Convert audio adapter → FP8 (via quant_adapter.py)
        if not args.skip_audio_fp8:
            step5_convert_audio_adapter_to_fp8(output_dir)

    except Exception as e:
        logger.error(f"\n{'=' * 80}")
        logger.error("PIPELINE FAILED")
        logger.error(f"{'=' * 80}")
        logger.error(f"Error: {e}")
        sys.exit(1)

    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("✓ PIPELINE COMPLETED SUCCESSFULLY!")
    logger.info("=" * 80)
    logger.info(f"\nOutput directory: {output_dir}\n")
    logger.info("Generated files:")
    logger.info("  ✓ merged.safetensors                  (FP32, R2V+distill merged)")
    logger.info("  ✓ merged_bf16.safetensors             (BF16, from merged FP32)")
    if not args.skip_merged_fp8:
        logger.info("  ✓ merged_fp8.safetensors              (FP8, from merged FP32)")
    logger.info("  ✓ audio_adapter_model.safetensors     (BF16)")
    if not args.skip_audio_fp8:
        logger.info("  ✓ audio_adapter_model_fp8.safetensors (FP8)")
    logger.info(f"\nTemp files: {temp_dir}")
    logger.info("\nConversion flow:")
    logger.info("  1. R2V (FP32) + Distill (FP32) --LoRA--> merged.safetensors (FP32)")
    logger.info("  2. merged.safetensors (FP32) --> merged_bf16.safetensors")
    logger.info("  3. merged.safetensors (FP32) --> merged_fp8.safetensors")
    logger.info("  4. audio_adapter --> audio_adapter_model.safetensors (BF16)")
    logger.info("  5. audio_adapter_model.safetensors --> audio_adapter_model_fp8.safetensors")


if __name__ == "__main__":
    main()
