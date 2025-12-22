"""
Main CLI entry point for model conversion.

Supports both YAML config and command-line arguments.
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, Optional

from loguru import logger

from ..core.config_parser import ConfigParser
from ..core.registry import CONVERTER_REGISTRY, QUANTIZER_REGISTRY
from ..core.weight_io import WeightIO
from ..lora.lora_processor import LoRAProcessor


class ModelConverter:
    """Main model conversion orchestrator."""

    def __init__(self, config: Dict):
        """
        Initialize converter.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.parser = ConfigParser()
        self.parser.config = config
        
        # Initialize components
        self._init_converter()
        self._init_quantizer()
        self._init_lora_processor()

    def _init_converter(self):
        """Initialize model converter."""
        model_type = self.config["source"]["type"]
        
        if model_type not in CONVERTER_REGISTRY:
            raise ValueError(
                f"Unsupported model type: {model_type}. "
                f"Available: {CONVERTER_REGISTRY.list()}"
            )
        
        converter_class = CONVERTER_REGISTRY.get(model_type)
        self.converter = converter_class(self.config)
        logger.info(f"Initialized converter: {self.converter}")

    def _init_quantizer(self):
        """Initialize quantizer if needed."""
        self.quantizer = None
        
        if "quantization" not in self.config:
            return
        
        quant_config = self.config["quantization"]
        method = quant_config.get("method")
        
        if not method:
            return
        
        if method not in QUANTIZER_REGISTRY:
            raise ValueError(
                f"Unsupported quantization method: {method}. "
                f"Available: {QUANTIZER_REGISTRY.list()}"
            )
        
        # Get model-specific quantization config
        model_quant_config = self.converter.get_quantization_config()
        
        quantizer_class = QUANTIZER_REGISTRY.get(method)
        self.quantizer = quantizer_class(
            target_modules=model_quant_config.get("target_modules", []),
            key_idx=model_quant_config.get("key_idx", 2),
            ignore_keys=model_quant_config.get("ignore_keys", []),
        )
        logger.info(f"Initialized quantizer: {self.quantizer}")

    def _init_lora_processor(self):
        """Initialize LoRA processor if needed."""
        self.lora_processor = None
        
        if "lora" not in self.config or not self.config["lora"].get("enabled"):
            return
        
        # Get key mapping rules if format conversion is involved
        key_mapping_rules = None
        if self.config.get("target", {}).get("format") != "lightx2v":
            # Converting format, apply key mapping to LoRA
            direction = "forward"  # Assuming forward conversion
            key_mapping_rules = self.converter.get_key_mapping_rules(direction)
        
        self.lora_processor = LoRAProcessor(key_mapping_rules=key_mapping_rules)
        logger.info("Initialized LoRA processor")

    def convert(self) -> Path:
        """
        Execute the full conversion pipeline.
        
        Returns:
            Path to output directory
        """
        logger.info("=" * 80)
        logger.info("STARTING MODEL CONVERSION")
        logger.info("=" * 80)
        logger.info(f"Source: {self.config['source']['path']}")
        logger.info(f"Target: {self.config['target']['format']}")
        logger.info(f"Output: {self.config['output']['path']}")
        logger.info("=" * 80)
        
        # Step 1: Load weights
        logger.info("\n[Step 1/5] Loading model weights...")
        device = self.config.get("performance", {}).get("device", "cpu")
        weights = WeightIO.load(self.config["source"]["path"], device=device)
        logger.info(f"Loaded {len(weights)} weights")
        
        # Step 2: Apply format conversion (if needed)
        target_format = self.config["target"]["format"]
        if target_format != "lightx2v":
            logger.info(f"\n[Step 2/5] Converting to {target_format} format...")
            weights = self.converter.forward_convert(weights)
        else:
            logger.info("\n[Step 2/5] Skipping format conversion (target is LightX2V)")
        
        # Step 3: Apply LoRA (if configured)
        if self.lora_processor:
            logger.info("\n[Step 3/5] Applying LoRA...")
            weights = self.lora_processor.process_from_config(
                weights, self.config["lora"]
            )
        else:
            logger.info("\n[Step 3/5] Skipping LoRA (not configured)")
        
        # Step 4: Apply quantization (if configured)
        if self.quantizer:
            logger.info("\n[Step 4/5] Applying quantization...")
            
            # Get model-specific config for quantization
            model_quant_config = self.converter.get_quantization_config()
            
            # Apply quantization
            weights = self.quantizer.quantize_model(
                weights=weights,
                adapter_keys=model_quant_config.get("adapter_keys"),
                comfyui_mode=self.config["target"]["format"] == "comfyui",
                comfyui_keys=self.converter.get_comfyui_keys() if hasattr(self.converter, "get_comfyui_keys") else None,
            )
        else:
            logger.info("\n[Step 4/5] Skipping quantization (not configured)")
        
        # Step 5: Save weights
        logger.info("\n[Step 5/5] Saving converted weights...")
        output_path = Path(self.config["output"]["path"])
        output_name = self.config["output"]["name"]
        layout = self.config["target"].get("layout", "single_file")
        save_format = "safetensors"  # Default to safetensors
        
        saved_files = WeightIO.save(
            weights=weights,
            output_path=output_path,
            output_name=output_name,
            save_format=save_format,
            layout=layout,
        )
        
        logger.info("=" * 80)
        logger.info("✓ CONVERSION COMPLETED SUCCESSFULLY!")
        logger.info("=" * 80)
        logger.info(f"Output directory: {output_path}")
        logger.info(f"Saved {len(saved_files)} file(s)")
        logger.info("=" * 80)
        
        return output_path


def convert_model(config: Dict) -> Path:
    """
    Convert model with given configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Path to output directory
    """
    converter = ModelConverter(config)
    return converter.convert()


def build_config_from_args(args: argparse.Namespace) -> Dict:
    """
    Build configuration from command-line arguments.
    
    Args:
        args: Parsed command-line arguments
        
    Returns:
        Configuration dictionary
    """
    config = {
        "version": "1.0",
        "source": {
            "type": args.model,
            "path": args.source,
            "format": "auto",
        },
        "target": {
            "format": args.target_format,
            "layout": args.layout,
        },
        "output": {
            "path": args.output,
            "name": args.output_name or "converted_model",
        },
    }
    
    # Add quantization if specified
    if args.quantization:
        config["quantization"] = {
            "method": args.quantization,
            "options": {},
        }
    
    # Add LoRA if specified
    if args.lora_paths:
        config["lora"] = {
            "enabled": True,
            "paths": args.lora_paths,
            "strengths": args.lora_strengths or [1.0],
            "alphas": args.lora_alphas or None,
        }
    
    # Add performance config
    config["performance"] = {
        "device": args.device,
        "parallel": True,
    }
    
    return config


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="LightX2V Model Converter - Convert models between formats and apply quantization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Using config file
  python -m model_converter convert --config configs/wan_dit_int8_dcu.yaml
  
  # Quick mode from command line
  python -m model_converter convert \\
    --model wan_dit \\
    --source /path/to/model \\
    --target-format lightx2v \\
    --quantization int8 \\
    --output /path/to/output
  
  # With LoRA
  python -m model_converter convert \\
    --model wan_dit \\
    --source /path/to/model \\
    --lora-paths /path/to/lora1.safetensors /path/to/lora2.safetensors \\
    --lora-strengths 1.0 0.8 \\
    --output /path/to/output
        """,
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Convert command
    convert_parser = subparsers.add_parser("convert", help="Convert model")
    
    # Config file OR command-line args
    config_group = convert_parser.add_mutually_exclusive_group(required=True)
    config_group.add_argument(
        "--config",
        type=str,
        help="Path to YAML configuration file",
    )
    config_group.add_argument(
        "--model",
        type=str,
        choices=["wan_dit", "hunyuan_dit", "qwen_image_dit", "wan_t5", "wan_clip"],
        help="Model type (for quick mode)",
    )
    
    # Source/target (required for quick mode)
    convert_parser.add_argument(
        "--source",
        type=str,
        help="Source model path (file or directory)",
    )
    convert_parser.add_argument(
        "--target-format",
        type=str,
        choices=["lightx2v", "diffusers", "comfyui"],
        default="lightx2v",
        help="Target format (default: lightx2v)",
    )
    convert_parser.add_argument(
        "--output",
        type=str,
        help="Output directory path",
    )
    convert_parser.add_argument(
        "--output-name",
        type=str,
        help="Output file name (default: converted_model)",
    )
    
    # Quantization
    convert_parser.add_argument(
        "--quantization",
        type=str,
        choices=["int8", "fp8", "nvfp4", "mxfp4", "mxfp6", "mxfp8"],
        help="Quantization method",
    )
    
    # LoRA
    convert_parser.add_argument(
        "--lora-paths",
        type=str,
        nargs="+",
        help="Path(s) to LoRA file(s)",
    )
    convert_parser.add_argument(
        "--lora-strengths",
        type=float,
        nargs="+",
        help="LoRA strength values (default: 1.0 for each)",
    )
    convert_parser.add_argument(
        "--lora-alphas",
        type=float,
        nargs="+",
        help="LoRA alpha values (optional)",
    )
    
    # Output format
    convert_parser.add_argument(
        "--layout",
        type=str,
        choices=["single_file", "by_block", "chunked"],
        default="single_file",
        help="Output layout (default: single_file)",
    )
    
    # Performance
    convert_parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device for quantization (default: cuda:0)",
    )
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    if args.command == "convert":
        # Load config
        if args.config:
            # Config file mode
            config_parser = ConfigParser()
            config = config_parser.load(args.config)
        else:
            # Quick mode from command-line args
            if not args.source or not args.output:
                logger.error("--source and --output are required in quick mode")
                sys.exit(1)
            
            config = build_config_from_args(args)
        
        # Validate config
        try:
            parser_obj = ConfigParser()
            parser_obj.load_from_dict(config)
        except Exception as e:
            logger.error(f"Configuration validation failed: {e}")
            sys.exit(1)
        
        # Convert
        try:
            output_path = convert_model(config)
            logger.info(f"\n✓ Conversion successful! Output: {output_path}")
            sys.exit(0)
        except Exception as e:
            logger.error(f"\n✗ Conversion failed: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)


if __name__ == "__main__":
    main()

