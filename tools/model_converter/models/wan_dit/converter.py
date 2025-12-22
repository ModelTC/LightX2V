"""Wan DiT model converter implementation."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml
from loguru import logger

from ...core.base_converter import BaseConverter
from ...core.registry import CONVERTER_REGISTRY
from .key_mappings import get_wan_dit_key_mappings


@CONVERTER_REGISTRY.register("wan_dit")
class WanDiTConverter(BaseConverter):
    """Converter for Wan DiT models."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Wan DiT converter.
        
        Args:
            config: Configuration dictionary
        """
        super().__init__(config)
        
        # Load model-specific config
        config_path = Path(__file__).parent / "config.yaml"
        with open(config_path, "r") as f:
            self.model_config = yaml.safe_load(f)
        
        # Determine variant
        self.variant = self._detect_variant()
        logger.info(f"Detected Wan DiT variant: {self.variant}")

    def _detect_variant(self) -> str:
        """Detect Wan DiT variant from config or path."""
        # Check config first
        if "variant" in self.config.get("source", {}):
            return self.config["source"]["variant"]
        
        # Try to detect from path
        path_str = str(self.source_path).lower()
        if "animate" in path_str:
            return "wan_animate_dit"
        
        # Default to standard wan_dit
        return "wan_dit"

    def get_key_mapping_rules(self, direction: str) -> List[Tuple[str, str]]:
        """
        Get key mapping rules for Wan DiT.
        
        Args:
            direction: "forward" or "backward"
            
        Returns:
            List of (pattern, replacement) regex tuples
        """
        return get_wan_dit_key_mappings(direction)

    def get_quantization_config(self) -> Dict[str, Any]:
        """
        Get Wan DiT quantization configuration.
        
        Returns:
            Dictionary with quantization settings
        """
        # Get base config
        quant_config = self.model_config["quantization"].copy()
        
        # Override with variant-specific config if available
        if self.variant in self.model_config["variants"]:
            variant_config = self.model_config["variants"][self.variant]["quantization"]
            quant_config.update(variant_config)
        
        # Override with user config if provided
        if "quantization" in self.config:
            user_quant = self.config["quantization"]
            if "options" in user_quant:
                if "target_modules" in user_quant["options"]:
                    quant_config["target_modules"] = user_quant["options"]["target_modules"]
        
        return quant_config

    def get_comfyui_keys(self) -> Optional[List[str]]:
        """
        Get keys to quantize in ComfyUI mode.
        
        Returns:
            List of key names or None
        """
        # ComfyUI doesn't have specific keys for Wan DiT
        # Rely on target_modules instead
        return None

    def validate_source(self) -> bool:
        """
        Validate Wan DiT source model.
        
        Returns:
            True if valid
        """
        # Call parent validation
        super().validate_source()
        
        # Additional Wan DiT specific checks
        # Could check for required layers, config files, etc.
        
        return True

    def get_model_info(self) -> Dict[str, Any]:
        """Get Wan DiT model information."""
        info = super().get_model_info()
        info.update({
            "variant": self.variant,
            "supported_formats": self.model_config["formats"]["supported"],
        })
        return info

    def __repr__(self) -> str:
        return f"WanDiTConverter(variant={self.variant}, source={self.source_path.name})"

