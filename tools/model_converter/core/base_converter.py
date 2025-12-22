"""
Base converter class that all model converters should inherit from.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from loguru import logger


class BaseConverter(ABC):
    """
    Abstract base class for model converters.
    
    Each model type (e.g., Wan DiT, Qwen DiT) should implement this interface.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize converter with configuration.
        
        Args:
            config: Configuration dictionary parsed from YAML
        """
        self.config = config
        self.model_type = config["source"]["type"]
        self.source_path = Path(config["source"]["path"])
        self.target_format = config["target"]["format"]
        
    @abstractmethod
    def get_key_mapping_rules(
        self, direction: str
    ) -> List[Tuple[str, str]]:
        """
        Get key mapping rules for format conversion.
        
        Args:
            direction: "forward" (LightX2V → Diffusers) or 
                      "backward" (Diffusers → LightX2V)
        
        Returns:
            List of (pattern, replacement) tuples for regex substitution
        """
        pass

    @abstractmethod
    def get_quantization_config(self) -> Dict[str, Any]:
        """
        Get model-specific quantization configuration.
        
        Returns:
            Dictionary with:
                - target_modules: List of module names to quantize
                - key_idx: Index in key split for module identification
                - ignore_keys: Keys to skip during quantization
                - adapter_keys: Special adapter keys (optional)
        """
        pass

    def validate_source(self) -> bool:
        """
        Validate that source model exists and is valid.
        
        Returns:
            True if valid, raises exception otherwise
        """
        if not self.source_path.exists():
            raise FileNotFoundError(f"Source path not found: {self.source_path}")
        
        if self.source_path.is_file():
            valid_exts = [".pt", ".pth", ".safetensors"]
            if self.source_path.suffix not in valid_exts:
                raise ValueError(
                    f"Invalid file format: {self.source_path.suffix}. "
                    f"Expected: {valid_exts}"
                )
        elif self.source_path.is_dir():
            # Check for safetensors or pth files
            has_weights = (
                list(self.source_path.glob("*.safetensors"))
                or list(self.source_path.glob("*.pth"))
                or list(self.source_path.glob("*.pt"))
            )
            if not has_weights:
                raise ValueError(
                    f"No weight files found in directory: {self.source_path}"
                )
        else:
            raise ValueError(f"Invalid source path: {self.source_path}")
        
        return True

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get model metadata and info.
        
        Returns:
            Dictionary with model information
        """
        return {
            "type": self.model_type,
            "source_path": str(self.source_path),
            "target_format": self.target_format,
        }

    def forward_convert(self, weights: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Convert weights from LightX2V format to target format.
        
        Args:
            weights: Source weights dictionary
            
        Returns:
            Converted weights dictionary
        """
        rules = self.get_key_mapping_rules("forward")
        return self._apply_key_mapping(weights, rules)

    def backward_convert(self, weights: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Convert weights from target format to LightX2V format.
        
        Args:
            weights: Source weights dictionary
            
        Returns:
            Converted weights dictionary
        """
        rules = self.get_key_mapping_rules("backward")
        return self._apply_key_mapping(weights, rules)

    def _apply_key_mapping(
        self, 
        weights: Dict[str, torch.Tensor],
        rules: List[Tuple[str, str]]
    ) -> Dict[str, torch.Tensor]:
        """
        Apply key mapping rules to weights dictionary.
        
        Args:
            weights: Source weights
            rules: List of (pattern, replacement) regex rules
            
        Returns:
            Weights with converted keys
        """
        import re
        from tqdm import tqdm
        
        # Pre-compile regex patterns
        compiled_rules = [
            (re.compile(pattern), replacement) 
            for pattern, replacement in rules
        ]
        
        converted = {}
        for key in tqdm(weights.keys(), desc="Converting keys"):
            new_key = key
            for pattern, replacement in compiled_rules:
                new_key = pattern.sub(replacement, new_key)
            converted[new_key] = weights[key]
        
        logger.info(f"Converted {len(weights)} keys")
        return converted

    def prepare_for_quantization(
        self, weights: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Prepare weights for quantization (optional preprocessing).
        
        Args:
            weights: Model weights
            
        Returns:
            Preprocessed weights
        """
        # Default: no preprocessing
        return weights

    def post_process_quantized(
        self, weights: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Post-process quantized weights (optional).
        
        Args:
            weights: Quantized weights
            
        Returns:
            Post-processed weights
        """
        # Default: no post-processing
        return weights

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"model={self.model_type}, "
            f"source={self.source_path.name})"
        )

