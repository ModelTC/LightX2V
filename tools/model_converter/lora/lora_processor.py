"""
LoRA processor - wraps the existing LoRALoader with new architecture integration.

Reuses the battle-tested LoRALoader from tools/convert/lora_loader.py
"""

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from loguru import logger
from safetensors import safe_open

# Import the original LoRALoader
sys.path.append(str(Path(__file__).parent.parent.parent.parent))
try:
    from tools.convert.lora_loader import LoRALoader
except ImportError:
    logger.error("Failed to import LoRALoader. Make sure tools/convert/lora_loader.py exists.")
    raise


class LoRAProcessor:
    """
    LoRA processing wrapper for the new architecture.
    
    Handles:
    - Loading LoRA weights from various formats
    - Merging multiple LoRAs
    - Key mapping for format conversion
    - Integration with quantization pipeline
    """

    def __init__(
        self,
        key_mapping_rules: Optional[List[Tuple[str, str]]] = None,
    ):
        """
        Initialize LoRA processor.
        
        Args:
            key_mapping_rules: Optional key mapping rules for format conversion
        """
        self.key_mapping_rules = key_mapping_rules
        self.loader = LoRALoader(key_mapping_rules=key_mapping_rules)

    def load_lora_weights(self, lora_path: str) -> Dict[str, torch.Tensor]:
        """
        Load LoRA weights from file.
        
        Args:
            lora_path: Path to LoRA file (.safetensors or .pt)
            
        Returns:
            Dictionary of LoRA weights
        """
        lora_path = Path(lora_path)
        
        if not lora_path.exists():
            raise FileNotFoundError(f"LoRA file not found: {lora_path}")
        
        logger.info(f"Loading LoRA from: {lora_path}")
        
        if lora_path.suffix == ".safetensors":
            with safe_open(str(lora_path), framework="pt") as f:
                lora_weights = {k: f.get_tensor(k) for k in f.keys()}
        elif lora_path.suffix in [".pt", ".pth"]:
            lora_weights = torch.load(lora_path, map_location="cpu", weights_only=True)
        else:
            raise ValueError(f"Unsupported LoRA format: {lora_path.suffix}")
        
        logger.info(f"Loaded {len(lora_weights)} LoRA weights")
        return lora_weights

    def apply_lora(
        self,
        model_weights: Dict[str, torch.Tensor],
        lora_path: str,
        alpha: Optional[float] = None,
        strength: float = 1.0,
    ) -> int:
        """
        Apply single LoRA to model weights.
        
        Args:
            model_weights: Model weights (modified in-place)
            lora_path: Path to LoRA file
            alpha: Alpha scaling factor (if None, use LoRA's built-in alpha)
            strength: Additional strength multiplier
            
        Returns:
            Number of successfully applied LoRA weights
        """
        lora_weights = self.load_lora_weights(lora_path)
        
        applied = self.loader.apply_lora(
            weight_dict=model_weights,
            lora_weights=lora_weights,
            alpha=alpha,
            strength=strength,
        )
        
        logger.info(f"Applied LoRA from {Path(lora_path).name}: {applied} weights modified")
        return applied

    def apply_multiple_loras(
        self,
        model_weights: Dict[str, torch.Tensor],
        lora_paths: List[str],
        alphas: Optional[List[float]] = None,
        strengths: Optional[List[float]] = None,
    ) -> List[int]:
        """
        Apply multiple LoRAs sequentially.
        
        Args:
            model_weights: Model weights (modified in-place)
            lora_paths: List of LoRA file paths
            alphas: List of alpha values (one per LoRA)
            strengths: List of strength values (one per LoRA)
            
        Returns:
            List of applied counts for each LoRA
        """
        if not lora_paths:
            return []
        
        # Normalize alphas
        if alphas is None:
            alphas = [None] * len(lora_paths)
        elif len(alphas) == 1 and len(lora_paths) > 1:
            alphas = alphas * len(lora_paths)
        elif len(alphas) != len(lora_paths):
            raise ValueError(
                f"Number of alphas ({len(alphas)}) must match "
                f"number of LoRA paths ({len(lora_paths)}) or be 1"
            )
        
        # Normalize strengths
        if strengths is None:
            strengths = [1.0] * len(lora_paths)
        elif len(strengths) == 1 and len(lora_paths) > 1:
            strengths = strengths * len(lora_paths)
        elif len(strengths) != len(lora_paths):
            raise ValueError(
                f"Number of strengths ({len(strengths)}) must match "
                f"number of LoRA paths ({len(lora_paths)}) or be 1"
            )
        
        results = []
        for idx, lora_path in enumerate(lora_paths):
            logger.info(f"Applying LoRA {idx + 1}/{len(lora_paths)}: {Path(lora_path).name}")
            applied = self.apply_lora(
                model_weights=model_weights,
                lora_path=lora_path,
                alpha=alphas[idx],
                strength=strengths[idx],
            )
            results.append(applied)
        
        logger.info(f"Applied {len(lora_paths)} LoRAs total")
        return results

    def process_from_config(
        self,
        model_weights: Dict[str, torch.Tensor],
        lora_config: Dict[str, Any],
    ) -> Dict[str, torch.Tensor]:
        """
        Process LoRA based on configuration.
        
        Args:
            model_weights: Model weights
            lora_config: LoRA configuration from YAML
            
        Returns:
            Modified model weights
        """
        if not lora_config.get("enabled", False):
            logger.info("LoRA processing disabled")
            return model_weights
        
        lora_paths = lora_config.get("paths", [])
        if not lora_paths:
            logger.warning("LoRA enabled but no paths provided")
            return model_weights
        
        alphas = lora_config.get("alphas")
        strengths = lora_config.get("strengths")
        
        self.apply_multiple_loras(
            model_weights=model_weights,
            lora_paths=lora_paths,
            alphas=alphas,
            strengths=strengths,
        )
        
        return model_weights

    def update_key_mapping_rules(self, rules: List[Tuple[str, str]]):
        """
        Update key mapping rules.
        
        Args:
            rules: New key mapping rules
        """
        self.key_mapping_rules = rules
        self.loader = LoRALoader(key_mapping_rules=rules)
        logger.info("Updated LoRA key mapping rules")

    def get_lora_info(self, lora_path: str) -> Dict[str, Any]:
        """
        Get information about a LoRA file.
        
        Args:
            lora_path: Path to LoRA file
            
        Returns:
            Dictionary with LoRA information
        """
        lora_weights = self.load_lora_weights(lora_path)
        
        lora_alphas = self.loader.extract_lora_alphas(lora_weights)
        lora_pairs = self.loader.extract_lora_pairs(lora_weights)
        lora_diffs = self.loader.extract_lora_diffs(lora_weights)
        
        return {
            "path": lora_path,
            "total_weights": len(lora_weights),
            "num_pairs": len(lora_pairs),
            "num_diffs": len(lora_diffs),
            "has_alphas": len(lora_alphas) > 0,
            "alpha_values": list(lora_alphas.values())[:5],  # Sample
        }

    def __repr__(self) -> str:
        has_rules = self.key_mapping_rules is not None
        return f"LoRAProcessor(has_key_mapping={has_rules})"

