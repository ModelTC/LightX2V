"""
LoRA (Low-Rank Adaptation) loader with support for multiple format patterns.

Supported formats:
- Standard: {key}.lora_up.weight and {key}.lora_down.weight
- Diffusers: {key}_lora.up.weight and {key}_lora.down.weight
- Diffusers v2: {key}.lora_B.weight and {key}.lora_A.weight (B=up, A=down)
- Diffusers v3: {key}.lora.up.weight and {key}.lora.down.weight
- Mochi: {key}.lora_B and {key}.lora_A (no .weight suffix)
- Transformers: {key}.lora_linear_layer.up.weight and {key}.lora_linear_layer.down.weight
- Qwen: {key}.lora_B.default.weight and {key}.lora_A.default.weight
"""

import re
from enum import Enum
from typing import Dict, List, Optional, Tuple

import torch
from loguru import logger

from .envs import *
from .quant_utils import dequant_naive_inplace as dequant
from .quant_utils import quant_naive_inplace as quant


class LoRAFormat(Enum):
    """Enum for different LoRA format patterns."""

    STANDARD = "standard"
    DIFFUSERS = "diffusers"
    DIFFUSERS_V2 = "diffusers_v2"
    DIFFUSERS_V3 = "diffusers_v3"
    MOCHI = "mochi"
    TRANSFORMERS = "transformers"
    QWEN = "qwen"


class LoRAPatternDefinition:
    """Defines a single LoRA format pattern and how to extract its components."""

    def __init__(
        self,
        format_name: LoRAFormat,
        up_suffix: str,
        down_suffix: str,
        has_weight_suffix: bool = True,
        mid_suffix: Optional[str] = None,
    ):
        """
        Args:
            format_name: The LoRA format type
            up_suffix: Suffix for the up (B) weight matrix (e.g., ".lora_up.weight")
            down_suffix: Suffix for the down (A) weight matrix (e.g., ".lora_down.weight")
            has_weight_suffix: Whether the format includes .weight suffix
            mid_suffix: Optional suffix for mid weight (only used in standard format)
        """
        self.format_name = format_name
        self.up_suffix = up_suffix
        self.down_suffix = down_suffix
        self.has_weight_suffix = has_weight_suffix
        self.mid_suffix = mid_suffix

    def get_base_key(self, key: str, detected_suffix: str) -> Optional[str]:
        """Extract base key by removing the detected suffix."""
        if key.endswith(detected_suffix):
            return key[: -len(detected_suffix)]
        return None


class LoRAPatternMatcher:
    """Detects and matches LoRA format patterns in state dicts."""

    def __init__(self):
        """Initialize the pattern matcher with all supported formats."""
        self.patterns: Dict[LoRAFormat, LoRAPatternDefinition] = {
            LoRAFormat.STANDARD: LoRAPatternDefinition(
                LoRAFormat.STANDARD,
                up_suffix=".lora_up.weight",
                down_suffix=".lora_down.weight",
                mid_suffix=".lora_mid.weight",
            ),
            LoRAFormat.DIFFUSERS: LoRAPatternDefinition(
                LoRAFormat.DIFFUSERS,
                up_suffix="_lora.up.weight",
                down_suffix="_lora.down.weight",
            ),
            LoRAFormat.DIFFUSERS_V2: LoRAPatternDefinition(
                LoRAFormat.DIFFUSERS_V2,
                up_suffix=".lora_B.weight",
                down_suffix=".lora_A.weight",
            ),
            LoRAFormat.DIFFUSERS_V3: LoRAPatternDefinition(
                LoRAFormat.DIFFUSERS_V3,
                up_suffix=".lora.up.weight",
                down_suffix=".lora.down.weight",
            ),
            LoRAFormat.MOCHI: LoRAPatternDefinition(
                LoRAFormat.MOCHI,
                up_suffix=".lora_B",
                down_suffix=".lora_A",
                has_weight_suffix=False,
            ),
            LoRAFormat.TRANSFORMERS: LoRAPatternDefinition(
                LoRAFormat.TRANSFORMERS,
                up_suffix=".lora_linear_layer.up.weight",
                down_suffix=".lora_linear_layer.down.weight",
            ),
            LoRAFormat.QWEN: LoRAPatternDefinition(
                LoRAFormat.QWEN,
                up_suffix=".lora_B.default.weight",
                down_suffix=".lora_A.default.weight",
            ),
        }

    def detect_format(self, key: str, lora_weights: Dict) -> Optional[Tuple[LoRAFormat, str]]:
        """
        Detect the LoRA format of a given key.

        Args:
            key: The weight key to check
            lora_weights: The full LoRA weights dictionary

        Returns:
            Tuple of (LoRAFormat, detected_suffix) if format detected, None otherwise
        """
        for format_type, pattern in self.patterns.items():
            if key.endswith(pattern.up_suffix):
                return (format_type, pattern.up_suffix)
        return None

    def extract_lora_pair(
        self,
        key: str,
        lora_weights: Dict,
        lora_alphas: Dict,
    ) -> Optional[Dict]:
        """
        Extract a complete LoRA pair (up and down weights) from the state dict.

        Args:
            key: The up weight key
            lora_weights: The full LoRA weights dictionary
            lora_alphas: Dictionary of alpha values by base key

        Returns:
            Dictionary with extracted LoRA information, or None if pair is incomplete
        """
        format_detected = self.detect_format(key, lora_weights)
        if format_detected is None:
            return None

        format_type, up_suffix = format_detected
        pattern = self.patterns[format_type]

        # Extract base key
        base_key = pattern.get_base_key(key, up_suffix)
        if base_key is None:
            return None

        # Check if down weight exists
        down_key = base_key + pattern.down_suffix
        if down_key not in lora_weights:
            return None

        # Check for mid weight (only for standard format)
        mid_key = None
        if pattern.mid_suffix:
            mid_key = base_key + pattern.mid_suffix
            if mid_key not in lora_weights:
                mid_key = None

        # Get alpha value
        alpha = lora_alphas.get(base_key, None)

        return {
            "format": format_type,
            "base_key": base_key,
            "up_key": key,
            "down_key": down_key,
            "mid_key": mid_key,
            "alpha": alpha,
        }


class LoRALoader:
    """Loads and applies LoRA weights to model weights using pattern matching."""

    def __init__(self, key_mapping_rules: Optional[List[Tuple[str, str]]] = None):
        """
        Args:
            key_mapping_rules: Optional list of (pattern, replacement) regex rules for key mapping
        """
        self.pattern_matcher = LoRAPatternMatcher()
        self.key_mapping_rules = key_mapping_rules or []
        self._compile_rules()

    def _compile_rules(self):
        """Pre-compile regex patterns for better performance."""
        self.compiled_rules = [(re.compile(pattern), replacement) for pattern, replacement in self.key_mapping_rules]

    def _apply_key_mapping(self, key: str) -> str:
        """Apply key mapping rules to a key."""
        for pattern, replacement in self.compiled_rules:
            key = pattern.sub(replacement, key)
        return key

    def _get_model_key(
        self,
        lora_key: str,
        base_key: str,
        suffix_to_remove: str,
        suffix_to_add: str = ".weight",
    ) -> Optional[str]:
        """
        Extract the model weight key from LoRA key with proper prefix handling.

        Args:
            lora_key: The original LoRA key
            base_key: The base key after removing LoRA suffix
            suffix_to_remove: The suffix that was removed
            suffix_to_add: The suffix to add for model key

        Returns:
            The model key, or None if extraction fails
        """
        # For Qwen models, keep transformer_blocks prefix
        if base_key.startswith("transformer_blocks.") and len(base_key.split(".")) > 1:
            if base_key.split(".")[1].isdigit():
                # Keep the full path for Qwen models
                model_key = base_key + suffix_to_add
            else:
                # Remove common prefixes for other models
                model_key = self._remove_prefixes(base_key) + suffix_to_add
        else:
            # Remove common prefixes for other models
            model_key = self._remove_prefixes(base_key) + suffix_to_add

        # Apply key mapping rules if provided
        if self.compiled_rules:
            model_key = self._apply_key_mapping(model_key)

        return model_key

    @staticmethod
    def _remove_prefixes(key: str) -> str:
        """Remove common model prefixes from a key."""
        prefixes_to_remove = ["diffusion_model.", "model.", "unet."]
        for prefix in prefixes_to_remove:
            if key.startswith(prefix):
                return key[len(prefix) :]
        return key

    def extract_lora_alphas(self, lora_weights: Dict) -> Dict:
        """Extract LoRA alpha values from the state dict."""
        lora_alphas = {}
        for key in lora_weights.keys():
            if key.endswith(".alpha"):
                base_key = key[:-6]  # Remove .alpha
                lora_alphas[base_key] = lora_weights[key].item()
        return lora_alphas

    def extract_lora_pairs(self, lora_weights: Dict) -> Dict[str, Dict]:
        """
        Extract all LoRA pairs from the state dict, mapping to model keys.

        Args:
            lora_weights: The LoRA state dictionary

        Returns:
            Dictionary mapping model keys to LoRA pair information
        """
        lora_alphas = self.extract_lora_alphas(lora_weights)
        lora_pairs = {}

        for key in lora_weights.keys():
            # Skip alpha parameters
            if key.endswith(".alpha"):
                continue

            # Try to extract LoRA pair
            pair_info = self.pattern_matcher.extract_lora_pair(key, lora_weights, lora_alphas)
            if pair_info is None:
                continue

            # Determine the suffix to remove and add based on format
            format_type = pair_info["format"]
            pattern = self.pattern_matcher.patterns[format_type]

            # Get the model key
            model_key = self._get_model_key(
                pair_info["up_key"],
                pair_info["base_key"],
                pattern.up_suffix,
                ".weight",
            )

            if model_key is None:
                logger.warning(f"Failed to extract model key from LoRA key: {key}")
                continue

            lora_pairs[model_key] = pair_info

        return lora_pairs

    def extract_lora_diffs(self, lora_weights: Dict) -> Dict[str, Dict]:
        """
        Extract diff-style LoRA weights (direct addition, not matrix multiplication).

        Args:
            lora_weights: The LoRA state dictionary

        Returns:
            Dictionary mapping model keys to diff information
        """
        lora_diffs = {}

        # Define diff patterns: (suffix_to_check, suffix_to_remove, suffix_to_add)
        diff_patterns = [
            (".diff", ".diff", ".weight"),
            (".diff_b", ".diff_b", ".bias"),
            (".diff_m", ".diff_m", ".modulation"),
        ]

        for key in lora_weights.keys():
            for check_suffix, remove_suffix, add_suffix in diff_patterns:
                if key.endswith(check_suffix):
                    base_key = key[: -len(remove_suffix)]
                    model_key = self._get_model_key(key, base_key, remove_suffix, add_suffix)

                    if model_key:
                        lora_diffs[model_key] = {
                            "diff_key": key,
                            "type": check_suffix,
                        }
                    break

        return lora_diffs

    def apply_lora(
        self,
        weight_dict: Dict[str, torch.Tensor],
        lora_weights: Dict[str, torch.Tensor],
        alpha: float = None,
        strength: float = 1.0,
    ) -> int:
        """
        Apply LoRA weights to model weights.

        Args:
            weight_dict: The model weights dictionary (will be modified in place)
            lora_weights: The LoRA weights dictionary
            alpha: Global alpha scaling factor
            strength: Additional strength factor for LoRA deltas

        Returns:
            Number of LoRA weights successfully applied
        """
        # Extract LoRA pairs, diffs, and alphas
        lora_pairs = self.extract_lora_pairs(lora_weights)
        lora_diffs = self.extract_lora_diffs(lora_weights)

        applied_count = 0
        used_lora_keys = set()

        # Apply LoRA pairs (matrix multiplication)
        for model_key, pair_info in lora_pairs.items():
            if model_key not in weight_dict:
                logger.debug(f"Model key not found: {model_key}")
                continue

            param = weight_dict[model_key]
            param_dtype = param.dtype
            up_key = pair_info["up_key"]
            down_key = pair_info["down_key"]

            # Track used keys
            used_lora_keys.add(up_key)
            used_lora_keys.add(down_key)
            if pair_info["mid_key"]:
                used_lora_keys.add(pair_info["mid_key"])

            if param_dtype not in [torch.int8, torch.float8_e4m3fn]:
                lora_up = lora_weights[up_key].to(param.device, param_dtype)
                lora_down = lora_weights[down_key].to(param.device, param_dtype)
            else:
                lora_up = lora_weights[up_key].to(param.device)
                lora_down = lora_weights[down_key].to(param.device)

            # Get LoRA-specific alpha if available, otherwise use global alpha
            # Apply LoRA: W' = W + (alpha/rank) * B @ A
            # where B = up (out_features, rank), A = down (rank, in_features)
            if pair_info["alpha"]:
                lora_scale = pair_info["alpha"] / lora_down.shape[0]
            elif alpha is not None:
                lora_scale = alpha / lora_down.shape[0]
            else:
                lora_scale = 1

            if len(lora_down.shape) == 2 and len(lora_up.shape) == 2:
                lora_delta = torch.mm(lora_up, lora_down) * lora_scale
                if strength is not None:
                    lora_delta = lora_delta * float(strength)

                if param_dtype in [torch.int8, torch.float8_e4m3fn]:
                    param.data = dequant(param.data, weight_dict[model_key + "_scale"], GET_DTYPE())
                param.data += lora_delta
                if param_dtype in [torch.int8, torch.float8_e4m3fn]:
                    param.data, weight_dict[model_key + "_scale"] = quant(param.data, param_dtype)
                applied_count += 1
                logger.debug(f"Applied LoRA to {model_key} with lora_scale={lora_scale}")
            else:
                logger.warning(f"Unexpected LoRA shape for {model_key}: down={lora_down.shape}, up={lora_up.shape}")

        # Apply diff weights (direct addition)
        for model_key, diff_info in lora_diffs.items():
            if model_key not in weight_dict:
                logger.debug(f"Model key not found for diff: {model_key}")
                continue

            param = weight_dict[model_key]
            param_dtype = param.dtype
            diff_key = diff_info["diff_key"]

            # Track used keys
            used_lora_keys.add(diff_key)

            if param_dtype not in [torch.int8, torch.float8_e4m3fn]:
                lora_diff = lora_weights[diff_key].to(param.device, param_dtype)
            else:
                lora_diff = lora_weights[diff_key].to(param.device)
            if param_dtype in [torch.int8, torch.float8_e4m3fn]:
                param.data = dequant(param.data, weight_dict[model_key + "_scale"], GET_DTYPE())
            if alpha is not None:
                param.data += lora_diff * alpha * (float(strength) if strength is not None else 1.0)
            else:
                param.data += lora_diff * (float(strength) if strength is not None else 1.0)
            if param_dtype in [torch.int8, torch.float8_e4m3fn]:
                param.data, weight_dict[model_key + "_scale"] = quant(param.data, param_dtype)
            applied_count += 1
            logger.debug(f"Applied LoRA diff to {model_key} (type: {diff_info['type']})")

        # Warn about unused keys
        all_lora_keys = set(k for k in lora_weights.keys() if not k.endswith(".alpha"))
        unused_lora_keys = all_lora_keys - used_lora_keys

        if unused_lora_keys:
            logger.warning(f"Found {len(unused_lora_keys)} unused LoRA weights - this may indicate key mismatch:")
            for key in list(unused_lora_keys)[:10]:  # Show first 10
                logger.warning(f"  Unused: {key}")
            if len(unused_lora_keys) > 10:
                logger.warning(f"  ... and {len(unused_lora_keys) - 10} more")

        logger.info(f"Applied {applied_count} LoRA weight adjustments out of {len(lora_pairs) + len(lora_diffs)} possible")

        if applied_count == 0 and (lora_pairs or lora_diffs):
            logger.error("No LoRA weights were applied! Check for key name mismatches.")
            logger.info("Model weight keys sample: " + str(list(weight_dict.keys())[:5]))
            logger.info("LoRA pairs keys sample: " + str(list(lora_pairs.keys())[:5]))
            logger.info("LoRA diffs keys sample: " + str(list(lora_diffs.keys())[:5]))

        return applied_count
