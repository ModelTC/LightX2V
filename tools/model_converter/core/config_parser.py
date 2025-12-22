"""
Configuration parser for model conversion.

Supports YAML configuration files with:
- Variable substitution
- Template inheritance
- Validation
"""

import os
import re
from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from loguru import logger


class ConfigParser:
    """Parse and validate conversion configuration."""

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize parser.
        
        Args:
            config_path: Path to YAML config file (optional)
        """
        self.config_path = Path(config_path) if config_path else None
        self.config: Dict[str, Any] = {}

    def load(self, config_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Load configuration from YAML file.
        
        Args:
            config_path: Path to config file
            
        Returns:
            Parsed configuration dictionary
        """
        if config_path:
            self.config_path = Path(config_path)
        
        if not self.config_path or not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
        
        logger.info(f"Loading config from: {self.config_path}")
        
        with open(self.config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)
        
        # Apply variable substitution
        self.config = self._substitute_variables(self.config)
        
        # Validate configuration
        self.validate()
        
        return self.config

    def load_from_dict(self, config_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Load configuration from dictionary.
        
        Args:
            config_dict: Configuration dictionary
            
        Returns:
            Validated configuration
        """
        self.config = config_dict
        self.config = self._substitute_variables(self.config)
        self.validate()
        return self.config

    def _substitute_variables(self, config: Any) -> Any:
        """
        Recursively substitute ${VAR} with environment variables or config values.
        
        Args:
            config: Configuration value (dict, list, str, or other)
            
        Returns:
            Configuration with substituted variables
        """
        if isinstance(config, dict):
            return {k: self._substitute_variables(v) for k, v in config.items()}
        elif isinstance(config, list):
            return [self._substitute_variables(item) for item in config]
        elif isinstance(config, str):
            return self._substitute_string(config)
        else:
            return config

    def _substitute_string(self, value: str) -> str:
        """
        Substitute variables in string.
        
        Supports:
        - ${ENV_VAR}: Environment variable
        - ${config.path.to.value}: Config value reference
        
        Args:
            value: String with potential variables
            
        Returns:
            String with substituted values
        """
        # Pattern: ${var_name} or ${config.path}
        pattern = r"\$\{([^}]+)\}"
        
        def replacer(match):
            var_name = match.group(1)
            
            # Try environment variable first
            if var_name in os.environ:
                return os.environ[var_name]
            
            # Try config reference (e.g., ${output.path})
            if var_name.startswith("config."):
                path = var_name[7:].split(".")  # Remove "config." prefix
                try:
                    result = self.config
                    for key in path:
                        result = result[key]
                    return str(result)
                except (KeyError, TypeError):
                    logger.warning(f"Config reference not found: {var_name}")
                    return match.group(0)  # Return original if not found
            
            # Return original if not found
            logger.warning(f"Variable not found: {var_name}")
            return match.group(0)
        
        return re.sub(pattern, replacer, value)

    def validate(self) -> bool:
        """
        Validate configuration structure and required fields.
        
        Returns:
            True if valid
            
        Raises:
            ValueError: If configuration is invalid
        """
        if not self.config:
            raise ValueError("Empty configuration")
        
        # Check required top-level keys
        required_keys = ["source", "target", "output"]
        for key in required_keys:
            if key not in self.config:
                raise ValueError(f"Missing required key: {key}")
        
        # Validate source
        self._validate_source()
        
        # Validate target
        self._validate_target()
        
        # Validate output
        self._validate_output()
        
        # Validate quantization (if present)
        if "quantization" in self.config:
            self._validate_quantization()
        
        # Validate lora (if present)
        if "lora" in self.config:
            self._validate_lora()
        
        logger.info("Configuration validation passed")
        return True

    def _validate_source(self):
        """Validate source configuration."""
        source = self.config["source"]
        
        if "type" not in source:
            raise ValueError("source.type is required")
        
        if "path" not in source:
            raise ValueError("source.path is required")
        
        # Check path exists
        path = Path(source["path"])
        if not path.exists():
            raise FileNotFoundError(f"Source path not found: {path}")

    def _validate_target(self):
        """Validate target configuration."""
        target = self.config["target"]
        
        if "format" not in target:
            raise ValueError("target.format is required")
        
        valid_formats = ["lightx2v", "diffusers", "comfyui"]
        if target["format"] not in valid_formats:
            raise ValueError(
                f"Invalid target.format: {target['format']}. "
                f"Valid: {valid_formats}"
            )
        
        if "layout" in target:
            valid_layouts = ["single_file", "by_block", "chunked"]
            if target["layout"] not in valid_layouts:
                raise ValueError(
                    f"Invalid target.layout: {target['layout']}. "
                    f"Valid: {valid_layouts}"
                )

    def _validate_output(self):
        """Validate output configuration."""
        output = self.config["output"]
        
        if "path" not in output:
            raise ValueError("output.path is required")
        
        if "name" not in output:
            # Provide default
            self.config["output"]["name"] = "converted_model"

    def _validate_quantization(self):
        """Validate quantization configuration."""
        quant = self.config["quantization"]
        
        if "method" not in quant:
            raise ValueError("quantization.method is required when quantization is enabled")
        
        valid_methods = ["int8", "fp8", "nvfp4", "mxfp4", "mxfp6", "mxfp8"]
        if quant["method"] not in valid_methods:
            raise ValueError(
                f"Invalid quantization.method: {quant['method']}. "
                f"Valid: {valid_methods}"
            )
        
        # Remove backend if present (deprecated)
        if "backend" in quant:
            logger.warning("quantization.backend is deprecated and will be ignored. All platforms use the same quantization method.")
            quant.pop("backend")

    def _validate_lora(self):
        """Validate LoRA configuration."""
        lora = self.config["lora"]
        
        if not lora.get("enabled", False):
            return
        
        if "paths" not in lora:
            raise ValueError("lora.paths is required when lora.enabled=true")
        
        # Check paths exist
        for path in lora["paths"]:
            if not Path(path).exists():
                raise FileNotFoundError(f"LoRA path not found: {path}")

    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Get config value by dot-separated path.
        
        Args:
            key_path: Dot-separated key path (e.g., "source.type")
            default: Default value if not found
            
        Returns:
            Config value or default
        """
        keys = key_path.split(".")
        value = self.config
        
        try:
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            return default

    def set(self, key_path: str, value: Any):
        """
        Set config value by dot-separated path.
        
        Args:
            key_path: Dot-separated key path
            value: Value to set
        """
        keys = key_path.split(".")
        config = self.config
        
        for key in keys[:-1]:
            if key not in config:
                config[key] = {}
            config = config[key]
        
        config[keys[-1]] = value

    def merge(self, other_config: Dict[str, Any]):
        """
        Merge another configuration into this one.
        
        Args:
            other_config: Configuration to merge
        """
        self.config = self._deep_merge(self.config, other_config)

    @staticmethod
    def _deep_merge(base: Dict, update: Dict) -> Dict:
        """
        Deep merge two dictionaries.
        
        Args:
            base: Base dictionary
            update: Dictionary to merge in
            
        Returns:
            Merged dictionary
        """
        result = base.copy()
        
        for key, value in update.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = ConfigParser._deep_merge(result[key], value)
            else:
                result[key] = value
        
        return result

    def save(self, output_path: str):
        """
        Save configuration to file.
        
        Args:
            output_path: Path to save config
        """
        with open(output_path, "w", encoding="utf-8") as f:
            yaml.dump(self.config, f, default_flow_style=False, allow_unicode=True)
        
        logger.info(f"Configuration saved to: {output_path}")

    def __repr__(self) -> str:
        return f"ConfigParser(source={self.get('source.type')}, target={self.get('target.format')})"

