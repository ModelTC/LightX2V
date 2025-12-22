"""Tests for configuration parser."""

import pytest
import yaml
from pathlib import Path
import tempfile

from ..core.config_parser import ConfigParser


def test_load_valid_config():
    """Test loading a valid configuration."""
    config_data = {
        "version": "1.0",
        "source": {
            "type": "wan_dit",
            "path": "/path/to/model",
        },
        "target": {
            "format": "lightx2v",
        },
        "output": {
            "path": "/path/to/output",
            "name": "converted",
        },
    }
    
    parser = ConfigParser()
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(config_data, f)
        config_path = f.name
    
    try:
        # Note: This will fail validation because paths don't exist
        # In real tests, use mock paths or actual test fixtures
        config = parser.load_from_dict(config_data.copy())
        # Make paths exist for validation
        Path(config_data["source"]["path"]).mkdir(parents=True, exist_ok=True)
        Path(config_data["output"]["path"]).mkdir(parents=True, exist_ok=True)
        config = parser.load_from_dict(config_data)
        
        assert config["source"]["type"] == "wan_dit"
        assert config["target"]["format"] == "lightx2v"
    finally:
        Path(config_path).unlink(missing_ok=True)


def test_variable_substitution():
    """Test environment variable substitution."""
    import os
    
    os.environ["TEST_MODEL_PATH"] = "/test/model"
    os.environ["TEST_OUTPUT_PATH"] = "/test/output"
    
    config_data = {
        "version": "1.0",
        "source": {
            "type": "wan_dit",
            "path": "${TEST_MODEL_PATH}",
        },
        "output": {
            "path": "${TEST_OUTPUT_PATH}",
        },
        "target": {"format": "lightx2v"},
    }
    
    parser = ConfigParser()
    config = parser._substitute_variables(config_data)
    
    assert config["source"]["path"] == "/test/model"
    assert config["output"]["path"] == "/test/output"


def test_missing_required_fields():
    """Test validation of missing required fields."""
    config_data = {
        "version": "1.0",
        "source": {
            "type": "wan_dit",
        },
        # Missing path
    }
    
    parser = ConfigParser()
    
    with pytest.raises(ValueError):
        parser.load_from_dict(config_data)


def test_get_set_config_values():
    """Test getting and setting config values."""
    config_data = {
        "version": "1.0",
        "source": {
            "type": "wan_dit",
        },
    }
    
    parser = ConfigParser()
    parser.config = config_data
    
    # Test get
    assert parser.get("source.type") == "wan_dit"
    assert parser.get("nonexistent.key", "default") == "default"
    
    # Test set
    parser.set("source.path", "/new/path")
    assert parser.config["source"]["path"] == "/new/path"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

