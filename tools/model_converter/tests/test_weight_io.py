"""Tests for weight I/O operations."""

import pytest
import torch
import tempfile
from pathlib import Path

from ..core.weight_io import WeightIO


def test_save_and_load_single_file():
    """Test saving and loading as single file."""
    weights = {
        "layer1.weight": torch.randn(128, 256),
        "layer2.weight": torch.randn(256, 512),
        "layer1.bias": torch.randn(128),
    }
    
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir)
        
        # Save
        saved_files = WeightIO.save(
            weights=weights,
            output_path=output_path,
            output_name="test_model",
            save_format="safetensors",
            layout="single_file",
        )
        
        assert len(saved_files) == 1
        assert saved_files[0].exists()
        
        # Load
        loaded_weights = WeightIO.load(saved_files[0])
        
        assert len(loaded_weights) == len(weights)
        for key in weights.keys():
            assert key in loaded_weights
            assert torch.allclose(weights[key], loaded_weights[key])


def test_save_by_block():
    """Test saving by block."""
    weights = {
        "blocks.0.attn.weight": torch.randn(64, 128),
        "blocks.0.ffn.weight": torch.randn(128, 256),
        "blocks.1.attn.weight": torch.randn(64, 128),
        "blocks.1.ffn.weight": torch.randn(128, 256),
        "head.weight": torch.randn(256, 10),
    }
    
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir)
        
        # Save
        saved_files = WeightIO.save(
            weights=weights,
            output_path=output_path,
            output_name="test_model",
            save_format="safetensors",
            layout="by_block",
        )
        
        # Should have multiple files (block_0, block_1, non_block)
        assert len(saved_files) >= 3
        
        # Check block files exist
        assert (output_path / "block_0.safetensors").exists()
        assert (output_path / "block_1.safetensors").exists()
        assert (output_path / "non_block.safetensors").exists()


def test_load_directory():
    """Test loading from directory with multiple files."""
    weights1 = {
        "layer1.weight": torch.randn(64, 128),
    }
    weights2 = {
        "layer2.weight": torch.randn(128, 256),
    }
    
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir)
        
        # Save two separate files
        from safetensors import torch as st
        st.save_file(weights1, str(output_path / "part1.safetensors"))
        st.save_file(weights2, str(output_path / "part2.safetensors"))
        
        # Load directory
        loaded_weights = WeightIO.load(output_path)
        
        assert len(loaded_weights) == 2
        assert "layer1.weight" in loaded_weights
        assert "layer2.weight" in loaded_weights


def test_pytorch_format():
    """Test PyTorch .pth format."""
    weights = {
        "layer.weight": torch.randn(32, 64),
    }
    
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir)
        
        # Save as .pth
        saved_files = WeightIO.save(
            weights=weights,
            output_path=output_path,
            output_name="test_model",
            save_format="pytorch",
        )
        
        assert saved_files[0].suffix == ".pth"
        
        # Load
        loaded_weights = WeightIO.load(saved_files[0])
        assert "layer.weight" in loaded_weights


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

