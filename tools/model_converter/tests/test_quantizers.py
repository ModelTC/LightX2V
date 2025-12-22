"""Tests for quantizers."""

import pytest
import torch

from ..quantization import INT8Quantizer, BaseQuantizer


def test_int8_quantizer_initialization():
    """Test INT8 quantizer initialization."""
    quantizer = INT8Quantizer(
        backend="cpu",
        target_modules=["attn", "ffn"],
        key_idx=2,
    )
    
    assert quantizer.backend == "cpu"
    assert "attn" in quantizer.target_modules
    assert quantizer.key_idx == 2


def test_int8_weight_quantization():
    """Test INT8 weight quantization."""
    quantizer = INT8Quantizer(backend="cpu")
    
    # Create a test weight tensor
    weight = torch.randn(128, 256)
    
    # Quantize
    w_q, scales, extra = quantizer.quantize_weight(weight, per_channel=True)
    
    # Verify output types
    assert w_q.dtype == torch.int8
    assert scales.dtype == torch.float32
    assert w_q.shape == weight.shape
    assert scales.shape[0] == weight.shape[0]  # Per-channel scales


def test_quantization_range():
    """Test that quantized values are in valid range."""
    quantizer = INT8Quantizer(backend="cpu")
    
    weight = torch.randn(64, 128)
    w_q, scales, extra = quantizer.quantize_weight(weight)
    
    # INT8 range: [-128, 127]
    assert w_q.min() >= -128
    assert w_q.max() <= 127


def test_quantization_reconstruction():
    """Test that dequantized weights are close to original."""
    quantizer = INT8Quantizer(backend="cpu")
    
    weight = torch.randn(32, 64)
    w_q, scales, extra = quantizer.quantize_weight(weight, per_channel=True)
    
    # Dequantize
    w_dequant = w_q.float() * scales
    
    # Check reconstruction error
    error = (weight - w_dequant).abs().mean()
    assert error < 0.1  # Reasonable reconstruction


def test_weight_validation():
    """Test weight validation."""
    quantizer = INT8Quantizer(backend="cpu")
    
    # Test invalid dimensions
    weight_1d = torch.randn(128)
    with pytest.raises(ValueError):
        quantizer.validate_weight(weight_1d)
    
    # Test NaN values
    weight_nan = torch.randn(32, 64)
    weight_nan[0, 0] = float('nan')
    with pytest.raises(ValueError):
        quantizer.validate_weight(weight_nan)


def test_should_quantize():
    """Test module selection for quantization."""
    quantizer = INT8Quantizer(
        backend="cpu",
        target_modules=["attn", "ffn"],
        key_idx=2,
    )
    
    # Should quantize
    assert quantizer._should_quantize(
        "blocks.0.attn.weight",
        adapter_keys=None,
        comfyui_mode=False,
        comfyui_keys=None,
    )
    
    assert quantizer._should_quantize(
        "blocks.5.ffn.weight",
        adapter_keys=None,
        comfyui_mode=False,
        comfyui_keys=None,
    )
    
    # Should not quantize
    assert not quantizer._should_quantize(
        "blocks.0.norm.weight",
        adapter_keys=None,
        comfyui_mode=False,
        comfyui_keys=None,
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

