"""Exactness regression for DreamZero's torch-compatible fused RMSNorm."""

import unittest
from unittest.mock import patch

import torch

from lightx2v.common.ops.norm.torch_compatible_rms_norm import can_use_torch_rms_norm_5120, torch_rms_norm_5120

SUPPORTED = torch.cuda.is_available() and torch.version.hip is None and torch.__version__.split(".")[:2] == ["2", "10"]


class TestDreamZeroRMSNorm(unittest.TestCase):
    @unittest.skipUnless(SUPPORTED, "PyTorch 2.10 CUDA reduction contract")
    def test_exact_torch_rms_norm(self):
        for rows in (16, 17, 880, 1785):
            for diff_shape in (None, (), (1,), (5120,)):
                with self.subTest(rows=rows, diff_shape=diff_shape):
                    generator = torch.Generator(device="cuda").manual_seed(1140)
                    x = torch.randn((rows, 5120), device="cuda", dtype=torch.bfloat16, generator=generator) * 13
                    weight = torch.randn((5120,), device="cuda", dtype=x.dtype, generator=generator)
                    delta = None if diff_shape is None else torch.randn(diff_shape, device="cuda", dtype=x.dtype, generator=generator)
                    # Independent eager reference, including both BF16 rounding boundaries.
                    normalized = (x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + 1e-6)).to(x.dtype)
                    reference = normalized * (weight if delta is None else weight + delta)
                    assert can_use_torch_rms_norm_5120(x, weight, delta)
                    actual = torch_rms_norm_5120(x, weight, delta, 1e-6)
                    assert torch.equal(reference, actual)

    @unittest.skipUnless(SUPPORTED, "PyTorch 2.10 CUDA reduction contract")
    def test_dispatch_guard(self):
        x = torch.zeros((17, 5120), device="cuda", dtype=torch.bfloat16)
        weight = torch.ones((5120,), device="cuda", dtype=x.dtype)
        assert can_use_torch_rms_norm_5120(x, weight, None)
        assert not can_use_torch_rms_norm_5120(x[:15], weight, None)
        assert not can_use_torch_rms_norm_5120(x[:, ::2], weight[::2], None)
        assert not can_use_torch_rms_norm_5120(x.float(), weight.float(), None)
        assert not can_use_torch_rms_norm_5120(x.cpu(), weight.cpu(), None)
        assert not can_use_torch_rms_norm_5120(x, None, None)
        assert not can_use_torch_rms_norm_5120(x, weight, torch.zeros(2, device=x.device, dtype=x.dtype))
        assert not can_use_torch_rms_norm_5120(x, weight, weight.float())
        with patch.object(torch, "__version__", "2.9.0"):
            assert not can_use_torch_rms_norm_5120(x, weight, None)

    def test_cpu_helper_falls_back(self):
        from lightx2v.common.ops.norm.rms_norm_weight import RMSWeight
        from lightx2v.models.networks.wan.infer.dreamzero.transformer_infer import DreamZeroTransformerInfer

        transformer = object.__new__(DreamZeroTransformerInfer)
        transformer.fuse_rms_norm = True
        norm = object.__new__(RMSWeight)
        norm.infer_dtype = norm.sensitive_layer_dtype = torch.bfloat16
        norm.eps = 1e-6
        norm.weight = torch.ones(32, dtype=torch.bfloat16)
        norm.weight_diff = torch.zeros(32, dtype=torch.bfloat16)
        x = torch.randn(17, 32).bfloat16()
        assert torch.equal(transformer._apply_attention_rms_norm(norm, x), norm.apply(x))


if __name__ == "__main__":
    unittest.main()
