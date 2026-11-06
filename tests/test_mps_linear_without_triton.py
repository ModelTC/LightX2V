"""MPS model imports and ordinary Linear must not require CUDA Triton kernels."""

import os
import subprocess
import sys
import unittest
from pathlib import Path

import torch


@unittest.skipUnless(torch.backends.mps.is_available(), "MPS device required")
class MpsWithoutTritonTests(unittest.TestCase):
    def test_h3_import_and_unfused_linear(self):
        code = """
import sys
import torch
import torch.nn.functional as F
from lightx2v.models.runners.minimax_h3.minimax_h3_runner import MiniMaxH3Runner
from lightx2v.common.ops.mm.mm_weight import MMWeight
from lightx2v.common.ops.mm.optimized_linear import apply_swiglu_linear, make_gelu_linear
torch.manual_seed(17)
with torch.inference_mode():
    x = torch.randn(129, 64, device='mps', dtype=torch.bfloat16)
    gate = torch.randn_like(x)
    weight = torch.randn(32, 64, device='mps', dtype=torch.bfloat16)
    linear = MMWeight('probe.weight')
    linear.load({'probe.weight': weight})
    torch.testing.assert_close(apply_swiglu_linear(linear, x, gate), linear.apply(x * F.silu(gate)), rtol=0, atol=0)
    torch.testing.assert_close(make_gelu_linear(linear).apply(x), linear.apply(F.gelu(x, approximate='tanh')), rtol=0, atol=0)
assert 'triton' not in sys.modules
"""
        repo = Path(__file__).resolve().parents[1]
        env = dict(os.environ, PLATFORM="mps", DTYPE="BF16", PYTHONPATH=str(repo), PYTHONDONTWRITEBYTECODE="1")
        result = subprocess.run([sys.executable, "-B", "-c", code], env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=60)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
