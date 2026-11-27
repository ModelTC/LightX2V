"""CPU value and gradient reference for the independent fold implementation."""

import unittest

import torch
import torch.nn.functional as F

import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location("sf3d_ascend", Path(__file__).resolve().parents[1] / "ascend.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
fold2d = module.fold2d


class FoldEquivalence(unittest.TestCase):
    def test_values_and_gradients(self):
        torch.manual_seed(42)
        for unbatched in (False, True):
            for kernel, stride, padding, dilation in [
                ((3, 3), 1, 0, 1),
                ((2, 3), (2, 1), (1, 2), (2, 1)),
                ((1, 1), 2, 0, 1),
            ]:
                kwargs = dict(output_size=(13, 17), kernel_size=kernel, stride=stride, padding=padding, dilation=dilation)
                x = torch.randn(2, 3, 13, 17, dtype=torch.float64)
                a = F.unfold(x, kernel, stride=stride, padding=padding, dilation=dilation)
                if unbatched:
                    a = a[0]
                a = a.detach().requires_grad_(True)
                b = a.detach().clone().requires_grad_(True)
                ref, actual = F.fold(a, **kwargs), fold2d(b, **kwargs)
                torch.testing.assert_close(ref, actual, atol=1e-12, rtol=1e-12)
                gradient = torch.randn_like(ref)
                ref.backward(gradient)
                actual.backward(gradient)
                torch.testing.assert_close(a.grad, b.grad, atol=1e-12, rtol=1e-12)

    def test_invalid_geometry(self):
        with self.assertRaises(ValueError):
            fold2d(torch.zeros(1, 27, 9), (13, 17), (3, 3))


if __name__ == "__main__":
    unittest.main()
