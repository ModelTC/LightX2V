import os
import unittest
from tempfile import TemporaryDirectory
from unittest.mock import patch

import torch
from lightx2v_kernel_apple.int8_gemm import int8_matmul
from lightx2v_kernel_apple.int8_gemm.gemm import _kernels


class MetalLibraryTests(unittest.TestCase):
    def test_missing_directory(self):
        for directory in (None, ""):
            with self.subTest(directory=directory), self.assertRaisesRegex(RuntimeError, "LIGHTX2V_METALLIB_DIR"):
                _kernels(4096, 4096, directory)

    def test_missing_shape(self):
        with TemporaryDirectory() as directory, self.assertRaisesRegex(FileNotFoundError, "int8_k4096_n4096.metallib"):
            _kernels(4096, 4096, directory)


@unittest.skipUnless(torch.backends.mps.is_available() and os.environ.get("LIGHTX2V_METALLIB_DIR"), "MPS and LIGHTX2V_METALLIB_DIR are required")
class Int8MatmulTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)
        compiler = patch("torch.mps.compile_shader", side_effect=AssertionError("Unexpected source compilation"))
        compiler.start()
        self.addCleanup(compiler.stop)

    def matrices(self, m, k, n):
        # Power-of-two row scales make the online INT8 values exact, so the
        # result can be checked against an independent CPU INT32 reference.
        qx = torch.randint(-127, 128, (m, k), dtype=torch.int8)
        if m:
            qx[:, 0] = 127
        xs = 2.0 ** (torch.arange(m) % 4 - 2)
        x = (qx.float() * xs[:, None]).to(torch.bfloat16).to("mps")
        qw = torch.randint(-128, 128, (n, k), dtype=torch.int8)
        ws = torch.full((n,), 1 / 128)
        bias = torch.randn(n, dtype=torch.bfloat16)
        columns = sorted({0, n // 2, n - 1})
        acc = qx.to(torch.int32) @ qw[columns].to(torch.int32).T
        expected = (acc.float() * xs[:, None] * ws[columns] + bias[columns].float()).to(torch.bfloat16)
        return x, qw.to("mps").T, ws.to("mps"), bias.to("mps"), columns, expected

    def test_dynamic_shapes_and_bias(self):
        for m, k, n in ((1, 4096, 4096), (19, 4096, 4096), (129, 4096, 4096), (128, 4096, 12288), (129, 12288, 4096)):
            with self.subTest(m=m, k=k, n=n):
                x, weight, scale, bias, columns, expected = self.matrices(m, k, n)
                output = int8_matmul(x, weight, scale, bias)
                self.assertEqual(output.dtype, torch.bfloat16)
                self.assertEqual(output.shape, (m, n))
                torch.testing.assert_close(output[:, columns].cpu(), expected, rtol=0, atol=0)

    def test_independent_outputs_and_weight_updates(self):
        x, weight, scale, bias, _, _ = self.matrices(129, 4096, 4096)
        first = int8_matmul(x, weight, scale, bias)
        saved = first.cpu()
        weight.zero_()
        second = int8_matmul(x, weight, scale, bias)
        self.assertNotEqual(first.data_ptr(), second.data_ptr())
        torch.testing.assert_close(second.cpu(), bias.cpu().expand(129, 4096), rtol=0, atol=0)
        torch.testing.assert_close(first.cpu(), saved, rtol=0, atol=0)

    def test_no_bias_strided_input_and_zero_rows(self):
        x, weight, scale, _, _, _ = self.matrices(129, 4096, 4096)
        reference = int8_matmul(x, weight, scale).cpu()
        storage = torch.empty(129, 8192, device="mps", dtype=torch.bfloat16)
        storage[:, ::2] = x
        torch.testing.assert_close(int8_matmul(storage[:, ::2], weight, scale).cpu(), reference, rtol=0, atol=0)
        torch.testing.assert_close(int8_matmul(torch.zeros_like(x), weight, scale).cpu(), torch.zeros_like(reference), rtol=0, atol=0)
        self.assertEqual(int8_matmul(x[:0], weight, scale).shape, (0, 4096))


if __name__ == "__main__":
    unittest.main()
