import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import torch
from lightx2v_kernel_apple.int8_gemm import int8_matmul
from lightx2v_kernel_apple.int8_gemm.compile_metallib import parse_shape
from lightx2v_kernel_apple.int8_gemm.gemm import _kernels


class MetalLibraryTests(unittest.TestCase):
    def test_compile_shape_contract(self):
        import argparse

        for shape in ("5376x28672", "5376x7168", "2688x96768", "128x128"):
            self.assertEqual(parse_shape(shape), tuple(map(int, shape.split("x"))))
        for shape in ("0x128", "129x128", "128x129", "invalid"):
            with self.subTest(shape=shape), self.assertRaises(argparse.ArgumentTypeError):
                parse_shape(shape)

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

    def require_libraries(self, *shapes):
        directory = Path(os.environ["LIGHTX2V_METALLIB_DIR"])
        missing = [f"int8_k{k}_n{n}.metallib" for k, n in shapes if not (directory / f"int8_k{k}_n{n}.metallib").is_file()]
        if missing:
            self.skipTest("Compile the additional test shapes first: " + ", ".join(missing))

    def matrices(self, m, k, n):
        # Power-of-two row scales make the online INT8 values exact, so the
        # result can be checked against an independent CPU INT32 reference.
        qx = torch.randint(-127, 128, (m, k), dtype=torch.int8)
        if m:
            qx[:, 0] = 127
        xs = 2.0 ** (torch.arange(m) % 4 - 2)
        x = (qx.float() * xs[:, None]).to(torch.bfloat16).to("mps")
        qw = torch.randint(-128, 128, (n, k), dtype=torch.int8)
        # Different exact scales across channels also verify scale indexing.
        ws = 2.0 ** (torch.arange(n) % 4 - 7)
        bias = torch.randn(n, dtype=torch.bfloat16)
        columns = list(range(n)) if n <= 256 else sorted({0, 31, 63, 64, 127, 128, 255, 256, n // 2, n - 1})
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

    def test_short_k_and_quantizer_tail(self):
        self.require_libraries((128, 128), (256, 128), (384, 256), (768, 128), (1280, 256))
        for m, k, n in ((1, 128, 128), (127, 256, 128), (128, 384, 256), (129, 768, 128), (257, 1280, 256)):
            with self.subTest(m=m, k=k, n=n):
                x, weight, scale, bias, columns, expected = self.matrices(m, k, n)
                torch.testing.assert_close(int8_matmul(x, weight, scale, bias)[:, columns].cpu(), expected, rtol=0, atol=0)

    def test_h3_native_shapes(self):
        self.require_libraries((5376, 28672), (5376, 7168), (7168, 5376), (14336, 5376), (2688, 96768))
        for m, k, n in ((1, 5376, 28672), (129, 5376, 28672), (128, 5376, 7168), (257, 7168, 5376), (129, 14336, 5376), (19, 2688, 96768)):
            with self.subTest(m=m, k=k, n=n):
                x, weight, scale, bias, columns, expected = self.matrices(m, k, n)
                pointer = weight.untyped_storage().data_ptr()
                output = int8_matmul(x, weight, scale, bias)
                torch.testing.assert_close(output[:, columns].cpu(), expected, rtol=0, atol=0)
                self.assertEqual(weight.shape, (k, n))
                self.assertEqual(weight.untyped_storage().data_ptr(), pointer)

    def test_h3_final_column_amax_and_zero_rows(self):
        self.require_libraries((5376, 7168))
        x, weight, scale, _, columns, _ = self.matrices(129, 5376, 7168)
        x.zero_()
        x[-1, -1] = 127
        expected = torch.zeros(129, len(columns), dtype=torch.bfloat16)
        expected[-1] = (127 * weight[-1, columns].cpu().float() * scale[columns].cpu()).to(torch.bfloat16)
        torch.testing.assert_close(int8_matmul(x, weight, scale)[:, columns].cpu(), expected, rtol=0, atol=0)

    def test_h3_strides_offsets_and_independent_outputs(self):
        self.require_libraries((5376, 7168))
        x, weight, scale, bias, columns, expected = self.matrices(128, 5376, 7168)
        strided = torch.empty(128, 5376 * 2, device="mps", dtype=torch.bfloat16)
        strided[:, ::2] = x
        offset = torch.empty(x.numel() + 1, device="mps", dtype=torch.bfloat16)[1:].view_as(x)
        offset.copy_(x)
        first = int8_matmul(strided[:, ::2], weight, scale, bias)
        saved = first.cpu()
        torch.testing.assert_close(first[:, columns].cpu(), expected, rtol=0, atol=0)
        torch.testing.assert_close(int8_matmul(offset, weight, scale, bias).cpu(), saved, rtol=0, atol=0)
        weight.zero_()
        second = int8_matmul(x, weight, scale, bias)
        self.assertNotEqual(first.data_ptr(), second.data_ptr())
        torch.testing.assert_close(first.cpu(), saved, rtol=0, atol=0)
        torch.testing.assert_close(second.cpu(), bias.cpu().expand_as(second), rtol=0, atol=0)
        self.assertEqual(int8_matmul(x[:0], weight, scale).shape, (0, 7168))

    def test_native_dimension_and_layout_rejections(self):
        for k, n in ((129, 128), (128, 129), (0, 128)):
            with self.subTest(k=k, n=n), self.assertRaisesRegex(ValueError, "divisible by 128"):
                int8_matmul(torch.empty(1, k, device="mps", dtype=torch.bfloat16), torch.empty(n, k, device="mps", dtype=torch.int8).T, torch.ones(n, device="mps", dtype=torch.float32))
        with self.assertRaisesRegex(ValueError, "strides"):
            int8_matmul(torch.empty(1, 128, device="mps", dtype=torch.bfloat16), torch.empty(128, 256, device="mps", dtype=torch.int8), torch.ones(256, device="mps", dtype=torch.float32))

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
