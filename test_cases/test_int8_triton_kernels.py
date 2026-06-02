import importlib.util
import unittest
from pathlib import Path

import torch

_DTYPE_TOLERANCES = {
    torch.float32: {"rtol": 1e-5, "atol": 1e-3},
    torch.float16: {"rtol": 1e-3, "atol": 5e-1},
    torch.bfloat16: {"rtol": 1e-2, "atol": 2.0},
}


def _load_triton_kernels():
    module_path = Path(__file__).resolve().parents[1] / "lightx2v" / "common" / "ops" / "mm" / "triton_kernels.py"
    spec = importlib.util.spec_from_file_location("triton_kernels_under_test", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _ones_case(m, n, k):
    a = torch.ones((m, k), device="cuda", dtype=torch.int8)
    b = torch.ones((n, k), device="cuda", dtype=torch.int8)
    a_scales = torch.ones(m, device="cuda", dtype=torch.float32)
    b_scales = torch.ones(n, device="cuda", dtype=torch.float32)
    return a, b, a_scales, b_scales


def _random_case(m, n, k):
    generator = torch.Generator()
    generator.manual_seed(896)
    a = torch.randint(-2, 3, (m, k), generator=generator, dtype=torch.int8).cuda()
    b = torch.randint(-2, 3, (n, k), generator=generator, dtype=torch.int8).cuda()
    a_scales = ((torch.arange(m, device="cuda", dtype=torch.float32) % 5) + 1) / 8
    b_scales = ((torch.arange(n, device="cuda", dtype=torch.float32) % 7) + 1) / 7
    return a, b, a_scales, b_scales


def _reference(a, b, a_scales, b_scales, bias=None):
    out = (a.float() @ b.float().t()) * a_scales[:, None] * b_scales[None, :]
    if bias is not None:
        out = out + bias[None, :]
    return out


def _assert_close_to_reference(out, expected, dtype, rtol=0, atol=0):
    torch.cuda.synchronize()
    assert out.dtype == dtype
    assert out.shape == expected.shape
    torch.testing.assert_close(out.float(), expected, rtol=rtol, atol=atol)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for Triton INT8 GEMM tests")
class TestInt8TritonKernels(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.triton_kernels = _load_triton_kernels()

    def test_int8_gemm_default_output_dtype_is_float16_and_does_not_overflow(self):
        a, b, a_scales, b_scales = _ones_case(128, 128, 128)
        expected = _reference(a, b, a_scales, b_scales)

        out = self.triton_kernels.int8_gemm_triton(a, b, a_scales, b_scales)

        _assert_close_to_reference(out, expected, torch.float16)

    def test_int8_gemm_bias_default_output_dtype_is_float16_and_does_not_overflow(self):
        a, b, a_scales, b_scales = _ones_case(128, 128, 128)
        bias = torch.zeros(128, device="cuda", dtype=torch.float32)
        expected = _reference(a, b, a_scales, b_scales, bias)

        out = self.triton_kernels.int8_gemm_bias_triton(a, b, bias, a_scales, b_scales)

        _assert_close_to_reference(out, expected, torch.float16)

    def test_int8_gemm_explicit_output_dtype_is_preserved(self):
        for dtype in [torch.bfloat16, torch.float16]:
            for with_bias in [False, True]:
                with self.subTest(dtype=dtype, with_bias=with_bias):
                    a, b, a_scales, b_scales = _ones_case(128, 128, 128)
                    bias = torch.zeros(128, device="cuda", dtype=torch.float32) if with_bias else None
                    expected = _reference(a, b, a_scales, b_scales, bias)

                    if with_bias:
                        out = self.triton_kernels.int8_gemm_bias_triton(a, b, bias, a_scales, b_scales, output_dtype=dtype)
                    else:
                        out = self.triton_kernels.int8_gemm_triton(a, b, a_scales, b_scales, output_dtype=dtype)

                    _assert_close_to_reference(out, expected, dtype)

    def test_int8_gemm_handles_k_tail_and_block_k_64_multiples(self):
        for k in [64, 127, 129, 192]:
            for with_bias in [False, True]:
                with self.subTest(k=k, with_bias=with_bias):
                    a, b, a_scales, b_scales = _ones_case(128, 128, k)
                    bias = None
                    if with_bias:
                        bias = (torch.arange(128, device="cuda", dtype=torch.float32) % 5) - 2
                    expected = _reference(a, b, a_scales, b_scales, bias)

                    if with_bias:
                        out = self.triton_kernels.int8_gemm_bias_triton(
                            a,
                            b,
                            bias,
                            a_scales,
                            b_scales,
                            output_dtype=torch.bfloat16,
                        )
                    else:
                        out = self.triton_kernels.int8_gemm_triton(
                            a,
                            b,
                            a_scales,
                            b_scales,
                            output_dtype=torch.bfloat16,
                        )

                    _assert_close_to_reference(out, expected, torch.bfloat16)

    def test_int8_gemm_random_scales_bias_and_tail_shapes(self):
        for dtype in [torch.float32, torch.float16, torch.bfloat16]:
            for with_bias in [False, True]:
                with self.subTest(dtype=dtype, with_bias=with_bias):
                    m, n, k = 130, 129, 127
                    a, b, a_scales, b_scales = _random_case(m, n, k)
                    bias = None
                    if with_bias:
                        bias = ((torch.arange(n, device="cuda", dtype=torch.float32) % 11) - 5) / 6
                    expected = _reference(a, b, a_scales, b_scales, bias)
                    tolerances = _DTYPE_TOLERANCES[dtype]

                    if with_bias:
                        out = self.triton_kernels.int8_gemm_bias_triton(
                            a,
                            b,
                            bias,
                            a_scales,
                            b_scales,
                            output_dtype=dtype,
                        )
                    else:
                        out = self.triton_kernels.int8_gemm_triton(
                            a,
                            b,
                            a_scales,
                            b_scales,
                            output_dtype=dtype,
                        )

                    _assert_close_to_reference(out, expected, dtype, **tolerances)

    def test_int8_gemm_handles_non_tile_mn(self):
        a, b, a_scales, b_scales = _ones_case(130, 129, 128)
        expected = _reference(a, b, a_scales, b_scales)

        out = self.triton_kernels.int8_gemm_triton(a, b, a_scales, b_scales, output_dtype=torch.bfloat16)

        _assert_close_to_reference(out, expected, torch.bfloat16)

    def test_int8_gemm_bias_uses_valid_scale_indices_for_non_tile_mn(self):
        m, n, k = 130, 129, 128
        a = torch.ones((m, k), device="cuda", dtype=torch.int8)
        b = torch.ones((n, k), device="cuda", dtype=torch.int8)
        a_scales = (torch.arange(m, device="cuda", dtype=torch.float32) % 5) + 1
        b_scales = (torch.arange(n, device="cuda", dtype=torch.float32) % 7) + 1
        bias = (torch.arange(n, device="cuda", dtype=torch.float32) % 3) + 1
        expected = _reference(a, b, a_scales, b_scales, bias)

        out = self.triton_kernels.int8_gemm_bias_triton(a, b, bias, a_scales, b_scales, output_dtype=torch.float32)

        _assert_close_to_reference(out, expected, torch.float32)


if __name__ == "__main__":
    unittest.main()
