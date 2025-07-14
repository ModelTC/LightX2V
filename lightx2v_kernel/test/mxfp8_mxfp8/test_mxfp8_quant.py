import unittest
import torch
from lightx2v_kernel.gemm import cutlass_scaled_mxfp8_mm
from lightx2v_kernel.gemm import scaled_fp8_quant
from torch.nn.functional import linear
from lightx2v_kernel.utils import error, benchmark


class TestQuantBF162MXFP8(unittest.TestCase):
    def setUp(self):
        self.tokens = [257, 512, 1024, 13325, 32130, 32760]  # , 75348
        self.channels = [1536, 5120, 8960]  # , 13824
        self.hiddenDims = [1536, 3072, 5120, 8960, 12800]  # , 13824

        self.device = "cuda"
        self.dtype = torch.bfloat16

    def test_accuracy(self):
        """Test the accuracy of quantization from BF16 to MXFP8."""
        for m in self.tokens:
            for k in self.hiddenDims:
                for n in self.channels:
                    with self.subTest(shape=[m, k, n]):
                        activation = torch.randn(m, k, dtype=self.dtype, device=self.device)
                        activation_quant_pred, activation_scale_pred = scaled_fp8_quant(activation)

                        weight = torch.randn(n, k, dtype=self.dtype, device=self.device)
                        weight_quant_pred, weight_scale_pred = scaled_fp8_quant(weight)

                        alpha = torch.tensor(1.0, device=self.device, dtype=torch.float32)
                        mm_pred = cutlass_scaled_mxfp8_mm(activation_quant_pred, weight_quant_pred, activation_scale_pred, weight_scale_pred, alpha=alpha)

                        mm_real = linear(activation, weight, bias=None).to(torch.bfloat16)

                        self.assertTrue(error(mm_pred, mm_real) < 1e-2, f"Accuracy test failed for shape {m, k, n}: Error {error(mm_pred, mm_real)} exceeds threshold.")

    def test_performance(self):
        """Benchmark the performance of Activation quantization from BF16 to MXFP8."""
        for m in self.tokens:
            for k in self.hiddenDims:
                with self.subTest(shape=[m, k]):
                    input = torch.randn(m, k, dtype=self.dtype, device=self.device)
                    shape = [m, k]
                    tflops = 2 * (m * k / 1024**4)
                    benchmark(scaled_fp8_quant, shape, tflops, 100, input)


if __name__ == "__main__":
    unittest.main()
