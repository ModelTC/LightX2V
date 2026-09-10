import importlib.util
import os
import unittest
from pathlib import Path

import torch

TRITON_AVAILABLE = importlib.util.find_spec("triton") is not None


@unittest.skipUnless(TRITON_AVAILABLE, "Triton is not installed")
class TestTritonInt8Quantization(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ["TRITON_INTERPRET"] = "1"
        module_path = Path(__file__).parents[1] / "lightx2v" / "common" / "ops" / "mm" / "triton_kernels.py"
        spec = importlib.util.spec_from_file_location("triton_kernels", module_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        cls.kernel = module.int8_quantize_kernel

    def test_rounds_to_nearest_instead_of_truncating(self):
        values = torch.tensor(
            [[1.0, 0.5, -0.5, 0.25, -0.25, 0.1, -0.1, 0.0]],
            dtype=torch.float32,
        )
        quantized = torch.empty_like(values, dtype=torch.int8)
        scales = torch.empty(values.shape[0], dtype=torch.float32)

        self.kernel[(values.shape[0],)](
            values,
            quantized,
            scales,
            values.shape[1],
            BLOCK_SIZE=8,
            num_warps=1,
        )

        expected = torch.tensor(
            [[127, 64, -64, 32, -32, 13, -13, 0]],
            dtype=torch.int8,
        )
        torch.testing.assert_close(quantized, expected, rtol=0, atol=0)
        torch.testing.assert_close(scales, torch.tensor([1.0 / 127.0]))


if __name__ == "__main__":
    unittest.main()
