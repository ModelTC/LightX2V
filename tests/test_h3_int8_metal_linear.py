"""Verify H3's production Linear factory consumes native INT8 checkpoint storage."""

import os
import unittest
from pathlib import Path

import torch


@unittest.skipUnless(torch.backends.mps.is_available() and (Path(os.getenv("LIGHTX2V_METALLIB_DIR") or "/") / "int8_k5376_n7168.metallib").is_file(), "MPS and native H3 metallibs required")
class H3Int8MetalLinearTests(unittest.TestCase):
    def setUp(self):
        from lightx2v.models.networks.minimax_h3.weights.transformer_weights import _linear

        self.make_linear = lambda name, bias=False: _linear({"dit_quant_scheme": "int8-metal"}, name, bias=bias)

    def test_storage_descriptions_preserve_checkpoint_shapes(self):
        from lightx2v_platform.ops.offload.weight_storage import TensorMetadata

        for k, n in ((5376, 28672), (5376, 7168), (7168, 5376), (14336, 5376), (2688, 96768)):
            with self.subTest(k=k, n=n):
                linear = self.make_linear("probe", bias=True)
                metadata = {"probe.weight": TensorMetadata((n, k), torch.int8), "probe.weight_scale": TensorMetadata((n, 1), torch.bfloat16), "probe.bias": TensorMetadata((n,), torch.bfloat16)}
                storage = {x.attr: x for x in linear.describe_storage(metadata).tensors}
                self.assertEqual(storage["weight"].shape, (n, k))
                self.assertEqual(storage["weight"].dtype, torch.int8)
                self.assertTrue(storage["weight"].transpose)
                self.assertEqual(storage["weight_scale"].dtype, torch.float32)

    def test_native_factory_loading_reuse_and_reload(self):
        torch.manual_seed(73)
        m, k, n = 129, 5376, 7168
        with torch.inference_mode():
            qx = torch.randint(-127, 128, (m, k), dtype=torch.int8)
            qx[:, 0] = 127
            x = qx.to(torch.bfloat16).to("mps")
            qweight = torch.randint(-128, 128, (n, k), dtype=torch.int8, device="mps")
            scales = torch.full((n, 1), 1 / 128, dtype=torch.bfloat16, device="mps")
            bias = torch.randn(n, dtype=torch.bfloat16, device="mps")
            columns = [0, 63, 64, 127, 128, n // 2, n - 1]
            reference = ((qx.int() @ qweight[columns].cpu().int().T).float() / 128 + bias[columns].cpu().float()).to(torch.bfloat16)
            linear = self.make_linear("probe", bias=True)
            pointer = qweight.untyped_storage().data_ptr()
            linear.load({"probe.weight": qweight, "probe.weight_scale": scales, "probe.bias": bias})
            self.assertEqual(linear.weight.shape, (k, n))
            self.assertEqual(linear.weight.stride(), (1, k))
            self.assertEqual(linear.weight.untyped_storage().data_ptr(), pointer)
            self.assertIsNone(linear.pin_weight)
            self.assertEqual(linear.weight_scale.dtype, torch.float32)
            first = linear.apply(x)
            torch.testing.assert_close(first[:, columns].cpu(), reference, rtol=0, atol=0)
            saved = first.cpu()
            torch.testing.assert_close(linear.apply(x).cpu(), saved, rtol=0, atol=0)
            self.assertEqual(linear.weight.untyped_storage().data_ptr(), pointer)
            replacement = torch.zeros_like(qweight)
            linear.load({"probe.weight": replacement, "probe.weight_scale": scales, "probe.bias": bias})
            torch.testing.assert_close(linear.apply(x).cpu(), bias.cpu().expand(m, n), rtol=0, atol=0)
            torch.testing.assert_close(first.cpu(), saved, rtol=0, atol=0)

    def test_floating_checkpoint_is_rejected(self):
        linear = self.make_linear("probe")
        with self.assertRaisesRegex(ValueError, "implicit quantization/dequantization"):
            linear.load({"probe.weight": torch.zeros(128, 128, dtype=torch.bfloat16, device="mps")})


if __name__ == "__main__":
    unittest.main()
