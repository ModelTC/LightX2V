import json
import tempfile
import unittest
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from lightx2v.models.networks.minimax_h3.int8_metal import linear_quant_scheme, validate_metal_checkpoint
from tools.convert.convert_h3_mlp_int8_metal import convert, file_hash, quantize_weight


class H3MlpConversionTests(unittest.TestCase):
    def test_scope_selection(self):
        config = {"dit_quant_scheme": "int8-metal", "dit_quant_scope": "mlp"}
        self.assertEqual(linear_quant_scheme(config, "transformer_blocks.0.ff.net.0.proj"), "int8-metal")
        self.assertEqual(linear_quant_scheme(config, "transformer_blocks.0.ff.net.2"), "int8-metal")
        for name in ("transformer_blocks.0.attn.to_q", "transformer_blocks.0.adaln_proj.linear", "token_refiner.refiner_blocks.0.ff.net.2"):
            self.assertEqual(linear_quant_scheme(config, name), "Default")
        with self.assertRaises(ValueError):
            linear_quant_scheme(dict(config, dit_quant_scope="typo"), "probe")
        with self.assertRaises(ValueError):
            linear_quant_scheme(dict(config, dit_quant_scheme="fp8-sgl"), "probe")

    def test_sharded_checkpoint_preserves_other_tensors(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "source"
            source.mkdir()
            torch.manual_seed(42)
            weights = {
                "transformer_blocks.0.ff.net.0.proj.weight": torch.randn(256, 128).bfloat16(),
                "transformer_blocks.0.ff.net.2.weight": torch.zeros(128, 128).bfloat16(),
                "transformer_blocks.0.attn.to_q.weight": torch.randn(128, 128).bfloat16(),
                "proj_in.weight": torch.randn(128, 96),
                "fixture_scalar": torch.tensor(-0.0),
                "transformer_blocks.0.norm1.weight": torch.randn(128).bfloat16(),
            }
            save_file(weights, str(source / "model.safetensors"))
            index = source / "model.safetensors.index.json"
            index.write_text(json.dumps({"metadata": {}, "weight_map": {k: "model.safetensors" for k in weights}}))
            before = file_hash(source / "model.safetensors")
            out = Path(tmp) / "output"
            result = convert(source, out)
            self.assertEqual(file_hash(source / "model.safetensors"), before)
            actual = load_file(str(out / "model.safetensors"))
            for key, value in weights.items():
                if ".ff." in key:
                    self.assertEqual(actual[key].dtype, torch.int8)
                    self.assertEqual(actual[key + "_scale"].dtype, torch.float32)
                    self.assertEqual(actual[key + "_scale"].shape, (value.shape[0], 1))
                    self.assertTrue(torch.all(actual[key + "_scale"] > 0))
                    restored = actual[key].float() * actual[key + "_scale"]
                    self.assertLessEqual((restored - value.float()).abs().max().item(), actual[key + "_scale"].max().item() / 2 + 1e-6)
                else:
                    torch.testing.assert_close(actual[key], value, rtol=0, atol=0)
            self.assertEqual(len(result["quantized_tensors"]), 2)
            validate_metal_checkpoint({"dit_quantized_ckpt": str(out), "dit_quant_scope": "mlp"})
            with self.assertRaises(ValueError):
                validate_metal_checkpoint({"dit_quantized_ckpt": str(out), "dit_quant_scope": "all"})

    def test_converter_rejects_nonfinite(self):
        with self.assertRaisesRegex(ValueError, "Nonfinite"):
            quantize_weight(torch.full((128, 128), float("nan"), dtype=torch.bfloat16), "cpu")


if __name__ == "__main__":
    unittest.main()
