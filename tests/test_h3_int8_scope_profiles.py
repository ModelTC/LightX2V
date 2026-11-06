import json
import tempfile
import unittest
from pathlib import Path

import torch
from safetensors.torch import save_file

from lightx2v.models.networks.minimax_h3.int8_metal import linear_component, linear_quant_scheme, load_metal_checkpoint


class ScopeProfileTests(unittest.TestCase):
    def test_profile_membership(self):
        names = {"fc1": "ff.net.0.proj", "fc2": "ff.net.2", "qkv": "attn.to_q", "out": "attn.to_out.0", "adaln": "adaln_proj.linear"}
        expected = {"fc1": {"fc1"}, "fc2": {"fc2"}, "mlp": {"fc1", "fc2"}, "mlp_out": {"fc1", "fc2", "out"}, "mlp_qkv": {"fc1", "fc2", "qkv"}, "dit_proj": {"fc1", "fc2", "qkv", "out"}}
        for scope, selected in expected.items():
            config = {"dit_quant_scheme": "int8-metal", "dit_quant_scope": scope}
            for component, suffix in names.items():
                name = "transformer_blocks.7." + suffix
                self.assertEqual(linear_quant_scheme(config, name), "int8-metal" if component in selected else "Default")
            self.assertEqual(linear_quant_scheme(config, "token_refiner.refiner_blocks.0.ff.net.2"), "Default")
        self.assertEqual(linear_component("transformer_blocks.0.attn.to_k"), "qkv")
        self.assertEqual(linear_component("transformer_blocks.0.attn.to_v"), "qkv")

    def test_loader_honors_nested_index_and_ignores_unselected_payloads(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "original").mkdir()
            (root / "quantized").mkdir()
            fc1 = "transformer_blocks.0.ff.net.0.proj.weight"
            fc2 = "transformer_blocks.0.ff.net.2.weight"
            original = {fc1: torch.ones(128, 128, dtype=torch.bfloat16), fc2: torch.ones(128, 128, dtype=torch.bfloat16), "proj_in.bias": torch.tensor([-0.0, 1.5], dtype=torch.float32)}
            quantized = {fc1: torch.ones(128, 128, dtype=torch.int8), fc1 + "_scale": torch.full((128, 1), 0.125), fc2: torch.zeros(128, 128, dtype=torch.int8), fc2 + "_scale": torch.ones(128, 1)}
            save_file(original, str(root / "original/a.safetensors"))
            save_file(quantized, str(root / "quantized/b.safetensors"))
            mapping = {fc1: "quantized/b.safetensors", fc1 + "_scale": "quantized/b.safetensors", fc2: "original/a.safetensors", "proj_in.bias": "original/a.safetensors"}
            (root / "model.safetensors.index.json").write_text(json.dumps({"metadata": {"lightx2v_quant_scheme": "int8-metal", "lightx2v_quant_scope": "fc1"}, "weight_map": mapping}))
            config = {"dit_quantized_ckpt": str(root), "dit_quant_scope": "fc1"}
            loaded = load_metal_checkpoint(config, "cpu")
            self.assertEqual(set(loaded), set(mapping))
            self.assertEqual(loaded[fc1].dtype, torch.int8)
            self.assertEqual(loaded[fc2].dtype, torch.bfloat16)
            torch.testing.assert_close(loaded[fc2], original[fc2], rtol=0, atol=0)
            self.assertTrue(torch.equal(loaded["proj_in.bias"].view(torch.uint8), original["proj_in.bias"].view(torch.uint8)))
            filtered = load_metal_checkpoint(config, "cpu", remove_keys=(".ff.net.2",))
            self.assertNotIn(fc2, filtered)


if __name__ == "__main__":
    unittest.main()
