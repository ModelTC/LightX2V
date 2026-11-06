import json
import os
import tempfile
import unittest
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from lightx2v.models.networks.minimax_h3.int8_metal import validate_metal_checkpoint
from tools.convert.convert_h3_int8_metal import convert, file_hash


class FullCheckpointTests(unittest.TestCase):
    def test_standalone_shards_select_all_six_projections(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "source"
            source.mkdir()
            torch.manual_seed(17)
            suffixes = ("ff.net.0.proj", "ff.net.2", "attn.to_q", "attn.to_k", "attn.to_v", "attn.to_out.0")
            tensors = {f"transformer_blocks.0.{suffix}.weight": torch.randn(128, 128).bfloat16() for suffix in suffixes}
            tensors.update({"transformer_blocks.0.adaln_proj.linear.weight": torch.randn(128, 128).bfloat16(), "proj_out.bias": torch.tensor([-0.0, 1.25]), "norm.weight": torch.ones(128)})
            mapping = {}
            for number, keys in enumerate((list(tensors)[:3], list(tensors)[3:])):
                file = f"shard{number}.safetensors"
                save_file({key: tensors[key] for key in keys}, str(source / file))
                mapping.update(dict.fromkeys(keys, file))
            (source / "model.safetensors.index.json").write_text(json.dumps({"weight_map": mapping}))
            original_hashes = {file: file_hash(source / file) for file in set(mapping.values())}
            output = Path(tmp) / "output"
            manifest = convert(source, output)
            index = validate_metal_checkpoint({"dit_quantized_ckpt": str(output), "dit_quant_scope": "dit_proj"})
            self.assertEqual(len(manifest["quantized_tensors"]), 6)
            self.assertEqual(index["metadata"]["quantized_linear_count"], 6)
            actual = {}
            for file in set(mapping.values()):
                self.assertFalse((output / file).is_symlink())
                self.assertEqual(file_hash(source / file), original_hashes[file])
                actual.update(load_file(str(output / file)))
            for name, value in tensors.items():
                if name in manifest["quantized_tensors"]:
                    self.assertEqual(actual[name].dtype, torch.int8)
                    restored = actual[name].float() * actual[name + "_scale"]
                    self.assertTrue(torch.all((restored - value.float()).abs() <= actual[name + "_scale"] / 2 + 1e-6))
                else:
                    self.assertTrue(torch.equal(value.reshape(-1).view(torch.uint8), actual[name].reshape(-1).view(torch.uint8)))

    def test_checkpoint_rejects_missing_or_wrong_scale(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            name = "transformer_blocks.0.attn.to_q.weight"
            for scale in (None, torch.ones(128, 1, dtype=torch.bfloat16), torch.ones(1, 128)):
                tensors = {name: torch.ones(128, 128, dtype=torch.int8)}
                if scale is not None:
                    tensors[name + "_scale"] = scale
                save_file(tensors, str(root / "model.safetensors"))
                index = {"metadata": {"lightx2v_quant_scheme": "int8-metal", "lightx2v_quant_scope": "dit_proj"}, "weight_map": dict.fromkeys(tensors, "model.safetensors")}
                (root / "model.safetensors.index.json").write_text(json.dumps(index))
                with self.assertRaisesRegex(ValueError, "scale"):
                    validate_metal_checkpoint({"dit_quantized_ckpt": str(root), "dit_quant_scope": "dit_proj"})

    @unittest.skipUnless(
        torch.backends.mps.is_available() and (Path(os.getenv("LIGHTX2V_METALLIB_DIR") or "/") / "int8_k128_n128.metallib").is_file(),
        "MPS and compiled INT8 metallib required",
    )
    def test_direct_resident_mixed_storage_and_integer_oracle(self):
        from lightx2v.common.modules.weight_module import WeightModule
        from lightx2v.common.offload.mps_resident import load_resident_groups
        from lightx2v.common.offload.safetensors_checkpoint import SafetensorsCheckpoint
        from lightx2v.models.networks.minimax_h3.weights.transformer_weights import _linear

        with tempfile.TemporaryDirectory() as tmp:
            torch.manual_seed(19)
            name = "transformer_blocks.0.attn.to_q"
            q = torch.randint(-127, 128, (128, 128), dtype=torch.int8)
            values = {name + ".weight": q, name + ".weight_scale": torch.full((128, 1), 1 / 128), "proj_in.weight": torch.randn(128, 128).bfloat16()}
            save_file(values, str(Path(tmp) / "model.safetensors"))
            root = WeightModule()
            config = {"dit_quant_scheme": "int8-metal", "dit_quant_scope": "dit_proj"}
            root.add_module("quantized", _linear(config, name))
            root.add_module("floating", _linear(config, "proj_in"))
            stats = load_resident_groups(SafetensorsCheckpoint(tmp), [("fixture", root)])
            self.assertEqual(stats["weight_bytes"], sum(value.numel() * value.element_size() for value in values.values()))
            self.assertEqual(stats["weight_count"], 3)
            x = torch.randint(-127, 128, (129, 128)).bfloat16()
            x[:, 0] = 127
            expected = ((x.int() @ q.int().t()).float() / 128).bfloat16()
            actual = root.quantized.apply(x.to("mps")).cpu()
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            torch.testing.assert_close(root.floating.weight.cpu().t(), values["proj_in.weight"], rtol=0, atol=0)
            for obj in (root.quantized, root.floating):
                self.assertIsNone(getattr(obj, "pin_weight", None))
            with self.assertRaisesRegex(ValueError, "match exactly"):
                load_resident_groups(SafetensorsCheckpoint(tmp), [("only_one", root.quantized)])


if __name__ == "__main__":
    unittest.main()
