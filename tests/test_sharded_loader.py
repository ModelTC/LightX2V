"""CPU tests for the generic FSDP2 safetensors loader.

The safetensors reader is replaced with a tiny in-memory reader so these tests
exercise key resolution, validation, slicing and file lifetime without a GPU or
an actual model checkpoint.
"""

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


HELPER = Path(__file__).resolve().parents[1] / "lightx2v_train/lightx2v_train/runtime/sharded_loader.py"
spec = importlib.util.spec_from_file_location("sharded_loader_under_test", HELPER)
loader = importlib.util.module_from_spec(spec)
spec.loader.exec_module(loader)

try:
    import torch
except ImportError:
    torch = None


class FakeSlice:
    def __init__(self, tensor):
        self.tensor = tensor

    def get_shape(self):
        return tuple(self.tensor.shape)

    def get_dtype(self):
        return "F32" if self.tensor.dtype is torch.float32 else "F16"

    def __getitem__(self, index):
        return self.tensor[index]


class FakeShard:
    def __init__(self, values):
        self.values = values
        self.closed = False

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.closed = True

    def keys(self):
        return list(self.values)

    def get_slice(self, key):
        if key not in self.values and key.startswith("base."):
            key = "linear." + key.split(".", 1)[1]
        return FakeSlice(self.values[key])


class FakeModule:
    def __init__(self):
        self.loaded = {}

    def load_state_dict(self, state, *, strict, assign):
        self.loaded.update({key: value.clone() for key, value in state.items()})
        return type("Incompatible", (), {"unexpected_keys": []})()


@unittest.skipIf(torch is None, "PyTorch is not installed")
class ShardedLoaderTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.values = {
            "part-1.safetensors": {"base.weight": torch.arange(16, dtype=torch.float32).reshape(4, 4)},
            "part-2.safetensors": {"base.bias": torch.arange(4, dtype=torch.float32)},
        }
        for filename in self.values:
            (self.root / filename).touch()

    def tearDown(self):
        self.tmp.cleanup()

    def fake_open(self, path, **_):
        return FakeShard(self.values[Path(path).name])

    def write_index(self, weight_map):
        index = self.root / "model.safetensors.index.json"
        index.write_text(json.dumps({"weight_map": weight_map}), encoding="utf-8")
        return index

    def test_indexed_load_resolves_keys_and_releases_each_file(self):
        self.write_index({"base.weight": "part-1.safetensors", "base.bias": "part-2.safetensors"})
        module = FakeModule()
        templates = {
            "base_layer.weight": torch.empty(4, 4),
            "base_layer.bias": torch.empty(4),
        }
        with patch.object(loader, "_get_safe_open", return_value=self.fake_open):
            metadata = loader.load_fsdp2_safetensors(
                module,
                self.root,
                templates=templates,
                key_resolver=lambda key, _: key.replace("base.", "base_layer."),
            )
        self.assertEqual(metadata["loaded_key_count"], 2)
        self.assertEqual(metadata["files"], ["part-1.safetensors", "part-2.safetensors"])
        torch.testing.assert_close(module.loaded["base_layer.weight"], self.values["part-1.safetensors"]["base.weight"])

    def test_single_file_checkpoint_is_supported(self):
        single = self.root / "single.safetensors"
        single.touch()
        self.values["single.safetensors"] = {"weight": torch.ones(2, 2)}
        module = FakeModule()
        with patch.object(loader, "_get_safe_open", return_value=self.fake_open):
            metadata = loader.load_fsdp2_safetensors(
                module,
                single,
                templates={"weight": torch.empty(2, 2)},
            )
        self.assertEqual(metadata["checkpoint_key_count"], 1)
        torch.testing.assert_close(module.loaded["weight"], torch.ones(2, 2))

    def test_shape_dtype_and_duplicate_errors_are_strict(self):
        self.write_index({"base.weight": "part-1.safetensors"})
        with patch.object(loader, "_get_safe_open", return_value=self.fake_open):
            with self.assertRaises(ValueError):
                loader.load_fsdp2_safetensors(
                    FakeModule(), self.root, templates={"base.weight": torch.empty(3, 3)}
                )

            with self.assertRaises(TypeError):
                loader.load_fsdp2_safetensors(
                    FakeModule(), self.root, templates={"base.weight": torch.empty(4, 4, dtype=torch.float16)}
                )

            with self.assertRaises(ValueError):
                self.write_index({"base.weight": "part-1.safetensors", "base.bias": "part-1.safetensors"})
                loader.load_fsdp2_safetensors(
                    FakeModule(),
                    self.root,
                    templates={"a": torch.empty(4, 4)},
                    key_resolver=lambda *_: "a",
                )

    def test_missing_and_unresolved_keys_require_explicit_policy(self):
        self.write_index({"base.weight": "part-1.safetensors", "extra": "part-2.safetensors"})
        with patch.object(loader, "_get_safe_open", return_value=self.fake_open):
            with self.assertRaises(KeyError):
                loader.load_fsdp2_safetensors(
                    FakeModule(), self.root, templates={"base.weight": torch.empty(4, 4)}
                )
            module = FakeModule()
            metadata = loader.load_fsdp2_safetensors(
                module,
                self.root,
                templates={"base.weight": torch.empty(4, 4), "missing": torch.empty(1)},
                ignored_checkpoint_keys={"extra"},
                ignored_model_keys={"missing"},
                key_resolver=lambda key, _: None if key == "extra" else key,
            )
        self.assertEqual(metadata["ignored_checkpoint_keys"], ["extra"])

    def test_missing_initializer_must_match_template(self):
        self.write_index({"base.weight": "part-1.safetensors"})
        with patch.object(loader, "_get_safe_open", return_value=self.fake_open):
            with self.assertRaises(ValueError):
                loader.load_fsdp2_safetensors(
                    FakeModule(),
                    self.root,
                    templates={"base.weight": torch.empty(4, 4), "missing": torch.empty(2)},
                    initialize_missing=lambda *_: torch.empty(3),
                )


class ShardedLoadConfigTests(unittest.TestCase):
    def test_new_bool_and_mapping(self):
        self.assertTrue(loader.sharded_load_enabled({"distributed": {"fsdp2": {"sharded_load": True}}}))
        normalized = loader.normalize_sharded_load_config(
            {"distributed": {"fsdp2": {"sharded_load": {"enabled": True, "index_filename": "x"}}}}
        )
        self.assertEqual(normalized["options"], {"index_filename": "x"})
        self.assertFalse(normalized["deprecated_alias"])

    def test_legacy_alias_is_recorded_and_conflicts_are_rejected(self):
        normalized = loader.normalize_sharded_load_config(
            {"distributed": {"fsdp2": {"serialized_load": True}}}
        )
        self.assertTrue(normalized["enabled"])
        self.assertTrue(normalized["deprecated_alias"])
        with self.assertRaises(ValueError):
            loader.normalize_sharded_load_config(
                {"distributed": {"fsdp2": {"sharded_load": False, "serialized_load": True}}}
            )


if __name__ == "__main__":
    unittest.main()
