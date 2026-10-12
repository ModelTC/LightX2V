"""CPU regression tests; no inference kernels or model downloads required."""

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch

import torch
from safetensors.torch import save_file


def load_merge_modules():
    # Skip LightX2V's package initializer, which loads accelerator kernels.
    root = Path(__file__).resolve().parents[1]
    packages = {}
    for name in ("lightx2v", "lightx2v.utils", "lightx2v.models", "lightx2v.models.networks", "lightx2v.models.networks.minimax_h3", "lightx2v_platform", "lightx2v_platform.base"):
        packages[name] = ModuleType(name)
        packages[name].__path__ = []
    platform = ModuleType("lightx2v_platform.base.global_var")
    platform.AI_DEVICE = "cpu"
    packages[platform.__name__] = platform
    modules = []
    with patch.dict(sys.modules, packages):
        for name in ("lightx2v.utils.envs", "lightx2v.utils.lora_loader", "lightx2v.models.networks.lora_adapter", "lightx2v.models.networks.minimax_h3.lora"):
            spec = importlib.util.spec_from_file_location(name, root / (name.replace(".", "/") + ".py"))
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            spec.loader.exec_module(module)
            modules.append(module)
    return modules[1].LoRALoader, modules[2].LoraAdapter, modules[3].MiniMaxH3LoraAdapter


class TestFP32Merge(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.Loader, cls.Adapter, cls.H3Adapter = load_merge_modules()

    def setUp(self):
        generator = torch.Generator().manual_seed(42)
        self.up = torch.randn(16, 4, generator=generator)
        self.down = torch.randn(4, 12, generator=generator)
        self.weight = torch.randn(16, 12, generator=generator).bfloat16()
        self.lora = {"layer.lora_up.weight": self.up, "layer.lora_down.weight": self.down}
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = str(Path(self.directory.name) / "lora.safetensors")

    def model(self, weight, **config):
        return SimpleNamespace(
            config={"cpu_offload": True, "lora_merge_device": "cpu", **config},
            original_weight_dict={"layer.weight": weight},
            use_tp=False,
            _apply_weights=Mock(),
        )

    def test_loader_fp32_under_autocast(self):
        for dtype in (torch.bfloat16, torch.float32):
            with self.subTest(dtype=dtype):
                weight = self.weight.to(dtype).clone()
                ptr = weight.data_ptr()
                expected = (weight.float() + (self.up @ self.down) * 2.0 * 0.37).to(dtype)
                with torch.autocast("cpu", dtype=torch.bfloat16):
                    count = self.Loader().apply_lora({"layer.weight": weight}, self.lora, alpha=8, strength=0.37, force_fp32=True)
                self.assertEqual(count, 1)
                self.assertEqual(weight.dtype, dtype)
                self.assertEqual(weight.data_ptr(), ptr)
                torch.testing.assert_close(weight, expected, rtol=0, atol=0)

    def test_loader_default_preserves_bf16_arithmetic(self):
        weight = self.weight.clone()
        expected = weight + (self.up.bfloat16() @ self.down.bfloat16()) * 2.0 * 0.37
        self.Loader().apply_lora({"layer.weight": weight}, self.lora, alpha=8, strength=0.37)
        torch.testing.assert_close(weight, expected, rtol=0, atol=0)
        fp32_result = (self.weight.float() + (self.up @ self.down) * 2.0 * 0.37).bfloat16()
        self.assertFalse(torch.equal(weight, fp32_result))

    def test_diff_weights(self):
        for force in (False, True):
            for suffix, target in (("diff", "weight"), ("diff_b", "bias"), ("diff_m", "modulation")):
                with self.subTest(force=force, suffix=suffix):
                    weight = self.weight.clone()
                    diff = self.up @ self.down
                    if force:
                        expected = (weight.float() + diff * 0.3 * 0.7).bfloat16()
                    else:
                        expected = weight + diff.bfloat16() * 0.3 * 0.7
                    count = self.Loader().apply_lora({f"layer.{target}": weight}, {f"layer.{suffix}": diff}, alpha=0.3, strength=0.7, force_fp32=force)
                    self.assertEqual(count, 1)
                    torch.testing.assert_close(weight, expected, rtol=0, atol=0)

    def test_adapter_preserves_fp32_checkpoint_factors(self):
        save_file(self.lora, self.path)
        model = self.model(self.weight.clone(), lora_merge_force_fp32=True)
        adapter = self.Adapter(model)
        loaded = adapter._load_lora_file(self.path)
        for key, value in self.lora.items():
            self.assertEqual(loaded[key].dtype, torch.float32)
            torch.testing.assert_close(loaded[key], value, rtol=0, atol=0)
        expected = (self.weight.float() + (self.up @ self.down) * 0.37).bfloat16()
        adapter.apply_lora([{"path": self.path, "strength": 0.37}])
        torch.testing.assert_close(model.original_weight_dict["layer.weight"], expected, rtol=0, atol=0)
        model._apply_weights.assert_called_once()
        self.assertFalse(self.Adapter(self.model(self.weight)).merge_force_fp32)

    def test_h3_fp32_merge_and_tp_shards(self):
        save_file({"base_model.model.layer.lora_A.default.weight": self.down, "base_model.model.layer.lora_B.default.weight": self.up}, self.path)
        for dtype in (torch.bfloat16, torch.float32):
            for split in (None, "row", "col", "ff_fused_col"):
                for rank in range(2 if split else 1):
                    with self.subTest(dtype=dtype, split=split, rank=rank):
                        base = self.weight.to(dtype)
                        expected = base.float().clone().add_(self.up @ self.down, alpha=2.0 * 0.37).to(dtype)
                        if split == "row":
                            base, expected = base.chunk(2, dim=1)[rank], expected.chunk(2, dim=1)[rank]
                        elif split == "col":
                            base, expected = base.chunk(2, dim=0)[rank], expected.chunk(2, dim=0)[rank]
                        elif split == "ff_fused_col":
                            base = torch.cat([part.chunk(2, dim=0)[rank] for part in base.chunk(2, dim=0)])
                            expected = torch.cat([part.chunk(2, dim=0)[rank] for part in expected.chunk(2, dim=0)])
                        weight = base.clone()
                        ptr = weight.data_ptr()
                        model = self.model(weight, lora_merge_force_fp32=True)
                        model.use_tp = split is not None
                        model.tp_size, model.tp_rank = 2, rank
                        model._tp_split_type = lambda key: split
                        with torch.autocast("cpu", dtype=torch.bfloat16):
                            count = self.H3Adapter(model)._merge_file(self.path, alpha=8, strength=0.37)
                        self.assertEqual(count, 1)
                        self.assertEqual(weight.dtype, dtype)
                        self.assertEqual(weight.data_ptr(), ptr)
                        torch.testing.assert_close(weight, expected, rtol=0, atol=0)

    def test_h3_default_and_checkpoint_alpha(self):
        save_file({**self.lora, "layer.alpha": torch.tensor(8.0)}, self.path)
        weight = self.weight.clone()
        expected = weight.clone().add_(self.up.bfloat16() @ self.down.bfloat16(), alpha=2.0 * 0.37)
        model = self.model(weight)
        count = self.H3Adapter(model).apply_lora([{"path": self.path, "alpha": 99, "strength": 0.37}])
        self.assertEqual(count, 1)
        torch.testing.assert_close(weight, expected, rtol=0, atol=0)
        model._apply_weights.assert_called_once()


if __name__ == "__main__":
    unittest.main()
