import ast
import importlib.util
import re
import sys
import types
import unittest
from pathlib import Path
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parents[1]
NETWORKS_ROOT = REPO_ROOT / "lightx2v" / "models" / "networks"


class LayerNormWeightRoutingTest(unittest.TestCase):
    def test_dit_infer_does_not_call_functional_layer_norm_directly(self):
        offenders = []
        for path in NETWORKS_ROOT.rglob("*.py"):
            tree = ast.parse(path.read_text(), filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "layer_norm":
                    offenders.append(f"{path.relative_to(REPO_ROOT)}:{node.lineno}")

        self.assertEqual(offenders, [], f"DiT LayerNorm calls must route through registered weights: {offenders}")

    def test_dit_weights_do_not_force_the_torch_layer_norm_registry(self):
        hardcoded = []
        pattern = re.compile(r"LN_WEIGHT_REGISTER\[\s*['\"]torch['\"]\s*\]")
        for path in NETWORKS_ROOT.rglob("*.py"):
            if "weights" not in path.relative_to(NETWORKS_ROOT).parts:
                continue
            for line_number, line in enumerate(path.read_text().splitlines(), start=1):
                if pattern.search(line):
                    hardcoded.append(f"{path.relative_to(REPO_ROOT)}:{line_number}")

        self.assertEqual(hardcoded, [], f"LayerNorm registry selection must come from config: {hardcoded}")

    def test_platform_layer_norm_supports_no_affine_weight_lifecycle(self):
        fake_torch = types.ModuleType("torch")
        fake_torch.bfloat16 = object()
        fake_torch.float16 = object()
        fake_torch.float32 = object()

        fake_safetensors = types.ModuleType("safetensors")
        fake_safetensors.safe_open = None

        fake_platform = types.ModuleType("lightx2v_platform")
        fake_platform.__path__ = []
        fake_base = types.ModuleType("lightx2v_platform.base")
        fake_base.__path__ = []
        fake_global_var = types.ModuleType("lightx2v_platform.base.global_var")
        fake_global_var.AI_DEVICE = "cpu"

        stubs = {
            "torch": fake_torch,
            "safetensors": fake_safetensors,
            "lightx2v_platform": fake_platform,
            "lightx2v_platform.base": fake_base,
            "lightx2v_platform.base.global_var": fake_global_var,
        }
        module_path = REPO_ROOT / "lightx2v_platform" / "ops" / "norm" / "norm_template.py"
        spec = importlib.util.spec_from_file_location("_layer_norm_template_under_test", module_path)
        module = importlib.util.module_from_spec(spec)

        with mock.patch.dict(sys.modules, stubs):
            spec.loader.exec_module(module)

        class DummyLayerNorm(module.LayerNormWeightTemplate):
            def apply(self, input_tensor):
                return input_tensor

        layer_norm = DummyLayerNorm(eps=1e-5)
        layer_norm.load({})
        layer_norm.to_cpu()
        self.assertEqual(layer_norm.state_dict(), {})
        layer_norm.load_state_dict({}, block_index=0)
        layer_norm.load_state_dict_from_disk(block_index=0)
        self.assertIsNone(layer_norm.weight)
        self.assertIsNone(layer_norm.bias)


if __name__ == "__main__":
    unittest.main()
