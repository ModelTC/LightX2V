import json
import os
import unittest
from pathlib import Path

os.environ.setdefault("SKIP_PLATFORM_CHECK", "1")

from lightx2v.models.schedulers.wan.scheduler_factory import get_wan_distill_method
from lightx2v.utils.set_config import validate_thor_config


class ThorConfigValidationTest(unittest.TestCase):
    @staticmethod
    def make_config(**overrides):
        config = {
            "model_cls": "wan2.2_moe",
            "distill_method": "dmd2",
            "task": "i2v",
            "dit_quant_scheme": "nvfp4",
        }
        config.update(overrides)
        return config

    def test_supported_model_task_combinations(self):
        for distill_method in (None, "dmd2"):
            for task in ("i2v", "t2v"):
                with self.subTest(distill_method=distill_method, task=task):
                    config = self.make_config(thor=True, task=task)
                    if distill_method is None:
                        config.pop("distill_method")
                    validate_thor_config(config)
                    self.assertEqual(get_wan_distill_method(config), distill_method)

    def test_retired_distill_model_name_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "model_cls"):
            validate_thor_config(self.make_config(thor=True, model_cls="wan2.2_moe_distill"))

    def test_thor_presets_select_dmd2(self):
        config_dir = Path(__file__).resolve().parents[1] / "configs" / "wan22" / "thor"
        paths = sorted(config_dir.glob("*.json"))
        self.assertTrue(paths)
        for path in paths:
            with self.subTest(config=path.name):
                config = json.loads(path.read_text())
                config.update(model_cls="wan2.2_moe", task="i2v")
                self.assertIs(config["thor"], True)
                validate_thor_config(config)
                self.assertEqual(get_wan_distill_method(config), "dmd2")

    def test_disabled_or_missing_thor_is_ignored(self):
        validate_thor_config({})
        validate_thor_config({"thor": False, "model_cls": "wan2.1", "dit_quant_scheme": "fp8"})

    def test_thor_must_be_boolean(self):
        for value in (None, 0, 1, "true", [], {}):
            with self.subTest(value=value):
                with self.assertRaisesRegex(TypeError, "thor must be a boolean"):
                    validate_thor_config(self.make_config(thor=value))

    def test_unsupported_configurations_are_rejected(self):
        for overrides, message in (
            ({"model_cls": "wan2.1"}, "model_cls"),
            ({"task": "flf2v"}, "task"),
            ({"dit_quant_scheme": "fp8"}, "dit_quant_scheme"),
        ):
            with self.subTest(overrides=overrides):
                with self.assertRaisesRegex(ValueError, message):
                    validate_thor_config(self.make_config(thor=True, **overrides))

    def test_legacy_fields_are_accepted_without_mutating_config(self):
        legacy_options = {
            "nvfp4_ffn0_gelu_fusion": True,
            "nvfp4_ffn2_residual_gate_fusion": False,
            "nvfp4_ffn_split_n_parts": 4,
            "nvfp4_ffn_split_n_stride_workaround": True,
            "nvfp4_large_gemm_cublaslt": False,
            "nvfp4_large_gemm_cublaslt_algorithm": 3,
            "nvfp4_qkv_cublaslt": True,
            "nvfp4_qkv_cublaslt_algorithm": 2,
        }
        for options in ({key: value} for key, value in legacy_options.items()):
            with self.subTest(options=options):
                config = dict(options)
                validate_thor_config(config)
                self.assertEqual(config, options)
                self.assertNotIn("thor", config)
        for thor in (False, True):
            with self.subTest(thor=thor):
                config = self.make_config(thor=thor, **legacy_options)
                expected = dict(config)
                validate_thor_config(config)
                self.assertEqual(config, expected)


if __name__ == "__main__":
    unittest.main()
