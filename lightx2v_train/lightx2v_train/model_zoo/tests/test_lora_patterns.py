"""CPU regressions for per-layer rank/alpha through configuration and exports."""

import copy
import json
import tempfile
import unittest
from pathlib import Path

import torch
from peft import LoraConfig
from safetensors.torch import load_file

from lightx2v_train.model_capabilities import TrainableModelCapability
from lightx2v_train.model_zoo.tests.test_lora_precision import _TinyModel
from lightx2v_train.trainers.base import BaseTrainer


class LoRAPatternTest(unittest.TestCase):
    def setUp(self):
        state = torch.get_rng_state()
        self.addCleanup(torch.set_rng_state, state)

    @staticmethod
    def model():
        model = _TinyModel()
        model.transformer.adaln_proj = torch.nn.Module()
        model.transformer.adaln_proj.linear = torch.nn.Linear(5, 3, dtype=torch.bfloat16)
        return model

    @staticmethod
    def options():
        return dict(rank=128, alpha=128, param_dtype="fp32", target_modules=["projection", "adaln_proj.linear"], rank_pattern={"adaln_proj.linear": 16}, alpha_pattern={"adaln_proj.linear": 16})

    def assert_patterns(self, model):
        for layer, rank in [(model.transformer.projection, 128), (model.transformer.adaln_proj.linear, 16)]:
            self.assertEqual(layer.r["default"], rank)
            self.assertEqual(layer.lora_alpha["default"], rank)
            self.assertEqual(layer.scaling["default"], 1)
        for name, param in model.transformer.named_parameters():
            self.assertEqual(param.requires_grad, "lora_" in name)
            self.assertEqual(param.dtype, torch.float32 if "lora_" in name else torch.bfloat16)

    def test_common_capability_preserves_per_layer_patterns(self):
        model = self.model()
        model.ensure_capabilities().require(TrainableModelCapability).configure("lora", self.options())
        self.assert_patterns(model)

    def test_base_trainer_forwards_patterns(self):
        model = self.model()
        trainer = BaseTrainer.__new__(BaseTrainer)
        options = self.options()
        trainer.train_type = "lora"
        for name, value in options.items():
            setattr(trainer, f"lora_{name}", value)
        trainer._setup_trainable_model(model)
        self.assert_patterns(model)

    def test_mixed_rank_export_sidecar_reconstructs_scaling_and_resume_delta(self):
        source = self.model()
        target = copy.deepcopy(source)
        source.ensure_capabilities().require(TrainableModelCapability).configure("lora", self.options())
        with torch.no_grad():
            for name, param in source.transformer.named_parameters():
                if "lora_" in name:
                    param.normal_(std=0.02)
        with tempfile.TemporaryDirectory() as folder:
            source.save_lora_weights(folder)
            saved = json.loads((Path(folder) / "adapter_config.json").read_text())
            self.assertEqual((saved["r"], saved["lora_alpha"]), (128, 128))
            self.assertEqual(saved["rank_pattern"], {"adaln_proj.linear": 16})
            self.assertEqual(saved["alpha_pattern"], {"adaln_proj.linear": 16})
            weights = load_file(str(Path(folder) / "pytorch_lora_weights.safetensors"))
            self.assertTrue(all("lora_" in key or ".lora." in key for key in weights))
            target.transformer.add_adapter(LoraConfig.from_pretrained(folder))
            target.set_lora_param_dtype("fp32")
            target.set_lora_trainable()
            target.load_lora_weights_for_resume(folder)
        self.assert_patterns(target)
        for name in ["projection", "adaln_proj.linear"]:
            actual = target.transformer.get_submodule(name)
            expected = source.transformer.get_submodule(name)
            torch.testing.assert_close(actual.get_delta_weight("default"), expected.get_delta_weight("default"), rtol=0, atol=0)

    def test_uniform_rank_legacy_export_keeps_existing_artifact_format(self):
        model = self.model()
        model.ensure_capabilities().require(TrainableModelCapability).configure("lora", dict(rank=2, alpha=4, target_modules=["projection"]))
        with tempfile.TemporaryDirectory() as folder:
            model.save_lora_weights(folder)
            self.assertFalse((Path(folder) / "adapter_config.json").exists())
            self.assertTrue((Path(folder) / "pytorch_lora_weights.safetensors").is_file())


if __name__ == "__main__":
    unittest.main()
