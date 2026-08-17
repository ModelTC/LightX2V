import importlib.util
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import torch


def _package(name):
    module = types.ModuleType(name)
    module.__path__ = []
    return module


def _stub_modules():
    class BaseTransformerModel:
        pass

    fastwam_infer = types.ModuleType("lightx2v.models.networks.wan.infer.fastwam")
    fastwam_infer.FastWAMPreInfer = type("FastWAMPreInfer", (), {})
    fastwam_infer.FastWAMTransformerInfer = type("FastWAMTransformerInfer", (), {})

    fastwam_weights = types.ModuleType("lightx2v.models.networks.wan.weights.fastwam")
    fastwam_weights.FastWAMPreWeights = type("FastWAMPreWeights", (), {})
    fastwam_weights.FastWAMTransformerWeights = type("FastWAMTransformerWeights", (), {})

    fastwam_scheduler = types.ModuleType("lightx2v.models.schedulers.wan.fastwam")
    fastwam_scheduler.FastWAMActionScheduler = type("FastWAMActionScheduler", (), {})

    base_model = types.ModuleType("lightx2v.models.networks.base_model")
    base_model.BaseTransformerModel = BaseTransformerModel

    envs = types.ModuleType("lightx2v.utils.envs")
    envs.GET_DTYPE = lambda: torch.float32
    envs.GET_SENSITIVE_DTYPE = lambda: torch.float64

    ops = types.ModuleType("lightx2v.common.ops")
    ops.__all__ = []

    return {
        "lightx2v": _package("lightx2v"),
        "lightx2v.common": _package("lightx2v.common"),
        "lightx2v.common.ops": ops,
        "lightx2v.models": _package("lightx2v.models"),
        "lightx2v.models.networks": _package("lightx2v.models.networks"),
        "lightx2v.models.networks.base_model": base_model,
        "lightx2v.models.networks.wan": _package("lightx2v.models.networks.wan"),
        "lightx2v.models.networks.wan.infer": _package("lightx2v.models.networks.wan.infer"),
        "lightx2v.models.networks.wan.infer.fastwam": fastwam_infer,
        "lightx2v.models.networks.wan.weights": _package("lightx2v.models.networks.wan.weights"),
        "lightx2v.models.networks.wan.weights.fastwam": fastwam_weights,
        "lightx2v.models.schedulers": _package("lightx2v.models.schedulers"),
        "lightx2v.models.schedulers.wan": _package("lightx2v.models.schedulers.wan"),
        "lightx2v.models.schedulers.wan.fastwam": fastwam_scheduler,
        "lightx2v.utils": _package("lightx2v.utils"),
        "lightx2v.utils.envs": envs,
    }


def _load_fastwam_model_module():
    module_path = Path(__file__).resolve().parents[1] / "lightx2v/models/networks/wan/fastwam_model.py"
    spec = importlib.util.spec_from_file_location("_fastwam_model_under_test", module_path)
    module = importlib.util.module_from_spec(spec)
    with mock.patch.dict(sys.modules, _stub_modules()):
        spec.loader.exec_module(module)
    return module


class FastWAMCheckpointMmapTest(unittest.TestCase):
    def _run_load_ckpt(self, extra_config=None, payload=None):
        module = _load_fastwam_model_module()
        model = object.__new__(module.FastWAMNativeModel)
        model.device = torch.device("cpu")

        calls = []

        def fake_load(path, **kwargs):
            calls.append((path, kwargs))
            return payload or {
                "mot": {"blocks.0.weight": torch.ones(2, 2)},
                "proprio_encoder": {
                    "weight": torch.ones(1, 2),
                    "bias": torch.zeros(1),
                },
            }

        with tempfile.NamedTemporaryFile(suffix=".pt") as ckpt:
            config = {"adapter_model_path": ckpt.name}
            if extra_config:
                config.update(extra_config)
            model.config = config
            with mock.patch.object(module.torch, "load", side_effect=fake_load):
                weight_dict = model._load_ckpt(unified_dtype=True, sensitive_layer=set())

        self.assertEqual(len(calls), 1)
        return calls[0], weight_dict

    def test_default_config_passes_mmap_false_to_torch_load(self):
        (_path, kwargs), weight_dict = self._run_load_ckpt()

        self.assertEqual(kwargs["map_location"], "cpu")
        self.assertIs(kwargs["weights_only"], True)
        self.assertIs(kwargs["mmap"], False)
        self.assertIn("blocks.0.weight", weight_dict)
        self.assertIn("proprio_encoder.weight", weight_dict)
        self.assertIn("proprio_encoder.bias", weight_dict)

    def test_explicit_config_passes_mmap_true_to_torch_load(self):
        (_path, kwargs), _weight_dict = self._run_load_ckpt({"checkpoint_mmap": True})

        self.assertIs(kwargs["mmap"], True)

    def test_checkpoint_schema_still_requires_mot(self):
        module = _load_fastwam_model_module()
        model = object.__new__(module.FastWAMNativeModel)
        model.device = torch.device("cpu")

        with tempfile.NamedTemporaryFile(suffix=".pt") as ckpt:
            model.config = {"adapter_model_path": ckpt.name, "checkpoint_mmap": True}
            with mock.patch.object(module.torch, "load", return_value={"not_mot": {}}):
                with self.assertRaisesRegex(ValueError, "must contain `mot`"):
                    model._load_ckpt(unified_dtype=True, sensitive_layer=set())


if __name__ == "__main__":
    unittest.main()
