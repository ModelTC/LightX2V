"""Verify that developer measurement preserves normal inference calls and cleanup."""

import importlib.util
from contextlib import nullcontext
import sys
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch

from evaluate import MODEL, record_model_events, record_pipeline_stage_times
import evaluate as evaluation


class Event:
    def __init__(self, enable_timing):
        assert enable_timing
        self.recorded = False

    def record(self):
        self.recorded = True


class MeasurementTests(unittest.TestCase):
    def test_calls_preserve_results_and_restore_class_methods(self):
        class Model:
            def forward(self, value, *, scale):
                return value * scale

        model = Model()
        with record_model_events([(model, "forward", "forward")], SimpleNamespace(Event=Event)) as events:
            self.assertEqual(model.forward(3, scale=4), 12)
            self.assertEqual(model.forward(5, scale=2), 10)
        self.assertNotIn("forward", vars(model))
        self.assertEqual(model.forward(6, scale=3), 18)
        self.assertEqual(len(events), 2)
        self.assertTrue(all(start.recorded and end.recorded for _, start, end in events))

    def test_failure_restores_all_hooks_and_existing_instance_override(self):
        original = Mock(side_effect=RuntimeError("model failure"))
        model = SimpleNamespace(forward=original)
        sampler = SimpleNamespace(sample_once=lambda value: value)
        sample_once = sampler.sample_once
        with self.assertRaisesRegex(RuntimeError, "model failure"):
            with record_model_events([(sampler, "sample_once", "step"), (model, "forward", "forward")], SimpleNamespace(Event=Event)):
                model.forward("image")
        self.assertIs(model.forward, original)
        self.assertIs(sampler.sample_once, sample_once)
        original.assert_called_once_with("image")


class AdapterTests(unittest.TestCase):
    def test_native_adapter_uses_public_request_api(self):
        # Only the external framework is replaced; exercise the actual adapter.
        factory = ModuleType("lightx2v.models.runners.runner_factory")
        settings_module = ModuleType("lightx2v.utils.set_config")
        modules = {"torch": ModuleType("torch"), factory.__name__: factory, settings_module.__name__: settings_module}
        with patch.dict(sys.modules, modules):
            spec = importlib.util.spec_from_file_location("boundary_test_adapters", Path(__file__).resolve().parents[1] / "evaluate.py")
            adapters = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(adapters)
            for model in (MODEL,):
                with self.subTest(model=model):
                    prepared, prediction = object(), object()
                    runner = SimpleNamespace(prepare_request=Mock(return_value=prepared), run_request=Mock(return_value=prediction))
                    factory.build_runner = Mock(return_value=runner)
                    settings_module.build_startup_config = Mock(side_effect=lambda settings: settings)
                    paths = {name: Path("/weights") / name for name in ("trellis2", "base", "dino", "rembg")}
                    adapter = adapters.LightX2VAdapter(paths, Path("/unused"))
                    with patch.object(adapters, "record_pipeline_stage_times", return_value=nullcontext({})):
                        result = adapter.predict(["images/a.png", "images/b.png"], seed=42)
                    self.assertIs(result, prediction)
                    expected = {"seed": 42, "image_path": "/data/images/a.png"}
                    self.assertEqual(settings_module.build_startup_config.call_args.args[0]["config_json"], str(adapters.config_path("model.json")))
                    runner.prepare_request.assert_called_once_with(expected)
                    runner.run_request.assert_called_once_with(prepared)
                    self.assertEqual(set(vars(runner)), {"prepare_request", "run_request"})


class PipelineStageTests(unittest.TestCase):
    def test_shared_callable_class_has_disjoint_stages_and_restores_methods(self):
        class Callable:
            def __call__(self, value):
                return value

        class Runner:
            def __init__(self):
                self.encoder = Callable()
                self.models = {name: Callable() for name in (
                    "sparse_structure_decoder", "shape_slat_decoder", "tex_slat_decoder")}

            def _sample(self, stage, value):
                return value

        runner = Runner()
        original = Callable.__call__
        module = sys.modules[Runner.__module__]
        filler = Mock(side_effect=lambda value: value)
        device = SimpleNamespace(synchronize=Mock())
        with patch.object(module, "fill_mesh_holes", filler, create=True), patch.object(evaluation, "device_api", return_value=device):
            with record_pipeline_stage_times(runner) as timings:
                self.assertEqual(runner.encoder(1), 1)
                for model in runner.models.values():
                    self.assertEqual(model(2), 2)
                for stage in ("sparse_structure", "shape_slat", "tex_slat"):
                    self.assertEqual(runner._sample(stage, 3), 3)
                self.assertEqual(module.fill_mesh_holes(4), 4)
            self.assertIs(module.fill_mesh_holes, filler)
        self.assertIs(Callable.__call__, original)
        self.assertNotIn("_sample", vars(runner))
        self.assertEqual(len(timings), 8)
        self.assertEqual(device.synchronize.call_count, 16)

    def test_exception_restores_class_and_runner_hooks(self):
        class Callable:
            def __call__(self, value):
                raise RuntimeError("expected stage failure")

        class Runner:
            encoder = Callable()
            models = {key: Callable() for key in ("sparse_structure_decoder", "shape_slat_decoder", "tex_slat_decoder")}

            def _sample(self, stage):
                return stage

        runner, original = Runner(), Callable.__call__
        module = sys.modules[Runner.__module__]
        with patch.object(module, "fill_mesh_holes", Mock(), create=True), patch.object(evaluation, "device_api", return_value=SimpleNamespace(synchronize=Mock())):
            with self.assertRaisesRegex(RuntimeError, "expected stage failure"):
                with record_pipeline_stage_times(runner):
                    runner.encoder(0)
        self.assertIs(Callable.__call__, original)
        self.assertNotIn("_sample", vars(runner))


if __name__ == "__main__":
    unittest.main()
