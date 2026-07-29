import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch


REPO_ROOT = Path(__file__).resolve().parents[1]


def _package(name):
    module = types.ModuleType(name)
    module.__path__ = []
    return module


def _identity_decorator(*args, **kwargs):
    def decorate(func):
        return func

    return decorate


def _load_model_module():
    torch_module = _package("torch")
    torch_distributed = types.ModuleType("torch.distributed")
    torch_functional = types.ModuleType("torch.nn.functional")
    torch_nn = _package("torch.nn")
    torch_nn.functional = torch_functional
    torch_module.distributed = torch_distributed
    torch_module.nn = torch_nn
    torch_module.no_grad = _identity_decorator

    class _BaseTransformerModel:
        pass

    base_model = types.ModuleType("lightx2v.models.networks.base_model")
    base_model.BaseTransformerModel = _BaseTransformerModel

    class _Infer:
        pass

    infer_modules = {}
    for module_name, class_names in {
        "lightx2v.models.networks.flux2.infer.feature_caching.transformer_infer": ("Flux2TransformerInferAdaCaching",),
        "lightx2v.models.networks.flux2.infer.offload.transformer_infer": ("Flux2OffloadTransformerInfer",),
        "lightx2v.models.networks.flux2.infer.post_infer": ("Flux2PostInfer",),
        "lightx2v.models.networks.flux2.infer.pre_infer": ("Flux2DevPreInfer", "Flux2PreInfer"),
        "lightx2v.models.networks.flux2.infer.transformer_infer": ("Flux2TransformerInfer",),
    }.items():
        stub = types.ModuleType(module_name)
        for class_name in class_names:
            setattr(stub, class_name, _Infer)
        infer_modules[module_name] = stub

    post_weights = types.ModuleType("lightx2v.models.networks.flux2.weights.post_weights")
    post_weights.Flux2PostWeights = object
    pre_weights = types.ModuleType("lightx2v.models.networks.flux2.weights.pre_weights")
    pre_weights.Flux2DevPreWeights = object
    pre_weights.Flux2PreWeights = object

    def preserve(module):
        module.preserve_calls += 1

    def release(module):
        module.release_calls += 1

    transformer_weights = types.ModuleType("lightx2v.models.networks.flux2.weights.transformer_weights")
    transformer_weights.Flux2TransformerWeights = object
    transformer_weights.preserve_weight_module_cpu_tensors = preserve
    transformer_weights.release_weight_module_device_tensors = release
    transformer_weights.validate_flux2_block_slab_config = lambda config, ai_device: False

    custom_compiler = types.ModuleType("lightx2v.utils.custom_compiler")
    custom_compiler.compiled_method = _identity_decorator
    global_var = types.ModuleType("lightx2v_platform.base.global_var")
    global_var.AI_DEVICE = None
    platform_base = _package("lightx2v_platform.base")
    platform_base.global_var = global_var

    stub_modules = {
        "torch": torch_module,
        "torch.distributed": torch_distributed,
        "torch.nn": torch_nn,
        "torch.nn.functional": torch_functional,
        "lightx2v": _package("lightx2v"),
        "lightx2v.models": _package("lightx2v.models"),
        "lightx2v.models.networks": _package("lightx2v.models.networks"),
        "lightx2v.models.networks.base_model": base_model,
        "lightx2v.models.networks.flux2": _package("lightx2v.models.networks.flux2"),
        "lightx2v.models.networks.flux2.infer": _package("lightx2v.models.networks.flux2.infer"),
        "lightx2v.models.networks.flux2.weights": _package("lightx2v.models.networks.flux2.weights"),
        "lightx2v.models.networks.flux2.weights.post_weights": post_weights,
        "lightx2v.models.networks.flux2.weights.pre_weights": pre_weights,
        "lightx2v.models.networks.flux2.weights.transformer_weights": transformer_weights,
        "lightx2v.utils": _package("lightx2v.utils"),
        "lightx2v.utils.custom_compiler": custom_compiler,
        "lightx2v_platform": _package("lightx2v_platform"),
        "lightx2v_platform.base": platform_base,
        "lightx2v_platform.base.global_var": global_var,
        **infer_modules,
    }

    module_path = REPO_ROOT / "lightx2v" / "models" / "networks" / "flux2" / "model.py"
    spec = importlib.util.spec_from_file_location("_flux2_model_lifecycle_under_test", module_path)
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, stub_modules):
        spec.loader.exec_module(module)
    return module


class _ProfileContext:
    def __call__(self, func):
        return func

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False


def _profile_context(*args, **kwargs):
    return _ProfileContext()


def _load_runner_module():
    torch_module = _package("torch")
    torch_module.cpu = types.SimpleNamespace(empty_cache=lambda: None)

    model_stub = types.ModuleType("lightx2v.models.networks.flux2.model")
    model_stub.Flux2DevTransformerModel = object
    model_stub.Flux2KleinTransformerModel = object
    default_runner = types.ModuleType("lightx2v.models.runners.default_runner")
    default_runner.DefaultRunner = object

    scheduler_base = types.ModuleType("lightx2v.models.schedulers.flux2.scheduler")
    scheduler_base.Flux2DevScheduler = object
    scheduler_base.Flux2Scheduler = object
    scheduler_caching = types.ModuleType("lightx2v.models.schedulers.flux2.feature_caching.scheduler")
    scheduler_caching.Flux2DevSchedulerCaching = object
    scheduler_caching.Flux2SchedulerCaching = object
    vae_module = types.ModuleType("lightx2v.models.video_encoders.hf.flux2.vae")
    vae_module.Flux2VAE = object

    profiler = types.ModuleType("lightx2v.utils.profiler")
    profiler.ProfilingContext4DebugL1 = _profile_context
    profiler.ProfilingContext4DebugL2 = _profile_context
    registry = types.ModuleType("lightx2v.utils.registry_factory")
    registry.RUNNER_REGISTER = lambda name: (lambda cls: cls)
    utils = types.ModuleType("lightx2v.utils.utils")
    utils.is_main_process = lambda: True
    global_var = types.ModuleType("lightx2v_platform.base.global_var")
    global_var.AI_DEVICE = "cpu"

    stub_modules = {
        "torch": torch_module,
        "lightx2v": _package("lightx2v"),
        "lightx2v.models": _package("lightx2v.models"),
        "lightx2v.models.networks": _package("lightx2v.models.networks"),
        "lightx2v.models.networks.flux2": _package("lightx2v.models.networks.flux2"),
        "lightx2v.models.networks.flux2.model": model_stub,
        "lightx2v.models.runners": _package("lightx2v.models.runners"),
        "lightx2v.models.runners.default_runner": default_runner,
        "lightx2v.models.schedulers": _package("lightx2v.models.schedulers"),
        "lightx2v.models.schedulers.flux2": _package("lightx2v.models.schedulers.flux2"),
        "lightx2v.models.schedulers.flux2.scheduler": scheduler_base,
        "lightx2v.models.schedulers.flux2.feature_caching": _package("lightx2v.models.schedulers.flux2.feature_caching"),
        "lightx2v.models.schedulers.flux2.feature_caching.scheduler": scheduler_caching,
        "lightx2v.models.video_encoders": _package("lightx2v.models.video_encoders"),
        "lightx2v.models.video_encoders.hf": _package("lightx2v.models.video_encoders.hf"),
        "lightx2v.models.video_encoders.hf.flux2": _package("lightx2v.models.video_encoders.hf.flux2"),
        "lightx2v.models.video_encoders.hf.flux2.vae": vae_module,
        "lightx2v.utils": _package("lightx2v.utils"),
        "lightx2v.utils.profiler": profiler,
        "lightx2v.utils.registry_factory": registry,
        "lightx2v.utils.utils": utils,
        "lightx2v_platform": _package("lightx2v_platform"),
        "lightx2v_platform.base": _package("lightx2v_platform.base"),
        "lightx2v_platform.base.global_var": global_var,
    }

    module_path = REPO_ROOT / "lightx2v" / "models" / "runners" / "flux2" / "flux2_runner.py"
    spec = importlib.util.spec_from_file_location("_flux2_runner_lifecycle_under_test", module_path)
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, stub_modules):
        spec.loader.exec_module(module)
    return module


model_module = _load_model_module()
runner_module = _load_runner_module()


class _Weight:
    def __init__(self, fail=False):
        self.fail = fail
        self.to_cuda_calls = 0
        self.release_calls = 0
        self.preserve_calls = 0

    def to_cuda(self):
        self.to_cuda_calls += 1
        if self.fail:
            raise RuntimeError("prepare failed")


class _TransformerWeights:
    def __init__(self):
        self.non_block_to_cuda_calls = 0
        self.resident_to_cuda_calls = 0
        self.release_non_block_calls = 0
        self.release_resident_calls = 0

    def non_block_weights_to_cuda(self):
        self.non_block_to_cuda_calls += 1

    def resident_blocks_to_cuda(self):
        self.resident_to_cuda_calls += 1

    def release_non_block_weights(self):
        self.release_non_block_calls += 1

    def release_resident_blocks(self):
        self.release_resident_calls += 1


class _Manager:
    def __init__(self):
        self.flush_calls = 0

    def flush(self):
        self.flush_calls += 1


def _make_block_model(step_index=3, post_fail=False):
    model_type = model_module._Flux2TransformerModelBase
    model = object.__new__(model_type)
    model.cpu_offload = True
    model.offload_granularity = "block"
    model._offload_weights_loaded = False
    model._offload_weights_preparing = False
    model.scheduler = types.SimpleNamespace(step_index=step_index, infer_steps=10)
    model.pre_weight = _Weight()
    model.post_weight = _Weight(fail=post_fail)
    model.transformer_weights = _TransformerWeights()
    double_manager = _Manager()
    single_manager = _Manager()
    model.transformer_infer = types.SimpleNamespace(
        use_npu_event_offload=True,
        offload_manager_double=double_manager,
        offload_manager_single=single_manager,
    )
    model.sync_calls = 0

    def sync():
        model.sync_calls += 1

    model._sync_device = sync
    return model


class Flux2OffloadModelLifecycleTest(unittest.TestCase):
    def test_nonzero_cold_start_loads_once_and_repeated_prepare_is_idempotent(self):
        model = _make_block_model(step_index=4)

        model._prepare_infer_weights()
        model._prepare_infer_weights()

        self.assertTrue(model._offload_weights_loaded)
        self.assertEqual(model.pre_weight.to_cuda_calls, 1)
        self.assertEqual(model.post_weight.to_cuda_calls, 1)
        self.assertEqual(model.transformer_weights.non_block_to_cuda_calls, 1)
        self.assertEqual(model.transformer_weights.resident_to_cuda_calls, 1)

        self.assertTrue(model.force_cleanup_offload_weights())
        self.assertFalse(model._offload_weights_loaded)
        self.assertEqual(model.transformer_infer.offload_manager_double.flush_calls, 1)
        self.assertEqual(model.transformer_infer.offload_manager_single.flush_calls, 1)
        self.assertEqual(model.sync_calls, 1)
        self.assertFalse(model.force_cleanup_offload_weights())

    def test_partial_prepare_failure_is_cleaned_and_original_error_is_preserved(self):
        model = _make_block_model(post_fail=True)

        with self.assertRaisesRegex(RuntimeError, "prepare failed"):
            model._prepare_infer_weights()

        self.assertFalse(model._offload_weights_loaded)
        self.assertFalse(model._offload_weights_preparing)
        self.assertEqual(model.pre_weight.release_calls, 1)
        self.assertEqual(model.post_weight.release_calls, 1)
        self.assertEqual(model.transformer_infer.offload_manager_double.flush_calls, 1)

    def test_model_granularity_prepare_is_idempotent_for_nonzero_step(self):
        model = _make_block_model(step_index=7)
        model.offload_granularity = "model"
        model.to_cuda_calls = 0
        model.to_cpu_calls = 0
        model.to_cuda = lambda: setattr(model, "to_cuda_calls", model.to_cuda_calls + 1)
        model.to_cpu = lambda: setattr(model, "to_cpu_calls", model.to_cpu_calls + 1)

        model._prepare_infer_weights()
        model._prepare_infer_weights()
        model.force_cleanup_offload_weights()

        self.assertEqual(model.to_cuda_calls, 1)
        self.assertEqual(model.to_cpu_calls, 1)


class _Scheduler:
    def __init__(self):
        self.infer_steps = 50
        self.latents = "latents"
        self.generator = "generator"
        self.steps = []

    def step_pre(self, step_index):
        self.steps.append(("pre", step_index))

    def step_post(self):
        self.steps.append(("post",))


class _RunnerModel:
    def __init__(self, infer_error=None):
        self.scheduler = _Scheduler()
        self.infer_error = infer_error
        self.infer_calls = 0
        self.cleanup_calls = 0

    def infer(self, inputs):
        self.infer_calls += 1
        if self.infer_error is not None:
            raise self.infer_error

    def force_cleanup_offload_weights(self):
        self.cleanup_calls += 1


def _make_runner(model):
    runner = object.__new__(runner_module.Flux2BaseRunner)
    runner.model = model
    runner.inputs = {"input": True}
    runner.progress_callback = None
    return runner


class Flux2RunnerLifecycleTest(unittest.TestCase):
    def test_short_run_forces_cleanup(self):
        model = _RunnerModel()
        runner = _make_runner(model)

        result = runner.run(total_steps=1)

        self.assertEqual(result, ("latents", "generator"))
        self.assertEqual(model.infer_calls, 1)
        self.assertEqual(model.cleanup_calls, 1)

    def test_infer_error_forces_cleanup_and_preserves_original_error(self):
        original_error = ValueError("infer failed")
        model = _RunnerModel(infer_error=original_error)
        runner = _make_runner(model)

        with self.assertRaisesRegex(ValueError, "infer failed") as caught:
            runner.run(total_steps=3)

        self.assertIs(caught.exception, original_error)
        self.assertEqual(model.cleanup_calls, 1)


if __name__ == "__main__":
    unittest.main()
