"""CPU-only checks for immutable Qwen2.5-VL offload sources and recovery."""

import ast
import importlib.util
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn


@pytest.fixture
def adapter():
    # Import the leaf file without initializing LightX2V's device/model registry.
    path = Path(__file__).resolve().parents[1] / "lightx2v/common/offload/module_adapter.py"
    spec = importlib.util.spec_from_file_location("_qwen25_test_module_adapter", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def fake_pinning(monkeypatch):
    pinned_storages = set()
    allocations = []

    def pin(tensor, *args, **kwargs):
        copy = torch.empty_strided(tensor.shape, tensor.stride(), dtype=tensor.dtype)
        copy.copy_(tensor)
        pinned_storages.add(copy.untyped_storage()._cdata)
        allocations.append(copy)
        return copy

    def is_pinned(tensor, *args, **kwargs):
        return tensor.untyped_storage()._cdata in pinned_storages

    monkeypatch.setattr(torch.Tensor, "pin_memory", pin)
    monkeypatch.setattr(torch.Tensor, "is_pinned", is_pinned)
    return allocations


@pytest.fixture
def cuda_sync(monkeypatch):
    devices = []
    monkeypatch.setattr(torch.cuda, "synchronize", devices.append)
    return devices


class FakeDeviceModule(nn.Module):
    """Replace tensors on upload, as _apply may, without allocating on a GPU."""

    def __init__(self):
        super().__init__()
        storage = torch.arange(24, dtype=torch.float32).reshape(4, 6)
        self.weight = nn.Parameter(storage, requires_grad=False)
        self.tied = self.weight
        self.transpose = nn.Parameter(storage.t(), requires_grad=False)
        self.slice = nn.Parameter(storage[1:, 1:5:2], requires_grad=False)
        self.independent = nn.Parameter(torch.arange(12, dtype=torch.float32).reshape(3, 4).t(), requires_grad=False)
        self.register_buffer("counter", torch.tensor(1.0), persistent=False)
        self.uploads = []
        self.fail_after = None
        self.eval()

    def to(self, *args, **kwargs):
        device = torch.device(kwargs.get("device", args[0] if args else "cpu"))
        assert device.type == "cuda", "Immutable parameters must be restored without a module.to(cpu) copy"
        self.uploads.append(kwargs)
        replaced = 0
        for child in self.modules():
            for name, parameter in tuple(child._parameters.items()):
                if parameter is None:
                    continue
                child._parameters[name] = nn.Parameter(parameter.detach().clone(), requires_grad=parameter.requires_grad)
                replaced += 1
                if replaced == self.fail_after:
                    raise RuntimeError("injected partial H2D failure")
            for name, buffer in tuple(child._buffers.items()):
                if buffer is not None:
                    child._buffers[name] = buffer.clone()
        return self


def parameter_snapshot(model):
    return {name: (parameter, parameter.detach().clone(), parameter.shape, parameter.stride(), parameter.storage_offset()) for name, parameter in model.named_parameters(remove_duplicate=False)}


def assert_restored(model, snapshot):
    for name, (original, value, shape, stride, offset) in snapshot.items():
        parameter = model.get_parameter(name)
        assert parameter is original
        assert torch.equal(parameter, value)
        assert parameter.shape == shape
        assert parameter.stride() == stride
        assert parameter.storage_offset() == offset
        assert parameter.device.type == "cpu"
    assert model.weight is model.tied
    assert model.weight.untyped_storage()._cdata == model.transpose.untyped_storage()._cdata
    assert model.weight.untyped_storage()._cdata == model.slice.untyped_storage()._cdata


def test_pin_preserves_parameter_objects_aliases_and_noncontiguous_views(adapter, fake_pinning):
    model = FakeDeviceModule()
    snapshot = parameter_snapshot(model)

    adapter.PinnedModuleCPUWeights(model)

    assert_restored(model, snapshot)
    assert all(parameter.is_pinned() for parameter in model.parameters())
    # Three distinct Parameter views and a tied name share one allocation.
    assert len(fake_pinning) == 2


def test_repeated_uploads_reuse_cpu_sources_without_copying_device_weights_back(adapter, fake_pinning, cuda_sync):
    model = FakeDeviceModule()
    snapshot = parameter_snapshot(model)
    weights = adapter.PinnedModuleCPUWeights(model)
    backing = {name: parameter.data_ptr() for name, parameter in model.named_parameters(remove_duplicate=False)}

    for _ in range(3):
        try:
            weights.activate("cuda:0")
            # Corrupt only the temporary device-side copy; CPU sources are immutable.
            model.weight.data.fill_(-100)
        finally:
            weights.restore()
        assert_restored(model, snapshot)
        assert {name: parameter.data_ptr() for name, parameter in model.named_parameters(remove_duplicate=False)} == backing

    assert len(fake_pinning) == 2
    assert len(model.uploads) == 3
    assert all(call["non_blocking"] for call in model.uploads)
    assert len(cuda_sync) >= 3


def test_restore_retains_updated_and_new_nonpersistent_buffers(adapter, fake_pinning, cuda_sync):
    model = FakeDeviceModule()
    weights = adapter.PinnedModuleCPUWeights(model)

    for value in (7.0, 11.0):
        try:
            weights.activate("cuda:0")
            model.counter.add_(2)
            model.register_buffer("runtime", torch.tensor(value), persistent=False)
        finally:
            weights.restore()
        assert model.runtime.item() == value
        assert model.counter.device.type == model.runtime.device.type == "cpu"

    assert model.counter.item() == 5.0
    assert "counter" not in model.state_dict()
    assert "runtime" not in model.state_dict()


def test_partial_upload_failure_can_restore_all_parameter_aliases(adapter, fake_pinning, cuda_sync):
    model = FakeDeviceModule()
    snapshot = parameter_snapshot(model)
    weights = adapter.PinnedModuleCPUWeights(model)
    model.fail_after = 2

    with pytest.raises(RuntimeError, match="partial H2D"):
        try:
            weights.activate("cuda:0")
        finally:
            weights.restore()

    assert_restored(model, snapshot)
    assert cuda_sync
    # Recovery must also permit the next request to run successfully.
    model.fail_after = None
    try:
        weights.activate("cuda:0")
    finally:
        weights.restore()
    assert_restored(model, snapshot)


def test_forward_failure_keeps_latest_buffers_and_cpu_sources(adapter, fake_pinning, cuda_sync):
    model = FakeDeviceModule()
    snapshot = parameter_snapshot(model)
    weights = adapter.PinnedModuleCPUWeights(model)

    with pytest.raises(RuntimeError, match="forward failure"):
        try:
            weights.activate("cuda:0")
            model.counter.add_(3)
            model.register_buffer("runtime", torch.tensor(9.0), persistent=False)
            raise RuntimeError("injected forward failure")
        finally:
            weights.restore()

    assert_restored(model, snapshot)
    assert model.counter.item() == 4.0
    assert model.runtime.item() == 9.0


def test_pin_failure_preserves_pageable_sources_and_uses_blocking_upload(adapter, monkeypatch, cuda_sync):
    model = FakeDeviceModule()
    snapshot = parameter_snapshot(model)

    def fail_pin(*args, **kwargs):
        raise RuntimeError("injected host allocation failure")

    monkeypatch.setattr(torch.Tensor, "pin_memory", fail_pin)
    monkeypatch.setattr(torch.Tensor, "is_pinned", lambda tensor, *args, **kwargs: False)
    weights = adapter.PinnedModuleCPUWeights(model)
    assert_restored(model, snapshot)

    try:
        weights.activate("cuda:0")
    finally:
        weights.restore()

    assert_restored(model, snapshot)
    assert model.uploads[0]["non_blocking"] is False


def test_training_module_is_rejected(adapter, fake_pinning):
    with pytest.raises(ValueError, match="eval|inference|training"):
        adapter.PinnedModuleCPUWeights(FakeDeviceModule().train())


def isolated_encoder(device="cuda"):
    """Exercise real integration methods without importing model dependencies."""
    path = Path(__file__).resolve().parents[1] / "lightx2v/models/input_encoders/hf/qwen25/qwen25_vlforconditionalgeneration.py"
    tree = ast.parse(path.read_text())
    encoder = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "Qwen25_VLForConditionalGeneration_TextEncoder")
    methods = [node for node in encoder.body if isinstance(node, ast.FunctionDef) and node.name in {"__init__", "load", "infer"}]
    isolated = ast.Module(body=[ast.ClassDef(name="IsolatedEncoder", bases=[], keywords=[], body=methods, decorator_list=[])], type_ignores=[])
    events = []
    model = FakeDeviceModule()

    class FakeWeights:
        def __init__(self, module):
            assert module is model
            events.append("prepare")

        def activate(self, target):
            assert target == device
            events.append("activate")

        def restore(self):
            events.append("restore")

    source = SimpleNamespace(from_pretrained=lambda *args, **kwargs: model)
    namespace = {
        "torch": torch,
        "os": os,
        "AI_DEVICE": device,
        "Qwen2_5_VLForConditionalGeneration": source,
        "Qwen2Tokenizer": SimpleNamespace(from_pretrained=lambda *args, **kwargs: object()),
        "Qwen2VLProcessor": SimpleNamespace(from_pretrained=lambda *args, **kwargs: object()),
        "VaeImageProcessor": lambda *args, **kwargs: object(),
        "PinnedModuleCPUWeights": FakeWeights,
        "torch_device_module": SimpleNamespace(empty_cache=lambda: events.append("empty_cache")),
        "gc": SimpleNamespace(collect=lambda: events.append("gc")),
    }
    exec(compile(ast.fix_missing_locations(isolated), str(path), "exec"), namespace)
    return namespace["IsolatedEncoder"], events


@pytest.mark.parametrize(
    "overrides,device,enabled",
    [
        ({}, "cuda", True),
        ({"cpu_offload": False}, "cuda", False),
        ({"shared_cpu_weights": False}, "cuda", False),
        ({"qwen25vl_cpu_offload": False}, "cuda", False),
        ({"qwen25vl_cpu_offload": True}, "cuda", True),
        ({"cpu_offload": False, "qwen25vl_cpu_offload": True}, "cuda", False),
        ({"qwen25vl_quantized": True}, "cuda", False),
        ({"task": "i2i"}, "cuda", False),
        ({"layered": True}, "cuda", False),
        ({}, "cpu", False),
    ],
)
def test_qwen25_pinning_requires_supported_shared_offload_configuration(overrides, device, enabled):
    encoder_type, events = isolated_encoder(device)
    config = {
        "model_path": "/unused/test-model",
        "task": "t2i",
        "cpu_offload": True,
        "shared_cpu_weights": True,
        "prompt_template_encode": "{}",
        "prompt_template_encode_start_idx": 0,
        "qwen25vl_quant_scheme": "int4",
        "qwen25vl_quantized_ckpt": "/unused/quantized",
        "vae_scale_factor": 8,
        "use_en_prompt": False,
        **overrides,
    }

    encoder = encoder_type(config)

    assert (encoder._pinned_cpu_weights is not None) is enabled
    assert events == (["prepare"] if enabled else [])


@pytest.mark.parametrize("failure_stage", [None, "activate", "infer"])
def test_qwen25_wrapper_restores_after_success_or_partial_failure(failure_stage):
    encoder_type, events = isolated_encoder()
    encoder = encoder_type.__new__(encoder_type)
    output = object()

    def activate(device):
        events.append("activate")
        if failure_stage == "activate":
            raise RuntimeError("injected activate failure")

    def infer(text, image_list):
        assert text == ["a photo of a bench"]
        assert image_list is None
        assert not torch.is_grad_enabled()
        events.append("infer")
        if failure_stage == "infer":
            raise RuntimeError("injected infer failure")
        return output

    encoder._pinned_cpu_weights = SimpleNamespace(activate=activate, restore=lambda: events.append("restore"))
    encoder._infer = infer
    if failure_stage is None:
        assert encoder.infer(["a photo of a bench"]) is output
    else:
        with pytest.raises(RuntimeError, match=f"injected {failure_stage} failure"):
            encoder.infer(["a photo of a bench"])

    assert events == ["activate"] + ([] if failure_stage == "activate" else ["infer"]) + ["restore", "empty_cache", "gc"]


def test_qwen25_wrapper_keeps_existing_path_without_pinned_sources():
    encoder_type, events = isolated_encoder()
    encoder = encoder_type.__new__(encoder_type)
    encoder._pinned_cpu_weights = None
    output = object()

    def infer(text, image_list):
        assert not torch.is_grad_enabled()
        assert image_list == ["image"]
        events.append("original_infer")
        return output

    encoder._infer = infer

    assert encoder.infer(["prompt"], ["image"]) is output
    assert events == ["original_infer"]
