"""LongCat enables pinned text offload without requiring shared DiT weights."""

import ast
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn


def isolated_encoder(device="cuda"):
    # Load real integration methods without initializing platform/model imports.
    path = Path(__file__).resolve().parents[1] / "lightx2v/models/input_encoders/hf/longcat/longcat_text_encoder.py"
    tree = ast.parse(path.read_text())
    encoder = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "LongCatImageTextEncoder")
    method_names = {"__init__", "load", "infer", "rewrite_prompt", "_rewrite_prompt", "_run_with_pinned_weights"}
    methods = [node for node in encoder.body if isinstance(node, ast.FunctionDef) and node.name in method_names]
    isolated = ast.Module(body=[ast.ClassDef(name="IsolatedEncoder", bases=[], keywords=[], body=methods, decorator_list=[])], type_ignores=[])
    events = []

    class FakeModel:
        def to(self, target):
            events.append(f"to:{target}")
            return self

    model = FakeModel()

    class FakeWeights:
        def __init__(self, module):
            assert module is model
            events.append("prepare")

        def activate(self, target):
            events.append(f"activate:{target}")

        def restore(self):
            events.append("restore")

    namespace = {
        "torch": torch,
        "os": os,
        "AI_DEVICE": device,
        "GET_DTYPE": lambda: torch.bfloat16,
        "get_prompt_language": lambda text: "en",
        "SYSTEM_PROMPT_EN": "Rewrite this prompt:",
        "SYSTEM_PROMPT_ZH": "Rewrite this prompt:",
        "Qwen2_5_VLForConditionalGeneration": SimpleNamespace(from_pretrained=lambda *args, **kwargs: model),
        "Qwen2Tokenizer": SimpleNamespace(from_pretrained=lambda *args, **kwargs: object()),
        "Qwen2VLProcessor": SimpleNamespace(from_pretrained=lambda *args, **kwargs: object()),
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
        ({"shared_cpu_weights": False}, "cuda", True),
        ({"enable_prompt_rewrite": True}, "cuda", True),
        ({"cpu_offload": False}, "cuda", False),
        ({"cpu_offload": False, "shared_cpu_weights": True}, "cuda", False),
        ({"task": "i2i"}, "cuda", False),
        ({"task": "t2i"}, "cuda", True),
        ({}, "cpu", False),
    ],
)
def test_longcat_pins_only_cuda_t2i_cpu_offload(overrides, device, enabled):
    encoder_type, events = isolated_encoder(device)
    # No task/shared flag: LongCat defaults to T2I and has no shared DiT adapter.
    config = {"model_path": "/unused/test-model", "cpu_offload": True, **overrides}

    encoder = encoder_type(config)

    assert (encoder._pinned_cpu_weights is not None) is enabled
    assert events.count("prepare") == int(enabled)


@pytest.mark.parametrize("operation", ["infer", "rewrite_prompt"])
@pytest.mark.parametrize("failure_stage", [None, "activate", "forward"])
def test_longcat_wrappers_restore_after_success_and_failure(operation, failure_stage):
    encoder_type, events = isolated_encoder()
    encoder = encoder_type.__new__(encoder_type)
    output = object()

    def activate(device):
        assert torch.device(device).type == "cuda"
        events.append("activate")
        if failure_stage == "activate":
            raise RuntimeError("injected activate failure")

    def forward(text, extra=None):
        assert text == ["a photo of a bench"]
        assert not torch.is_grad_enabled()
        if operation == "infer":
            assert extra is None
        else:
            assert extra is None or torch.device(extra).type == "cuda"
        events.append("forward")
        if failure_stage == "forward":
            raise RuntimeError("injected forward failure")
        return output

    encoder._pinned_cpu_weights = SimpleNamespace(activate=activate, restore=lambda: events.append("restore"))
    setattr(encoder, "_infer" if operation == "infer" else "_rewrite_prompt", forward)
    if failure_stage is None:
        assert getattr(encoder, operation)(["a photo of a bench"]) is output
    else:
        with pytest.raises(RuntimeError, match=f"injected {failure_stage} failure"):
            getattr(encoder, operation)(["a photo of a bench"])

    assert events == ["activate"] + ([] if failure_stage == "activate" else ["forward"]) + ["restore", "empty_cache", "gc"]


@pytest.mark.parametrize("operation", ["infer", "rewrite_prompt"])
def test_longcat_wrapper_preserves_existing_path_when_pinning_is_disabled(operation):
    encoder_type, events = isolated_encoder()
    encoder = encoder_type.__new__(encoder_type)
    encoder._pinned_cpu_weights = None
    output = object()

    def forward(text, extra=None):
        assert text == ["prompt"]
        assert not torch.is_grad_enabled()
        if operation == "infer":
            assert extra == ["unused image"]
        else:
            assert extra == "cpu"
        events.append("original")
        return output

    setattr(encoder, "_infer" if operation == "infer" else "_rewrite_prompt", forward)

    extra = ["unused image"] if operation == "infer" else "cpu"
    assert getattr(encoder, operation)(["prompt"], extra) is output
    assert events == ["original"]


def test_explicit_cpu_rewrite_keeps_pinned_cpu_sources_and_skips_cuda_activation():
    encoder_type, events = isolated_encoder()
    encoder = encoder_type.__new__(encoder_type)
    sources = {"weight": torch.arange(4)}
    original_source = sources["weight"]

    def forbidden(*args):
        raise AssertionError("An explicit CPU rewrite must not activate CUDA or release pinned CPU sources")

    weights = SimpleNamespace(activate=forbidden, restore=forbidden, sources=sources)
    encoder._pinned_cpu_weights = weights

    def rewrite(prompt, device=None):
        assert prompt == "draw a bench"
        assert torch.device(device).type == "cpu"
        assert not torch.is_grad_enabled()
        events.append("cpu_rewrite")
        return ["a photo of a bench"]

    encoder._rewrite_prompt = rewrite

    assert encoder.rewrite_prompt("draw a bench", device="cpu") == ["a photo of a bench"]
    assert events == ["cpu_rewrite"]
    assert encoder._pinned_cpu_weights is weights
    assert weights.sources["weight"] is original_source


def test_rewrite_honors_explicit_cuda_device():
    encoder_type, events = isolated_encoder()
    encoder = encoder_type.__new__(encoder_type)
    encoder._pinned_cpu_weights = SimpleNamespace(activate=lambda device: events.append(f"activate:{device}"), restore=lambda: events.append("restore"))

    def rewrite(prompt, device=None):
        assert str(device) == "cuda:1"
        events.append("rewrite")
        return ["rewritten"]

    encoder._rewrite_prompt = rewrite

    assert encoder.rewrite_prompt("prompt", device="cuda:1") == ["rewritten"]
    assert events == ["activate:cuda:1", "rewrite", "restore", "empty_cache", "gc"]


def test_actual_cpu_rewrite_preserves_source_storage_through_original_cpu_transfers(monkeypatch):
    encoder_type, events = isolated_encoder()
    encoder = encoder_type.__new__(encoder_type)

    class CPUModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.arange(4.0), requires_grad=False)

        def to(self, device):
            assert torch.device(device).type == "cpu"
            events.append("model:cpu")
            return super().to(device)

        def generate(self, input_ids, max_new_tokens):
            assert not torch.is_grad_enabled()
            assert max_new_tokens == 512
            return torch.cat((input_ids, torch.tensor([[9]])), dim=1)

    class Inputs(dict):
        def __init__(self):
            super().__init__(input_ids=torch.tensor([[1, 2]]))
            self.input_ids = self["input_ids"]

        def to(self, device):
            assert torch.device(device).type == "cpu"
            events.append("inputs:cpu")
            return self

    class Processor:
        def apply_chat_template(self, message, **kwargs):
            return "rewrite"

        def __call__(self, **kwargs):
            return Inputs()

        def batch_decode(self, tokens, **kwargs):
            assert len(tokens) == 1 and torch.equal(tokens[0], torch.tensor([9]))
            return ["a photo of a bench"]

    model = CPUModel().eval()
    source = model.weight.detach()
    original_storage = source.untyped_storage()._cdata
    # Mark this CPU source as pinned without calling a CUDA host allocator.
    monkeypatch.setattr(torch.Tensor, "is_pinned", lambda tensor, *args, **kwargs: tensor.untyped_storage()._cdata == original_storage)

    def forbidden(*args):
        raise AssertionError("An explicit CPU rewrite must retain CPU sources without a CUDA activation")

    weights = SimpleNamespace(sources={"weight": source}, activate=forbidden, restore=forbidden)
    encoder._pinned_cpu_weights = weights
    encoder.text_encoder = model
    encoder.processor = Processor()
    encoder.cpu_offload = True
    encoder.tokenizer_max_length = 512

    assert encoder.rewrite_prompt("draw a bench", device=torch.device("cpu")) == ["a photo of a bench"]
    assert model.weight.data_ptr() == source.data_ptr()
    assert model.weight.is_pinned()
    assert weights.sources["weight"] is source
    assert events == ["inputs:cpu", "model:cpu", "model:cpu", "empty_cache", "gc"]
