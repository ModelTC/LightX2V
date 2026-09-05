import importlib.util
import sys
import types
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).parents[3]


@pytest.fixture
def modules(monkeypatch):
    # Keep geometry tests independent of accelerator initialization and media dependencies.
    for name in (
        "lightx2v",
        "lightx2v.models",
        "lightx2v.models.networks",
        "lightx2v.models.networks.minimax_h3",
        "lightx2v.models.schedulers",
        "lightx2v.models.schedulers.minimax_h3",
        "lightx2v_platform",
        "lightx2v_platform.base",
    ):
        module = types.ModuleType(name)
        module.__path__ = []
        monkeypatch.setitem(sys.modules, name, module)
    for name, attrs in (
        ("lightx2v.models.schedulers.scheduler", {"BaseScheduler": object}),
        ("lightx2v_platform.base.global_var", {"AI_DEVICE": "cpu"}),
        ("lightx2v.models.networks.minimax_h3.packing_ref2av", {"build_ref2av_packed_sequence": None}),
    ):
        module = types.ModuleType(name)
        module.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules, name, module)

    def load(name):
        spec = importlib.util.spec_from_file_location(name, REPO_ROOT / (name.replace(".", "/") + ".py"))
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, name, module)
        spec.loader.exec_module(module)
        return module

    return load("lightx2v.models.networks.minimax_h3.packing"), load("lightx2v.models.schedulers.minimax_h3.scheduler")


@pytest.mark.parametrize("anchors", [(), ("first", "last")])
@pytest.mark.parametrize("device", ["cpu", pytest.param("mps", marks=pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS unavailable"))])
def test_layout_device_boundary_preserves_geometry_and_metadata(modules, anchors, device):
    packing, scheduler = modules
    layout = packing.build_packed_sequence(torch.ones(1, dtype=torch.long), 37, 2, 2, 207, keyframe_anchors=anchors)
    assert layout.position_ids.device.type == "cpu"
    assert layout.position_ids.dtype == torch.float64
    expected = layout.position_ids.to(torch.float32)

    actual = scheduler._layout_to_device(layout, device)

    assert actual.position_ids.device.type == device
    assert actual.position_ids.dtype == torch.float32
    assert torch.isfinite(actual.position_ids).all()
    assert torch.equal(actual.position_ids.cpu(), expected)
    assert layout.position_ids.dtype == torch.float64
    for name in ("token_tags", "video_indices", "audio_indices", "text_indices"):
        value = getattr(actual, name)
        assert value.device.type == device
        assert value.dtype == torch.long
        assert torch.equal(value.cpu(), getattr(layout, name))
    for name in ("sequence_length", "num_condition_video_rows", "num_condition_audio_rows"):
        assert getattr(actual, name) == getattr(layout, name)
