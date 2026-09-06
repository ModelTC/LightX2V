import importlib.util
import sys
import types
from pathlib import Path

import pytest
import torch


@pytest.fixture()
def h3_rope(monkeypatch):
    root = Path(__file__).parents[3]
    spec = importlib.util.spec_from_file_location("h3_rope_weight_helpers", Path(__file__).with_name("test_transformer_disk_streaming.py"))
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    _, weights, _ = helper.h3_modules.__wrapped__(monkeypatch)

    registry = sys.modules["lightx2v.utils.registry_factory"]
    rope_registry = registry.ROPE_REGISTER
    monkeypatch.setattr(registry, "ROPE_REGISTER", lambda name: lambda cls: cls)
    magi = types.ModuleType("lightx2v.common.magi_custom_op_mode")
    magi.use_magi_custom_ops = lambda: False
    monkeypatch.setitem(sys.modules, magi.__name__, magi)
    package = types.ModuleType("h3_precision_rope")
    package.__path__ = [str(root / "lightx2v/common/ops/rope")]
    monkeypatch.setitem(sys.modules, package.__name__, package)
    spec = importlib.util.spec_from_file_location(package.__name__ + ".torch_rope", Path(package.__path__[0]) / "torch_rope.py")
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    monkeypatch.setattr(registry, "ROPE_REGISTER", rope_registry)
    monkeypatch.setitem(rope_registry, "torch_real_rope", module.TorchRealRope)
    monkeypatch.setitem(weights.ROPE_REGISTER, "torch_real_rope", module.TorchRealRope)
    return weights.MiniMaxH3AttentionWeights("transformer_blocks.0.attn", {}).rope, module.TorchRealRope


def test_h3_precision_does_not_change_generic_default(h3_rope):
    rope, generic = h3_rope
    assert rope.layout == "split_half"
    assert rope.compute_dtype == torch.bfloat16
    assert generic().compute_dtype == torch.float32


@pytest.mark.parametrize("length", [1, 17, 52])
def test_h3_bf16_rope_matches_reference_exactly(h3_rope, length):
    rope, _ = h3_rope
    generator = torch.Generator().manual_seed(123)
    q, k = [torch.randn(length, 56, 128, generator=generator, dtype=torch.bfloat16) for _ in range(2)]
    angles = torch.randn(length, 48, generator=generator, dtype=torch.float32)
    angles = torch.cat((angles, angles), dim=-1)
    cos, sin = angles.cos(), angles.sin()

    def reference(x):
        rotary, passthrough = x[..., :96], x[..., 96:]
        first, second = rotary.chunk(2, dim=-1)
        rotated = torch.cat((-second, first), dim=-1)
        out = rotary * cos.to(x.dtype)[:, None, :] + rotated * sin.to(x.dtype)[:, None, :]
        return torch.cat((out, passthrough), dim=-1)

    actual = rope.apply(q, k, (cos, sin), rotary_dim=96)
    for x, out in zip((q, k), actual):
        assert out.shape == x.shape
        assert out.dtype == torch.bfloat16
        assert torch.equal(out, reference(x))
        assert torch.equal(out[..., 96:], x[..., 96:])
