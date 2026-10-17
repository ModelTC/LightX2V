"""ProxyDINO dense backend dispatch preserves batched layout and graph ownership."""

import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import lightx2v.common.ops  # noqa: F401
from lightx2v.common.modules.weight_module import WeightModule
from lightx2v.models.input_encoders.edenlive.proxy_dino import ProxyDINO, _ProxyAttention


@pytest.mark.parametrize("backend", ["flash_attn2", "sage_attn2"])
def test_dense_backends_receive_batch_layout_without_varlen_metadata(monkeypatch, backend):
    calls = []

    def attention(q, k, v, **kwargs):
        calls.append((q, k, v, kwargs))
        return q + k + v

    if backend == "flash_attn2":
        monkeypatch.setitem(sys.modules, "flash_attn", SimpleNamespace(flash_attn_func=attention))
    else:
        monkeypatch.setitem(sys.modules, "sageattention", SimpleNamespace(sageattn=attention))
    operator = _ProxyAttention(backend)
    q, k, v = [torch.randn(3, 17, 16, 64) for _ in range(3)]
    actual = operator.apply(q, k, v)
    assert actual.shape == q.shape
    assert torch.equal(actual, q + k + v)
    assert calls[0][:3] == (q, k, v)
    assert calls[0][3] == ({"dropout_p": 0.0, "causal": False} if backend == "flash_attn2" else {"tensor_layout": "NHD", "is_causal": False})


def test_backend_switch_and_weight_load_drop_old_captures(monkeypatch):
    monkeypatch.setitem(sys.modules, "flash_attn", SimpleNamespace(flash_attn_func=lambda q, k, v, **kwargs: q))
    encoder = ProxyDINO.__new__(ProxyDINO)
    WeightModule.__init__(encoder)
    cleared = []
    encoder._cuda_graph = SimpleNamespace(clear=lambda: cleared.append(True))
    encoder.set_attention_type("torch_sdpa")
    encoder.set_attention_type("flash_attn2")
    assert encoder.attention_type == "flash_attn2"
    assert len(cleared) == 2
    encoder.set_attention_type("flash_attn2")
    assert len(cleared) == 2
    previous = encoder.attention
    with pytest.raises(ValueError, match="Unsupported ProxyDINO attention_type"):
        encoder.set_attention_type("sparse")
    assert encoder.attention is previous and len(cleared) == 2
    encoder.load({})
    assert len(cleared) == 3


@pytest.fixture(scope="module")
def real_encoder():
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    root = Path(os.environ.get("EDENLIVE_PROXY_TEST_MODEL", Path(__file__).resolve().parents[2] / "EdenLive-x2v"))
    if not (root / "vision-encoder/projection.safetensors").is_file():
        pytest.skip("Set EDENLIVE_PROXY_TEST_MODEL to an EdenLive model directory")
    encoder = ProxyDINO(str(root), "bilinear", use_cuda_graph=True)
    yield encoder
    encoder._cuda_graph.clear()


@pytest.mark.parametrize("backend", ["torch_sdpa", "flash_attn2", "sage_attn2"])
def test_real_backends_eager_and_graph_are_exact_and_switch_rewarms(real_encoder, backend):
    encoder = real_encoder
    encoder.set_attention_type(backend)
    generator = torch.Generator(device="cuda").manual_seed(20261016)
    images = torch.randint(0, 256, (3, 3, 704, 1280), device="cuda", dtype=torch.uint8, generator=generator)
    expected = encoder._infer_eager(images)
    retained = []
    for offset in range(4):
        current = images if offset < 3 else images.bitwise_xor(255)
        reference = expected if offset < 3 else encoder._infer_eager(current)
        actual = encoder.infer(current)
        assert torch.equal(actual.view(torch.int32), reference.view(torch.int32))
        assert actual.shape == (1, 1024, 3, 30, 52)
        assert torch.isfinite(actual).all()
        retained.append((actual, actual.clone()))
    assert next(iter(encoder._cuda_graph._entries.values())).graph is not None
    assert all(torch.equal(value.view(torch.int32), original.view(torch.int32)) for value, original in retained)
