import importlib.util
import sys
import types
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F


@pytest.fixture()
def sdpa(monkeypatch):
    root = Path(__file__).parents[3]
    registry = types.ModuleType("lightx2v.utils.registry_factory")
    registry.ATTN_WEIGHT_REGISTER = lambda name: lambda cls: cls
    monkeypatch.setitem(sys.modules, registry.__name__, registry)
    package = types.ModuleType("chunked_sdpa_test")
    package.__path__ = [str(root / "lightx2v/common/ops/attn")]
    monkeypatch.setitem(sys.modules, package.__name__, package)
    name = package.__name__ + ".torch_sdpa"
    spec = importlib.util.spec_from_file_location(name, root / "lightx2v/common/ops/attn/torch_sdpa.py")
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("length", [16, 19])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_dense_numerical_parity(sdpa, length, dtype):
    torch.manual_seed(123)
    q, k, v = [torch.randn(1, 4, length, 8, dtype=dtype) for _ in range(3)]
    expected = F.scaled_dot_product_attention(q, k, v)
    actual = sdpa._query_chunked_sdpa(q, k, v, 8)
    assert actual.shape == expected.shape and actual.dtype == dtype
    tolerance = 2e-2 if dtype == torch.bfloat16 else 1e-5
    torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)


@pytest.mark.parametrize(
    "change",
    [
        {},
        {"chunk_size": 0},
        {"scope": None},
        {"scope": "video_vae"},
        {"attn_mask": torch.ones(1)},
        {"causal": True},
        {"drop_rate": 0.1},
        {"device": "cpu"},
        {"device": "cuda"},
        {"shape": (1, 8, 19, 128)},
    ],
)
def test_applicability(sdpa, change):
    args = {"chunk_size": 8, "scope": "minimax_h3_dit", "attn_mask": None, "causal": False, "drop_rate": 0}
    device = change.get("device", "mps")
    shape = change.get("shape", (1, 56, 19, 128))
    tensor = types.SimpleNamespace(device=torch.device(device), ndim=4, shape=shape)
    args.update({k: v for k, v in change.items() if k not in ("device", "shape")})
    assert sdpa._use_h3_mps_query_chunks(tensor, tensor, tensor, **args) == (not change)


@pytest.mark.parametrize("options", [{}, {"attn_mask": torch.ones(9, 9, dtype=torch.bool)}, {"causal": True}, {"drop_rate": 0.2}])
def test_original_path_preserved_on_cpu(sdpa, monkeypatch, options):
    q, k, v = [torch.randn(9, 4, 8) for _ in range(3)]
    calls = []
    original = F.scaled_dot_product_attention

    def record(*args, **kwargs):
        calls.append(kwargs)
        return original(*args, **kwargs)

    monkeypatch.setattr(sdpa.F, "scaled_dot_product_attention", record)
    result = sdpa.TorchSDPAWeight().apply(q, k, v, attention_scope="minimax_h3_dit", mps_sdpa_query_chunk_size=512, **options)
    assert result.shape == (9, 32) and result.dtype == q.dtype
    assert len(calls) == 1
    assert calls[0]["is_causal"] == options.get("causal", False)
    assert calls[0]["dropout_p"] == options.get("drop_rate", 0)
    assert calls[0]["attn_mask"] is options.get("attn_mask")


def test_query_chunks_keep_complete_key_value_context(sdpa, monkeypatch):
    q, k, v = [torch.randn(1, 4, 19, 8) for _ in range(3)]
    lengths = []
    original = F.scaled_dot_product_attention

    def record(query, key, value, **kwargs):
        assert key is k and value is v
        assert kwargs == {"attn_mask": None, "dropout_p": 0.0, "is_causal": False}
        lengths.append(query.shape[2])
        return original(query, key, value, **kwargs)

    monkeypatch.setattr(sdpa.F, "scaled_dot_product_attention", record)
    assert sdpa._query_chunked_sdpa(q, k, v, 8).shape == q.shape
    assert lengths == [8, 8, 3]
