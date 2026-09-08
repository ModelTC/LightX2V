from types import SimpleNamespace

import pytest
import torch

from lightx2v_platform.ops.attn.intel_xpu import xpu_sla_attn


def test_three_sycl_sla_apis_are_forwarded(monkeypatch):
    calls = []

    def record(name):
        def call(*args):
            calls.append((name, args))
            return args[0]

        return call

    monkeypatch.setattr(
        xpu_sla_attn,
        "_sycl_kernels",
        SimpleNamespace(
            sla_block_map=record("map"),
            sparse_block_attention=record("sparse"),
            sla_sparse_attention=record("combined"),
        ),
    )
    q = torch.empty(1, 9, 2, 4)
    lut = torch.empty(1, 2, 1, 1, dtype=torch.int32)

    xpu_sla_attn.sla_block_map(q, q, 0.15, 128, 128)
    xpu_sla_attn.sparse_block_attention(q, q, q, lut, 128, 128, None)
    xpu_sla_attn.sla_sparse_attention(q, q, q, 0.15, 128, 128, None)

    assert [name for name, _ in calls] == ["map", "sparse", "combined"]
    assert calls[0][1][2:] == (0.15, 128, 128)
    assert calls[2][1][3:] == (0.15, 128, 128, None)


def test_missing_sycl_sla_api_has_actionable_error(monkeypatch):
    monkeypatch.setattr(xpu_sla_attn, "_sycl_kernels", SimpleNamespace())
    with pytest.raises(RuntimeError, match="rebuild/install lightx2v_kernel_xpu"):
        xpu_sla_attn.sla_block_map(torch.empty(1), torch.empty(1))


def test_dynamic_sparse_attention_uses_xpu_combined_api(monkeypatch):
    from lightx2v.common.ops.attn.dynamic_sparse_attn import DynamicSparseAttnWeight

    def fail_if_cuda_is_queried(*args, **kwargs):
        raise AssertionError("the Intel XPU backend must not query CUDA")

    received = {}

    def fake_sla(q, k, v, keep_ratio, block_q, block_k, scale=None):
        received.update(
            shape=tuple(q.shape),
            keep_ratio=keep_ratio,
            block_q=block_q,
            block_k=block_k,
            scale=scale,
        )
        return q

    monkeypatch.setattr(torch.cuda, "current_device", fail_if_cuda_is_queried)
    monkeypatch.setattr(xpu_sla_attn, "sla_sparse_attention", fake_sla)
    attention = DynamicSparseAttnWeight({"operator": "intel_xpu", "sparsity_ratio": 0.85})
    q = torch.empty(17, 7, 128, dtype=torch.bfloat16)
    cu_seqlens = torch.tensor([0, 17], dtype=torch.int32)

    output = attention.apply(q, q, q, cu_seqlens, cu_seqlens, 17, 17)

    assert output.shape == (17, 7 * 128)
    assert received == {
        "shape": (1, 17, 7, 128),
        "keep_ratio": pytest.approx(0.15),
        "block_q": 128,
        "block_k": 128,
        "scale": None,
    }
