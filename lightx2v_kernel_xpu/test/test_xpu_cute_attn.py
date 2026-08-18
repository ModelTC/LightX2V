import pytest
import torch

if not torch.xpu.is_available():
    pytest.skip("XPU is unavailable", allow_module_level=True)

from lightx2v_platform.ops.attn.intel_xpu import xpu_cute_attn


def test_4d_batch_without_cu_seqlens_is_processed_independently(monkeypatch):
    calls = []

    def fake_cute_sdp(q, k, v):
        calls.append((q.clone(), k.clone(), v.clone()))
        return q

    monkeypatch.setattr(xpu_cute_attn, "_cute_sdp", fake_cute_sdp)
    q = torch.arange(2 * 3 * 2 * 4).reshape(2, 3, 2, 4)

    actual = xpu_cute_attn.IntelXpuCuteAttnWeight().apply(q, q, q)

    assert len(calls) == 2
    assert all(call[0].shape == (1, 3, 2, 4) for call in calls)
    torch.testing.assert_close(calls[0][0], q[0].unsqueeze(0))
    torch.testing.assert_close(calls[1][0], q[1].unsqueeze(0))
    torch.testing.assert_close(actual, q.reshape(6, 8))
