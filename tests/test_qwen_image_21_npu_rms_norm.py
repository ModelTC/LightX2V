from types import SimpleNamespace

import torch

from lightx2v_platform.ops.norm.ascend_npu import qwen_image_21_npu_rms_norm as npu_rms_norm_module
from lightx2v_platform.ops.norm.ascend_npu.qwen_image_21_npu_rms_norm import QwenImage21NpuRmsNormWeight


def _reference(input_tensor, weight, eps):
    input_dtype = input_tensor.dtype
    variance = input_tensor.to(torch.float32).pow(2).mean(-1, keepdim=True)
    output = input_tensor * torch.rsqrt(variance + eps)
    if weight is not None:
        if weight.dtype in (torch.float16, torch.bfloat16):
            output = output.to(weight.dtype)
        output = output * weight
    return output.to(input_dtype)


def _norm(weight, eps=1e-6):
    norm = QwenImage21NpuRmsNormWeight("norm.weight", eps=eps)
    norm.weight = weight
    return norm


def test_cpu_fallback_matches_fp32_variance(monkeypatch):
    monkeypatch.setattr(npu_rms_norm_module, "torch_npu", None)
    input_tensor = torch.randn(17, 128, dtype=torch.bfloat16)
    weight = torch.randn(128, dtype=torch.bfloat16)
    norm = _norm(weight)

    output = norm.apply(input_tensor)

    torch.testing.assert_close(output, _reference(input_tensor, weight, norm.eps), rtol=0, atol=0)


def test_mixed_dtype_falls_back_when_native_rms_norm_is_available(monkeypatch):
    def fail_if_called(*args, **kwargs):
        raise AssertionError("Mixed-dtype tensors must not be passed to npu_rms_norm")

    monkeypatch.setattr(npu_rms_norm_module, "torch_npu", SimpleNamespace(npu_rms_norm=fail_if_called))
    monkeypatch.setattr(npu_rms_norm_module, "_is_npu_tensor", lambda tensor: True)
    input_tensor = torch.randn(17, 128, dtype=torch.float32)
    weight = torch.randn(128, dtype=torch.bfloat16)
    norm = _norm(weight)

    output = norm.apply(input_tensor)

    torch.testing.assert_close(output, _reference(input_tensor, weight, norm.eps), rtol=0, atol=0)


def test_npu_path_calls_native_rms_norm(monkeypatch):
    calls = []

    def fake_npu_rms_norm(input_tensor, weight, eps):
        calls.append((input_tensor, weight, eps))
        return _reference(input_tensor, weight, eps), torch.empty(0)

    monkeypatch.setattr(npu_rms_norm_module, "torch_npu", SimpleNamespace(npu_rms_norm=fake_npu_rms_norm))
    monkeypatch.setattr(npu_rms_norm_module, "_is_npu_tensor", lambda tensor: True)
    input_tensor = torch.randn(23, 128, dtype=torch.bfloat16)
    weight = torch.randn(128, dtype=torch.bfloat16)
    norm = _norm(weight, eps=1e-5)

    output = norm.apply(input_tensor)

    assert len(calls) == 1
    assert calls[0][0] is input_tensor
    torch.testing.assert_close(calls[0][1], torch.ones_like(weight), rtol=0, atol=0)
    assert calls[0][2] == norm.eps
    assert output.dtype == input_tensor.dtype
    torch.testing.assert_close(output, _reference(input_tensor, weight, norm.eps), rtol=0, atol=0)

    second_output = norm.apply(input_tensor)
    assert calls[1][1] is calls[0][1]
    torch.testing.assert_close(second_output, output, rtol=0, atol=0)


def test_weight_diff_is_applied_when_present(monkeypatch):
    monkeypatch.setattr(npu_rms_norm_module, "torch_npu", None)
    input_tensor = torch.randn(9, 64, dtype=torch.bfloat16)
    weight = torch.randn(64, dtype=torch.bfloat16)
    weight_diff = torch.randn(64, dtype=torch.float32).mul_(0.01)
    norm = _norm(weight)
    norm.weight_diff = weight_diff

    output = norm.apply(input_tensor)
    actual_weight = weight + weight_diff.to(torch.bfloat16)

    torch.testing.assert_close(output, _reference(input_tensor, actual_weight, norm.eps), rtol=0, atol=0)
