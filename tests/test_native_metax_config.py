"""Protect existing platform admission while adding MetaX BF16 inference."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from lightx2v.models.runners import native_inference


@pytest.mark.parametrize("platform,device", [("cuda", "cuda"), ("cambricon_mlu", "mlu"), ("metax_cuda", "cuda")])
def test_supported_platform_admission(monkeypatch, platform, device):
    monkeypatch.setenv("PLATFORM", platform)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: True)
    monkeypatch.setattr(torch, "mlu", SimpleNamespace(is_available=lambda: True, is_bf16_supported=lambda: True), raising=False)
    monkeypatch.setattr(native_inference, "GET_DTYPE", lambda: torch.bfloat16)
    with patch.object(torch, "device", side_effect=lambda name: name):
        config = {"implementation_status": "implemented", "inference_dtype": "bf16"}
        assert native_inference.validate_a100_config(config) == device
        assert config["device"] == device


@pytest.mark.parametrize("platform", ["cuda", "metax_cuda"])
def test_bf16_hardware_admission(monkeypatch, platform):
    monkeypatch.setenv("PLATFORM", platform)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: False)
    monkeypatch.setattr(native_inference, "GET_DTYPE", lambda: torch.bfloat16)
    with pytest.raises(NotImplementedError, match="BF16"):
        native_inference.validate_a100_config({})


@pytest.mark.parametrize("platform", ["ascend_npu", "unknown"])
def test_unimplemented_platforms_remain_explicit(monkeypatch, platform):
    monkeypatch.setenv("PLATFORM", platform)
    with pytest.raises(RuntimeError, match="No available device"):
        native_inference.validate_a100_config({})


def test_metax_keeps_precision_and_feature_guards(monkeypatch):
    monkeypatch.setenv("PLATFORM", "metax_cuda")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: True)
    monkeypatch.setattr(native_inference, "GET_DTYPE", lambda: torch.bfloat16)
    for config, exception in [
        ({"implementation_status": "placeholder"}, NotImplementedError),
        ({"inference_dtype": "fp16"}, ValueError),
        ({"parallel": True}, NotImplementedError),
        ({"feature_caching": "cached"}, NotImplementedError),
    ]:
        with pytest.raises(exception):
            native_inference.validate_a100_config(config)
