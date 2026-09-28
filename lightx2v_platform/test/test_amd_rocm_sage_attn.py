"""CPU checks for the AMD SageAttention2 wrapper; the GPU kernel is mocked."""

import importlib
import os
import sys
from types import ModuleType
from unittest.mock import Mock

import pytest
import torch
import torch.nn.functional as F


@pytest.fixture
def sage_module():
    if os.getenv("PLATFORM") == "amd_rocm":
        # Follow production initialization before platform registries are read.
        import lightx2v  # noqa: F401

    return importlib.import_module("lightx2v_platform.ops.attn.amd_rocm.sage_attn")


@pytest.fixture
def backend(sage_module, monkeypatch):
    def sageattn(q, k, v, tensor_layout, is_causal=False, sm_scale=None, return_lse=False):
        assert tensor_layout == "NHD"
        assert q.ndim == k.ndim == v.ndim == 4
        assert all(x.is_contiguous() for x in (q, k, v))
        q, k, v = (x.transpose(1, 2) for x in (q, k, v))
        output = F.scaled_dot_product_attention(q, k, v, is_causal=is_causal, scale=sm_scale).transpose(1, 2)
        if return_lse:
            scale = q.shape[-1] ** -0.5 if sm_scale is None else sm_scale
            lse = ((q @ k.transpose(-2, -1)) * scale).logsumexp(-1)
            return output, lse
        return output

    module = ModuleType("sageattention")
    module.sageattn = sageattn
    monkeypatch.setitem(sys.modules, "sageattention", module)
    monkeypatch.setattr(torch.version, "hip", "test")
    patches = Mock()
    monkeypatch.setattr(sage_module, "apply_rocm_sage_patches", patches)
    instance = sage_module.AmdSageAttn2Weight()
    patches.assert_called_once_with()
    return instance


def test_dedicated_registration(sage_module):
    from lightx2v_platform.registry_factory import PLATFORM_ATTN_WEIGHT_REGISTER

    assert PLATFORM_ATTN_WEIGHT_REGISTER["sage_attn2_amd_rocm"] is sage_module.AmdSageAttn2Weight
    assert "sage_attn2" not in PLATFORM_ATTN_WEIGHT_REGISTER


def test_rejects_non_rocm(sage_module, monkeypatch):
    monkeypatch.setattr(torch.version, "hip", None)
    with pytest.raises(RuntimeError, match="requires AMD ROCm"):
        sage_module.AmdSageAttn2Weight()


def test_sageattention_required_only_at_construction(sage_module, monkeypatch):
    monkeypatch.setattr(torch.version, "hip", "test")
    monkeypatch.setitem(sys.modules, "sageattention", None)
    patches = Mock()
    monkeypatch.setattr(sage_module, "apply_rocm_sage_patches", patches)
    with pytest.raises(ImportError):
        sage_module.AmdSageAttn2Weight()
    patches.assert_not_called()


@pytest.mark.parametrize("batched", [False, True])
@pytest.mark.parametrize("with_lse", [False, True])
def test_attention_layout_and_lse(backend, sage_module, batched, with_lse):
    torch.manual_seed(42)
    batch, heads, width = 2 if batched else 1, 3, 8
    # Unequal query/KV lengths cover prefix-cache attention; transpose also
    # verifies that the wrapper makes noncontiguous inputs contiguous.
    q, k, v = [torch.randn(batch, heads, length, width).transpose(1, 2) for length in (5, 9, 9)]
    scale = 0.25 if with_lse else None
    reference = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), scale=scale)
    reference = reference.transpose(1, 2).reshape(batch * 5, heads * width)
    inputs = (q, k, v) if batched else (q[0], k[0], v[0])
    if with_lse:
        output, lse = backend.apply_with_lse(*inputs, softmax_scale=scale)
        expected_lse = ((q.transpose(1, 2) @ k.transpose(1, 2).transpose(-2, -1)) * scale).logsumexp(-1)
        torch.testing.assert_close(lse, expected_lse.transpose(1, 2).reshape(batch * 5, heads))
    else:
        output = backend.apply(*inputs)
    torch.testing.assert_close(output, reference)
    sage_module.apply_rocm_sage_patches.assert_called_once_with()


def test_invalid_input_rank(backend):
    x = torch.zeros(5, 8)
    with pytest.raises(ValueError, match="expects 3D or 4D"):
        backend.apply(x, x, x)


@pytest.mark.parametrize("max_stages,validated,expected_calls", [(0, True, 0), (2, False, 0), (2, True, 1)])
def test_workaround_gating(sage_module, monkeypatch, max_stages, validated, expected_calls):
    monkeypatch.setattr(sage_module, "_MAX_STAGES", max_stages)
    monkeypatch.setattr(sage_module, "_is_validated_rocm_arch", lambda: validated)
    patches = [Mock(), Mock(), Mock()]
    for name, patch in zip(("_clamp_sageattn_kernels", "_clamp_triton_compile", "_force_inductor_single_thread"), patches):
        monkeypatch.setattr(sage_module, name, patch)
    sage_module.apply_rocm_sage_patches()
    assert all(patch.call_count == expected_calls for patch in patches)
