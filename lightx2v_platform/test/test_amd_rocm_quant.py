"""AMD ROCm quantized GEMM correctness checks.

Run inside the ROCm container (a ROCm GPU and PLATFORM=amd_rocm are required;
otherwise every test is skipped, so this stays safe to collect on CPU CI):

    PYTHONPATH=/path-to-LightX2V PLATFORM=amd_rocm python -m pytest \\
        lightx2v_platform/test/test_amd_rocm_quant.py -v

It checks that fp8-rocm / int8-rocm register into the production
``MM_WEIGHT_REGISTER`` (the platform ops must merge into the same registry
inference looks them up in), and that their GEMM matches a BF16 reference across
several M — including the int8 M<=16 BF16-fallback boundary. These torch-native
ops do not import aiter or sageattention, so registration works without them.
"""

import os

import pytest
import torch

_ON_ROCM = getattr(torch.version, "hip", None) is not None and os.getenv("PLATFORM") == "amd_rocm" and torch.cuda.is_available()

pytestmark = pytest.mark.skipif(not _ON_ROCM, reason="requires PLATFORM=amd_rocm on a ROCm GPU")


def _gcn_arch():
    try:
        return torch.cuda.get_device_properties(torch.cuda.current_device()).gcnArchName
    except Exception:
        return ""


def _mm_register():
    # Import via the lightx2v package (lazily, so collecting this module on a
    # non-ROCm host never fails): lightx2v/__init__ runs set_ai_device before
    # registry_factory takes its one-time platform snapshot — the same order
    # production uses, so the platform schemes are merged into MM_WEIGHT_REGISTER.
    from lightx2v.utils.registry_factory import MM_WEIGHT_REGISTER

    return MM_WEIGHT_REGISTER


def _rel_err(out, ref):
    return ((out.float() - ref.float()).abs().mean() / ref.float().abs().mean().clamp_min(1e-6)).item()


def _check(scheme, tol):
    torch.manual_seed(42)
    register = _mm_register()
    assert scheme in register, f"{scheme} not merged into the production MM_WEIGHT_REGISTER"
    n, k = 512, 1024
    weight = torch.randn(n, k, device="cuda", dtype=torch.bfloat16)
    linear = register[scheme]("w.weight", None)
    linear.load({"w.weight": weight.clone()})
    ref_w = weight.float()
    for m in (1, 16, 17, 31, 64):
        x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
        ref = (x.float() @ ref_w.t()).to(torch.bfloat16)
        err = _rel_err(linear.apply(x), ref)
        assert err < tol, f"{scheme} M={m}: rel_err {err:.4f} >= tol {tol}"


def test_schemes_registered():
    register = _mm_register()
    assert "fp8-rocm" in register
    assert "int8-rocm" in register


def test_int8_rocm():
    _check("int8-rocm", tol=0.04)


@pytest.mark.skipif(not _gcn_arch().startswith("gfx1201"), reason="fp8-rocm requires gfx1201 (RDNA4)")
def test_fp8_rocm():
    _check("fp8-rocm", tol=0.08)
