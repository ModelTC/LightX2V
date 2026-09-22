"""AMD ROCm quantized GEMM correctness checks.

Run inside the ROCm container:
    PYTHONPATH=/path-to-LightX2V PLATFORM=amd_rocm python lightx2v_platform/test/test_amd_rocm_quant.py

What it checks:
- The fp8-rocm / int8-rocm scheme keys register (importing set_ai_device with
  aiter and sageattention absent must still succeed).
- Quantized GEMM output matches a BF16 matmul reference across several M,
  including the int8 M<=16 BF16-fallback boundary (M = 1, 16, 17, 31, 64).

fp8-rocm requires gfx1201 (RDNA4) and is skipped on other archs; int8-rocm runs
on gfx1100 / gfx1201.
"""

# isort: off
import lightx2v_platform.set_ai_device  # noqa: F401  must run first: initializes AI device + registers platform ops
import torch

from lightx2v_platform.registry_factory import PLATFORM_MM_WEIGHT_REGISTER

# isort: on


def _arch():
    try:
        return torch.cuda.get_device_properties(0).gcnArchName
    except Exception:
        return ""


def _rel_err(out, ref):
    return ((out.float() - ref.float()).abs().mean() / ref.float().abs().mean().clamp_min(1e-6)).item()


def _check_scheme(scheme, tol):
    n, k = 512, 1024
    weight = torch.randn(n, k, device="cuda", dtype=torch.bfloat16)
    linear = PLATFORM_MM_WEIGHT_REGISTER[scheme]("w.weight", None)
    linear.load({"w.weight": weight.clone()})
    ref_w = weight.float()
    for m in (1, 16, 17, 31, 64):
        x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
        ref = (x.float() @ ref_w.t()).to(torch.bfloat16)
        out = linear.apply(x)
        err = _rel_err(out, ref)
        assert err < tol, f"{scheme} M={m}: rel_err {err:.4f} >= tol {tol}"
        print(f"  {scheme:<10} M={m:>3}  rel_err={err:.4f}  OK")


def main():
    assert "fp8-rocm" in PLATFORM_MM_WEIGHT_REGISTER, "fp8-rocm not registered"
    assert "int8-rocm" in PLATFORM_MM_WEIGHT_REGISTER, "int8-rocm not registered"
    print("registration OK (no aiter / sageattention required)")

    if not torch.cuda.is_available():
        print("no GPU available; skipping numerical checks")
        return

    arch = _arch()
    print(f"arch: {arch}")
    if arch.startswith("gfx1201"):
        _check_scheme("fp8-rocm", tol=0.08)
    else:
        print("  fp8-rocm skipped (requires gfx1201)")
    _check_scheme("int8-rocm", tol=0.04)
    print("ALL PASS")


if __name__ == "__main__":
    main()
