import pytest
import torch

from lightx2v_kernel.gemm import (
    cutlass_scaled_nvfp4_mm,
    cutlass_scaled_nvfp4_mm_split_n_stride,
    scaled_nvfp4_quant,
)


FLOAT4_E2M1_MAX = 6.0
FLOAT8_E4M3_MAX = torch.finfo(torch.float8_e4m3fn).max


@pytest.mark.parametrize("split_n_parts,bias_enabled", [(2, False), (4, True)])
@torch.inference_mode()
def test_split_n_stride_matches_full_gemm(split_n_parts, bias_enabled):
    m, n, k = 129, 512, 256
    activation = torch.randn((m, k), dtype=torch.bfloat16, device="cuda")
    weight = torch.randn((n, k), dtype=torch.bfloat16, device="cuda")
    bias = torch.randn((n,), dtype=torch.bfloat16, device="cuda") if bias_enabled else None

    activation_global_scale = ((FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX) / activation.abs().max()).float()
    weight_global_scale = ((FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX) / weight.abs().max()).float()
    alpha = 1.0 / (activation_global_scale * weight_global_scale)
    activation_fp4, activation_scale = scaled_nvfp4_quant(activation, activation_global_scale)
    weight_fp4, weight_scale = scaled_nvfp4_quant(weight, weight_global_scale)

    expected = cutlass_scaled_nvfp4_mm(
        activation_fp4,
        weight_fp4,
        activation_scale,
        weight_scale,
        alpha,
        bias,
    )
    actual = cutlass_scaled_nvfp4_mm_split_n_stride(
        activation_fp4,
        weight_fp4,
        activation_scale,
        weight_scale,
        alpha,
        bias,
        split_n_parts=split_n_parts,
    )

    torch.testing.assert_close(actual, expected, atol=1e-1, rtol=1e-1)
