import torch
from lightx2v_kernel.gemm import cutlass_scaled_mxfp6_mxfp8_mm


def test_cutlass_scaled_mxfp6_mxfp8_mm_sm120():
    m, k, n = 1024, 2048, 4096

    input_shape = (m, k)
    weight_shape = (n, k)

    input_tensor_quant = (torch.rand((input_shape[0], input_shape[1]), device="cuda") * 10).to(torch.uint8)
    weight = (torch.rand((weight_shape[0], weight_shape[1] * 3 // 4), device="cuda") * 10).to(torch.uint8)

    print(f"shape: {input_tensor_quant.shape}, {weight.shape}")

    input_tensor_scale = torch.rand((input_shape[0], input_shape[1] // 32), device="cuda").to(torch.float8_e8m0fnu)
    weight_scale = torch.rand(weight_shape[0], weight_shape[1] // 32, device="cuda").to(torch.float8_e8m0fnu)

    print(f"shape: {input_tensor_scale.shape}, {weight_scale.shape}")

    alpha = torch.tensor(0.0002765655517578125, device="cuda", dtype=torch.float32)
    bias = None

    out = cutlass_scaled_mxfp6_mxfp8_mm(input_tensor_quant, weight, input_tensor_scale, weight_scale, alpha, bias)
    print(f"out: {out}, shape: {out.shape}")


if __name__ == "__main__":
    test_cutlass_scaled_mxfp6_mxfp8_mm_sm120()
