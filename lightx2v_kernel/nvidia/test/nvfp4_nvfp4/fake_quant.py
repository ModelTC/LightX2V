import torch


BLOCK_SIZE = 16

FLOAT4_E2M1_MAX = 6.0
FLOAT8_E4M3_MAX = torch.finfo(torch.float8_e4m3fn).max


def cast_to_fp4(x):
    sign = torch.sign(x)
    x = torch.abs(x)
    x[(x >= 0.0) & (x <= 0.25)] = 0.0
    x[(x > 0.25) & (x < 0.75)] = 0.5
    x[(x >= 0.75) & (x <= 1.25)] = 1.0
    x[(x > 1.25) & (x < 1.75)] = 1.5
    x[(x >= 1.75) & (x <= 2.5)] = 2.0
    x[(x > 2.5) & (x < 3.5)] = 3.0
    x[(x >= 3.5) & (x <= 5.0)] = 4.0
    x[x > 5.0] = 6.0
    return x * sign


def get_reciprocal(x):
    if isinstance(x, torch.Tensor):
        return torch.where(x == 0, torch.tensor(0.0, dtype=x.dtype), 1.0 / x)
    elif isinstance(x, (float, int)):
        return 0.0 if x == 0 else 1.0 / x
    else:
        raise TypeError("Input must be a float, int, or a torch.Tensor.")


def ref_nvfp4_quant(x, global_scale):
    assert global_scale.dtype == torch.float32
    assert x.ndim == 2
    m, n = x.shape
    x = torch.reshape(x, (m, n // BLOCK_SIZE, BLOCK_SIZE))
    vec_max = torch.max(torch.abs(x), dim=-1, keepdim=True)[0].to(torch.float32)
    scale = global_scale * (vec_max * get_reciprocal(FLOAT4_E2M1_MAX))
    scale = scale.to(torch.float8_e4m3fn).to(torch.float32)
    # output_scale = get_reciprocal(scale * get_reciprocal(global_scale))
    output_scale = global_scale * get_reciprocal(scale)

    scaled_x = x.to(torch.float32) * output_scale
    clipped_x = torch.clamp(scaled_x, -6.0, 6.0).reshape(m, n)
    return cast_to_fp4(clipped_x), scale.squeeze(-1)


if __name__ == "__main__":
    x = torch.randn(1, 16, dtype=torch.bfloat16).cuda()
    print(f"x: {x}, {x.shape}")
    global_scale = (6.0 * 448.0 / torch.max(torch.abs(x))).to(torch.float32).cuda()
    quant_x, scale = ref_nvfp4_quant(x, global_scale)
    print(f"quant_x: {quant_x}, {quant_x.shape}")
    print(f"scale: {scale}, {scale.shape}")
