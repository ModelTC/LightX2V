import torch
from lightx2v_kernel.gemm import cutlass_scaled_nvfp4_mm


"""
input_shape = (1024, 2048)
weight_shape = (4096, 2048)

input_tensor_quant = (torch.rand((1024, 1024), device="cuda") * 10).to(torch.uint8)
weight = (torch.rand((4096, 1024), device="cuda") * 10).to(torch.uint8)
input_tensor_scale = torch.rand(1024, 128, device="cuda").to(torch.float8_e4m3fn)
weight_scale = torch.rand(4096, 128, device="cuda").to(torch.float8_e4m3fn)
alpha = torch.tensor(0.0002765655517578125, device="cuda").to(torch.float32)
bias = None
"""


def test_mm(input_tensor_quant, weight, input_tensor_scale, weight_scale, alpha, bias):
    output_tensor = cutlass_scaled_nvfp4_mm(input_tensor_quant, weight, input_tensor_scale, weight_scale, alpha=alpha, bias=bias)
    return output_tensor


def test_tflops(input_shape, weight_shape, num_warmup=10, num_runs=100):
    """
    测试test_mm函数的TFLOPS性能
    """

    # 创建输入数据
    input_tensor_quant = (torch.rand((input_shape[0], input_shape[1] // 2), device="cuda") * 10).to(torch.uint8)
    weight = (torch.rand((weight_shape[0], weight_shape[1] // 2), device="cuda") * 10).to(torch.uint8)

    input_tensor_scale = torch.rand(((input_shape[0] + 128 - 1) // 128) * 128, (input_shape[1] // 16 + 4 - 1) // 4 * 4, device="cuda").to(torch.float8_e4m3fn)
    weight_scale = torch.rand(weight_shape[0], weight_shape[1] // 16, device="cuda").to(torch.float8_e4m3fn)
    alpha = torch.tensor(0.0002765655517578125, device="cuda", dtype=torch.float32)
    bias = None

    # 预热GPU
    for _ in range(num_warmup):
        test_mm(input_tensor_quant, weight, input_tensor_scale, weight_scale, alpha, bias)

    # 同步GPU
    torch.cuda.synchronize()

    # 创建GPU事件用于精确计时
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    # 测量时间
    start_event.record()
    for _ in range(num_runs):
        result = test_mm(input_tensor_quant, weight, input_tensor_scale, weight_scale, alpha, bias)
    end_event.record()

    # 同步并计算时间
    torch.cuda.synchronize()
    elapsed_time_ms = start_event.elapsed_time(end_event)
    elapsed_time_s = elapsed_time_ms / 1000.0

    # 计算FLOPS
    # 矩阵乘法 A(M x K) @ B(K x N) = C(M x N)
    # M = batch_size, K = input_dim, N = output_dim
    M = input_shape[0]
    K = input_shape[1]
    N = weight_shape[0]

    # 每次矩阵乘法的FLOPS = 2 * M * N * K （每个输出元素需要K次乘法和K次加法）
    flops_per_run = 2 * M * N * K
    total_flops = flops_per_run * num_runs

    # 计算TFLOPS (万亿次浮点运算每秒)
    tflops = total_flops / (elapsed_time_s * 1e12)

    print(f"测试结果:")
    print(f"  输入形状: {input_shape} (M={M}, K={K})")
    print(f"  权重形状: {weight_shape} (N={N}, K={K})")
    print(f"  输出形状: ({M}, {N})")
    print(f"  运行次数: {num_runs}")
    print(f"  总执行时间: {elapsed_time_ms:.2f} ms")
    print(f"  平均每次执行时间: {elapsed_time_ms / num_runs:.4f} ms")
    print(f"  每次运行FLOPS: {flops_per_run / 1e9:.2f} GFLOPS")
    print(f"  总FLOPS: {total_flops / 1e12:.2f} TFLOPS")
    print(f"  计算性能: {tflops:.2f} TFLOPS")

    return tflops


if __name__ == "__main__":
    # 测试不同大小的矩阵乘法
    # (m,k) (n,k)
    test_cases = [
        ((32130, 5120), (5120, 5120)),
        ((512, 5120), (5120, 5120)),
        ((257, 5120), (5120, 5120)),
        ((32130, 5120), (13824, 5120)),
        ((32130, 13824), (5120, 13824)),
        ((75348, 5120), (5120, 5120)),
        ((75348, 5120), (13824, 5120)),
        ((75348, 13824), (5120, 13824)),
        ((32760, 1536), (1536, 1536)),
        ((512, 1536), (1536, 1536)),
        ((32760, 1536), (8960, 1536)),
        ((32760, 8960), (1536, 8960)),
    ]

    print("=== test_mm TFLOPS性能测试 ===\n")

    for i, (input_shape, weight_shape) in enumerate(test_cases):
        print(f"测试 {i + 1}: 输入形状 {input_shape}, 权重形状 {weight_shape}")
        print("-" * 60)

        tflops = test_tflops(input_shape, weight_shape)
        print(f"✓ 成功完成测试，性能: {tflops:.2f} TFLOPS\n")

    print("=== 测试完成 ===")
