import torch
from lightx2v_kernel.gemm import scaled_nvfp4_quant


input_global_scale = torch.tensor(808.0, dtype=torch.float32).cuda()


def quantize_fp4(x):
    return scaled_nvfp4_quant(x, input_global_scale)


def test_memory_bandwidth(func, x, num_warmup=10, num_runs=100):
    """
    测试函数的显存带宽
    """
    # 预热GPU
    for _ in range(num_warmup):
        func(x)

    # 同步GPU
    torch.cuda.synchronize()

    # 创建GPU事件用于精确计时
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    # 测量时间
    start_event.record()
    for _ in range(num_runs):
        result = func(x)
    end_event.record()

    # 同步并计算时间
    torch.cuda.synchronize()
    elapsed_time_ms = start_event.elapsed_time(end_event)
    elapsed_time_s = elapsed_time_ms / 1000.0

    # 计算数据量
    input_bytes = x.numel() * x.element_size()  # 输入数据字节数

    # FP4量化后，每个元素占用0.5字节
    output_bytes = x.numel() * 0.5  # FP4输出数据字节数

    scale_bytes = x.numel() / 16  # group_size = 16

    # 总数据传输量（读取输入 + 写入输出 + scale）
    total_bytes = (input_bytes + output_bytes + scale_bytes) * num_runs

    # 计算带宽
    bandwidth_gbps = (total_bytes / elapsed_time_s) / (1024**3)  # GB/s

    print(f"测试结果:")
    print(f"  输入张量形状: {x.shape}")
    print(f"  输入数据类型: {x.dtype}")
    print(f"  运行次数: {num_runs}")
    print(f"  总执行时间: {elapsed_time_ms:.2f} ms")
    print(f"  平均每次执行时间: {elapsed_time_ms / num_runs:.4f} ms")
    print(f"  输入数据大小: {input_bytes / (1024**2):.2f} MB")
    print(f"  输出数据大小: {output_bytes / (1024**2):.2f} MB")
    print(f"  总数据传输量: {total_bytes / (1024**3):.2f} GB")
    print(f"  显存带宽: {bandwidth_gbps:.2f} GB/s")

    return bandwidth_gbps


if __name__ == "__main__":
    # 测试不同大小的张量
    test_sizes = [
        # (1, 1024),
        # (1, 2048),
        # (1, 4096),
        # (1, 8192),
        # (1, 16384),
        # (1, 32768),
        # (2, 1024),
        # (2, 2048),
        # (2, 4096),
        # (2, 8192),
        # (2, 16384),
        # (2, 32768),
        # (4, 1024),
        # (4, 2048),
        # (4, 4096),
        # (4, 8192),
        # (4, 16384),
        # (4, 32768),
        # (128, 1024),
        # (128, 2048),
        # (128, 4096),
        # (128, 8192),
        # (128, 16384),
        # (128, 32768),
        # (512, 1024),
        # (512, 2048),
        # (512, 4096),
        # (512, 8192),
        # (512, 16384),
        # (512, 32768),
        # (1024, 1024),
        # (1024, 2048),
        # (1024, 4096),
        # (1024, 8192),
        # (1024, 16384),
        # (1024, 32768),
        # (2048, 1024),
        # (2048, 2048),
        # (2048, 4096),
        # (2048, 8192),
        # (2048, 16384),
        # (2048, 32768),
        # (4096, 1024),
        # (4096, 2048),
        # (4096, 4096),
        # (4096, 8192),
        # (4096, 16384),
        # (4096, 32768),
        # (8192, 1024),
        # (8192, 2048),
        # (8192, 4096),
        # (8192, 8192),
        # (8192, 16384),
        # (8192, 32768),
        # (16384, 1024),
        # (16384, 2048),
        # (16384, 4096),
        # (16384, 8192),
        # (16384, 16384),
        # (16384, 32768),
        # (32768, 1024),
        # (32768, 2048),
        # (32768, 4096),
        # (32768, 8192),
        # (32768, 16384),
        # (32768, 32768),
        (32130, 5120),
        (512, 5120),
        (257, 5120),
        (32130, 13824),
        (75348, 5120),
        (75348, 13824),
        (32760, 1536),
        (512, 1536),
        (32760, 8960),
    ]

    print("=== quantize_fp4 显存带宽测试 ===\n")

    for i, (h, w) in enumerate(test_sizes):
        print(f"测试 {i + 1}: 张量大小 ({h}, {w})")
        print("-" * 50)

        x = torch.randn(h, w, dtype=torch.bfloat16).cuda()

        try:
            bandwidth = test_memory_bandwidth(quantize_fp4, x)
            print(f"✓ 成功完成测试，带宽: {bandwidth:.2f} GB/s\n")
        except Exception as e:
            print(f"✗ 测试失败: {e}\n")

    print("=== 测试完成 ===")
