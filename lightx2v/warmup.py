import json
import time

import torch

from lightx2v.common.ops.mm.triton_kernels import fp8_gemm_bias_triton, fp8_gemm_triton, fp8_quantize_triton, int8_gemm_bias_triton, int8_gemm_triton, int8_quantize_triton
from lightx2v.utils.registry_factory import (
    ATTN_WEIGHT_REGISTER,
)


# ====================== 调优功能（移除 fuse_gelu） ======================
def generate_test_data(M, K, N, dtype=torch.bfloat16, device="cuda"):
    """生成测试数据"""
    a = torch.randn((M, K), dtype=dtype, device=device)
    b = torch.randn((K, N), dtype=dtype, device=device)
    bias = torch.randn(N, dtype=dtype, device=device)
    return a, b, bias


def quantize_data(data, quant_type="int8"):
    if quant_type == "int8":
        return int8_quantize_triton(data)
    elif quant_type == "fp8":
        return fp8_quantize_triton(data)
    else:
        raise ValueError(f"不支持的量化类型: {quant_type}")


def get_qkv(seq_len):
    batch_size = 1
    head_num = 40
    head_dim = 128
    q = torch.randn((batch_size, seq_len, head_num, head_dim), dtype=torch.bfloat16, device="cuda")
    k = torch.randn((batch_size, seq_len, head_num, head_dim), dtype=torch.bfloat16, device="cuda")
    v = torch.randn((batch_size, seq_len, head_num, head_dim), dtype=torch.bfloat16, device="cuda")
    return q, k, v


def time_op(op, *args, warmup=1, iters=3):
    for i in range(warmup):
        print(i, "warm")
        if hasattr(op, "apply"):
            _ = op.apply(*args)
        else:
            _ = op(*args)
    torch.cuda.synchronize()

    # 正式计时
    start = time.time()
    for i in range(iters):
        print(i, "run")
        if hasattr(op, "apply"):
            _ = op.apply(*args)
        else:
            _ = op(*args)
    torch.cuda.synchronize()
    end = time.time()

    # 计算平均耗时 (ms)
    avg_time = (end - start) / iters * 1000
    return avg_time


def tune_quant_gemm(test_sizes, quant_type="int8", output_dtype=torch.bfloat16, save_results=True):
    """调优量化GEMM算子（默认不融合GELU）"""
    assert quant_type in ["int8", "fp8"], f"无效的量化类型: {quant_type}"
    device = "cuda"
    dtype = torch.bfloat16  # 原始数据类型
    results = []

    print(f"\n========== 开始调优 {quant_type.upper()} GEMM 算子 ==========")
    print(f"输出类型: {output_dtype}")
    print("=" * 80)

    for idx, (M, K, N) in enumerate(test_sizes):
        print(f"\n[{idx + 1}/{len(test_sizes)}] 测试尺寸: M={M}, K={K}, N={N}")

        # 1. 生成测试数据
        a, b, bias = generate_test_data(M, K, N, dtype=dtype, device=device)

        # 2. 量化激活和权重
        print(f"  - 量化{quant_type}数据...")
        a_quant, a_scales = quantize_data(a, quant_type=quant_type)
        b_quant, b_scales = quantize_data(b, quant_type=quant_type)

        # 3. 测试带bias和不带bias两种情况
        for has_bias in [False, True]:
            print(f"  - 测试{'带' if has_bias else '不带'}bias...")

            # 选择对应的算子
            if quant_type == "int8":
                if has_bias:
                    gemm_op = int8_gemm_bias_triton
                    op_args = (a_quant, b_quant.t(), bias, a_scales, b_scales, False, output_dtype)
                else:
                    gemm_op = int8_gemm_triton
                    op_args = (a_quant, b_quant.t(), a_scales, b_scales, False, output_dtype)
            else:  # fp8
                if has_bias:
                    gemm_op = fp8_gemm_bias_triton
                    op_args = (a_quant, b_quant.t(), bias, a_scales, b_scales, False, output_dtype)
                else:
                    gemm_op = fp8_gemm_triton
                    op_args = (a_quant, b_quant.t(), a_scales, b_scales, False, output_dtype)

            avg_time_ms = time_op(gemm_op, *op_args)
            print(avg_time_ms)

    return results


def warmup_attn(config):
    if config["resolution"] == "480p":
        seq_lens = [32760, 18900]
    elif config["resolution"] == "540p":
        seq_lens = [42840, 27216]
    elif config["resolution"] == "720p":
        seq_lens = [75600]
    for seq_len in seq_lens:
        q, k, v = get_qkv(seq_len)
        attn_op = ATTN_WEIGHT_REGISTER[config["self_attn_1_type"]]()
        print(attn_op)
        op_args = (q, k, v, None, None, q.shape[0], None)
        avg_time_ms = time_op(attn_op, *op_args)
        print(avg_time_ms)


def warmup_triton_mm(config):
    if "resolution" not in config:
        return
    test_sizes = [
        (512, 5120, 5120),
        (257, 5120, 5120),
    ]
    if config["resolution"] == "480p":
        test_size = [
            (32760, 5120, 5120),
            (32760, 5120, 13824),
            (32760, 13824, 5120),
            (32760, 13824, 13824),
            (18900, 5120, 5120),
            (18900, 5120, 13824),
            (18900, 13824, 5120),
            (18900, 13824, 13824),
        ]
    elif config["resolution"] == "540p":
        test_size = [(42840, 5120, 5120), (42840, 5120, 13824), (42840, 13824, 5120), (42840, 13824, 13824), (27216, 5120, 5120), (27216, 5120, 13824), (27216, 13824, 5120), (27216, 13824, 13824)]
    if config["resolution"] == "720p":
        test_size = [
            (75600, 5120, 5120),
            (75600, 13824, 5120),
            (75600, 13824, 13824),
            (75600, 5120, 13824),
        ]

    test_sizes.extend(test_size)

    if config["dit_quant_scheme"] == "int8-triton":
        int8_results = tune_quant_gemm(test_sizes=test_sizes, quant_type="int8", output_dtype=torch.bfloat16, save_results=False)
    elif config["dit_quant_scheme"] == "fp8-triton":
        fp8_results = tune_quant_gemm(test_sizes=test_sizes, quant_type="fp8", output_dtype=torch.bfloat16, save_results=False)


def warmup(config):
    warmup_attn(config)
    warmup_triton_mm(config)

if __name__ == "__main__":
    config = {
        "resolution": "720p",
        "resize_mode": "adaptive",
        "self_attn_1_type": "sage_attn2",
        "dit_quant_scheme": "int8-triton",
    }
    warmup(config)