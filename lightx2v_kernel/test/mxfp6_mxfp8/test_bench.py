import torch
from lightx2v_kernel.gemm import scaled_fp8_quant, scaled_fp6_quant, cutlass_scaled_mxfp6_mxfp8_mm
import time


class MMWeightMxfp8ActMxfp6:
    def __init__(self, weight, bias):
        self.load_fp6_weight(weight, bias)
        self.act_quant_func = self.act_quant_fp8
        self.set_alpha()

    @torch.no_grad()
    def apply(self, input_tensor):
        input_tensor_quant, input_tensor_scale = self.act_quant_func(input_tensor)
        output_tensor = cutlass_scaled_mxfp6_mxfp8_mm(input_tensor_quant, self.weight, input_tensor_scale, self.weight_scale, alpha=self.alpha, bias=self.bias)
        return output_tensor

    @torch.no_grad()
    def load_fp6_weight(self, weight, bias):
        self.weight, self.weight_scale = scaled_fp6_quant(weight)
        self.bias = bias

    def set_alpha(self):
        self.alpha = torch.tensor(1.0, dtype=torch.float32, device=self.weight.device)

    @torch.no_grad()
    def act_quant_fp8(self, x):
        return scaled_fp8_quant(x)


def test_speed(m, k, n):
    with torch.no_grad():
        input_tensor = torch.randn(m, k, dtype=torch.bfloat16).cuda()
        weight = torch.randn(n, k, dtype=torch.bfloat16, device="cuda")
        # bias = torch.randn(1, n, dtype=torch.bfloat16).cuda()
        bias = None

        mm = MMWeightMxfp8ActMxfp6(weight, bias)

        # warmup
        output_tensor = mm.apply(input_tensor)

        torch.cuda.synchronize()
        start_time = time.time()
        for i in range(100):
            output_tensor = mm.apply(input_tensor)
        torch.cuda.synchronize()
        end_time = time.time()

        lightx2v_kernel_time = (end_time - start_time) / 100
        print(f"lightx2v-kernel time: {lightx2v_kernel_time}")

        input_tensor = torch.randn(m, n, dtype=torch.bfloat16).cuda()
        weight = torch.randn(k, n, dtype=torch.bfloat16, device="cuda")
        bias = torch.randn(1, k, dtype=torch.bfloat16).cuda()

        linear = torch.nn.Linear(k, n, bias=False).cuda()
        linear.weight.data = weight
        # linear.bias.data = bias

        # warmup
        ref_output_tensor = linear(input_tensor)

        torch.cuda.synchronize()
        start_time = time.time()
        for i in range(100):
            ref_output_tensor = linear(input_tensor)
        torch.cuda.synchronize()
        end_time = time.time()

        ref_time = (end_time - start_time) / 100
        print(f"ref time: {ref_time}")

        print(f"speedup: {ref_time / lightx2v_kernel_time:.3f}")


def test_accuracy(m, k, n):
    with torch.no_grad():
        input_tensor = torch.randn(m, k, dtype=torch.bfloat16).cuda()
        weight = torch.randn(n, k, dtype=torch.bfloat16, device="cuda")
        # bias = torch.randn(1, n, dtype=torch.bfloat16).cuda()
        bias = None

        linear = torch.nn.Linear(k, n, bias=False).cuda()
        linear.weight.data = weight
        # linear.bias.data = bias

        ref_output_tensor = linear(input_tensor)

        mm = MMWeightMxfp8ActMxfp6(weight, bias)

        output_tensor = mm.apply(input_tensor)

        # print(f"ref_output_tensor: {ref_output_tensor}")
        # print(f"output_tensor: {output_tensor}")

        # cosine
        cos = torch.nn.functional.cosine_similarity(ref_output_tensor.flatten(), output_tensor.flatten(), dim=0)
        print(f"cos : {cos}")


if __name__ == "__main__":
    test_sizes = [
        (32130, 5120, 5120),
        (512, 5120, 5120),
        (257, 5120, 5120),
        (32130, 5120, 13824),
        (32130, 13824, 5120),
        (75348, 5120, 5120),
        (75348, 13824, 5120),
        (32760, 1536, 1536),
        (512, 1536, 1536),
        (32760, 1536, 8960),
        (32760, 8960, 1536),
    ]

    for i, (m, k, n) in enumerate(test_sizes):
        print("-" * 30)
        print(f"测试 {i + 1}: 张量大小 ({m}, {k}, {n})")
        test_accuracy(m, k, n)
        test_speed(m, k, n)
