import torch

from lightx2v_kernel.kv_cache import dequantize_kv_cache_fp4


E2M1_TO_FLOAT = torch.tensor(
    [
        0.0,
        0.5,
        1.0,
        1.5,
        2.0,
        3.0,
        4.0,
        6.0,
        -0.0,
        -0.5,
        -1.0,
        -1.5,
        -2.0,
        -3.0,
        -4.0,
        -6.0,
    ],
    dtype=torch.float32,
)


def blocked_scale_index(row: int, scale_col: int, scale_cols: int) -> int:
    row_block = row // 128
    row_in_block = row - row_block * 128
    scale_col_block = scale_col // 4
    scale_col_in_block = scale_col - scale_col_block * 4
    scale_col_blocks = scale_cols // 4
    logical_block = row_block * scale_col_blocks + scale_col_block
    return ((logical_block * 32 + (row_in_block & 31)) * 16 + (row_in_block >> 5) * 4 + scale_col_in_block)


def reference_dequant(values, scales, amax, *, num_heads, block_token_size, e2m1_max, e4m3_max, dtype):
    max_blocks = len(values)
    packed_cols = values[0].size(1)
    head_dim = packed_cols * 2
    scale_cols = head_dim // 16
    out = torch.empty(1, max_blocks * block_token_size, num_heads, head_dim, device=values[0].device, dtype=torch.float32)
    lut = E2M1_TO_FLOAT.to(values[0].device)

    for block_idx, block_values in enumerate(values):
        block_scales = scales[block_idx].float()
        global_scale = amax[block_idx].float()[0] / (e2m1_max * e4m3_max)
        for token in range(block_token_size):
            for head in range(num_heads):
                row = token * num_heads + head
                for col_pair in range(packed_cols):
                    packed = int(block_values[row, col_pair].item())
                    scale_col = (col_pair * 2) // 16
                    scale = block_scales[blocked_scale_index(row, scale_col, scale_cols)]
                    out_token = block_idx * block_token_size + token
                    out[0, out_token, head, col_pair * 2] = lut[packed & 0x0F] * scale * global_scale
                    out[0, out_token, head, col_pair * 2 + 1] = lut[(packed >> 4) & 0x0F] * scale * global_scale
    return out.to(dtype)


def benchmark_cuda(fn, warmup=10, repeat=100):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeat):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / repeat


def benchmark_wall(fn, warmup=2, repeat=5):
    import time

    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(repeat):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) * 1000.0 / repeat


def main():
    if not torch.cuda.is_available():
        print("CUDA is not available, skip kv dequant test.")
        return

    torch.manual_seed(0)
    num_blocks = 3
    num_heads = 4
    block_token_size = 16
    head_dim = 128
    packed_cols = head_dim // 2
    scale_cols = head_dim // 16
    rows_padded = 128
    scale_len = (rows_padded // 128) * (scale_cols // 4) * 32 * 16
    e2m1_max, e4m3_max = 6.0, 256.0

    values = [
        torch.randint(0, 256, (rows_padded, packed_cols), device="cuda", dtype=torch.uint8)
        for _ in range(num_blocks)
    ]
    scales = [
        torch.ones(scale_len, device="cuda", dtype=torch.float8_e4m3fn)
        for _ in range(num_blocks)
    ]
    amax = [
        torch.tensor([e2m1_max * e4m3_max], device="cuda", dtype=torch.float32)
        for _ in range(num_blocks)
    ]

    out = dequantize_kv_cache_fp4(
        values,
        scales,
        amax,
        num_heads=num_heads,
        block_token_size=block_token_size,
        dtype=torch.float16,
        e2m1_max=e2m1_max,
        e4m3_max=e4m3_max,
    )
    ref = reference_dequant(
        values,
        scales,
        amax,
        num_heads=num_heads,
        block_token_size=block_token_size,
        e2m1_max=e2m1_max,
        e4m3_max=e4m3_max,
        dtype=torch.float16,
    )

    diff = (out.float() - ref.float()).abs()
    max_diff = diff.max()
    max_idx = int(diff.reshape(-1).argmax().item())
    flat_out = out.reshape(-1)
    flat_ref = ref.reshape(-1)

    print("output shape:", tuple(out.shape), "dtype:", out.dtype)
    print("first 16 out:", flat_out[:16].detach().cpu().tolist())
    print("first 16 ref:", flat_ref[:16].detach().cpu().tolist())
    print("max_abs_diff:", max_diff.item(), "at flat index:", max_idx)
    print("out[max_idx]:", flat_out[max_idx].item(), "ref[max_idx]:", flat_ref[max_idx].item())

    def run_cuda():
        return dequantize_kv_cache_fp4(
            values,
            scales,
            amax,
            num_heads=num_heads,
            block_token_size=block_token_size,
            dtype=torch.float16,
            e2m1_max=e2m1_max,
            e4m3_max=e4m3_max,
        )

    def run_reference():
        return reference_dequant(
            values,
            scales,
            amax,
            num_heads=num_heads,
            block_token_size=block_token_size,
            e2m1_max=e2m1_max,
            e4m3_max=e4m3_max,
            dtype=torch.float16,
        )

    cuda_ms = benchmark_cuda(run_cuda)
    ref_ms = benchmark_wall(run_reference)
    print(f"cuda kernel avg time: {cuda_ms:.4f} ms")
    print(f"python reference avg time: {ref_ms:.4f} ms")
    print(f"speedup vs reference: {ref_ms / cuda_ms:.2f}x")

    torch.testing.assert_close(out, ref, rtol=0, atol=0)
    print("kv dequant CUDA test passed.")


if __name__ == "__main__":
    main()
