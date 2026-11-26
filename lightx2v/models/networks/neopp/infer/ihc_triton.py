"""Two-stream HY4 iHC kernels. The trained FP32 gate projection stays in PyTorch."""

import torch
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def _read_kernel(states, pre, out, D: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    col = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = col < D
    first = tl.load(states + row * 2 * D + col, mask, 0)
    second = tl.load(states + row * 2 * D + D + col, mask, 0)
    p0, p1 = tl.load(pre + row * 2), tl.load(pre + row * 2 + 1)
    tl.store(out + row * D + col, p0 * first + p1 * second, mask)


@triton.jit
def _write_kernel(
    states,
    update,
    post,
    out,
    tokens: tl.constexpr,
    width: tl.constexpr,
    stride_row,
    stride_stream,
    BLOCK: tl.constexpr,
):
    offset = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offset < tokens * 2 * width
    row = offset // (2 * width)
    stream = offset // width % 2
    col = offset % width
    x = tl.load(states + row * stride_row + stream * stride_stream + col, mask, 0).to(tl.float32)
    u = tl.load(update + row * width + col, mask, 0).to(tl.float32)
    p = tl.load(post + row * 2 + stream, mask, 0)
    tl.store(out + offset, x + p * u, mask)


def read(states, w):
    n, streams, d = states.shape
    assert streams == 2 and states.dtype == torch.bfloat16
    flat = states.flatten(1).float().contiguous()
    with torch.autocast(device_type=states.device.type, enabled=False):
        inv_rms = torch.rsqrt(flat.square().mean(-1, keepdim=True) + 1e-5)
        logits = F.linear(flat, w.proj_rw.tensor.float().T) * inv_rms
        first, second = logits.chunk(2, dim=-1)
        pre = torch.sigmoid(w.bias_pre.tensor + w.alpha_pre.tensor * first) + 1e-6
        post = 2 * torch.sigmoid(w.bias_post.tensor + w.alpha_post.tensor * second) + 1e-6
    values = states.new_empty(n, d)
    block = 512
    _read_kernel[(n, triton.cdiv(d, block))](flat, pre, values, d, block, enable_fp_fusion=False)
    return values, post


def write(states, update, post):
    n, streams, d = states.shape
    assert streams == 2 and states.stride(-1) == 1 and update.is_contiguous()
    out = states.new_empty(n, streams, d)
    block = 1024
    _write_kernel[(triton.cdiv(out.numel(), block),)](
        states,
        update,
        post,
        out,
        n,
        d,
        states.stride(0),
        states.stride(1),
        block,
        enable_fp_fusion=False,
    )
    return out
