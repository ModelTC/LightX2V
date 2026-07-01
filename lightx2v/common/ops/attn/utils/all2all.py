import torch
import torch.distributed as dist

try:
    from sageattn3_sparse import dequant_fp4 as dequant_fp4_sage3
    from sageattn3_sparse import quant_fp4 as quant_fp4_sage3
except ImportError:
    quant_fp4_sage3 = None
    dequant_fp4_sage3 = None


def _fp8_all_to_all(input_t, group=None):
    """All-to-all with per-token fp8 compression along the last dim.

    ``input_t`` must be contiguous with dim 0 == world_size (the all-to-all
    split dim). Only the quantized payload + per-token scale cross the wire,
    roughly halving bf16/fp16 communication volume. Returns the dequantized
    tensor in ``input_t``'s original dtype.
    """
    from lightx2v.utils.quant_utils import dequant_fp8_vllm, quant_fp8_vllm

    orig_dtype = input_t.dtype
    shape = input_t.shape
    hidden = shape[-1]
    q, scale = quant_fp8_vllm(input_t.reshape(-1, hidden).contiguous())
    q = q.reshape(shape)
    scale = scale.reshape(*shape[:-1], 1).contiguous()
    out_q = torch.empty_like(q)
    out_scale = torch.empty_like(scale)
    dist.all_to_all_single(out_q, q, group=group)
    dist.all_to_all_single(out_scale, scale, group=group)
    return dequant_fp8_vllm(out_q, out_scale, orig_dtype)


def _fp4_all_to_all(input_t, group=None):
    """All-to-all with SageAttention3 FP4 compression along the last dim."""
    if quant_fp4_sage3 is None or dequant_fp4_sage3 is None:
        raise ImportError("sageattn3_sparse quant_fp4/dequant_fp4 is required for seq_p_fp4_comm.")

    shape = input_t.shape
    hidden = shape[-1]
    q, scale = quant_fp4_sage3(input_t.reshape(1, 1, -1, hidden).contiguous())
    q = q.reshape(*shape[:-1], hidden // 2).contiguous()
    scale = scale.reshape(*shape[:-1], hidden // 16).contiguous()
    out_q = torch.empty_like(q)
    out_scale = torch.empty_like(scale)
    dist.all_to_all_single(out_q, q, group=group)
    dist.all_to_all_single(out_scale, scale, group=group)
    return dequant_fp4_sage3(
        out_q.reshape(1, 1, -1, hidden // 2),
        out_scale.reshape(1, 1, -1, hidden // 16),
    ).reshape(shape)


def all2all_seq2head(input, group=None, use_fp8_comm=False, use_fp4_comm=False):
    """
    将输入张量从 [seq_len/N, heads, hidden_dims] 转换为 [seq_len, heads/N, hidden_dims] 的格式。

    参数:
        input (torch.Tensor): 输入张量，形状为 [seq_len/N, heads, hidden_dims]

    返回:
        torch.Tensor: 转换后的输出张量，形状为 [seq_len, heads/N, hidden_dims]
    """
    # 确保输入是一个3D张量
    assert input.dim() == 3, f"input must be 3D tensor"
    assert not (use_fp8_comm and use_fp4_comm), "use_fp8_comm and use_fp4_comm can't be enabled at the same time."

    # 获取当前进程的世界大小
    world_size = dist.get_world_size(group=group)

    # 获取输入张量的形状
    shard_seq_len, heads, hidden_dims = input.shape
    seq_len = shard_seq_len * world_size  # 计算总序列长度
    shard_heads = heads // world_size  # 计算每个进程处理的头数

    # 重塑输入张量以便进行 all-to-all 操作
    input_t = (
        input.reshape(shard_seq_len, world_size, shard_heads, hidden_dims)  # 重塑为 [shard_seq_len, world_size, shard_heads, hidden_dims]
        .transpose(0, 1)  # 转置以便进行 all-to-all 操作
        .contiguous()  # 确保内存连续
    )

    # 执行 all-to-all 操作，将输入张量的内容分发到所有进程
    if use_fp8_comm:
        output = _fp8_all_to_all(input_t, group=group)
    elif use_fp4_comm:
        output = _fp4_all_to_all(input_t, group=group)
    else:
        output = torch.empty_like(input_t)
        dist.all_to_all_single(output, input_t, group=group)

    # 重塑输出张量为 [seq_len, heads/N, hidden_dims] 形状
    output = output.reshape(seq_len, shard_heads, hidden_dims).contiguous()

    return output  # 返回转换后的输出张量


def all2all_head2seq(input, group=None):
    """
    将输入张量从 [seq_len, heads/N, hidden_dims] 转换为 [seq_len/N, heads, hidden_dims] 的格式。

    参数:
        input (torch.Tensor): 输入张量，形状为 [seq_len, heads/N, hidden_dims]

    返回:
        torch.Tensor: 转换后的输出张量，形状为 [seq_len/N, heads, hidden_dims]
    """
    # 确保输入是一个3D张量
    assert input.dim() == 3, f"input must be 3D tensor"

    # 获取当前进程的世界大小
    world_size = dist.get_world_size(group=group)

    # 获取输入张量的形状
    seq_len, shard_heads, hidden_dims = input.shape
    heads = shard_heads * world_size  # 计算总头数
    shard_seq_len = seq_len // world_size  # 计算每个进程处理的序列长度

    # 重塑输入张量以便进行 all-to-all 操作
    input_t = (
        input.reshape(world_size, shard_seq_len, shard_heads, hidden_dims)  # 重塑为 [world_size, shard_seq_len, shard_heads, hidden_dims]
        .transpose(1, 2)  # 转置以便进行 all-to-all 操作
        .contiguous()  # 确保内存连续
        .reshape(world_size, shard_heads, shard_seq_len, hidden_dims)  # 再次重塑为 [world_size, shard_heads, shard_seq_len, hidden_dims]
    )

    # 创建一个与输入张量相同形状的输出张量
    output = torch.empty_like(input_t)

    # 执行 all-to-all 操作，将输入张量的内容分发到所有进程
    dist.all_to_all_single(output, input_t, group=group)

    # 重塑输出张量为 [heads, shard_seq_len, hidden_dims] 形状
    output = output.reshape(heads, shard_seq_len, hidden_dims)

    # 转置输出张量并重塑为 [shard_seq_len, heads, hidden_dims] 形状
    output = output.transpose(0, 1).contiguous().reshape(shard_seq_len, heads, hidden_dims)

    return output  # 返回转换后的输出张量
