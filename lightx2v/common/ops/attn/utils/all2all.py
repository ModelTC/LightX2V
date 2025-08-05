import torch
import torch._dynamo as dynamo
import torch.distributed as dist


@dynamo.disable
def all2all_seq2head(input, group=None):
    """
    将输入张量从 [seq_len/N, heads, hidden_dims] 转换为 [seq_len, heads/N, hidden_dims] 的格式。

    参数:
        input (torch.Tensor): 输入张量，形状为 [seq_len/N, heads, hidden_dims]

    返回:
        torch.Tensor: 转换后的输出张量，形状为 [seq_len, heads/N, hidden_dims]
    """
    # 确保输入是一个3D张量
    assert input.dim() == 3, f"input must be 3D tensor"

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

    # 创建一个与输入张量相同形状的输出张量
    output = torch.empty_like(input_t)

    # 执行 all-to-all 操作，将输入张量的内容分发到所有进程
    dist.all_to_all_single(output, input_t, group=group)

    # 重塑输出张量为 [seq_len, heads/N, hidden_dims] 形状
    output = output.reshape(seq_len, shard_heads, hidden_dims).contiguous()

    return output  # 返回转换后的输出张量


@dynamo.disable
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
