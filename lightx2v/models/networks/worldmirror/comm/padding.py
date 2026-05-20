import torch
import torch.nn.functional as F


def minimal_pad_to_divisible(tensor: torch.Tensor, sp_size: int, dim: int = 1, pad_value: float = 0.0):
    """
    对三维或更高维度的tensor在指定维度进行最小化padding，使其长度能被 sp_size 整除。

    Args:
        tensor: 输入的PyTorch tensor (例如：[B, L, C] 或 [B, H, W, C] 等)。
        sp_size: 要求的最小分割尺寸。
        dim: 需要进行padding的维度索引（默认为 1，即第二维）。
        pad_value: 填充的值（默认为 0.0）。

    Returns:
        padded_tensor: 填充后的 tensor。
    """

    current_size = tensor.size(dim)

    # 计算需要填充的长度
    # (sp_size - current_size % sp_size) % sp_size
    # 保证了如果 current_size 已经是 sp_size 的倍数，padding_len 为 0。
    # 否则，计算出最小的填充长度。
    padding_len = (sp_size - current_size % sp_size) % sp_size

    if padding_len == 0:
        # 如果长度已经可以整除，直接返回原 tensor
        return tensor, 0

    # 构建 pad 元组
    # torch.nn.functional.pad 的 pad 参数是从**最末尾的维度**开始，**成对** (后填充, 前填充) 指定的。
    # 假设你的 tensor 是 [D0, D1, D2]
    # 如果 dim=1 (第二维, D1)，pad 应该是 (0, 0, padding_len, 0, 0, 0, ...)
    #
    # 由于我们需要在第二维 (dim=1) 的末尾进行填充，我们需要确定 pad 元组中对应 dim=1 的位置。
    # 维度数量 D = tensor.dim()
    # dim=0 对应 pad 元组的最后两位
    # dim=1 对应 pad 元组的倒数第 4, 3 位
    # dim=2 对应 pad 元组的倒数第 6, 5 位 (对于三维 tensor，即前两位)

    # 在 dim 维度进行 '后填充' (在末尾添加)
    # padding_dims 是一个长度为 2 * D 的元组，所有维度默认不填充
    padding_dims = [0] * (2 * tensor.dim())

    # 对应 dim 维度的 '后填充' (即 pad 元组中的偶数索引位置，从后往前数)
    # 填充的位置是 (2 * tensor.dim() - 2 * dim - 2)
    # 例如：D=3, dim=1 -> 2*3 - 2*1 - 2 = 2
    # pad 元组为 (d2_start, d2_end, d1_start, d1_end, d0_start, d0_end)
    # 我们要填充 d1_end，它在索引 2 的位置

    # F.pad 要求的是 (最后维度 start, 最后维度 end, 倒数第二维度 start, 倒数第二维度 end, ...)
    # 我们的 dim=1 是倒数第 (D - 1 - dim) + 1 = D - dim 个维度
    # 它在 pad 元组中是倒数第 2 * (D - dim) 位和倒数第 2 * (D - dim) - 1 位
    #
    # 填充位置的索引 (从 0 开始, 从左往右):
    # (2 * (tensor.dim() - dim - 1)) 是 '前填充' 的位置
    # (2 * (tensor.dim() - dim - 1) + 1) 是 '后填充' 的位置
    pad_index = 2 * (tensor.dim() - dim - 1) + 1

    if pad_index < len(padding_dims):
        padding_dims[pad_index] = padding_len
    else:
        raise ValueError("Invalid dimension index.")

    # 转换回 tuple
    pad = tuple(padding_dims)

    # 使用 F.pad 进行填充，模式为 'constant'
    padded_tensor = F.pad(tensor, pad=pad, mode="constant", value=pad_value)

    return padded_tensor, padding_len


def depad_by_length(padded_tensor: torch.Tensor, depadding_len: int, dim: int = 1) -> torch.Tensor:
    """
    在指定维度上去除末尾的 padding 部分。

    Args:
        padded_tensor: 已经经过 padding 的 PyTorch tensor。
        depadding_len: 需要从末尾去除的长度。
        dim: 需要去除 padding 的维度索引（默认为 1，即第二维）。

    Returns:
        depadded_tensor: 去除 padding 后的 tensor。
    """

    # 检查去除长度是否合理
    current_size = padded_tensor.size(dim)
    if depadding_len < 0:
        raise ValueError("depadding_len 必须是非负数。")
    if depadding_len > current_size:
        raise ValueError(f"要去除的长度 {depadding_len} 大于当前维度长度 {current_size}。")

    # 计算去除 padding 后的目标长度
    target_size = current_size - depadding_len

    # 构造切片操作所需的索引元组
    # 对于所有维度，我们默认使用完整的切片 `:`
    slices = [slice(None)] * padded_tensor.dim()

    # 在指定维度 dim 上，我们只取从 0 到 target_size 的部分
    # Python 切片 [0:target_size] 会保留 target_size 个元素，即去除了末尾的 depadding_len
    slices[dim] = slice(0, target_size)

    # 使用元组解包进行切片操作
    depadded_tensor = padded_tensor[tuple(slices)]

    return depadded_tensor


def pad_by_length(padded_tensor: torch.Tensor, padding_len: int, dim: int = 1, pad_value: float = 0.0) -> torch.Tensor:
    if padding_len < 0:
        raise ValueError("padding_len 必须是非负数。")

    if dim < 0 or dim >= padded_tensor.dim():
        raise ValueError(f"维度索引 {dim} 超出有效范围 [0, {padded_tensor.dim() - 1}]。")

    # 构建padding参数
    # F.pad需要为每个维度指定左右两边的padding长度
    # 格式为: (最后一个维度的左边, 最后一个维度的右边, 倒数第二个维度的左边, 倒数第二个维度的右边, ...)
    pad_tuple = [0] * (2 * padded_tensor.dim())

    # 将指定维度右边的padding长度设置为padding_len
    # F.pad的维度顺序是从最后一个维度开始的，所以需要进行转换
    pad_idx = 2 * (padded_tensor.dim() - 1 - dim) + 1
    pad_tuple[pad_idx] = padding_len

    # 调用F.pad进行padding
    padded_tensor = F.pad(padded_tensor, pad=tuple(pad_tuple), mode="constant", value=pad_value)

    return padded_tensor
