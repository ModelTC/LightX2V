import torch.nn.functional as F


def safe_pad_operation(x, pad):
    return F.pad(x, pad)
