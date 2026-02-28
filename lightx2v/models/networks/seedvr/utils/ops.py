import torch.nn.functional as F


def slice_inputs(x, dim=0):
    return x


def gather_outputs(x, gather_dim=0, padding_dim=0, unpad_shape=None, cache=None):
    return x


def gather_seq_scatter_heads_qkv(x, seq_dim=0, qkv_shape=None, cache=None):
    return x


def gather_heads_scatter_seq(x, head_dim=1, seq_dim=0):
    return x


def safe_pad_operation(x, pad):
    return F.pad(x, pad)
