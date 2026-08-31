import torch.nn.functional as F

from lightx2v.models.video_encoders.hf.seedvr.common.distributed import ops as distributed_ops

gather_heads_scatter_seq = distributed_ops.gather_heads_scatter_seq
gather_outputs = distributed_ops.gather_outputs
gather_seq_scatter_heads_qkv = distributed_ops.gather_seq_scatter_heads_qkv
slice_inputs = distributed_ops.slice_inputs


def safe_pad_operation(x, pad):
    return F.pad(x, pad)
