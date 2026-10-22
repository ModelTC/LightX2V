"""Reuse LightX2V Ulysses collectives around full-grid local attention."""

import torch
import torch.distributed as dist
import torch.nn.functional as F

from lightx2v.common.ops.attn.utils.all2all import all2all_head2seq, all2all_seq2head


class SequenceParallel:
    def __init__(self, group=None):
        self.group = group
        self.size = dist.get_world_size(group) if group is not None else 1
        self.rank = dist.get_rank(group) if group is not None else 0

    def split(self, x):
        if self.size == 1:
            return x
        return F.pad(x, (0, 0, 0, -x.shape[0] % self.size)).chunk(self.size)[self.rank].contiguous()

    def gather(self, x, tokens):
        if self.size == 1:
            return x
        output = torch.empty((x.shape[0] * self.size, *x.shape[1:]), device=x.device, dtype=x.dtype)
        dist.all_gather_into_tensor(output, x.contiguous(), group=self.group)
        return output[:tokens]

    def to_heads(self, x, tokens):
        return all2all_seq2head(x, group=self.group)[:tokens] if self.size > 1 else x

    def to_sequence(self, x):
        if self.size == 1:
            return x.flatten(1)
        x = F.pad(x, (0, 0, 0, 0, 0, -x.shape[0] % self.size))
        return all2all_head2seq(x, group=self.group).flatten(1)
