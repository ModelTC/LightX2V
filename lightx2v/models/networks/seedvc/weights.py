"""Checkpoint operators and bounded CPU block offload for Seed-VC."""

import torch
import torch.nn.functional as F

from lightx2v.common.modules.weight_module import WeightModule
from lightx2v.utils.registry_factory import ATTN_WEIGHT_REGISTER, MM_WEIGHT_REGISTER, RMS_WEIGHT_REGISTER


def linear(name, state, device, size=1, rank=0, split=None, group=None):
    weight = state[name + ".weight"]
    bias = state.get(name + ".bias")
    if split and size > 1:
        weight = weight.chunk(size, dim=0 if split == "col" else 1)[rank].contiguous()
        if bias is not None and split == "col":
            bias = bias.chunk(size)[rank].contiguous()
        op = MM_WEIGHT_REGISTER["TensorParallel"](name + ".weight", name + ".bias" if bias is not None else None, tp_group=group, tp_rank=rank, tp_size=size, split_dim=split)
        owner = op._mm
    else:
        op = MM_WEIGHT_REGISTER["Default"](name + ".weight", name + ".bias" if bias is not None else None)
        owner = op
    owner.weight = weight.to(device=device, dtype=torch.float32).t()
    owner.bias = bias.to(device=device, dtype=torch.float32) if bias is not None else None
    if split == "row" and size > 1 and bias is not None:
        op._row_split_bias, owner.bias = owner.bias, None
    return op


def project(op, x):
    out = op.apply(x.reshape(-1, x.shape[-1]))
    return out.reshape(*x.shape[:-1], out.shape[-1])


@RMS_WEIGHT_REGISTER("seedvc")
class SeedVCRMS:
    def __init__(self, weight, eps=1e-5):
        self.weight = weight
        self.eps = eps

    def apply(self, x):
        value = x.float()
        value = value * torch.rsqrt((value * value).mean(-1, keepdim=True) + self.eps)
        return value.to(x.dtype) * self.weight


@ATTN_WEIGHT_REGISTER("seedvc_sdpa")
class SeedVCAttention:
    def apply(self, q, k, v, mask):
        out = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), attn_mask=mask)
        return out.transpose(1, 2).contiguous().flatten(2)


class SeedVCBlock(WeightModule):
    def __init__(self, index, state, config, device, size=1, rank=0, group=None):
        super().__init__()
        p = f"transformer.layers.{index}."
        # Fused QKV checkpoints must be split by Q/K/V before TP head sharding.
        q, k, v = state[p + "attention.wqkv.weight"].chunk(3)
        local = state | {p + "attention." + key + ".weight": value for key, value in zip(("q", "k", "v"), (q, k, v), strict=True)}
        for name, suffix, split in [
            ("q", "attention.q", "col"),
            ("k", "attention.k", "col"),
            ("v", "attention.v", "col"),
            ("o", "attention.wo", "row"),
            ("gate", "feed_forward.w1", "col"),
            ("up", "feed_forward.w3", "col"),
            ("down", "feed_forward.w2", "row"),
            ("attention_adapt", "attention_norm.project_layer", None),
            ("ffn_adapt", "ffn_norm.project_layer", None),
        ]:
            self.add_module(name, linear(p + suffix, local, device, size, rank, split, group))
        for name in ("attention_norm", "ffn_norm"):
            self.add_module(name, SeedVCRMS(state[p + name + ".norm.weight"].to(device)))
        if config["uvit_skip_connection"]:
            self.add_module("skip", linear(p + "skip_in_linear", state, device))
        self.heads = config["num_heads"] // size
        self.head_dim = config["hidden_dim"] // config["num_heads"]
        self.time_as_token = config["time_as_token"]
        self.attention = ATTN_WEIGHT_REGISTER["seedvc_sdpa"]()

    def tensor_owners(self):
        for module in self._modules.values():
            owner = getattr(module, "_mm", module)
            for name in ("weight", "bias"):
                value = getattr(owner, name, None)
                if value is not None:
                    yield owner, name, value


class BlockOffload:
    """Pinned CPU blocks, two reusable GPU slots; all transfers outside compile."""

    def __init__(self, blocks, make_slot, device):
        self.blocks = blocks
        self.slots = [make_slot(), make_slot()]
        self.stream = torch.cuda.Stream(device=device)
        self.device = device
        self.ready = [torch.cuda.Event(), torch.cuda.Event()]
        self.used = [torch.cuda.Event(), torch.cuda.Event()]
        self.busy = [False, False]
        self.transfer_count = self.transfer_bytes = 0
        for block in blocks:
            for owner, name, value in block.tensor_owners():
                setattr(owner, name, value.pin_memory())

    def copy(self, index):
        slot = index % 2
        with torch.cuda.stream(self.stream):
            if self.busy[slot]:
                self.stream.wait_event(self.used[slot])
            for (_, _, source), (_, _, target) in zip(self.blocks[index].tensor_owners(), self.slots[slot].tensor_owners(), strict=True):
                target.copy_(source, non_blocking=True)
                self.transfer_bytes += source.numel() * source.element_size()
            self.ready[slot].record(self.stream)
            self.transfer_count += 1

    def iterate(self):
        current = torch.cuda.current_stream(self.device)
        self.copy(0)
        for index in range(len(self.blocks)):
            slot = index % 2
            current.wait_event(self.ready[slot])
            if index + 1 < len(self.blocks):
                self.copy(index + 1)
            try:
                yield index, self.slots[slot]
            finally:
                self.used[slot].record(current)
                self.busy[slot] = True

    def close(self):
        self.stream.synchronize()
        torch.cuda.current_stream(self.device).synchronize()
        self.slots.clear()
        self.blocks.clear()
