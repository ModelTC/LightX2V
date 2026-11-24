"""Checkpoint-native operators and bounded, double-buffered block offload."""

import torch

from lightx2v.common.modules.weight_module import WeightModule
from lightx2v.utils.envs import GET_DTYPE, GET_SENSITIVE_DTYPE
from lightx2v.utils.registry_factory import ATTN_WEIGHT_REGISTER, MM_WEIGHT_REGISTER, RMS_WEIGHT_REGISTER


@RMS_WEIGHT_REGISTER("yue2")
class YuE2RMS:
    """Keep the release's FP32 reduction and two BF16 multiplication boundaries."""

    def __init__(self, weight, eps):
        self.weight = weight
        self.eps = eps
        # The reference explicitly requires at least FP32 accumulation.
        self.accum_dtype = torch.promote_types(GET_SENSITIVE_DTYPE(), torch.float32)

    def apply(self, x):
        scale = torch.rsqrt(x.to(self.accum_dtype).pow(2).mean(-1, keepdim=True) + self.eps).to(x.dtype)
        return x * scale * self.weight


@ATTN_WEIGHT_REGISTER("yue2_sdpa")
class YuE2Attention:
    def apply(self, q, k, v, causal=False):
        # [T,heads,D] -> [B,heads,T,D]; native SDPA retains grouped KV heads.
        out = torch.nn.functional.scaled_dot_product_attention(
            q.transpose(0, 1)[None],
            k.transpose(0, 1)[None],
            v.transpose(0, 1)[None],
            is_causal=causal,
            enable_gqa=q.shape[1] != k.shape[1],
        )
        return out[0].transpose(0, 1).flatten(1)


def linear(name, state, device, tp_size=1, tp_rank=0, split=None, group=None):
    weight = state[name + ".weight"]
    if split is not None and tp_size > 1:
        dim = 0 if split == "col" else 1
        if weight.shape[dim] % tp_size:
            raise ValueError(f"Cannot shard {name} across {tp_size} ranks")
        weight = weight.chunk(tp_size, dim=dim)[tp_rank].contiguous()
    bias = state.get(name + ".bias")
    if bias is not None and split == "col" and tp_size > 1:
        bias = bias.chunk(tp_size)[tp_rank].contiguous()
    if split is not None and tp_size > 1:
        op = MM_WEIGHT_REGISTER["TensorParallel"](name + ".weight", name + ".bias" if bias is not None else None, tp_group=group, tp_rank=tp_rank, tp_size=tp_size, split_dim=split)
        owner = op._mm
    else:
        op = MM_WEIGHT_REGISTER["Default"](name + ".weight", name + ".bias" if bias is not None else None)
        owner = op
    # Same [input,output] storage contract as MMWeight.load, with an explicit
    # CPU destination for block offload; no load-time GPU allocation is hidden.
    owner.weight = weight.to(device=device, dtype=GET_DTYPE()).t()
    owner.bias = None if bias is None else bias.to(device=device, dtype=GET_DTYPE())
    if split == "row" and tp_size > 1 and bias is not None:
        op._row_split_bias, owner.bias = owner.bias, None
    return op


class YuE2Block(WeightModule):
    def __init__(self, index, phase, state, config, device, tp_size, tp_rank, group):
        super().__init__()
        p = f"model.layers.{index}."
        attn = "self_attn" if phase == "ar" else "nar_self_attn"
        mlp = "mlp" if phase == "ar" else "nar_mlp"
        for short, suffix, split in [
            ("q", f"{attn}.q_proj", "col"),
            ("k", f"{attn}.k_proj", "col"),
            ("v", f"{attn}.v_proj", "col"),
            ("o", f"{attn}.o_proj", "row"),
            ("gate", f"{mlp}.gate_proj", "col"),
            ("up", f"{mlp}.up_proj", "col"),
            ("down", f"{mlp}.down_proj", "row"),
        ]:
            self.add_module(short, linear(p + suffix, state, device, tp_size, tp_rank, split, group))
        for short, suffix in [
            ("input_norm", "input_layernorm" if phase == "ar" else "nar_input_layernorm"),
            ("mlp_norm", "post_attention_layernorm" if phase == "ar" else "nar_pre_mlp_layernorm"),
            ("q_norm", f"{attn}.q_norm"),
            ("k_norm", f"{attn}.k_norm"),
        ]:
            self.add_module(short, RMS_WEIGHT_REGISTER["yue2"](state[p + suffix + ".weight"].to(device), config["rms_norm_eps"]))
        self.head_dim = config["head_dim"]
        self.q_heads = config["num_attention_heads"] // tp_size
        self.kv_heads = config["num_key_value_heads"] // tp_size
        self.attention = ATTN_WEIGHT_REGISTER["yue2_sdpa"]()

    def tensor_owners(self):
        for module in self._modules.values():
            owner = getattr(module, "_mm", module)
            for attr in ("weight", "bias"):
                value = getattr(owner, attr, None)
                if value is not None:
                    yield owner, attr, value


class BlockOffload:
    """CPU owns every core block; each branch has two reusable GPU slots.

    Copies, stream waits and slot selection are outside compiled computation.
    KV is request state and never stored in a reused weight slot.
    """

    def __init__(self, blocks, make_slot, device):
        self.blocks = blocks
        self.device = device
        self.stream = torch.cuda.Stream(device=device)
        self.slots = {phase: [make_slot(phase), make_slot(phase)] for phase in blocks}
        self.loaded = {p: [torch.cuda.Event(), torch.cuda.Event()] for p in blocks}
        self.used = {p: [torch.cuda.Event(), torch.cuda.Event()] for p in blocks}
        self.has_used = {p: [False, False] for p in blocks}
        self.transfer_bytes = 0
        self.transfer_count = 0
        for branch in blocks.values():
            for block in branch:
                for owner, attr, value in block.tensor_owners():
                    setattr(owner, attr, value.pin_memory())

    def _copy(self, phase, index):
        slot = index % 2
        with torch.cuda.stream(self.stream):
            if self.has_used[phase][slot]:
                self.stream.wait_event(self.used[phase][slot])
            source = self.blocks[phase][index].tensor_owners()
            target = self.slots[phase][slot].tensor_owners()
            for (_, _, src), (_, _, dst) in zip(source, target, strict=True):
                dst.copy_(src, non_blocking=True)
                self.transfer_bytes += src.numel() * src.element_size()
            self.loaded[phase][slot].record(self.stream)
            self.transfer_count += 1

    def iterate(self, phase):
        self._copy(phase, 0)
        current = torch.cuda.current_stream(self.device)
        for i in range(len(self.blocks[phase])):
            slot = i % 2
            current.wait_event(self.loaded[phase][slot])
            if i + 1 < len(self.blocks[phase]):
                self._copy(phase, i + 1)
            try:
                yield i, self.slots[phase][slot]
            finally:
                # Also fence the slot if inference is cancelled mid-iteration.
                self.used[phase][slot].record(current)
                self.has_used[phase][slot] = True

    def close(self):
        self.stream.synchronize()
        torch.cuda.current_stream(self.device).synchronize()
        self.slots.clear()
        self.blocks.clear()
