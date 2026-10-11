"""CPU block sources and bounded GPU execution slots for HunyuanImage3.

Routed experts copy directly into each slot's fused MoE pack. No per-layer
resident pack is built, and KV caches continue to use logical layer indices.
"""

import re

import torch
from loguru import logger

from lightx2v.common.modules.weight_module import WeightModule
from lightx2v.common.offload.event_manager import EventSlotWeightAsyncStreamManager
from lightx2v.common.ops.mm.mm_weight import MMWeightTP
from lightx2v.models.networks.hunyuan_image3.weights.transformer_weights import HunyuanImage3TransformerBlock


def iter_weight_leaves(module):
    """Walk through TP wrappers without visiting their concrete weights twice."""
    if isinstance(module, MMWeightTP):
        yield module._mm
    elif isinstance(module, WeightModule):
        for child in module._modules.values():
            if child is not None:
                yield from iter_weight_leaves(child)
    else:
        yield module


def _extract_row_biases(module):
    if isinstance(module, MMWeightTP):
        module._extract_row_split_bias()
    elif isinstance(module, WeightModule):
        for child in module._modules.values():
            if child is not None:
                _extract_row_biases(child)


def _suffix(name):
    return name.split(".", 3)[3]


def block_source_state(block):
    # Generic state_dict also exports empty LoRA/diff placeholders. They are
    # not checkpoint weights and need not be pinned or copied in this path.
    state = {}
    for leaf in iter_weight_leaves(block):
        for name, attr, _ in leaf.base_attrs:
            value = getattr(leaf, f"pin_{attr}", None)
            state[name] = value if value is not None else getattr(leaf, attr)
    return state


class _BlockSource:
    def __init__(self, block):
        self.tensors = block_source_state(block)
        for name, tensor in self.tensors.items():
            if tensor.device.type != "cpu" or not tensor.is_pinned():
                raise ValueError(f"HunyuanImage3 offload source must be pinned CPU storage: {name}")

    def state_dict(self):
        return self.tensors


def block_signature(block, state):
    layout = tuple(sorted((_suffix(name), tuple(t.shape), t.dtype, t.stride()) for name, t in state.items()))
    phase = block.compute_phases[1]
    moe = phase.moe if phase.is_moe else None
    semantics = None if moe is None else (moe.num_experts, moe.moe_topk, moe.micro_shard_count, moe.moe_backend)
    return layout, semantics


class HunyuanImage3BlockSlot:
    """An executable block with stable GPU buffers, including packed experts."""

    def __init__(self, source, config, device):
        self.block = HunyuanImage3TransformerBlock(source.block_index, config, "Default")
        source_state = block_source_state(source)
        self.copies = []
        self.nbytes = 0
        expert_leaves = []
        for leaf in iter_weight_leaves(self.block):
            for name, attr, transpose in leaf.base_attrs:
                value = source_state[name]
                setattr(leaf, f"pin_{attr}", None)
                match = re.search(r"\.mlp\.experts\.(\d+)\.(gate_and_up_proj|down_proj)\.weight$", name)
                if match:
                    # Execution uses the fused pack, not these individual MMs.
                    setattr(leaf, attr, None)
                    expert_leaves.append((name, value, int(match[1]), match[2]))
                    continue
                # Match the resident loader's dense checkpoint layout before
                # the operator transpose. Private pinned TP row slices can
                # retain gaps; copying those gaps to CUDA changes MM strides.
                shape = tuple(reversed(value.shape)) if transpose else value.shape
                target = torch.empty(shape, device=device, dtype=value.dtype)
                if transpose:
                    target = target.t()
                setattr(leaf, attr, target)
                self.copies.append((_suffix(name), target, "direct"))
                self.nbytes += target.numel() * target.element_size()
        _extract_row_biases(self.block)

        if expert_leaves:
            moe = self.block.compute_phases[1].moe
            gate = next(value for _, value, _, kind in expert_leaves if kind == "gate_and_up_proj")
            hidden, gate_up = gate.shape  # Operator layout is [in, out].
            micro = moe.micro_shard_count
            if gate_up % (2 * micro):
                raise ValueError("HunyuanImage3 expert gate/up shape is incompatible with micro shards")
            width = gate_up // (2 * micro)
            moe.moe_fc1_weight = torch.empty((micro, moe.num_experts, 2 * width, hidden), device=device, dtype=gate.dtype)
            moe.moe_fc2_weight = torch.empty((micro, moe.num_experts, hidden, width), device=device, dtype=gate.dtype)
            for name, value, expert, kind in expert_leaves:
                if kind == "gate_and_up_proj":
                    target, transform = moe.moe_fc1_weight[:, expert], "gate_up"
                    expected = (hidden, 2 * micro * width)
                else:
                    target, transform = moe.moe_fc2_weight[:, expert], "down"
                    expected = (micro * width, hidden)
                if tuple(value.shape) != expected or value.dtype != gate.dtype:
                    raise ValueError(f"Incompatible HunyuanImage3 expert layout: {name}")
                self.copies.append((_suffix(name), target, transform))
            self.nbytes += (moe.moe_fc1_weight.numel() + moe.moe_fc2_weight.numel()) * gate.element_size()
            moe._moe_weights_initialized = True
            moe._build_fused_moe_backends()

    def load_state_dict(self, source, block_index, adapter_block_index=None):
        for suffix, target, transform in self.copies:
            value = source[f"model.layers.{block_index}.{suffix}"]
            if transform == "gate_up":
                value = value.t().reshape(target.shape)
            elif transform == "down":
                micro, hidden, width = target.shape
                value = value.t().reshape(hidden, micro, width).permute(1, 0, 2)
            target.copy_(value, non_blocking=True)


class HunyuanImage3BlockOffload:
    def __init__(self, blocks, config, device):
        self.device = torch.device(device)
        self.blocks = blocks
        self.sources = [_BlockSource(block) for block in blocks]
        self.managers = {}
        self.assignments = []
        self.completion = None
        self.closed = False
        next_slots = {}
        with torch.cuda.device(self.device):
            for index, block in enumerate(blocks):
                signature = block_signature(block, self.sources[index].tensors)
                if signature not in self.managers:
                    slots = [HunyuanImage3BlockSlot(block, config, self.device) for _ in range(2)]
                    manager = EventSlotWeightAsyncStreamManager("block")
                    manager.init_cuda_buffer(blocks_cuda_buffer=slots)
                    self.managers[signature] = manager
                    next_slots[manager] = 0
                manager = self.managers[signature]
                slot = next_slots[manager]
                self.assignments.append((manager, slot))
                # Alternate within each family; block order is fixed across requests.
                next_slots[manager] = 1 - slot
        nbytes = sum(slot.nbytes for manager in self.managers.values() for slot in manager.cuda_buffers)
        logger.info("HunyuanImage3 block offload: {} CPU blocks, {} block families, {} GPU slots, {:.3f} GiB slot weights", len(blocks), len(self.managers), 2 * len(self.managers), nbytes / 1024**3)

    def _prefetch(self, index):
        manager, slot = self.assignments[index]
        manager.prefetch_to_slot(slot, index, self.sources)
        return manager, slot

    def infer(self, compute_block, hidden_states, pre_infer_out):
        if self.closed:
            raise RuntimeError("HunyuanImage3 block offload has been closed")
        with torch.cuda.device(self.device):
            stream = torch.cuda.current_stream(self.device)
            if self.completion is not None:
                stream.wait_event(self.completion)
            for manager in self.managers.values():
                if self.completion is not None:
                    manager.cuda_load_stream.wait_event(self.completion)
                manager.reset_slots()
            try:
                pending = self._prefetch(0)
                for index in range(len(self.blocks)):
                    manager, slot = pending
                    buffer = manager.wait_ready(slot, stream)
                    if index + 1 < len(self.blocks):
                        pending = self._prefetch(index + 1)
                    hidden_states = compute_block(index, buffer.block, hidden_states, pre_infer_out)
                    manager.record_free(slot, stream)
                self.completion = torch.cuda.Event()
                self.completion.record(stream)
                return hidden_states
            except BaseException:
                torch.cuda.synchronize(self.device)
                for manager in self.managers.values():
                    manager.reset_slots()
                self.completion = None
                raise

    def close(self):
        if self.closed:
            return
        torch.cuda.synchronize(self.device)
        self.assignments.clear()
        self.managers.clear()
        self.sources.clear()
        self.completion = None
        self.closed = True
