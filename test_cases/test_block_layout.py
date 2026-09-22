"""Single-GPU tests for direct loading and immutable contiguous offload."""

import re

import pytest
import torch

import lightx2v.common.ops  # noqa: F401
from lightx2v.common.offload.block_layout import ContiguousBlockTransfer
from lightx2v.common.offload.manager import WeightAsyncStreamManager
from lightx2v.common.ops.mm.mm_weight import MMWeightWfp8channelAfp8channeldynamicVllm
from lightx2v.common.ops.norm.rms_norm_weight import RMSWeightTemplate
from lightx2v.common.ops.tensor.tensor import DefaultTensor
from lightx2v.models.networks.wan.weights.block_layout import validate_contiguous_config
from lightx2v.models.networks.wan.weights.transformer_weights import WanTransformerAttentionBlock

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires one CUDA GPU")


def _config(contiguous=True):
    config = {
        "model_cls": "wan2.1",
        "task": "i2v",
        "dit_quantized": True,
        "dit_quant_scheme": "fp8-vllm",
        "cpu_offload": True,
        "offload_granularity": "block",
        "seq_parallel": False,
        "self_attn_1_type": "flash_attn3",
        "cross_attn_1_type": "flash_attn3",
        "cross_attn_2_type": "flash_attn3",
        "rms_norm_type": "torch",
    }
    if contiguous:
        config["cpu_offload_layout"] = "contiguous"
    return config


def _unloaded(index, cuda=False, contiguous=True):
    block = WanTransformerAttentionBlock(index, "i2v", "fp8-vllm", _config(contiguous), create_cuda_buffer=cuda)
    weights = {}
    for _, leaf in block.named_weight_leaves():
        for name, attr, _ in getattr(leaf, "base_attrs", ()):
            if isinstance(leaf, MMWeightWfp8channelAfp8channeldynamicVllm) and attr == "weight":
                value = (torch.arange(24).reshape(4, 6) + index).to(torch.float8_e4m3fn)
            elif attr == "weight_scale":
                value = torch.full((4, 1), index + 0.3125, dtype=torch.bfloat16)
            else:
                value = torch.arange(4, dtype=torch.bfloat16) + index + 0.25
            weights[name] = value
        if isinstance(leaf, DefaultTensor):
            weights[leaf.tensor_name] = torch.full((1, 6, 4), index + 0.5, dtype=torch.bfloat16)
    return block, weights


def _block(index, cuda=False, contiguous=True):
    block, weights = _unloaded(index, cuda, contiguous)
    block.load(weights)
    if contiguous and not cuda:
        assert not weights  # No checkpoint copy remains in the source mapping.
    return block


def _state(block):
    return {re.sub(r"\.\d+", ".0", key, count=1): tensor for key, tensor in block.state_dict().items()}


def _assert_equal(actual, expected):
    a, b = _state(actual), _state(expected)
    assert a.keys() == b.keys()
    for key, x in a.items():
        y = b[key]
        assert (x.shape, x.stride(), x.dtype) == (y.shape, y.stride(), y.dtype), key
        assert torch.equal(x.reshape(-1).view(torch.uint8).cpu(), y.reshape(-1).view(torch.uint8).cpu()), key


def test_cpu_load_fills_final_views_without_individual_pin_allocations(monkeypatch):
    baseline = _block(0, contiguous=False)

    def forbidden(*args, **kwargs):
        raise AssertionError("an individual pinned tensor was allocated")

    monkeypatch.setattr("lightx2v.common.ops.utils.create_pin_tensor", forbidden)
    monkeypatch.setattr(DefaultTensor, "_create_cpu_pin_tensor", forbidden)
    block = _block(0)
    _assert_equal(block, baseline)
    storage = block.block_buffer.storage
    tensors = [t for t in block.state_dict().values() if t.device.type == "cpu"]
    assert len(tensors) == 44
    assert all(t.is_pinned() and t.untyped_storage().data_ptr() == storage.data_ptr() for t in tensors)
    assert all(t.data_ptr() % 256 == 0 for t in tensors)
    assert any(not t.is_contiguous() for t in tensors)
    assert all(t.is_pinned() for name, t in block.state_dict().items() if name.endswith("weight_scale"))


@pytest.mark.parametrize("contiguous", [False, True])
def test_manager_initial_load_swap_and_wraparound(contiguous):
    blocks = [_block(i, contiguous=contiguous) for i in range(3)]
    slots = [_block(i, cuda=True, contiguous=contiguous) for i in range(2)]
    reference = _block(0, cuda=True, contiguous=False)
    manager = WeightAsyncStreamManager("block")
    manager.init_cuda_buffer(slots)
    if contiguous:
        manager.init_contiguous_blocks(blocks)
        versions = [b.block_buffer.storage._version for b in blocks]
        pointers = [s.block_buffer.storage.data_ptr() for s in slots]
    manager.init_first_buffer(blocks)
    for index in (0, 1, 2, 0, 1, 2):
        manager.prefetch_weights((index + 1) % 3, blocks)
        reference.load_state_dict(blocks[index].state_dict(), index)
        torch.cuda.synchronize()
        _assert_equal(manager.cuda_buffers[0], reference)
        with torch.cuda.stream(manager.compute_stream):
            result = manager.cuda_buffers[0].compute_phases[0].self_attn_q.weight.float().sum()
        manager.swap_blocks()
        assert torch.equal(result.cpu(), blocks[index].compute_phases[0].self_attn_q.pin_weight.float().sum())
    if contiguous:
        assert versions == [b.block_buffer.storage._version for b in blocks]
        assert pointers == [s.block_buffer.storage.data_ptr() for s in slots]
        for slot in slots:
            tensors = [getattr(leaf, f"{attr}_cuda_buffer") for _, leaf in slot.named_weight_leaves() for _, attr, _ in getattr(leaf, "base_attrs", ())]
            assert all(t.untyped_storage().data_ptr() == slot.block_buffer.storage.data_ptr() for t in tensors)
        manager.contiguous_transfer.close()


def test_transfer_waits_for_setup_stream():
    setup = torch.cuda.Stream()
    with torch.cuda.stream(setup):
        blocks = [_block(i) for i in range(2)]
        slots = [_block(i, cuda=True) for i in range(2)]
        torch.cuda._sleep(2_000_000)
        for block in blocks:
            for value in block.block_auxiliary.values():
                value.fill_(3.5)
        transfer = ContiguousBlockTransfer(blocks, slots)
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        transfer.copy(0, slots[0])
    transfer.close()
    _assert_equal(slots[0], blocks[0])
    with pytest.raises(RuntimeError, match="closed"):
        transfer.copy(1, slots[1])


def test_transfer_uses_auxiliary_tensors_after_loading(monkeypatch):
    original = RMSWeightTemplate.load

    def replace_diff(self, weight_dict):
        original(self, weight_dict)
        self.weight_diff = torch.full_like(self.weight_diff, 3.5)

    monkeypatch.setattr(RMSWeightTemplate, "load", replace_diff)
    blocks = [_block(0)]
    slots = [_block(i, cuda=True) for i in range(2)]
    manager = WeightAsyncStreamManager("block")
    manager.init_cuda_buffer(slots)
    manager.init_contiguous_blocks(blocks)
    for value in slots[0].block_auxiliary.values():
        value.zero_()
    torch.cuda.synchronize()
    manager.init_first_buffer(blocks)
    _assert_equal(manager.cuda_buffers[0], blocks[0])
    assert all(tensor.item() == 3.5 for tensor in manager.cuda_buffers[0].block_auxiliary.values())


def test_rejects_unmapped_checkpoint_weights_before_consumption():
    block, weights = _unloaded(0)
    weights["blocks.0.unknown"] = torch.ones(2)
    keys = set(weights)
    with pytest.raises(ValueError, match="Unmapped"):
        block.load(weights)
    assert set(weights) == keys
    assert not hasattr(block, "block_buffer")


def test_rejects_postprocessing_that_reallocates(monkeypatch):
    original = MMWeightWfp8channelAfp8channeldynamicVllm.post_process

    def reallocate(self):
        original(self)
        if getattr(self, "pin_weight_scale", None) is not None:
            self.pin_weight_scale = self.pin_weight_scale.clone()

    monkeypatch.setattr(MMWeightWfp8channelAfp8channeldynamicVllm, "post_process", reallocate)
    with pytest.raises(ValueError, match="escaped"):
        _block(0)


def test_rejects_different_block_layouts():
    blocks = [_block(0)]
    block, weights = _unloaded(1)
    weights["blocks.1.modulation"] = torch.ones((1, 6, 5), dtype=torch.bfloat16)
    block.load(weights)
    blocks.append(block)
    with pytest.raises(ValueError, match="layouts"):
        ContiguousBlockTransfer(blocks, [_block(i, cuda=True) for i in range(2)])


@pytest.mark.parametrize(
    "key,value", [("shared_cpu_weights", True), ("lazy_load", True), ("parallel", {"tensor_p": 2}), ("lora_path", "adapter"), ("enable_cuda_graph", True), ("dit_quant_scheme", "nvfp4")]
)
def test_rejects_unsupported_config(key, value):
    config = _config()
    config[key] = value
    with pytest.raises(ValueError):
        validate_contiguous_config(config)
