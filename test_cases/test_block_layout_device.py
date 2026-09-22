"""Run on CUDA for regression, or PLATFORM=ascend_npu on an Ascend device."""

import re

import pytest
import torch

from lightx2v.common.offload.manager import WeightAsyncStreamManager
from lightx2v.common.offload.timing import TransferTimer
from lightx2v.common.ops.tensor.tensor import DefaultTensor
from lightx2v.models.networks.wan.weights.block_layout import validate_contiguous_config
from lightx2v.models.networks.wan.weights.transformer_weights import WanTransformerAttentionBlock
from lightx2v.utils.registry_factory import LN_WEIGHT_REGISTER, MM_WEIGHT_REGISTER, RMS_WEIGHT_REGISTER
from lightx2v_platform.base.global_var import AI_DEVICE
from lightx2v_platform.ops.mm.ascend_npu.mm_weight import MMWeightWint8channelAint8channeldynamicNpu
from lightx2v_platform.ops.norm.ascend_npu.npu_layer_norm import NpuLayerNormWeight
from lightx2v_platform.ops.norm.ascend_npu.npu_rms_norm import NpuRmsNormWeight

device_module = getattr(torch, AI_DEVICE)
pytestmark = pytest.mark.skipif(AI_DEVICE not in ("cuda", "npu") or not device_module.is_available(), reason="requires CUDA or Ascend")


@pytest.fixture(autouse=True)
def platform_operators(monkeypatch):
    # Exercise the real platform loaders on CUDA too; only NPU executes INT8 MM.
    monkeypatch.setitem(MM_WEIGHT_REGISTER, "int8-npu", MMWeightWint8channelAint8channeldynamicNpu)
    monkeypatch.setitem(RMS_WEIGHT_REGISTER, "npu_rms_norm", NpuRmsNormWeight)
    monkeypatch.setitem(LN_WEIGHT_REGISTER, "npu_layer_norm", NpuLayerNormWeight)
    if AI_DEVICE == "npu":
        torch.npu.config.allow_internal_format = False


def make_block(index, scheme, contiguous, device_buffer=False):
    attention = "npu_flash_attn" if AI_DEVICE == "npu" else "torch_sdpa"
    config = {
        "model_cls": "wan2.1",
        "task": "i2v",
        "dit_quantized": scheme != "Default",
        "dit_quant_scheme": scheme,
        "cpu_offload": True,
        "offload_granularity": "block",
        "seq_parallel": False,
        "self_attn_1_type": attention,
        "cross_attn_1_type": attention,
        "cross_attn_2_type": attention,
        "rms_norm_type": "npu_rms_norm",
        "layer_norm_type": "npu_layer_norm",
        "cpu_offload_layout": "contiguous" if contiguous else "per_tensor",
    }
    if AI_DEVICE == "npu" or scheme == "Default":
        validate_contiguous_config(config)
    block = WanTransformerAttentionBlock(index, "i2v", scheme, config, create_cuda_buffer=device_buffer)
    weights = {}
    for _, leaf in block.named_weight_leaves():
        for name, attr, transpose in getattr(leaf, "base_attrs", ()):
            if transpose:
                dtype = torch.int8 if scheme == "int8-npu" else torch.bfloat16
                value = (torch.arange(1024).reshape(32, 32) % 16 + index).to(dtype)
            elif attr == "weight_scale":
                value = torch.full((32, 1), 0.125 + index / 32, dtype=torch.bfloat16)
            else:
                value = torch.full((32,), 0.25 + index, dtype=torch.bfloat16)
            weights[name] = value
        if isinstance(leaf, DefaultTensor):
            weights[leaf.tensor_name] = torch.full((1, 6, 32), index + 0.5, dtype=torch.bfloat16)
    block.load(weights)
    if contiguous and not device_buffer:
        assert not weights
    return block


def state(block):
    return {re.sub(r"\.\d+", ".0", name, count=1): t for name, t in block.state_dict().items() if t is not None}


@pytest.mark.parametrize("scheme", ["Default", "int8-npu"])
def test_platform_loaders_and_double_buffer_roundtrip(scheme):
    baseline = [make_block(i, scheme, False) for i in range(3)]
    blocks = [make_block(i, scheme, True) for i in range(3)]
    for expected, actual in zip(baseline, blocks):
        a, b = state(expected), state(actual)
        assert a.keys() == b.keys()
        for name in a:
            assert torch.equal(a[name], b[name]), name
            assert a[name].stride() == b[name].stride()
            assert b[name].is_pinned()
            assert b[name].untyped_storage().data_ptr() == actual.block_buffer.storage.data_ptr()
    versions = [block.block_buffer.storage._version for block in blocks]
    slots = [make_block(i, scheme, True, True) for i in range(2)]
    reference = make_block(0, scheme, False, True)
    manager = WeightAsyncStreamManager("block")
    manager.init_cuda_buffer(slots)
    manager.init_contiguous_blocks(blocks)
    manager.init_first_buffer(blocks)
    for index in (0, 1, 2, 0, 1, 2):
        reference.load_state_dict(baseline[index].state_dict(), index)
        device_module.synchronize()
        active = manager.cuda_buffers[0]
        for name, value in state(active).items():
            assert torch.equal(value.cpu(), state(reference)[name].cpu()), name
        if scheme == "Default" or AI_DEVICE == "npu":
            x = torch.arange(64, device=AI_DEVICE).to(torch.bfloat16).reshape(2, 32) / 32
            q = active.compute_phases[0].self_attn_q
            expected = reference.compute_phases[0].self_attn_q.apply(x)
            torch.testing.assert_close(q.apply(x), expected, rtol=0, atol=0)
        manager.prefetch_weights((index + 1) % 3, blocks)
        manager.swap_blocks()
    assert versions == [block.block_buffer.storage._version for block in blocks]
    manager.contiguous_transfer.close()


@pytest.mark.parametrize("contiguous", [False, True])
def test_bounded_timer_records_actual_manager_loads(contiguous):
    blocks = [make_block(i, "Default", contiguous) for i in range(2)]
    slots = [make_block(i, "Default", contiguous, True) for i in range(2)]
    manager = WeightAsyncStreamManager("block")
    manager.init_cuda_buffer(slots)
    if contiguous:
        manager.init_contiguous_blocks(blocks)
    timer = TransferTimer(device_module, [{"h2d_bytes": 4096}] * 2, limit=2)
    manager.transfer_timer = timer
    manager.init_first_buffer(blocks)
    manager.prefetch_weights(1, blocks)
    manager.swap_blocks()
    manager.prefetch_weights(0, blocks)
    device_module.synchronize()
    rows = timer.collect()
    assert [(row["block_index"], row["phase"]) for row in rows] == [(0, "initial"), (1, "prefetch")]
    assert all(row["submit_ms"] >= 0 and row["load_stream_ms"] > 0 for row in rows)


def test_npu_int8_rejects_fp8_checkpoint():
    op = MMWeightWint8channelAint8channeldynamicNpu("blocks.0.q.weight", None)
    with pytest.raises(ValueError, match="INT8 checkpoint"):
        op.load({"blocks.0.q.weight": torch.zeros(32, 32).to(torch.float8_e4m3fn)})


@pytest.mark.parametrize(
    "dtype,scheme,accepted", [(torch.bfloat16, "Default", True), (torch.int8, "int8-npu", True), (torch.float8_e4m3fn, "Default", False), (torch.float8_e4m3fn, "int8-npu", False)]
)
def test_checkpoint_precision_preflight(tmp_path, dtype, scheme, accepted):
    from safetensors.torch import save_file

    from scripts.wan.layout.benchmark import validate_npu_checkpoint

    checkpoint = tmp_path / "block.safetensors"
    save_file({"blocks.0.self_attn.q.weight": torch.zeros(32, 32).to(dtype)}, checkpoint)
    if accepted:
        validate_npu_checkpoint(checkpoint, scheme)
    else:
        with pytest.raises(ValueError, match="No implicit FP8 conversion"):
            validate_npu_checkpoint(checkpoint, scheme)


def test_t5_npu_offload_norm_uses_portable_compute(monkeypatch):
    from lightx2v.models.input_encoders.hf.wan.t5 import model

    monkeypatch.setattr(model, "AI_DEVICE", "npu")
    block = model.T5OffloadSelfAttention(0, None, create_cuda_buffer=True)
    for norm in (block.norm1, block.norm2):
        assert isinstance(norm, RMS_WEIGHT_REGISTER["torch"])
        weight = torch.linspace(0.5, 1.5, 32, dtype=torch.bfloat16)
        norm.load({norm.weight_name: weight})
        norm.load_state_dict({norm.weight_name: weight}, 0)
        x = torch.arange(64, device=AI_DEVICE).to(torch.bfloat16).reshape(1, 2, 32) / 32
        expected = (x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + norm.eps) * weight.to(AI_DEVICE)).to(torch.bfloat16)
        torch.testing.assert_close(norm.apply(x), expected, rtol=0.02, atol=0.02)
