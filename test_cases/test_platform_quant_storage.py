"""Shared per-channel storage, with native kernels only on their own devices."""

from dataclasses import replace

import pytest
import torch

from lightx2v.common.modules.weight_module import WeightModule
from lightx2v.common.offload.block_loader import OffloadGroup, prepare_contiguous_groups
from lightx2v.common.offload.manager import WeightAsyncStreamManager
from lightx2v_platform.base.global_var import AI_DEVICE, PLATFORM
from lightx2v_platform.base.offload import get_block_offload_backend
from lightx2v_platform.ops.mm.ascend_npu.mm_weight import MMWeightWint8channelAint8channeldynamicNpu
from lightx2v_platform.ops.mm.cambricon_mlu.mm_weight import MMWeightWint8channelAint8channeldynamicMlu
from lightx2v_platform.ops.mm.mthreads_musa.mm_weight import MMWeightWfp8channelAfp8tokendynamicMusa
from lightx2v_platform.ops.mm.template import MMWeightPerChannelQuantTemplate
from lightx2v_platform.ops.weight_storage import TensorMetadata
from lightx2v_platform.registry_factory import PLATFORM_DEVICE_REGISTER


class TemplateQuantMM(MMWeightPerChannelQuantTemplate):
    checkpoint_dtype = torch.int8

    def apply(self, x):
        weight = self.weight.t() if self.weight_need_transpose else self.weight
        return torch.nn.functional.linear(x.float(), weight.float() * self.weight_scale.reshape(-1, 1), self.bias.float() if self.bias is not None else None)


@pytest.fixture(
    params=[
        (None, TemplateQuantMM),
        ("ascend_npu", MMWeightWint8channelAint8channeldynamicNpu),
        ("cambricon_mlu", MMWeightWint8channelAint8channeldynamicMlu),
        ("musa", MMWeightWfp8channelAfp8tokendynamicMusa),
    ]
)
def implementation(request):
    return request.param


def weights(cls, index=0, bias=True):
    prefix = f"blocks.{index}.proj"
    source = {
        prefix + ".weight": (torch.arange(32 * 64).reshape(32, 64) % 7 + index).to(cls.checkpoint_dtype),
        prefix + ".weight_scale": torch.full((32, 1), 0.125 + index / 32, dtype=torch.bfloat16),
    }
    if bias:
        source[prefix + ".bias"] = torch.linspace(-0.5, 0.5, 32, dtype=torch.bfloat16)
    return source


def block(cls, index=0, bias=True, device_buffer=False):
    owner = WeightModule()
    prefix = f"blocks.{index}.proj"
    owner.add_module("proj", cls(prefix + ".weight", prefix + ".bias" if bias else None, create_cuda_buffer=device_buffer))
    return owner


@pytest.mark.parametrize("force_fp32", [False, True])
def test_metadata_matches_baseline_scale_bias_and_transpose(implementation, force_fp32):
    _, cls = implementation
    owner = block(cls)
    owner.proj.bias_force_fp32 = force_fp32
    source = weights(cls)
    metadata = {name: TensorMetadata(tuple(t.shape), t.dtype) for name, t in source.items()}
    name = "blocks.0.proj.bias"
    metadata[name] = replace(metadata[name], dtype=torch.float32, loaded_dtype=torch.float16)
    specs = {spec.attr: spec for spec in owner.proj.describe_storage(metadata).tensors}
    assert specs["weight"].dtype == cls.checkpoint_dtype
    assert specs["weight"].transpose == (cls is not MMWeightWint8channelAint8channeldynamicMlu)
    assert specs["weight_scale"].dtype == torch.float32
    assert specs["bias"].dtype == (torch.float32 if force_fp32 else torch.float16)
    # A subclass's final layout setting must also govern binding, not a stale mapping.
    owner.proj.weight_need_transpose = not owner.proj.weight_need_transpose
    updated = owner.proj.describe_storage(metadata).tensors[0]
    assert updated.transpose != specs["weight"].transpose
    assert owner.proj.base_attrs[0][2] == updated.transpose


@pytest.mark.parametrize("field", ["dtype", "loaded_dtype"])
def test_rejects_incompatible_checkpoint_before_loading(implementation, field):
    _, cls = implementation
    source = weights(cls)
    metadata = {name: TensorMetadata(tuple(t.shape), t.dtype) for name, t in source.items()}
    name = "blocks.0.proj.weight"
    metadata[name] = replace(metadata[name], **{field: torch.bfloat16})
    with pytest.raises(ValueError, match="checkpoint dtype"):
        block(cls).proj.describe_storage(metadata)
    assert set(source) == {"blocks.0.proj.weight", "blocks.0.proj.weight_scale", "blocks.0.proj.bias"}


@pytest.mark.parametrize("bias", [False, True])
def test_per_tensor_and_contiguous_quantized_blocks_match(implementation, bias):
    kernel_platform, cls = implementation
    if kernel_platform is not None and PLATFORM not in ("cuda", kernel_platform):
        pytest.skip("vendor storage is checked on CUDA or its native platform")
    if not PLATFORM_DEVICE_REGISTER[PLATFORM].is_available():
        pytest.skip("requires an accelerator")
    backend = get_block_offload_backend()
    backend.prepare()
    baseline = [block(cls, index, bias) for index in range(3)]
    reference = block(cls, bias=bias, device_buffer=True)
    source = weights(cls, bias=bias)
    reference.load(source)
    assert source  # Device-slot allocation must not consume the CPU block's weights.
    for index, owner in enumerate(baseline):
        source = weights(cls, index, bias)
        owner.load(source)
        assert not source

    blocks = [block(cls, index, bias) for index in range(3)]
    slots = [block(cls, bias=bias, device_buffer=True) for _ in range(2)]
    group = OffloadGroup(blocks, slots, tuple(f"blocks.{i}." for i in range(3)))
    source = {name: tensor for i in range(3) for name, tensor in weights(cls, i, bias).items()}
    prepare_contiguous_groups([group], source)
    for owner in (*slots, *blocks):
        owner.load(source)
    assert not source
    for owner, expected in zip(blocks, baseline):
        assert owner.state_dict().keys() == expected.state_dict().keys()
        for name, tensor in owner.state_dict().items():
            assert tensor.is_pinned()
            assert tensor.untyped_storage().data_ptr() == owner.block_buffer.storage.untyped_storage().data_ptr()
            ref = expected.state_dict()[name]
            assert (tensor.dtype, tensor.shape, tensor.stride()) == (ref.dtype, ref.shape, ref.stride())
            torch.testing.assert_close(tensor.float(), ref.float(), rtol=0, atol=0)

    module = backend.device_module(slots[0].block_buffer.storage.device)
    manager = WeightAsyncStreamManager("block")
    manager.init_contiguous_groups([group])
    manager.init_first_buffer(blocks)
    pointers = [[t.data_ptr() for t in slot.state_dict().values()] for slot in slots]
    versions = [owner.block_buffer.storage._version for owner in blocks]
    x = torch.linspace(-1, 1, 16 * 64, device=AI_DEVICE, dtype=torch.bfloat16).reshape(16, 64)
    try:
        for index in (0, 1, 2, 0, 1, 2):
            reference.load_state_dict(baseline[index].state_dict(), index)
            module.synchronize()
            actual = manager.cuda_buffers[0]
            for name, tensor in actual.state_dict().items():
                ref = reference.state_dict()[name]
                assert (tensor.dtype, tensor.shape, tensor.stride()) == (ref.dtype, ref.shape, ref.stride())
                torch.testing.assert_close(tensor.float(), ref.float(), rtol=0, atol=0)
            assert (actual.proj.bias is None) == (not bias)
            if kernel_platform is None or kernel_platform == PLATFORM:
                with torch.no_grad(), module.stream(manager.compute_stream):
                    torch.testing.assert_close(actual.proj.apply(x), reference.proj.apply(x), rtol=0, atol=0)
            manager.prefetch_weights((index + 1) % len(blocks), blocks)
            manager.swap_blocks()
    finally:
        manager.compute_stream.synchronize()
        manager.contiguous_transfer.close()
    assert pointers == [[t.data_ptr() for t in slot.state_dict().values()] for slot in slots]
    assert versions == [owner.block_buffer.storage._version for owner in blocks]
