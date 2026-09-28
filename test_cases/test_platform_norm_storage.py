"""Platform norms inherit storage support without chip-specific loading code."""

import pytest
import torch

from lightx2v.common.modules.weight_module import WeightModule
from lightx2v.common.offload.block_loader import OffloadGroup, prepare_contiguous_groups
from lightx2v.common.offload.manager import WeightAsyncStreamManager
from lightx2v_platform.base.global_var import AI_DEVICE, PLATFORM
from lightx2v_platform.ops.norm import norm_template
from lightx2v_platform.ops.weight_storage import TensorMetadata
from lightx2v_platform.registry_factory import PLATFORM_DEVICE_REGISTER


class TemplateRMS(norm_template.RMSWeightTemplate):
    def apply(self, x):
        return torch.nn.functional.rms_norm(x, (x.shape[-1],), self.weight, self.eps)


class TemplateLayerNorm(norm_template.LayerNormWeightTemplate):
    def apply(self, x):
        return torch.nn.functional.layer_norm(x, (x.shape[-1],), self.weight, self.bias, self.eps)


@pytest.fixture(params=[torch.bfloat16, torch.float16])
def norm_dtype(request, monkeypatch):
    monkeypatch.setattr(norm_template, "GET_DTYPE", lambda: request.param)
    monkeypatch.setattr(norm_template, "GET_SENSITIVE_DTYPE", lambda: request.param)
    return request.param


@pytest.fixture(params=["template", "ascend"])
def norm_classes(request):
    if request.param == "ascend":
        from lightx2v_platform.ops.norm.ascend_npu.npu_layer_norm import NpuLayerNormWeight
        from lightx2v_platform.ops.norm.ascend_npu.npu_rms_norm import NpuRmsNormWeight

        return NpuRmsNormWeight, NpuLayerNormWeight
    return TemplateRMS, TemplateLayerNorm


def make_block(classes, index=0, device_buffer=False):
    rms, ln = classes
    prefix = f"blocks.{index}."
    options = {"create_cuda_buffer": device_buffer}
    block = WeightModule()
    block.add_module("rms", rms(prefix + "rms.weight", **options))
    block.add_module("rms_plain", rms(None, **options))
    block.add_module("ln", ln(prefix + "ln.weight", prefix + "ln.bias", **options))
    block.add_module("ln_weight_only", ln(prefix + "ln_weight_only.weight", **options))
    block.add_module("ln_plain", ln(**options))
    return block


def source_weights(dtype, index):
    return {f"blocks.{index}.{name}": torch.linspace(0.125, 1.0, 32, dtype=dtype) + index for name in ("rms.weight", "ln.weight", "ln.bias", "ln_weight_only.weight")}


def test_descriptions_use_checkpoint_shape_and_inference_dtype(norm_classes, norm_dtype):
    block = make_block(norm_classes)
    source = source_weights(torch.float32, 0)
    metadata = {name: TensorMetadata(tuple(tensor.shape), tensor.dtype) for name, tensor in source.items()}
    descriptions = [leaf.describe_storage(metadata) for _, leaf in block.named_weight_leaves()]
    specs = [spec for description in descriptions for spec in description.tensors]
    assert {spec.name for spec in specs} == set(source)
    assert len(specs) == len(source)
    assert all(spec.dtype == norm_dtype and spec.shape == (32,) and not spec.transpose for spec in specs)
    assert all(not description.auxiliary for description in descriptions)


@pytest.mark.parametrize("name", ["rms.weight", "ln.weight", "ln.bias"])
def test_invalid_checkpoint_dtype_is_rejected(norm_classes, name):
    source = source_weights(torch.bfloat16, 0)
    metadata = {key: TensorMetadata(tuple(tensor.shape), tensor.dtype) for key, tensor in source.items()}
    # Preserve raw checkpoint dtype even when a prior loader already cast to BF16.
    metadata["blocks.0." + name] = TensorMetadata((32,), torch.int8, torch.bfloat16)
    block = make_block(norm_classes)
    with pytest.raises(ValueError, match="checkpoint dtype"):
        for _, leaf in block.named_weight_leaves():
            leaf.describe_storage(metadata)


@pytest.mark.parametrize("mode", ["default", "device", "lazy_cpu"])
def test_parameter_free_norms_remain_stateless(norm_classes, mode):
    rms, ln = norm_classes
    options = dict(create_cuda_buffer=mode == "device", create_cpu_buffer=mode == "lazy_cpu", lazy_load=mode == "lazy_cpu")
    for norm in (rms(None, **options), ln(**options)):
        norm.load({})
        norm.load_state_dict({}, block_index=1)
        norm.load_state_dict_from_disk(block_index=1)
        norm.to_cuda()
        norm.to_cpu()
        assert norm.state_dict() == {}
        assert norm.weight is None
        assert not norm.describe_storage({}).tensors


def test_baseline_and_contiguous_norms_match_across_slot_reuse(norm_classes, norm_dtype):
    module = getattr(torch, AI_DEVICE)
    if not PLATFORM_DEVICE_REGISTER[PLATFORM].is_available():
        pytest.skip("requires one accelerator for pinned memory and device copies")
    from lightx2v_platform.base.offload import get_block_offload_backend

    get_block_offload_backend().prepare()
    baseline = [make_block(norm_classes, index) for index in range(3)]
    reference = make_block(norm_classes, device_buffer=True)
    reference.load(source_weights(norm_dtype, 0))
    for index, block in enumerate(baseline):
        block.load(source_weights(norm_dtype, index))

    blocks = [make_block(norm_classes, index) for index in range(3)]
    slots = [make_block(norm_classes, device_buffer=True) for _ in range(2)]
    sources = {name: tensor for index in range(3) for name, tensor in source_weights(norm_dtype, index).items()}
    group = OffloadGroup(blocks, slots, tuple(f"blocks.{index}." for index in range(3)))
    prepare_contiguous_groups([group], sources)
    # Match model loading order: allocate both device slots before consuming CPU sources.
    for block in (*slots, *blocks):
        block.load(sources)
    assert not sources
    for block, expected in zip(blocks, baseline):
        actual_state, expected_state = block.state_dict(), expected.state_dict()
        assert actual_state.keys() == expected_state.keys()
        for name, tensor in actual_state.items():
            assert tensor.is_pinned()
            assert tensor.untyped_storage().data_ptr() == block.block_buffer.storage.untyped_storage().data_ptr()
            torch.testing.assert_close(tensor, expected_state[name], rtol=0, atol=0)

    manager = WeightAsyncStreamManager("block")
    manager.init_contiguous_groups([group])
    manager.init_first_buffer(blocks)
    pointers = [[tensor.data_ptr() for tensor in slot.state_dict().values()] for slot in slots]
    versions = [block.block_buffer.storage._version for block in blocks]
    x = torch.linspace(-1, 1, 64, device=AI_DEVICE, dtype=norm_dtype).reshape(2, 32)
    try:
        for index in (0, 1, 2, 0, 1, 2):
            reference.load_state_dict(baseline[index].state_dict(), index)
            module.synchronize()
            with torch.no_grad(), module.stream(manager.compute_stream):
                for (path, actual), (_, expected) in zip(manager.cuda_buffers[0].named_weight_leaves(), reference.named_weight_leaves()):
                    torch.testing.assert_close(actual.apply(x), expected.apply(x), rtol=0, atol=0, msg=path)
            manager.prefetch_weights((index + 1) % len(blocks), blocks)
            manager.swap_blocks()
    finally:
        manager.compute_stream.synchronize()
        manager.contiguous_transfer.close()
    assert pointers == [[tensor.data_ptr() for tensor in slot.state_dict().values()] for slot in slots]
    assert versions == [block.block_buffer.storage._version for block in blocks]
