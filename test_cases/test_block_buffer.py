"""Storage boundaries and block transfers; PLATFORM=ascend_npu selects NPU tests."""

import gc
import importlib.util
import os
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

# Keep storage tests usable without importing model and kernel dependencies.
module_spec = importlib.util.spec_from_file_location("block_buffer_under_test", Path(__file__).parents[1] / "lightx2v/common/offload/block_layout.py")
buffers = importlib.util.module_from_spec(module_spec)
sys.modules[module_spec.name] = buffers
module_spec.loader.exec_module(buffers)


@pytest.fixture
def layout():
    return buffers.BlockLayout.build(
        [
            (("q", "weight"), "q", (2, 3), torch.bfloat16, True),
            (("scale", "weight"), "scale", (3,), torch.float32, False),
            (("z", "weight"), "z", (5,), torch.int8, False),
        ]
    )


def aligned_storage(nbytes, device="cpu", pin_memory=False):
    allocation = torch.empty(nbytes + 255, dtype=torch.uint8, device=device, pin_memory=pin_memory)
    start = (-allocation.data_ptr()) % 256
    return allocation.narrow(0, start, nbytes)


def sources_for(layout, index=0):
    return {spec.name: (torch.arange(spec.nbytes // spec.dtype.itemsize).reshape(spec.shape) + index).to(spec.dtype) for spec in layout.tensors}


def test_external_region_preserves_storage_and_transposed_values(layout):
    parent = aligned_storage(256 + layout.nbytes + 256)
    parent.fill_(113)
    buffer = buffers.BlockBuffer(layout, parent[256:])
    assert buffer.storage.data_ptr() == parent.data_ptr() + 256
    assert buffer.storage.numel() == layout.nbytes
    assert torch.all(parent == 113)  # Binding does not initialize or copy data.

    sources = sources_for(layout)
    context = buffers.BlockLoadContext(dict(sources), buffer)
    for spec in layout.tensors:
        actual = context.take(spec.name, transpose=spec.transpose)
        expected = sources[spec.name].t() if spec.transpose else sources[spec.name]
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        assert actual.stride() == expected.stride()
        assert actual.data_ptr() % 256 == 0
        assert actual.untyped_storage().data_ptr() == parent.untyped_storage().data_ptr()
        view = buffer.view(spec, operator=True)
        assert (view.data_ptr(), view.stride()) == (actual.data_ptr(), actual.stride())
    context.finish()
    assert not context.sources
    assert torch.all(parent[:256] == 113)
    assert torch.all(parent[256 + layout.nbytes :] == 113)


def test_tensor_view_keeps_external_allocation_alive(layout):
    parent = aligned_storage(layout.nbytes)
    buffer = buffers.BlockBuffer(layout, parent)
    view = buffer.view(layout.tensors[0], operator=True)
    view.fill_(7)
    pointer = view.data_ptr()
    del parent, buffer
    gc.collect()
    assert view.data_ptr() == pointer
    torch.testing.assert_close(view, torch.full((3, 2), 7, dtype=torch.bfloat16))
    view.add_(1)
    assert torch.all(view == 8)


@pytest.mark.parametrize("case", ["dtype", "rank", "strided", "short", "address"])
def test_invalid_external_storage_is_rejected(layout, case):
    parent = aligned_storage(2 * layout.nbytes + 256)
    invalid = {
        "dtype": parent.view(torch.int8),
        "rank": parent.unsqueeze(0),
        "strided": parent[::2],
        "short": parent[: layout.nbytes - 1],
        "address": parent[1:],
    }[case]
    with pytest.raises(ValueError):
        buffers.BlockBuffer(layout, invalid)


@pytest.mark.parametrize("offset", [-256, 1, 768])
def test_tensor_view_cannot_escape_block_region(layout, offset):
    # Parent storage is large enough; only the block's own range may be used.
    buffer = buffers.BlockBuffer(layout, aligned_storage(2048))
    spec = replace(layout.tensors[0], offset=offset)
    with pytest.raises(ValueError):
        buffer.view(spec)


@pytest.mark.parametrize("shape", [(0, 3), (-2, -3)])
def test_invalid_dimensions_are_rejected(shape):
    with pytest.raises(ValueError):
        buffers.BlockLayout.build([(("q", "weight"), "q", shape, torch.bfloat16, False)])


def test_loading_requires_every_planned_tensor(layout):
    buffer = buffers.BlockBuffer(layout, aligned_storage(layout.nbytes))
    context = buffers.BlockLoadContext(sources_for(layout), buffer)
    context.take("q")
    with pytest.raises(ValueError, match="complete block layout"):
        context.finish()
    with pytest.raises(KeyError):
        context.take("q")
    context.take("scale")
    context.take("z")
    context.finish()


def test_allocate_only_adds_alignment_slack_when_required(monkeypatch, layout):
    # Simulate both aligned and unaligned backends without a GPU dependency.
    original_empty = torch.empty
    for remainder in (0, 64):
        calls = []

        def allocate(size, **kwargs):
            calls.append((size, kwargs))
            parent = original_empty(size + 512, dtype=torch.uint8)
            start = (-parent.data_ptr()) % 256 + remainder
            return parent.narrow(0, start, size).fill_(113)

        with monkeypatch.context() as patch:
            patch.setattr(buffers.torch, "empty", allocate)
            buffer = buffers.BlockBuffer.allocate(layout, "cpu")
        sizes = [size for size, _ in calls]
        assert sizes == ([layout.nbytes] if remainder == 0 else [layout.nbytes, layout.nbytes + 255])
        assert all(kwargs["pin_memory"] for _, kwargs in calls)
        assert buffer.storage.data_ptr() % 256 == 0
        assert buffer.storage.numel() == layout.nbytes
        end = 0
        for spec in layout.tensors:
            assert torch.all(buffer.storage[end : spec.offset] == 0)
            assert torch.all(buffer.storage[spec.offset : spec.offset + spec.nbytes] == 113)
            end = spec.offset + spec.nbytes


@pytest.fixture
def accelerator():
    device = "npu" if os.environ.get("PLATFORM") == "ascend_npu" else "cuda"
    if device == "npu":
        pytest.importorskip("torch_npu")
        torch.npu.config.allow_internal_format = False
    module = getattr(torch, device)
    if not module.is_available():
        pytest.skip(f"requires one {device} device")
    return device, module


def test_real_allocation_is_pinned_and_exactly_block_sized(layout, accelerator):
    device, _ = accelerator
    cpu = buffers.BlockBuffer.allocate(layout, "cpu")
    target = buffers.BlockBuffer.allocate(layout, device)
    assert cpu.storage.is_pinned()
    for block in (cpu, target):
        assert block.storage.data_ptr() % 256 == 0
        assert block.storage.numel() == layout.nbytes


def test_async_double_buffer_copy_preserves_external_guards(layout, accelerator):
    device, module = accelerator
    blocks, slots, parents = [], [], []
    for index in range(3):
        parent = aligned_storage(256 + layout.nbytes + 256, pin_memory=True).fill_(113)
        buffer = buffers.BlockBuffer(layout, parent[256:])
        context = buffers.BlockLoadContext(sources_for(layout, index), buffer)
        for spec in layout.tensors:
            context.take(spec.name)
        context.finish()
        assert buffer.storage.is_pinned()
        blocks.append(SimpleNamespace(block_buffer=buffer, block_auxiliary={"scale": torch.tensor(float(index), device=device)}))
        parents.append(parent)
    for _ in range(2):
        parent = aligned_storage(256 + layout.nbytes + 256, device=device).fill_(113)
        buffer = buffers.BlockBuffer(layout, parent[256:])
        context_sources = sources_for(layout)
        context = buffers.BlockLoadContext(context_sources, buffer)
        for spec in layout.tensors:
            context.take(spec.name)
        context.finish()
        assert set(context_sources) == {spec.name for spec in layout.tensors}
        slots.append(SimpleNamespace(block_buffer=buffer, block_auxiliary={"scale": torch.empty((), device=device)}))
        parents.append(parent)

    pointers = [slot.block_buffer.storage.data_ptr() for slot in slots]
    source_copies = [parent.clone() for parent in parents[:3]]
    from lightx2v_platform.base.offload import TorchBlockOffload

    transfer = buffers.ContiguousBlockTransfer(blocks, slots, backend=TorchBlockOffload)
    load_stream, compute_stream = module.Stream(), module.Stream()
    transfer.copy(0, slots[0])
    module.synchronize()
    try:
        for index in (0, 1, 2, 0, 1, 2):
            with module.stream(load_stream):
                transfer.copy((index + 1) % 3, slots[1])
            with module.stream(compute_stream):
                values = [slots[0].block_buffer.view(spec, operator=True).clone() for spec in layout.tensors]
                scale = slots[0].block_auxiliary["scale"].clone()
            load_stream.synchronize()
            compute_stream.synchronize()
            for spec, value in zip(layout.tensors, values):
                expected = blocks[index].block_buffer.view(spec, operator=True)
                torch.testing.assert_close(value.cpu(), expected, rtol=0, atol=0)
            assert scale.item() == index
            slots.reverse()
    finally:
        compute_stream.synchronize()
        transfer.close()
    assert pointers == [slot.block_buffer.storage.data_ptr() for slot in slots]
    for parent, expected in zip(parents, source_copies):
        assert torch.equal(parent, expected)
    for parent in parents:
        assert torch.all(parent[:256].cpu() == 113)
        assert torch.all(parent[256 + layout.nbytes :].cpu() == 113)
