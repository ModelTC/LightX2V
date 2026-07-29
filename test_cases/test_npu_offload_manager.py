import importlib.util
import sys
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch


class _StubTensor:
    def __init__(self, numel, dtype, device):
        self._numel = numel
        self.dtype = dtype
        self.device = device

    def dim(self):
        return 1

    def is_contiguous(self):
        return True

    def numel(self):
        return self._numel


def _package(name):
    module = ModuleType(name)
    module.__path__ = []
    return module


def _load_manager_module():
    """Load manager.py without importing torch or the LightX2V package."""
    torch_module = ModuleType("torch")
    torch_module.__version__ = "2.9.0"
    torch_module.cuda = ModuleType("torch.cuda")
    torch_module.Tensor = _StubTensor
    torch_module.uint8 = "uint8"
    torch_module.empty = lambda shape, dtype, device: _StubTensor(shape[0], dtype, device)

    profiler_module = ModuleType("lightx2v.utils.profiler")

    def excluded_profiling_context(*args, **kwargs):
        return lambda function: function

    profiler_module.ExcludedProfilingContext = excluded_profiling_context

    global_var_module = ModuleType("lightx2v_platform.base.global_var")
    global_var_module.AI_DEVICE = "cuda"

    block_slab_module = ModuleType("lightx2v.common.offload.block_slab")
    block_slab_module.carve_block_slab = lambda raw, layout: {}
    block_slab_module.copy_block_slab_ = lambda destination, source, **kwargs: destination

    stub_modules = {
        "torch": torch_module,
        "lightx2v": _package("lightx2v"),
        "lightx2v.common": _package("lightx2v.common"),
        "lightx2v.common.offload": _package("lightx2v.common.offload"),
        "lightx2v.common.offload.block_slab": block_slab_module,
        "lightx2v.utils": _package("lightx2v.utils"),
        "lightx2v.utils.profiler": profiler_module,
        "lightx2v_platform": _package("lightx2v_platform"),
        "lightx2v_platform.base": _package("lightx2v_platform.base"),
        "lightx2v_platform.base.global_var": global_var_module,
    }

    module_path = Path(__file__).resolve().parents[1] / "lightx2v" / "common" / "offload" / "manager.py"
    spec = importlib.util.spec_from_file_location("_npu_offload_manager_under_test", module_path)
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, stub_modules):
        spec.loader.exec_module(module)
    return module


manager_module = _load_manager_module()


class _FakeStream:
    def __init__(self, name, trace):
        self.name = name
        self.trace = trace

    def wait_event(self, event):
        self.trace.append(("wait_event", self.name, event.name))

    def synchronize(self):
        self.trace.append(("synchronize", self.name))


class _FakeEvent:
    def __init__(self, name, trace):
        self.name = name
        self.trace = trace

    def record(self, stream):
        self.trace.append(("record", self.name, stream.name))


class _FakeStreamContext:
    def __init__(self, stream, trace):
        self.stream = stream
        self.trace = trace

    def __enter__(self):
        self.trace.append(("enter_stream", self.stream.name))

    def __exit__(self, exc_type, exc_value, traceback):
        self.trace.append(("exit_stream", self.stream.name))


class _FakeDeviceModule:
    def __init__(self, trace):
        self.trace = trace
        self.stream_count = 0
        self.event_count = 0

    def Stream(self, priority):
        name = f"created-stream-{self.stream_count}"
        self.stream_count += 1
        self.trace.append(("create_stream", name, priority))
        return _FakeStream(name, self.trace)

    def Event(self):
        name = f"event-{self.event_count}"
        self.event_count += 1
        self.trace.append(("create_event", name))
        return _FakeEvent(name, self.trace)

    def stream(self, stream):
        return _FakeStreamContext(stream, self.trace)

    def synchronize(self):
        self.trace.append(("device_synchronize",))


class _FakeBlock:
    def __init__(self, block_idx):
        self.block_idx = block_idx

    def state_dict(self):
        return {"block_idx": self.block_idx}


class _FakeBuffer:
    def __init__(self, name, trace):
        self.name = name
        self.trace = trace

    def load_state_dict(self, state_dict, block_idx, adapter_block_idx):
        self.trace.append(("load", self.name, state_dict["block_idx"], block_idx, adapter_block_idx))


class WeightAsyncStreamManagerNpuTest(unittest.TestCase):
    def _make_npu_manager(self):
        trace = []
        device_module = _FakeDeviceModule(trace)
        load_stream = _FakeStream("shared-load", trace)
        compute_stream = _FakeStream("shared-compute", trace)
        patches = (
            patch.object(manager_module, "AI_DEVICE", "npu"),
            patch.object(manager_module, "torch_device_module", device_module),
        )
        for active_patch in patches:
            active_patch.start()
            self.addCleanup(active_patch.stop)

        manager = manager_module.WeightAsyncStreamManager(
            offload_granularity="block",
            load_stream=load_stream,
            compute_stream=compute_stream,
        )
        buffers = [_FakeBuffer("slot-0", trace), _FakeBuffer("slot-1", trace)]
        manager.init_cuda_buffer(blocks_cuda_buffer=buffers)
        return manager, buffers, load_stream, compute_stream, trace

    def test_injected_streams_are_reused(self):
        manager, _, load_stream, compute_stream, trace = self._make_npu_manager()

        self.assertIs(manager.cuda_load_stream, load_stream)
        self.assertIs(manager.compute_stream, compute_stream)
        created_streams = [entry for entry in trace if entry[0] == "create_stream"]
        self.assertEqual(created_streams, [("create_stream", "created-stream-0", 0)])

    def test_event_slots_pipeline_without_steady_state_host_sync(self):
        manager, buffers, _, _, trace = self._make_npu_manager()
        blocks = [_FakeBlock(0), _FakeBlock(1), _FakeBlock(2)]
        trace.clear()

        first_buffer = manager.prefetch_to_slot(0, 0, blocks)
        self.assertIs(first_buffer, buffers[0])
        self.assertIs(manager.wait_ready(0), buffers[0])
        trace.append(("compute", 0))
        manager.record_free(0)

        second_buffer = manager.prefetch_to_slot(0, 2, blocks, adapter_block_idx=7)
        self.assertIs(second_buffer, buffers[0])
        self.assertIs(manager.wait_ready(0), buffers[0])
        trace.append(("compute", 2))
        manager.record_free(0)

        self.assertNotIn(("synchronize", "shared-load"), trace)
        self.assertNotIn(("synchronize", "shared-compute"), trace)

        first_ready_record = trace.index(("record", "event-0", "shared-load"))
        first_compute_wait = trace.index(("wait_event", "shared-compute", "event-0"))
        first_compute = trace.index(("compute", 0))
        first_free_record = trace.index(("record", "event-2", "shared-compute"))
        slot_reuse_wait = trace.index(("wait_event", "shared-load", "event-2"))
        second_load = trace.index(("load", "slot-0", 2, 2, 7))
        self.assertLess(first_ready_record, first_compute_wait)
        self.assertLess(first_compute_wait, first_compute)
        self.assertLess(first_compute, first_free_record)
        self.assertLess(first_free_record, slot_reuse_wait)
        self.assertLess(slot_reuse_wait, second_load)

        manager.flush()
        self.assertEqual(trace[-2:], [("synchronize", "shared-load"), ("synchronize", "shared-compute")])

    def test_slot_cannot_be_reused_until_compute_records_free(self):
        manager, _, _, _, _ = self._make_npu_manager()
        blocks = [_FakeBlock(0), _FakeBlock(1)]

        manager.prefetch_to_slot(0, 0, blocks)
        with self.assertRaisesRegex(RuntimeError, "record_free"):
            manager.prefetch_to_slot(0, 1, blocks)
        with self.assertRaisesRegex(RuntimeError, "wait_ready"):
            manager.record_free(0)

    def test_block_slab_path_uses_one_raw_copy_then_existing_loader(self):
        manager, _, _, _, trace = self._make_npu_manager()
        slab = SimpleNamespace(
            raw=object(),
            layout=SimpleNamespace(nbytes=128),
        )

        def carve(raw, layout):
            self.assertEqual(raw.numel(), 128)
            return {"block_idx": 1}

        def copy_raw(destination, source, *, nbytes, non_blocking):
            trace.append(("raw_copy", nbytes, non_blocking))
            self.assertIs(source, slab.raw)
            return destination

        with (
            patch.object(manager_module, "carve_block_slab", side_effect=carve),
            patch.object(manager_module, "copy_block_slab_", side_effect=copy_raw),
        ):
            manager.init_block_slabs({1: slab})
            trace.clear()
            manager.prefetch_to_slot(0, 1, [_FakeBlock(0), _FakeBlock(1)])

        raw_copy = trace.index(("raw_copy", 128, True))
        typed_scatter = trace.index(("load", "slot-0", 1, 1, None))
        ready_record = trace.index(("record", "event-0", "shared-load"))
        self.assertLess(raw_copy, typed_scatter)
        self.assertLess(typed_scatter, ready_record)

    def test_legacy_cuda_swap_still_synchronizes_and_swaps(self):
        trace = []
        device_module = _FakeDeviceModule(trace)
        load_stream = _FakeStream("cuda-load", trace)
        compute_stream = _FakeStream("cuda-compute", trace)
        with (
            patch.object(manager_module, "AI_DEVICE", "cuda"),
            patch.object(manager_module, "torch_device_module", device_module),
        ):
            manager = manager_module.WeightAsyncStreamManager(
                offload_granularity="block",
                load_stream=load_stream,
                compute_stream=compute_stream,
            )
            buffers = [object(), object()]
            manager.init_cuda_buffer(blocks_cuda_buffer=buffers)
            manager.swap_blocks()

        self.assertEqual(trace[-2:], [("synchronize", "cuda-load"), ("synchronize", "cuda-compute")])
        self.assertEqual(manager.cuda_buffers, [buffers[1], buffers[0]])
        self.assertEqual(device_module.event_count, 0)

    def test_legacy_xpu_swap_keeps_device_wide_sync(self):
        trace = []
        device_module = _FakeDeviceModule(trace)
        with (
            patch.object(manager_module, "AI_DEVICE", "xpu"),
            patch.object(manager_module, "torch_device_module", device_module),
        ):
            manager = manager_module.WeightAsyncStreamManager(offload_granularity="block")
            buffers = [object(), object()]
            manager.init_cuda_buffer(blocks_cuda_buffer=buffers)
            trace.clear()
            manager.swap_blocks()

        self.assertEqual(trace, [("device_synchronize",)])
        self.assertEqual(manager.cuda_buffers, [buffers[1], buffers[0]])


if __name__ == "__main__":
    unittest.main()
