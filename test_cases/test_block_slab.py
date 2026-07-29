import importlib.util
import sys
import unittest
import warnings
from pathlib import Path
from unittest import mock

import torch
from torch.utils._python_dispatch import TorchDispatchMode


def _load_block_slab_module():
    module_path = Path(__file__).resolve().parents[1] / "lightx2v" / "common" / "offload" / "block_slab.py"
    spec = importlib.util.spec_from_file_location("block_slab_under_test", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


block_slab = _load_block_slab_module()


class _CopyCounter(TorchDispatchMode):
    def __init__(self):
        super().__init__()
        self.copy_count = 0

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if func is torch.ops.aten.copy_.default:
            self.copy_count += 1
        return func(*args, **(kwargs or {}))


class BlockSlabTest(unittest.TestCase):
    def make_tensors(self):
        return {
            "matrix": torch.arange(12, dtype=torch.float32).reshape(3, 4).t(),
            "bias": torch.tensor([3, -2, 7], dtype=torch.int16),
            "scalar": torch.tensor(9, dtype=torch.int64),
            "empty": torch.empty((0, 2), dtype=torch.float16),
        }

    def test_pack_preserves_values_and_aligns_entries(self):
        tensors = self.make_tensors()
        self.assertFalse(tensors["matrix"].is_contiguous())

        layout = block_slab.build_block_slab_layout(tensors, alignment=64)
        packed = block_slab.pack_cpu_block_slab(tensors, layout=layout, pin_memory=False)

        self.assertEqual(packed.raw.dtype, torch.uint8)
        self.assertEqual(packed.raw.dim(), 1)
        self.assertTrue(packed.raw.is_contiguous())
        self.assertFalse(packed.is_pinned)
        self.assertEqual(packed.raw.numel(), layout.nbytes)

        for name, entry in layout.entries.items():
            self.assertEqual(entry.offset % 64, 0)
            self.assertEqual(entry.offset % entry.dtype.itemsize, 0)
            self.assertEqual(packed.views[name].untyped_storage().data_ptr(), packed.raw.untyped_storage().data_ptr())
            self.assertTrue(packed.views[name].is_contiguous())
            torch.testing.assert_close(packed.views[name], tensors[name])

    def test_raw_slot_carving_and_copy_use_one_copy_operation(self):
        tensors = self.make_tensors()
        packed = block_slab.pack_cpu_block_slab(tensors, pin_memory=False)
        slot = block_slab.allocate_block_slab_slot(packed.layout, "cpu", slot_nbytes=packed.layout.nbytes + 128)
        slot.raw.zero_()

        counter = _CopyCounter()
        with counter:
            result = block_slab.copy_block_slab_(
                slot.raw,
                packed.raw,
                nbytes=packed.layout.nbytes,
                non_blocking=True,
            )

        self.assertIs(result, slot.raw)
        self.assertEqual(counter.copy_count, 1)
        self.assertTrue(torch.count_nonzero(slot.raw[packed.layout.nbytes :]) == 0)
        for name in tensors:
            torch.testing.assert_close(slot.views[name], tensors[name])

    def test_pin_failure_warns_and_falls_back(self):
        tensors = {"weight": torch.arange(8, dtype=torch.float32)}
        with mock.patch.object(block_slab, "_allocate_pinned_uint8", side_effect=RuntimeError("pin unavailable")):
            with self.assertWarnsRegex(RuntimeWarning, "falling back to regular CPU memory"):
                packed = block_slab.pack_cpu_block_slab(tensors, pin_memory=True, strict_pin=False)

        self.assertFalse(packed.is_pinned)
        torch.testing.assert_close(packed.views["weight"], tensors["weight"])

    def test_strict_pin_failure_raises(self):
        tensors = {"weight": torch.arange(8, dtype=torch.float32)}
        with mock.patch.object(block_slab, "_allocate_pinned_uint8", side_effect=RuntimeError("pin unavailable")):
            with self.assertRaisesRegex(RuntimeError, "failed to allocate"):
                block_slab.pack_cpu_block_slab(tensors, pin_memory=True, strict_pin=True)

    def test_supplied_layout_rejects_mismatched_tensors(self):
        layout = block_slab.build_block_slab_layout({"weight": torch.ones(2, dtype=torch.float32)})

        with self.assertRaisesRegex(ValueError, "names do not match"):
            block_slab.pack_cpu_block_slab({"other": torch.ones(2)}, layout=layout, pin_memory=False)
        with self.assertRaisesRegex(ValueError, "dtype"):
            block_slab.pack_cpu_block_slab({"weight": torch.ones(2, dtype=torch.float16)}, layout=layout, pin_memory=False)
        with self.assertRaisesRegex(ValueError, "shape"):
            block_slab.pack_cpu_block_slab({"weight": torch.ones(3, dtype=torch.float32)}, layout=layout, pin_memory=False)

    def test_invalid_raw_buffers_and_slot_sizes_raise(self):
        tensors = {"weight": torch.ones(4, dtype=torch.float32)}
        layout = block_slab.build_block_slab_layout(tensors)

        with self.assertRaisesRegex(ValueError, "too small"):
            block_slab.carve_block_slab(torch.empty(layout.nbytes - 1, dtype=torch.uint8), layout)
        with self.assertRaisesRegex(ValueError, "uint8"):
            block_slab.carve_block_slab(torch.empty(layout.nbytes, dtype=torch.float32), layout)
        with self.assertRaisesRegex(ValueError, "slot_nbytes is too small"):
            block_slab.allocate_block_slab_slot(layout, "cpu", slot_nbytes=layout.nbytes - 1)

    def test_empty_layout_uses_a_valid_raw_allocation(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            packed = block_slab.pack_cpu_block_slab({}, pin_memory=False)

        self.assertEqual(packed.layout.nbytes, 1)
        self.assertEqual(packed.raw.numel(), 1)
        self.assertEqual(packed.views, {})

    def test_alignment_must_be_positive(self):
        with self.assertRaisesRegex(ValueError, "positive integer"):
            block_slab.build_block_slab_layout({"weight": torch.ones(1)}, alignment=0)


if __name__ == "__main__":
    unittest.main()
