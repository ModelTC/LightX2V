import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import torch
from safetensors import safe_open
from safetensors.torch import save_file

os.environ.setdefault("SKIP_PLATFORM_CHECK", "1")


def ensure_lightx2v_pipeline_stub():
    if "lightx2v.pipeline" not in sys.modules:
        pipeline_stub = types.ModuleType("lightx2v.pipeline")
        pipeline_stub.LightX2VPipeline = object
        sys.modules["lightx2v.pipeline"] = pipeline_stub


class MmapPinnedWeightsTest(unittest.TestCase):
    def setUp(self):
        ensure_lightx2v_pipeline_stub()
        from lightx2v.common.ops import utils

        utils.reset_pinned_weight_stats()

    def _write_safetensors(self, tensors):
        tmpdir = tempfile.TemporaryDirectory()
        path = Path(tmpdir.name) / "weights.safetensors"
        save_file(tensors, str(path))
        self.addCleanup(tmpdir.cleanup)
        return path

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for cudaHostRegister")
    def test_safetensors_tensor_registers_without_copy(self):
        from lightx2v.common.ops.utils import create_pin_tensor, get_pinned_weight_stats

        path = self._write_safetensors({"w": torch.arange(16, dtype=torch.float32).reshape(4, 4)})
        with safe_open(str(path), framework="pt", device="cpu") as f:
            tensor = f.get_tensor("w")
            original_ptr = tensor.data_ptr()
            pinned = create_pin_tensor(tensor)

        self.assertEqual(pinned.data_ptr(), original_ptr)
        self.assertTrue(pinned.is_pinned())
        stats = get_pinned_weight_stats()
        self.assertEqual(stats["fallback_count"], 0)
        self.assertGreaterEqual(stats["registered_bytes"], pinned.numel() * pinned.element_size())

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for cudaHostRegister")
    def test_registered_tensor_survives_safe_open_close_for_h2d(self):
        from lightx2v.common.ops.utils import create_pin_tensor, is_readonly_pinned_source

        source = torch.arange(16, dtype=torch.float32).reshape(4, 4)
        path = self._write_safetensors({"w": source})
        with safe_open(str(path), framework="pt", device="cpu") as f:
            pinned = create_pin_tensor(f.get_tensor("w"))

        self.assertTrue(is_readonly_pinned_source(pinned))
        self.assertTrue(torch.equal(pinned, source))
        cuda_copy = pinned.to("cuda", non_blocking=True)
        torch.cuda.synchronize()
        self.assertTrue(torch.equal(cuda_copy.cpu(), source))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for cudaHostRegister")
    def test_transposed_view_keeps_registered_base(self):
        from lightx2v.common.ops.utils import create_pin_tensor

        path = self._write_safetensors({"w": torch.arange(12, dtype=torch.float32).reshape(3, 4)})
        with safe_open(str(path), framework="pt", device="cpu") as f:
            tensor = f.get_tensor("w")
            original_ptr = tensor.data_ptr()
            pinned = create_pin_tensor(tensor, transpose=True)

        self.assertEqual(tuple(pinned.shape), (4, 3))
        self.assertEqual(pinned.data_ptr(), original_ptr)
        self.assertTrue(pinned.is_pinned())
        self.assertTrue(hasattr(pinned, "_lightx2v_cuda_host_registered_base"))

    def test_dtype_conversion_falls_back_to_copy(self):
        from lightx2v.common.ops.utils import create_pin_tensor, get_pinned_weight_stats

        source = torch.arange(8, dtype=torch.float32)
        pinned = create_pin_tensor(source, dtype=torch.float16)

        self.assertEqual(pinned.dtype, torch.float16)
        self.assertNotEqual(pinned.data_ptr(), source.data_ptr())
        stats = get_pinned_weight_stats()
        self.assertEqual(stats["registered_count"], 0)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for cudaHostRegister")
    def test_safetensors_dtype_conversion_does_not_register_converted_copy(self):
        from lightx2v.common.ops.utils import create_pin_tensor, get_pinned_weight_stats

        path = self._write_safetensors({"w": torch.arange(8, dtype=torch.float32)})
        with safe_open(str(path), framework="pt", device="cpu") as f:
            tensor = f.get_tensor("w")
            original_ptr = tensor.data_ptr()
            pinned = create_pin_tensor(tensor, dtype=torch.float16)

        self.assertEqual(pinned.dtype, torch.float16)
        self.assertNotEqual(pinned.data_ptr(), original_ptr)
        stats = get_pinned_weight_stats()
        self.assertEqual(stats["registered_count"], 0)
        self.assertEqual(stats["fallback_count"], 0)

    def test_registration_failure_falls_back(self):
        from lightx2v.common.ops import utils

        source = torch.arange(8, dtype=torch.float32)
        with mock.patch.object(utils, "cuda_register_host_tensor", side_effect=RuntimeError("forced failure")):
            pinned = utils.create_pin_tensor(source)

        self.assertNotEqual(pinned.data_ptr(), source.data_ptr())
        self.assertEqual(utils.get_pinned_weight_stats()["fallback_count"], 1)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for cudaHostRegister")
    def test_to_cpu_does_not_write_back_readonly_pinned_source(self):
        from lightx2v.common.ops.utils import create_pin_tensor, move_tensor_back_to_cpu

        class Holder:
            pass

        source = torch.arange(8, dtype=torch.float32)
        path = self._write_safetensors({"w": source})
        with safe_open(str(path), framework="pt", device="cpu") as f:
            holder = Holder()
            holder.pin_weight = create_pin_tensor(f.get_tensor("w"))
            holder.weight = torch.full((8,), 99.0, device="cuda")

        original_ptr = holder.pin_weight.data_ptr()
        move_tensor_back_to_cpu(holder, "weight")

        self.assertEqual(holder.weight.data_ptr(), original_ptr)
        self.assertTrue(torch.equal(holder.weight, source))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for cudaHostRegister")
    def test_mm_weight_to_cpu_does_not_write_back_readonly_source(self):
        import lightx2v.common.ops.mm.mm_weight as mm_weight

        source = torch.arange(12, dtype=torch.float32).reshape(3, 4)
        path = self._write_safetensors({"blocks.0.ffn.0.weight": source})
        weight = mm_weight.MMWeight(
            "blocks.0.ffn.0.weight",
            None,
            create_cpu_buffer=True,
            lazy_load=True,
            lazy_load_file=str(path),
        )
        weight.load({})
        original = weight.pin_weight.clone()
        original_ptr = weight.pin_weight.data_ptr()
        weight.weight = torch.full(tuple(weight.pin_weight.shape), 99.0, device="cuda")

        weight.to_cpu()

        self.assertEqual(weight.weight.data_ptr(), original_ptr)
        self.assertTrue(torch.equal(weight.weight, original))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for pinned copy-back")
    def test_to_cpu_preserves_copy_back_for_writable_pin_buffer(self):
        from lightx2v.common.ops.utils import create_writable_pin_tensor, move_tensor_back_to_cpu

        class Holder:
            pass

        holder = Holder()
        holder.pin_weight = create_writable_pin_tensor(torch.zeros(4, dtype=torch.float32))
        holder.weight = torch.arange(4, dtype=torch.float32, device="cuda")
        move_tensor_back_to_cpu(holder, "weight")

        self.assertEqual(holder.weight.data_ptr(), holder.pin_weight.data_ptr())
        self.assertTrue(torch.equal(holder.weight, torch.arange(4, dtype=torch.float32)))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for cudaHostRegister")
    def test_mm_weight_lazy_reload_uses_registered_mmap_tensor(self):
        import lightx2v.common.ops.mm.mm_weight as mm_weight

        path = self._write_safetensors(
            {
                "blocks.0.ffn.0.weight": torch.arange(12, dtype=torch.float32).reshape(3, 4),
                "blocks.0.ffn.0.bias": torch.arange(4, dtype=torch.float32),
            }
        )
        weight = mm_weight.MMWeight(
            "blocks.0.ffn.0.weight",
            "blocks.0.ffn.0.bias",
            create_cpu_buffer=True,
            lazy_load=True,
            lazy_load_file=str(path),
        )
        weight.load({})
        initial_ptr = weight.pin_weight.data_ptr()

        weight.load_state_dict_from_disk(0)

        self.assertEqual(tuple(weight.pin_weight.shape), (4, 3))
        self.assertTrue(weight.pin_weight.is_pinned())
        self.assertNotEqual(weight.pin_weight.data_ptr(), initial_ptr)
        self.assertTrue(hasattr(weight.pin_weight, "_lightx2v_cuda_host_registered_base"))
        self.assertTrue(weight.pin_bias.is_pinned())

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for cudaHostRegister")
    def test_quant_lazy_reload_preserves_scale_and_bias_dtypes(self):
        import lightx2v.common.ops.mm.mm_weight as mm_weight

        path = self._write_safetensors(
            {
                "blocks.0.attn.q.weight": torch.arange(12, dtype=torch.float32).reshape(3, 4),
                "blocks.0.attn.q.weight_scale": torch.arange(4, dtype=torch.float16),
                "blocks.0.attn.q.bias": torch.arange(4, dtype=torch.float32),
            }
        )
        weight = mm_weight.MMWeightWfp8channelAfp8channeldynamicVllm(
            "blocks.0.attn.q.weight",
            "blocks.0.attn.q.bias",
            create_cpu_buffer=True,
            lazy_load=True,
            lazy_load_file=str(path),
        )
        weight.bias_force_fp32 = False

        weight.load_state_dict_from_disk(0)

        self.assertEqual(weight.pin_weight_scale.dtype, torch.float32)
        self.assertEqual(weight.pin_bias.dtype, weight.infer_dtype)


if __name__ == "__main__":
    unittest.main()
