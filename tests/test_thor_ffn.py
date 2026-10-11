"""Regression tests for Thor FFN adapter semantics and tensor lifetime."""

import gc
import importlib.util
import os
import unittest
import weakref
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn.functional as F

# Load the adapter without importing unrelated optional runtime backends.
_SOURCE = Path(__file__).resolve().parents[1] / "lightx2v/common/ops/mm/thor_ffn/backend.py"
_SPEC = importlib.util.spec_from_file_location("thor_ffn_backend", _SOURCE)
backend = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(backend)


class TestThorFFN(unittest.TestCase):
    def setUp(self):
        if hasattr(backend._plan_cache, "entry"):
            del backend._plan_cache.entry

    def test_down_uses_current_weight_bias_and_preserves_lora(self):
        from lightx2v.common.ops.mm.mm_weight import MMWeight

        mm = object.__new__(MMWeight)
        mm.weight = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        mm.bias = torch.tensor([0.1, 0.2])
        mm.weight_diff = torch.ones_like(mm.weight)
        mm.bias_diff = torch.ones_like(mm.bias)
        mm.has_lora_branch = True
        mm.apply_lora = lambda x: torch.full((x.shape[0], 2), 0.5)
        x = torch.tensor([[1.0, 2.0]])
        native = SimpleNamespace(run=lambda x, w, b: x @ w.t() + b)
        with patch.object(backend, "down_extension", return_value=native):
            expected = x @ (mm.weight + mm.weight_diff) + mm.bias + mm.bias_diff + 0.5
            torch.testing.assert_close(backend.down_bias(mm, x), expected)
            mm.weight_diff.mul_(2)
            expected = x @ (mm.weight + mm.weight_diff) + mm.bias + mm.bias_diff + 0.5
            torch.testing.assert_close(backend.down_bias(mm, x), expected)

    def test_down_preserves_other_backend_dispatch(self):
        mm = SimpleNamespace(apply=lambda x: x + 2)
        with patch.object(backend, "down_extension", side_effect=AssertionError("unexpected native call")):
            torch.testing.assert_close(backend.down_bias(mm, torch.ones(1)), torch.full((1,), 3.0))

    def test_lora_is_added_before_gelu_without_loading_extension(self):
        x = torch.tensor([[1.0, -2.0]])
        weight = torch.tensor([[0.5, -1.0], [0.25, 0.5]])
        down = torch.tensor([[1.0], [0.5]])
        up = torch.tensor([[0.5, -2.0]])
        bias = torch.tensor([0.1, -0.2])
        x = x + 0.25
        mm = SimpleNamespace(has_lora_branch=True, apply=lambda v: v @ weight + bias + (v @ down) @ up)
        expected = F.gelu(x @ weight + bias + (x @ down) @ up, approximate="tanh")
        with patch.object(backend, "extension", side_effect=AssertionError("LoRA must use fallback")):
            torch.testing.assert_close(backend.up_gelu(mm, x), expected, rtol=0, atol=0)

    def test_call_does_not_retain_weights_or_activation_buffers(self):
        refs = []

        class Plan:
            def __init__(self, x, w, b, y):
                refs.extend(weakref.ref(t) for t in (x, w, b, y))

            def run_with(self, x, w, b, y):
                refs.extend(weakref.ref(t) for t in (x, w, b, y))
                y.copy_(F.gelu(x @ w.t() + b, approximate="tanh"))

        mm = SimpleNamespace(has_lora_branch=False)
        mm.weight = torch.randn(3, 4)
        mm.bias = torch.randn(4)
        mm._get_actual_weight = lambda: mm.weight
        mm._get_actual_bias = lambda: mm.bias
        with patch.object(backend, "extension", return_value=SimpleNamespace(prepare=Plan)), patch.object(torch.cuda, "current_stream", return_value=SimpleNamespace(cuda_stream=0)):
            for rows in (2, 5, 2):
                x = torch.randn(rows, 3)
                y = backend.up_gelu(mm, x)
                torch.testing.assert_close(y, F.gelu(x @ mm.weight + mm.bias, approximate="tanh"))
                del x, y
        # Simulate releasing/offloading the owning module's weights.
        mm.weight = mm.bias = None
        gc.collect()
        self.assertTrue(all(ref() is None for ref in refs))

    def test_reuses_plan_across_layers_but_not_shapes_or_streams(self):
        prepared = []

        class Plan:
            def __init__(self, *args):
                prepared.append(self)

            def run_with(self, x, w, b, y):
                y.copy_(F.gelu(x @ w.t() + b, approximate="tanh"))

        def layer(scale):
            return SimpleNamespace(
                has_lora_branch=False,
                _get_actual_weight=lambda: torch.full((3, 4), scale),
                _get_actual_bias=lambda: torch.zeros(4),
            )

        stream = SimpleNamespace(cuda_stream=0)
        with patch.object(backend, "extension", return_value=SimpleNamespace(prepare=Plan)), patch.object(torch.cuda, "current_stream", return_value=stream):
            first = backend.up_gelu(layer(1.0), torch.ones(2, 3))
            saved = first.clone()
            second = backend.up_gelu(layer(2.0), torch.ones(2, 3))
            self.assertEqual(len(prepared), 1)
            self.assertFalse(torch.equal(first, second))
            torch.testing.assert_close(first, saved, rtol=0, atol=0)
            backend.up_gelu(layer(1.0), torch.ones(5, 3))
            self.assertEqual(len(prepared), 2)
            stream.cuda_stream = 1
            backend.up_gelu(layer(1.0), torch.ones(5, 3))
            self.assertEqual(len(prepared), 3)

    @unittest.skipUnless(os.environ.get("THOR_FFN_RUN_CUDA_TESTS") == "1", "opt-in Thor CUDA/JIT test")
    def test_native_cached_plan_releases_tensor_storage(self):
        backend.extension()
        torch.cuda.synchronize()
        gc.collect()
        before = torch.cuda.memory_allocated()
        with torch.inference_mode():
            mm = SimpleNamespace(has_lora_branch=False)
            mm.weight = torch.zeros((13824, 5120), device="cuda", dtype=torch.bfloat16).t()
            mm.bias = torch.zeros(13824, device="cuda", dtype=torch.bfloat16)
            mm._get_actual_weight = lambda: mm.weight
            mm._get_actual_bias = lambda: mm.bias
            for rows in (17, 33, 17):
                x = torch.ones((rows, 5120), device="cuda", dtype=torch.bfloat16)
                y = backend.up_gelu(mm, x)
                self.assertEqual(torch.count_nonzero(y).item(), 0)
                mm.bias.fill_(1)
                z = backend.up_gelu(mm, x)
                torch.testing.assert_close(z, F.gelu(torch.ones_like(z), approximate="tanh"))
                self.assertEqual(torch.count_nonzero(y).item(), 0)
                mm.bias.zero_()
                del x, y, z
            mm.weight = mm.bias = None
        gc.collect()
        torch.cuda.synchronize()
        # Includes C++ at::Tensor owners, which Python weakrefs cannot detect.
        self.assertLess(torch.cuda.memory_allocated() - before, 1024 * 1024)


if __name__ == "__main__":
    unittest.main()
