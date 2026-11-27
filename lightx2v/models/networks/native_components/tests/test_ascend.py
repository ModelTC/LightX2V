"""Precision policy regression tests, independent of accelerator availability."""
import importlib.util
import os
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import torch


def load_adapter(path):
    spec = importlib.util.spec_from_file_location("native_ascend", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class PrecisionPolicy(unittest.TestCase):
    def setUp(self):
        root = Path(__file__).resolve().parents[1]
        self.adapter = load_adapter(root / "ascend.py")

    def test_other_platforms_never_access_npu(self):
        for platform in ("cuda", "metax_cuda", "cambricon_mlu"):
            with patch.dict(os.environ, PLATFORM=platform):
                with patch.dict(sys.modules, torch_npu=None):
                    x = torch.ones(3, dtype=torch.bfloat16)
                    wrapped = self.adapter.preserve_native_precision(lambda value: value)
                    self.assertIs(wrapped(x), x)

    def test_flags_restored_after_nested_failure_and_dtypes_preserved(self):
        backend = types.SimpleNamespace(
            conv=types.SimpleNamespace(allow_hf32=True),
            matmul=types.SimpleNamespace(allow_hf32=False),
        )
        with patch.dict(os.environ, PLATFORM="ascend_npu"), patch.dict(sys.modules, torch_npu=types.ModuleType("torch_npu")), patch.object(torch, "npu", backend, create=True):
            x, y = torch.ones(3, dtype=torch.bfloat16), torch.ones(3, dtype=torch.float32)

            @self.adapter.preserve_native_precision
            def request(a, b):
                self.assertFalse(backend.conv.allow_hf32)
                self.assertFalse(backend.matmul.allow_hf32)
                self.assertEqual(a.dtype, torch.bfloat16)
                self.assertEqual(b.dtype, torch.float32)
                with self.adapter.native_precision():
                    raise ValueError("request failure")

            with self.assertRaisesRegex(ValueError, "request failure"):
                request(x, y)
            self.assertTrue(backend.conv.allow_hf32)
            self.assertFalse(backend.matmul.allow_hf32)


if __name__ == "__main__":
    unittest.main()
