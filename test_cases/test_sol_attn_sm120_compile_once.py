import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch

os.environ.setdefault("SKIP_PLATFORM_CHECK", "1")

from lightx2v.common.ops.attn.sol_attn import (  # noqa: E402
    SolAttnWeight,
    _CompiledSolAttnWithKeywordStream,
    _torch_stream_handle,
)


class SolAttnSm120CompileOnceTest(unittest.TestCase):
    def test_unified_torch_stream_uses_native_handle(self):
        self.assertEqual(_torch_stream_handle(SimpleNamespace(native_handle=1234)), 1234)
        self.assertEqual(_torch_stream_handle(SimpleNamespace(native_handle=lambda: 5678)), 5678)
        self.assertEqual(_torch_stream_handle(SimpleNamespace(cuda_stream=9012, native_handle=1234)), 9012)

    def test_tvm_ffi_keyword_stream_is_forwarded_positionally(self):
        compiled = mock.Mock(return_value="output")
        wrapped = _CompiledSolAttnWithKeywordStream(compiled)

        self.assertEqual(wrapped("q", "k", stream="stream"), "output")
        compiled.assert_called_once_with("q", "k", "stream")

    def test_compile_once_mode_routes_to_fixed_interface(self):
        backend = SolAttnWeight()
        backend.set_config(
            {
                "compile_mode": "sm120_compile_once",
                "tau": 1.5,
                "thresh_type": "diag",
                "strict": True,
            }
        )
        q = torch.randn(65, 1, 128, dtype=torch.bfloat16)
        k = torch.randn_like(q)
        v = torch.randn_like(q)

        with (
            mock.patch.object(SolAttnWeight, "_ineligibility_reason", return_value=None),
            mock.patch(
                "lightx2v.common.ops.attn.sol_attn._run_sol_attn_sm120_compile_once",
                return_value=v.unsqueeze(0),
            ) as fixed_kernel,
            mock.patch("lightx2v.common.ops.attn.sol_attn._run_sol_attn") as default_kernel,
            mock.patch("torch.cuda.get_device_capability", return_value=(12, 0)),
        ):
            actual = backend.apply(q, k, v)

        fixed_kernel.assert_called_once()
        default_kernel.assert_not_called()
        self.assertEqual(actual.shape, (65, 128))
        torch.testing.assert_close(actual, v.reshape(65, 128))

    def test_compile_once_rejects_incompatible_settings(self):
        backend = SolAttnWeight()
        with self.assertRaisesRegex(ValueError, "compile_mode must be one of"):
            backend.set_config({"compile_mode": "unknown"})
        with self.assertRaisesRegex(ValueError, "requires thresh_type='diag'"):
            backend.set_config(
                {
                    "compile_mode": "sm120_compile_once",
                    "thresh_type": "exact",
                }
            )
        with self.assertRaisesRegex(ValueError, "requires kv_splits='auto' or 1"):
            backend.set_config(
                {
                    "compile_mode": "sm120_compile_once",
                    "kv_splits": 2,
                }
            )

    def test_compile_and_persistent_keys_ignore_token_count(self):
        from lightx2v.common.ops.attn.sol_attn_sm120_compile_once import (
            _compile_key,
            _persistent_cache_fingerprint,
        )

        q_short = torch.empty(1, 64, 1, 128, dtype=torch.bfloat16)
        q_long = torch.empty(1, 129, 1, 128, dtype=torch.bfloat16)

        self.assertEqual(
            _compile_key(q_short, (12, 0), 1),
            _compile_key(q_long, (12, 0), 1),
        )
        interface = SimpleNamespace(__file__=__file__)
        with (
            mock.patch(
                "lightx2v.common.ops.attn.sol_attn_sm120_compile_once._sm120_source_digest",
                return_value="kernel-source",
            ),
            mock.patch(
                "lightx2v.common.ops.attn.sol_attn_sm120_compile_once._distribution_version",
                return_value="test-version",
            ),
        ):
            self.assertEqual(
                _persistent_cache_fingerprint(q_short, (12, 0), 1, interface),
                _persistent_cache_fingerprint(q_long, (12, 0), 1, interface),
            )

    def test_persistent_cache_hit_skips_cute_compile(self):
        from lightx2v.common.ops.attn.sol_attn_sm120_compile_once import _load_or_compile_persistent

        compiled = object()
        interface = SimpleNamespace(
            _compiled={},
            _to_cute_tensors=mock.Mock(return_value=["cute-args"]),
            _compile_sm120=mock.Mock(),
        )
        with tempfile.TemporaryDirectory() as directory:
            artifact_path = Path(directory) / "kernel.o"
            artifact_path.touch()
            with mock.patch(
                "lightx2v.common.ops.attn.sol_attn_sm120_compile_once._load_persistent_compiled",
                return_value=compiled,
            ):
                actual = _load_or_compile_persistent(
                    interface,
                    "key",
                    ["tensors"],
                    0.5,
                    0,
                    1,
                    "stream",
                    artifact_path,
                )

        self.assertEqual(actual, (compiled, ["cute-args"]))
        self.assertIs(interface._compiled["key"], compiled)
        interface._compile_sm120.assert_not_called()

    def test_persistent_cache_miss_exports_fresh_compile(self):
        from lightx2v.common.ops.attn.sol_attn_sm120_compile_once import _load_or_compile_persistent

        compiled = object()
        interface = SimpleNamespace(
            _compiled={},
            _to_cute_tensors=mock.Mock(),
            _compile_sm120=mock.Mock(return_value=(compiled, ["fresh-args"])),
        )
        with tempfile.TemporaryDirectory() as directory:
            artifact_path = Path(directory) / "kernel.o"
            with mock.patch("lightx2v.common.ops.attn.sol_attn_sm120_compile_once._export_persistent_compiled") as export:
                actual = _load_or_compile_persistent(
                    interface,
                    "key",
                    ["tensors"],
                    0.5,
                    0,
                    1,
                    "stream",
                    artifact_path,
                )

        self.assertEqual(actual, (compiled, ["fresh-args"]))
        interface._compile_sm120.assert_called_once_with("key", ["tensors"], 0.5, 0, 1, "stream")
        export.assert_called_once_with(compiled, artifact_path)


if __name__ == "__main__":
    unittest.main()
