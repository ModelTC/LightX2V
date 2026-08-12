import os
import unittest
from contextlib import ExitStack
from types import SimpleNamespace
from unittest import mock

import torch
import torch.nn.functional as F

os.environ.setdefault("SKIP_PLATFORM_CHECK", "1")

from lightx2v.common.ops.attn.sol_attn import SolAttnWeight, _morton3d_indices  # noqa: E402
from lightx2v.models.networks.minimax_h3.infer.transformer_infer import MiniMaxH3TransformerInfer  # noqa: E402
from lightx2v.models.networks.wan.infer.transformer_infer import WanTransformerInfer  # noqa: E402
from lightx2v.utils.registry_factory import ATTN_WEIGHT_REGISTER  # noqa: E402


class SolAttnBackendTest(unittest.TestCase):
    def test_backend_is_registered(self):
        self.assertIs(ATTN_WEIGHT_REGISTER["sol_attn"], SolAttnWeight)

    def test_cpu_call_falls_back_to_sdpa(self):
        torch.manual_seed(0)
        q = torch.randn(9, 2, 8)
        k = torch.randn(9, 2, 8)
        v = torch.randn(9, 2, 8)
        actual = SolAttnWeight().apply(q, k, v)
        expected = (
            F.scaled_dot_product_attention(
                q.unsqueeze(0).transpose(1, 2),
                k.unsqueeze(0).transpose(1, 2),
                v.unsqueeze(0).transpose(1, 2),
            )
            .transpose(1, 2)
            .reshape(9, -1)
        )
        torch.testing.assert_close(actual, expected)

    def test_strict_mode_rejects_ineligible_call(self):
        backend = SolAttnWeight()
        backend.set_config({"strict": True})
        with self.assertRaisesRegex(RuntimeError, "same CUDA device"):
            backend.apply(
                torch.randn(4, 1, 128, dtype=torch.bfloat16),
                torch.randn(4, 1, 128, dtype=torch.bfloat16),
                torch.randn(4, 1, 128, dtype=torch.bfloat16),
            )

    def test_public_kernel_arguments_and_output_layout(self):
        backend = SolAttnWeight()
        backend.set_config(
            {
                "tau": 1.25,
                "thresh_type": "exact",
                "kv_splits": 1,
                "strict": True,
                "dense_steps": 8,
                "dense_layers": [0],
            }
        )
        q = torch.randn(7, 2, 128, dtype=torch.bfloat16)
        k = torch.randn_like(q)
        v = torch.randn_like(q)

        def fake_kernel(q_bthd, k_bthd, v_bthd, **kwargs):
            self.assertEqual(q_bthd.shape, (1, 7, 2, 128))
            self.assertTrue(q_bthd.is_contiguous())
            self.assertEqual(kwargs["tau"], 1.25)
            self.assertEqual(kwargs["thresh_type"], "exact")
            self.assertEqual(kwargs["kv_splits"], 1)
            return v_bthd + 1

        with (
            mock.patch.object(SolAttnWeight, "_ineligibility_reason", return_value=None),
            mock.patch("lightx2v.common.ops.attn.sol_attn._load_sol_attn", return_value=fake_kernel),
            mock.patch("torch.cuda.get_device_capability", return_value=(9, 0)),
        ):
            actual = backend.apply(q, k, v, scheduler=SimpleNamespace(step_index=8), block_idx=1)
        self.assertEqual(actual.shape, (7, 256))
        torch.testing.assert_close(actual, (v + 1).reshape(7, 256))

    def test_dense_warmup_step_bypasses_sol_kernel(self):
        backend = SolAttnWeight()
        backend.set_config({"dense_steps": 8, "strict": True})
        torch.manual_seed(1)
        q = torch.randn(7, 2, 128)
        k = torch.randn_like(q)
        v = torch.randn_like(q)

        dense_output = torch.randn(7, 256)
        with ExitStack() as stack:
            load_kernel = stack.enter_context(mock.patch("lightx2v.common.ops.attn.sol_attn._load_sol_attn"))
            dense_apply = stack.enter_context(mock.patch.object(backend.dense_backend, "apply", return_value=dense_output))
            actual = backend.apply(q, k, v, scheduler=SimpleNamespace(step_index=7), block_idx=1)

        load_kernel.assert_not_called()
        dense_apply.assert_called_once()
        self.assertIs(actual, dense_output)

    def test_dense_backend_can_use_sage_attention2(self):
        backend = SolAttnWeight()
        backend.set_config({"dense_backend": "sage_attn2"})

        self.assertEqual(backend.dense_backend_name, "sage_attn2")
        self.assertEqual(type(backend.dense_backend).__name__, "SageAttn2Weight")

    def test_invalid_dense_backend_is_rejected(self):
        backend = SolAttnWeight()
        with self.assertRaisesRegex(ValueError, "dense_backend must be one of"):
            backend.set_config({"dense_backend": "unknown"})

    def test_dense_layer_bypasses_sol_kernel_after_warmup(self):
        backend = SolAttnWeight()
        backend.set_config({"dense_steps": 8, "dense_layers": "0-1,4", "strict": True})
        torch.manual_seed(2)
        q = torch.randn(7, 2, 128)
        k = torch.randn_like(q)
        v = torch.randn_like(q)

        with mock.patch("lightx2v.common.ops.attn.sol_attn._load_sol_attn") as load_kernel:
            actual = backend.apply(q, k, v, scheduler=SimpleNamespace(step_index=8), block_idx=0)

        expected = (
            F.scaled_dot_product_attention(
                q.unsqueeze(0).transpose(1, 2),
                k.unsqueeze(0).transpose(1, 2),
                v.unsqueeze(0).transpose(1, 2),
            )
            .transpose(1, 2)
            .reshape(7, -1)
        )
        self.assertEqual(backend.dense_layers, frozenset({0, 1, 4}))
        load_kernel.assert_not_called()
        torch.testing.assert_close(actual, expected)

    def test_missing_dense_guard_metadata_uses_dense_attention(self):
        backend = SolAttnWeight()
        backend.set_config({"dense_steps": 8, "strict": True})
        q = torch.randn(7, 2, 128)
        k = torch.randn_like(q)
        v = torch.randn_like(q)

        with mock.patch("lightx2v.common.ops.attn.sol_attn._load_sol_attn") as load_kernel:
            actual = backend.apply(q, k, v)

        self.assertEqual(actual.shape, (7, 256))
        load_kernel.assert_not_called()

    def test_morton_permutation_round_trip(self):
        permutation, inverse = _morton3d_indices((3, 4, 5), torch.device("cpu"))
        values = torch.arange(60)
        torch.testing.assert_close(values.index_select(0, permutation).index_select(0, inverse), values)

    def test_preordered_morton_skips_per_attention_reorder(self):
        backend = SolAttnWeight()
        backend.set_config({"reorder": "morton3d", "strict": True})
        q = torch.randn(8, 2, 128, dtype=torch.bfloat16)
        k = torch.randn_like(q)
        v = torch.randn_like(q)

        def fake_kernel(q_bthd, _k_bthd, _v_bthd, **_kwargs):
            return q_bthd

        with ExitStack() as stack:
            stack.enter_context(mock.patch.object(SolAttnWeight, "_ineligibility_reason", return_value=None))
            stack.enter_context(mock.patch("lightx2v.common.ops.attn.sol_attn._load_sol_attn", return_value=fake_kernel))
            morton_indices = stack.enter_context(mock.patch("lightx2v.common.ops.attn.sol_attn._morton3d_indices"))
            stack.enter_context(mock.patch("torch.cuda.get_device_capability", return_value=(9, 0)))
            actual = backend.apply(q, k, v, sol_morton_preordered=True)

        morton_indices.assert_not_called()
        torch.testing.assert_close(actual, q.reshape(8, -1))

    def test_wan_global_morton_wraps_the_block_stack_once(self):
        infer = WanTransformerInfer.__new__(WanTransformerInfer)
        infer.config = {"sol_attn_setting": {"strict": True}}
        infer._sol_morton_preordered = False
        infer._sol_morton_log_keys = set()
        infer.rope_positions = None
        infer.cos_sin = torch.arange(8).view(8, 1, 1)

        original_x = torch.arange(24).view(8, 3)
        original_cos_sin = infer.cos_sin
        pre_infer_out = SimpleNamespace(
            x=original_x,
            grid_sizes=SimpleNamespace(tuple=(2, 2, 2)),
        )
        permutation, _ = _morton3d_indices((2, 2, 2), torch.device("cpu"))

        def fake_block_stack(_blocks, x, pre):
            self.assertTrue(infer._sol_morton_preordered)
            torch.testing.assert_close(x, original_x.index_select(0, permutation))
            torch.testing.assert_close(pre.x, x)
            torch.testing.assert_close(infer.cos_sin, original_cos_sin.index_select(0, permutation))
            return x + 1

        infer.infer_func = fake_block_stack
        actual = infer._infer_main_blocks_with_morton([], pre_infer_out)

        torch.testing.assert_close(actual, original_x + 1)
        self.assertIs(pre_infer_out.x, original_x)
        self.assertIs(infer.cos_sin, original_cos_sin)
        self.assertFalse(infer._sol_morton_preordered)

    def test_minimax_h3_passes_dense_guard_metadata(self):
        class Identity:
            @staticmethod
            def apply(value):
                return value

        class Rope:
            @staticmethod
            def apply(q, k, *_args, **_kwargs):
                return q, k

        class Attention:
            call = None

            @classmethod
            def apply(cls, **kwargs):
                cls.call = kwargs
                return kwargs["v"].reshape(kwargs["v"].shape[0], -1)

        infer = MiniMaxH3TransformerInfer.__new__(MiniMaxH3TransformerInfer)
        infer.num_heads = 2
        infer.head_dim = 4
        infer.infer_dtype = torch.float32
        infer.scheduler = SimpleNamespace(step_index=6)
        infer.block_idx = 3
        weights = SimpleNamespace(
            to_q=Identity(),
            to_k=Identity(),
            to_v=Identity(),
            norm_q=Identity(),
            norm_k=Identity(),
            rope=Rope(),
            calculate=Attention(),
            to_out=Identity(),
        )
        pre_infer_out = SimpleNamespace(
            rotary_emb=(torch.empty(5, 4), torch.empty(5, 4)),
            sequence_parallel_state=None,
        )

        actual = infer._attention(weights, torch.randn(5, 8), pre_infer_out)

        self.assertEqual(actual.shape, (5, 8))
        self.assertIs(Attention.call["scheduler"], infer.scheduler)
        self.assertEqual(Attention.call["block_idx"], 3)
        self.assertFalse(Attention.call["causal"])


if __name__ == "__main__":
    unittest.main()
