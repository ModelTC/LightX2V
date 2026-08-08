import os
import unittest
from unittest import mock

import torch
import torch.nn.functional as F

os.environ.setdefault("SKIP_PLATFORM_CHECK", "1")

from lightx2v.common.ops.attn.sol_attn import SolAttnWeight, _morton3d_indices  # noqa: E402
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
        expected = F.scaled_dot_product_attention(
            q.unsqueeze(0).transpose(1, 2),
            k.unsqueeze(0).transpose(1, 2),
            v.unsqueeze(0).transpose(1, 2),
        ).transpose(1, 2).reshape(9, -1)
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
        backend.set_config({"tau": 1.25, "thresh_type": "exact", "kv_splits": 1, "strict": True})
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
            actual = backend.apply(q, k, v)
        self.assertEqual(actual.shape, (7, 256))
        torch.testing.assert_close(actual, (v + 1).reshape(7, 256))

    def test_morton_permutation_round_trip(self):
        permutation, inverse = _morton3d_indices((3, 4, 5), torch.device("cpu"))
        values = torch.arange(60)
        torch.testing.assert_close(values.index_select(0, permutation).index_select(0, inverse), values)


if __name__ == "__main__":
    unittest.main()
