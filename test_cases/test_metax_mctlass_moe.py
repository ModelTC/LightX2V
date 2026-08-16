import importlib.util
import os
import unittest
from unittest import mock

import torch
import torch.nn.functional as F

os.environ.setdefault("PLATFORM", "unit_test")

from lightx2v_platform.ops.moe import template as moe_template  # noqa: E402
from lightx2v_platform.ops.moe.metax_cuda import metax_mctlass_moe as metax_moe  # noqa: E402
from lightx2v_platform.registry_factory import PLATFORM_FUSED_MOE_REGISTER  # noqa: E402


def _reference_moe(input, expert_ids, scales, fc1_weight, fc2_weight, fc1_bias, fc2_bias):
    output = torch.zeros(input.shape[0], fc2_weight.shape[1], dtype=torch.float32, device=input.device)
    for token_index in range(input.shape[0]):
        for route_index in range(expert_ids.shape[1]):
            expert_index = int(expert_ids[token_index, route_index])
            hidden = F.linear(input[token_index : token_index + 1], fc1_weight[expert_index], fc1_bias[expert_index])
            hidden = F.gelu(hidden, approximate="none")
            expert_output = F.linear(hidden, fc2_weight[expert_index], fc2_bias[expert_index])
            output[token_index].add_(expert_output[0].float() * scales[token_index, route_index])
    return output


class MetaxMctlassFusedMoETest(unittest.TestCase):
    def test_backend_is_registered_without_loading_vendor_extensions(self):
        self.assertIs(PLATFORM_FUSED_MOE_REGISTER["metax_mctlass_moe"], metax_moe.MetaxMctlassFusedMoE)

    def test_rejects_non_bf16_before_loading_vendor_extensions(self):
        with mock.patch.object(metax_moe, "_load_metax_moe_ops") as load_ops:
            with self.assertRaisesRegex(TypeError, "only torch.bfloat16 weights"):
                metax_moe.MetaxMctlassFusedMoE(
                    torch.randn(3, 8, 4, dtype=torch.float16),
                    torch.randn(3, 5, 8, dtype=torch.float16),
                    "gelu",
                )
            load_ops.assert_not_called()

    def test_loader_rejects_missing_device_before_importing_extensions(self):
        with (
            mock.patch.object(torch.cuda, "is_available", return_value=False),
            mock.patch.object(metax_moe, "import_module") as import_module,
        ):
            with self.assertRaisesRegex(RuntimeError, "available MetaX CUDA device"):
                metax_moe._load_metax_moe_ops()
            import_module.assert_not_called()

    def test_preserves_packed_weight_layout_and_registers_biases(self):
        fc1_weight = torch.randn(3, 4, 8, dtype=torch.bfloat16).transpose(1, 2)
        fc2_weight = torch.randn(3, 8, 5, dtype=torch.bfloat16).transpose(1, 2)
        fc1_bias = torch.randn(3, 16, dtype=torch.bfloat16)[:, ::2]
        fc2_bias = torch.randn(3, 10, dtype=torch.bfloat16)[:, ::2]

        with mock.patch.object(metax_moe, "_load_metax_moe_ops", return_value=(object(), object(), object())):
            backend = metax_moe.MetaxMctlassFusedMoE(
                fc1_weight,
                fc2_weight,
                "gelu",
                fc1_bias,
                fc2_bias,
            )

        self.assertEqual(backend.grouped_fc1_weight.shape, (3, 8, 4))
        self.assertEqual(backend.grouped_fc2_weight.shape, (3, 5, 8))
        self.assertEqual(backend.grouped_fc1_bias.shape, (3, 8))
        self.assertEqual(backend.grouped_fc2_bias.shape, (3, 5))
        self.assertTrue(backend.grouped_fc1_weight.is_contiguous())
        self.assertTrue(backend.grouped_fc2_weight.is_contiguous())
        self.assertTrue(backend.grouped_fc1_bias.is_contiguous())
        self.assertTrue(backend.grouped_fc2_bias.is_contiguous())
        self.assertEqual(
            dict(backend.named_parameters()).keys(),
            {"grouped_fc1_weight", "grouped_fc2_weight", "grouped_fc1_bias", "grouped_fc2_bias"},
        )

    def test_c500_bf16_matches_reference(self):
        if not torch.cuda.is_available() or "MetaX" not in torch.cuda.get_device_name(0):
            self.skipTest("requires a MetaX CUDA device")
        if importlib.util.find_spec("mctlassEx") is None or importlib.util.find_spec("mcoplib._moe_C") is None:
            self.skipTest("requires mctlassEx and mcoplib._moe_C")

        torch.manual_seed(17)
        device = torch.device("cuda")
        num_experts, num_tokens, top_k = 4, 13, 2
        hidden_size, intermediate_size = 256, 256
        input = (torch.randn(num_tokens, hidden_size, device=device, dtype=torch.bfloat16) * 0.1).contiguous()
        fc1_weight = (torch.randn(num_experts, intermediate_size, hidden_size, device=device, dtype=torch.bfloat16) * 0.03).contiguous()
        fc2_weight = (torch.randn(num_experts, hidden_size, intermediate_size, device=device, dtype=torch.bfloat16) * 0.03).contiguous()
        fc1_bias = (torch.randn(num_experts, intermediate_size, device=device, dtype=torch.bfloat16) * 0.01).contiguous()
        fc2_bias = (torch.randn(num_experts, hidden_size, device=device, dtype=torch.bfloat16) * 0.01).contiguous()
        expert_ids = torch.tensor(
            [[0, 3], [1, 1], [0, 3], [3, 0], [1, 3], [0, 0], [3, 1], [1, 0], [0, 3], [1, 1], [3, 0], [1, 3], [0, 1]],
            dtype=torch.int64,
            device=device,
        )
        scales = torch.softmax(torch.randn(num_tokens, top_k, device=device), dim=-1)

        backend = metax_moe.MetaxMctlassFusedMoE(
            fc1_weight,
            fc2_weight,
            "gelu",
            fc1_bias,
            fc2_bias,
        )
        backend.to_cpu()
        self.assertEqual(backend.weight_device.type, "cpu")
        with mock.patch.object(moe_template, "AI_DEVICE", "cuda"):
            backend.to_cuda()
        self.assertEqual(backend.weight_device.type, "cuda")
        output = torch.full((num_tokens, hidden_size), torch.nan, dtype=torch.bfloat16, device=device)
        actual = backend.apply(input, expert_ids, scales, output=output)
        expected = _reference_moe(input, expert_ids, scales, fc1_weight, fc2_weight, fc1_bias, fc2_bias)
        torch.cuda.synchronize()

        relative_l2 = (actual.float() - expected).norm() / expected.norm()
        cosine = F.cosine_similarity(actual.flatten().float(), expected.flatten(), dim=0)
        self.assertIs(actual, output)
        self.assertLess(relative_l2.item(), 0.02)
        self.assertGreater(cosine.item(), 0.999)


if __name__ == "__main__":
    unittest.main()
