import os
import unittest
from types import SimpleNamespace

import torch
import torch.nn.functional as F

os.environ.setdefault("SKIP_PLATFORM_CHECK", "1")

from lightx2v.common.modules.weight_module import WeightModule  # noqa: E402
from lightx2v.common.ops.moe.fused_moe import TorchExpertLoopFusedMoE, TorchGroupedMMFusedMoE, create_local_fused_moe  # noqa: E402
from lightx2v.models.networks.hunyuan3d.weights.transformer_weights import Hunyuan3DMoEWeights  # noqa: E402
from lightx2v.models.networks.lingbot_video.infer.transformer_infer import LingBotVideoTransformerInfer  # noqa: E402
from lightx2v.models.networks.lingbot_video.weights.transformer_weights import LingBotVideoFFNWeights  # noqa: E402
from lightx2v.utils.registry_factory import FUSED_MOE_REGISTER  # noqa: E402


def _reference(input, expert_ids, scales, fc1_weight, fc2_weight, fc1_bias, fc2_bias, activation):
    output = torch.zeros(input.shape[0], fc2_weight.shape[1], dtype=torch.float32, device=input.device)
    for token_idx in range(input.shape[0]):
        for route_idx in range(expert_ids.shape[1]):
            expert_idx = int(expert_ids[token_idx, route_idx])
            hidden = F.linear(input[token_idx : token_idx + 1], fc1_weight[expert_idx], fc1_bias[expert_idx])
            if activation == "gelu":
                hidden = F.gelu(hidden, approximate="none")
            else:
                value, gate = hidden.chunk(2, dim=-1)
                hidden = value * F.silu(gate)
            expert_output = F.linear(hidden, fc2_weight[expert_idx], fc2_bias[expert_idx])
            output[token_idx].add_(expert_output[0].float(), alpha=float(scales[token_idx, route_idx]))
    return output.to(input.dtype)


def _split_swiglu_reference(input, expert_ids, scales, w1, w2, w3):
    output = torch.zeros_like(input, dtype=torch.float32)
    for token_idx in range(input.shape[0]):
        for route_idx in range(expert_ids.shape[1]):
            expert_idx = int(expert_ids[token_idx, route_idx])
            token = input[token_idx : token_idx + 1]
            gate = F.linear(token, w1[expert_idx])
            value = F.linear(token, w3[expert_idx])
            expert_output = F.linear(F.silu(gate) * value, w2[expert_idx])
            output[token_idx].add_(expert_output[0].float(), alpha=float(scales[token_idx, route_idx]))
    return output.to(input.dtype)


class TorchFusedMoETest(unittest.TestCase):
    def test_backends_are_registered(self):
        self.assertIs(FUSED_MOE_REGISTER["torch_expert_loop"], TorchExpertLoopFusedMoE)
        self.assertIs(FUSED_MOE_REGISTER["torch_grouped_mm"], TorchGroupedMMFusedMoE)
        for backend in ("flashinfer", "multi_micro", "torch_expert_loop", "torch_grouped_mm"):
            self.assertTrue(issubclass(FUSED_MOE_REGISTER[backend], WeightModule))

    def test_hunyuan3d_accepts_local_backends(self):
        for backend in ("torch_expert_loop", "flashinfer"):
            weights = Hunyuan3DMoEWeights({"moe_backend": backend, "num_experts": 2}, 0, "Default")
            self.assertEqual(weights.moe_backend, backend)

    def test_hunyuan3d_transfers_routed_weights_to_fused_module(self):
        weights = Hunyuan3DMoEWeights({"moe_backend": "torch_expert_loop", "num_experts": 2}, 0, "Default")
        for expert in weights.experts:
            expert.fc1.pin_weight = torch.randn(4, 8)
            expert.fc1.pin_bias = torch.randn(8)
            expert.fc2.pin_weight = torch.randn(8, 4)
            expert.fc2.pin_bias = torch.randn(4)

        weights._build_fused_moe()

        self.assertIs(weights._modules["fused_moe"], weights.fused_moe)
        self.assertIsInstance(weights.fused_moe, WeightModule)
        self.assertEqual(weights.fused_moe.fc1_weights[0].shape, (8, 4))
        for expert in weights.experts:
            self.assertIsNone(expert.fc1.pin_weight)
            self.assertIsNone(expert.fc2.pin_weight)
        for method in ("to_cuda", "to_cpu", "to_cuda_async", "to_cpu_async"):
            self.assertNotIn(method, Hunyuan3DMoEWeights.__dict__)

    def test_local_factory_preserves_backend_weight_layout(self):
        fc1_weight = tuple(torch.randn(3, 8, 4).unbind())
        fc2_weight = tuple(torch.randn(3, 5, 8).unbind())
        fc1_bias = tuple(torch.randn(3, 8).unbind())
        fc2_bias = tuple(torch.randn(3, 5).unbind())

        expert_loop = create_local_fused_moe("torch_expert_loop", fc1_weight, fc2_weight, "gelu", fc1_bias, fc2_bias)
        grouped_mm = create_local_fused_moe("torch_grouped_mm", fc1_weight, fc2_weight, "gelu", fc1_bias, fc2_bias)
        flashinfer = create_local_fused_moe("flashinfer", fc1_weight, fc2_weight, "gelu", fc1_bias, fc2_bias)

        self.assertEqual(expert_loop.fc1_weights[0].data_ptr(), fc1_weight[0].data_ptr())
        self.assertEqual(grouped_mm.fc1_weights, ())
        self.assertEqual(grouped_mm.grouped_fc1_weight.shape, (3, 8, 4))
        self.assertEqual(flashinfer.shards[0].fc1_weight.shape, (3, 8, 4))
        self.assertTrue(grouped_mm.grouped_fc1_weight.is_contiguous())

        self.assertEqual(flashinfer.shards[0].fc1_bias.shape, (3, 8))
        self.assertEqual(flashinfer.tp_size, 1)
        self.assertEqual(flashinfer.tune_max_num_tokens, 8192)

    def test_split_swiglu_matches_reference(self):
        torch.manual_seed(11)
        num_experts, num_tokens, top_k = 4, 6, 2
        hidden_size, intermediate_size = 5, 7
        input = torch.randn(num_tokens, hidden_size)
        expert_ids = torch.tensor([[0, 3], [1, 1], [3, 0], [0, 1], [3, 3], [1, 0]])
        scales = torch.randn(num_tokens, top_k)
        w1 = torch.randn(num_experts, intermediate_size, hidden_size)
        w2 = torch.randn(num_experts, hidden_size, intermediate_size)
        w3 = torch.randn(num_experts, intermediate_size, hidden_size)

        backend = create_local_fused_moe(
            "torch_expert_loop",
            w3,
            w2,
            "swiglu",
            fc1_gate_weight=w1,
        )
        actual = backend.apply(input, expert_ids, scales)
        expected = _split_swiglu_reference(input, expert_ids, scales, w1, w2, w3)

        torch.testing.assert_close(actual, expected)
        self.assertEqual(backend.fc1_weights[0].data_ptr(), w3[0].data_ptr())
        self.assertEqual(backend.fc1_gate_weights[0].data_ptr(), w1[0].data_ptr())
        self.assertEqual(backend.fc2_weights[0].data_ptr(), w2[0].data_ptr())

    def test_lingbot_default_backend_binds_split_weights(self):
        weights = LingBotVideoFFNWeights("blocks.0", 0, {"num_experts": 3})
        w1 = torch.randn(3, 7, 5)
        w2 = torch.randn(3, 5, 7)
        w3 = torch.randn(3, 7, 5)
        weights.experts.w1.tensor = w1
        weights.experts.w2.tensor = w2
        weights.experts.w3.tensor = w3

        weights._build_fused_moe()

        self.assertEqual(weights.moe_backend, "torch_grouped_mm")
        self.assertIsInstance(weights.fused_moe, TorchGroupedMMFusedMoE)
        self.assertEqual(weights.fused_moe.grouped_fc1_weight.data_ptr(), w3.data_ptr())
        self.assertEqual(weights.fused_moe.grouped_fc1_gate_weight.data_ptr(), w1.data_ptr())
        self.assertEqual(weights.fused_moe.grouped_fc2_weight.data_ptr(), w2.data_ptr())

    def test_lingbot_infer_calls_routed_module_once_and_adds_shared_expert(self):
        class Backend:
            def __init__(self):
                self.calls = 0

            def apply(self, hidden_states, top_indices, top_scores):
                self.calls += 1
                self.args = (hidden_states, top_indices, top_scores)
                return hidden_states + 1

        infer = object.__new__(LingBotVideoTransformerInfer)
        top_indices = torch.tensor([[0, 1], [1, 0]])
        top_scores = torch.tensor([[0.6, 0.4], [0.3, 0.7]])
        infer._route = lambda weights, hidden_states: (top_indices, top_scores)
        infer._dense_mlp = lambda weights, hidden_states: hidden_states * 2
        backend = Backend()
        weights = SimpleNamespace(fused_moe=backend, shared_experts=object())
        hidden_states = torch.randn(2, 4)

        actual = infer._moe(weights, hidden_states)

        self.assertEqual(backend.calls, 1)
        self.assertIs(backend.args[0], hidden_states)
        self.assertIs(backend.args[1], top_indices)
        self.assertIs(backend.args[2], top_scores)
        torch.testing.assert_close(actual, hidden_states * 3 + 1)

    def test_local_flashinfer_rejects_mismatched_experts(self):
        fc1_weight = torch.randn(3, 8, 4)
        fc2_weight = torch.randn(2, 5, 8)
        with self.assertRaisesRegex(ValueError, "same number of experts"):
            create_local_fused_moe("flashinfer", fc1_weight, fc2_weight, "gelu")

    def test_expert_loop_matches_reference(self):
        torch.manual_seed(5)
        num_experts, num_tokens, top_k = 4, 7, 2
        hidden_size, intermediate_size, output_size = 6, 9, 5
        expert_ids = torch.tensor([[0, 2], [1, 1], [0, 2], [2, 0], [1, 2], [0, 0], [2, 1]])
        scales = torch.randn(num_tokens, top_k, dtype=torch.float64)
        storage = torch.randn(num_tokens, hidden_size * 2)

        for activation in ("gelu", "swiglu"):
            fc1_size = intermediate_size if activation == "gelu" else 2 * intermediate_size
            input = storage[:, ::2]
            fc1_weight = torch.randn(num_experts, fc1_size, hidden_size)
            fc2_weight = torch.randn(num_experts, output_size, intermediate_size)
            fc1_bias = torch.randn(num_experts, fc1_size)
            fc2_bias = torch.randn(num_experts, output_size)
            backend = TorchExpertLoopFusedMoE(
                tuple(fc1_weight.unbind()),
                tuple(fc2_weight.unbind()),
                activation=activation,
                fc1_bias=tuple(fc1_bias.unbind()),
                fc2_bias=tuple(fc2_bias.unbind()),
            )
            output = torch.full((num_tokens, output_size), 123.0)
            actual = backend.apply(input, expert_ids, scales, output=output)
            expected = _reference(input, expert_ids, scales, fc1_weight, fc2_weight, fc1_bias, fc2_bias, activation)

            self.assertIs(actual, output)
            torch.testing.assert_close(actual, expected)

    def test_empty_input_preserves_output_identity(self):
        backend = TorchExpertLoopFusedMoE(
            torch.randn(3, 8, 4),
            torch.randn(3, 5, 8),
            activation="gelu",
        )
        output = torch.empty(0, 5)
        actual = backend.apply(
            torch.empty(0, 4),
            torch.empty(0, 2, dtype=torch.int64),
            torch.empty(0, 2),
            output=output,
        )
        self.assertIs(actual, output)
        self.assertEqual(actual.shape, (0, 5))

    def test_routing_dtypes_are_validated(self):
        backend = TorchExpertLoopFusedMoE(
            torch.randn(3, 8, 4),
            torch.randn(3, 5, 8),
            activation="gelu",
        )
        input = torch.randn(2, 4)

        with self.assertRaisesRegex(TypeError, "integer dtype"):
            backend.apply(input, torch.ones(2, 1), torch.ones(2, 1))
        with self.assertRaisesRegex(TypeError, "floating-point dtype"):
            backend.apply(input, torch.ones(2, 1, dtype=torch.int64), torch.ones(2, 1, dtype=torch.int64))

    def test_grouped_mm_matches_expert_loop(self):
        if not torch.cuda.is_available() or not hasattr(torch, "_grouped_mm"):
            self.skipTest("torch._grouped_mm CUDA backend is unavailable")
        if torch.cuda.get_device_capability() != (9, 0):
            self.skipTest("grouped_mm test requires an SM90 GPU")

        torch.manual_seed(9)
        device = torch.device("cuda")
        num_experts, num_tokens, top_k = 4, 13, 2
        hidden_size, intermediate_size, output_size = 16, 32, 24
        input = torch.randn(num_tokens, hidden_size, device=device, dtype=torch.bfloat16)
        expert_ids = torch.tensor(
            [[0, 2], [3, 1], [1, 1], [0, 3], [0, 0], [3, 1], [1, 0], [0, 3], [1, 1], [0, 3], [3, 0], [1, 3], [0, 1]],
            device=device,
        )
        scales = torch.randn(num_tokens, top_k, device=device)
        fc1_weight = torch.randn(num_experts, intermediate_size, hidden_size, device=device, dtype=torch.bfloat16)
        fc2_weight = torch.randn(num_experts, output_size, intermediate_size, device=device, dtype=torch.bfloat16)
        fc1_bias = torch.randn(num_experts, intermediate_size, device=device, dtype=torch.bfloat16)
        fc2_bias = torch.randn(num_experts, output_size, device=device, dtype=torch.bfloat16)

        loop = TorchExpertLoopFusedMoE(fc1_weight, fc2_weight, activation="gelu", fc1_bias=fc1_bias, fc2_bias=fc2_bias)
        grouped = TorchGroupedMMFusedMoE(fc1_weight, fc2_weight, activation="gelu", fc1_bias=fc1_bias, fc2_bias=fc2_bias)
        expected = loop.apply(input, expert_ids, scales).float()
        actual = grouped.apply(input, expert_ids, scales).float()
        relative_l2 = (actual - expected).norm() / expected.norm()
        cosine = F.cosine_similarity(actual.flatten(), expected.flatten(), dim=0)

        self.assertLess(relative_l2.item(), 0.02)
        self.assertGreater(cosine.item(), 0.999)

    def test_split_swiglu_grouped_mm_matches_expert_loop(self):
        if not torch.cuda.is_available() or not hasattr(torch, "_grouped_mm"):
            self.skipTest("torch._grouped_mm CUDA backend is unavailable")
        if torch.cuda.get_device_capability() != (9, 0):
            self.skipTest("grouped_mm test requires an SM90 GPU")

        torch.manual_seed(13)
        device = torch.device("cuda")
        num_experts, num_tokens, top_k = 4, 13, 2
        hidden_size, intermediate_size = 16, 24
        input = torch.randn(num_tokens, hidden_size, device=device, dtype=torch.bfloat16)
        expert_ids = torch.tensor(
            [[0, 3], [3, 1], [1, 1], [0, 3], [0, 0], [3, 1], [1, 0], [0, 3], [1, 1], [0, 3], [3, 0], [1, 3], [0, 1]],
            device=device,
        )
        scales = torch.randn(num_tokens, top_k, device=device)
        w1 = torch.randn(num_experts, intermediate_size, hidden_size, device=device, dtype=torch.bfloat16)
        w2 = torch.randn(num_experts, hidden_size, intermediate_size, device=device, dtype=torch.bfloat16)
        w3 = torch.randn(num_experts, intermediate_size, hidden_size, device=device, dtype=torch.bfloat16)

        loop = create_local_fused_moe("torch_expert_loop", w3, w2, "swiglu", fc1_gate_weight=w1)
        grouped = create_local_fused_moe("torch_grouped_mm", w3, w2, "swiglu", fc1_gate_weight=w1)
        expected = loop.apply(input, expert_ids, scales).float()
        actual = grouped.apply(input, expert_ids, scales).float()
        relative_l2 = (actual - expected).norm() / expected.norm()
        cosine = F.cosine_similarity(actual.flatten(), expected.flatten(), dim=0)

        self.assertEqual(grouped.grouped_fc1_weight.data_ptr(), w3.data_ptr())
        self.assertEqual(grouped.grouped_fc1_gate_weight.data_ptr(), w1.data_ptr())
        self.assertEqual(grouped.grouped_fc2_weight.data_ptr(), w2.data_ptr())
        self.assertLess(relative_l2.item(), 0.02)
        self.assertGreater(cosine.item(), 0.999)


if __name__ == "__main__":
    unittest.main()
