import os
import unittest
from types import SimpleNamespace
from unittest import mock

import torch
import torch.nn.functional as F

os.environ.setdefault("SKIP_PLATFORM_CHECK", "1")

from lightx2v.common.ops.moe.fused_moe import TorchExpertLoopFusedMoE  # noqa: E402
from lightx2v.models.networks.hunyuan_image3.config import normalize_hunyuan_image3_config  # noqa: E402
from lightx2v.models.networks.hunyuan_image3.weights.common import (  # noqa: E402
    HunyuanImage3MoEWeights,
    _MicroRouteFusedMoE,
)
from lightx2v.utils.registry_factory import FUSED_MOE_REGISTER  # noqa: E402


def _phase_aware_config(moe_backend, micro_shard_count=2):
    return {
        "task": "t2i",
        "bot_task": "image",
        "moe_backend": moe_backend,
        "hidden_size": 4096,
        "num_attention_heads": 4,
        "num_key_value_heads": 4,
        "intermediate_size": 8,
        "moe_intermediate_size": 1536,
        "num_experts": 64,
        "moe_topk": 8,
        "num_shared_expert": 1,
        "vocab_size": 8,
        "parallel": {
            "phase_aware": True,
            "storage_tensor_p_size": 1,
            "ar": {"tensor_p_size": micro_shard_count, "seq_p_size": 1},
            "denoise": {"tensor_p_size": 1, "seq_p_size": micro_shard_count},
            "cfg_p_size": 1,
            "pipeline_parallel": False,
            "cfg_mode": "batch",
        },
    }


def _resident_weights(moe_backend):
    weights = object.__new__(HunyuanImage3MoEWeights)
    weights.num_experts = 3
    weights.moe_backend = moe_backend
    weights.parallel_context = SimpleNamespace(local_micro_shard_id=1)
    weights.micro_shard_count = 2
    weights.storage_tp_rank = 0
    weights.logical_tp_size = 2
    weights.tune_max_num_tokens = 64
    weights.moe_fc1_weight = torch.randn(2, 3, 6, 4)
    weights.moe_fc2_weight = torch.randn(2, 3, 4, 3)
    weights._moe_weights_initialized = True
    weights._fused_moe_by_phase = {}
    return weights


class _FakeBackend:
    def __init__(self, name):
        self.name = name

    def apply(self, input, token_selected_experts, token_final_scales, output=None):
        raise AssertionError("backend execution is outside this construction test")


class HunyuanImage3FusedMoETest(unittest.TestCase):
    def test_legacy_moe_keys_are_rejected(self):
        legacy_values = {
            "moe_impl": "flashinfer",
            "flashinfer_multi_micro": True,
            "flashinfer_multi_micro_backend": "grouped_mm",
        }
        for key, value in legacy_values.items():
            with self.subTest(key=key):
                config = {"moe_backend": "flashinfer", key: value}
                with self.assertRaisesRegex(ValueError, "legacy MoE keys"):
                    normalize_hunyuan_image3_config(config)

    def test_backend_validation_accepts_only_the_three_model_backends(self):
        for backend in ("flashinfer", "multi_micro", "torch_expert_loop"):
            with self.subTest(backend=backend):
                config = _phase_aware_config(f" {backend.upper()} ")
                normalized = normalize_hunyuan_image3_config(config)
                self.assertEqual(normalized["moe_backend"], backend)

        with self.assertRaisesRegex(ValueError, "requires moe_backend"):
            normalize_hunyuan_image3_config({})
        for backend in ("eager", "torch", "torch_grouped_mm"):
            with self.subTest(invalid_backend=backend):
                with self.assertRaisesRegex(ValueError, "moe_backend must be one of"):
                    normalize_hunyuan_image3_config({"moe_backend": backend})

    def test_multi_micro_requires_phase_aware_two_micro_topology(self):
        with self.assertRaisesRegex(ValueError, "phase-aware parallel configuration"):
            normalize_hunyuan_image3_config({"moe_backend": "multi_micro"})

        non_phase_aware = {
            "moe_backend": "multi_micro",
            "parallel": {"phase_aware": False},
        }
        with self.assertRaisesRegex(ValueError, "requires parallel.phase_aware=true"):
            normalize_hunyuan_image3_config(non_phase_aware)

        for micro_shard_count in (1, 4):
            with self.subTest(micro_shard_count=micro_shard_count):
                config = _phase_aware_config("multi_micro", micro_shard_count)
                with self.assertRaisesRegex(ValueError, "exactly two micro shards"):
                    normalize_hunyuan_image3_config(config)

        normalized = normalize_hunyuan_image3_config(_phase_aware_config("multi_micro", 2))
        self.assertEqual(normalized["parallel"]["micro_shard_count"], 2)

    def test_multi_micro_validates_the_fixed_kernel_contract(self):
        invalid_values = (
            ("num_experts", 32, "64 experts"),
            ("moe_topk", 4, "top-8 routing"),
            ("hidden_size", 2048, "hidden_size=4096"),
            ("moe_intermediate_size", 768, "= 768"),
        )
        for key, value, message in invalid_values:
            with self.subTest(key=key):
                config = _phase_aware_config("multi_micro", 2)
                config[key] = value
                with self.assertRaisesRegex(ValueError, message):
                    normalize_hunyuan_image3_config(config)

    def test_micro_route_adapter_expands_ids_and_repeats_scales(self):
        class RecordingBackend:
            def __init__(self):
                self.expert_ids = None
                self.scales = None

            def apply(self, input, expert_ids, scales, output=None):
                self.expert_ids = expert_ids.clone()
                self.scales = scales.clone()
                result = output if output is not None else torch.empty_like(input)
                result.zero_()
                return result

        recorder = RecordingBackend()
        backend = _MicroRouteFusedMoE(recorder, num_experts=3, micro_shard_count=2)
        input = torch.randn(2, 4)
        expert_ids = torch.tensor([[1, 0], [2, 1]])
        scales = torch.tensor([[0.25, 0.75], [0.4, 0.6]])
        output = torch.empty_like(input)

        actual = backend.apply(input, expert_ids, scales, output)

        self.assertIs(actual, output)
        torch.testing.assert_close(
            recorder.expert_ids,
            torch.tensor([[1, 4, 0, 3], [2, 5, 1, 4]]),
        )
        torch.testing.assert_close(
            recorder.scales,
            torch.tensor([[0.25, 0.25, 0.75, 0.75], [0.4, 0.4, 0.6, 0.6]]),
        )

    def test_torch_micro_major_route_adapter_matches_explicit_reference(self):
        torch.manual_seed(23)
        micro_shard_count = 2
        num_experts = 3
        num_tokens = 5
        top_k = 2
        hidden_size = 4
        intermediate_size = 3

        input = torch.randn(num_tokens, hidden_size, dtype=torch.float64)
        expert_ids = torch.tensor([[0, 2], [1, 0], [2, 1], [0, 1], [2, 0]])
        scales = torch.randn(num_tokens, top_k, dtype=torch.float32)
        fc1_weight = torch.randn(
            micro_shard_count,
            num_experts,
            2 * intermediate_size,
            hidden_size,
            dtype=torch.float64,
        )
        fc2_weight = torch.randn(
            micro_shard_count,
            num_experts,
            hidden_size,
            intermediate_size,
            dtype=torch.float64,
        )

        packed_backend = TorchExpertLoopFusedMoE(
            fc1_weight.flatten(0, 1),
            fc2_weight.flatten(0, 1),
            "swiglu",
        )
        backend = _MicroRouteFusedMoE(packed_backend, num_experts, micro_shard_count)
        output = torch.full((num_tokens, hidden_size), float("nan"), dtype=input.dtype)
        actual = backend.apply(input, expert_ids, scales, output)

        expected = torch.zeros_like(output)
        for token_index in range(num_tokens):
            for route_index in range(top_k):
                expert_index = int(expert_ids[token_index, route_index])
                scale = float(scales[token_index, route_index])
                for micro_index in range(micro_shard_count):
                    projected = F.linear(input[token_index], fc1_weight[micro_index, expert_index])
                    value, gate = projected.chunk(2)
                    partial = F.linear(value * F.silu(gate), fc2_weight[micro_index, expert_index])
                    expected[token_index].add_(partial, alpha=scale)

        self.assertIs(actual, output)
        torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)

    def test_torch_micro_major_route_adapter_accepts_empty_batch(self):
        num_experts = 3
        hidden_size = 4
        intermediate_size = 3
        fc1_weight = torch.randn(2, num_experts, 2 * intermediate_size, hidden_size)
        fc2_weight = torch.randn(2, num_experts, hidden_size, intermediate_size)
        packed_backend = TorchExpertLoopFusedMoE(
            fc1_weight.flatten(0, 1),
            fc2_weight.flatten(0, 1),
            "swiglu",
        )
        backend = _MicroRouteFusedMoE(packed_backend, num_experts, 2)
        output = torch.empty(0, hidden_size)

        actual = backend.apply(
            torch.empty(0, hidden_size),
            torch.empty(0, 2, dtype=torch.int64),
            torch.empty(0, 2),
            output,
        )

        self.assertIs(actual, output)

    def test_backend_builder_uses_only_the_selected_registry_keys(self):
        expected_calls = {
            "flashinfer": ["flashinfer", "flashinfer"],
            "multi_micro": ["flashinfer", "multi_micro"],
            "torch_expert_loop": ["torch_expert_loop", "torch_expert_loop"],
        }

        for backend, expected in expected_calls.items():
            with self.subTest(backend=backend):
                calls = []

                def factory(name):
                    def build(*args, **kwargs):
                        calls.append((name, args, kwargs))
                        return _FakeBackend(name)

                    return build

                replacements = {
                    name: factory(name)
                    for name in ("flashinfer", "multi_micro", "torch_grouped_mm", "torch_expert_loop")
                }
                weights = _resident_weights(backend)
                with mock.patch.dict(FUSED_MOE_REGISTER._dict, replacements):
                    weights._build_fused_moe_backends()

                self.assertEqual([name for name, _, _ in calls], expected)
                self.assertEqual(set(weights._fused_moe_by_phase), {"ar", "denoise"})

                if backend == "torch_expert_loop":
                    self.assertEqual(calls[0][1][0].shape, (3, 6, 4))
                    self.assertEqual(calls[1][1][0].shape, (6, 6, 4))
                    self.assertIsInstance(weights._fused_moe_by_phase["denoise"], _MicroRouteFusedMoE)
                elif backend == "multi_micro":
                    self.assertEqual(calls[1][1][0].shape, (2, 3, 6, 4))
                    self.assertEqual(calls[1][1][1].shape, (2, 3, 4, 3))


if __name__ == "__main__":
    unittest.main()
