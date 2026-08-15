import os
import unittest
from types import SimpleNamespace
from unittest import mock

import torch

os.environ.setdefault("SKIP_PLATFORM_CHECK", "1")

from lightx2v.models.networks.hunyuan_image3.config import normalize_hunyuan_image3_config  # noqa: E402
from lightx2v.models.networks.hunyuan_image3.weights.common import HunyuanImage3MoEWeights, _MicroRouteFusedMoE  # noqa: E402
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


class _RecordingBackend:
    def __init__(self):
        self.call = None

    def apply(self, input, token_selected_experts, token_final_scales, output=None):
        self.call = (input, token_selected_experts, token_final_scales, output)
        if output is None:
            return torch.zeros_like(input)
        output.zero_()
        return output


class HunyuanImage3FusedMoETest(unittest.TestCase):
    def test_micro_route_adapter_expands_experts_and_scales(self):
        backend = _RecordingBackend()
        adapter = _MicroRouteFusedMoE(backend, num_experts=3, micro_shard_count=2)
        input_tensor = torch.randn(1, 4)
        selected_experts = torch.tensor([[1, 0]], dtype=torch.int32)
        final_scales = torch.tensor([[0.25, 0.75]])
        output = torch.empty_like(input_tensor)

        result = adapter.apply(input_tensor, selected_experts, final_scales, output)

        self.assertIs(result, output)
        self.assertIs(adapter._modules["fused_moe"], backend)
        _, expanded_experts, expanded_scales, forwarded_output = backend.call
        torch.testing.assert_close(expanded_experts, torch.tensor([[1, 4, 0, 3]]))
        torch.testing.assert_close(expanded_scales, torch.tensor([[0.25, 0.25, 0.75, 0.75]]))
        self.assertIs(forwarded_output, output)

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

    def test_backend_builder_uses_only_the_selected_registry_keys(self):
        expected_calls = {
            "flashinfer": ["flashinfer", "flashinfer"],
            "multi_micro": ["flashinfer", "multi_micro"],
            "torch_grouped_mm": ["torch_grouped_mm", "torch_grouped_mm"],
        }

        for backend, expected in expected_calls.items():
            with self.subTest(backend=backend):
                calls = []

                def factory(name):
                    def build(*args, **kwargs):
                        calls.append((name, args, kwargs))
                        return _FakeBackend(name)

                    return build

                replacements = {name: factory(name) for name in ("flashinfer", "multi_micro", "torch_grouped_mm")}
                weights = _resident_weights(backend)
                with mock.patch.dict(FUSED_MOE_REGISTER._dict, replacements):
                    weights._build_fused_moe_backends()

                self.assertEqual([name for name, _, _ in calls], expected)
                self.assertEqual(set(weights._fused_moe_by_phase), {"ar", "denoise"})

                if backend == "multi_micro":
                    self.assertEqual(calls[1][1][0].shape, (2, 3, 6, 4))
                    self.assertEqual(calls[1][1][1].shape, (2, 3, 4, 3))

                if backend == "torch_grouped_mm":
                    self.assertEqual(calls[0][1][0].shape, (3, 6, 4))
                    self.assertEqual(calls[0][1][1].shape, (3, 4, 3))
                    self.assertEqual(calls[1][1][0].shape, (6, 6, 4))
                    self.assertEqual(calls[1][1][1].shape, (6, 4, 3))
                    denoise_backend = weights._fused_moe_by_phase["denoise"]
                    self.assertIsInstance(denoise_backend, _MicroRouteFusedMoE)
                    self.assertIs(denoise_backend._modules["fused_moe"], denoise_backend.fused_moe)

    def test_non_phase_aware_grouped_backend_uses_one_micro_shard(self):
        calls = []

        def build(*args, **kwargs):
            calls.append((args, kwargs))
            return _FakeBackend("torch_grouped_mm")

        weights = _resident_weights("torch_grouped_mm")
        weights.parallel_context = None
        weights.micro_shard_count = 1
        weights.logical_tp_size = 1
        weights.moe_fc1_weight = weights.moe_fc1_weight[:1]
        weights.moe_fc2_weight = weights.moe_fc2_weight[:1]
        with mock.patch.dict(FUSED_MOE_REGISTER._dict, {"torch_grouped_mm": build}):
            weights._build_fused_moe_backends()

        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][0][0].shape, (3, 6, 4))
        self.assertEqual(calls[0][0][1].shape, (3, 4, 3))
        self.assertEqual(set(weights._fused_moe_by_phase), {"default"})


if __name__ == "__main__":
    unittest.main()
