import os
import unittest
from types import SimpleNamespace

import torch
import torch.nn.functional as F

try:
    import torch_npu
except (ImportError, RuntimeError):
    torch_npu = None


def _npu_is_available():
    if torch_npu is None:
        return False
    try:
        return torch.npu.is_available()
    except RuntimeError:
        return False


def _make_moe_checkpoint(num_experts, hidden_dim, intermediate_dim, device, dtype):
    prefix = "blocks.0.moe"
    checkpoint = {
        f"{prefix}.gate.weight": torch.randn(num_experts, hidden_dim, device=device, dtype=dtype),
        f"{prefix}.shared_experts.net.0.proj.weight": torch.randn(intermediate_dim, hidden_dim, device=device, dtype=dtype),
        f"{prefix}.shared_experts.net.0.proj.bias": torch.randn(intermediate_dim, device=device, dtype=dtype),
        f"{prefix}.shared_experts.net.2.weight": torch.randn(hidden_dim, intermediate_dim, device=device, dtype=dtype),
        f"{prefix}.shared_experts.net.2.bias": torch.randn(hidden_dim, device=device, dtype=dtype),
    }
    fc1_weights, fc2_weights = [], []
    for expert_idx in range(num_experts):
        expert_prefix = f"{prefix}.experts.{expert_idx}"
        fc1_weight = torch.randn(intermediate_dim, hidden_dim, device=device, dtype=dtype)
        fc2_weight = torch.randn(hidden_dim, intermediate_dim, device=device, dtype=dtype)
        checkpoint[f"{expert_prefix}.net.0.proj.weight"] = fc1_weight
        checkpoint[f"{expert_prefix}.net.0.proj.bias"] = torch.randn(intermediate_dim, device=device, dtype=dtype)
        checkpoint[f"{expert_prefix}.net.2.weight"] = fc2_weight
        checkpoint[f"{expert_prefix}.net.2.bias"] = torch.randn(hidden_dim, device=device, dtype=dtype)
        fc1_weights.append(fc1_weight)
        fc2_weights.append(fc2_weight)
    return checkpoint, fc1_weights, fc2_weights


@unittest.skipUnless(_npu_is_available(), "Ascend NPU is not available")
class Hunyuan3DNpuMoeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ.setdefault("PLATFORM", "ascend_npu")
        os.environ.setdefault("AI_DEVICE", "npu")

    def test_fp16_bias_empty_experts_and_unpermute_matches_reference(self):
        from lightx2v.models.networks.hunyuan3d.infer.moe_npu import infer_routed_moe_npu

        torch.manual_seed(2026)
        device = torch.device("npu")
        dtype = torch.float16
        num_tokens, hidden_dim, intermediate_dim = 8, 64, 128
        num_experts, top_k = 8, 2

        flat = (torch.randn(num_tokens, hidden_dim, device=device, dtype=dtype) / 10).contiguous()
        fc1_weight = (torch.randn(num_experts, hidden_dim, intermediate_dim, device=device, dtype=dtype) / 10).contiguous()
        fc2_weight = (torch.randn(num_experts, intermediate_dim, hidden_dim, device=device, dtype=dtype) / 10).contiguous()
        fc1_bias = (torch.randn(num_experts, intermediate_dim, device=device, dtype=dtype) / 10).contiguous()
        fc2_bias = (torch.randn(num_experts, hidden_dim, device=device, dtype=dtype) / 10).contiguous()

        routes = torch.tensor(
            [[0, 1], [0, 2], [1, 2], [0, 4], [1, 4], [2, 4], [0, 1], [0, 2]],
            device=device,
            dtype=torch.int64,
        )
        logits = torch.full((num_tokens, num_experts), -8.0, device=device, dtype=dtype)
        logits.scatter_(1, routes[:, :1], 4.0)
        logits.scatter_(1, routes[:, 1:], 3.0)

        moe_weights = SimpleNamespace(
            moe_top_k=top_k,
            num_experts=num_experts,
            _npu_cache_ready=True,
            _npu_fc1_weight=fc1_weight,
            _npu_fc2_weight=fc2_weight,
            _npu_fc1_bias=fc1_bias,
            _npu_fc2_bias=fc2_bias,
            _ensure_npu_weights=lambda: None,
        )

        actual = infer_routed_moe_npu(moe_weights, flat, logits)

        topk_weight, topk_idx, _ = torch_npu.npu_moe_gating_top_k_softmax(logits, None, k=top_k)
        _, _, expert_counts, _ = torch_npu.npu_moe_init_routing_v2(
            flat,
            topk_idx,
            active_num=num_tokens * top_k,
            expert_num=num_experts,
            drop_pad_mode=0,
            expert_tokens_num_type=1,
            expert_tokens_num_flag=True,
            quant_mode=-1,
            active_expert_range=[0, num_experts],
            row_idx_type=0,
        )
        self.assertEqual(expert_counts.cpu().tolist(), [5, 4, 4, 0, 3, 0, 0, 0])

        flat_cpu = flat.cpu().float()
        fc1_weight_cpu = fc1_weight.cpu().float()
        fc2_weight_cpu = fc2_weight.cpu().float()
        fc1_bias_cpu = fc1_bias.cpu().float()
        fc2_bias_cpu = fc2_bias.cpu().float()
        topk_weight_cpu = topk_weight.cpu().float()
        topk_idx_cpu = topk_idx.cpu()
        expected = torch.zeros(num_tokens, hidden_dim, dtype=torch.float32)
        for token_idx in range(num_tokens):
            for route_idx in range(top_k):
                expert_idx = int(topk_idx_cpu[token_idx, route_idx])
                expert_hidden = F.gelu(
                    flat_cpu[token_idx] @ fc1_weight_cpu[expert_idx] + fc1_bias_cpu[expert_idx],
                    approximate="none",
                )
                expert_output = expert_hidden @ fc2_weight_cpu[expert_idx] + fc2_bias_cpu[expert_idx]
                expected[token_idx] += expert_output * topk_weight_cpu[token_idx, route_idx]

        self.assertEqual(actual.dtype, dtype)
        self.assertEqual(tuple(actual.shape), (num_tokens, hidden_dim))
        torch.testing.assert_close(actual.cpu().float(), expected, rtol=2e-2, atol=2e-2)

    def test_real_weight_loader_packs_and_invalidates_cache(self):
        from lightx2v.models.networks.hunyuan3d.weights.transformer_weights import Hunyuan3DMoEWeights

        torch.manual_seed(2027)
        device = torch.device("npu")
        dtype = torch.float16
        num_experts, hidden_dim, intermediate_dim = 3, 16, 32
        checkpoint, fc1_weights, fc2_weights = _make_moe_checkpoint(num_experts, hidden_dim, intermediate_dim, device, dtype)
        moe_weights = Hunyuan3DMoEWeights(
            {"num_experts": num_experts, "moe_top_k": 2, "moe_backend": "npu"},
            block_idx=0,
            mm_type="Default",
        )
        moe_weights.load(checkpoint)

        self.assertEqual(tuple(moe_weights._npu_fc1_weight.shape), (num_experts, hidden_dim, intermediate_dim))
        self.assertEqual(tuple(moe_weights._npu_fc2_weight.shape), (num_experts, intermediate_dim, hidden_dim))
        self.assertEqual(moe_weights._npu_fc1_bias.dtype, dtype)
        self.assertEqual(moe_weights._npu_fc2_bias.dtype, dtype)
        torch.testing.assert_close(moe_weights._npu_fc1_weight[0], fc1_weights[0].t(), rtol=0, atol=0)
        torch.testing.assert_close(moe_weights._npu_fc2_weight[0], fc2_weights[0].t(), rtol=0, atol=0)

        packed_data_ptr = moe_weights._npu_fc1_weight.data_ptr()
        moe_weights._ensure_npu_weights()
        self.assertEqual(moe_weights._npu_fc1_weight.data_ptr(), packed_data_ptr)

        moe_weights.register_diff({})
        self.assertFalse(moe_weights._npu_cache_ready)
        self.assertFalse(hasattr(moe_weights, "_npu_fc1_weight"))
        moe_weights._ensure_npu_weights()

        moe_weights.to_cpu()
        self.assertFalse(moe_weights._npu_cache_ready)
        self.assertFalse(hasattr(moe_weights, "_npu_fc1_weight"))
        moe_weights.to_cuda()
        self.assertTrue(moe_weights._npu_cache_ready)
        torch.testing.assert_close(moe_weights._npu_fc1_weight[0], fc1_weights[0].t(), rtol=0, atol=0)

        moe_weights.experts[0].fc1.has_lora_branch = True
        moe_weights._clear_packed_weights()
        with self.assertRaisesRegex(NotImplementedError, "routed-expert LoRA"):
            moe_weights._ensure_npu_weights()

        with self.assertRaisesRegex(ValueError, "floating-point routed experts"):
            Hunyuan3DMoEWeights(
                {"num_experts": num_experts, "moe_backend": "npu"},
                block_idx=0,
                mm_type="int8-npu",
            )

    def test_bf16_loader_converts_grouped_matmul_bias_to_fp32(self):
        from lightx2v.models.networks.hunyuan3d.weights.transformer_weights import Hunyuan3DMoEWeights

        device = torch.device("npu")
        checkpoint, _, _ = _make_moe_checkpoint(2, 8, 16, device, torch.bfloat16)
        moe_weights = Hunyuan3DMoEWeights(
            {"num_experts": 2, "moe_top_k": 2, "moe_backend": "npu"},
            block_idx=0,
            mm_type="Default",
        )
        moe_weights.load(checkpoint)

        self.assertEqual(moe_weights._npu_fc1_weight.dtype, torch.bfloat16)
        self.assertEqual(moe_weights._npu_fc2_weight.dtype, torch.bfloat16)
        self.assertEqual(moe_weights._npu_fc1_bias.dtype, torch.float32)
        self.assertEqual(moe_weights._npu_fc2_bias.dtype, torch.float32)


if __name__ == "__main__":
    unittest.main()
