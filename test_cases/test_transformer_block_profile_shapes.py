import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from lightx2v.common.ops.mm.mm_weight import MMWeightTP
from lightx2v.models.networks.hunyuan3d.infer.block_profile import Hunyuan3DBlockProfile
from lightx2v.models.networks.minimax_h3.infer.block_profile import MiniMaxH3BlockProfile
from lightx2v.models.networks.wan.infer.block_profile import WanBlockProfile
from lightx2v.utils import op_shape_trace as ost


def linear(out_features, in_features, *, dtype=torch.bfloat16):
    weight = torch.empty(out_features, in_features, dtype=dtype)
    return SimpleNamespace(_get_actual_weight=lambda: weight)


def attention(hidden):
    return SimpleNamespace(
        has_fused_qkv=False,
        to_q=linear(hidden, hidden),
        to_k=linear(hidden, hidden),
        to_v=linear(hidden, hidden),
        to_qkv=None,
        out_proj=linear(hidden, hidden),
    )


def h3_linear(k, n, *, lora_rank=0):
    linear = SimpleNamespace(
        weight=torch.empty(k, n),
        has_lora_branch=lora_rank > 0,
    )
    linear._get_actual_weight = lambda: linear.weight
    if lora_rank:
        linear.lora_down = torch.empty(lora_rank, k)
    return linear


def h3_tp_linear(k, n):
    linear = MMWeightTP("weight", None)
    linear._mm.weight = torch.empty(k, n)
    return linear


class TransformerBlockProfileShapeTest(unittest.TestCase):
    def test_hunyuan_batch_and_moe_shapes(self):
        hidden = 8
        profile = Hunyuan3DBlockProfile(
            {
                "num_heads": 2,
                "hidden_size": hidden,
                "moe_intermediate_size": 16,
                "moe_top_k": 2,
                "moe_backend": "torch_expert_loop",
            }
        )
        block = SimpleNamespace(
            skip_linear=None,
            attn1=attention(hidden),
            attn2=attention(hidden),
            moe=SimpleNamespace(
                gate=linear(4, hidden),
                shared_experts=SimpleNamespace(
                    fc1=linear(16, hidden),
                    fc2=linear(hidden, 16),
                ),
            ),
            mlp=None,
        )
        profile.bind(block, cond_len=5, hidden_states=torch.empty(2, 3, hidden))

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "ops.jsonl"
            try:
                ost.begin_recording(path)
                profile.self_attn()
                profile.cross_attn()
                profile.moe([1, 2, 3, 6])
                ost.flush(path)
            finally:
                ost.end_recording()
            entries = [json.loads(line) for line in path.read_text().splitlines()]

        by_tag = {entry["tag"]: entry for entry in entries}
        self.assertEqual(by_tag["self_sdpa"]["B"], 2)
        self.assertEqual(by_tag["self_sdpa"]["flops"], 4 * 2 * 2 * 3 * 3 * 4)
        self.assertEqual(by_tag["cross_sdpa"]["flops"], 4 * 2 * 2 * 3 * 5 * 4)
        self.assertEqual(by_tag["moe_routed"]["intermediate"], 16)
        self.assertEqual(by_tag["moe_routed"]["routed_tokens"], 12)

    def test_minimax_h3_inventory_uses_runtime_sp_and_cache_state(self):
        hidden = 8
        block = SimpleNamespace(
            adaln=h3_linear(hidden, 6 * hidden),
            attn=SimpleNamespace(
                to_q=h3_tp_linear(hidden, hidden),
                to_k=h3_linear(hidden, hidden),
                to_v=h3_linear(hidden, hidden),
                to_out=h3_linear(hidden, hidden),
            ),
            ff=SimpleNamespace(
                in_proj=h3_linear(hidden, 4 * hidden),
                out_proj=h3_linear(2 * hidden, hidden, lora_rank=2),
            ),
        )
        pre_infer_out = SimpleNamespace(
            temb=torch.empty(4, hidden),
            sequence_parallel_state=SimpleNamespace(aux_length=2, main_shard_length=3),
        )
        profile = MiniMaxH3BlockProfile(
            {
                "attention_head_dim": 2,
                "attn_type": "dynamic_sparse_attn",
            },
            num_heads=4,
            seq_p_size=2,
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "inventory.json"
            profile.bind(block, torch.empty(5, hidden), pre_infer_out)
            profile.write_inventory(path, block_idx=7)
            inventory = json.loads(path.read_text())

            profile.bind(block, torch.empty(5, hidden), pre_infer_out, include_adaln=False)
            profile.write_inventory(path, block_idx=7)
            cached_inventory = json.loads(path.read_text())

        by_tag = {operation["tag"]: operation for operation in inventory["linear_operations"]}
        self.assertEqual(by_tag["adaln"]["shape_mnk"], [4, 6 * hidden, hidden])
        self.assertEqual(by_tag["attn_q"]["shape_mnk"], [5, hidden, hidden])
        self.assertEqual(by_tag["ffn_in_fused"]["shape_mnk"], [5, 4 * hidden, hidden])
        self.assertEqual(by_tag["ffn_out"]["lora_flops"], 2 * 5 * 2 * (hidden + 2 * hidden))
        self.assertEqual(inventory["attention"]["shape_bhsd"], [1, 2, 8, 2])
        self.assertEqual(inventory["attention"]["flops_semantics"], "dense-equivalent")
        self.assertNotIn("adaln", {operation["tag"] for operation in cached_inventory["linear_operations"]})

    def test_wan_packed_fp4_restores_logical_k(self):
        hidden = 8

        def packed_linear(out_features, in_features):
            return linear(out_features, in_features // 2, dtype=torch.uint8)

        block = SimpleNamespace(
            compute_phases=[
                SimpleNamespace(
                    self_attn_q=packed_linear(hidden, hidden),
                    self_attn_k=packed_linear(hidden, hidden),
                    self_attn_v=packed_linear(hidden, hidden),
                    self_attn_o=packed_linear(hidden, hidden),
                ),
                SimpleNamespace(
                    cross_attn_q=packed_linear(hidden, hidden),
                    cross_attn_k=packed_linear(hidden, hidden),
                    cross_attn_v=packed_linear(hidden, hidden),
                    cross_attn_o=packed_linear(hidden, hidden),
                ),
                SimpleNamespace(
                    ffn_0=packed_linear(16, hidden),
                    ffn_2=packed_linear(hidden, 16),
                ),
            ]
        )
        profile = WanBlockProfile(
            {
                "num_heads": 2,
                "dim": hidden,
                "task": "t2v",
                "dit_quant_scheme": "nvfp4",
            }
        )
        profile.bind(
            block,
            torch.empty(3, hidden),
            SimpleNamespace(context=torch.empty(5, hidden)),
        )

        self.assertEqual(profile._gemms["self_q"].k, hidden)
        self.assertEqual(profile._gemms["ffn_2"].k, 16)


if __name__ == "__main__":
    unittest.main()
