import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from lightx2v.models.networks.hunyuan3d.infer.block_profile import Hunyuan3DBlockProfile
from lightx2v.models.networks.wan.infer.block_profile import WanBlockProfile
from lightx2v.utils import op_shape_trace as ost


def linear(out_features, in_features, *, dtype=torch.bfloat16):
    weight = torch.empty(out_features, in_features, dtype=dtype)
    return SimpleNamespace(_get_actual_weight=lambda: weight)


def attention(hidden):
    return SimpleNamespace(
        to_q=linear(hidden, hidden),
        to_k=linear(hidden, hidden),
        to_v=linear(hidden, hidden),
        out_proj=linear(hidden, hidden),
    )


class TransformerBlockProfileShapeTest(unittest.TestCase):
    def test_hunyuan_batch_and_moe_shapes(self):
        hidden = 8
        profile = Hunyuan3DBlockProfile(
            {
                "num_heads": 2,
                "hidden_size": hidden,
                "moe_top_k": 2,
            }
        )
        expert = SimpleNamespace(fc1=linear(16, hidden))
        block = SimpleNamespace(
            skip_linear=None,
            attn1=attention(hidden),
            attn2=attention(hidden),
            moe=SimpleNamespace(
                gate=linear(4, hidden),
                experts=[expert],
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
