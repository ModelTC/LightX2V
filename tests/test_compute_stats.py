"""Pure compute contracts: independent token-area oracles, tails, layouts, and the actual SLA router."""

import json

import pytest
import torch

from lightx2v.common.ops.attn.template import AttnWeightTemplate
from lightx2v.common.ops.compute_stats import range_token_pairs, selected_token_pairs
from tools.compute_stats import attention_flops, gemm_flops


def test_math_known_macs_and_int64_area():
    assert gemm_flops(3, 5, 7, 2) == 420
    assert attention_flops(13, 4, 7) == 286
    assert selected_token_pairs(torch.tensor([[1, 0], [1, 1]]), torch.tensor([2, 1]), torch.tensor([3, 2])) == 11
    ranges = torch.tensor([[0, 50000]], dtype=torch.int32)
    assert range_token_pairs(ranges, ranges) == 2500000000


@pytest.mark.parametrize("layout", ["SHD", "BHSD", "BSHD"])
@pytest.mark.parametrize("sparsity,count", [(0.0, 3), (0.2, 2), (0.5, 1), (0.8, 0), (1.0, 0)])
@pytest.mark.parametrize("minimum", [0, 1])
def test_shared_sparsity_helper_shape_only_and_independent_token_pair_oracle(layout, sparsity, count, minimum):
    batch, heads, sq, sk = (1 if layout == "SHD" else 2), 2, 5, 7
    q, k, v = torch.empty(batch, heads, sq, 3, device="meta"), torch.empty(batch, heads, sk, 3, device="meta"), torch.empty(batch, heads, sk, 2, device="meta")
    if layout == "SHD":
        inputs = [x[0].transpose(0, 1) for x in (q, k, v)]
    elif layout == "BSHD":
        inputs = [x.transpose(1, 2) for x in (q, k, v)]
    else:
        inputs = [x for x in (q, k, v)]
    basic = AttnWeightTemplate.compute_sparse_stats_from_sparsity(*inputs, sparsity, 2, 3, min_blocks=minimum, layout=layout)
    retained = max(minimum, count)
    mask = torch.zeros(batch, heads, 3, 3, dtype=torch.bool)
    for b in range(batch):
        for h in range(heads):
            for row in range(3):
                for offset in range(retained):
                    mask[b, h, row, (row + h + offset) % 3] = True
    pairs = sum(int(mask[b, h, i // 2, j // 3]) for b in range(batch) for h in range(heads) for i in range(sq) for j in range(sk))
    exact = AttnWeightTemplate.compute_sparse_stats(q, k, v, mask, 2, 3)
    assert basic["sparse"]["tiles_per_query_block"] == retained
    assert basic["work"]["dense_equivalent_flops"] == exact["work"]["dense_equivalent_flops"] == batch * heads * sq * sk * 10
    assert basic["work"]["scheduled_tile_flops"] == exact["work"]["scheduled_tile_flops"] == int(mask.sum()) * 2 * 3 * 10
    assert exact["work"]["selected_logical_flops"] == pairs * 10
    assert basic["work"]["selected_logical_flops"] == (pairs * 10 if retained in (0, 3) else None)


@pytest.mark.parametrize("sparsity", [0.8, 0.9, 0.7])
@pytest.mark.parametrize("kv_heads", [1, 2])
def test_shared_sparsity_matches_actual_router_float_rounding(monkeypatch, sparsity, kv_heads):
    from math import ceil

    from lightx2v.common.ops.attn.utils import sla_util

    def cpu_pool(x, block, center=None):
        return torch.zeros(x.shape[0], x.shape[1], ceil(x.shape[-2] / block), x.shape[-1])

    monkeypatch.setattr(sla_util, "mean_pool", cpu_pool)
    q, k = torch.zeros(1, 2, 5, 4), torch.zeros(1, kv_heads, 20, 4)
    mask, _, actual_topk = sla_util.get_block_map(q, k, 1 - sparsity, BLKQ=2, BLKK=2)
    stats = AttnWeightTemplate.compute_sparse_stats_from_sparsity(q, k, k, sparsity, 2, 2, min_blocks=1)
    actual = AttnWeightTemplate.compute_sparse_stats(q, k, k, mask, 2, 2)
    assert stats["sparse"]["tiles_per_query_block"] == actual_topk
    assert actual["work"]["selected_logical_flops"] == stats["work"]["selected_logical_flops"]
    assert actual["work"]["scheduled_tile_flops"] == stats["work"]["scheduled_tile_flops"]
    assert stats["work"]["scheduled_tile_flops"] == int(mask.sum()) * 2 * 2 * 16


def test_shared_variable_mask_uses_actual_sizes_and_leaves_kernel_padding_unknown():
    q, k, v = torch.empty(2, 2, 5, 4), torch.empty(2, 2, 7, 4), torch.empty(2, 2, 7, 3)
    q_sizes = torch.tensor([[[2, 0, 3], [1, 3, 1]], [[4, 1, 0], [0, 2, 3]]], dtype=torch.int32)
    k_sizes = torch.tensor([[[1, 2, 4], [4, 0, 3]], [[2, 2, 3], [1, 5, 1]]], dtype=torch.int32)
    mask = torch.arange(36).reshape(2, 2, 3, 3) % 3 != 1
    actual_pairs = 0
    for b in range(2):
        for h in range(2):
            token_mask = mask[b, h].repeat_interleave(q_sizes[b, h].long(), dim=0).repeat_interleave(k_sizes[b, h].long(), dim=1)
            assert token_mask.shape == (5, 7)
            actual_pairs += int(token_mask.sum())
    result = AttnWeightTemplate.compute_sparse_stats(q, k, v, mask, q_block_sizes=q_sizes, k_block_sizes=k_sizes)
    assert result["work"]["selected_logical_flops"] == actual_pairs * 14
    assert result["work"]["dense_equivalent_flops"] == 2 * 2 * 5 * 7 * 14
    assert result["work"]["scheduled_tile_flops"] is None
    assert result["routing"]["q_block_sizes"] == q_sizes.tolist()
    assert result["routing"]["block_q"] is None and result["routing"]["block_k"] is None
    assert json.loads(json.dumps(result)) == result


@pytest.mark.parametrize("kv_heads", [1, 2])
def test_shared_mask_token_layout_and_gqa(kv_heads):
    q, k, v = torch.empty(3, 2, 4), torch.empty(5, 2, 4), torch.empty(5, 2, 3)
    mask = torch.zeros(1, 2, 3, 5)
    mask[:, :, 1, 2] = 1
    assert AttnWeightTemplate.compute_sparse_stats(q, k, v, mask, layout="SHD")["work"]["selected_logical_flops"] == 2 * 14
    q, k = torch.empty(1, 4, 3, 4), torch.empty(1, kv_heads, 5, 4)
    facts = AttnWeightTemplate.compute_sparse_stats(q, k, k, torch.ones(1, 4, 3, 5))
    assert facts["shape"]["heads"] == 4 and facts["shape"]["kv_heads"] == kv_heads
    assert facts["work"]["selected_logical_flops"] == facts["work"]["dense_equivalent_flops"] == 960
