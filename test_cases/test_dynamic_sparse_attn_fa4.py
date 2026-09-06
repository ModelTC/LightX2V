import pytest
import torch

from lightx2v.common.ops.attn import dynamic_sparse_attn
from lightx2v.common.ops.attn.utils.sla_util import (
    block_lut_to_ordinal_metadata,
    centered_mean_pool,
    get_block_lut,
    get_block_map,
    mean_pool,
)
from lightx2v.common.ops.attn.utils.sla_util_blhd import (
    centered_mean_pool as centered_mean_pool_blhd,
)
from lightx2v.common.ops.attn.utils.sla_util_blhd import (
    get_block_lut_blhd,
)
from lightx2v.common.ops.attn.utils.sparge_util import block_map_ordinal_lut_triton


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_cutedsl_builds_lut_and_preserves_input_dtype(monkeypatch, dtype):
    q = torch.randn((256, 2, 128), dtype=dtype)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    expected_lut = torch.zeros((1, 2, 1, 1), dtype=torch.int64)

    def fake_get_block_lut_blhd(qt, kt, **kwargs):
        assert qt.shape == (1, 256, 2, 128)
        assert kt.shape == (1, 256, 2, 128)
        assert qt.dtype == dtype
        assert kt.dtype == dtype
        assert kwargs["BLKQ"] == 256
        assert kwargs["BLKK"] == 128
        return expected_lut, 1, 2

    def fake_cutedsl_sparse_fmha(qt, kt, vt, cu_seqlens, max_seqlen, block_count, block_indices):
        assert qt.dtype == dtype
        assert kt.dtype == dtype
        assert vt.dtype == dtype
        assert qt.is_contiguous() and kt.is_contiguous() and vt.is_contiguous()
        assert cu_seqlens.tolist() == [0, 256]
        assert max_seqlen == 256
        torch.testing.assert_close(block_count, torch.ones((1, 2, 1), dtype=torch.int32))
        assert block_indices.shape == (1, 2, 1, 2)
        return torch.zeros_like(qt)

    monkeypatch.setattr(dynamic_sparse_attn, "get_block_lut_blhd", fake_get_block_lut_blhd)
    attn = object.__new__(dynamic_sparse_attn.DynamicSparseAttnWeight)
    attn.BLKQ, attn.BLKK = 256, 128
    attn.topk = 0.5
    attn.cutedsl_sparse_fmha = fake_cutedsl_sparse_fmha

    out = attn.apply_cutedsl(q, k, v)

    assert out.shape == (256, 2 * 128)
    assert out.dtype == dtype


def test_block_lut_to_ordinal_metadata_sorts_and_pads():
    lut = torch.tensor([[[[5, 1, 3], [4, 0, 2]]]])

    full_block_idx, full_block_cnt = block_lut_to_ordinal_metadata(lut, num_k_blocks=6)

    expected_idx = torch.tensor([[[[1, 3, 5, 0, 0, 0], [0, 2, 4, 0, 0, 0]]]], dtype=torch.int32)
    expected_cnt = torch.tensor([[[3, 3]]], dtype=torch.int32)
    torch.testing.assert_close(full_block_idx, expected_idx, atol=0, rtol=0)
    torch.testing.assert_close(full_block_cnt, expected_cnt, atol=0, rtol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Triton mean pooling requires CUDA")
def test_fa4_lut_matches_existing_map_path_for_gqa_and_partial_blocks():
    torch.manual_seed(42)
    q = torch.randn((1, 4, 257, 64), device="cuda", dtype=torch.bfloat16)
    k = torch.randn((1, 2, 259, 64), device="cuda", dtype=torch.bfloat16)

    k_mean = torch.mean(k, dim=-2, keepdim=True)
    old_pooled_k = mean_pool(k - k_mean, 128)
    new_pooled_k = centered_mean_pool(k, k_mean, 128)
    torch.testing.assert_close(new_pooled_k, old_pooled_k, atol=0, rtol=0)

    sparse_map, old_lut, old_topk = get_block_map(q, k, topk_ratio=0.67, BLKQ=128, BLKK=128)
    new_lut, new_topk, num_k_blocks = get_block_lut(q, k, topk_ratio=0.67, BLKQ=128, BLKK=128)

    torch.testing.assert_close(
        torch.sort(new_lut, dim=-1).values,
        torch.sort(old_lut, dim=-1).values,
        atol=0,
        rtol=0,
    )
    assert new_topk == old_topk
    assert num_k_blocks == 3

    old_idx, old_cnt = block_map_ordinal_lut_triton(sparse_map)
    new_idx, new_cnt = block_lut_to_ordinal_metadata(new_lut, num_k_blocks)
    torch.testing.assert_close(new_idx, old_idx, atol=0, rtol=0)
    torch.testing.assert_close(new_cnt, old_cnt, atol=0, rtol=0)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("block_q,block_k", [(128, 128), (256, 128)])
@pytest.mark.skipif(not torch.cuda.is_available(), reason="Triton mean pooling requires CUDA")
def test_blhd_lut_matches_bhld_without_full_tensor_clones(dtype, block_q, block_k):
    torch.manual_seed(123)
    q = torch.randn((1, 257, 4, 64), device="cuda", dtype=dtype)
    k = torch.randn((1, 259, 2, 64), device="cuda", dtype=dtype)

    q_bhld = q.transpose(1, 2).contiguous()
    k_bhld = k.transpose(1, 2).contiguous()
    expected_lut, expected_topk, expected_k_blocks = get_block_lut(q_bhld, k_bhld, topk_ratio=0.67, BLKQ=block_q, BLKK=block_k)
    actual_lut, actual_topk, actual_k_blocks = get_block_lut_blhd(q, k, topk_ratio=0.67, BLKQ=block_q, BLKK=block_k)

    torch.testing.assert_close(
        torch.sort(actual_lut, dim=-1).values,
        torch.sort(expected_lut, dim=-1).values,
        atol=0,
        rtol=0,
    )
    assert actual_topk == expected_topk
    assert actual_k_blocks == expected_k_blocks

    k_center_blhd = torch.mean(k, dim=1, keepdim=True)
    pooled_k_blhd = centered_mean_pool_blhd(k, k_center_blhd, block_k)
    k_center_bhld = torch.mean(k_bhld, dim=-2, keepdim=True)
    pooled_k_bhld = centered_mean_pool(k_bhld, k_center_bhld, block_k)
    torch.testing.assert_close(pooled_k_blhd, pooled_k_bhld, atol=0, rtol=0)
