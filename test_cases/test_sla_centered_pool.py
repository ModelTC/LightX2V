"""SLA centering fusion preserves pooled keys and the selected block mask."""

import pytest
import torch

from lightx2v.common.ops.attn.utils.sla_util import get_block_map, mean_pool

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def _assert_same_bits(actual, expected):
    assert actual.shape == expected.shape and actual.dtype == expected.dtype
    assert torch.equal(actual.view(torch.uint8), expected.view(torch.uint8))


def _keys(shape, dtype, distribution):
    torch.manual_seed(20261016)
    x = torch.randn(shape, device="cuda", dtype=dtype)
    if distribution == "large_bias":
        x = (x * 8 + 1000).to(dtype)
    elif distribution == "alternating":
        x[..., ::2, :] *= 64
        x[..., 1::2, :] *= 0.03125
    return x


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("block", [64, 128])
@pytest.mark.parametrize("length", [1, 63, 64, 65, 127, 128, 129])
def test_centered_pool_preserves_partial_blocks_and_subtraction_rounding(dtype, block, length):
    k = _keys((2, 3, length, 128), dtype, "alternating")
    center = k.mean(dim=-2, keepdim=True)
    expected = mean_pool(k - center, block)
    _assert_same_bits(mean_pool(k, block, center), expected)


@pytest.mark.parametrize("distribution", ["normal", "large_bias", "alternating"])
@pytest.mark.parametrize("block", [64, 128])
def test_full_720p_window_pooled_keys_and_final_router_mask_are_exact(distribution, block):
    k = _keys((1, 24, 15840, 128), torch.bfloat16, distribution)
    q = torch.randn(1, 24, 2640, 128, device="cuda", dtype=torch.bfloat16)
    center = k.mean(dim=-2, keepdim=True)
    _assert_same_bits(mean_pool(k, block, center), mean_pool(k - center, block))
    reference = get_block_map(q, k, 0.2, BLKQ=block, BLKK=block, fuse_centered_pool=False)
    actual = get_block_map(q, k, 0.2, BLKQ=block, BLKK=block)
    assert actual[2] == reference[2]
    assert torch.equal(actual[0], reference[0])
    assert torch.equal(actual[1], reference[1])


def test_grouped_query_attention_repeats_identical_pooled_keys():
    k = _keys((2, 2, 257, 64), torch.bfloat16, "normal")
    q = torch.randn(2, 6, 129, 64, device="cuda", dtype=torch.bfloat16)
    reference = get_block_map(q, k, 0.2, fuse_centered_pool=False)
    actual = get_block_map(q, k, 0.2)
    assert actual[2] == reference[2]
    assert torch.equal(actual[0], reference[0])
    assert torch.equal(actual[1], reference[1])


def test_centered_pool_and_router_are_cuda_graph_safe():
    k = _keys((1, 2, 257, 128), torch.bfloat16, "normal")
    q = torch.randn(1, 2, 129, 128, device="cuda", dtype=torch.bfloat16)
    reference = get_block_map(q, k, 0.2, fuse_centered_pool=False)
    for _ in range(3):
        get_block_map(q, k, 0.2)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        mask, lut, topk = get_block_map(q, k, 0.2)
    graph.replay()
    assert torch.equal(mask, reference[0]) and torch.equal(lut, reference[1]) and topk == reference[2]
    k.add_(0.125)
    reference = get_block_map(q, k, 0.2, fuse_centered_pool=False)
    graph.replay()
    assert torch.equal(mask, reference[0]) and torch.equal(lut, reference[1])
