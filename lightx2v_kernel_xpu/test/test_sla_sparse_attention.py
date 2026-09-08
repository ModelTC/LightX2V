import math

import pytest
import torch

from sycl_kernels.sla import (
    sla_block_map,
    sparse_block_attention,
    sparse_block_attention_reference,
)


def _explicit_router(q, k, keep_ratio, block_q, block_k):
    # Small test-only implementation matching LightX2V's original BHLD SLA.
    q_bhld = q.permute(0, 2, 1, 3)
    k_bhld = k.permute(0, 2, 1, 3)
    k_bhld = k_bhld - k_bhld.mean(dim=2, keepdim=True)

    def pool(x, block):
        chunks = [x[:, :, start : start + block].mean(dim=2) for start in range(0, x.shape[2], block)]
        return torch.stack(chunks, dim=2)

    pq, pk = pool(q_bhld, block_q), pool(k_bhld, block_k)
    if pq.shape[1] != pk.shape[1]:
        pk = pk.repeat_interleave(pq.shape[1] // pk.shape[1], dim=1)
    scores = pq @ pk.transpose(-1, -2)
    topk = max(1, min(scores.shape[-1], int(keep_ratio * scores.shape[-1])))
    return torch.topk(scores, topk, dim=-1, sorted=False).indices


@pytest.mark.parametrize("length", [17, 32, 35])
def test_sla_router_matches_original_semantics(length):
    torch.manual_seed(3)
    q = torch.randn(1, length, 4, 8)
    k = torch.randn(1, length, 2, 8)
    actual = sla_block_map(q, k, keep_ratio=0.5, block_q=8, block_k=8)
    expected = _explicit_router(q, k, 0.5, 8, 8)
    # topk(sorted=False) order is not contractual; compare selected sets.
    torch.testing.assert_close(actual.sort(dim=-1).values.long(), expected.sort(dim=-1).values)


def test_full_lut_reference_matches_dense_gqa():
    torch.manual_seed(4)
    q = torch.randn(1, 19, 4, 16, dtype=torch.bfloat16)
    k = torch.randn(1, 19, 2, 16, dtype=torch.bfloat16)
    v = torch.randn_like(k)
    block = 8
    blocks = math.ceil(q.shape[1] / block)
    lut = torch.arange(blocks, dtype=torch.int32).view(1, 1, 1, blocks).expand(1, 4, blocks, blocks)
    actual = sparse_block_attention_reference(q, k, v, lut, block, block)
    expected = torch.nn.functional.scaled_dot_product_attention(
        q.transpose(1, 2),
        k.repeat_interleave(2, dim=2).transpose(1, 2),
        v.repeat_interleave(2, dim=2).transpose(1, 2),
    ).transpose(1, 2)
    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)


@pytest.mark.skipif(not torch.xpu.is_available(), reason="XPU is unavailable")
def test_xpu_sparse_kernel_matches_reference():
    torch.manual_seed(5)
    q = torch.randn(1, 257, 4, 128, device="xpu", dtype=torch.bfloat16)
    k = torch.randn(1, 257, 2, 128, device="xpu", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    lut = sla_block_map(q, k, keep_ratio=0.5, block_q=128, block_k=128)
    actual = sparse_block_attention(q, k, v, lut, 128, 128)
    expected = sparse_block_attention_reference(q, k, v, lut, 128, 128)
    torch.xpu.synchronize()
    torch.testing.assert_close(actual, expected, rtol=3e-2, atol=3e-2)
