"""SLA AR forward obeys selected blocks for rectangular and partial windows."""

import pytest
import torch
import torch.nn.functional as F

from lightx2v.common.ops.attn.kernels.sla_kernel_ar import _attention_ar
from lightx2v.common.ops.attn.utils.sla_util import get_block_map, mean_pool

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.mark.parametrize("shape,dtype", [((1, 129, 257, 3, 64), torch.bfloat16), ((2, 257, 389, 2, 128), torch.float16), ((1, 2640, 15840, 2, 128), torch.bfloat16)])
def test_all_selected_blocks_match_torch_sdpa(shape, dtype):
    batch, nq, nk, heads, dim = shape
    torch.manual_seed(20261016)
    q = torch.randn(batch, nq, heads, dim, device="cuda", dtype=dtype)
    k = torch.randn(batch, nk, heads, dim, device="cuda", dtype=dtype)
    v = torch.randn_like(k)
    qb, kb = (nq + 127) // 128, (nk + 127) // 128
    mask = torch.ones(batch, heads, qb, kb, device="cuda", dtype=torch.int8)
    lut = torch.arange(kb, device="cuda").view(1, 1, 1, kb).expand(batch, heads, qb, kb).contiguous()
    actual = _attention_ar.apply(q, k, v, mask, lut, kb, 128, 128)
    expected = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)).transpose(1, 2)
    assert actual.shape == q.shape and torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, atol=0.003 if dtype == torch.bfloat16 else 0.0005, rtol=0.02)


@pytest.mark.parametrize("nq,nk", [(129, 257), (2640, 15840)])
def test_selected_first_and_partial_last_key_blocks_match_sdpa(nq, nk):
    torch.manual_seed(42)
    q = torch.randn(1, nq, 2, 128, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(1, nk, 2, 128, device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    qb, kb = (nq + 127) // 128, (nk + 127) // 128
    lut = torch.tensor([0, kb - 1], device="cuda").view(1, 1, 1, 2).expand(1, 2, qb, 2).contiguous()
    mask = torch.zeros(1, 2, qb, kb, device="cuda", dtype=torch.int8).scatter_(-1, lut, 1)
    actual = _attention_ar.apply(q, k, v, mask, lut, 2, 128, 128)
    indices = torch.cat((torch.arange(128), torch.arange((kb - 1) * 128, nk))).to("cuda")
    expected = F.scaled_dot_product_attention(q.transpose(1, 2), k[:, indices].transpose(1, 2), v[:, indices].transpose(1, 2)).transpose(1, 2)
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, atol=0.004, rtol=0.02)


def test_router_pools_partial_blocks_and_keeps_one_key_for_short_sequences():
    q = torch.full((1, 2, 65, 128), 2.0, device="cuda", dtype=torch.bfloat16)
    k = torch.full((1, 2, 129, 128), 3.0, device="cuda", dtype=torch.bfloat16)
    assert torch.equal(mean_pool(q, 128), torch.full((1, 2, 1, 128), 2.0, device="cuda", dtype=torch.bfloat16))
    assert torch.equal(mean_pool(k, 128), torch.full((1, 2, 2, 128), 3.0, device="cuda", dtype=torch.bfloat16))
    mask, lut, topk = get_block_map(q, k, 0.2, BLKQ=128, BLKK=128)
    assert topk == 1 and lut.shape == (1, 2, 1, 1)
    assert torch.all(mask.sum(-1) == 1)


def test_d128_rectangular_kernel_compiles_fullgraph():
    q = torch.randn(1, 129, 2, 128, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(1, 257, 2, 128, device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    mask = torch.ones(1, 2, 2, 3, device="cuda", dtype=torch.int8)
    lut = torch.arange(3, device="cuda").view(1, 1, 1, 3).expand(1, 2, 2, 3).contiguous()

    def attention(q, k, v, mask, lut):
        return _attention_ar.apply(q, k, v, mask, lut, 3, 128, 128)

    compiled = torch.compile(attention, fullgraph=True, dynamic=False)
    with torch.no_grad():
        actual = compiled(q, k, v, mask, lut)
    expected = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)).transpose(1, 2)
    torch.testing.assert_close(actual, expected, atol=0.003, rtol=0.02)


def test_rejects_layout_and_empty_selection_before_launch():
    q = torch.zeros(1, 129, 2, 64, device="cuda", dtype=torch.bfloat16)
    k = torch.zeros(1, 257, 2, 64, device="cuda", dtype=torch.bfloat16)
    mask = torch.ones(1, 2, 2, 3, device="cuda", dtype=torch.int8)
    lut = torch.zeros(1, 2, 2, 1, device="cuda", dtype=torch.long)
    with pytest.raises(ValueError, match="at least one"):
        _attention_ar.apply(q, k, k, mask, lut[..., :0], 0, 128, 128)
    with pytest.raises(ValueError, match="contiguous"):
        _attention_ar.apply(q[:, ::2], k, k, mask, lut, 1, 128, 128)
    with pytest.raises(ValueError, match="LUT shape"):
        _attention_ar.apply(q, k, k, mask, lut[:, :, :1].contiguous(), 1, 128, 128)
