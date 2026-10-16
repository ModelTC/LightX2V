import pytest
import torch

from lightx2v.common.ops.attn.sage_attn import SageAttn2Weight

sageattention = pytest.importorskip("sageattention")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="SageAttention requires CUDA")


@pytest.mark.parametrize("shape", [(1, 32, 64, 4, 64), (1, 2640, 512, 24, 128), (2, 64, 128, 4, 64)])
def test_sage_attention_fullgraph_matches_native(shape):
    batch, queries, keys, heads, dim = shape
    q = torch.randn(batch, queries, heads, dim, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(batch, keys, heads, dim, device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    originals = [tensor.clone() for tensor in [q, k, v]]
    expected = sageattention.sageattn(q, k, v, tensor_layout="NHD").reshape(batch * queries, heads * dim)
    weight = SageAttn2Weight()
    compiled = torch.compile(weight.apply, fullgraph=True)
    for attention in [weight.apply, compiled]:
        actual = attention(q, k, v)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        for original, tensor in zip(originals, [q, k, v]):
            torch.testing.assert_close(tensor, original, rtol=0, atol=0)
