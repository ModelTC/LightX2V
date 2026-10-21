import pytest
import torch
from reconstruction_imports import cpu_imports

with cpu_imports():
    import lightx2v.common.ops.attn.torch_sdpa  # noqa: F401
    from lightx2v.common.ops.inference_module import InferenceAttention


@pytest.mark.parametrize("batch", [1, 2])
def test_attention_layout_and_cross_attention(batch):
    torch.manual_seed(3)
    q = torch.randn(batch, 3, 7, 8, dtype=torch.bfloat16)
    k = torch.randn(batch, 3, 11, 8, dtype=torch.bfloat16)
    v = torch.randn_like(k)
    actual = InferenceAttention("torch_sdpa")(q, k, v)
    expected = torch.nn.functional.scaled_dot_product_attention(q, k, v)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
