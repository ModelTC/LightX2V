import pytest
import torch

from lightx2v.common.ops.attn.sage_sm90_fused import SageAttn2Sm90FusedWeight, make_transposed_v, update_transposed_v

sageattention = pytest.importorskip("sageattention")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0), reason="SM90 required")


@pytest.mark.parametrize("shape", [(2, 257, 4, 64), (1, 2640, 24, 128)])
def test_transposed_v_partial_updates_match_native(shape):
    from sageattention import _fused

    v = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    workspace = make_transposed_v(v[:, :0], shape[1])
    for start, end in [(0, 33), (33, shape[1])]:
        update_transposed_v(v[:, start:end].contiguous(), workspace, start)
    padded_v = torch.nn.functional.pad(v, (0, 0, 0, 0, 0, workspace.shape[-1] - shape[1]))
    expected = torch.empty_like(workspace)
    _fused.transpose_pad_permute_cuda(padded_v, expected, 0)
    torch.testing.assert_close(workspace, expected, rtol=0, atol=0)


@pytest.mark.parametrize("shape", [(1, 64, 257, 4, 64), (1, 2640, 7920, 24, 128), (2, 65, 512, 4, 128)])
def test_fused_attention_matches_native(shape):
    batch, queries, keys, heads, dim = shape
    torch.manual_seed(42)
    q = torch.randn(batch, queries, heads, dim, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(batch, keys, heads, dim, device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    expected = sageattention.sageattn(q, k, v, tensor_layout="NHD").reshape(batch * queries, heads * dim)
    weight = SageAttn2Sm90FusedWeight()
    torch.testing.assert_close(weight.apply(q, k, v), expected, rtol=0, atol=0)
    compiled = torch.compile(weight.apply, fullgraph=True)
    torch.testing.assert_close(compiled(q, k, v), expected, rtol=0, atol=0)
