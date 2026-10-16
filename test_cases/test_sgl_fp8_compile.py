import pytest
import torch

from lightx2v.common.ops.mm.sgl_kernel import sgl_fp8_per_token_quant

sgl_kernel = pytest.importorskip("sgl_kernel")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="SGL FP8 requires CUDA")


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("shape", [(17, 256), (2640, 3072)])
def test_sgl_fp8_quant_fullgraph_matches_native(shape, dtype):
    x = torch.randn(shape, device="cuda", dtype=dtype)
    original = x.clone()
    expected_q = torch.empty(shape, device=x.device, dtype=torch.float8_e4m3fn)
    expected_scale = torch.empty((shape[0], 1), device=x.device, dtype=torch.float32)
    sgl_kernel.sgl_per_token_quant_fp8(x, expected_q, expected_scale)
    compiled = torch.compile(sgl_fp8_per_token_quant, fullgraph=True)
    for quantize in [sgl_fp8_per_token_quant, compiled]:
        q, scale = quantize(x)
        torch.testing.assert_close(q.view(torch.uint8), expected_q.view(torch.uint8), rtol=0, atol=0)
        torch.testing.assert_close(scale, expected_scale, rtol=0, atol=0)
        torch.testing.assert_close(x, original, rtol=0, atol=0)


def test_sgl_fp8_linear_fullgraph_matches_eager():
    from lightx2v.common.ops.mm.mm_weight import MMWeightWfp8channelAfp8channeldynamicSgl

    weight = MMWeightWfp8channelAfp8channeldynamicSgl("weight", None)
    weight.weight = torch.randn(128, 256, device="cuda", dtype=torch.bfloat16).to(torch.float8_e4m3fn).t()
    weight.weight_scale = torch.ones(128, 1, device="cuda", dtype=torch.float32)
    x = torch.randn(17, 256, device="cuda", dtype=torch.bfloat16)
    with torch.no_grad():
        expected = weight.apply(x)
        actual = torch.compile(weight.apply, fullgraph=True)(x)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
