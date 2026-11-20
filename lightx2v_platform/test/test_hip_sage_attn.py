"""
hip_sage_amd_rocm smoke test (AMD gfx1201 only):
PYTHONPATH=/path-to-LightX2V python lightx2v_platform/test/test_hip_sage_attn.py
"""

import pytest
import torch

from lightx2v_platform.ops.attn.amd_rocm import hip_sage

pytestmark = pytest.mark.skipif(not hip_sage.is_supported_device(), reason=f"requires an AMD {hip_sage.ARCH} GPU")


def _sdpa(q, k, v):
    q, k, v = (t.transpose(0, 1).float() for t in (q, k, v))
    return torch.nn.functional.scaled_dot_product_attention(q, k, v).transpose(0, 1)


def test_precision_policy():
    from lightx2v_platform.ops.attn.amd_rocm.hip_sage_attn import AmdHipSageAttnWeight

    w = AmdHipSageAttnWeight()
    assert [w.bits_for(b) for b in (None, 0, 44, 45, 49)] == [8] * 5
    w.set_config({"model": "minimax_h3", "mixed_precision": True})
    assert [w.bits_for(b) for b in (None, 0, 44, 45, 49)] == [8, 4, 4, 8, 8]
    with pytest.raises(ValueError):
        AmdHipSageAttnWeight().set_config({"model": "other", "mixed_precision": True})


@pytest.mark.parametrize("bits,min_cos", [(4, 0.97), (8, 0.998)])
def test_attention_matches_sdpa(bits, min_cos):
    torch.manual_seed(0)
    q, k, v = (torch.randn(4100, 6, 128, device="cuda", dtype=torch.bfloat16) for _ in range(3))
    out = hip_sage.attention(q, k, v, None, bits)
    cos = torch.nn.functional.cosine_similarity(out.float().flatten(), _sdpa(q, k, v).flatten(), 0).item()
    assert out.shape == q.shape and cos > min_cos, f"int{bits}: cos {cos:.5f}"


if __name__ == "__main__":
    test_precision_policy()
    for bits, min_cos in ((4, 0.97), (8, 0.998)):
        test_attention_matches_sdpa(bits, min_cos)
    print("hip_sage_amd_rocm: OK")
