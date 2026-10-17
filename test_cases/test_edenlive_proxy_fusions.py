"""ProxyDINO exact RoPE and optional approximate final normalization."""

import pytest
import torch
import torch.nn.functional as F
from torch._subclasses.fake_tensor import FakeTensorMode

from lightx2v.models.input_encoders.edenlive.proxy_dino import ProxyDINO
from lightx2v.models.input_encoders.edenlive.proxy_fusions import proxy_normalize, proxy_rope_qk

cuda_only = pytest.mark.skipif(not torch.cuda.is_available(), reason="Proxy fusions require CUDA")


def _angles(height, width, head_dim=64):
    # Use the encoder's real frequency/coordinate construction.
    encoder = ProxyDINO.__new__(ProxyDINO)
    encoder.config = {"rope_theta": 100.0}
    encoder.head_dim = head_dim
    encoder._rope_cache = {}
    return encoder._rope(height, width, torch.device("cuda"))


def _rope_reference(q, k, cos, sin, prefix=5):
    outputs = (q.clone(), k.clone())
    for tensor in outputs:
        patches = tensor[:, prefix:].float()
        first, second = patches.chunk(2, dim=-1)
        tensor[:, prefix:] = (patches * cos + torch.cat((-second, first), dim=-1) * sin).to(tensor.dtype)
    return outputs


def _qk(batch, sequence, heads, width, layout, dtype):
    if layout == "contiguous":
        return tuple(torch.randn(batch, sequence, heads, width, device="cuda", dtype=dtype) for _ in range(2))
    if layout == "packed":
        packed = torch.randn(batch, sequence, 3 * heads * width, device="cuda", dtype=dtype)
        q, k, _ = packed.split(heads * width, dim=-1)
        return q.view(batch, sequence, heads, width), k.view(batch, sequence, heads, width)
    storage = torch.randn(batch, sequence + 2, heads, width * 2, device="cuda", dtype=dtype)
    other = torch.randn_like(storage)
    return storage[:, 1:-1, :, ::2], other[:, 1:-1, :, ::2]


@cuda_only
@pytest.mark.parametrize("batch", [1, 3])
@pytest.mark.parametrize("height,width", [(2, 3), (25, 35), (30, 52)])
@pytest.mark.parametrize("layout", ["contiguous", "packed", "strided"])
def test_rope_is_bit_exact_for_real_angles_and_streaming_shapes(batch, height, width, layout):
    torch.manual_seed(123)
    sequence = 5 + height * width
    q, k = _qk(batch, sequence, 16, 64, layout, torch.bfloat16)
    original_q, original_k = q.clone(), k.clone()
    cos, sin = _angles(height, width)
    actual = proxy_rope_qk(q, k, cos, sin, 5)
    expected = _rope_reference(q, k, cos, sin, 5)
    for output, reference in zip(actual, expected):
        assert torch.equal(output.view(torch.int16), reference.view(torch.int16))
        assert output.is_contiguous()
    assert torch.equal(q, original_q) and torch.equal(k, original_k)


@cuda_only
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_rope_other_floating_dtypes_and_strided_angles(dtype):
    q, k = _qk(3, 11, 16, 64, "packed", dtype)
    cos, sin = _angles(2, 3)
    cos_storage = torch.zeros(6, 16, 128, device="cuda")
    sin_storage = torch.zeros_like(cos_storage)
    cos_storage[:, :, ::2] = cos.expand(-1, 16, -1)
    sin_storage[:, :, ::2] = sin.expand(-1, 16, -1)
    cos, sin = cos_storage[:, :, ::2], sin_storage[:, :, ::2]
    for output, reference in zip(proxy_rope_qk(q, k, cos, sin), _rope_reference(q, k, cos, sin)):
        assert torch.equal(output, reference)


@cuda_only
def test_prefix_preserves_every_bf16_bit_pattern():
    q, k = _qk(1, 64, 16, 64, "contiguous", torch.bfloat16)
    bits = torch.arange(-32768, 32768, device="cuda", dtype=torch.int32).to(torch.int16)
    q.copy_(bits.view(torch.bfloat16).reshape_as(q))
    k.copy_(bits.flip(0).view(torch.bfloat16).reshape_as(k))
    cos = sin = torch.empty(0, 1, 64, device="cuda")
    actual_q, actual_k = proxy_rope_qk(q, k, cos, sin, 64)
    assert torch.equal(actual_q.view(torch.int16), q.view(torch.int16))
    assert torch.equal(actual_k.view(torch.int16), k.view(torch.int16))


@cuda_only
@pytest.mark.parametrize("shape,prefix", [((0, 5, 16, 64), 5), ((3, 0, 16, 64), 0), ((3, 5, 16, 64), 5)])
def test_rope_empty_batches_or_patch_regions(shape, prefix):
    q, k = (torch.randn(shape, device="cuda", dtype=torch.bfloat16) for _ in range(2))
    angles = torch.empty(shape[1] - prefix, 1, shape[3], device="cuda")
    actual = proxy_rope_qk(q, k, angles, angles, prefix)
    assert torch.equal(actual[0], q) and torch.equal(actual[1], k)


@cuda_only
def test_rope_fullgraph_and_cuda_graph_replay_are_bit_exact():
    q, k = _qk(3, 1565, 16, 64, "packed", torch.bfloat16)
    cos, sin = _angles(30, 52)
    compiled = torch.compile(proxy_rope_qk, fullgraph=True)
    expected = _rope_reference(q, k, cos, sin)
    for output, reference in zip(compiled(q, k, cos, sin), expected):
        assert torch.equal(output, reference)
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(3):
            compiled(q, k, cos, sin)
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=side):
        actual = compiled(q, k, cos, sin)
    q.normal_()
    k.normal_()
    graph.replay()
    for output, reference in zip(actual, _rope_reference(q, k, cos, sin)):
        assert torch.equal(output, reference)


@cuda_only
@pytest.mark.parametrize("rows,width", [(0, 1024), (880, 1024), (1560 * 3, 1024), (31, 1003)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_normalize_numerical_error_and_contract(rows, width, dtype):
    torch.manual_seed(123)
    x = torch.randn(rows, width * 2, device="cuda", dtype=torch.bfloat16)[:, ::2]
    actual = proxy_normalize(x, dtype)
    reference = F.normalize(x.float(), dim=-1)
    assert actual.shape == reference.shape and actual.dtype == dtype
    assert actual.is_contiguous()
    torch.testing.assert_close(actual.float(), reference, rtol=3e-6 if dtype == torch.float32 else 4e-3, atol=1e-7 if dtype == torch.float32 else 1e-6)


@cuda_only
def test_normalize_zero_tiny_and_large_rows():
    factors = torch.tensor([0.0, 1e-30, 1e-15, 1e-12, 1.0, 1e30], device="cuda", dtype=torch.bfloat16)
    x = torch.randn(6, 1024, device="cuda", dtype=torch.bfloat16) * factors[:, None]
    torch.testing.assert_close(proxy_normalize(x), F.normalize(x.float(), dim=-1), rtol=3e-6, atol=1e-7)


@cuda_only
def test_normalize_fullgraph_and_cuda_graph():
    x = torch.randn(4680, 1024, device="cuda", dtype=torch.bfloat16)
    compiled = torch.compile(proxy_normalize, fullgraph=True)
    torch.testing.assert_close(compiled(x), F.normalize(x.float(), dim=-1), rtol=3e-6, atol=1e-7)
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(3):
            compiled(x)
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=side):
        actual = compiled(x)
    x.normal_()
    graph.replay()
    torch.testing.assert_close(actual, F.normalize(x.float(), dim=-1), rtol=3e-6, atol=1e-7)


def test_fake_metadata_and_bad_contracts():
    with FakeTensorMode():
        q = torch.empty(3, 1565, 16, 64, device="cuda", dtype=torch.bfloat16)
        k = torch.empty_like(q)
        cos = torch.empty(1560, 1, 64, device="cuda", dtype=torch.float32)
        for output in proxy_rope_qk(q, k, cos, cos):
            assert output.shape == q.shape and output.dtype == q.dtype and output.is_contiguous()
        with pytest.raises(ValueError, match="valid prefix"):
            proxy_rope_qk(q, k, cos, cos, -1)
        with pytest.raises(ValueError, match="FP32"):
            proxy_rope_qk(q, k, cos.bfloat16(), cos)
        with pytest.raises(ValueError, match="angles must have shape"):
            proxy_rope_qk(q, k, cos[:10], cos)
        x = torch.empty(4680, 1024, device="cuda", dtype=torch.bfloat16)
        assert proxy_normalize(x).dtype == torch.float32
        assert proxy_normalize(x, torch.bfloat16).dtype == torch.bfloat16
        with pytest.raises(ValueError, match="BF16 or FP32"):
            proxy_normalize(x, torch.float16)
        with pytest.raises(ValueError, match="positive"):
            proxy_normalize(x, eps=0)
