import pytest
import torch

from lightx2v.common.ops.rope import ChunkedRope, TorchComplexRope, TorchRealRope, get_rope_module


def test_complex_and_real_interleaved_match():
    torch.manual_seed(0)
    q = torch.randn(7, 3, 8)
    k = torch.randn_like(q)
    angles = torch.randn(7, 4)
    freqs = torch.polar(torch.ones_like(angles), angles).unsqueeze(1)

    complex_out = TorchComplexRope().apply(q, k, freqs)
    real_out = TorchRealRope(layout="interleaved").apply(q, k, freqs)

    torch.testing.assert_close(real_out[0], complex_out[0])
    torch.testing.assert_close(real_out[1], complex_out[1])


def test_real_split_half_accepts_full_and_half_width_freqs():
    torch.manual_seed(1)
    x = torch.randn(2, 5, 3, 8)
    angles = torch.randn(2, 5, 3, 4)
    half_freqs = (angles.cos(), angles.sin())
    full_freqs = tuple(item.repeat(1, 1, 1, 2) for item in half_freqs)
    rope = TorchRealRope(layout="split_half")

    half_out = rope.apply_single(x, half_freqs)
    full_out = rope.apply_single(x, full_freqs)

    torch.testing.assert_close(half_out, full_out)


def test_real_rope_preserves_non_rotary_tail():
    torch.manual_seed(2)
    x = torch.randn(6, 2, 10)
    angles = torch.randn(6, 4)
    output = TorchRealRope(layout="split_half").apply_single(
        x,
        (angles.cos(), angles.sin()),
        rotary_dim=8,
    )

    torch.testing.assert_close(output[..., 8:], x[..., 8:])


def test_chunked_rope_matches_inner_and_preserves_tail_tokens():
    torch.manual_seed(3)
    q = torch.randn(9, 2, 8)
    k = torch.randn_like(q)
    freqs = torch.polar(torch.ones(7, 1, 4), torch.randn(7, 1, 4))
    inner = TorchComplexRope()
    chunked = ChunkedRope(inner, chunk_size=3)

    q_out, k_out = chunked.apply(q, k, freqs)
    q_expected, k_expected = inner.apply(q[:7], k[:7], freqs)

    torch.testing.assert_close(q_out[:7], q_expected)
    torch.testing.assert_close(k_out[:7], k_expected)
    torch.testing.assert_close(q_out[7:], q[7:])
    torch.testing.assert_close(k_out[7:], k[7:])


def test_builder_uses_explicit_rope_module_names():
    assert isinstance(get_rope_module("torch_complex_rope"), TorchComplexRope)
    assert isinstance(get_rope_module("torch_real_rope"), TorchRealRope)
    assert isinstance(get_rope_module("torch_naive"), TorchRealRope)
    with pytest.raises(ValueError, match="Unsupported rope_type: torch"):
        get_rope_module("torch")
