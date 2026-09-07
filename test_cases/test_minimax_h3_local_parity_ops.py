import json
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from lightx2v.models.networks.minimax_h3.infer.sglang_fused import (
    _apply_qk_neox_rope_local,
    _silu_mul_with_activation_rounding_inplace,
    apply_qk_norm_rope_sglang,
    apply_qk_norm_sglang,
    apply_vae_rope_sglang,
    indexed_gate_sglang,
    indexed_scale_shift_sglang,
    prepare_vae_rope_sglang,
    scaled_residual_add_vae_sglang,
)
from lightx2v.models.networks.minimax_h3.infer.transformer_infer import MiniMaxH3TransformerInfer

REPO_ROOT = Path(__file__).resolve().parents[1]
NVIDIA_CUDA_AVAILABLE = torch.cuda.is_available() and getattr(torch.version, "hip", None) is None
CUDA_ONLY = pytest.mark.skipif(not NVIDIA_CUDA_AVAILABLE, reason="requires NVIDIA CUDA Triton kernels")


class _Norm:
    def __init__(self, weight: torch.Tensor, eps: float = 1e-5):
        self.weight = weight
        self.eps = eps

    def _get_actual_weight(self) -> torch.Tensor:
        return self.weight


def _neox_rope_reference(
    hidden_states: torch.Tensor,
    cache: torch.Tensor,
    positions: torch.Tensor,
) -> torch.Tensor:
    rotary_dim = cache.shape[-1]
    half = rotary_dim // 2
    flat = hidden_states.reshape(-1, hidden_states.shape[-2], hidden_states.shape[-1])
    selected = cache.index_select(0, positions.repeat(flat.shape[0] // positions.numel()))
    cos = selected[:, None, :half]
    sin = selected[:, None, half:]
    first = flat[..., :half]
    second = flat[..., half:rotary_dim]
    first_cos = (first * cos).to(flat.dtype)
    second_sin = (second * sin).to(flat.dtype)
    second_cos = (second * cos).to(flat.dtype)
    first_sin = (first * sin).to(flat.dtype)
    output = torch.cat(
        (
            (first_cos - second_sin).to(flat.dtype),
            (second_cos + first_sin).to(flat.dtype),
            flat[..., rotary_dim:],
        ),
        dim=-1,
    )
    return output.reshape(hidden_states.shape)


def test_h3_parity_configuration_has_no_checkout_paths():
    with (REPO_ROOT / "configs/minimax_h3/minimax_h3_ref2av.json").open(encoding="utf-8") as handle:
        config = json.load(handle)

    assert config["sgl_aligned"] is True
    assert "h3_sglang_parity_ops" not in config
    assert "h3_packed_sequence_alignment" not in config
    assert "h3_rng_mode" not in config
    assert "h3_step_update" not in config
    assert "sglang_compatible_export" not in config
    assert "h3_sglang_root" not in config
    assert "sglang_ffmpeg_path" not in config

    checked_files = (
        REPO_ROOT / "lightx2v/models/networks/minimax_h3/infer/sglang_fused.py",
        REPO_ROOT / "lightx2v/models/networks/minimax_h3/infer/transformer_infer.py",
        REPO_ROOT / "lightx2v/models/video_encoders/hf/minimax_h3/video_vae.py",
        REPO_ROOT / "lightx2v/models/runners/minimax_h3/minimax_h3_runner.py",
        REPO_ROOT / "scripts/minimax_h3/run_minimax_h3_ref2av.sh",
    )
    combined_source = "\n".join(path.read_text(encoding="utf-8") for path in checked_files)
    assert "/data/wushuo1/sglang" not in combined_source
    assert "configure_sglang_fused_ops" not in combined_source
    assert "import sgl_kernel" not in combined_source
    assert "from sglang" not in combined_source


def test_h3_parity_initializes_without_sglang_root():
    transformer = MiniMaxH3TransformerInfer({"sgl_aligned": True})
    assert transformer.sglang_parity_ops is True


@CUDA_ONLY
def test_qknorm_matches_fp32_reference_on_production_strides():
    torch.manual_seed(17)
    tokens, heads, head_dim = 65, 7, 128
    packed = torch.randn(tokens, 3 * heads * head_dim, device="cuda", dtype=torch.bfloat16)
    q, k, value = (part.unflatten(-1, (heads, head_dim)) for part in packed.split(heads * head_dim, dim=-1))
    q_before = q.clone()
    k_before = k.clone()
    value_before = value.clone()
    q_weight = torch.randn(head_dim, device="cuda", dtype=torch.bfloat16)
    k_weight = torch.randn_like(q_weight)
    q_norm = _Norm(q_weight)
    k_norm = _Norm(k_weight)
    q_pointer, k_pointer = q.data_ptr(), k.data_ptr()
    q_stride, k_stride = q.stride(), k.stride()

    actual_q, actual_k = apply_qk_norm_sglang(q, k, q_norm, k_norm)
    expected_q = F.rms_norm(q_before.float(), (head_dim,), q_weight.float(), q_norm.eps).to(torch.bfloat16)
    expected_k = F.rms_norm(k_before.float(), (head_dim,), k_weight.float(), k_norm.eps).to(torch.bfloat16)

    assert torch.equal(actual_q, expected_q)
    assert torch.equal(actual_k, expected_k)
    assert actual_q.data_ptr() == q_pointer and actual_q.stride() == q_stride
    assert actual_k.data_ptr() == k_pointer and actual_k.stride() == k_stride
    assert torch.equal(value, value_before)


@CUDA_ONLY
def test_qknorm_rope_is_bit_exact_and_does_not_touch_value_slice():
    torch.manual_seed(19)
    tokens, heads, head_dim, rotary_dim = 33, 7, 128, 96
    packed = torch.randn(tokens, 3 * heads * head_dim, device="cuda", dtype=torch.bfloat16)
    q, k, value = (part.unflatten(-1, (heads, head_dim)) for part in packed.split(heads * head_dim, dim=-1))
    q_before = q.clone()
    k_before = k.clone()
    value_before = value.clone()
    q_weight = torch.randn(head_dim, device="cuda", dtype=torch.bfloat16)
    k_weight = torch.randn_like(q_weight)
    q_norm = _Norm(q_weight)
    k_norm = _Norm(k_weight)
    cache = torch.randn(tokens, rotary_dim, device="cuda", dtype=torch.bfloat16)
    positions = torch.arange(tokens, device="cuda", dtype=torch.long)
    q_pointer, k_pointer = q.data_ptr(), k.data_ptr()
    q_stride, k_stride = q.stride(), k.stride()

    actual_q, actual_k = apply_qk_norm_rope_sglang(q, k, q_norm, k_norm, (cache, positions))
    normalized_q = F.rms_norm(q_before.float(), (head_dim,), q_weight.float(), q_norm.eps).to(torch.bfloat16)
    normalized_k = F.rms_norm(k_before.float(), (head_dim,), k_weight.float(), k_norm.eps).to(torch.bfloat16)
    expected_q = _neox_rope_reference(normalized_q, cache, positions)
    expected_k = _neox_rope_reference(normalized_k, cache, positions)

    assert torch.equal(actual_q, expected_q)
    assert torch.equal(actual_k, expected_k)
    assert actual_q.data_ptr() == q_pointer and actual_q.stride() == q_stride
    assert actual_k.data_ptr() == k_pointer and actual_k.stride() == k_stride
    assert torch.equal(value, value_before)


@CUDA_ONLY
def test_qknorm_rope_rejects_invalid_positions_before_mutating_qk():
    q = torch.randn(2, 1, 128, device="cuda", dtype=torch.bfloat16)
    k = torch.randn_like(q)
    q_before = q.clone()
    k_before = k.clone()
    q_norm = _Norm(torch.ones(128, device="cuda", dtype=torch.bfloat16))
    k_norm = _Norm(torch.ones(128, device="cuda", dtype=torch.bfloat16))
    cache = torch.ones(1, 96, device="cuda", dtype=torch.bfloat16)
    positions = torch.empty(0, device="cuda", dtype=torch.long)

    with pytest.raises(ValueError, match="must not be empty"):
        apply_qk_norm_rope_sglang(q, k, q_norm, k_norm, (cache, positions))

    assert torch.equal(q, q_before)
    assert torch.equal(k, k_before)


@CUDA_ONLY
def test_qknorm_rejects_overlapping_token_stride():
    q_storage = torch.randn(320, device="cuda", dtype=torch.bfloat16)
    k_storage = torch.randn_like(q_storage)
    q = q_storage.as_strided((2, 2, 128), (64, 128, 1))
    k = k_storage.as_strided((2, 2, 128), (64, 128, 1))
    q_norm = _Norm(torch.ones(128, device="cuda", dtype=torch.bfloat16))
    k_norm = _Norm(torch.ones(128, device="cuda", dtype=torch.bfloat16))

    with pytest.raises(ValueError, match="Overlapping.*token strides"):
        apply_qk_norm_sglang(q, k, q_norm, k_norm)


@CUDA_ONLY
def test_indexed_modulation_matches_explicit_bf16_rounding():
    torch.manual_seed(23)
    rows, hidden_size, states = 11, 5376, 4
    indices = torch.randint(states, (rows,), device="cuda")
    shift = torch.randn(states, hidden_size, device="cuda", dtype=torch.bfloat16)
    scale = torch.randn_like(shift)
    gate = torch.randn_like(shift)
    other = torch.randn(rows, hidden_size, device="cuda", dtype=torch.bfloat16)
    source = torch.randn_like(other)

    one_plus_scale = (1.0 + scale.index_select(0, indices).float()).to(torch.bfloat16).float()
    scaled = (source.float() * one_plus_scale).to(torch.bfloat16).float()
    expected_scale_shift = (scaled + shift.index_select(0, indices).float()).to(torch.bfloat16)
    actual_scale_shift = indexed_scale_shift_sglang(source.clone(), shift, scale, indices)

    product = (gate.index_select(0, indices).float() * other.float()).to(torch.bfloat16).float()
    expected_gate = (source.float() + product).to(torch.bfloat16)
    actual_gate = indexed_gate_sglang(source.clone(), gate, other, indices)

    assert torch.equal(actual_scale_shift, expected_scale_shift)
    assert torch.equal(actual_gate, expected_gate)


def test_indexed_modulation_rejects_noncontiguous_last_dimension():
    rows, hidden_size, states = 3, 8, 2
    x = torch.zeros(rows, hidden_size * 2, dtype=torch.bfloat16)[:, ::2]
    table = torch.zeros(states, hidden_size, dtype=torch.bfloat16)
    indices = torch.zeros(rows, dtype=torch.long)

    with pytest.raises(ValueError, match="contiguous last dimension"):
        indexed_scale_shift_sglang(x, table, table, indices)


@CUDA_ONLY
def test_silu_mul_preserves_the_activation_rounding_boundary():
    gate = torch.full((1, 16), -5.0, device="cuda", dtype=torch.bfloat16)
    value = torch.full((1, 16), 0.1, device="cuda", dtype=torch.bfloat16)
    packed = torch.cat((gate, value), dim=-1)
    value_before = packed[..., 16:].clone()
    pointer = packed.data_ptr()

    actual = _silu_mul_with_activation_rounding_inplace(packed)
    activated = F.silu(gate.float()).to(torch.bfloat16).float()
    expected = (activated * value.float()).to(torch.bfloat16)

    assert torch.equal(actual, expected)
    assert actual.data_ptr() == pointer
    assert actual[0, 0].item() == -0.0033416748046875
    assert torch.equal(packed[..., 16:], value_before)


@CUDA_ONLY
def test_scaled_residual_add_matches_uncontracted_eager_ops():
    torch.manual_seed(29)
    with torch.inference_mode():
        residual = torch.randn(9, 64, device="cuda", dtype=torch.float32)
        hidden_states = torch.randn(9, 64, device="cuda", dtype=torch.float16)
        scale = torch.randn(64, device="cuda", dtype=torch.float32)
        product = hidden_states * scale
        expected = residual + product
        actual = scaled_residual_add_vae_sglang(residual, hidden_states, scale)

    assert torch.equal(actual, expected)


@CUDA_ONLY
@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16))
def test_vae_rope_matches_explicit_rounding_and_preserves_tail(dtype):
    torch.manual_seed(31)
    batch, tokens, heads, head_dim, rotary_dim = 1, 19, 32, 64, 48
    packed = torch.randn(batch, tokens, heads, 3 * head_dim, device="cuda", dtype=dtype)
    query, key, value = packed.split(head_dim, dim=-1)
    query_before = query.clone()
    key_before = key.clone()
    value_before = value.clone()
    angles = torch.randn(tokens, rotary_dim // 2, device="cuda")
    cos = torch.cat((angles.cos(), angles.cos()), dim=-1).view(1, tokens, 1, rotary_dim)
    sin = torch.cat((angles.sin(), angles.sin()), dim=-1).view(1, tokens, 1, rotary_dim)
    prepared = prepare_vae_rope_sglang((cos, sin), dtype=dtype)
    _, _, cache, positions = prepared
    expected_q = _neox_rope_reference(query_before, cache, positions)
    expected_k = _neox_rope_reference(key_before, cache, positions)

    actual_q, actual_k = apply_vae_rope_sglang(query, key, prepared)

    assert torch.equal(actual_q, expected_q)
    assert torch.equal(actual_k, expected_k)
    assert actual_q.is_contiguous() and actual_k.is_contiguous()
    assert torch.equal(actual_q[..., rotary_dim:], query_before[..., rotary_dim:])
    assert torch.equal(actual_k[..., rotary_dim:], key_before[..., rotary_dim:])
    assert torch.equal(query, query_before)
    assert torch.equal(key, key_before)
    assert torch.equal(value, value_before)


@CUDA_ONLY
def test_vae_rope_accepts_noncontiguous_cache_and_positions():
    torch.manual_seed(37)
    tokens, heads, head_dim, rotary_dim = 9, 3, 64, 48
    query = torch.randn(tokens, heads, head_dim, device="cuda", dtype=torch.float16)
    key = torch.randn_like(query)
    cache_storage = torch.randn(tokens * 2, rotary_dim * 2, device="cuda", dtype=torch.float16)
    cache = cache_storage[::2, ::2]
    position_storage = torch.empty(tokens * 2, device="cuda", dtype=torch.long)
    position_storage[::2] = torch.arange(tokens, device="cuda", dtype=torch.long)
    positions = position_storage[::2]
    expected_q = _neox_rope_reference(query, cache, positions)
    expected_k = _neox_rope_reference(key, cache, positions)

    actual_q, actual_k = apply_vae_rope_sglang(query, key, (None, None, cache, positions))

    assert torch.equal(actual_q, expected_q)
    assert torch.equal(actual_k, expected_k)


@CUDA_ONLY
def test_local_rope_materializes_expanded_token_layout():
    torch.manual_seed(41)
    tokens, heads, head_dim, rotary_dim = 5, 2, 64, 48
    query = torch.randn(1, heads, head_dim, device="cuda", dtype=torch.float16).expand(tokens, -1, -1)
    key = torch.randn(1, heads, head_dim, device="cuda", dtype=torch.float16).expand(tokens, -1, -1)
    cache = torch.randn(tokens, rotary_dim, device="cuda", dtype=torch.float16)
    positions = torch.arange(tokens, device="cuda", dtype=torch.long)
    expected_q = _neox_rope_reference(query, cache, positions)
    expected_k = _neox_rope_reference(key, cache, positions)

    actual_q, actual_k = _apply_qk_neox_rope_local(query, key, cache, positions)

    assert actual_q.is_contiguous() and actual_k.is_contiguous()
    assert torch.equal(actual_q, expected_q)
    assert torch.equal(actual_k, expected_k)


def test_vae_rope_rejects_empty_positions():
    query = torch.empty(1, 2, 64, dtype=torch.float16)
    key = torch.empty_like(query)
    cache = torch.empty(1, 48, dtype=torch.float16)
    positions = torch.empty(0, dtype=torch.long)

    with pytest.raises(ValueError, match="must not be empty"):
        apply_vae_rope_sglang(query, key, (None, None, cache, positions))
