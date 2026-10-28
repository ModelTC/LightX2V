from types import SimpleNamespace

import torch

from lightx2v.common.ops.rope.torch_rope import TorchRealRope
from lightx2v.models.networks.qwen_image_21 import model as qwen_image_21_model
from lightx2v.models.networks.qwen_image_21.infer.offload.transformer_infer import _record_stream
from lightx2v.models.networks.qwen_image_21.infer.pre_infer import _slice_rotary
from lightx2v.models.networks.qwen_image_21.model import QwenImage21TransformerModel
from lightx2v_platform.ops.rope.ascend_npu import qwen_image_21_npu_rope as npu_rope_module
from lightx2v_platform.ops.rope.ascend_npu.qwen_image_21_npu_rope import QwenImage21NpuRope


def _raw_freqs(length, rotary_dim):
    phase = torch.randn(length, rotary_dim // 2, dtype=torch.float32)
    return torch.cat((phase.cos(), phase.sin()), dim=-1)


def _fake_rotary_mul(calls):
    def apply(x, cos, sin, rotary_mode):
        calls.append((x.dtype, cos.dtype, sin.dtype, x.shape, cos.shape, rotary_mode))
        pairs = x.reshape(*x.shape[:-1], -1, 2)
        first, second = pairs.unbind(dim=-1)
        rotated = torch.stack((-second, first), dim=-1).flatten(-2)
        return x * cos + rotated * sin

    return apply


def test_prepare_freqs_caches_expanded_fp32_tables():
    rope = QwenImage21NpuRope()
    raw = _raw_freqs(length=7, rotary_dim=8)

    cos, sin = rope.prepare_freqs(raw, rotary_dim=8)

    assert cos.shape == (7, 8)
    assert sin.shape == (7, 8)
    assert cos.dtype == torch.float32
    assert sin.dtype == torch.float32
    torch.testing.assert_close(cos[:, 0::2], raw[:, :4])
    torch.testing.assert_close(cos[:, 1::2], raw[:, :4])
    torch.testing.assert_close(sin[:, 0::2], raw[:, 4:])
    torch.testing.assert_close(sin[:, 1::2], raw[:, 4:])


def test_tuple_tables_slice_condition_and_target_without_rebuilding():
    rope = QwenImage21NpuRope()
    tables = rope.prepare_freqs(_raw_freqs(length=9, rotary_dim=8), rotary_dim=8)

    condition = _slice_rotary(tables, 0, 3)
    target = _slice_rotary(tables, 3, None)

    assert condition[0].shape == (3, 8)
    assert target[0].shape == (6, 8)
    assert condition[0].untyped_storage().data_ptr() == tables[0].untyped_storage().data_ptr()
    assert target[1].untyped_storage().data_ptr() == tables[1].untyped_storage().data_ptr()


def test_tensor_table_slice_preserves_non_npu_representation():
    table = torch.randn(9, 8)

    target = _slice_rotary(table, 3, None)

    assert torch.is_tensor(target)
    torch.testing.assert_close(target, table[3:])
    assert target.untyped_storage().data_ptr() == table.untyped_storage().data_ptr()


def test_resident_offload_records_tuple_rotary_tables():
    stream = object()
    calls = []

    class FakeTensor:
        def record_stream(self, actual_stream):
            calls.append((self, actual_stream))

    hidden_states = FakeTensor()
    cos = FakeTensor()
    sin = FakeTensor()

    _record_stream(hidden_states, stream)
    _record_stream((cos, sin), stream)
    _record_stream(None, stream)

    assert calls == [(hidden_states, stream), (cos, stream), (sin, stream)]


def test_sequence_parallel_slices_tuple_tables(monkeypatch):
    monkeypatch.setattr(qwen_image_21_model.dist, "get_world_size", lambda group: 2)
    monkeypatch.setattr(qwen_image_21_model.dist, "get_rank", lambda group: 1)
    model = object.__new__(QwenImage21TransformerModel)
    model.seq_p_group = object()
    cos = torch.randn(8, 6).transpose(0, 1)
    sin = torch.randn(8, 6).transpose(0, 1)
    state = SimpleNamespace(
        hidden_states=torch.randn(6, 4),
        rotary=(cos, sin),
        rotary_positions=None,
    )

    result = model._seq_parallel_pre_process(state)

    assert result.hidden_states.shape == (3, 4)
    assert all(item.shape == (3, 8) for item in result.rotary)
    assert all(item.is_contiguous() for item in result.rotary)
    torch.testing.assert_close(result.rotary[0], cos[3:6])
    torch.testing.assert_close(result.rotary[1], sin[3:6])


def test_npu_path_uses_fp32_and_returns_projection_dtype(monkeypatch):
    calls = []
    fake_npu = SimpleNamespace(npu_rotary_mul=_fake_rotary_mul(calls))
    monkeypatch.setattr(npu_rope_module, "torch_npu", fake_npu)
    monkeypatch.setattr(npu_rope_module, "_is_npu_tensor", lambda tensor: True)

    q = torch.randn(5, 2, 8, dtype=torch.bfloat16)
    k = torch.randn(5, 2, 8, dtype=torch.bfloat16)
    rope = QwenImage21NpuRope()
    tables = rope.prepare_freqs(_raw_freqs(length=5, rotary_dim=8), rotary_dim=8)

    q_output, k_output = rope.apply(q, k, tables)
    reference = TorchRealRope(layout="interleaved", compute_dtype=torch.float32)
    q_reference, k_reference = reference.apply(q, k, _raw_from_tables(tables))

    assert q_output.dtype == torch.bfloat16
    assert k_output.dtype == torch.bfloat16
    assert len(calls) == 2
    assert all(call[:3] == (torch.float32, torch.float32, torch.float32) for call in calls)
    assert all(call[3] == torch.Size([5, 1, 2, 8]) for call in calls)
    assert all(call[4] == torch.Size([5, 1, 1, 8]) for call in calls)
    assert all(call[5] == "interleave" for call in calls)
    torch.testing.assert_close(q_output, q_reference)
    torch.testing.assert_close(k_output, k_reference)


def _raw_from_tables(tables):
    cos, sin = tables
    return torch.cat((cos[:, 0::2], sin[:, 0::2]), dim=-1)


def test_cpu_fallback_matches_torch_real_rope_when_torch_npu_is_available(monkeypatch):
    def fail_if_called(*args, **kwargs):
        raise AssertionError("CPU tensors must not be passed to npu_rotary_mul")

    monkeypatch.setattr(npu_rope_module, "torch_npu", SimpleNamespace(npu_rotary_mul=fail_if_called))
    q = torch.randn(6, 3, 8, dtype=torch.bfloat16)
    k = torch.randn(6, 3, 8, dtype=torch.bfloat16)
    raw = _raw_freqs(length=6, rotary_dim=8)
    rope = QwenImage21NpuRope()

    q_output, k_output = rope.apply(q, k, rope.prepare_freqs(raw, rotary_dim=8))
    reference = TorchRealRope(layout="interleaved", compute_dtype=torch.float32)
    q_reference, k_reference = reference.apply(q, k, raw)

    torch.testing.assert_close(q_output, q_reference)
    torch.testing.assert_close(k_output, k_reference)
