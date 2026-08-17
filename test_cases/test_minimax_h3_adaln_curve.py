from types import SimpleNamespace

import torch
import torch.nn.functional as F

from lightx2v.models.networks.minimax_h3.infer.pre_infer import interpolate_adaln_curve
from lightx2v.models.networks.minimax_h3.infer.transformer_infer import MiniMaxH3TransformerInfer


class _RecordingProjection:
    def __init__(self):
        self.call_count = 0
        self.last_input = None

    def apply(self, value):
        self.call_count += 1
        self.last_input = value
        return value


def _make_transformer_infer(*, curve):
    infer = MiniMaxH3TransformerInfer.__new__(MiniMaxH3TransformerInfer)
    infer.h3_adaln_curve = curve
    infer.infer_dtype = torch.bfloat16
    infer.hidden_size = 1
    infer.tp_size = 1
    return infer


def test_interpolate_adaln_curve_clamps_and_interpolates():
    table = torch.tensor([[0.0, 10.0], [2.0, 12.0], [4.0, 14.0]], dtype=torch.float32)
    timesteps = torch.tensor([-1.0, 0.0, 0.25, 0.5, 0.75, 1.0, 2.0], dtype=torch.float32)
    expected = torch.tensor(
        [
            [0.0, 10.0],
            [0.0, 10.0],
            [1.0, 11.0],
            [2.0, 12.0],
            [3.0, 13.0],
            [4.0, 14.0],
            [4.0, 14.0],
        ],
        dtype=torch.float32,
    )
    torch.testing.assert_close(interpolate_adaln_curve(table, timesteps), expected, rtol=0.0, atol=0.0)


def test_interpolate_adaln_curve_rejects_invalid_inputs():
    table = torch.zeros(1, 8)
    timesteps = torch.zeros(1)
    try:
        interpolate_adaln_curve(table, timesteps)
    except ValueError as error:
        assert "grid>=2" in str(error)
    else:
        raise AssertionError("invalid curve table was accepted")


def test_curve_adaln_integrates_with_request_cache():
    infer = _make_transformer_infer(curve=True)
    infer._adaln_cache = {}
    infer._current_adaln_tables = None
    infer._adaln_cache_hit = False
    infer.scheduler = SimpleNamespace(unique_timesteps_cpu=torch.tensor([0.0, 0.5], dtype=torch.float32))

    projection = _RecordingProjection()
    weights = SimpleNamespace(adaln=projection)
    pre_infer_out = SimpleNamespace(temb=torch.arange(12, dtype=torch.float32).reshape(2, 6))

    infer._prepare_adaln_cache()
    first = infer._get_or_build_adaln(0, weights, pre_infer_out)
    assert projection.last_input is pre_infer_out.temb
    assert first.dtype == torch.float32

    infer._prepare_adaln_cache()
    second = infer._get_or_build_adaln(0, weights, pre_infer_out)
    assert infer._adaln_cache_hit
    assert second is first
    assert projection.call_count == 1


def test_full_checkpoint_adaln_preserves_activation_and_dtype():
    infer = _make_transformer_infer(curve=False)
    projection = _RecordingProjection()
    weights = SimpleNamespace(adaln=projection)
    temb = torch.arange(12, dtype=torch.float32).reshape(2, 6)

    result = infer._compute_adaln_table(weights, SimpleNamespace(temb=temb))
    expected = F.silu(temb).to(torch.bfloat16)

    torch.testing.assert_close(projection.last_input, expected, rtol=0.0, atol=0.0)
    torch.testing.assert_close(result, expected, rtol=0.0, atol=0.0)
