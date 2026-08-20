import torch

from lightx2v.models.networks.minimax_h3.infer.pre_infer import interpolate_adaln_curve


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
