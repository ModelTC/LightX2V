import pytest
import torch
from lightx2v_train.trainers.dmd.config import DmdConfig
from lightx2v_train.trainers.dmd.score_sampling import (
    ContinuousUniformScoreSigmaSampler,
    DiscreteTimestepScoreSigmaSampler,
    ScoreSigmaContext,
    build_score_sigma_sampler,
)


def _context(**overrides):
    values = {
        "denoised_timestep_from": None,
        "denoised_timestep_to": None,
        "num_train_timesteps": 1000,
        "device": torch.device("cpu"),
    }
    values.update(overrides)
    return ScoreSigmaContext(**values)


def test_discrete_score_sampler_preserves_rollout_bounds_shift_and_clamp(monkeypatch):
    sampled_bounds = []

    def fake_randint(low, high, size, **kwargs):
        sampled_bounds.append((low, high, size))
        return torch.tensor([400], device=kwargs["device"], dtype=kwargs["dtype"])

    monkeypatch.setattr(torch, "randint", fake_randint)
    sampler = DiscreteTimestepScoreSigmaSampler(
        sample_min_timestep=50,
        clamp_min_timestep=20,
        clamp_max_timestep=980,
        timestep_shift=2.0,
        use_rollout_min=True,
        use_rollout_max=True,
    )

    sigma = sampler.sample(
        _context(
            denoised_timestep_from=900,
            denoised_timestep_to=200,
        )
    )

    assert sampled_bounds == [(200, 900, (1,))]
    assert sigma.item() == pytest.approx(2.0 * 0.4 / (1.0 + 0.4))


def test_discrete_score_sampler_clamps_after_sampling():
    sampler = DiscreteTimestepScoreSigmaSampler(
        sample_min_timestep=0,
        clamp_min_timestep=999,
        clamp_max_timestep=1000,
    )
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        sigma = sampler.sample(_context())

    assert sigma.item() == pytest.approx(0.999)


def test_continuous_score_sampler_returns_unshifted_fp32_sigma():
    sampler = ContinuousUniformScoreSigmaSampler(min_sigma=0.2, max_sigma=0.3)

    sigma = sampler.sample(_context())

    assert sigma.shape == (1,)
    assert sigma.dtype == torch.float32
    assert 0.2 <= sigma.item() < 0.3


def test_score_sampler_builder_defaults_to_legacy_discrete_policy():
    sampler = build_score_sigma_sampler(
        None,
        sample_min_timestep=10,
        clamp_min_timestep=20,
        clamp_max_timestep=980,
        timestep_shift=5.0,
        use_rollout_min=True,
        use_rollout_max=False,
    )

    assert sampler == DiscreteTimestepScoreSigmaSampler(
        sample_min_timestep=10,
        clamp_min_timestep=20,
        clamp_max_timestep=980,
        timestep_shift=5.0,
        use_rollout_min=True,
        use_rollout_max=False,
    )


@pytest.mark.parametrize(
    ("config", "message"),
    [
        ("continuous_uniform", "must be a mapping"),
        ({"type": "unknown"}, "Unsupported"),
        ({"type": "continuous_uniform", "min_sigma": 0.8, "max_sigma": 0.2}, "bounds"),
    ],
)
def test_score_sampler_builder_rejects_invalid_config(config, message):
    with pytest.raises(ValueError, match=message):
        build_score_sigma_sampler(
            config,
            sample_min_timestep=0,
            clamp_min_timestep=20,
            clamp_max_timestep=980,
            timestep_shift=5.0,
            use_rollout_min=False,
            use_rollout_max=False,
        )


def test_dmd_config_parses_an_optional_latent_dtype():
    config = {
        "training": {
            "student": {"train_type": "full"},
            "fake": {"train_type": "full"},
            "teacher": {},
            "dmd": {"latent_dtype": "fp32"},
        }
    }

    assert DmdConfig.from_mapping(config).latent_dtype == torch.float32
