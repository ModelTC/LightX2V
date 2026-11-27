"""Loss, sampling and optimizer invariants for H3 recovery training."""

import pytest
import torch

from .loss import RelativeFeatureMSE, feature_distillation_loss, flow_matching_losses
from .checkpoint import load_model_checkpoint
from .optimization import learning_rate_factor, optimizer_groups, sample_base_sigma
from .runtime import shift_sigma


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_relative_features_match_reference_gradient(dtype):
    student = torch.randn(19, 64, dtype=dtype, requires_grad=True)
    teacher = torch.randn_like(student)
    actual = RelativeFeatureMSE.apply(student, teacher)
    expected = (student.float() - teacher.float()).square().mean() / teacher.float().square().mean()
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(torch.autograd.grad(actual, student)[0], torch.autograd.grad(expected, student)[0])


def test_relative_features_measure_error_independently_of_output_and_layer_count():
    teacher = torch.ones(3, 16)
    student = torch.full_like(teacher, 2.0, requires_grad=True)
    output = torch.tensor(0.7)
    loss = feature_distillation_loss(output, [student], [teacher], "relative")
    torch.testing.assert_close(loss, torch.tensor(1.0))
    torch.testing.assert_close(loss, feature_distillation_loss(output * 100, [student] * 47, [teacher] * 47, "relative"))
    improved = feature_distillation_loss(output, [student * 0.75], [teacher], "relative")
    assert improved < loss


def test_h3_flow_target_sign_and_modality_weights():
    clean = {"video": torch.full((1, 20, 96), 3.0), "audio": torch.full((1, 4, 32), 2.0)}
    noise = {name: torch.ones_like(value) for name, value in clean.items()}
    prediction = {name: value - noise[name] for name, value in clean.items()}
    assert all(value == 0 for value in flow_matching_losses(prediction, clean, noise).values())
    assert all(value > 0 for value in flow_matching_losses({name: -value for name, value in prediction.items()}, clean, noise).values())


def test_sigma_mixture_covers_detail_without_breaking_joint_schedule():
    config = {"flow_shift": {"video": 12.0, "audio": 3.0}, "uniform_video_sigma_probability": 0.5}
    torch.manual_seed(51)
    base = torch.stack([sample_base_sigma(config) for _ in range(10000)])
    video = shift_sigma(base, 12)
    audio = shift_sigma(base, 3)
    assert 0.26 < float((video < 0.5).float().mean()) < 0.32
    assert (audio <= video).all() and (base >= 0).all() and (base <= 1).all()


def test_learning_rate_schedule_resume_is_step_determined():
    assert learning_rate_factor(0, 100000, 100, 0.2) == 0.01
    assert learning_rate_factor(99, 100000, 100, 0.2) == 1
    assert learning_rate_factor(100, 100000, 100, 0.2) == 1
    assert learning_rate_factor(99999, 100000, 100, 0.2) == pytest.approx(0.2)


def test_unfrozen_shared_is_in_optimizer():
    model = torch.nn.Module()
    model.ff = torch.nn.Module()
    model.ff.shared = torch.nn.Linear(8, 8)
    model.ff.router = torch.nn.Linear(8, 4, bias=False)
    model.ff.up_gate = torch.nn.Parameter(torch.zeros(4, 8, 8))
    groups = optimizer_groups(model, {"learning_rate": 5e-5, "router_learning_rate": 1e-4})
    assert {g["name"] for g in groups} == {"shared", "router", "experts"}
    assert sum(len(g["params"]) for g in groups) == len(list(model.parameters()))
    assert next(g for g in groups if g["name"] == "router")["lr"] == 1e-4


def test_model_warmstart_retains_newly_unfrozen_weights(tmp_path):
    import torch.distributed.checkpoint as dcp

    model = torch.nn.Module()
    model.shared = torch.nn.Parameter(torch.full((4, 8), 7.0))
    model.experts = torch.nn.Parameter(torch.zeros(4, 8))
    expected = torch.randn(4, 8)
    dcp.save({"model": {"experts": expected}}, checkpoint_id=tmp_path / "checkpoint")
    count = load_model_checkpoint(model, tmp_path / "checkpoint")
    assert count == 1
    torch.testing.assert_close(model.experts, expected)
    torch.testing.assert_close(model.shared, torch.full((4, 8), 7.0))
