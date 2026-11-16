"""The latent gradient bridge must preserve PDMD/GAN updates and RNG state."""

import weakref
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from lightx2v_train.trainers.dmd.official_pdmd_core import joint_map
from lightx2v_train.trainers.dmd.official_pdmd_training import OfficialH3PdmdTraining
from lightx2v_train.trainers.dmd.tests.test_h3_image_gan import gan_fixture

from lightx2v_train.trainers.dmd.tests import test_official_pdmd_gan as gan_fixtures


class _SavedTensor:
    def __init__(self, tensor):
        self.tensor = tensor


def _fixture(kind, recompute, accum, stochastic):
    torch.manual_seed(123)
    if kind == "joint":
        fixture = gan_fixtures.GanPDMDTest()
        fixture.setUp()
        trainer, trajectory = fixture.t, fixture.trajectory
        engine = OfficialH3PdmdTraining.__new__(OfficialH3PdmdTraining)
        engine.trainer, engine.dmd, engine.gan = trainer, trainer.dmd_config, fixture.gan
        engine.seed = 17
        real = fixture.real
    else:
        engine, trainer, real, _ = gan_fixture()
        trajectory = engine._trajectory(0, 0, 0)

    engine.dmd["gan"]["student_recompute"] = recompute
    engine.critic_steps, engine.accum = 5, accum
    engine.consumed_microbatches = 5 * accum
    engine.cached = [
        replace(trajectory, condition={**trajectory.condition, "bias": 0.04 + 0.07 * micro})
        for micro in range(accum)
    ]
    engine.route_loss_scale = lambda offset: (0.37, 1.6)[offset % 2]
    # SGD makes erroneous loss/accumulation scaling visible in the actual update.
    trainer.optimizer = torch.optim.SGD(trainer.trainable_params, lr=0.013)
    trainer.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(trainer.optimizer, lambda _: 1)
    trainer.max_grad_norm = float("inf")
    gradients = []
    trainer._sync_sequence_parallel_grads = lambda parameters: gradients.extend(parameter.grad.detach().clone() for parameter in parameters)
    trainer._set_student_gradient_sync = Mock()
    saved_critic_tensors = []

    def pack_saved_tensor(tensor):
        saved = _SavedTensor(tensor)
        saved_critic_tensors.append(weakref.ref(saved))
        return saved

    generator_loss = engine.gan.generator

    def track_critic_graph(*args, **kwargs):
        with torch.autograd.graph.saved_tensors_hooks(pack_saved_tensor, lambda saved: saved.tensor):
            return generator_loss(*args, **kwargs)

    engine.gan.generator = track_critic_graph
    calls = {name: [] for name in ("student", "fake", "teacher")}
    for name in calls:
        capability = getattr(trainer, name)
        predict = capability.predict_velocity

        def tracked(state, sigma, condition, *, role=name, original=predict):
            if recompute and role == "student" and torch.is_grad_enabled():
                assert saved_critic_tensors, "The test must exercise a differentiable critic graph"
                assert all(saved() is None for saved in saved_critic_tensors), "Critic activations survived into the student replay"
            calls[role].append(torch.is_grad_enabled())
            value = original(state, sigma, condition)
            if stochastic:
                value = joint_map(value, lambda tensor, _: tensor * (0.9 + 0.2 * torch.rand_like(tensor)))
            return value

        capability.predict_velocity = tracked
    return engine, trainer, real, gradients, calls


def _student_update(kind, recompute, accum, stochastic):
    engine, trainer, real, gradients, calls = _fixture(kind, recompute, accum, stochastic)
    frozen_parameters = [*trainer.fake_trainable_params, *trainer.dmd2_gan_trainable_params, *trainer.teacher.denoiser().parameters()]
    frozen_before = [parameter.detach().clone() for parameter in frozen_parameters]
    student_before = [parameter.detach().clone() for parameter in trainer.trainable_params]
    torch.manual_seed(9123)
    role, metrics = engine._update(iter(()), 5)
    rng = torch.get_rng_state()
    assert role == "student"
    assert engine.cached == []
    assert engine.consumed_microbatches == 5 * accum
    real.load.assert_not_called()
    for parameter, before in zip(frozen_parameters, frozen_before):
        torch.testing.assert_close(parameter, before, atol=0, rtol=0)
        assert parameter.grad is None
    assert any(not torch.equal(parameter, before) for parameter, before in zip(trainer.trainable_params, student_before))
    assert all(torch.isfinite(gradient).all() for gradient in gradients)
    assert calls["teacher"] == [False] * accum
    assert calls["fake"] == [False] * accum
    assert calls["student"] == ([False, True] * accum if recompute else [True] * accum)
    assert [call.args[0] for call in trainer._set_student_gradient_sync.call_args_list] == [micro == accum - 1 for micro in range(accum)]
    return SimpleNamespace(parameters=[parameter.detach().clone() for parameter in trainer.trainable_params], gradients=gradients, metrics=metrics, rng=rng)


@pytest.mark.parametrize("kind", ["joint", "tensor"])
@pytest.mark.parametrize("accum", [1, 2])
@pytest.mark.parametrize("stochastic", [False, True])
def test_recomputed_generator_matches_connected_graph(kind, accum, stochastic):
    baseline = _student_update(kind, False, accum, stochastic)
    recomputed = _student_update(kind, True, accum, stochastic)
    assert baseline.metrics.keys() == recomputed.metrics.keys()
    for name in baseline.metrics:
        assert recomputed.metrics[name] == pytest.approx(baseline.metrics[name], rel=2e-6, abs=2e-7), name
    for expected, actual in zip(baseline.gradients, recomputed.gradients):
        torch.testing.assert_close(actual, expected, rtol=2e-6, atol=2e-7)
    for expected, actual in zip(baseline.parameters, recomputed.parameters):
        torch.testing.assert_close(actual, expected, rtol=2e-6, atol=2e-7)
    torch.testing.assert_close(recomputed.rng, baseline.rng, rtol=0, atol=0)
