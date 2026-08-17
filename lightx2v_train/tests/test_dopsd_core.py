from contextlib import contextmanager

import pytest
import torch
from lightx2v_train.model_capabilities import (
    AdapterBankCapability,
    DopsdCapability,
    DopsdPreparedBatch,
    DopsdPreparedTeacherBatch,
)
from lightx2v_train.schedulers.flow_matching import RectifiedFlowMatchingScheduler
from lightx2v_train.trainers.dopsd_core import DopsdConfig, DopsdObjective


class FakeAdapterBank(AdapterBankCapability):
    def __init__(self):
        self.active = "student"
        self.training = True
        self.student = torch.nn.Parameter(torch.tensor(1.0))
        self.teacher = torch.nn.Parameter(torch.tensor(0.25), requires_grad=False)

    def configure_pair(self, *args, **kwargs):
        pass

    def parameters(self, adapter_name):
        return (self.student,)

    @contextmanager
    def activate(self, adapter_name, training=None):
        previous_adapter = self.active
        previous_training = self.training
        self.active = adapter_name
        if training is not None:
            self.training = training
        try:
            yield
        finally:
            self.active = previous_adapter
            self.training = previous_training

    def set_trainable(self, adapter_name):
        self.active = adapter_name

    def copy(self, source_adapter, target_adapter):
        self.teacher.copy_(self.student.detach())

    def ema_update(self, source_adapter, target_adapter, decay):
        self.teacher.mul_(decay).add_(self.student.detach(), alpha=1.0 - decay)

    def load(self, path, adapter_name, weights_subdir=None):
        pass

    def save(self, path, adapter_name, weights_subdir=None):
        pass


class FakeDopsdModel(DopsdCapability):
    def __init__(self, adapters):
        self.adapters = adapters

    @property
    def device(self):
        return torch.device("cpu")

    @staticmethod
    def _initial_state():
        return torch.ones(1, 2, 3, 4)

    def prepare_training_batch(self, batch, teacher_prompts, running_dtype):
        return DopsdPreparedBatch(
            initial_state=self._initial_state(),
            state_ids=torch.zeros(1, 6, 1),
            student_condition="student-condition",
            teacher_condition="teacher-condition",
            teacher_reference="reference",
            latent_hw=(3, 4),
        )

    def prepare_teacher_batch(self, reference_image, teacher_prompts, running_dtype, generator=None):
        return DopsdPreparedTeacherBatch(
            initial_state=self._initial_state(),
            state_ids=torch.zeros(1, 6, 1),
            condition="teacher-condition",
            reference="reference",
            latent_hw=(3, 4),
            height=24,
            width=32,
        )

    def predict_velocity(self, state, time, condition, state_ids, reference=None):
        weight = self.adapters.student if self.adapters.active == "student" else self.adapters.teacher
        return torch.ones_like(state) * weight

    def decode_state(self, state, state_ids):
        return state


class FakeScheduler:
    def build_inference_sigmas(self, num_steps, latent_hw=None):
        assert num_steps == 2
        assert latent_hw == (3, 4)
        return torch.tensor([1.0, 0.5, 0.0])


def _config(**overrides):
    values = {
        "num_training_steps": 2,
        "step_loss_weights": (2.0, 1.0),
    }
    values.update(overrides)
    config = DopsdConfig(**values)
    config.validate()
    return config


def test_dopsd_objective_matches_weighted_x0_regression_and_backpropagates():
    adapters = FakeAdapterBank()
    objective = DopsdObjective(
        FakeDopsdModel(adapters),
        adapters,
        FakeScheduler(),
        _config(),
        torch.float32,
    )
    result = objective.compute_loss(
        {"conditioning": {"prompt": "a prompt"}},
        collect_trajectory=True,
    )

    # teacher velocity=0.25 and student velocity=1.0, so the x0 delta is
    # sigma * 0.75 at sigmas 1.0 and 0.5.
    assert result.loss.item() == pytest.approx((2.0 * 0.75**2 + 0.375**2) / 3.0)
    assert len(result.student_trajectory) == 2
    assert result.state_ids is not None
    result.loss.backward()
    assert adapters.student.grad.item() == pytest.approx(1.125)
    assert adapters.teacher.grad is None
    assert adapters.active == "student"
    assert adapters.training is True


def test_teacher_rollout_uses_the_same_schedule_and_restores_adapter_state():
    adapters = FakeAdapterBank()
    objective = DopsdObjective(
        FakeDopsdModel(adapters),
        adapters,
        FakeScheduler(),
        _config(),
        torch.float32,
    )
    rollout = objective.rollout_teacher(
        torch.zeros(1, 3, 24, 32),
        ["teacher prompt"],
        num_steps=2,
    )

    assert torch.allclose(rollout.state, torch.full_like(rollout.state, 0.75))
    assert adapters.active == "student"
    assert adapters.training is True


@pytest.mark.parametrize(
    ("dopsd", "message"),
    [
        ({"num_training_steps": 0}, "num_training_steps"),
        ({"ema_decay": 1.0}, "ema_decay"),
        ({"student_adapter": "same", "teacher_adapter": "same"}, "must be different"),
        ({"step_loss_weights": [0.0, 0.0]}, "at least one positive"),
        ({"trajectory_every_iters": 0}, "trajectory_every_iters"),
    ],
)
def test_dopsd_config_rejects_invalid_values(dopsd, message):
    with pytest.raises(ValueError, match=message):
        DopsdConfig.from_training_config({"train_type": "lora", "dopsd": dopsd})


def test_dopsd_config_rejects_full_training():
    with pytest.raises(ValueError, match="must be 'lora'"):
        DopsdConfig.from_training_config({"train_type": "full"})


def test_scheduler_can_build_a_schedule_without_mutating_inference_state():
    scheduler = RectifiedFlowMatchingScheduler(
        {
            "model": {"running_dtype": "fp32"},
            "scheduler": {"num_train_timesteps": 1000},
        }
    )
    sigmas = scheduler.build_inference_sigmas(2, latent_hw=(3, 4))

    assert torch.allclose(sigmas.cpu(), torch.tensor([1.0, 0.5, 0.0]))
    assert scheduler.infer_sigmas is None
    assert scheduler.infer_timesteps is None
