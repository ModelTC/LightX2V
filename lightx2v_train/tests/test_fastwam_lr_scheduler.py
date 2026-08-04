import io

import pytest
import torch
from lightx2v_train.trainers.fastwam import FastWAMTrainer
from torch.optim.lr_scheduler import ConstantLR, CosineAnnealingLR, SequentialLR


def _build_trainer(*, scheduler_name="cosine", eta_min_ratio=0.01, max_train_iters=20, warmup_iters=4):
    trainer = FastWAMTrainer(
        {
            "model": {},
            "training": {
                "output_dir": "unused",
                "max_train_iters": max_train_iters,
                "lr_scheduler": scheduler_name,
                "lr_warmup_iters": warmup_iters,
                "lr_eta_min_ratio": eta_min_ratio,
                "optimizer": {"learning_rate": 1.0e-4},
            },
        }
    )
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    trainer.optimizer = torch.optim.AdamW([parameter], lr=trainer.learning_rate)
    return trainer, parameter


def _step(optimizer, scheduler, count):
    for _ in range(count):
        optimizer.step()
        scheduler.step()


def test_cosine_scheduler_uses_configured_base_lr_for_eta_min():
    trainer, _ = _build_trainer()

    scheduler = trainer._build_lr_scheduler()

    assert isinstance(scheduler, SequentialLR)
    assert isinstance(scheduler._schedulers[1], CosineAnnealingLR)
    assert scheduler._schedulers[1].eta_min == pytest.approx(1.0e-6)
    assert scheduler.get_last_lr() == pytest.approx([trainer.learning_rate / trainer.lr_warmup_iters])

    _step(trainer.optimizer, scheduler, trainer.max_train_iters)
    assert trainer.optimizer.param_groups[0]["lr"] == pytest.approx(1.0e-6)


def test_cosine_scheduler_allows_zero_eta_min_ratio():
    trainer, _ = _build_trainer(eta_min_ratio=0.0)
    scheduler = trainer._build_lr_scheduler()

    assert scheduler._schedulers[1].eta_min == 0.0

    _step(trainer.optimizer, scheduler, trainer.max_train_iters)
    assert trainer.optimizer.param_groups[0]["lr"] == pytest.approx(0.0, abs=1.0e-15)


def test_constant_scheduler_keeps_base_lr_after_warmup():
    trainer, _ = _build_trainer(scheduler_name="constant")
    scheduler = trainer._build_lr_scheduler()

    assert isinstance(scheduler, SequentialLR)
    assert isinstance(scheduler._schedulers[1], ConstantLR)

    _step(trainer.optimizer, scheduler, trainer.lr_warmup_iters)
    assert trainer.optimizer.param_groups[0]["lr"] == pytest.approx(1.0e-4)

    _step(trainer.optimizer, scheduler, trainer.max_train_iters - trainer.lr_warmup_iters)
    assert trainer.optimizer.param_groups[0]["lr"] == pytest.approx(1.0e-4)


@pytest.mark.parametrize("scheduler_name", ["cosine", "constant"])
def test_new_sequential_lr_checkpoint_round_trip_continues_same_curve(scheduler_name):
    trainer, _ = _build_trainer(scheduler_name=scheduler_name, max_train_iters=12, warmup_iters=3)
    scheduler = trainer._build_lr_scheduler()
    _step(trainer.optimizer, scheduler, 7)

    checkpoint_buffer = io.BytesIO()
    torch.save(
        {
            "optimizer": trainer.optimizer.state_dict(),
            "lr_scheduler": scheduler.state_dict(),
        },
        checkpoint_buffer,
    )
    checkpoint_buffer.seek(0)
    checkpoint = torch.load(checkpoint_buffer, map_location="cpu", weights_only=False)

    resumed_trainer, _ = _build_trainer(scheduler_name=scheduler_name, max_train_iters=12, warmup_iters=3)
    resumed_scheduler = resumed_trainer._build_lr_scheduler()
    resumed_trainer.optimizer.load_state_dict(checkpoint["optimizer"])
    resumed_scheduler.load_state_dict(checkpoint["lr_scheduler"])

    assert resumed_scheduler.last_epoch == scheduler.last_epoch
    assert resumed_scheduler.get_last_lr() == pytest.approx(scheduler.get_last_lr())
    if scheduler_name == "cosine":
        assert resumed_scheduler._schedulers[1].eta_min == pytest.approx(1.0e-6)

    for _ in range(5):
        trainer.optimizer.step()
        scheduler.step()
        resumed_trainer.optimizer.step()
        resumed_scheduler.step()
        assert resumed_scheduler.get_last_lr() == pytest.approx(scheduler.get_last_lr())


@pytest.mark.parametrize(
    ("scheduler_name", "eta_min_ratio", "warmup_iters"),
    [
        ("linear", 0.01, 4),
        ("cosine", -0.01, 4),
        ("cosine", 1.01, 4),
        ("cosine", 0.01, 0),
        ("cosine", 0.01, 20),
    ],
)
def test_invalid_scheduler_configuration_fails_early(scheduler_name, eta_min_ratio, warmup_iters):
    trainer, _ = _build_trainer(
        scheduler_name=scheduler_name,
        eta_min_ratio=eta_min_ratio,
        warmup_iters=warmup_iters,
    )

    with pytest.raises(ValueError):
        trainer._build_lr_scheduler()
