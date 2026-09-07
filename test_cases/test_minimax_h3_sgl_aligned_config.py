import pytest

from lightx2v.models.networks.minimax_h3.config import MiniMaxH3SGLAlignment, resolve_minimax_h3_sgl_alignment


def test_sgl_aligned_true_resolves_complete_profile():
    config = {"sgl_aligned": True, "h3_rng_mode": "sglang"}

    resolved = resolve_minimax_h3_sgl_alignment(config)

    assert resolved == MiniMaxH3SGLAlignment(
        parity_ops=True,
        packed_sequence_alignment=64,
        rng_mode="sglang",
        step_update="sglang_reference_blend",
        compatible_export=True,
    )
    assert config == {"sgl_aligned": True, "h3_rng_mode": "sglang"}


def test_sgl_aligned_false_preserves_legacy_overrides():
    config = {
        "sgl_aligned": False,
        "h3_step_update": "training_euler",
        "h3_packed_sequence_alignment": 32,
    }

    resolved = resolve_minimax_h3_sgl_alignment(config)

    assert resolved == MiniMaxH3SGLAlignment(
        parity_ops=False,
        packed_sequence_alignment=32,
        rng_mode="legacy_stream",
        step_update="training_euler",
        compatible_export=False,
    )


def test_missing_sgl_aligned_preserves_legacy_overrides():
    config = {"h3_step_update": "training_euler", "h3_packed_sequence_alignment": 32}

    resolved = resolve_minimax_h3_sgl_alignment(config)

    assert config == {"h3_step_update": "training_euler", "h3_packed_sequence_alignment": 32}
    assert resolved.step_update == "training_euler"
    assert resolved.packed_sequence_alignment == 32


def test_scheduler_consumes_raw_sgl_aligned_config(monkeypatch):
    from lightx2v.models.schedulers.minimax_h3 import scheduler as scheduler_module

    monkeypatch.setattr(scheduler_module, "AI_DEVICE", "cpu")
    scheduler = scheduler_module.MiniMaxH3Scheduler({"infer_steps": 2, "sgl_aligned": True})

    assert scheduler.packed_sequence_alignment == 64
    assert scheduler.rng_mode == "sglang"
    assert scheduler.step_update == "sglang_reference_blend"


def test_sgl_aligned_conflict_is_rejected_without_partial_update():
    config = {"sgl_aligned": True, "h3_rng_mode": "legacy_stream"}
    original = config.copy()

    with pytest.raises(ValueError, match=r"sgl_aligned=True conflicts.*h3_rng_mode"):
        resolve_minimax_h3_sgl_alignment(config)

    assert config == original


@pytest.mark.parametrize("value", (None, 1, "true"))
def test_sgl_aligned_requires_a_boolean(value):
    with pytest.raises(ValueError, match="sgl_aligned must be true or false"):
        resolve_minimax_h3_sgl_alignment({"sgl_aligned": value})
