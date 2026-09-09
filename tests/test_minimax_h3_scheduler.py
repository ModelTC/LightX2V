from unittest.mock import Mock

import pytest
import torch

from lightx2v.models.runners.minimax_h3.minimax_h3_runner import MiniMaxH3Runner
from lightx2v.models.schedulers.minimax_h3.scheduler import MiniMaxH3Scheduler
from lightx2v.utils.profiler import no_sync_profiling


@pytest.fixture(autouse=True)
def cpu_scheduler(monkeypatch):
    monkeypatch.setattr("lightx2v.models.schedulers.minimax_h3.scheduler.AI_DEVICE", "cpu")


@pytest.mark.parametrize("infer_steps", [1, 4, 5, 8, 29, 30])
@pytest.mark.parametrize("step_update", ["reference_blend", "training_euler"])
def test_runner_executes_configured_steps_and_reaches_terminal_zero(infer_steps, step_update):
    scheduler = MiniMaxH3Scheduler({"infer_steps": infer_steps, "video_flow_shift": 6.0, "audio_flow_shift": 3.0, "h3_step_update": step_update})
    scheduler.prepare(seed=42, num_frames=124, height=32, width=32, text_token_tags=torch.tensor([1]))
    initial_video = scheduler.video_latents.clone()
    initial_audio = scheduler.audio_latents.clone()

    def predict_velocity(inputs):
        scheduler.video_noise_pred = torch.ones_like(scheduler.video_latents)
        scheduler.audio_noise_pred = torch.ones_like(scheduler.audio_latents)

    runner = MiniMaxH3Runner.__new__(MiniMaxH3Runner)
    runner.scheduler = scheduler
    runner.inputs = {}
    runner.model = Mock(infer=Mock(side_effect=predict_velocity))
    runner.check_stop = Mock()
    runner.progress_callback = Mock()

    with no_sync_profiling():
        video, audio = runner.run_segment()

    assert runner.model.infer.call_count == infer_steps
    runner.progress_callback.assert_called_with(100, 100)
    assert scheduler.video_sigmas[scheduler.step_index + 1] == 0
    assert scheduler.audio_sigmas[scheduler.step_index + 1] == 0
    torch.testing.assert_close(video, initial_video + 1, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(audio, initial_audio + 1, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("infer_steps", [0, -1])
def test_scheduler_rejects_nonpositive_steps(infer_steps):
    with pytest.raises(ValueError, match="infer_steps must be at least 1"):
        MiniMaxH3Scheduler({"infer_steps": infer_steps})
