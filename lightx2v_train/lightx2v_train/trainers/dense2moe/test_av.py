"""Joint modality gradients and H3's separate audio/video time conditioning."""

from fractions import Fraction

import av
import numpy as np
import pytest
import torch

from .data import load_media
from .loss import modality_losses
from .runtime import AVSample, shift_sigma


def test_audio_loss_is_not_diluted_by_video_token_count():
    prediction = {"video": torch.ones(1, 500, 96, requires_grad=True), "audio": torch.full((1, 10, 32), 2.0, requires_grad=True)}
    target = {name: torch.zeros_like(value) for name, value in prediction.items()}
    losses = modality_losses(prediction, target)
    loss = sum(losses.values())
    loss.backward()
    assert losses["video"] == 1 and losses["audio"] == 4
    for name, value in prediction.items():
        torch.testing.assert_close(value.grad, 2 * value.detach() / value.numel())


def test_joint_forward_preserves_stereo_order_and_modality_times():
    condition = {"text_token_tags": torch.ones(3, dtype=torch.long), "prompt_embeds": torch.zeros(1, 3, 5120)}
    sample = AVSample(condition, {"width": 64, "height": 64, "num_frames": 124}, "cpu")
    latents = {name: torch.randn(shape) for name, shape in sample.shapes.items()}
    sigmas = {"video": shift_sigma(torch.tensor(0.5), 12), "audio": shift_sigma(torch.tensor(0.5), 3)}
    captured = {}

    def model(**kwargs):
        captured.update(kwargs)
        return kwargs["hidden_states"], kwargs["audio_hidden_states"]

    outputs = sample.forward(model, latents, sigmas)
    assert outputs["audio"] is latents["audio"] and outputs["video"] is latents["video"]
    times = captured["timestep"][captured["timestep_indices"]]
    for name in latents:
        indices = getattr(sample.layout, name + "_indices")
        torch.testing.assert_close(times[indices], torch.full_like(times[indices], 1 - sigmas[name]))
    audio_indices = sample.layout.audio_indices
    positions = sample.layout.position_ids[audio_indices]
    torch.testing.assert_close(positions[:207, 0], positions[207:, 0])
    assert (positions[:207, 2] < positions[207:, 2]).all()


@pytest.mark.parametrize("audio_delay", [0, 3200])
def test_media_uses_shared_timeline_and_only_pads_rounding_tail(tmp_path, audio_delay):
    path = tmp_path / "sample.mkv"
    frames = 124
    samples = round(frames / 24 * 32000)
    with av.open(str(path), "w") as container:
        video = container.add_stream("ffv1", rate=24)
        video.width, video.height, video.pix_fmt = 32, 32, "yuv444p"
        audio = container.add_stream("pcm_f32le", rate=32000)
        audio.layout = "stereo"
        for index in range(frames):
            frame = av.VideoFrame.from_ndarray(np.full((32, 32, 3), 80, dtype=np.uint8), format="rgb24")
            frame.pts, frame.time_base = index + 24, Fraction(1, 24)
            for packet in video.encode(frame):
                container.mux(packet)
        for packet in video.encode():
            container.mux(packet)
        values = np.tile(np.array([0.2, -0.3], dtype=np.float32), (samples, 1))
        frame = av.AudioFrame.from_ndarray(values.reshape(1, -1), format="flt", layout="stereo")
        frame.sample_rate, frame.time_base, frame.pts = 32000, Fraction(1, 32000), 32000 + audio_delay
        for packet in audio.encode(frame):
            container.mux(packet)
        for packet in audio.encode():
            container.mux(packet)
    record = {"width": 32, "height": 32, "num_frames": frames}
    if audio_delay:
        with pytest.raises(ValueError, match="Audio does not cover"):
            load_media(path, record)
    else:
        video, audio = load_media(path, record)
        assert video.shape == (1, 3, frames, 32, 32)
        assert audio.shape == (2, 207 * 800)
        torch.testing.assert_close(audio[:, :samples], torch.tensor([[0.2], [-0.3]]).expand(2, samples))
        assert (audio[:, samples:] == 0).all()
