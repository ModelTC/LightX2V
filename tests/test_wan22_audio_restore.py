import inspect
import json
from pathlib import Path
from unittest.mock import Mock

import pytest
import torch

from lightx2v.models.runners.base_runner import BaseRunner
from lightx2v.models.runners.wan import wan_audio_runner
from lightx2v.models.runners.wan.wan_audio_runner import Wan22AudioRunner, WanAudioRunner
from lightx2v.models.schedulers.wan.audio import scheduler as audio_scheduler
from lightx2v.pipeline import LightX2VPipeline
from lightx2v.utils.lockable_dict import LockableDict
from lightx2v.utils.set_config import build_startup_config


def test_legacy_i2v_initializes_and_runs_audio_encoder(monkeypatch):
    runner = object.__new__(Wan22AudioRunner)
    BaseRunner.__init__(
        runner,
        LockableDict({"model_cls": "wan2.2_audio", "task": "i2v", "vae_stride": [4, 16, 16], "lazy_load": True, "cpu_offload": True, "use_image_encoder": False}),
    )
    runner.init_modules()
    runner.input_info = runner.prepare_request({"image_path": "portrait.png", "audio_path": "speech.wav"})
    runner.read_image_input = Mock(return_value=("image", [48, 5, 4, 4], [64, 64]))
    runner.run_vae_encoder = Mock(return_value="image latents")
    runner.read_audio_input = Mock(return_value=(["audio segment"], 17, None, 1))
    runner.run_text_encoder = Mock(return_value="text embeddings")
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)

    result = runner.run_input_encoder()

    runner.read_image_input.assert_called_once_with("portrait.png")
    runner.read_audio_input.assert_called_once_with("speech.wav")
    assert result["audio_segments"] == ["audio segment"]
    assert result["image_encoder_output"]["vae_encoder_out"] == "image latents"
    assert runner.input_info.audio_num == 1
    assert runner.input_info.seed == 42


@pytest.mark.parametrize("scheduler_cls", [audio_scheduler.EulerScheduler, audio_scheduler.ConsistencyModelScheduler])
def test_audio_scheduler_preserves_previous_frames(monkeypatch, scheduler_cls):
    monkeypatch.setattr(audio_scheduler, "AI_DEVICE", "cpu")
    scheduler = scheduler_cls({"model_cls": "wan2.2_audio", "sample_shift": 1, "seq_parallel": False, "sample_guide_scale": 1, "dim": 8, "num_heads": 1, "parallel": None})
    shape = (2, 3, 4, 4)
    scheduler.prepare(42, shape, infer_steps=2)
    previous_latents = torch.full(shape, 7.0)
    scheduler.reset(42, shape, {"prev_latents": previous_latents, "prev_len": 1})

    torch.testing.assert_close(scheduler.latents[:, :1], previous_latents[:, :1])
    new_frames = scheduler.latents[:, 1:].clone()
    scheduler.noise_pred = torch.ones(shape)
    scheduler.step_post()

    torch.testing.assert_close(scheduler.latents[:, :1], previous_latents[:, :1])
    assert not torch.equal(scheduler.latents[:, 1:], new_frames)

    scheduler.clear()
    assert scheduler.prev_latents is None
    assert scheduler.prev_len == 0

    scheduler.prepare(42, shape, infer_steps=2)
    fresh_scheduler = scheduler_cls(scheduler.config)
    fresh_scheduler.prepare(42, shape, infer_steps=2)
    torch.testing.assert_close(scheduler.latents, fresh_scheduler.latents)

    scheduler.clear()
    scheduler.prepare(42, (2, 3, 8, 8), infer_steps=2)
    assert scheduler.latents.shape == (2, 3, 8, 8)


def test_rs2v_is_not_advertised_by_wan22_audio():
    runner = object.__new__(Wan22AudioRunner)
    with pytest.raises(ValueError, match="Wan22AudioRunner does not support task 'rs2v'"):
        BaseRunner.__init__(runner, {"model_cls": "wan2.2_audio", "task": "rs2v"})

    seko_runner = object.__new__(WanAudioRunner)
    BaseRunner.__init__(seko_runner, {"model_cls": "seko_talk", "task": "rs2v", "vae_stride": [4, 8, 8]})
    input_info = seko_runner.prepare_request({"image_path": "portrait.png", "audio_path": "speech.wav"})
    assert input_info.task == "rs2v"


@pytest.mark.parametrize("cpu_offload", [False, True])
def test_vae_loading_preserves_independent_instances_and_options(tmp_path, monkeypatch, cpu_offload):
    vae_path = tmp_path / "Wan2.2_VAE.pth"
    vae_path.touch()
    runner = object.__new__(Wan22AudioRunner)
    BaseRunner.__init__(runner, {"model_cls": "wan2.2_audio", "task": "s2v", "model_path": str(tmp_path), "cpu_offload": cpu_offload, "vae_offload_cache": True, "dummy_model": True})
    signature = inspect.signature(wan_audio_runner.Wan2_2_VAE)
    instances = [object(), object()]
    constructor = Mock(side_effect=instances)
    monkeypatch.setattr(wan_audio_runner, "Wan2_2_VAE", constructor)

    encoder, decoder = runner.load_vae()

    assert encoder is instances[0]
    assert decoder is instances[1]
    assert constructor.call_count == 2
    for call in constructor.call_args_list:
        arguments = signature.bind(**call.kwargs)
        arguments.apply_defaults()
        assert arguments.arguments["vae_path"] == str(vae_path)
        assert arguments.arguments["dtype"] == torch.float32
        assert arguments.arguments["device"] == torch.device("cpu" if cpu_offload else wan_audio_runner.AI_DEVICE)
        assert arguments.arguments["cpu_offload"] is cpu_offload
        assert arguments.arguments["offload_cache"] is True
        assert arguments.arguments["dummy_model"] is True


@pytest.mark.parametrize("python_entry", [False, True])
@pytest.mark.parametrize("overrides", [{}, {"vae_stride": [4, 32, 32], "num_channels_latents": 64, "use_image_encoder": True}])
def test_wan22_audio_startup_uses_json_settings(tmp_path, python_entry, overrides):
    startup = {"model_cls": "wan2.2_audio", "task": "s2v", "model_path": str(tmp_path)}
    if python_entry:
        startup = LightX2VPipeline(**startup).startup_config
    preset = Path(__file__).resolve().parents[1] / "configs/seko_talk/seko_talk_08_5B_base.json"
    settings = json.loads(preset.read_text())
    settings.update(overrides)
    config_path = tmp_path / "deployment.json"
    config_path.write_text(json.dumps(settings))
    config = build_startup_config({**startup, "config_json": str(config_path)})

    assert config["vae_stride"] == settings["vae_stride"]
    assert config["num_channels_latents"] == settings["num_channels_latents"]
    assert config["use_image_encoder"] is settings["use_image_encoder"]


def test_wan22_audio_pipeline_defaults_without_json(tmp_path):
    pipeline = LightX2VPipeline(model_cls="wan2.2_audio", task="s2v", model_path=str(tmp_path))
    config = build_startup_config(pipeline.startup_config)

    assert config["vae_stride"] == (4, 16, 16)
    assert config["num_channels_latents"] == 48
    assert config["use_image_encoder"] is False
