import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from lightx2v.models.runners.base_runner import BaseRunner
from lightx2v.shot_runner import rs2v_infer, stream_infer
from lightx2v.shot_runner.shot_base import ShotPipeline, load_clip_configs


@pytest.mark.parametrize("task", ["s2v", "rs2v"])
def test_shot_loads_startup_task_from_clip_json(tmp_path, task):
    model_path = tmp_path / "model"
    model_path.mkdir()
    (model_path / "config.json").write_text('{"hidden_size": 256}')
    (tmp_path / "clip.json").write_text(json.dumps({"model_cls": "seko_talk", "model_path": str(model_path), "task": task, "target_video_length": 81}))
    main_path = tmp_path / "main.json"
    main_path.write_text(json.dumps({"lightx2v_path": str(tmp_path), "clip_configs": [{"name": "clip", "path": "clip.json"}]}))

    clips = load_clip_configs(main_path)

    assert len(clips) == 1
    assert clips[0].name == "clip"
    assert clips[0].config_json["task"] == task
    assert clips[0].config_json["hidden_size"] == 256
    assert clips[0].config_json["target_video_length"] == 81
    assert clips[0].config_json["warmup"] is False


@pytest.mark.parametrize("request_fields", [{}, {"seed": 0, "target_shape": [480, 832], "save_result_path": "chosen.mp4"}])
def test_shot_uses_flat_specs_and_caller_content(request_fields):
    pipeline = object.__new__(ShotPipeline)
    config = {
        "task": "rs2v",
        "target_video_length": 81,
        "target_shape": [720, 1280],
    }
    args = SimpleNamespace(prompt="request", image_path="input.png", audio_path="input.wav", **request_fields)

    result = pipeline.prepare_input_info(args, config)

    assert result.task == "rs2v"
    assert result.prompt == "request"
    assert result.negative_prompt == ""
    assert result.image_path == "input.png"
    assert result.audio_path == "input.wav"
    assert result.seed == request_fields.get("seed")
    assert result.save_result_path == request_fields.get("save_result_path")
    assert result.target_shape == request_fields.get("target_shape", [720, 1280])
    assert result.target_video_length == 81


@pytest.mark.parametrize(("module", "pipeline_name"), [(rs2v_infer, "ShotRS2VPipeline"), (stream_infer, "ShotStreamPipeline")])
@pytest.mark.parametrize("explicit_shape", [False, True])
def test_shot_cli_keeps_json_shape_unless_explicitly_overridden(monkeypatch, module, pipeline_name, explicit_shape):
    config = {"task": "rs2v" if module is rs2v_infer else "s2v", "target_shape": [720, 1280]}
    pipeline = object.__new__(ShotPipeline)
    inputs = []
    pipeline.generate = lambda args: inputs.append(pipeline.prepare_input_info(args, config))
    monkeypatch.setattr(module, "load_clip_configs", lambda path: [])
    monkeypatch.setattr(module, pipeline_name, lambda configs: pipeline)
    argv = ["shot", "--config_json", "deployment.json"]
    if explicit_shape:
        argv += ["--target_shape", "480", "832"]
    monkeypatch.setattr("sys.argv", argv)

    module.main()

    assert inputs[0].target_shape == ([480, 832] if explicit_shape else [720, 1280])
    assert inputs[0].save_result_path is None


@pytest.mark.parametrize(("request_fields", "expected_frames"), [({}, 81), ({"video_duration": 1.0}, 17), ({"video_duration": 1.0, "target_video_length": 49}, 49)])
def test_rs2v_keeps_duration_and_explicit_frame_priority(monkeypatch, request_fields, expected_frames):
    pipeline = object.__new__(rs2v_infer.ShotRS2VPipeline)
    runner = object.__new__(BaseRunner)
    runner.config = {"task": "rs2v", "target_video_length": 81, "target_fps": 16, "audio_sr": 16000, "vae_stride": (4, 8, 8)}
    observed_frames = []
    runner._run_input_encoder_local_rs2v_static = lambda: observed_frames.append(runner.input_info.target_video_length)
    runner._run_input_encoder_local_rs2v_dynamic = lambda: None
    runner.check_stop = lambda: None
    runner.run_clip_main = lambda: (torch.zeros(1, 3, runner.input_info.target_video_length, 1, 1), runner.input_info.audio_clip, torch.zeros(1, 1, 1, 1))
    pipeline.clip_generators = {"rs2v_clip": runner}
    pipeline.progress_callback = None
    monkeypatch.setattr(rs2v_infer, "load_audio_file", lambda path: (torch.zeros(1, 16000), 16000))
    save = Mock()
    monkeypatch.setattr(rs2v_infer, "save_to_video", save)
    args = SimpleNamespace(prompt="request", image_path="input.png", audio_path="input.wav", **request_fields)

    pipeline.generate(args)

    assert observed_frames == [expected_frames]
    assert runner.input_info.seed == 42
    assert runner.input_info.save_result_path is None
    save.assert_not_called()


def test_shot_stream_omitted_output_does_not_write_temporary_files(monkeypatch):
    pipeline = object.__new__(stream_infer.ShotStreamPipeline)
    config = {"task": "s2v", "target_video_length": 33, "prev_frame_length": 1, "target_fps": 16, "audio_sr": 16000}
    runner = object.__new__(BaseRunner)
    runner.config = config
    runner.prev_frame_length = 1
    runner.run_clip_pipeline = lambda inputs: (torch.zeros(1, 3, 33, 1, 1), inputs.audio_clip, None)
    pipeline.clip_generators = {"s2v_clip": runner, "f2v_clip": runner}
    monkeypatch.setattr(stream_infer, "load_audio_file", lambda path: (torch.zeros(1, 33000), 16000))
    save = Mock()
    monkeypatch.setattr(stream_infer, "save_to_video", save)
    monkeypatch.setattr(stream_infer, "save_audio", save)

    pipeline.generate(SimpleNamespace(prompt="request", image_path="input.png", audio_path="input.wav"))

    save.assert_not_called()
