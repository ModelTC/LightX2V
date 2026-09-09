import json
from functools import partial

import pytest
import torch
from PIL import Image

from lightx2v.models.runners.base_runner import BaseRunner
from lightx2v.models.runners.default_runner import DefaultRunner
from lightx2v.models.runners.wan.wan_audio_runner import WanAudioARRunner
from lightx2v.models.runners.wan.wan_runner import Wan22DenseRunner, Wan22MoeRunner, WanRunner


@pytest.fixture
def cpu_runner(tmp_path, monkeypatch):
    image_path = tmp_path / "square.png"
    Image.new("RGB", (64, 64), "red").save(image_path)
    config = {
        "model_cls": "wan2.1",
        "task": "i2v",
        "target_height": 480,
        "target_width": 832,
        "target_video_length": 5,
        "vae_stride": [4, 8, 8],
        "patch_size": [1, 2, 2],
        "enable_cfg": True,
        "use_image_encoder": False,
        "enable_reuse": True,
        "reuse_cache_path": str(tmp_path / "cache"),
    }
    runner = object.__new__(WanRunner)
    monkeypatch.setattr(runner, "set_init_device", lambda: setattr(runner, "init_device", torch.device("cpu")))
    monkeypatch.setattr(runner, "init_scheduler", lambda: None)
    DefaultRunner.__init__(runner, config)
    runner._gc_frozen = True
    runner.run_input_encoder = runner._run_input_encoder_local_i2v
    # Keep request preparation, image resizing, latent sizing and disk-cache IO real.
    monkeypatch.setattr(runner, "get_vae_encoder_output", lambda *args: torch.ones(1))
    monkeypatch.setattr(runner, "run_text_encoder", lambda info: {"context": torch.ones(1)})
    monkeypatch.setattr(runner, "maybe_empty_cache", lambda: None)
    monkeypatch.setattr(runner, "end_run", lambda: None)
    monkeypatch.setattr(runner, "run_main", lambda: tuple(runner.input_info.latent_shape))
    monkeypatch.setattr(runner, "load_reuse_state", partial(runner.load_reuse_state, map_location="cpu"))
    return runner


@pytest.mark.parametrize(
    ("resize_config", "expected_key_shape"),
    [({}, [480, 832]), ({"resize_mode": "adaptive"}, None), ({"resize_mode": "fixed_shape", "fixed_shape": [64, 96]}, None)],
)
def test_three_requests_reuse_after_real_image_preprocessing(cpu_runner, monkeypatch, resize_config, expected_key_shape, tmp_path):
    runner = cpu_runner
    runner.config.update(resize_config)
    request = {"prompt": "same", "image_path": str(tmp_path / "square.png"), "save_result_path": str(tmp_path / "result.mp4")}
    first = runner.prepare_request({"task": runner.config["task"], **request})
    assert first.target_shape == [480, 832]
    first_result = runner.run_pipeline(first)
    with open(runner.reuse_cache_dir + "/manifest.json") as file:
        saved_key = json.load(file)["reuse_key"]
    if expected_key_shape is None:
        assert "target_shape" not in saved_key
    else:
        assert saved_key["target_shape"] == expected_key_shape
    assert "resize_mode" not in saved_key
    assert "resize_config" not in saved_key
    if resize_config.get("resize_mode") == "adaptive":
        assert first.target_shape == [480, 480]
    elif resize_config.get("resize_mode") == "fixed_shape":
        assert first.target_shape == [64, 96]

    def unexpected_encode():
        pytest.fail("Reusing a successful request must not rerun input encoders")

    monkeypatch.setattr(runner, "run_input_encoder", unexpected_encode)
    runner.set_reuse(True)
    for seed in (2, 3):
        next_input = runner.prepare_request({"task": runner.config["task"], **request, "seed": seed})
        assert next_input.target_shape == [480, 832]
        assert runner.run_pipeline(next_input) == first_result
        assert next_input.target_shape == first.target_shape
        assert next_input.latent_shape == first.latent_shape
        assert next_input is not first
    assert runner.config["target_height"] == 480
    assert runner.config["target_width"] == 832


@pytest.mark.parametrize(
    ("initial_config", "changed_request"),
    [
        ({}, {"target_shape": [720, 1280]}),
        ({"resize_mode": "adaptive"}, {"prompt": "changed"}),
        ({"resize_mode": "adaptive"}, {"negative_prompt": "changed"}),
        ({"resize_mode": "adaptive"}, {"target_video_length": 9}),
        ({"resize_mode": "adaptive"}, {"image_path": "different.png"}),
    ],
)
def test_reuse_rejects_changed_input_dependencies(cpu_runner, initial_config, changed_request, tmp_path):
    runner = cpu_runner
    runner.config.update(initial_config)
    request = {"image_path": str(tmp_path / "square.png"), "save_result_path": str(tmp_path / "result.mp4")}
    runner.run_pipeline(runner.prepare_request({"task": runner.config["task"], **request}))
    runner.set_reuse(True)
    with pytest.raises(ValueError, match="Reuse inputs must match"):
        runner.run_pipeline(runner.prepare_request({"task": runner.config["task"], **request, **changed_request}))


@pytest.mark.parametrize("runner_cls", [WanRunner, Wan22DenseRunner, Wan22MoeRunner])
@pytest.mark.parametrize("resize_mode", ["adaptive", "keep_ratio_fixed_area", "fixed_min_area", "fixed_max_area", "fixed_shape", "fixed_min_side"])
def test_standard_wan_resize_modes_own_image_output_shape(runner_cls, resize_mode):
    for task in runner_cls.supported_request_fields_by_task:
        runner = object.__new__(runner_cls)
        BaseRunner.__init__(runner, {"task": task, "resize_mode": resize_mode})
        fields = runner.get_supported_request_fields(task)
        assert ("target_shape" in fields) == (task == "t2v")
        del runner.config["resize_mode"]
        assert "target_shape" in runner.get_supported_request_fields(task)


def test_custom_audio_image_preprocessing_keeps_its_shape_contract():
    runner = object.__new__(WanAudioARRunner)
    BaseRunner.__init__(runner, {"task": "rs2v", "resize_mode": "adaptive"})
    fields = runner.get_supported_request_fields("rs2v")
    assert "target_shape" in fields


def test_request_shape_is_rejected_when_startup_resize_owns_it(cpu_runner):
    cpu_runner.config["resize_mode"] = "adaptive"
    with pytest.raises(ValueError, match="target_shape"):
        cpu_runner.prepare_request({"task": cpu_runner.config["task"], "target_shape": [720, 1280]})
