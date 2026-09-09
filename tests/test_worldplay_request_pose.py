import json
from unittest.mock import Mock

import pytest
import torch

from lightx2v.models.networks.worldplay import pose_utils
from lightx2v.models.runners.base_runner import BaseRunner
from lightx2v.models.runners.default_runner import DefaultRunner
from lightx2v.models.runners.worldplay.worldplay_ar_runner import WorldPlayARRunner
from lightx2v.models.runners.worldplay.worldplay_bi_runner import WorldPlayBIRunner
from lightx2v.models.runners.worldplay.worldplay_distill_runner import WorldPlayDistillRunner
from lightx2v.models.schedulers.hunyuan_video.scheduler import HunyuanVideo15Scheduler
from lightx2v.utils.input_info import UNSET
from lightx2v.utils.lockable_dict import LockableDict


@pytest.mark.parametrize("runner_cls", [WorldPlayARRunner, WorldPlayBIRunner, WorldPlayDistillRunner])
def test_worldplay_t2v_reaches_existing_encoder_and_conditioning(monkeypatch, runner_cls):
    monkeypatch.setattr(f"{runner_cls.__module__}.AI_DEVICE", "cpu")
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    instance = object.__new__(runner_cls)
    config = LockableDict({"task": "t2v", "hidden_size": 4, "target_video_length": 17, "vae_stride": [4, 16, 16]})
    BaseRunner.__init__(instance, config)
    instance.load_model = Mock()
    instance.vision_num_semantic_tokens = 3
    instance.get_latent_shape_with_target_hw = Mock(return_value=[2, 5, 2, 2])
    instance.run_text_encoder = Mock(return_value={"prompt_embeds": torch.zeros(1, 2, 4)})
    DefaultRunner.init_modules(instance)
    instance.input_info = instance.prepare_request({"prompt": "move forward", "pose": "w-4"})

    encoded = instance.run_input_encoder()

    assert instance.input_info.task == "t2v"
    assert instance.input_info.prompt == "move forward"
    assert encoded["pose_output"]["action"].shape == (1, 5)
    assert encoded["image_encoder_output"]["cond_latents"] is None
    assert torch.count_nonzero(encoded["image_encoder_output"]["siglip_output"]) == 0
    scheduler = object.__new__(HunyuanVideo15Scheduler)
    condition, mask = scheduler._prepare_cond_latents_and_mask("t2v", None, torch.ones(1, 2, 5, 2, 2), torch.zeros(5), False)
    assert torch.count_nonzero(condition) == torch.count_nonzero(mask) == 0
    with pytest.raises(ValueError, match="image_path"):
        instance.prepare_request({"image_path": "unused.png"})


@pytest.fixture
def runner():
    instance = object.__new__(WorldPlayARRunner)
    BaseRunner.__init__(instance, LockableDict({"task": "i2v", "model_cls": "worldplay_ar", "target_video_length": 125, "vae_stride": [4, 16, 16]}))
    return instance


@pytest.mark.parametrize("frame_fields", [{}, {"target_video_length": None}, {"target_video_length": UNSET}, {"target_video_length": 61}])
def test_pose_sets_request_length_without_changing_startup(runner, frame_fields):
    request = {"pose": "w-15", **frame_fields}
    original = request.copy()

    input_info = runner.prepare_request(request)

    assert input_info.target_video_length == 61
    assert len(input_info.pose) == 16
    assert request == original
    assert runner.config["target_video_length"] == 125
    assert runner.prepare_request({}).target_video_length == 125
    assert runner.prepare_request({}).pose is None


@pytest.mark.parametrize("requested_frames", [62, 125])
def test_explicit_frame_count_must_match_pose_exactly(runner, requested_frames):
    with pytest.raises(ValueError, match=f"pose corresponds to 61 frames, but num_frames is {requested_frames}"):
        runner.prepare_request({"pose": "w-15", "target_video_length": requested_frames})
    assert runner.config["target_video_length"] == 125


def test_without_pose_preserves_request_frame_override(runner):
    input_info = runner.prepare_request({"target_video_length": 121})
    assert input_info.target_video_length == 121
    assert input_info.pose is None


def test_prepared_pose_does_not_read_json_again(runner, tmp_path):
    pose = pose_utils.pose_string_to_json("w-15")
    path = tmp_path / "pose.json"
    path.write_text(json.dumps(pose))

    input_info = runner.prepare_request({"pose": str(path)})
    path.unlink()
    viewmats, intrinsics, actions = pose_utils.pose_to_input(input_info.pose, latent_num=16)

    assert input_info.target_video_length == 61
    assert input_info.pose == pose
    assert viewmats.shape == (16, 4, 4)
    assert intrinsics.shape == (16, 3, 3)
    assert actions.shape == (16,)


def test_command_pose_is_parsed_once(runner, monkeypatch):
    parse = Mock(wraps=pose_utils.pose_string_to_json)
    monkeypatch.setattr(pose_utils, "pose_string_to_json", parse)

    input_info = runner.prepare_request({"pose": "d-31"})
    pose_utils.pose_to_input(input_info.pose, latent_num=32)

    parse.assert_called_once_with("d-31")
    assert input_info.target_video_length == 125


def test_pose_dictionary_and_direct_shape_validation(runner):
    pose = pose_utils.pose_string_to_json("w-15,s-15")
    input_info = runner.prepare_request({"pose": pose, "target_video_length": 121})

    assert input_info.pose is pose
    assert input_info.target_video_length == 121
    with pytest.raises(ValueError, match="pose corresponds to 121 frames"):
        pose_utils.pose_to_input(pose, latent_num=32)
